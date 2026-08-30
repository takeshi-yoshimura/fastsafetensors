# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from collections import OrderedDict

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .common import init_logger
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType
from .tensor_factory import LazyTensorFactory

logger = init_logger(__name__)


class FilesBufferOnDevice:
    r"""Device buffer for .safetensors files.
        Users can call get_tensor(), get_sharded(), etc. to instantiate (sharded) tensors from the device buffer.
        Note that for multi-process loading, users must follow the single-program multiple-data (SPMD) paradigm, which is common for torch.distributed programs.
        In other words, users must ensure that every worker process calls the methods here in the same order.
        This is because methods here reuse torch.distributed operations: broadcast, scatter, recv, and send.
        They synchornously wait all the workers to execute copies among processes.

        Users should create this instance with SafeTensorsFileLoader.copy_files_to_device().
        Tensors returned from this buffer are valid only while the buffer stays open.
        Clone/copy returned tensors before close() if the tensor data must be used
        after this buffer is closed.

    Args:
        rank_loaders (Dict<rank, list(LazyTensorFacotry)>): Tensor factories per rank, which hold device pointers for buffers.
        pg (ProcessGroupBase): process group for calling distributed ops.
        auto_mem_delete (bool): automatically release device buffers when all the tensors are shuffled.
        keep_tensor (Callable[[str], bool], optional): If set, only tensors for
            which ``keep_tensor(name)`` is True are registered in ``key_to_rank_lidx``;
            others raise ``ValueError`` from ``get_tensor`` / ``get_filename`` /
            ``get_shape``. Subclasses that reimplement the registration loop must
            honor this.

    Examples:
        See examples/run_single.py and examples/run_parallel.py.
    """

    def __init__(
        self,
        rank_loaders: Dict[int, List[LazyTensorFactory]],
        pg: ProcessGroupBase,
        framework: FrameworkOpBase,
        auto_mem_delete: bool = True,
        keep_tensor: Optional[Callable[[str], bool]] = None,
    ):
        self.framework = framework
        self.rank_loaders: Dict[int, List[LazyTensorFactory]] = rank_loaders
        self.key_to_rank_lidx: Dict[str, Tuple[int, int]] = {}
        self.instantiated: Dict[int, Dict[int, Dict[str, bool]]] = {}  # rank, key name
        for rank, loaders in rank_loaders.items():
            self.instantiated[rank] = {}
            for lidx, loader in enumerate(loaders):
                for key in loader.metadata.tensors.keys():
                    if keep_tensor is not None and not keep_tensor(key):
                        continue
                    if key in self.key_to_rank_lidx:
                        raise Exception(
                            f"FilesBufferOnDevice: key {key} must be unique among files"
                        )
                    self.key_to_rank_lidx[key] = (rank, lidx)
                self.instantiated[rank][lidx] = {}
        self.pg = pg
        self.auto_mem_delete = auto_mem_delete and self.pg.size() > 1
        self._profile_runs = 0
        self._profile_tensors = 0
        self._profile_bytes = 0
        self._profile_wait_ns = 0
        self._profile_broadcast_ns = 0
        self._profile_view_ns = 0

    def close(self):
        """Release the backing device buffers.

        Any tensor returned from this FilesBufferOnDevice becomes invalid after
        close() unless the caller cloned/copied it to independent storage.
        """
        for _, loaders in self.rank_loaders.items():
            for loader in loaders:
                loader.free_dev_ptrs()
        self.rank_loaders = {}

    def get_filename(self, tensor_name: str) -> str:
        rank, lidx = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.src

    def get_shape(self, tensor_name: str) -> List[int]:
        rank, lidx = self._get_rank_lidx(tensor_name)
        return self.rank_loaders[rank][lidx].metadata.tensors[tensor_name].shape

    def _get_rank_lidx(self, tensor_name: str) -> Tuple[int, int]:
        if tensor_name not in self.key_to_rank_lidx:
            raise ValueError(f"_get_rank: key {tensor_name} was not found in files")
        return self.key_to_rank_lidx[tensor_name]

    def _get_tensor(
        self,
        rank: int,
        lidx: int,
        tensor_name: str,
        ret: TensorBase,
        device: Optional[Device],
        dtype: DType,
    ) -> TensorBase:
        loader = self.rank_loaders[rank][lidx]
        loader.wait_tensor(tensor_name)
        if self.auto_mem_delete:
            self.instantiated[rank][lidx][tensor_name] = True
            if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                if self.pg.rank() == rank:
                    logger.debug(
                        "_get_tensor: free_dev_ptrs, lidx=%d, src=%s",
                        lidx,
                        loader.metadata.src,
                    )
                loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def get_sharded_wrapped(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return a wrapped shard of tensor_name.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank, lidix = self._get_rank_lidx(tensor_name)
        self.rank_loaders[rank][lidix].wait_tensor(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle(self.pg, tensor_name, dim)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype)

    def get_sharded(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        partition a tensor instance with the key tensor_name at the dimension dim and return it.
        In multi-process loading, this eventually calls torch.distributed.scatter.
        A special dim is -1, which broadcast a tensor to all the ranks (== get_tensor()).
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_sharded_wrapped(tensor_name, dim, device, dtype).get_raw()

    def get_tensor_wrapped(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return a wrapped tensor by name.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_sharded_wrapped(tensor_name, -1, device, dtype)

    def get_tensor(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """
        get a tensor instance with the key tensor_name from a local or remote rank.
        In multi-process loading, this eventually calls torch.distributed.broadcast.
        So, every rank will allocate the same tensor at each device memroy.
        In single-process loading, this directly instantiates a tensor from the device buffer with zero copy.
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        return self.get_tensor_wrapped(tensor_name, device, dtype).get_raw()

    def _broadcast_run_wrapped(
        self,
        tensor_names: List[str],
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> List[TensorBase]:
        """Materialize a same-source run with one coalesced broadcast."""
        if not tensor_names:
            return []
        rank, first_lidx = self._get_rank_lidx(tensor_names[0])
        dsts: List[TensorBase] = []
        locations: List[Tuple[int, int, str]] = []
        source_tensors: List[Optional[TensorBase]] = []
        frames = []
        profile = os.environ.get("FASTSAFETENSORS_PROFILE") == "1"
        wait_begin = time.perf_counter_ns() if profile else 0
        for name in tensor_names:
            tensor_rank, lidx = self._get_rank_lidx(name)
            if tensor_rank != rank or lidx != first_lidx:
                raise ValueError("a broadcast run must have one source file")
            loader = self.rank_loaders[tensor_rank][lidx]
            loader.wait_tensor(name)
            frame = loader.metadata.tensors[name]
            frames.append(frame)
            source_tensors.append(
                loader.tensors[name] if self.pg.rank() == tensor_rank else None
            )
        if profile:
            self._profile_wait_ns += time.perf_counter_ns() - wait_begin

        # The optimized path broadcasts the loader's backing byte span directly
        # and returns views into it, avoiding source clones, receiver allocations
        # per tensor, and c10d's additional coalescing buffer.
        if device is None and dtype == DType.AUTO:
            loader = self.rank_loaders[rank][first_lidx]
            result = self.framework.broadcast_contiguous_run(
                self.pg, source_tensors, frames, rank, loader.device
            )
            if result is not None:
                run, broadcast_ns, view_ns = result
                if profile:
                    self._profile_runs += 1
                    self._profile_tensors += len(frames)
                    self._profile_bytes += sum(
                        frame.data_offsets[1] - frame.data_offsets[0]
                        for frame in frames
                    )
                    self._profile_broadcast_ns += broadcast_ns
                    self._profile_view_ns += view_ns
                return run

        for name, frame in zip(tensor_names, frames):
            if self.pg.rank() == rank:
                dst = self.rank_loaders[rank][first_lidx].tensors[name].clone().detach()
            else:
                dst = self.framework.get_empty_tensor(
                    frame.shape,
                    frame.dtype,
                    self.rank_loaders[rank][first_lidx].device,
                )
            dsts.append(dst)
            locations.append((rank, first_lidx, name))

        broadcast_begin = time.perf_counter_ns() if profile else 0
        self.pg.broadcast_many(dsts, rank)
        if profile:
            self._profile_runs += 1
            self._profile_tensors += len(frames)
            self._profile_bytes += sum(
                frame.data_offsets[1] - frame.data_offsets[0] for frame in frames
            )
            self._profile_broadcast_ns += time.perf_counter_ns() - broadcast_begin
        return [
            self._get_tensor(rank, lidx, name, dst, device, dtype)
            for dst, (rank, lidx, name) in zip(dsts, locations)
        ]

    def _prepare_exchange_run(
        self, tensor_names: List[str]
    ) -> Tuple[List[Optional[TensorBase]], List[Any]]:
        """Wait for the local source run while the peer waits for its own run."""
        rank, first_lidx = self._get_rank_lidx(tensor_names[0])
        source_tensors: List[Optional[TensorBase]] = []
        frames: List[Any] = []
        profile = os.environ.get("FASTSAFETENSORS_PROFILE") == "1"
        wait_begin = time.perf_counter_ns() if profile else 0
        for name in tensor_names:
            tensor_rank, lidx = self._get_rank_lidx(name)
            if tensor_rank != rank or lidx != first_lidx:
                raise ValueError("an exchange run must have one source file")
            loader = self.rank_loaders[rank][lidx]
            loader.wait_tensor(name)
            frames.append(loader.metadata.tensors[name])
            source_tensors.append(
                loader.tensors[name] if self.pg.rank() == rank else None
            )
        if profile:
            self._profile_wait_ns += time.perf_counter_ns() - wait_begin
        return source_tensors, frames

    def _exchange_run_pair(
        self, runs: List[List[str]]
    ) -> Optional[List[List[TensorBase]]]:
        """Exchange one rank-0 and one rank-1 run in both directions."""
        prepared = [self._prepare_exchange_run(run) for run in runs]
        source_tensors = [item[0] for item in prepared]
        frames = [item[1] for item in prepared]
        local_rank = self.pg.rank()
        local_source_rank, local_lidx = self._get_rank_lidx(runs[local_rank][0])
        if local_source_rank != local_rank:
            raise RuntimeError("paired exchange runs are not ordered by source rank")
        device = self.rank_loaders[local_rank][local_lidx].device
        result = self.framework.exchange_contiguous_runs(
            self.pg, source_tensors, frames, device
        )
        if result is None:
            return None

        outputs, exchange_ns, view_ns = result
        if os.environ.get("FASTSAFETENSORS_PROFILE") == "1":
            self._profile_runs += 2
            self._profile_tensors += sum(len(run_frames) for run_frames in frames)
            self._profile_bytes += sum(
                frame.data_offsets[1] - frame.data_offsets[0]
                for run_frames in frames
                for frame in run_frames
            )
            self._profile_broadcast_ns += exchange_ns
            self._profile_view_ns += view_ns
        return outputs

    def iter_tensors(self, tensor_names: List[str]) -> Iterator[Tuple[str, Any]]:
        """Yield tensors while coalescing consecutive same-source broadcasts.

        ``FASTSAFETENSORS_BROADCAST_RUN_BYTES`` and
        ``FASTSAFETENSORS_BROADCAST_RUN_TENSORS`` bound temporary receive
        memory, allocator metadata, and latency to the first tensor.  Setting
        either to zero restores per-tensor broadcasts.
        """
        if not tensor_names:
            return
        if self.pg.size() == 1:
            for name in tensor_names:
                yield name, self.get_tensor(name)
            return

        max_bytes = int(
            os.environ.get("FASTSAFETENSORS_BROADCAST_RUN_BYTES", str(16 << 20))
        )
        max_tensors = int(os.environ.get("FASTSAFETENSORS_BROADCAST_RUN_TENSORS", "64"))
        if max_bytes < 0:
            raise ValueError("FASTSAFETENSORS_BROADCAST_RUN_BYTES must be >= 0")
        if max_tensors < 0:
            raise ValueError("FASTSAFETENSORS_BROADCAST_RUN_TENSORS must be >= 0")
        if max_bytes == 0 or max_tensors == 0:
            for name in tensor_names:
                yield name, self.get_tensor(name)
            return

        runs: List[List[str]] = []
        run: List[str] = []
        run_location: Optional[Tuple[int, int]] = None
        run_bytes = 0
        run_end: Optional[int] = None

        def flush() -> None:
            nonlocal run, run_location, run_bytes, run_end
            if run:
                runs.append(run)
            run = []
            run_location = None
            run_bytes = 0
            run_end = None

        for name in tensor_names:
            rank, lidx = self._get_rank_lidx(name)
            frame = self.rank_loaders[rank][lidx].metadata.tensors[name]
            size = frame.data_offsets[1] - frame.data_offsets[0]
            if run and (
                (rank, lidx) != run_location
                or frame.data_offsets[0] != run_end
                or run_bytes + size > max_bytes
                or len(run) >= max_tensors
            ):
                flush()
            run.append(name)
            run_location = (rank, lidx)
            run_bytes += size
            run_end = frame.data_offsets[1]
        flush()

        # Pair only adjacent runs with opposite sources.  In particular, do
        # not bucket all rank-0 and rank-1 runs and interleave them: callers may
        # depend on checkpoint order to finish a module's temporary loading
        # state promptly, and changing that order can cause severe memory
        # pressure for large quantized/MoE models.
        #
        # Adjacent pairing preserves tensor_names exactly while still allowing
        # each rank to wait for its local read concurrently and exchange both
        # byte spans with one grouped send/recv.
        exchange_enabled = os.environ.get(
            "FASTSAFETENSORS_BIDIRECTIONAL_EXCHANGE", "1"
        ) not in ("0", "false", "False")
        if self.pg.size() == 2 and exchange_enabled:
            index = 0
            while index < len(runs):
                completed_run = runs[index]
                source_rank, _ = self._get_rank_lidx(completed_run[0])
                if index + 1 < len(runs):
                    next_run = runs[index + 1]
                    next_source_rank, _ = self._get_rank_lidx(next_run[0])
                else:
                    next_run = []
                    next_source_rank = source_rank

                if {source_rank, next_source_rank} == {0, 1}:
                    pair = [completed_run, next_run]
                    runs_by_rank = [pair[source_rank], pair[next_source_rank]]
                    outputs = self._exchange_run_pair(runs_by_rank)
                    if outputs is None:
                        for completed_run in pair:
                            tensors = self._broadcast_run_wrapped(completed_run)
                            for name, tensor in zip(completed_run, tensors):
                                yield name, tensor.get_raw()
                    else:
                        outputs_by_source = {
                            0: outputs[0],
                            1: outputs[1],
                        }
                        for completed_run in pair:
                            run_source, _ = self._get_rank_lidx(completed_run[0])
                            tensors = outputs_by_source[run_source]
                            for name, tensor in zip(completed_run, tensors):
                                yield name, tensor.get_raw()
                    index += 2
                    continue

                tensors = self._broadcast_run_wrapped(completed_run)
                for name, tensor in zip(completed_run, tensors):
                    yield name, tensor.get_raw()
                index += 1
            runs = []

        for completed_run in runs:
            tensors = self._broadcast_run_wrapped(completed_run)
            for name, tensor in zip(completed_run, tensors):
                yield name, tensor.get_raw()

        # The run collectives establish CUDA-stream dependencies but no longer
        # synchronize the whole device one run at a time.  Drain once per
        # loader batch before its backing allocation can be released.
        first_rank, first_lidx = self._get_rank_lidx(tensor_names[0])
        synchronize_begin = (
            time.perf_counter_ns()
            if os.environ.get("FASTSAFETENSORS_PROFILE") == "1"
            else 0
        )
        self.framework.synchronize(self.rank_loaders[first_rank][first_lidx].device)
        if synchronize_begin:
            self._profile_broadcast_ns += time.perf_counter_ns() - synchronize_begin
        if os.environ.get("FASTSAFETENSORS_PROFILE") == "1":
            print(
                "[FST_PROFILE] byte_runs "
                f"rank={self.pg.rank()} runs={self._profile_runs} "
                f"tensors={self._profile_tensors} bytes={self._profile_bytes} "
                f"wait_tensor_ms={self._profile_wait_ns / 1e6:.3f} "
                f"broadcast_ms={self._profile_broadcast_ns / 1e6:.3f} "
                f"views_ms={self._profile_view_ns / 1e6:.3f}",
                file=sys.stderr,
                flush=True,
            )

    def iter_local_tensors(self, tensor_names: List[str]):
        """Yield already-materialized tensors without the distributed path.

        This is only valid for a single-process buffer. The returned tensors
        borrow their backing device buffers and become invalid after close().
        """
        if self.pg.size() != 1:
            raise RuntimeError("iter_local_tensors requires a single-process group")
        for tensor_name in tensor_names:
            rank, lidx = self._get_rank_lidx(tensor_name)
            self.rank_loaders[rank][lidx].wait_tensor(tensor_name)
            yield (
                tensor_name,
                self.rank_loaders[rank][lidx].tensors[tensor_name].get_raw(),
            )

    def push_tensor(
        self,
        tensor_name: str,
        dst_rank: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Optional[Any]:
        """
        push a tensor instance with the key tensor_name from a rank to a destination rank dst_rank.
        In multi-process loading, this eventually calls torch.distributed.send if the rank has the tensor instance.
        The destination rank will call torch.distributed.recv.
        Other ranks do nothing.
        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank, lidix = self._get_rank_lidx(tensor_name)
        t = self.rank_loaders[rank][lidix].push(self.pg, tensor_name, dst_rank, rank)
        if t:
            return self._get_tensor(
                rank, lidix, tensor_name, t, device, dtype
            ).get_raw()
        return None

    def get_multi_cols(
        self,
        tensor_names: List[str],
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return concatenated column shards from tensor_names.

        The returned tensor must not be used after close() unless the caller
        cloned/copied it to independent storage.
        """
        rank_lidixs: Dict[Tuple[int, int], List[str]] = {}
        for tensor_name in tensor_names:
            ranklidx = self._get_rank_lidx(tensor_name)
            if ranklidx in rank_lidixs:
                rank_lidixs[ranklidx].append(tensor_name)
            else:
                rank_lidixs[ranklidx] = [tensor_name]
        ts: List[TensorBase] = []
        for (rank, lidix), tns in sorted(rank_lidixs.items(), key=lambda x: x[0]):
            ts.append(
                self.rank_loaders[rank][lidix].shuffle_multi_cols(self.pg, tns, dim)
            )
        if len(ts) == 1:
            # fastpath: tensors at the same layer are often in the same file
            return self._get_tensor(
                rank, lidix, rank_lidixs[(rank, lidix)][0], ts[0], device, dtype
            )
        ret = self.framework.concat_tensors(ts, dim=dim)
        if self.auto_mem_delete:
            for tensor_name in tensor_names:
                rank, lidx = self._get_rank_lidx(tensor_name)
                loader = self.rank_loaders[rank][lidx]
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "get_multi_cols: free_dev_ptrs, rank=%d, lidx=%d, src=%s",
                            rank,
                            lidx,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        return ret.to(device=device, dtype=dtype)

    def as_dict(self, tensor_shard_dim: OrderedDict[str, int]) -> Dict[str, TensorBase]:
        """Return tensors keyed by name according to the requested shard dims.

        Returned tensors must not be used after close() unless the caller
        cloned/copied them to independent storage.
        """
        tensors: Dict[str, TensorBase] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            rank, lidx = self._get_rank_lidx(tensor_name)
            loader = self.rank_loaders[rank][lidx]
            tensors[tensor_name] = loader.shuffle(self.pg, tensor_name, dim)
            if self.auto_mem_delete:
                self.instantiated[rank][lidx][tensor_name] = True
                if len(self.instantiated[rank][lidx]) == len(loader.metadata.tensors):
                    if self.pg.rank() == rank:
                        logger.debug(
                            "as_dict: free_dev_ptrs, rank=%d, src=%s",
                            rank,
                            loader.metadata.src,
                        )
                    loader.free_dev_ptrs()
        return tensors
