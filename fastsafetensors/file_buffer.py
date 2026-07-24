# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .common import init_logger
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType
from .tensor_factory import LazyTensorFactory

logger = init_logger(__name__)


class TensorConsumedError(ValueError):
    """Raised when a tensor name is requested after it has been consumed.

    A name is consumed by ``get_and_remove_tensor()`` /
    ``get_and_remove_tensor_wrapped()`` (and, in a later PR, ``drain_tensors()``).
    Consuming access hands the caller sole ownership of the name, so any
    subsequent reusable (``get_tensor``) or consuming (``get_and_remove_tensor``)
    acquisition of the same name fails with this error. It subclasses
    ``ValueError`` so existing ``except ValueError`` handlers around tensor
    acquisition keep working, while callers that want to distinguish "already
    consumed" from "unknown name" can catch this type.

    Metadata queries such as ``get_shape`` / ``get_filename`` remain valid after
    consumption and do not raise.
    """


class FilesBufferOnDevice:
    r"""Device buffer for .safetensors files.
        Users can call get_tensor(), get_sharded(), etc. to instantiate (sharded) tensors from the device buffer.
        Note that for multi-process loading, users must follow the single-program multiple-data (SPMD) paradigm, which is common for torch.distributed programs.
        In other words, users must ensure that every worker process calls the methods here in the same order.
        This is because methods here reuse torch.distributed operations: broadcast, scatter, recv, and send.
        They synchornously wait all the workers to execute copies among processes.

        Users should create this instance with SafeTensorsFileLoader.copy_files_to_device().
        Tensors returned from this buffer take shared ownership of their backing
        device allocation, so they remain valid after close(). The physical
        memory is released only once the buffer and every exported tensor are
        gone.

        Acquisition comes in two flavors. Reusable access (get_tensor(),
        get_sharded(), ...) leaves the name available to request again.
        Consuming access (get_and_remove_tensor()) hands over the name once:
        after it succeeds the name is removed from keys() and any further
        reusable or consuming acquisition of that name raises
        TensorConsumedError. Use keys() for the names still available and
        all_keys() for every registered name (metadata stays queryable after
        consumption).

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
        # Registered (available-at-load) names per factory, so each factory only
        # retains references to the tensors reachable through this buffer. This
        # replaces the old per-(rank, lidx) `instantiated` count check, which
        # compared against the full file metadata and mis-tracked filtered loads.
        factory_names: Dict[Tuple[int, int], set] = {}
        for rank, loaders in rank_loaders.items():
            for lidx, loader in enumerate(loaders):
                names = factory_names.setdefault((rank, lidx), set())
                for key in loader.metadata.tensors.keys():
                    if keep_tensor is not None and not keep_tensor(key):
                        continue
                    if key in self.key_to_rank_lidx:
                        raise Exception(
                            f"FilesBufferOnDevice: key {key} must be unique among files"
                        )
                    self.key_to_rank_lidx[key] = (rank, lidx)
                    names.add(key)
        for (rank, lidx), names in factory_names.items():
            rank_loaders[rank][lidx].retain_only(names)
        # Names consumed via get_and_remove_tensor()/drain_tensors().
        # Availability state, kept separate from the immutable key_to_rank_lidx
        # metadata so that metadata lookups keep working after a name is consumed.
        self._consumed_keys: Set[str] = set()
        self.pg = pg
        # auto_mem_delete eagerly drops internal references as tensors are handed
        # out (reusable access then cannot re-request the same name). It stays
        # gated to distributed groups for backward compatibility -- single-group
        # zero-copy access keeps names reusable until close. This flag is
        # transitional and is superseded by consuming access
        # (get_and_remove_*/drain_tensors); it is slated for deprecation.
        self.auto_mem_delete = auto_mem_delete and self.pg.size() > 1

    def close(self):
        """Release this buffer's references to its backing device allocations.

        This is a logical close: it drops the buffer-side references but leaves
        previously returned tensors valid. Each backing allocation is freed only
        once its final reference is gone. Idempotent: safe to call repeatedly.
        """
        for _, loaders in self.rank_loaders.items():
            for loader in loaders:
                loader.free_dev_ptrs()
        self.rank_loaders = {}

    def keys(self) -> List[str]:
        """Return the tensor names still available for acquisition.

        Excludes names already consumed via get_and_remove_tensor()/
        drain_tensors(). This is the preferred way to enumerate a buffer's
        tensors; prefer it over reaching into ``key_to_rank_lidx`` directly.
        """
        return [k for k in self.key_to_rank_lidx if k not in self._consumed_keys]

    def all_keys(self) -> List[str]:
        """Return every registered tensor name, including consumed ones.

        Backed by immutable metadata, so consumed names remain listed here and
        their get_shape()/get_filename() lookups keep working.
        """
        return list(self.key_to_rank_lidx.keys())

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

    def _require_available(self, tensor_name: str) -> Tuple[int, int]:
        """Validate that *tensor_name* can still be acquired.

        Raises ValueError if the name was never registered (or was filtered
        out) and TensorConsumedError if it has already been consumed. Returns
        the (rank, lidx) so callers can reuse the lookup.
        """
        rank_lidx = self._get_rank_lidx(tensor_name)
        if tensor_name in self._consumed_keys:
            raise TensorConsumedError(
                f"tensor {tensor_name} was consumed and is no longer available"
            )
        return rank_lidx

    def _release_internal(self, rank: int, lidx: int, tensor_name: str) -> None:
        """Drop the factory's internal reference to a handed-out tensor.

        The physical buffer is released only when the factory's retained set
        drains and no exported tensor still references it (shared ownership);
        there is no explicit last-tensor free.
        """
        self.rank_loaders[rank][lidx].drop_internal_reference(tensor_name)

    def _get_tensor(
        self,
        rank: int,
        lidx: int,
        tensor_name: str,
        ret: TensorBase,
        device: Optional[Device],
        dtype: DType,
        consume: bool = False,
    ) -> TensorBase:
        # Consuming access always transfers ownership out of the buffer;
        # auto_mem_delete does the same eagerly for reusable access.
        if consume or self.auto_mem_delete:
            self._release_internal(rank, lidx, tensor_name)
        return ret.to(device=device, dtype=dtype)

    def get_sharded_wrapped(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
        consume: bool = False,
    ) -> TensorBase:
        """Return a wrapped shard of tensor_name.

        The returned tensor keeps its backing allocation alive, so it stays
        valid after close(). Raises TensorConsumedError if the name was already
        consumed via get_and_remove_tensor()/drain_tensors(). When *consume* is
        True the factory's internal reference is dropped (see
        get_and_remove_sharded_wrapped).
        """
        rank, lidix = self._require_available(tensor_name)
        t = self.rank_loaders[rank][lidix].shuffle(self.pg, tensor_name, dim)
        return self._get_tensor(rank, lidix, tensor_name, t, device, dtype, consume)

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
        The returned tensor keeps its backing allocation alive, so it stays
        valid after close().
        """
        return self.get_sharded_wrapped(tensor_name, dim, device, dtype).get_raw()

    def get_tensor_wrapped(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Return a wrapped tensor by name.

        The returned tensor keeps its backing allocation alive, so it stays
        valid after close().
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
        The returned tensor keeps its backing allocation alive, so it stays
        valid after close().
        """
        return self.get_tensor_wrapped(tensor_name, device, dtype).get_raw()

    def get_and_remove_tensor_wrapped(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Consuming counterpart of get_tensor_wrapped().

        Returns the wrapped tensor and marks the name consumed: it is removed
        from keys() and any later reusable or consuming acquisition of the same
        name raises TensorConsumedError. The returned tensor retains shared
        ownership of the backing allocation, so no lifetime-related clone is
        required and it stays valid after close().

        The name is marked consumed only after the tensor is materialized (and,
        for distributed loads, after the collective completes) successfully; if
        acquisition raises, the name stays available for retry.
        """
        # get_sharded_wrapped runs _require_available first, so a consumed or
        # unknown name raises here before we mark anything.
        t = self.get_sharded_wrapped(tensor_name, -1, device, dtype, consume=True)
        self._consumed_keys.add(tensor_name)
        return t

    def get_and_remove_tensor(
        self,
        tensor_name: str,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """Consuming counterpart of get_tensor(); see get_and_remove_tensor_wrapped().

        Prefer this over get_tensor() for one-shot access: it makes the intent
        to consume explicit and lets the buffer report the name as no longer
        available.
        """
        return self.get_and_remove_tensor_wrapped(tensor_name, device, dtype).get_raw()

    def get_and_remove_sharded_wrapped(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Consuming counterpart of get_sharded_wrapped().

        Returns the wrapped shard and marks the name consumed (see
        get_and_remove_tensor_wrapped for the consuming contract).
        """
        t = self.get_sharded_wrapped(tensor_name, dim, device, dtype, consume=True)
        self._consumed_keys.add(tensor_name)
        return t

    def get_and_remove_sharded(
        self,
        tensor_name: str,
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> Any:
        """Consuming counterpart of get_sharded()."""
        return self.get_and_remove_sharded_wrapped(
            tensor_name, dim, device, dtype
        ).get_raw()

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
        The returned tensor keeps its backing allocation alive, so it stays
        valid after close().
        """
        rank, lidix = self._require_available(tensor_name)
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
        consume: bool = False,
    ) -> TensorBase:
        """Return concatenated column shards from tensor_names.

        The returned tensor keeps its backing allocation alive, so it stays
        valid after close(). When *consume* is True the input names' internal
        references are dropped (see get_and_remove_multi_cols).
        """
        rank_lidixs: Dict[Tuple[int, int], List[str]] = {}
        for tensor_name in tensor_names:
            ranklidx = self._require_available(tensor_name)
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
            ret = ts[0]
        else:
            ret = self.framework.concat_tensors(ts, dim=dim)
        # The result is an independent (concatenated/sharded) tensor; drop every
        # input name's internal reference so the source buffers can be released
        # by shared ownership. Applies to all names, not just the first.
        if consume or self.auto_mem_delete:
            for tensor_name in tensor_names:
                rank, lidx = self._get_rank_lidx(tensor_name)
                self._release_internal(rank, lidx, tensor_name)
        return ret.to(device=device, dtype=dtype)

    def get_and_remove_multi_cols(
        self,
        tensor_names: List[str],
        dim: int,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        """Consuming counterpart of get_multi_cols().

        Marks every input name consumed after the concatenation completes.
        """
        for tensor_name in tensor_names:
            self._require_available(tensor_name)
        ret = self.get_multi_cols(tensor_names, dim, device, dtype, consume=True)
        self._consumed_keys.update(tensor_names)
        return ret

    def as_dict(self, tensor_shard_dim: OrderedDict[str, int]) -> Dict[str, TensorBase]:
        """Return tensors keyed by name according to the requested shard dims.

        Returned tensors keep their backing allocation alive, so they stay
        valid after close().
        """
        tensors: Dict[str, TensorBase] = {}
        for tensor_name, dim in tensor_shard_dim.items():
            rank, lidx = self._require_available(tensor_name)
            loader = self.rank_loaders[rank][lidx]
            tensors[tensor_name] = loader.shuffle(self.pg, tensor_name, dim)
            # The returned tensor (a distributed shard, or a zero-copy view that
            # keeps its own allocation reference) lets us drop the factory's
            # internal reference under auto_mem_delete.
            if self.auto_mem_delete:
                self._release_internal(rank, lidx, tensor_name)
        return tensors
