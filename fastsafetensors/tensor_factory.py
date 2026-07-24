# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

from . import cpp as fstcpp
from .allocation import SharedDeviceAllocation
from .common import SafeTensorsMetadata, init_logger, is_debug
from .copier.base import CopierInterface, DummyDeviceBuffer
from .frameworks import FrameworkOpBase, ProcessGroupBase, TensorBase
from .st_types import Device, DType

logger = init_logger(__name__)


class LazyTensorFactory:
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        rank: int,
        local_rank: bool,
        factory_idx_bits: int,
        lidx: int,
        copier: Optional[CopierInterface],
        framework: FrameworkOpBase,
        disable_cache=True,
    ):
        self.framework = framework
        self.metadata = metadata
        self.device = device
        self.copier: Optional[CopierInterface] = None
        if local_rank:
            self.copier = copier
        self.tensors: Dict[str, TensorBase] = {}
        self.shuffled: Dict[str, TensorBase] = {}
        self.gbuf: Optional[fstcpp.gds_device_buffer] = None
        # Shared owner of gbuf. Created in wait_io and referenced by every
        # tensor materialized from the buffer, so the backing memory outlives
        # both this factory and any exported tensor. See SharedDeviceAllocation.
        self.allocation: Optional[SharedDeviceAllocation] = None
        self.rank = rank
        self.factory_idx_bits = factory_idx_bits
        self.lidx = lidx
        self.next_tag = 1
        self.disable_cache = disable_cache

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        if self.copier is not None:
            self.gbuf = self.copier.submit_io(use_buf_register, max_copy_block_size)
            if self.gbuf:
                logger.debug(
                    "submit_io: new buf, addr=0x%x", self.gbuf.get_base_address()
                )

    def wait_io(self, dtype: DType = DType.AUTO, noalign: bool = False):
        if self.copier is not None and self.gbuf is not None:
            # Create the shared owner before materializing tensors so that even
            # if wait_io raises, free_dev_ptrs() can release the buffer. Every
            # tensor built from gbuf takes a reference through its DLPack owner.
            self.allocation = SharedDeviceAllocation(
                self.gbuf,
                self.framework,
                self.device,
                owns_memory=not isinstance(self.gbuf, DummyDeviceBuffer),
            )
            self.tensors = self.copier.wait_io(
                self.gbuf, dtype=dtype, noalign=noalign, owner=self.allocation
            )
            if is_debug(logger):
                for name in self.tensors.keys():
                    logger.debug("wait_io: tensor=%s", name)
            self.copier = None

    def push(
        self,
        pg: ProcessGroupBase,
        tensor_name: str,
        dst_rank: int,
        src_rank: int,
    ) -> Optional[TensorBase]:
        if pg.size() == 1:
            return self.tensors[tensor_name]
        tag = (self.next_tag << self.factory_idx_bits) + self.lidx
        self.next_tag += 1
        if pg.rank() != dst_rank and pg.rank() != src_rank:
            logger.debug(
                "push: skip, tensor_name=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                dst_rank,
                pg.rank(),
                tag,
            )
            return None
        elif pg.rank() == dst_rank and src_rank == dst_rank:
            logger.debug(
                "push: nocopy, tensor_name=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                dst_rank,
                pg.rank(),
                tag,
            )
            return self.tensors[tensor_name].clone().detach()
        frame = self.metadata.tensors[tensor_name]
        if pg.rank() == src_rank:
            if tensor_name not in self.tensors:
                raise Exception(
                    f"push: tensor {tensor_name} was not found. released? lidx={self.lidx}"
                )
            t = self.tensors[tensor_name].clone().detach()
            logger.debug(
                "push: send, tensor_name=%s, shape=%s, dst_rank=%d, pg.rank()=%d, tag=%d",
                tensor_name,
                frame.shape,
                dst_rank,
                pg.rank(),
                tag,
            )
            pg.send(t, dst_rank, tag=tag)
            return None

        logger.debug(
            "push: recv, tensor_name=%s, shape=%s, src_rank=%d, pg.rank()=%d, tag=%d",
            tensor_name,
            frame.shape,
            src_rank,
            pg.rank(),
            tag,
        )

        t = self.framework.get_empty_tensor(frame.shape, frame.dtype, self.device)
        pg.recv(t, src_rank, tag=tag)
        return t

    def shuffle(self, pg: ProcessGroupBase, tensor_name: str, dim: int) -> TensorBase:
        if pg.size() == 1:
            # The returned tensor shares the backing gbuf lifetime; public APIs
            # document that callers must clone/copy before buffer close.
            return self.tensors[tensor_name]
        if tensor_name in self.shuffled:
            logger.debug("shuffle: use cache, tensor_name=%s", tensor_name)
            t = self.shuffled[tensor_name].clone().detach()
            return t
        frame = self.metadata.tensors[tensor_name]
        if dim == -1:
            if tensor_name in self.tensors:
                dst = self.tensors[tensor_name].clone().detach()
            else:
                dst = self.framework.get_empty_tensor(
                    frame.shape, frame.dtype, self.device
                )
            logger.debug(
                "shuffle: broadcast, tensor_name=%s, shape=%s, self.rank=%d, pg.rank()=%d, has_tensor=%s",
                tensor_name,
                frame.shape,
                self.rank,
                pg.rank(),
                tensor_name in self.tensors,
            )
            pg.broadcast(dst, self.rank)
        else:
            rank_slices: List[Tuple] = [() for i in range(0, pg.size())]
            size = frame.shape[dim]
            block_size = (size + pg.size() - 1) // pg.size()
            for rank in range(0, pg.size()):
                for i in range(0, len(frame.shape)):
                    if i < dim:
                        rank_slices[rank] += (slice(None, None, None),)
                    elif i == dim:
                        rank_slices[rank] += (
                            slice(rank * block_size, (rank + 1) * block_size, 1),
                        )
                        break
            scatter_list: List[TensorBase] = []
            new_frame = frame[rank_slices[pg.rank()]]
            dst = self.framework.get_empty_tensor(
                new_frame.shape, new_frame.dtype, self.device
            )
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(
                        f"shuffle: tensor {tensor_name} was not found, released? lidx={self.lidx}"
                    )
                t = self.tensors[tensor_name]
                scatter_list = [
                    t[
                        self.framework.get_native_slices(
                            frame.dtype, frame.shape, rank_slices[rank]
                        )
                    ].contiguous()
                    for rank in range(0, pg.size())
                ]  # scatter requires contiguous tensor
            logger.debug(
                "shuffle: scatter, tensor_name=%s, shape=%s->%s, self.rank=%d, pg.rank()=%d, rank_slices=%s, len(scatter_list)=%s",
                tensor_name,
                frame.shape,
                new_frame.shape,
                self.rank,
                pg.rank(),
                rank_slices,
                len(scatter_list),
            )
            pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        if not self.disable_cache:
            # Cache tensor for reuse within the same batch to improve performance.
            # Note: This requires additional (GPU) memory to store the cached tensors.
            # Enable this only if you have sufficient (GPU) memory and required.
            self.shuffled[tensor_name] = dst
        return dst

    def shuffle_multi_cols(
        self, pg: ProcessGroupBase, tensor_names: List[str], dim: int
    ) -> TensorBase:
        rank_tensors: List[List[TensorBase]] = [[] for i in range(0, pg.size())]
        new_shape: List[int] = []
        for tensor_name in tensor_names:
            frame = self.metadata.tensors[tensor_name]
            total_size = frame.shape[dim]
            block_size = (total_size + pg.size() - 1) // pg.size()
            shard_start = pg.rank() * block_size
            shard_stop = min((pg.rank() + 1) * block_size, total_size)
            shard_size = max(shard_stop - shard_start, 0)
            if len(new_shape) == 0:
                new_shape = list(frame.shape)
                new_shape[dim] = 0
            else:
                for dim2 in range(0, len(frame.shape)):
                    if dim2 != dim and frame.shape[dim2] != new_shape[dim2]:
                        raise Exception(
                            f"dim {dim2} mismatch: tensor {tensor_name} has {frame.shape} vs. {new_shape} (dim={dim})"
                        )
            new_shape[dim] += shard_size
            if self.rank == pg.rank():
                if tensor_name not in self.tensors:
                    raise Exception(
                        f"shuffle_multi_cols: tensor {tensor_name} was not found, released? lidx={self.lidx}"
                    )
                t = self.tensors[tensor_name]
                for rank in range(0, pg.size()):
                    rank_slices: Tuple[slice, ...] = ()
                    for i in range(0, len(frame.shape)):
                        if i < dim:
                            rank_slices += (slice(None, None, None),)
                        elif i == dim:
                            rank_slices += (
                                slice(rank * block_size, (rank + 1) * block_size, 1),
                            )
                            break
                    rank_tensors[rank].append(
                        t[
                            self.framework.get_native_slices(
                                frame.dtype, frame.shape, rank_slices
                            )
                        ]
                    )
        if pg.size() == 1:
            return self.framework.concat_tensors(rank_tensors[self.rank], dim=dim)
        scatter_list: List[TensorBase] = []

        if self.rank == pg.rank():
            for rank in range(0, pg.size()):
                scatter_list.append(
                    self.framework.concat_tensors(rank_tensors[rank], dim=dim)
                )
        logger.debug(
            "shuffle_multi_cols: scatter, tensor_name=%s, shape=%s->%s, self.rank=%d, pg.rank()=%d, len(scatter_list)=%s",
            tensor_name,
            frame.shape,
            new_shape,
            self.rank,
            pg.rank(),
            len(scatter_list),
        )
        dst = self.framework.get_empty_tensor(new_shape, frame.dtype, self.device)
        pg.scatter(dst, scatter_list=scatter_list, src=self.rank)
        return dst

    def retain_only(self, names: "set[str]") -> None:
        """Drop internal tensor references not in *names*.

        Called once after construction so the factory only keeps references to
        the tensors that are actually available in the owning buffer (e.g. after
        a tensor filter). Dropped tensors were materialized but never accessible;
        releasing their DLPack owners here reclaims those references early and
        lets the buffer-side allocation be released once the retained set drains.
        """
        if not self.tensors:
            return
        for name in [n for n in self.tensors if n not in names]:
            del self.tensors[name]

    def drop_internal_reference(self, name: str) -> None:
        """Release the factory's internal reference to *name*'s tensor.

        Used when a name is consumed (get_and_remove_*) or handed out under
        auto_mem_delete. Once the factory holds no more internal references, its
        buffer-side allocation reference is released so shared ownership alone
        decides when the physical buffer is freed: exported tensors that still
        reference it keep it alive; otherwise it is freed now. There is no
        explicit "last tensor" free call -- release follows from the reference
        count reaching zero.
        """
        self.tensors.pop(name, None)
        if not self.tensors and self.allocation is not None:
            logger.debug(
                "drop_internal_reference: release buf, addr=0x%x",
                self.allocation.get_base_address() if self.allocation.live else 0,
            )
            self.allocation.release()
            self.allocation = None
            self.gbuf = None

    def free_dev_ptrs(self):
        """Release this factory's reference to the backing allocation.

        This is a logical close: it drops the buffer-side reference and the
        factory's internal tensor references, but does not forcibly free memory
        that exported tensors still hold. The physical buffer is freed by
        SharedDeviceAllocation once its last reference disappears. Idempotent:
        safe to call repeatedly (e.g. from repeated close()).
        """
        self.tensors = {}
        if self.allocation is not None:
            logger.debug(
                "free_dev_ptrs: release buf, addr=0x%x",
                self.allocation.get_base_address() if self.allocation.live else 0,
            )
            self.allocation.release()
            self.allocation = None
            self.gbuf = None
        elif self.gbuf is not None and not isinstance(self.gbuf, DummyDeviceBuffer):
            # submit_io ran but wait_io never created the allocation (e.g. an
            # error before materialization): free the raw buffer directly.
            self.framework.free_tensor_memory(self.gbuf, self.device)
            logger.debug(
                "free_dev_ptrs: delete buf, addr=0x%x", self.gbuf.get_base_address()
            )
            self.gbuf = None
