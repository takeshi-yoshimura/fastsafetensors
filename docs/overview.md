Overview
=========

# Features

Fastsafetensors introduces three major features to optimize model loading performance:
1. Batched, lazy tensor instantiation.
2. GPU offloading for sharding, type conversions, and device pointer alignment.
3. GPU Direct Storage enablement for file loading from storage to GPU memory.

A major design difference from the original safetensors file loader is that fastsafetensors does *NOT* use `mmap`.
The original loader loads tensors on demand from memory-mapped files,
but unfortunately, it cannot fully utilize high-throughput I/O such as NVMe SSDs.
Therefore, we asynchronously transfer files in parallel to saturate storage throughput.
The loader then lazily instantiates tensors in GPU device memory with DLPack.

Another design change is to offload sharding and other tensor manipulations to GPUs.
The original loader provides slicing for sharding in user programs before copying to device memory. However, it incurs high CPU usage for host memory accesses.
Therefore, we introduce special APIs to run sharding with `torch.distributed` collective operations such as `broadcast` and `scatter`.
The offloading is also applied to other tensor manipulations such as type conversions.

The above two designs can be naturally extended to utilize device-to-device data transfers with GPU Direct Storage.
The technology helps minimize copy overheads from NVMe SSDs to GPU memory by bypassing host CPU and memory.

# Basic API usage

`SafeTensorsFileLoader` is a low-level entrypoint. To use it, pass either `SingleGroup()` for simple inference or `ProcessGroup()` (from `torch.distributed`) for tensor-parallel inference. The loader supports both CPU and CUDA devices, with optional GPU Direct Storage (GDS) support. You can specify the device and GDS settings using the `device` and `nogds` arguments, respectively. If GDS turns out to be unavailable at runtime (e.g., file handle registration fails), the loader logs a warning and falls back to the bounce-buffer (`nogds`) path instead of failing; you can also set `nogds=True` explicitly to skip GDS initialization. For more information on enabling GDS, please refer to the NVIDIA documentation.

The nogds reader keeps a fixed set of workers for the loader's lifetime. Each
worker alternates two bounce buffers and GPU streams to overlap `pread` with
host-to-device copies. `bbuf_size_kb` is the total host buffer budget, rounded
to whole KiB per slot and divided across `max_threads` workers and their two
buffers. For example, 8 workers and `bbuf_size_kb=256*1024` provide sixteen
16 MiB slots. Closing the loader drains outstanding copies and releases its
workers, streams and host buffers.

The nogds and unified-memory copiers prefer `O_DIRECT` on platforms that expose
the flag, except on known network filesystems, where buffered I/O preserves
kernel readahead and client caching. The nogds copier also reads RAM-backed
filesystems (tmpfs, ramfs) buffered, since its bounce buffers gain nothing from
bypassing a page cache that is the storage itself. It also reads ZFS buffered:
OpenZFS 2.3 and later serve a direct read one whole record at a time without
data prefetch, so a read smaller than the recordsize or not aligned to it
fetches every record it touches in full and demand-reads the partial edge
records through the ARC as well. With `recordsize=1M` the nogds copier's
default 512 KiB pieces read 3 to 4x the payload from disk and load 2 to 3x
slower than buffered; at the 128K default they still read 1.3x and load 25%
slower. On OpenZFS before 2.3, or a dataset with `direct=disabled`, the two
paths are equivalent. The unified-memory copier keeps direct reads on ZFS: its
16 MiB pieces loaded twice as fast as its mmap fallback there. The GDS copier
chooses `O_DIRECT` from the CUDA version; FGDS always uses it. Set
`FASTSAFETENSORS_ODIRECT=0` to opt into buffered reads, or
`FASTSAFETENSORS_ODIRECT=1` to prefer direct I/O regardless of filesystem type
(the `EINVAL` fallback below still applies); the one switch covers the nogds
and unified-memory copiers. Filesystem selection follows symlinks to the
checkpoint's actual location. The selection does not probe whether the file
accepts `O_DIRECT`. When a filesystem rejects it with `EINVAL`, the nogds
copier warns once and continues buffered: a rejected open is retried without
the flag, and a rejected read clears the flag on the descriptor and retries,
so a filesystem outside the lists does not fail a load that buffered I/O
would have completed. Other read failures on an `O_DIRECT` descriptor name
the flag and the `FASTSAFETENSORS_ODIRECT=0` switch in their error message.

When the nogds copier reads a checkpoint on a network filesystem with
`O_DIRECT` (so only with `FASTSAFETENSORS_ODIRECT=1`, since network
filesystems read buffered by default), it changes two defaults, chosen from
the first file it opens. Each worker reads through its own descriptor: the
Linux NFS client takes and drops a reference on a lock context, kept per open
file and process, for every page of a direct read, so 16 workers sharing the
copier's descriptor contend on one refcount. A worker reopens the file through
`/proc/self/fd` the first time it reads a request (`dup()` would share the
open file description); the descriptor closes with the request, and a failed
reopen falls back to the shared one. The bounce buffer also grows from 16 MiB
to 256 MiB of pinned host memory, so with 16 workers each read is 8 MiB
instead of 512 KiB and fewer round trips are needed. On NFS over RDMA
(200 Gb/s HDR, kernel 5.15), one process loading a 141 GB checkpoint through
`ParallelLoader` with its default arguments went from 9.6-10.2 GB/s to
19.5-21.4 GB/s and used about a quarter of the CPU; eight processes went from
21.8 to 23.7 GB/s. Each change alone stops short: per-worker descriptors with
512 KiB reads reached about 16 GB/s, and 8-16 MiB reads through a shared
descriptor 10-12 GB/s. An explicit `bbuf_size_kb` is kept, and
`FASTSAFETENSORS_FD_PER_THREAD=1` or `=0` forces the descriptors on or off for
any filesystem. Local filesystems keep no per-page lock context and keep the
previous defaults. The descriptors apply when the nogds copier does not use
mmap and are not available on Windows.

On Linux with libnuma and known GPU NUMA topology, `set_numa=True` places
nogds workers on the GPU's CPU node and prefers that node for their memory.
Pinned host buffers are allocated on a separate thread with the same policy,
leaving the caller's CPU affinity and memory policy unchanged. Other memory
nodes remain available if the preferred node is full. Set `set_numa=False`
to retain inherited placement; CPU readers and unavailable NUMA support also
retain inherited placement.

After creating a `SafeTensorsFileLoader` instance, map files to ranks with `.add_filenames()`, then call `.copy_files_to_device()` to load them and return a `FilesBufferOnDevice`. Retrieve whole tensors with `.get_tensor()` or distributed slices with `.get_sharded()`.

## Lifetime contract

By default, low-level tensor retrieval takes shared ownership of the backing allocation. Tensors and derived views stay valid after the buffer's `.close()`, which drops its own references. The allocation is released when its last owner disappears. A small view can retain an entire file or chunk allocation; `tensor.numel() * tensor.element_size()` does not describe all the memory it retains.

`ParallelLoader` preserves its existing delivery behavior: single-process outputs are cloned into independent storage. Distributed outputs own their receive or broadcast-run storage. Retaining an output does not keep its loader chunk alive, and the existing resident and yield-clone budget accounting is preserved.

For explicit borrowed access, use `copy_files_to_device(borrowed_tensors=True)` in a single-process low-level loader. Tensors and derived views do not hold an allocation owner. Complete every read, including asynchronous device work, before closing the buffer; aliases left in Python do not postpone release. Owning and borrowed storage are selected at materialization, not by `detach()` or slicing an owning tensor.

`ParallelLoader(..., borrowed_tensors=True, accumulate_resident=False)` likewise skips yield clones and returns non-owning views. Complete their use before requesting the next tensor or closing the iterator. Copy into independent storage if data must survive. This mode requires a single-process loader group (`pg=None` or `all_local=True`). Close the iterator on early exit before closing the loader.

The loader's own `close()` releases registrations and copier resources. Close
the low-level buffer, or exhaust/close the `ParallelLoader`/`AutoLoader` iterator,
before closing the loader. `loader.close()` does not close an active iterator.

`live_allocation_count()` / `live_allocation_bytes()` report the load allocations
fastsafetensors still owns, including allocations retained by exported owning
storage. They exclude independent clones, other framework tensors, pinned host
pools and allocator caches. `get_framework_op("pytorch").get_mem_used()` likewise
tracks load-buffer bytes, rather than total GPU memory. In owning mode, closing
every buffer is not enough to return these counters to zero: release all owning
tensors and their derived storage too.

`fastsafe_open` is an easier entrypoint. You can force GDS off and run in fallback mode if `nogds=True`.

```python
with fastsafe_open(filenames=[filename], nogds=True, device="cpu", debug_log=True) as f:
    for key in f.keys():
        t = f.get_tensor(key)  # stays valid after the block; no clone needed
```

# AutoLoader configuration

`AutoLoader` supports file-based configuration for loader type, pipeline mode, copy settings, and more.
See [Configuration Guide](./configuration.md) for defaults, examples, and all available options.

# Saving checkpoints

`ParallelSaver` saves a mapping of native framework tensors as size-balanced
safetensors shards. The destination is a filename prefix; returned paths can be passed
to `ParallelLoader`. Settings can be reused across saves, while metadata is
supplied for each checkpoint. Saves finish and release their file resources
before returning; no explicit close is needed.

```python
from fastsafetensors import ParallelSaver

saver = ParallelSaver(num_shards=8, num_threads=8, framework="pytorch")
paths = saver.save(tensors, "/cache/model", metadata={"version": "1"})
# /cache/model-00001-of-00008.safetensors, ...
```

Use `save_entries` for `WriteEntry` descriptions containing the on-disk dtype,
shape, byte size, and an opaque `source`. Pass a `fill` callback to `ParallelSaver`
to encode entries directly into the supplied `(buffer, entry)` pairs. Determine
encoded byte sizes before shard planning; variable-size encoders must prepare
their payloads first. Application-specific formats, layout metadata, and
compatibility checks belong to the caller.

`num_threads` controls default framework tensor writes; custom callbacks
control their own parallelism. A fill callback must complete all writes before
returning and release references to the supplied buffers.
`shard_metadata` optionally computes string metadata from each shard's entries;
fastsafetensors stores these values without interpreting them.

`metadata` is repeated in every shard; `shard0_metadata` is stored only in the
first shard. The lower-level `plan_shards`, `write_shards`, and `save_sharded`
functions remain available for custom planning and fill callbacks. Completed
shards are published by individual file renames; the set is not published
atomically.

`save` accepts native PyTorch tensors by default; use `framework="paddle"`
for native Paddle tensors. Applications can pass the tensor objects they
already hold. The saver adapts them internally without copying their storage
and without importing a tensor framework until native tensor access is needed.
Existing `TensorBase` wrappers from `get_tensor_wrapped` are also accepted.
Custom fill and shard metadata callbacks receive the original `source` objects,
including native tensors, without wrapping. `save_entries` with a custom fill
callback accepts arbitrary sources and does not load a framework adapter.
Device tensors are copied to host memory before files are persisted;
GPU Direct Storage writing is not implemented.

# ROCm

On ROCm, direct storage-to-GPU loading is supported through hipFile (ROCm >= 7.2): when `libhipfile.so` is available, the GDS code path uses it transparently. On older ROCm without hipFile, the loader falls back to the bounce-buffer (`nogds`) path.
A performance gain example with the `nogds` path can be found at [amd-perf.md](./amd-perf.md).

# Windows

From [PR#72](https://github.com/foundation-model-stack/fastsafetensors/pull/72):

On Linux, GDS uses cuFile to DMA data directly from NVMe into GPU memory. Windows has no cuFile — instead, it offers [DirectStorage](https://devblogs.microsoft.com/directx/directstorage-api-available-on-pc/), a DirectX 12 API designed for the same purpose.

Since DirectStorage writes into D3D12 resources (not CUDA buffers), we bridge the two APIs through CUDA external memory interop:

```
NVMe -> [DirectStorage] -> D3D12 shared buffer -> [cudaImportExternalMemory] -> CUDA device pointer
```

The key steps are:

1. Create a D3D12 committed resource with D3D12_HEAP_FLAG_SHARED so it can be exported
2. DirectStorage reads from NVMe into this D3D12 buffer via IDStorageQueue
3. Export the D3D12 resource as an NT handle via CreateSharedHandle
4. Import into CUDA via cudaImportExternalMemory + cudaExternalMemoryGetMappedBuffer to get a regular CUDA device pointer
5. Synchronize using a D3D12 fence imported as a cudaExternalSemaphore

All DirectStorage, D3D12, and DXGI libraries are loaded at runtime via LoadLibrary/GetProcAddress — no link-time SDK dependency on DirectStorage is required.

If the DirectStorage DLLs are not installed, the Windows loader falls back to
the bounce-buffer (`nogds`) path. Set `FASTSAFETENSORS_DSTORAGE_DLL_DIR` to an
absolute directory containing `dstoragecore.dll` and `dstorage.dll` to enable
DirectStorage; an invalid explicit directory remains an error.

The fallback resolves a framework-bundled CUDA runtime first (for example,
PyTorch's `torch\lib` directory), then checks system CUDA Toolkit locations.
`FASTSAFETENSORS_CUDART_LIB` can specify an absolute runtime DLL explicitly and
takes precedence over automatic discovery.


Spyre-native checkpoints
------------------------

`copier_type="spyre_native"` loads converted checkpoint bytes into Spyre tensors.
The checkpoint remains a standard safetensors file; its string metadata records
the device format and logical tensor description. `fastsafetensors.native` owns
the shared metadata convention: `NativeFormat`, `NativeTensorSpec`,
`native_metadata`, `native_format`, and `native_specs`. Backend layout values
are opaque to that layer. The inference layer owns model identity, TP rank,
cache policy and model traversal.
The Spyre backend owns `torch_spyre/1` checkpoint encoding and raw transfers;
torch-spyre defines the physical layout and supplies conversion primitives.

Use `ParallelLoader(..., device="spyre:0", copier_type="spyre_native",
all_local=True)` so each TP rank reads its local shards. Native tensors own their
storage and are yielded without cloning. Device-native payloads cannot be cast,
moved to a different device, sliced, or broadcast through a multi-rank loader
group. CPU/CUDA loaders retain upstream shared-allocation, readiness, borrowed
tensor and broadcast behavior.

Write converted host weights with the same `ParallelSaver` used for ordinary
tensors, using the Spyre backend's entry and callback helpers:

```python
from fastsafetensors import ParallelSaver
from fastsafetensors.copier.spyre import (
    native_entry, native_fill, native_shard_metadata,
)
from fastsafetensors.saver import plain_entry
from torch_spyre._C import dma_bytes, get_spyre_tensor_layout
from torch_spyre.layout import layout_to_json

# src is a loaded CPU tensor; dst is an allocated Spyre tensor with its final
# logical shape, strides, dtype and physical layout (no data transfer required).
entries = [native_entry(
    "weight", src, dst, nbytes=dma_bytes(dst),
    layout=layout_to_json(get_spyre_tensor_layout(dst)),
)]
# Ordinary CPU tensors can share the same shards.
entries.append(plain_entry("scale", scale))
paths = ParallelSaver(
    fill=native_fill, shard_metadata=native_shard_metadata,
).save_entries(entries, "/cache/weights")
```

The callbacks encode directly into saver-owned buffers. `dst` describes the
conversion target but is not filled by the save. The generic saver still treats
sources and application metadata as opaque; it contains no Spyre-specific logic.
Importing the backend does not import torch-spyre. Native conversion loads that
optional dependency only when needed. The version-1 `fst.native.*` keys and
payload representation are unchanged; model cache invalidation remains the
inference layer's responsibility.

The Spyre backend also provides `TransferRecorder` for deferred tensor transfers.
Calling it with a tensor allocates an eligible fp16 destination without filling
it; `place_with_layout` does the same with an explicit layout. Unsupported
tensors use the ordinary `.to()` path. Call `recorder.fill()` after collecting
the transfers to complete them through the bounded staging pipeline. Do not
read deferred destinations before filling them. `records` exposes each
`TransferRecord`'s source, destination, layout and transfer kind (`defer`, `to`,
or `same`) for callers that instead persist converted tensors. The recorder
retains detached source views so replacing a parameter does not release its
original storage. It owns no model traversal, vLLM state, or cache policy.

`records, entries = recorder.checkpoint_entries()` produces JSON-serializable
tensor replay records alongside entries for `ParallelSaver` (use `native_fill`
and `native_shard_metadata`). The backend owns `KEY_TRANSFERS`
(`torch_spyre.transfers`) and `KEY_HOST` (`torch_spyre.host`). Callers compute
application cache identity; the backend owns its on-disk representation as well.
Records retain the existing `same`/`to`/`raw` representation and tensor names
(`t0`, `t1`, ... by default; `prefix=` changes the prefix).

`TransferRestorer(device, headers, records)` replays those tensor operations;
`headers` maps each saved tensor name to its `SafeTensorsMetadata`. Invoke the
restorer for each tensor in the original order, or use `place_with_layout` for
explicit placement. It validates shape, dtype, placement, raw strides and native
payload descriptions, restores ordinary tensors with their recorded dense
strides, and allocates raw destinations without filling them. After traversal,
call `restorer.finish()` to check record exhaustion, then use `ParallelLoader`
with `destinations=restorer.destinations` and a matching `tensor_filter` to fill
the raw destinations. `finish()` does not perform DMA. Mismatches raise
`NativeFormatError`; discard partially restored state. Model traversal stays
outside this backend.

`tensor_metadata(tensor)` describes a tensor's logical size, stride and dtype.
`read_recorded_tensor(headers, record)` reads an ordinary payload, checks its
shape/dtype against the record and recovers dense strides. These helpers also
support caller-managed host state without importing an inference framework.

For models with CPU-only state, call `recorder.capture_host_state(model)` before
the normal `_apply(recorder)` traversal. Then
`metadata, entries = recorder.checkpoint_metadata(model)` records transfers and
the original tensor paths that still hold CPU tensors. Pass `metadata` as
`shard0_metadata` to `ParallelSaver.save_entries`. Paths cover registered state
(including non-persistent buffers), plain tensor attributes, dictionaries with
string/integer/boolean/None keys, lists, tuples, dataclasses and
`SimpleNamespace` objects. Cycles are not followed; arbitrary object internals
and properties are not inspected. Derived caches created by `_apply` are
regenerated by that traversal, not included in the host recipe.

`TransferRestorer.from_metadata(device, headers, metadata)` reads both backend
keys. Call `restorer.restore_host_state(model)` **before** replaying `_apply`:
some modules derive transfer sources from CPU tensors. Restoration resolves
only recorded paths (no rediscovery or evaluated Python), validates every
target's shape/dtype/device and object-alias group before swapping, and retains
tensor subclasses and Python attributes. Each distinct CPU tensor has one
ordinary payload; the metadata contains placement, paths and descriptions,
not tensor values. Recorded dense strides are recovered; distinct views sharing
storage are not reconstructed as shared storage. Call `check_materialized(model)`
after replay to reject remaining meta state in supported containers.

No vLLM/model imports, model methods, or global `Module.to` patches are required.
An application must construct a compatible skeleton before restoration; the
backend cannot recover values from meta or handle data-dependent construction.
Missing/malformed backend recipes are rejected rather than inferred from old
application metadata. Native payload/package versions remain unchanged.

The backend also owns the cache envelope, separate from the physical payload's
`torch_spyre/1` format. `cache_metadata(key)` returns two dictionaries:
metadata repeated in every shard and metadata stored only in shard zero. Pass
them to `ParallelSaver.save_entries` as `metadata=` and `shard0_metadata=` (merge
the latter with the transfer/host metadata). It writes:

| Key | Value |
| --- | --- |
| `torch_spyre.cache.format` | `torch_spyre-cache/3` |
| `torch_spyre.cache.key` | Canonically encoded caller-provided identity |
| `torch_spyre.cache.fp8` | Reserved payload recipe; currently `[]` only |

`read_cache_metadata(metadata, key)` checks format, identity and the absence of
unsupported FP8 payload recipes. It returns `None` on success and raises
`NativeFormatError` on mismatch. Compatibility relies on the caller's cache key
and tensor-restoration checks; plain model attributes are not saved or compared.
These helpers do not import an inference framework or compute model identity,
walk model attributes, or decide which kernels can be cached. Old
`spyre_inference.*` envelope keys are not accepted. The schema revision remains
3; physical payload and package versions are unchanged.
