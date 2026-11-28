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


Experimental Spyre-native checkpoints
-------------------------------------

`copier_type="spyre"` loads experimental `spyre.native.v1` checkpoints through
tensor-owned storage and a two-slot pinned pipeline. It requires PyTorch,
torch-spyre's buffer APIs and a single-process loader group
(`ParallelLoader(all_local=True)`). Native outputs cannot be sliced, cast or used
in collectives. CPU loading and fastsafetensors builds need no Spyre dependencies.

### Format and admission

`spyre.native.v1` is an experimental candidate, not an upstream safetensors
standard. Special values, rounding and cross-SDK compatibility are not yet
qualified; do not assume portable cached artifacts.

```json
{
  "profile": "spyre.native.v1",
  "spyre.overrides": "{\"tensors\":{\"weight\":{\"logical_shape\":[128,150],\"logical_dtype\":\"F16\"}}}"
}
```

Every tensor in a profiled shard is a `U8 [nbytes]` carrier with an exact storage
key in `spyre.overrides`. Ordinary tensors belong in separate unprofiled files;
additional application metadata may coexist. No URL, plugin, executable recipe,
arbitrary stride or runtime layout dump is accepted. Legacy `fst.native.*`
metadata must be regenerated, not reinterpreted.

`logical_shape` is required. Defaults are BF16, `dlfloat16` and `stick-last.v1`;
FP16 therefore records `logical_dtype="F16"`. FP32/`ieee-fp32` and I32/`int32`
are also admitted. I64 is not: on the tested runtime, allocation/DMA size disagrees
with the derived I32 payload for widths such as 65. I64, BOOL, FP8/QFP8WT, empty
tensors and other layouts stay portable and use the framework copy path.

Optional `target={"tp_size":N,"tp_rank":R}` identifies an already-prepared rank;
omission means TP1/rank0. These fields never request slicing. Pass matching
`tp_size`/`tp_rank` to the loader. Model-level identity policy belongs to the caller.

Limits are 16 logical axes, extents 1..2^31-1, 4 GiB per native tensor, 100,000
tensors per shard and 1 TiB aggregate native bytes per load plan. JSON is decoded
with Python's standard `json` module, including its last-value-wins handling of
duplicate keys; there are no additional JSON depth, key-length or nested-metadata
size limits. File/buffer readers retain the 100,000,000-byte header limit and the
existing offset, tensor-size and buffer-bound checks. Native schema, carriers,
target and aggregate budget are checked across the load plan before device
allocation or DMA.

### Fixed geometry

Each stick is 128 bytes (64 DLFloat16 lanes or 32 FP32/I32 lanes). The logical
last axis splits into outer sticks and inner lanes. Physical axes are
`[logical axis 1, ..., last outer-stick axis, logical axis 0, lane]` for rank >=2;
rank1 is `[outer stick, lane]`. Scalars occupy one stick. Singleton logical axes
have zero runtime stride-map entries; logical strides are canonical dense strides.
For `[128,150]` with DLFloat16, physical size is `[3,128,64]`, totaling 49,152 bytes,
not row-major `[128,192]`.

`tests/spyre/test_profile_conformance.py` compares derived/runtime layouts,
independently packs FP32/I32 fixtures, checks padding with two initializations and
round-trips admitted dtypes. DLFloat16 is not IEEE FP16/BF16. The tested runtime
passes finite fixtures `0, 1, -1, 0.5, 2, 1.5, -2`, including `1 = 0x3e00`
little-endian. Special values, rounding, overflow and underflow remain unqualified;
this is not a frozen interchange contract without those tests and SDK/DSO identities.

### Framework and transfer ownership

The Python adapter owns Tensor validation, allocation, storage handles and layout
JSON helpers. The copier uses `TensorBase`. fastsafetensors has no Torch, backend,
Flex or deeptools build dependency. The backend supplies two buffer functions:
`copy_buffer_ranges([(source, address_handle, physical_offset, length)], stream=...)`
and `convert_buffers([(output, input, logical_shape, dtype)])`.

Loading uses buffered `preadv` and two `pin_memory=True` slots. `bbuf_size_kb`
is their total exposed capacity (default 16 MiB), rounded down to two page-aligned
slots; at least two pages are required. Backing-cache over-allocation and idle
retention are not bounded by this setting. Pageable sources are rejected, not
staged. Export length and backing capacity are checked separately.

DMA uses physical byte ranges and the caller's captured stream. The reader
enqueues up to two owning tickets before waiting; single/tail work progresses
without `wait_read()`. Slots become reusable only after completion is proved.
Direct `copy_raw_tensors()` remains blocking. Existing stream-wide synchronization
may complete later work too; each ticket retains its exports, pinned storage,
destination handles and stream until its completion is checked. This binding holds
the GIL, so actual overlap/performance is not qualified.

Cancel/close drain started work. Partial enqueue failure with a successful drain
raises `TransferDrainedError` without poisoning. Unknown completion quarantines
owners for process lifetime and poisons future copies; remaining tickets retain
owners without synchronizing again. Restart the process, never retry/reset or
replace libraries. Metadata and selected destinations are preflighted across files
before I/O; later resource/submission failures still require drain.

Use one device per process and handles/mappings from the same process, device and
live runtime. Creation PID/device/runtime generation and storage identity are not
exposed or automatically checked. Inherited/stale objects are forbidden. Never
resize/rebind storage with a live handle, mutate storage before DMA completion,
or reset/tear down a runtime with live handles/mappings/work. Storage ownership
does not protect against caller mutation.

Conversion accepts little-endian dense input, outermost-first logical shapes and
F16/BF16/F32/I32 dtype names. The backend derives the standard layout (DLFloat16 for
F16/BF16); callers do not construct physical shapes or SDK descriptors. Conversion
is synchronous on the calling thread with the GIL released; callers choose
parallelism across independent calls. Fixed-profile tile/output planning remains
in the loader. Compact read/cast tensors are each <= `workspace_bytes` and may
coexist; no device Tensor is allocated. Output mmaps/page cache are outside that
budget. Single-chunk base allocations are limited to 4 GiB; automatic
`device_memory_budget` planning is unsupported. Older backends fail explicitly,
with no whole-tensor fallback.

### Standalone conversion and loading

The [standalone example](../examples/load_spyre.py) needs PyTorch, torch-spyre's
two buffer APIs and the matching SDK, but not spyre-inference, vLLM or Transformers.
It does not patch installed libraries. Build fastsafetensors normally with
`python -m pip install .`; source weights must already be local.

```bash
python examples/load_spyre.py convert /path/to/gemma-snapshot /output/gemma/model \
  --device spyre:0 --dtype F16 --num-shards 16 --workspace-bytes 8388608

python examples/load_spyre.py load /output/gemma --device spyre:0 \
  --host-buffer-mib 16 --batch-mib 3072 --verify-source /path/to/gemma-snapshot
```

`fastsafetensors.native.convert_to_spyre()` accepts a safetensors file, an HF index
or a directory (including snapshot symlinks). It preserves names/shapes and writes
to new paths, atomically **per shard**, not across the set. Floating-point input
becomes **FP16 by default**, including BF16 weights. No architecture/TP transform,
fusion or quantization is performed. Sequential compact read/cast tiles are each
bounded by `workspace_bytes` (default 8 MiB, minimum 512 bytes; both may coexist),
including wide rows. Conversion performs no DMA. Output mappings/page cache are
**not an RSS limit** and are separate from this workspace and loader slots.

The `ParallelSaver` helpers for encoding with an already allocated Spyre
destination (`native_entry`, `native_fill`, `native_shard_metadata`, and
`split_native_entries`) are available from `fastsafetensors.native`. They are
separate from the bounded standalone converter, which allocates no device tensor.

The `load` command uses `ParallelLoader` and releases weights after each delivery
chunk, so models larger than one card can be checked. `--batch-mib` must fit the
largest whole tensor. `--verify-source` checks every logical value against an
independent normal torch-spyre CPU-to-device-to-CPU copy, accounting for backend
rounding; omit it to load without comparison. This is not a performance benchmark.
Use `ParallelLoader` directly and retain tensors when integrating into a model.
`max_threads` bounds parallel `preadv`; `max_copy_block_size` limits each syscall.

Optional hardware tests (not CPU qualification):

```bash
TEST_FASTSAFETENSORS_FRAMEWORK=pytorch TEST_FASTSAFETENSORS_SPYRE=1 \
  python -m pytest -q tests/spyre
```
