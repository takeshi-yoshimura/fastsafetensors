# Windows Wheel Import Failure in the 0.3.3 Release Build

## Summary

The first `0.3.3` release workflow failed only for Windows wheels during the
`tests/smoke_import_cpp.py` import check. Linux x86_64 and Linux aarch64 wheels
completed successfully. The `0.3.3` tag was deleted pending this investigation.

The failure happens while importing the compiled `fastsafetensors.cpp` extension:

```text
ImportError: DLL load failed while importing cpp: The specified module could not be found.
```

This is before `cpp.load_library_functions("")` is called, so the immediate
failure is at Windows loader time for the `.pyd` or one of its native DLL
dependencies.

**Confirmed (2026-07-07):** the hosted runner image changed between the last
successful run and the failing run — from `windows-2025` to
`windows-2025-vs2026`, i.e. a new image with the Visual Studio 2026 toolchain.
See "Confirmed Runner Image Difference" below. The working hypothesis has been
revised accordingly: this is most likely a **build-toolchain change producing a
new MSVC C runtime DLL dependency**, not a DirectX runtime removal.

## Relevant Runs

Successful Windows wheel build:

- Run: https://github.com/foundation-model-stack/fastsafetensors/actions/runs/26941643302
- Commit: `f5f5911aff1903f847ebdeadb9335f7310c45edb`
- Date: 2026-06-04
- Result: success (`cp310`–`cp314` `win_amd64` all successful)

Failed Windows wheel build:

- Run: https://github.com/foundation-model-stack/fastsafetensors/actions/runs/28836212587
- Commit: `02dd37fb56ee692778856b075bf20275c3f8c0b7`
- Release branch/tag context: `0.3.3` (tag since deleted)
- Date: 2026-07-07
- Result: failure on all `win_amd64` wheels; Linux x86_64, Linux aarch64, and
  sdist succeeded

Both runs used the `windows-latest` label (verified via the GitHub API job
metadata).

## Failure Log Excerpt

```text
Successfully installed annotated-doc-0.0.4 colorama-0.4.6
fastsafetensors-0.3.2 markdown-it-py-4.2.0 mdurl-0.1.2
pygments-2.20.0 rich-15.0.0 shellingham-1.5.4 typer-0.26.8
+ python D:\a\fastsafetensors\fastsafetensors/tests/smoke_import_cpp.py
Traceback (most recent call last):
  File "D:\a\fastsafetensors\fastsafetensors\tests\smoke_import_cpp.py", line 44, in <module>
    main()
  File "D:\a\fastsafetensors\fastsafetensors\tests\smoke_import_cpp.py", line 35, in main
    cpp = _load_cpp_extension()
  File "D:\a\fastsafetensors\fastsafetensors\tests\smoke_import_cpp.py", line 22, in _load_cpp_extension
    module = importlib.util.module_from_spec(spec)
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 1176, in create_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
ImportError: DLL load failed while importing cpp: The specified module could not be found.
Error: cibuildwheel: Command python D:\a\fastsafetensors\fastsafetensors/tests/smoke_import_cpp.py failed with code 1.
```

Note: the `fastsafetensors-0.3.2` install line was caused by a version bump
mistake in `pyproject.toml` during the first 0.3.3 release attempt. That has
been fixed locally (`version = "0.3.3"`). The import failure itself is
independent of the package version string.

## Confirmed Runner Image Difference

From the "Set up job" log headers of both runs:

| | Successful (2026-06-04) | Failed (2026-07-07) |
|---|---|---|
| Runner version | 2.334.0 | 2.335.1 |
| OS | Windows Server 2025, 10.0.26100 | Windows Server 2025, 10.0.26100 |
| **Image** | **`windows-2025`** | **`windows-2025-vs2026`** |
| Image version | win25/20260525.149 | win25-vs2026/20260628.158 |

Key observations:

- The OS build is **identical** (10.0.26100). This rules out an OS migration
  removing DirectX runtime components.
- The image changed to a **Visual Studio 2026** variant. cibuildwheel builds
  with whatever MSVC toolset setuptools discovers on the machine, so the
  failing wheels were compiled and linked with a **different, newer MSVC
  toolchain** than the passing ones. The wheel binary itself differs between
  the two runs even though the C++ sources are effectively unchanged.

This answers former open questions 1–3: the environment did change, but the
change was the VS toolchain, not a Windows Server 2022→2025 jump (that
migration had already completed in late 2025).

## Revised Hypothesis

Most likely: the `.pyd` built by the VS2026 toolchain imports a **new MSVC C
runtime satellite DLL** that is not present (or not current) in `System32` on
the image or on end-user machines. Historical precedents for exactly this
failure mode and error message:

- VS2019 introduced `vcruntime140_1.dll` (x64) — binaries built with it failed
  to load on machines with an older redistributable.
- VS2019 16.8 introduced `msvcp140_atomic_wait.dll` with the same symptom.
- `actions/runner-images#10396` — ONNX import failure after a runner image
  update.

Supporting evidence:

- The error is "The specified **module** could not be found" (a DLL *file* is
  missing), not "the specified procedure could not be found" (an export is
  missing from an older DLL).
- All Python versions failed identically → shared native dependency, not a
  per-interpreter issue.
- `d3d12.dll` / `dxgi.dll` are demoted as suspects: they are standard OS
  components, the OS build is identical between runs, and loading them does
  not require a GPU. If the image had actually dropped them, `runner-images`
  would be flooded with reports — none were found.

Repository-side causes are effectively ruled out: the only C++ changes between
`f5f5911` and `02dd37f` (`ext.cpp`, `gpu_compat.h`) were inspected and are all
`dlopen`/`dlsym`-based with the GDS library path compiled out on Windows
(`gdsLib = nullptr`); they add no link-time or import-time dependencies.

## Implication for Released Wheels

This is not just a CI problem. If the missing DLL is a new CRT satellite,
**wheels built on the vs2026 image would fail the same way on end-user machines
that lack the newest VC redistributable** — even if CI happened to pass. The CI
smoke test acted as a canary. The durable fix is to bundle the CRT dependencies
into the wheel (delvewheel), which is standard practice for distributing
Windows wheels and is what auditwheel already does for the Linux wheels.

## Fixes

### 1. Dynamic loading of D3D12/DXGI (applied locally)

- Remove static links to `d3d12`, `dxgi`, `dxguid`, `uuid`, and `ole32` from
  `setup.py` and the corresponding `#pragma comment(lib, ...)` lines.
- Load `d3d12.dll` / `dxgi.dll` with `LoadLibraryExW(...,
  LOAD_LIBRARY_SEARCH_SYSTEM32)` inside `init_dstorage()`; resolve
  `D3D12CreateDevice` / `CreateDXGIFactory1` via `GetProcAddress`; define the
  required COM IIDs locally.

Reviewed: the four locally defined IIDs match the official GUID values; no
remaining references to the removed import libraries exist (`__uuidof` appears
only in `dstorage.h` comments; no `CoInitialize`/`CoCreateInstance` calls; COM
methods go through vtables and need no import library).

Caveat: this shrinks the import-time dependency surface (good hygiene for
GPU-less machines) but **does not fix the failure if the missing DLL is a CRT
satellite**, which is now the leading hypothesis.

### 2. delvewheel repair in cibuildwheel (applied to `publish.yaml`)

```yaml
CIBW_BEFORE_BUILD_WINDOWS: "pip install delvewheel"
CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: "delvewheel repair -w {dest_dir} {wheel}"
```

This vendors non-system dependent DLLs (notably the MSVC runtime satellites)
into `fastsafetensors.libs/` inside the wheel. delvewheel normally patches the
package `__init__.py` to register that directory; since
`tests/smoke_import_cpp.py` deliberately loads the extension without importing
the package, the smoke test now calls `os.add_dll_directory` on any `*.libs`
directory itself.

### 3. Temporary import-table diagnostic (applied to `publish.yaml`)

`tests/dump_pyd_imports.py` (requires `pefile`, wired via
`CIBW_TEST_REQUIRES_WINDOWS` / `CIBW_TEST_COMMAND_WINDOWS`) prints the DLL
import table of the built `.pyd` and whether each entry resolves from
`System32`, the extension directory, or the Python installation. Running this
once on the failing configuration identifies the missing DLL by name. Remove
the step once the root cause is recorded here.

## Verification Plan

1. Trigger `workflow_dispatch` on `main` with the fixes above.
2. Read the `dump_pyd_imports.py` output in the Windows jobs: confirm which
   DLLs the VS2026-built `.pyd` imports and which one was previously
   unresolved. Record it in this document.
3. Confirm the smoke import passes on `windows-latest` (vs2026 image).
4. Optionally cross-check by pinning `runs-on: windows-2025` (the pre-vs2026
   image label) on the previously failing commit — it should pass there
   without any fix, confirming the toolchain-change theory. Note: this label
   will presumably also move to vs2026 eventually, so pinning is a
   short-term mitigation only, not the fix.
5. Re-tag `0.3.3` once green.

## Remaining Open Questions

1. Which exact DLL was unresolved? (Answered by the `dump_pyd_imports.py`
   output on the next Windows run.)
2. Does delvewheel vendor it? (Expected yes for CRT satellites; if the missing
   DLL turns out to be something delvewheel excludes, revisit.)
3. Does the `windows-2025` label still map to the pre-vs2026 image, and for
   how long?
