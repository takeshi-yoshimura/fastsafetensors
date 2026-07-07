# SPDX-License-Identifier: Apache-2.0

"""Diagnostic run before the Windows wheel smoke test in CI.

Prints the DLL import table of the installed fastsafetensors.cpp extension and
where each imported DLL resolves from, so an import failure names the missing
DLL instead of Windows' generic "module could not be found". Also reports the
MSVC runtime DLLs present in System32 and their versions. Requires pefile.
"""

import importlib.machinery
import importlib.metadata
import os
import sys
from pathlib import Path


def _find_pyd():
    dist = importlib.metadata.distribution("fastsafetensors")
    dist_root = Path(dist.locate_file(""))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = dist_root / "fastsafetensors" / f"cpp{suffix}"
        if candidate.exists():
            return dist_root, candidate
    raise FileNotFoundError(
        f"fastsafetensors.cpp extension not found under {dist_root}"
    )


def main() -> None:
    if os.name != "nt":
        print("dump_pyd_imports: not Windows, nothing to do")
        return

    import pefile

    dist_root, pyd = _find_pyd()
    print(f"extension: {pyd}")

    search_dirs = {
        "System32": Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        "beside pyd": pyd.parent,
        "python dir": Path(sys.base_exec_prefix),
    }
    for libs_dir in dist_root.glob("*.libs"):
        search_dirs[libs_dir.name] = libs_dir

    pe = pefile.PE(str(pyd), fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
    )
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        name = entry.dll.decode()
        # API set names (api-ms-*, ext-ms-*) are resolved virtually by the OS
        # loader, not by file lookup, so absence on disk is not a failure.
        if name.lower().startswith(("api-ms-", "ext-ms-")):
            print(f"  imports {name}: API set (loader-resolved)")
            continue
        found_in = [label for label, d in search_dirs.items() if (d / name).exists()]
        status = ", ".join(found_in) if found_in else "*** NOT FOUND ***"
        print(f"  imports {name}: {status}")

    # The failing wheel was never inspected before delvewheel repair, so also
    # report which MSVC CRT DLLs the image itself provides and their versions;
    # this identifies what the unrepaired .pyd could(n't) have resolved.
    print("MSVC runtime DLLs in System32:")
    crt_names = [
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "concrt140.dll",
    ]
    for crt_name in crt_names:
        path = search_dirs["System32"] / crt_name
        if path.exists():
            print(f"  {crt_name}: {_file_version(pefile, path)}")
        else:
            print(f"  {crt_name}: *** NOT PRESENT ***")


def _file_version(pefile, path):
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )
        info = pe.VS_FIXEDFILEINFO[0]
        ms, ls = info.FileVersionMS, info.FileVersionLS
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception as e:
        return f"present (version unreadable: {e})"


if __name__ == "__main__":
    main()
