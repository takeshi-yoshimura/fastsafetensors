# SPDX-License-Identifier: Apache-2.0

"""Temporary diagnostic for the 0.3.3 Windows wheel investigation.

Prints the DLL import table of the installed fastsafetensors.cpp extension and
where each imported DLL resolves from, so an import failure names the missing
DLL instead of Windows' generic "module could not be found". Requires pefile.
See windows_runner_issue.md; remove once the root cause is recorded there.
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
        found_in = [label for label, d in search_dirs.items() if (d / name).exists()]
        status = ", ".join(found_in) if found_in else "*** NOT FOUND ***"
        print(f"  imports {name}: {status}")


if __name__ == "__main__":
    main()
