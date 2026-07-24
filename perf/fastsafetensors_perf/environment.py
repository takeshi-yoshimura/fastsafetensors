"""Host / GPU / version inventory.

Everything here is *detected*, never configured. The detected fields feed two
places: the ``environment`` block (full diagnostics) and the version/GPU/storage
identity fields (comparison compatibility). Detection degrades gracefully: torch,
pynvml, and CUDA may be absent (CPU-only unit host), in which case fields are
empty strings/None rather than errors.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .results import Identity, series_of


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return ""


def _git(args, cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _fastsafetensors_origin() -> Dict[str, Any]:
    """Where does the *imported* fastsafetensors resolve to, and is it the
    intended checkout/build (editable install) or a wheel?"""
    info: Dict[str, Any] = {"version": _version("fastsafetensors"), "path": "", "editable": None}
    try:
        import fastsafetensors

        path = os.path.dirname(os.path.abspath(fastsafetensors.__file__))
        info["path"] = path
        # Editable installs live inside the source tree (a cpp/ dir sits beside
        # the package); wheels land under site-packages.
        info["editable"] = "site-packages" not in path
    except Exception:
        pass
    return info


def _torch_versions() -> Dict[str, str]:
    out = {"torch": "", "cuda": "", "hip": ""}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = getattr(torch.version, "cuda", None) or ""
        out["hip"] = getattr(torch.version, "hip", None) or ""
    except Exception:
        pass
    return out


def _gpu_info(device_index: int = 0) -> Dict[str, Any]:
    """GPU marketing name, UUID, VRAM, compute capability, visible count."""
    info: Dict[str, Any] = {
        "name": "", "uuid": "", "total_memory_bytes": 0,
        "compute_capability": "", "visible_count": 0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["visible_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(device_index)
            info["name"] = props.name
            info["total_memory_bytes"] = int(props.total_memory)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["uuid"] = _safe(
                lambda: str(torch.cuda.get_device_properties(device_index).uuid), ""
            ) or ""
    except Exception:
        pass
    return info


def _storage_id(path: str) -> Dict[str, Any]:
    """Best-effort storage device / filesystem / mount identity for ``path``.

    ``storage_id`` (device:fstype) is an identity field: results on different
    storage are not comparable. Mount options are recorded for diagnostics only.
    """
    info: Dict[str, Any] = {"device": "", "fstype": "", "mount_options": "", "storage_id": ""}
    try:
        target = os.path.abspath(path)
        # Walk up to an existing directory (path may point at a file/alias).
        while target and not os.path.exists(target):
            parent = os.path.dirname(target)
            if parent == target:
                break
            target = parent
        best = {"mount": "", "device": "", "fstype": "", "options": ""}
        with open("/proc/mounts", "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 4:
                    continue
                dev, mount, fstype, options = parts[0], parts[1], parts[2], parts[3]
                if target == mount or target.startswith(mount.rstrip("/") + "/") or mount == "/":
                    if len(mount) >= len(best["mount"]):
                        best = {"mount": mount, "device": dev, "fstype": fstype, "options": options}
        info["device"] = os.path.basename(best["device"]) or best["device"]
        info["fstype"] = best["fstype"]
        info["mount_options"] = best["options"]
        info["storage_id"] = f"{info['device']}:{info['fstype']}"
    except Exception:
        pass
    return info


def _cpu_topology() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 0,
        "numa_nodes": 0,
    }
    try:
        numa_dir = "/sys/devices/system/node"
        if os.path.isdir(numa_dir):
            info["numa_nodes"] = sum(
                1 for n in os.listdir(numa_dir) if n.startswith("node") and n[4:].isdigit()
            )
    except Exception:
        pass
    return info


def _nvlink_topology() -> str:
    """Coarse PCIe/NVLink topology string from nvidia-smi, if available."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _driver_version() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


@dataclass
class Environment:
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.raw

    def gpu_model(self) -> str:
        return self.raw.get("gpu", {}).get("name", "")

    def storage_id(self) -> str:
        return self.raw.get("storage", {}).get("storage_id", "")

    def identity_versions(self) -> Dict[str, str]:
        v = self.raw.get("versions", {})
        return {
            "vllm_series": series_of(v.get("vllm", "")),
            "torch_series": series_of(v.get("torch", "")),
            "cuda_series": series_of(v.get("cuda", "")),
            "fastsafetensors_series": series_of(v.get("fastsafetensors", "")),
        }

    def fill_identity(self, ident: Identity, hardware_profile: str,
                      world_size: int) -> Identity:
        ident.hardware_profile = hardware_profile
        ident.gpu_model = self.gpu_model()
        ident.world_size = world_size
        ident.storage_id = self.storage_id()
        for k, val in self.identity_versions().items():
            setattr(ident, k, val)
        return ident


def collect(model_path: str = ".", repo_dir: Optional[str] = None,
            device_index: int = 0) -> Environment:
    """Collect the full environment inventory."""
    repo_dir = repo_dir or os.getcwd()
    torch_v = _torch_versions()
    fst = _fastsafetensors_origin()
    raw: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_topology(),
        "gpu": _gpu_info(device_index),
        "storage": _storage_id(model_path),
        "nvlink_topology": _nvlink_topology(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "versions": {
            "torch": torch_v["torch"],
            "cuda": torch_v["cuda"],
            "hip": torch_v["hip"],
            "driver": _driver_version(),
            "vllm": _version("vllm"),
            "fastsafetensors": fst["version"],
        },
        "fastsafetensors_origin": fst,
        "git_sha": _git(["rev-parse", "HEAD"], repo_dir),
        "git_dirty": bool(_git(["status", "--porcelain"], repo_dir)),
    }
    return Environment(raw=raw)
