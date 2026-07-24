"""Per-rank resource telemetry sampled across the timed region.

Captures what wall time alone hides: is the loader CPU-bound, is the disk the
ceiling, how much GPU memory and NVLink bandwidth a load actually costs. All
sampling is read-only (`/proc` + NVML) and touches neither the CUDA context nor
any collective, so the sampler thread is safe to run alongside NCCL work (unlike
the loader itself -- see worker._run_once_with_timeout).

Counters that are cumulative (CPU time, disk bytes, NVLink bytes) are read once
at start and once at stop; instantaneous gauges (GPU utilization, GPU memory,
host RSS) are polled by a background thread and reduced to mean/peak.

Everything degrades gracefully: no pynvml, no `/proc/self/io`, or no NVLink
leaves those fields at 0 rather than raising.
"""

from __future__ import annotations

import os
import resource
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


@dataclass
class TelemetryResult:
    cpu_user_pct: float = 0.0        # % of one core, averaged over the window
    cpu_system_pct: float = 0.0
    host_mem_increase_bytes: int = 0  # peak RSS during run - RSS at start
    disk_read_bytes: int = 0          # block-layer reads (0 on tmpfs/cache-hit)
    disk_read_bps: float = 0.0
    read_char_bytes: int = 0          # bytes via read()/pread() syscalls (rchar)
    read_char_bps: float = 0.0        # informative even on RAM/networked FS
    gpu_util_pct: float = 0.0         # mean of polled samples
    gpu_mem_used_bytes: int = 0       # peak device memory used (NVML), not torch
    nvlink_bytes: int = 0             # tx+rx over the window
    nvlink_bps: float = 0.0
    samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_user_pct": self.cpu_user_pct,
            "cpu_system_pct": self.cpu_system_pct,
            "host_mem_increase_bytes": self.host_mem_increase_bytes,
            "disk_read_bytes": self.disk_read_bytes,
            "disk_read_bps": self.disk_read_bps,
            "read_char_bytes": self.read_char_bytes,
            "read_char_bps": self.read_char_bps,
            "gpu_util_pct": self.gpu_util_pct,
            "gpu_mem_used_bytes": self.gpu_mem_used_bytes,
            "nvlink_bytes": self.nvlink_bytes,
            "nvlink_bps": self.nvlink_bps,
        }


def _read_proc_io() -> Optional[Dict[str, int]]:
    """Return {'read_bytes': block-layer reads, 'rchar': syscall bytes}."""
    try:
        out: Dict[str, int] = {}
        with open("/proc/self/io", "r") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    out["read_bytes"] = int(line.split()[1])
                elif line.startswith("rchar:"):
                    out["rchar"] = int(line.split()[1])
        return out or None
    except Exception:
        return None


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "r") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * _PAGE_SIZE
    except Exception:
        return 0


class _Nvml:
    """Thin, lazily-initialized NVML wrapper for one device."""

    def __init__(self, device_index: int):
        self.ok = False
        self._n = None
        self._h = None
        self._nvlink_fields = None
        try:
            import pynvml as N

            N.nvmlInit()
            self._n = N
            self._h = N.nvmlDeviceGetHandleByIndex(device_index)
            self._nvlink_fields = (
                getattr(N, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX", None),
                getattr(N, "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX", None),
            )
            self.ok = True
        except Exception:
            self.ok = False

    def util_and_mem(self):
        try:
            u = self._n.nvmlDeviceGetUtilizationRates(self._h).gpu
            m = self._n.nvmlDeviceGetMemoryInfo(self._h).used
            return int(u), int(m)
        except Exception:
            return None, None

    def nvlink_bytes(self) -> Optional[int]:
        """Cumulative NVLink TX+RX bytes for this device (fields are KiB)."""
        if not self.ok or not self._nvlink_fields or self._nvlink_fields[0] is None:
            return None
        try:
            queries = [(f, 0) for f in self._nvlink_fields if f is not None]
            vals = self._n.nvmlDeviceGetFieldValues(self._h, queries)
            total_kib = 0
            for v in vals:
                if v.nvmlReturn == 0:  # NVML_SUCCESS
                    total_kib += int(v.value.ullVal)
            return total_kib * 1024
        except Exception:
            return None


class ResourceMonitor:
    """Sample resources across a timed region. Use as a context manager."""

    def __init__(self, device_index: int = 0, sample_interval: float = 0.02,
                 enable_nvml: bool = True):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self._nvml = _Nvml(device_index) if enable_nvml else None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._raw: List[Dict[str, Optional[float]]] = []  # per-tick cumulative snapshots
        self.result = TelemetryResult()
        self.trace_samples: List[Dict[str, float]] = []  # instantaneous time-series

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self._io0 = _read_proc_io()
        self._rss0 = _read_rss_bytes()
        self._nvlink0 = self._nvml.nvlink_bytes() if self._nvml and self._nvml.ok else None
        # Seed the trace with the start snapshot at t=0 so the first interval has
        # a baseline for its instantaneous rates.
        self._raw = [{
            "t": 0.0, "rss": self._rss0, "util": None, "mem": None,
            "cpu_user_s": self._ru0.ru_utime, "cpu_sys_s": self._ru0.ru_stime,
            "read_bytes": (self._io0 or {}).get("read_bytes"),
            "rchar": (self._io0 or {}).get("rchar"),
            "nvlink": self._nvlink0,
        }]
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _snapshot(self) -> Dict[str, Optional[float]]:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        io = _read_proc_io() or {}
        u = m = None
        nvl = None
        if self._nvml and self._nvml.ok:
            u, m = self._nvml.util_and_mem()
            nvl = self._nvml.nvlink_bytes()
        return {
            "t": time.perf_counter() - self._t0, "rss": _read_rss_bytes(),
            "util": u, "mem": m,
            "cpu_user_s": ru.ru_utime, "cpu_sys_s": ru.ru_stime,
            "read_bytes": io.get("read_bytes"), "rchar": io.get("rchar"),
            "nvlink": nvl,
        }

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._raw.append(self._snapshot())
            self._stop.wait(self.sample_interval)

    def stop(self) -> TelemetryResult:
        if self._thread is None:
            return self.result
        self._stop.set()
        self._thread.join(timeout=5.0)
        elapsed = max(1e-9, time.perf_counter() - self._t0)
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        io1 = _read_proc_io()
        nvlink1 = self._nvml.nvlink_bytes() if self._nvml and self._nvml.ok else None

        r = self.result
        r.cpu_user_pct = (ru1.ru_utime - self._ru0.ru_utime) / elapsed * 100.0
        r.cpu_system_pct = (ru1.ru_stime - self._ru0.ru_stime) / elapsed * 100.0

        rss_samples = [s["rss"] for s in self._raw if s["rss"]]
        peak_rss = max(rss_samples) if rss_samples else _read_rss_bytes()
        r.host_mem_increase_bytes = max(0, peak_rss - self._rss0)

        if self._io0 is not None and io1 is not None:
            r.disk_read_bytes = max(0, io1.get("read_bytes", 0) - self._io0.get("read_bytes", 0))
            r.disk_read_bps = r.disk_read_bytes / elapsed
            r.read_char_bytes = max(0, io1.get("rchar", 0) - self._io0.get("rchar", 0))
            r.read_char_bps = r.read_char_bytes / elapsed

        util_samples = [s["util"] for s in self._raw if s["util"] is not None]
        mem_samples = [s["mem"] for s in self._raw if s["mem"] is not None]
        if util_samples:
            r.gpu_util_pct = sum(util_samples) / len(util_samples)
        if mem_samples:
            r.gpu_mem_used_bytes = max(mem_samples)
        if self._nvlink0 is not None and nvlink1 is not None:
            r.nvlink_bytes = max(0, nvlink1 - self._nvlink0)
            r.nvlink_bps = r.nvlink_bytes / elapsed
        r.samples = len(self._raw)

        self.trace_samples = self._build_trace()
        return r

    def _build_trace(self) -> List[Dict[str, float]]:
        """Instantaneous time-series from consecutive cumulative snapshots."""
        def _rate(a, b, key) -> float:
            if a.get(key) is None or b.get(key) is None:
                return 0.0
            return max(0.0, b[key] - a[key]) / dt

        trace: List[Dict[str, float]] = []
        for i in range(1, len(self._raw)):
            a, b = self._raw[i - 1], self._raw[i]
            dt = max(1e-6, b["t"] - a["t"])
            trace.append({
                "t": round(b["t"], 4),
                "cpu_user_pct": round((b["cpu_user_s"] - a["cpu_user_s"]) / dt * 100.0, 1),
                "cpu_system_pct": round((b["cpu_sys_s"] - a["cpu_sys_s"]) / dt * 100.0, 1),
                "gpu_util_pct": float(b["util"]) if b["util"] is not None else 0.0,
                "gpu_mem_gb": round(b["mem"] / 1e9, 3) if b["mem"] else 0.0,
                "host_rss_gb": round(b["rss"] / 1e9, 3) if b["rss"] else 0.0,
                "disk_gbps": round(_rate(a, b, "read_bytes") / 1e9, 3),
                "read_gbps": round(_rate(a, b, "rchar") / 1e9, 3),
                "nvlink_gbps": round(_rate(a, b, "nvlink") / 1e9, 3),
            })
        return trace

    def trace(self) -> List[Dict[str, float]]:
        return self.trace_samples
