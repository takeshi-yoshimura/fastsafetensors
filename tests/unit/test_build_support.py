from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastsafetensors.build_support import compiler_accepts, wants_gb10_target


def test_gb10_is_arm64_and_opt_in():
    assert wants_gb10_target("aarch64", {"FASTSAFETENSORS_CPU_TARGET": "gb10"})
    assert not wants_gb10_target("x86_64", {"FASTSAFETENSORS_CPU_TARGET": "gb10"})
    assert not wants_gb10_target("aarch64", {})


class Compiler:
    def __init__(self, reject=False):
        self.reject = reject

    def compile(self, sources, output_dir, extra_postargs):
        assert extra_postargs == ["-mcpu=gb10"]
        if self.reject:
            raise RuntimeError("unsupported")


def test_compiler_probe_and_fallback():
    assert compiler_accepts(Compiler(), "-mcpu=gb10")
    assert not compiler_accepts(Compiler(True), "-mcpu=gb10")
