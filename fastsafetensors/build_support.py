"""Build-time compiler feature detection."""

import os
import platform
import tempfile


def wants_gb10_target(machine=None, environ=None):
    env = os.environ if environ is None else environ
    arch = (platform.machine() if machine is None else machine).lower()
    return (
        arch in {"aarch64", "arm64"}
        and env.get("FASTSAFETENSORS_CPU_TARGET", "").lower() == "gb10"
    )


def compiler_accepts(compiler, flag):
    with tempfile.TemporaryDirectory(prefix="fastsafetensors-cc-") as tmp:
        source = os.path.join(tmp, "flag_test.cpp")
        with open(source, "w", encoding="utf-8") as stream:
            stream.write("int main() { return 0; }\n")
        try:
            compiler.compile([source], output_dir=tmp, extra_postargs=[flag])
        except Exception:
            return False
    return True
