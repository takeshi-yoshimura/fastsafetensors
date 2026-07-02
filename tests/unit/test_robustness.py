# SPDX-License-Identifier: Apache-2.0
"""Robustness tests: graceful degradation of I/O paths."""

import pytest

# ---- runtime GDS -> nogds fallback ----


def test_gds_copier_falls_back_to_nogds(input_files, framework, monkeypatch):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from test_fastsafetensors import load_safetensors_file

    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors.copier.gds import GdsFileCopier
    from fastsafetensors.st_types import Device

    def _boom(*a, **k):
        raise RuntimeError(
            "raw_gds_file_handle: cuFileHandleRegister returned an error = 5027"
        )

    monkeypatch.setattr(fstcpp, "gds_file_handle", _boom)

    device = Device.from_str("cpu")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    reader = fstcpp.gds_file_reader(4, False, 0)
    copier = GdsFileCopier(meta, device, reader, framework)
    gbuf = copier.submit_io(False, 10 * 1024 * 1024 * 1024)
    tensors = copier.wait_io(gbuf)
    expected = load_safetensors_file(input_files[0], device, framework)
    assert set(tensors.keys()) == set(expected.keys())
    for k, exp in expected.items():
        assert framework.is_equal(tensors[k], exp), k
    framework.free_tensor_memory(gbuf, device)
    # the fallback's bounce-buffer reader must not outlive the copy cycle
    assert fstcpp.get_cpp_metrics().bounce_buffer_bytes == 0


def test_gds_fallback_warns_once_and_shares_reader(
    input_files, framework, monkeypatch, caplog
):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    import logging

    from fastsafetensors import SafeTensorsMetadata
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors.copier import gds as gds_mod
    from fastsafetensors.st_types import Device

    monkeypatch.setattr(
        fstcpp,
        "gds_file_handle",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("error = 5027")),
    )
    monkeypatch.setattr(gds_mod, "_warned_gds_fallback", False)

    device = Device.from_str("cpu")
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    reader = fstcpp.gds_file_reader(4, False, 0)
    cache = []
    with caplog.at_level(logging.WARNING, logger="fastsafetensors.copier.gds"):
        c1 = gds_mod.GdsFileCopier(
            meta, device, reader, framework, fallback_cache=cache
        )
        g1 = c1.submit_io(False, 1 << 30)
        c1.wait_io(g1)
        c2 = gds_mod.GdsFileCopier(
            meta, device, reader, framework, fallback_cache=cache
        )
        g2 = c2.submit_io(False, 1 << 30)
        c2.wait_io(g2)
    assert caplog.text.count("falling back to the nogds copier") == 1  # warn once
    assert len(cache) == 1  # one shared nogds constructor for the whole loader
    framework.free_tensor_memory(g1, device)
    framework.free_tensor_memory(g2, device)
