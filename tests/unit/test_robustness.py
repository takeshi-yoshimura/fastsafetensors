# SPDX-License-Identifier: Apache-2.0
"""Robustness tests: filesystem-aware I/O paths and graceful degradation."""

import pytest

from fastsafetensors.common import get_fs_type

# ---- get_fs_type: longest-prefix mount matching ----

MOUNTS = """\
/dev/root / ext4 rw 0 0
nas:/vol /mnt/nfs nfs4 rw 0 0
/dev/nvme0n1p2 /data ext4 rw 0 0
nas:/models /data/remote nfs rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
/dev/sdb1 /mnt/with\\040space ext4 rw 0 0
"""


@pytest.fixture()
def mounts_file(tmp_path):
    p = tmp_path / "mounts"
    p.write_text(MOUNTS)
    return str(p)


def test_get_fs_type_basic(mounts_file):
    assert get_fs_type("/mnt/nfs/model.safetensors", mounts_file) == "nfs4"
    assert get_fs_type("/data/m/x.safetensors", mounts_file) == "ext4"
    assert get_fs_type("/somewhere/else", mounts_file) == "ext4"  # root fallback


def test_get_fs_type_longest_prefix_wins(mounts_file):
    # /data is ext4 but /data/remote is an NFS mount inside it
    assert get_fs_type("/data/remote/model.safetensors", mounts_file) == "nfs"


def test_get_fs_type_escaped_mountpoint(mounts_file):
    assert get_fs_type("/mnt/with space/f.safetensors", mounts_file) == "ext4"


def test_get_fs_type_unreadable_mounts(tmp_path):
    assert get_fs_type("/data/x", str(tmp_path / "nope")) == ""


# ---- O_DIRECT gating on network filesystems ----


def test_odirect_gating(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.delenv("FASTSAFETENSORS_ODIRECT", raising=False)
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    assert unified._odirect_ok("/mnt/nfs/f") is False
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    assert unified._odirect_ok("/data/f") is True
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "")  # unknown: allow
    assert unified._odirect_ok("/x") is True


def test_odirect_env_override(monkeypatch):
    from fastsafetensors.copier import unified

    monkeypatch.setattr(unified, "get_fs_type", lambda p: "nfs4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    assert unified._odirect_ok("/mnt/nfs/f") is True  # forced on
    monkeypatch.setattr(unified, "get_fs_type", lambda p: "ext4")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "0")
    assert unified._odirect_ok("/data/f") is False  # forced off


# ---- chunk plans must fail loudly on copiers without set_chunk ----


def test_chunk_plan_requires_set_chunk(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only")
    from fastsafetensors import SafeTensorsFileLoader, SafeTensorsMetadata

    loader = SafeTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")
    loader.add_filenames({0: [input_files[0]]})
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (chunk,) = meta.plan_chunks(meta.size_bytes)  # single whole-span chunk
    loader.set_chunk_plan({input_files[0]: chunk})

    class _NoChunkCopier:  # e.g. gds/dstorage: no set_chunk support
        def __init__(self, *a, **k):
            pass

    loader.copier_constructor = lambda m, d, f: _NoChunkCopier()
    with pytest.raises(NotImplementedError, match="set_chunk"):
        loader.copy_files_to_device()


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
