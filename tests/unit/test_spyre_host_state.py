# SPDX-License-Identifier: Apache-2.0
"""Backend-owned placement metadata, with no inference or accelerator dependency."""

import builtins
import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from fastsafetensors import ParallelSaver, SafeTensorsMetadata
from fastsafetensors.copier import spyre
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import NativeFormatError


@pytest.fixture(autouse=True)
def no_inference_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "spyre_inference", "torch_spyre"}:
            raise AssertionError(f"host state imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@dataclass(frozen=True, slots=True)
class Padding:
    pad: torch.Tensor


class Host(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.5), persistent=False)
        pad = torch.tensor([0.0, 1.0, 2.0])
        self.padding = Padding(pad)
        self.alias = [Padding(pad)]
        self.other = pad.clone()
        self.tail = SimpleNamespace(weight=torch.arange(6.0).reshape(2, 3).t())
        self.tables = {None: torch.tensor(1), 1: torch.tensor(2), "1": torch.empty(0)}
        self.cycle = {"tensor": torch.tensor(4)}
        self.cycle["self"] = self.cycle
        self.derived = None

    def _apply(self, fn, recurse=True):
        self.derived = self.padding.pad + 1
        return self


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.host = Host()


@pytest.fixture
def checkpoint(tmp_path):
    model = Model()
    recorder = spyre.TransferRecorder(torch.device("cpu"))
    recorder.capture_host_state(model)
    model._apply(recorder)
    recorder.records[0].kind = "to"
    metadata, entries = recorder.checkpoint_metadata(model)
    paths = ParallelSaver(num_shards=2).save_entries(
        entries, str(tmp_path / "state"), shard0_metadata=metadata
    )
    metas = [
        SafeTensorsMetadata.from_file(path, get_framework_op("pytorch"))
        for path in paths
    ]
    headers = {name: meta for meta in metas for name in meta.tensors}
    return model, metadata, headers


def test_backend_keys_round_trip_host_before_replay(checkpoint):
    source, metadata, headers = checkpoint
    assert set(metadata) == {"torch_spyre.transfers", "torch_spyre.host"}
    host = json.loads(metadata[spyre.KEY_HOST])
    assert all(record["device"] == "cpu" for record in host)
    assert not any(record["path"] == [["attr", "derived"]] for record in host)
    with torch.device("meta"):
        restored = Model()
    restorer = spyre.TransferRestorer.from_metadata(
        torch.device("cpu"), headers, metadata
    )
    restorer.restore_host_state(restored)
    restored._apply(restorer)
    restorer.finish()
    spyre.check_materialized(restored)
    assert restored.host.padding.pad is restored.host.alias[0].pad
    assert restored.host.cycle["self"] is restored.host.cycle
    assert restored.host.tail.weight.stride() == source.host.tail.weight.stride()
    for module, path, tensor in spyre._tensor_paths(source):
        torch.testing.assert_close(
            spyre._resolve_tensor(restored, module, path), tensor
        )


def test_restore_resolves_only_recorded_paths(checkpoint, monkeypatch):
    _, metadata, headers = checkpoint
    with torch.device("meta"):
        model = Model()
    monkeypatch.setattr(
        spyre, "_tensor_paths", lambda *args: pytest.fail("no discovery on read")
    )
    spyre.TransferRestorer.from_metadata(
        torch.device("cpu"), headers, metadata
    ).restore_host_state(model)
    assert model.host.tables[None].item() == 1
    assert model.host.tables[1].item() == 2
    assert model.host.tables["1"].numel() == 0


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "split", "merged"])
def test_validate_all_paths_and_aliases_before_swap(checkpoint, monkeypatch, change):
    _, metadata, headers = checkpoint
    with torch.device("meta"):
        model = Model()
        if change == "missing":
            del model.host.tables["1"]
        elif change == "shape":
            model.host.other = torch.empty(9)
        elif change == "dtype":
            model.host.other = model.host.other.half()
        elif change == "split":
            model.host.alias = [Padding(torch.empty(3))]
        else:
            model.host.other = model.host.padding.pad
    monkeypatch.setattr(
        spyre, "_swap_in", lambda *args: pytest.fail("validate before swap")
    )
    with pytest.raises(NativeFormatError):
        spyre.TransferRestorer.from_metadata(
            torch.device("cpu"), headers, metadata
        ).restore_host_state(model)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", [["call", "eval"]]),
        ("path", []),
        ("module", "missing"),
        ("device", "spyre"),
    ],
)
def test_reject_malformed_paths(checkpoint, field, value):
    _, metadata, headers = checkpoint
    metadata = copy.deepcopy(metadata)
    host = json.loads(metadata[spyre.KEY_HOST])
    host[0][field] = value
    metadata[spyre.KEY_HOST] = json.dumps(host)
    with pytest.raises(NativeFormatError):
        spyre.TransferRestorer.from_metadata(
            torch.device("cpu"), headers, metadata
        ).restore_host_state(Model())


@pytest.mark.parametrize("key", [spyre.KEY_TRANSFERS, spyre.KEY_HOST])
@pytest.mark.parametrize("value", [None, "[", "{}", "null", "[null]"])
def test_reject_missing_or_malformed_metadata(checkpoint, key, value):
    _, metadata, headers = checkpoint
    metadata = dict(metadata)
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value
    with pytest.raises(NativeFormatError):
        spyre.TransferRestorer.from_metadata(torch.device("cpu"), headers, metadata)


def test_swap_preserves_subclass_attributes_and_object_aliases():
    class Parameter(torch.nn.Parameter):
        pass

    target = Parameter(torch.empty(2, device="meta"), requires_grad=False)
    target.marker = object()
    alias, marker = target, target.marker
    spyre._swap_in(target, torch.tensor([1.0, 2.0]))
    assert target is alias and type(target) is Parameter and target.marker is marker
    torch.testing.assert_close(target, torch.tensor([1.0, 2.0]))


def test_remaining_meta_state_rejected():
    model = Model()
    model.host.tail.weight = torch.empty(2, device="meta")
    with pytest.raises(NativeFormatError, match="still on meta"):
        spyre.TransferRecorder(torch.device("cpu")).capture_host_state(model)


def test_capture_required_before_checkpoint_metadata():
    with pytest.raises(NativeFormatError, match="capture_host_state"):
        spyre.TransferRecorder(torch.device("cpu")).checkpoint_metadata(Model())


def test_only_original_paths_still_on_cpu_enter_host_recipe(monkeypatch):
    model = Model()
    recorder = spyre.TransferRecorder(torch.device("meta"))
    recorder.capture_host_state(model)
    model._apply(recorder)
    # A meta destination stands in for Spyre on this CPU-only test; skip only
    # the final materialization assertion, not path selection/serialization.
    monkeypatch.setattr(spyre, "check_materialized", lambda model: None)
    metadata, entries = recorder.checkpoint_metadata(model)
    host = json.loads(metadata[spyre.KEY_HOST])
    assert all(record["module"] == "host" for record in host)
    assert all(record["path"] != [["attr", "derived"]] for record in host)
    assert json.loads(metadata[spyre.KEY_TRANSFERS])[0]["kind"] == "to"
    assert len(entries) == 1 + len({record["name"] for record in host})


def test_host_tensor_removed_by_apply_is_not_recorded():
    model = Host()
    recorder = spyre.TransferRecorder(torch.device("cpu"))
    recorder.capture_host_state(model)
    model.other = None
    metadata, _ = recorder.checkpoint_metadata(model)
    assert all(
        record["path"] != [["attr", "other"]]
        for record in json.loads(metadata[spyre.KEY_HOST])
    )


def test_cache_envelope_round_trip_with_caller_identity(tmp_path):
    key = {"model": "test", "rank": 0, "config": {"dtype": "float16", "tp": 2}}
    metadata, shard0_metadata = spyre.cache_metadata(key)
    assert metadata == {
        "torch_spyre.cache.format": spyre.CACHE_FORMAT,
        "torch_spyre.cache.key": json.dumps(key, sort_keys=True),
    }
    assert shard0_metadata == {"torch_spyre.cache.fp8": "[]"}
    paths = ParallelSaver(num_shards=2).save_entries(
        [],
        str(tmp_path / "envelope"),
        metadata=metadata,
        shard0_metadata=shard0_metadata,
    )
    metas = [
        SafeTensorsMetadata.from_file(path, get_framework_op("pytorch"))
        for path in paths
    ]
    assert (
        spyre.read_cache_metadata(metas[0].metadata, dict(reversed(list(key.items()))))
        is None
    )
    assert metas[1].metadata[spyre.KEY_CACHE_FORMAT] == spyre.CACHE_FORMAT
    assert spyre.KEY_FP8 not in metas[1].metadata


@pytest.mark.parametrize(
    "field",
    [spyre.KEY_CACHE_FORMAT, spyre.KEY_CACHE_KEY, spyre.KEY_FP8],
)
def test_cache_envelope_requires_backend_keys(field):
    metadata, shard0 = spyre.cache_metadata({})
    metadata.update(shard0)
    metadata["spyre_inference." + field.rsplit(".", 1)[1]] = metadata.pop(field)
    with pytest.raises(NativeFormatError, match="invalid Spyre cache metadata"):
        spyre.read_cache_metadata(metadata, {})


@pytest.mark.parametrize("value", ["[", "null", "{}", '[{"path":"linear"}]'])
def test_cache_envelope_rejects_fp8_payload_recipes(value):
    metadata, shard0 = spyre.cache_metadata({})
    metadata.update(shard0)
    metadata[spyre.KEY_FP8] = value
    with pytest.raises(NativeFormatError):
        spyre.read_cache_metadata(metadata, {})


@pytest.mark.parametrize(
    "field,value",
    [
        (spyre.KEY_CACHE_FORMAT, "spyre-native-cache/3"),
        (spyre.KEY_CACHE_KEY, '{"other": 1}'),
        (spyre.KEY_CACHE_KEY, "["),
    ],
)
def test_cache_envelope_rejects_incompatible_identity_or_format(field, value):
    metadata, shard0 = spyre.cache_metadata({})
    metadata.update(shard0)
    metadata[field] = value
    with pytest.raises(NativeFormatError, match="another identity or format"):
        spyre.read_cache_metadata(metadata, {})
