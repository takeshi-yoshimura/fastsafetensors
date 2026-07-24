"""Schema validation, identity, and JSONL round-trip."""

from __future__ import annotations

import pytest

from fastsafetensors_perf import metrics
from fastsafetensors_perf.results import (
    IDENTITY_FIELDS,
    SCHEMA_VERSION,
    Identity,
    RankMetrics,
    SchemaError,
    make_aggregate_record,
    make_rank_record,
    read_records,
    series_of,
    validate_record,
    write_records,
)


def _identity(**over) -> Identity:
    base = dict(
        hardware_profile="a100", gpu_model="NVIDIA A100-SXM4-80GB", world_size=1,
        storage_id="nvme:ext4", model_alias="qwen3-8b", model_revision="rev1",
        model_fingerprint="abcd", mode="parallel", consumer="iterate",
        queue_size=0, backend="nogds", cache_policy="cold", tuning_key="",
        vllm_series="0.25", torch_series="2.6", cuda_series="12.4",
        fastsafetensors_series="0.3",
    )
    base.update(over)
    return Identity(**base)


def test_series_of():
    assert series_of("0.25.1") == "0.25"
    assert series_of("2.6.0+cu124") == "2.6"
    assert series_of("") == ""
    assert series_of(None) == ""


def test_identity_covers_all_fields():
    ident = _identity()
    d = ident.to_dict()
    assert set(d) == set(IDENTITY_FIELDS)
    assert Identity.from_dict(d).key() == ident.key()


def test_rank_record_validates_and_roundtrips(tmp_path):
    ident = _identity()
    env = {"hostname": "h1"}
    case = {"case_id": "c1"}
    m = RankMetrics(rank=0, repetition=0, wall_seconds=2.0, logical_bytes=100,
                    source_checkpoint_bytes=200)
    rec = make_rank_record(ident, env, case, m)
    validate_record(rec)
    assert rec["metrics"]["delivery_throughput_bps"] == 50.0
    assert rec["metrics"]["storage_throughput_bps"] == 100.0

    path = str(tmp_path / "out.jsonl")
    write_records(path, [rec], append=False)
    back = read_records(path)
    assert len(back) == 1
    assert back[0]["metrics"]["wall_seconds"] == 2.0


def test_validate_rejects_missing_identity_field():
    ident = _identity().to_dict()
    del ident["backend"]
    rec = {
        "schema_version": SCHEMA_VERSION, "kind": "rank",
        "identity": ident, "environment": {}, "case": {}, "metrics": {},
    }
    with pytest.raises(SchemaError):
        validate_record(rec)


def test_validate_rejects_incompatible_major():
    rec = {
        "schema_version": "99.0", "kind": "rank",
        "identity": _identity().to_dict(), "environment": {}, "case": {},
        "metrics": {},
    }
    with pytest.raises(SchemaError):
        validate_record(rec)


def test_aggregate_record_from_repetitions(tmp_path):
    ident = _identity()
    # two repetitions, one rank
    reps = [
        [RankMetrics(rank=0, repetition=0, wall_seconds=2.0, logical_bytes=100,
                     peak_cuda_allocated_bytes=1000).to_dict()],
        [RankMetrics(rank=0, repetition=1, wall_seconds=2.2, logical_bytes=100,
                     peak_cuda_allocated_bytes=1000).to_dict()],
    ]
    stats = metrics.aggregate_repetitions(reps)
    rec = make_aggregate_record(ident, {}, {}, stats,
                                cache_eviction_available=True,
                                n_repetitions=2, n_ranks=1, worst_status="ok")
    validate_record(rec)
    assert rec["aggregate"]["stats"]["wall_seconds"]["median"] == pytest.approx(2.1)
    assert rec["aggregate"]["worst_status"] == "ok"
