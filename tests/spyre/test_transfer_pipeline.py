# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""stage_to_spyre must be byte-identical to copy_tensor, whatever the layout."""


import os

import pytest

if os.environ.get("TEST_FASTSAFETENSORS_SPYRE") != "1":
    pytest.skip(
        "set TEST_FASTSAFETENSORS_SPYRE=1 to run on Spyre", allow_module_level=True
    )

import torch
import torch_spyre  # noqa: F401 -- registers the device
from torch_spyre._C import (
    SpyreTensorLayout,
    copy_tensor,
    get_device_dtype,
    get_elem_in_stick,
    spyre_empty_with_layout,
)

from fastsafetensors.copier import spyre as _transfer_pipeline
from fastsafetensors.copier.spyre import stage_to_spyre

DEVICE = torch.device("spyre")


def _default(t):
    return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=DEVICE)


def _swapped(t):
    layout = SpyreTensorLayout(list(t.shape), list(t.stride()), t.dtype, [1, 0])
    return spyre_empty_with_layout(t.size(), t.stride(), t.dtype, layout, device=DEVICE)


def _rows_outermost(t):
    rows, width = t.shape
    stick = get_elem_in_stick(t.dtype)
    layout = SpyreTensorLayout(
        [rows, width // stick, stick], [width, stick, 1], get_device_dtype(t.dtype)
    )
    return spyre_empty_with_layout(t.size(), t.stride(), t.dtype, layout, device=DEVICE)


def _pairs():
    g = torch.Generator().manual_seed(0)
    half = torch.float16
    return [
        (torch.randn(1024, 5376, generator=g).to(half), _default),
        (torch.randn(5376, generator=g).to(half), _default),
        (torch.randn(37, 100, generator=g).to(half), _default),
        (torch.randn(256, 128, generator=g).to(half).t(), _default),
        (torch.randn(4, 64, 192, generator=g).to(half), _default),
        (torch.randn(512, 1024, generator=g).to(half), _swapped),
        (torch.randn(512, 128, generator=g).to(half), _rows_outermost),
    ]


def _check(pairs):
    expected = []
    for src, make in pairs:
        ref = make(src)
        copy_tensor(src, ref, non_blocking=False)
        expected.append(ref.cpu())
    dsts = [make(src) for src, make in pairs]
    stage_to_spyre([(src, dst) for (src, _), dst in zip(pairs, dsts)], DEVICE)
    for i, (dst, want) in enumerate(zip(dsts, expected)):
        torch.testing.assert_close(dst.cpu(), want, rtol=0, atol=0, msg=f"pair {i}")


# Small limits exercise buffer reuse and tensors larger than a batch target.
@pytest.mark.parametrize(
    "max_buffers, max_batch_bytes",
    [(1, 1 << 30), (3, 1 << 30), (64, 1 << 30), (4, 100_000)],
)
def test_matches_copy_tensor(monkeypatch, max_buffers, max_batch_bytes):
    monkeypatch.setattr(_transfer_pipeline, "_MAX_BUFFERS", max_buffers)
    monkeypatch.setattr(_transfer_pipeline, "_MAX_BATCH_BYTES", max_batch_bytes)
    _check(_pairs())
