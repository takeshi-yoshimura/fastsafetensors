# SPDX-License-Identifier: Apache-2.0

import torch

from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.frameworks._torch import TorchOp, TorchTensor
from fastsafetensors.st_types import Device, DeviceType, DType


def test_flat_source_run_spans_non_resizable_dlpack_storages() -> None:
    """A contiguous run need not share one PyTorch Storage object."""
    device = Device(DeviceType.CPU, 0)
    backing = torch.arange(16, dtype=torch.uint8)
    parts = []
    for offset in (0, 8):
        capsule = from_cuda_buffer(
            backing.data_ptr() + offset, [8], [1], DType.U8, device
        )
        part = torch.from_dlpack(capsule)
        assert part.untyped_storage().nbytes() == 8
        parts.append(TorchTensor(device, DType.U8, part))

    flat = TorchOp._flat_source_run(parts, [8, 8])

    assert flat.untyped_storage().nbytes() == 16
    assert torch.equal(flat, backing)
    flat[9] = 123
    assert backing[9].item() == 123
