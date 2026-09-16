"""Differentiable, scan-parameterized Bruker CSI reconstruction operator.

This module implements the numerically validated ParaVision reconstruction stage
from 305 averaged complex spatial encodes to the retained complex voxel FIDs
(``fid_proc.64``).  It is deliberately not named an HR-to-measurement acquisition
operator: metabolite fitting, spectral preprocessing, scale alignment, and the
mapping from an HR metabolite image to raw encoded observations are separate
contracts that must be validated before real-data training.

Validated operation order per spatial axis:

1. place the 305 encodes on the 9 x 9 x 5 grid using ``RecoSortMaps``;
2. multiply by the scan-specific ``RECO_usr_wdw`` values;
3. multiply by ``exp(-i 2 pi RECO_rotate n)``;
4. apply the default normalized inverse FFT;
5. after all spatial axes, apply ``RECO_pc_lin`` first-order phase correction.

The class also exposes the exact Hermitian adjoint of this linear reconstruction
for numerical tests and future optimization-variable experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class BrukerCSIReconstructionContract:
    """Numerical parameters read from one scan's first-party Bruker files."""

    sort_maps: Sequence[int]
    window_z: Sequence[float]
    window_y: Sequence[float]
    window_x: Sequence[float]
    rotate_z: float
    rotate_y: float
    rotate_x: float
    phase_first_order_z_deg: float
    phase_first_order_y_deg: float
    phase_first_order_x_deg: float
    spatial_shape_zyx: tuple[int, int, int] = (5, 9, 9)

    def validate(self) -> None:
        z, y, x = self.spatial_shape_zyx
        maps = [int(value) for value in self.sort_maps]
        if len(maps) != 305:
            raise ValueError(f"Expected 305 RecoSortMaps entries, got {len(maps)}")
        if len(set(maps)) != len(maps):
            raise ValueError("RecoSortMaps contains duplicate output positions")
        if min(maps) < 0 or max(maps) >= z * y * x:
            raise ValueError("RecoSortMaps contains an out-of-grid position")
        for name, values, expected in (
            ("window_z", self.window_z, z),
            ("window_y", self.window_y, y),
            ("window_x", self.window_x, x),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} has {len(values)} entries, expected {expected}")


class BrukerCSIReconstructionOperator(nn.Module):
    """Linear vendor reconstruction and exact Hermitian adjoint.

    Input shape is ``(..., spectral_points, 305)`` and reconstructed output shape
    is ``(..., spectral_points, 5, 9, 9)``.  Complex tensors are required; no
    magnitude operation is performed because ``fid_proc.64`` is complex.
    """

    def __init__(self, contract: BrukerCSIReconstructionContract) -> None:
        super().__init__()
        contract.validate()
        self.spatial_shape_zyx = tuple(int(v) for v in contract.spatial_shape_zyx)
        self.register_buffer("sort_maps", torch.as_tensor(contract.sort_maps, dtype=torch.long))
        for name, values in (
            ("window_z", contract.window_z),
            ("window_y", contract.window_y),
            ("window_x", contract.window_x),
        ):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float64))
        self.register_buffer(
            "rotates_zyx",
            torch.tensor(
                [contract.rotate_z, contract.rotate_y, contract.rotate_x],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "phase_first_order_zyx_deg",
            torch.tensor(
                [
                    contract.phase_first_order_z_deg,
                    contract.phase_first_order_y_deg,
                    contract.phase_first_order_x_deg,
                ],
                dtype=torch.float64,
            ),
        )

    @staticmethod
    def _require_complex(value: torch.Tensor, name: str) -> None:
        if not torch.is_complex(value):
            raise TypeError(f"{name} must be a complex tensor, got {value.dtype}")

    def _window(self, axis_index: int, *, device, real_dtype) -> torch.Tensor:
        return (self.window_z, self.window_y, self.window_x)[axis_index].to(
            device=device, dtype=real_dtype
        )

    def _ramp(self, axis_index: int, *, device, real_dtype) -> torch.Tensor:
        length = self.spatial_shape_zyx[axis_index]
        coordinate = torch.arange(length, device=device, dtype=real_dtype)
        rotate = self.rotates_zyx[axis_index].to(device=device, dtype=real_dtype)
        return torch.exp(-2j * torch.pi * rotate * coordinate)

    def _phase_correction(self, *, device, real_dtype) -> torch.Tensor:
        coordinates = torch.meshgrid(
            *[
                torch.arange(length, device=device, dtype=real_dtype)
                for length in self.spatial_shape_zyx
            ],
            indexing="ij",
        )
        phase = torch.zeros(self.spatial_shape_zyx, device=device, dtype=real_dtype)
        degrees = self.phase_first_order_zyx_deg.to(device=device, dtype=real_dtype)
        for coordinate, degree in zip(coordinates, degrees):
            phase = phase + torch.deg2rad(degree) * coordinate
        return torch.exp(1j * phase)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct averaged complex encodes into complex voxel FIDs."""
        self._require_complex(encoded, "encoded")
        if encoded.ndim < 2 or encoded.shape[-1] != self.sort_maps.numel():
            raise ValueError(
                f"Expected (..., spectral_points, {self.sort_maps.numel()}) input, "
                f"got {tuple(encoded.shape)}"
            )
        real_dtype = encoded.real.dtype
        grid_flat = torch.zeros(
            (*encoded.shape[:-1], int(torch.tensor(self.spatial_shape_zyx).prod().item())),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        grid_flat = grid_flat.index_copy(-1, self.sort_maps.to(encoded.device), encoded)
        grid = grid_flat.reshape(*encoded.shape[:-1], *self.spatial_shape_zyx)

        for axis_index, dim in enumerate((-3, -2, -1)):
            shape = [1] * grid.ndim
            shape[dim] = self.spatial_shape_zyx[axis_index]
            window = self._window(axis_index, device=grid.device, real_dtype=real_dtype).reshape(shape)
            ramp = self._ramp(axis_index, device=grid.device, real_dtype=real_dtype).reshape(shape)
            grid = torch.fft.ifft(grid * window * ramp, dim=dim)

        phase_shape = [1] * (grid.ndim - 3) + list(self.spatial_shape_zyx)
        phase = self._phase_correction(device=grid.device, real_dtype=real_dtype).reshape(phase_shape)
        return grid * phase

    def adjoint(self, voxel_fids: torch.Tensor) -> torch.Tensor:
        """Apply the exact Hermitian adjoint of :meth:`forward`."""
        self._require_complex(voxel_fids, "voxel_fids")
        if voxel_fids.ndim < 4 or tuple(voxel_fids.shape[-3:]) != self.spatial_shape_zyx:
            raise ValueError(
                f"Expected (..., spectral_points, {self.spatial_shape_zyx}) input, "
                f"got {tuple(voxel_fids.shape)}"
            )
        real_dtype = voxel_fids.real.dtype
        phase_shape = [1] * (voxel_fids.ndim - 3) + list(self.spatial_shape_zyx)
        phase = self._phase_correction(
            device=voxel_fids.device, real_dtype=real_dtype
        ).reshape(phase_shape)
        grid = voxel_fids * phase.conj()

        for axis_index, dim in reversed(list(enumerate((-3, -2, -1)))):
            # For torch's default ifft (1/N normalization), the Hermitian
            # adjoint is fft with norm="forward" (also 1/N).
            grid = torch.fft.fft(grid, dim=dim, norm="forward")
            shape = [1] * grid.ndim
            shape[dim] = self.spatial_shape_zyx[axis_index]
            ramp = self._ramp(
                axis_index, device=grid.device, real_dtype=real_dtype
            ).reshape(shape)
            window = self._window(
                axis_index, device=grid.device, real_dtype=real_dtype
            ).reshape(shape)
            grid = grid * ramp.conj() * window

        flat = grid.reshape(*grid.shape[:-3], -1)
        return flat.index_select(-1, self.sort_maps.to(flat.device))
