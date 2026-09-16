"""Differentiable acquisition operators for quantitative 2-D MRSI/DMI maps.

The simulator forms a native low-resolution image by center-cropping the
high-resolution k-space, applying a window on the cropped grid, multiplying
by ``(N_lr / N_hr)^2`` and then applying an inverse FFT.  The scale factor is
essential: a spatially constant concentration must remain constant when the
matrix size changes.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _matrix_sizes(lr_matrix, batch_size: int) -> list[int]:
    if torch.is_tensor(lr_matrix):
        values: Iterable[int] = lr_matrix.detach().reshape(-1).cpu().tolist()
    elif isinstance(lr_matrix, (list, tuple)):
        values = lr_matrix
    else:
        values = [lr_matrix]
    sizes = [int(value) for value in values]
    if len(sizes) == 1 and batch_size > 1:
        sizes *= batch_size
    if len(sizes) != batch_size:
        raise ValueError(f"Expected {batch_size} LR matrix sizes, got {len(sizes)}")
    return sizes


def _window_2d(size: int, window: str, *, device, dtype) -> torch.Tensor:
    name = str(window or "none").lower()
    if name in ("none", "rect", "boxcar"):
        return torch.ones((size, size), device=device, dtype=dtype)
    if name == "hamming":
        one_d = torch.hamming_window(size, periodic=False, device=device, dtype=dtype)
    elif name == "hann":
        one_d = torch.hann_window(size, periodic=False, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported k-space window: {window}")
    window_2d = one_d[:, None] * one_d[None, :]
    return window_2d / window_2d.max().clamp_min(torch.finfo(dtype).eps)


def center_pad_native(native: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Center-pad ``(..., n, n)`` native data to a common HR-sized canvas."""
    h, w = int(target_hw[0]), int(target_hw[1])
    nh, nw = int(native.shape[-2]), int(native.shape[-1])
    if nh > h or nw > w:
        raise ValueError(f"Native shape {(nh, nw)} exceeds target shape {(h, w)}")
    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    return F.pad(native, (left, right, top, bottom))


def crop_padded_native(padded: torch.Tensor, lr_matrix: int) -> torch.Tensor:
    """Extract a centered native matrix from an HR-sized padded tensor."""
    n = int(lr_matrix)
    h, w = int(padded.shape[-2]), int(padded.shape[-1])
    if n <= 0 or n > min(h, w):
        raise ValueError(f"Invalid lr_matrix={n} for padded shape {(h, w)}")
    y0 = h // 2 - n // 2
    x0 = w // 2 - n // 2
    return padded[..., y0 : y0 + n, x0 : x0 + n]


def mrsi_native_forward_batch(
    hr: torch.Tensor,
    lr_matrix,
    *,
    window: str = "hamming",
    clamp_nonnegative: bool = True,
) -> torch.Tensor:
    """Apply the simulator-matched HR-to-native-LR operator.

    Returns center-padded native images with the same spatial shape as ``hr``.
    Padding permits ordinary batching even when samples use 16, 24 and 32
    acquisition matrices.  Values inside the native support retain intensive
    concentration semantics; they are not divided by the number of new pixels.
    """
    if hr.dim() != 4:
        raise ValueError(f"Expected BCHW input, got shape {tuple(hr.shape)}")
    batch, _channels, h, w = hr.shape
    if h != w:
        raise ValueError(f"Only square HR matrices are supported, got {(h, w)}")
    sizes = _matrix_sizes(lr_matrix, batch)
    x = hr.float()
    kspace = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    output = torch.zeros_like(x)
    for n in sorted(set(sizes)):
        if n <= 0 or n > h:
            raise ValueError(f"Invalid lr_matrix={n} for HR matrix {h}")
        indices = torch.tensor(
            [idx for idx, value in enumerate(sizes) if value == n],
            device=x.device,
            dtype=torch.long,
        )
        y0 = h // 2 - n // 2
        x0 = w // 2 - n // 2
        cropped = kspace.index_select(0, indices)[..., y0 : y0 + n, x0 : x0 + n]
        cropped = cropped * _window_2d(n, window, device=x.device, dtype=x.dtype)
        scale = float(n * n) / float(h * w)
        native = torch.fft.ifft2(torch.fft.ifftshift(cropped * scale, dim=(-2, -1))).real
        if clamp_nonnegative:
            native = native.clamp_min(0.0)
        output = output.index_copy(0, indices, center_pad_native(native, (h, w)))
    return output.to(dtype=hr.dtype)


def native_support_mask_batch(
    hr_mask: torch.Tensor | None,
    lr_matrix,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return center-padded soft native-resolution ROI masks."""
    if reference.dim() != 4:
        raise ValueError(f"Expected BCHW reference, got {tuple(reference.shape)}")
    batch, channels, h, w = reference.shape
    sizes = _matrix_sizes(lr_matrix, batch)
    if hr_mask is None:
        mask = torch.ones((batch, 1, h, w), device=reference.device, dtype=reference.dtype)
    else:
        mask = hr_mask.to(device=reference.device, dtype=reference.dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != batch or tuple(mask.shape[-2:]) != (h, w):
            raise ValueError(f"Mask shape {tuple(mask.shape)} is incompatible with {tuple(reference.shape)}")
    out = torch.zeros((batch, 1, h, w), device=reference.device, dtype=reference.dtype)
    for n in sorted(set(sizes)):
        indices = torch.tensor(
            [idx for idx, value in enumerate(sizes) if value == n],
            device=reference.device,
            dtype=torch.long,
        )
        native_mask = F.interpolate(mask.index_select(0, indices), size=(n, n), mode="area")
        out = out.index_copy(0, indices, center_pad_native(native_mask, (h, w)))
    out = out.clamp(0.0, 1.0)
    if channels != 1 and out.shape[1] == 1:
        out = out.expand(-1, channels, -1, -1)
    return out


def refine_native_data_consistency(
    sr_minus1_1: torch.Tensor,
    lr_native_padded: torch.Tensor,
    lr_matrix,
    *,
    hr_mask: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    window: str = "hamming",
    iterations: int = 10,
    learning_rate: float = 0.02,
    anchor_weight: float = 0.1,
) -> torch.Tensor:
    """Refine an SR sample against measured native LR with an SR anchor.

    This is an optional inference-time projection.  The native term uses a
    smooth L1 penalty to avoid forcing noisy magnitude observations exactly;
    the anchor prevents unconstrained high-frequency content from drifting.
    """
    if int(iterations) <= 0:
        return sr_minus1_1.detach()
    initial = ((sr_minus1_1.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
    target = lr_native_padded.detach().to(device=initial.device, dtype=initial.dtype).clamp(0.0, 1.0)
    native_weights = native_support_mask_batch(hr_mask, lr_matrix, reference=initial)
    if valid is not None:
        valid_t = valid.detach().to(device=initial.device, dtype=initial.dtype).reshape(-1, 1, 1, 1)
        native_weights = native_weights * valid_t
    native_denom = native_weights.sum().clamp_min(1.0)
    anchor_mask = (
        torch.ones_like(initial)
        if hr_mask is None else hr_mask.detach().to(device=initial.device, dtype=initial.dtype)
    )
    anchor_denom = anchor_mask.sum().clamp_min(1.0)

    with torch.enable_grad():
        refined = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([refined], lr=float(learning_rate))
        for _ in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            predicted_native = mrsi_native_forward_batch(
                refined,
                lr_matrix,
                window=window,
                clamp_nonnegative=True,
            )
            residual = predicted_native - target
            # Charbonnier-like robust data penalty, less eager to reproduce
            # magnitude-noise spikes than exact squared-error projection.
            data_loss = (
                (torch.sqrt(residual.square() + 1.0e-6) - 1.0e-3) * native_weights
            ).sum() / native_denom
            anchor_loss = ((refined - initial).square() * anchor_mask).sum() / anchor_denom
            (data_loss + float(anchor_weight) * anchor_loss).backward()
            optimizer.step()
            with torch.no_grad():
                refined.clamp_(0.0, 1.0)
        refined = refined.detach()
    return (refined * 2.0 - 1.0).to(dtype=sr_minus1_1.dtype)
