"""Improved, testable degradation / acquisition operators for water-film MRSI SR3.

Design goals (see reports/model_diagnosis.md):
  * Cover the *real* acquisition matrices (8x8, 12x12) that the shipped models
    never saw.  Training used a retained k-space block of 16..32 px on the
    64-grid; here we allow any even block in ``[min_block, max_block]``.
  * Correct IFFT amplitude normalization so that a spatially constant map stays
    constant regardless of matrix size.  The forward op reuses the already
    verified ``core.mrsi_physics.mrsi_native_forward_batch`` (scale = n^2/h^2).
  * Provide matching numpy utilities for inference (zero-pad up-sample and
    hard data-consistency) with the exact normalization factors used by the
    existing v5 inference script, so results stay comparable.

Nothing here overwrites data/prepare_mrsi_sr3_pairs.py or core/mrsi_physics.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from core.mrsi_physics import mrsi_native_forward_batch


# ---------------------------------------------------------------------------
# block sampling
# ---------------------------------------------------------------------------
def sample_kspace_block(rng: np.random.RandomState,
                        min_block: int = 8,
                        max_block: int = 32) -> int:
    """Draw an even retained-k-space block side length.

    The real water-film acquisitions use block 12 (ygh) and 8 (user); the
    shipped models only ever saw block in [16, 32].  This range deliberately
    starts at 8 so the model is trained on the out-of-distribution cases.
    """
    lo = int(min_block)
    hi = int(max_block)
    if lo % 2 != 0 or hi % 2 != 0:
        raise ValueError("block sizes must be even (square center block).")
    return int(rng.randint(lo // 2, hi // 2 + 1) * 2)


# ---------------------------------------------------------------------------
# differentiable training-side HR -> 64-grid LR condition
# ---------------------------------------------------------------------------
def degrade_hr_to_lr_condition(hr_01: torch.Tensor,
                               block: int,
                               window: str = "hamming") -> torch.Tensor:
    """HR in [0,1] (B,1,H,W) -> center-padded native LR on the same HxW grid.

    This is the *condition* image fed to the SR3 U-Net.  It is the measured
    native-LR image centered back onto the HR canvas.  A constant HR maps to a
    constant LR (intensive concentration semantics).
    """
    if hr_01.dim() != 4:
        raise ValueError(f"Expected BCHW, got {tuple(hr_01.shape)}")
    b, c, h, w = hr_01.shape
    if h != w:
        raise ValueError(f"Only square HR supported, got {(h, w)}")
    out = mrsi_native_forward_batch(hr_01, int(block), window=window,
                                   clamp_nonnegative=True)
    return out.clamp(0.0, 1.0).to(hr_01.dtype)


# ---------------------------------------------------------------------------
# numpy inference-side utilities (mirror infer_waterfilm_phantom_v5.py)
# ---------------------------------------------------------------------------
def zeropad_lr_to_canvas(lr_small: np.ndarray,
                         out_size: int = 64,
                         window: Optional[np.ndarray] = None) -> np.ndarray:
    """Zero-pad a small measured LR image onto the ``out_size`` canvas.

    Applies the IFFT normalization correction ``(out/h)*(out/w)`` so the
    amplitude matches the 64-grid training condition.
    """
    lr_small = np.asarray(lr_small, dtype=np.float32)
    h, w = lr_small.shape
    k = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    if window is not None:
        k = k * window.astype(np.float32)
    k_big = np.zeros((out_size, out_size), dtype=np.complex64)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_big[y0:y0 + h, x0:x0 + w] = k
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(k_big))).astype(np.float32)
    return img * (out_size / h) * (out_size / w)


def data_consistency(sr64: np.ndarray,
                     lr_small: np.ndarray,
                     out_size: int = 64) -> np.ndarray:
    """Hard DC: replace the center k-space of ``sr64`` with measured LR k-space.

    ``lr_small`` k-space is scaled by ``out_size^2 / (h*w)`` before insertion,
    matching the existing v5 post-processing.
    """
    sr64 = np.asarray(sr64, dtype=np.float32)
    lr_small = np.asarray(lr_small, dtype=np.float32)
    h, w = lr_small.shape
    k_sr = np.fft.fftshift(np.fft.fft2(sr64.astype(np.float32)))
    k_lr = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_lr_scaled = k_lr * (out_size ** 2) / (h * w)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_sr[y0:y0 + h, x0:x0 + w] = k_lr_scaled
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k_sr))).astype(np.float32)


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def radial_energy_profile(img: np.ndarray) -> np.ndarray:
    k = np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(img, dtype=np.float64)))) ** 2
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    Y, X = np.ogrid[:k.shape[0], :k.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    shells = np.arange(0, min(cy, cx) + 1)
    prof = np.zeros(len(shells))
    for s in shells:
        m = (r >= s - 0.5) & (r < s + 0.5)
        prof[s] = k[m].mean() if m.any() else 0.0
    tot = prof.sum()
    return prof / tot if tot > 0 else prof


def hf_fraction(img: np.ndarray) -> float:
    """Fraction of total power in the outer half of k-radius."""
    p = radial_energy_profile(img)
    return float(p[len(p) // 2:].sum())


def constant_image(size: int = 64, value: float = 0.7) -> torch.Tensor:
    return torch.full((1, 1, size, size), float(value), dtype=torch.float32)
