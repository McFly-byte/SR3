"""Improved, composable loss terms for the water-film SR3 fine-tuning.

These are *standalone* and do not modify model/sr3_modules/losses.py.  The
improved training entry can add them on top of the existing epsilon-prediction
objective.  All functions operate on tensors in the [-1, 1] convention used by
the SR3 pipeline unless noted otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from core.mrsi_physics import native_support_mask_batch
from model.sr3_modules.losses import native_acquisition_l1_sum


def _to01(x_m11: torch.Tensor) -> torch.Tensor:
    return ((x_m11 + 1.0) * 0.5).clamp(0.0, 1.0)


def charbonnier_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     mask: torch.Tensor | None = None,
                     eps: float = 1e-3) -> torch.Tensor:
    """Robust L1 (Charbonnier) averaged over valid pixels."""
    diff = pred - target
    if mask is None:
        denom = diff.numel()
        return torch.sqrt(diff.square() + eps ** 2).sum() / max(denom, 1)
    m = mask.to(pred.dtype)
    denom = m.sum().clamp_min(1.0)
    return (torch.sqrt(diff.square() + eps ** 2) * m).sum() / denom


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim_loss(pred: torch.Tensor,
             target: torch.Tensor,
             mask: torch.Tensor | None = None,
             window_size: int = 11,
             sigma: float = 1.5) -> torch.Tensor:
    """1 - mean(SSIM), on [-1,1] tensors.  Mask only trims the average."""
    ch = pred.shape[1]
    g1d = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = g1d[:, None] * g1d[None, :]
    window = window.expand(ch, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad, groups=ch)
    mu_t = F.conv2d(target, window, padding=pad, groups=ch)
    sp = F.conv2d(pred * pred, window, padding=pad, groups=ch) - mu_p ** 2
    st = F.conv2d(target * target, window, padding=pad, groups=ch) - mu_t ** 2
    spt = F.conv2d(pred * target, window, padding=pad, groups=ch) - mu_p * mu_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * spt + c2)) / \
               ((mu_p ** 2 + mu_t ** 2 + c1) * (sp + st + c2))
    if mask is not None:
        m = mask.to(ssim_map.dtype)
        denom = m.sum().clamp_min(1.0)
        return 1.0 - (ssim_map * m).sum() / denom
    return 1.0 - ssim_map.mean()


def lowfreq_consistency_loss(pred_x0_m11: torch.Tensor,
                             lr_native_padded: torch.Tensor,
                             lr_matrix,
                             mask: torch.Tensor | None = None,
                             window: str = "hamming") -> torch.Tensor:
    """Data-fidelity on the *native* acquisition grid (low-frequency lock).

    Wraps the verified ``native_acquisition_l1_sum``.  Returns a mean loss
    (not scaled by numel) so it can be weighted directly in the total.
    """
    val = native_acquisition_l1_sum(
        pred_x0_m11,
        lr_native_padded,
        lr_matrix,
        mask=mask,
        window=window,
    )
    # native_acquisition_l1_sum multiplies by numel for the original caller;
    # divide it back to obtain a plain mean term.
    return val / float(pred_x0_m11.numel())


class ImprovedLossWeights:
    """Container so the config stays declarative."""

    def __init__(self,
                 epsilon: float = 1.0,
                 charbonnier: float = 0.0,
                 ssim: float = 0.0,
                 lowfreq: float = 0.0):
        self.epsilon = float(epsilon)
        self.charbonnier = float(charbonnier)
        self.ssim = float(ssim)
        self.lowfreq = float(lowfreq)
