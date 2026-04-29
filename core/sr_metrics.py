import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _hann2d(h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=True, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=True, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    vals = x[mask]
    if vals.numel() == 0:
        mean = x.mean()
        std = x.std(unbiased=False).clamp(min=eps)
        return mean, std
    mean = vals.mean()
    std = vals.std(unbiased=False).clamp(min=eps)
    return mean, std


def _to_01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if float(x.min().item()) < -0.05:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    x2 = x.squeeze().float()
    if x2.dim() != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got {tuple(x2.shape)}")
    return x2


def _as_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(ref, dtype=torch.bool)
    m = mask.squeeze()
    if m.shape != ref.shape:
        raise ValueError(f"mask shape {tuple(m.shape)} must match image shape {tuple(ref.shape)}")
    return m > 0.5


@torch.no_grad()
def dmi_quant_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Masked DMI/MRSI quantitative metrics on normalized 2D maps."""
    x = _to_01(_as_2d(pred))
    y = _to_01(_as_2d(target))
    m = _as_mask(mask, y)
    mf = m.to(x.dtype)
    denom = mf.sum().clamp_min(1.0)

    diff = (x - y) * mf
    mse = (diff ** 2).sum() / denom
    mae = diff.abs().sum() / denom
    masked_psnr = 10.0 * torch.log10(torch.tensor(1.0, device=x.device, dtype=x.dtype) / mse.clamp_min(eps))

    x_vals = x[m]
    y_vals = y[m]
    if x_vals.numel() == 0:
        x_vals = x.reshape(-1)
        y_vals = y.reshape(-1)
    x_mean = x_vals.mean()
    y_mean = y_vals.mean()
    x_std = x_vals.std(unbiased=False)
    y_std = y_vals.std(unbiased=False)
    x_sum = (x * mf).sum()
    y_sum = (y * mf).sum()

    return {
        "masked_psnr": float(masked_psnr.item()),
        "masked_mae": float(mae.item()),
        "roi_mean_abs_err": float((x_mean - y_mean).abs().item()),
        "roi_mean_rel_err": float(((x_mean - y_mean).abs() / y_mean.abs().clamp_min(eps)).item()),
        "roi_std_abs_err": float((x_std - y_std).abs().item()),
        "roi_std_rel_err": float(((x_std - y_std).abs() / y_std.abs().clamp_min(eps)).item()),
        "roi_sum_abs_err": float((x_sum - y_sum).abs().item()),
        "roi_sum_rel_err": float(((x_sum - y_sum).abs() / y_sum.abs().clamp_min(eps)).item()),
    }


@torch.no_grad()
def false_hotspot_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    quantile: float = 0.95,
) -> Dict[str, float]:
    """Fraction of masked pixels that are hot in prediction but not in target."""
    x = _to_01(_as_2d(pred))
    y = _to_01(_as_2d(target))
    m = _as_mask(mask, y)
    vals = y[m]
    if vals.numel() == 0:
        vals = y.reshape(-1)
    thr = torch.quantile(vals, float(quantile)).clamp_min(1e-6)
    pred_hot = (x >= thr) & m
    target_hot = (y >= thr) & m
    false_hot = pred_hot & (~target_hot)
    denom = m.sum().clamp_min(1)
    pred_hot_denom = pred_hot.sum().clamp_min(1)
    return {
        "false_hotspot_rate": float(false_hot.sum().float().div(denom.float()).item()),
        "false_hotspot_precision_err": float(false_hot.sum().float().div(pred_hot_denom.float()).item()),
        "hotspot_threshold": float(thr.item()),
    }


def _kspace_truncate_2d(x: torch.Tensor, lowres_half: int, window: str = "hamming") -> torch.Tensor:
    h, w = x.shape
    div = int(lowres_half)
    if div <= 0:
        return x
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - div), min(h, cy + div)
    x0, x1 = max(0, cx - div), min(w, cx + div)
    k = torch.fft.fftshift(torch.fft.fft2(x.float()))
    if window and window.lower() not in ("none", "rect", "boxcar"):
        if window.lower() == "hamming":
            wy = torch.hamming_window(h, periodic=False, device=x.device, dtype=x.dtype)
            wx = torch.hamming_window(w, periodic=False, device=x.device, dtype=x.dtype)
        elif window.lower() == "hann":
            wy = torch.hann_window(h, periodic=False, device=x.device, dtype=x.dtype)
            wx = torch.hann_window(w, periodic=False, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported k-space window: {window}")
        k = k * (wy[:, None] * wx[None, :])
    k_small = torch.zeros_like(k)
    k_small[y0:y1, x0:x1] = k[y0:y1, x0:x1]
    return torch.fft.ifft2(torch.fft.ifftshift(k_small)).abs().to(x.dtype)


@torch.no_grad()
def degradation_consistency_2d(
    pred: torch.Tensor,
    lr: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    lowres_half: Optional[int] = None,
    window: str = "hamming",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compare D(pred) with observed LR under the same simple k-space truncation model."""
    x = _to_01(_as_2d(pred))
    y = _to_01(_as_2d(lr))
    if lowres_half is None:
        degraded = x
    else:
        degraded = _kspace_truncate_2d(x, int(lowres_half), window=window).clamp(0.0, 1.0)
    m = _as_mask(mask, y)
    mf = m.to(x.dtype)
    denom = mf.sum().clamp_min(1.0)
    diff = (degraded - y) * mf
    l1 = diff.abs().sum() / denom
    l2 = torch.sqrt((diff ** 2).sum() / denom)
    return {
        "degradation_l1": float(l1.item()),
        "degradation_rmse": float(l2.item()),
    }


@torch.no_grad()
def frc_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    apodize: bool = True,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    FRC-like spectral correlation between pred and target for a single 2D map.
    """
    pred2 = pred.squeeze()
    target2 = target.squeeze()
    if pred2.dim() != 2 or target2.dim() != 2:
        raise ValueError(f"frc_2d expects 2D tensors; got pred={tuple(pred2.shape)} target={tuple(target2.shape)}")

    h, w = int(pred2.shape[-2]), int(pred2.shape[-1])
    x = pred2.float()
    y = target2.float()

    if mask is None:
        m = (y != 0)
    else:
        m = mask.squeeze().bool()
        if m.shape != x.shape:
            raise ValueError(f"mask shape {tuple(m.shape)} must match image shape {tuple(x.shape)}")

    mu, sig = _masked_mean_std(y, m, eps=1e-6)
    x = (x - mu) / sig
    y = (y - mu) / sig

    mw = m.to(x.dtype)
    if apodize:
        mw = mw * _hann2d(h, w, device=x.device, dtype=x.dtype)
    x = x * mw
    y = y * mw

    X = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    Y = torch.fft.fftshift(torch.fft.fft2(y, norm="ortho"))

    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rbin = rr.floor().to(torch.int64)
    rmax = int(min(h, w) // 2)

    vals = []
    counts = []
    freqs = []
    for r in range(0, rmax + 1):
        sel = (rbin == r)
        n = int(sel.sum().item())
        if n <= 0:
            continue
        xk = X[sel]
        yk = Y[sel]
        num = (xk * torch.conj(yk)).sum()
        den = torch.sqrt((xk.abs() ** 2).sum() * (yk.abs() ** 2).sum()).clamp(min=eps)
        frc_r = (num.real / den).clamp(min=-1.0, max=1.0)
        vals.append(float(frc_r.item()))
        counts.append(float(n))
        freqs.append(float(r / max(1, rmax)))

    if len(vals) <= 1:
        return {"frc_auc_w": 0.0, "frc_hf_mean": 0.0, "frc_cutoff_1_7": 0.0}

    frc_t = torch.tensor(vals, device=x.device, dtype=torch.float32)
    cnt_t = torch.tensor(counts, device=x.device, dtype=torch.float32)
    f_t = torch.tensor(freqs, device=x.device, dtype=torch.float32)

    frc1 = frc_t[1:]
    cnt1 = cnt_t[1:]
    f1 = f_t[1:]
    auc_w = float((frc1 * cnt1).sum().div(cnt1.sum().clamp(min=eps)).item())

    hf_sel = (f1 >= 0.5)
    if hf_sel.any():
        frc_hf = frc1[hf_sel]
        cnt_hf = cnt1[hf_sel]
        hf_mean = float((frc_hf * cnt_hf).sum().div(cnt_hf.sum().clamp(min=eps)).item())
    else:
        hf_mean = 0.0

    thr = 1.0 / 7.0
    below = (frc1 < thr)
    if below.any():
        idx = int(torch.argmax(below.to(torch.int64)).item())
        cutoff = float(f1[idx].item())
    else:
        cutoff = 1.0

    return {"frc_auc_w": auc_w, "frc_hf_mean": hf_mean, "frc_cutoff_1_7": cutoff}


def _log_kernel_2d(
    sigma: float,
    trunc: float = 3.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    half = int(math.ceil(float(trunc) * float(sigma)))
    size = 2 * half + 1
    ax = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    s2 = float(sigma) ** 2
    kernel = ((r2 / (s2 ** 2)) - (2.0 / s2)) * torch.exp(-r2 / (2.0 * s2))
    kernel = kernel - kernel.mean()
    return kernel.view(size, size)


@torch.no_grad()
def hfen_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    sigma: float = 1.5,
    trunc: float = 3.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    pred2 = pred.squeeze()
    target2 = target.squeeze()
    if pred2.dim() != 2 or target2.dim() != 2:
        raise ValueError(f"hfen_2d expects 2D tensors; got pred={tuple(pred2.shape)} target={tuple(target2.shape)}")

    x = pred2.float()
    y = target2.float()
    if mask is None:
        m = (y != 0)
    else:
        m = mask.squeeze().bool()
        if m.shape != x.shape:
            raise ValueError(f"mask shape {tuple(m.shape)} must match image shape {tuple(x.shape)}")

    kernel = _log_kernel_2d(sigma=sigma, trunc=trunc, device=x.device, dtype=x.dtype).view(1, 1, -1, -1)
    pad = kernel.shape[-1] // 2
    x_hp = F.conv2d(x.view(1, 1, *x.shape), kernel, padding=pad).squeeze(0).squeeze(0)
    y_hp = F.conv2d(y.view(1, 1, *y.shape), kernel, padding=pad).squeeze(0).squeeze(0)

    m_float = m.to(x_hp.dtype)
    denom = m_float.sum().clamp_min(1.0)
    rmse = torch.sqrt((((x_hp - y_hp) ** 2) * m_float).sum() / denom)
    target_rms = torch.sqrt(((y_hp ** 2) * m_float).sum() / denom).clamp_min(eps)
    nrmse = rmse / target_rms
    return {"hfen_rmse": float(rmse.item()), "hfen_nrmse": float(nrmse.item())}


