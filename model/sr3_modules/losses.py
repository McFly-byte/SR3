import torch
import torch.nn.functional as F


def _mask_like(mask, ref):
    if mask is None:
        return torch.ones_like(ref)
    return mask.to(device=ref.device, dtype=ref.dtype)


def masked_l1_sum(pred, target, mask=None):
    m = _mask_like(mask, pred)
    return (torch.abs(pred - target) * m).sum()


def roi_mean_consistency_sum(pred, target, mask=None, eps=1e-6):
    m = _mask_like(mask, pred)
    reduce_dims = tuple(range(1, pred.dim()))
    denom = m.sum(dim=reduce_dims).clamp_min(1.0)
    pred_mean = (pred * m).sum(dim=reduce_dims) / denom
    target_mean = (target * m).sum(dim=reduce_dims) / denom
    rel = torch.abs(pred_mean - target_mean) / target_mean.abs().clamp_min(eps)
    return rel.mean() * float(pred.numel())


def gradient_l1_sum(pred, target, mask=None):
    m = _mask_like(mask, pred)
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mx = m[..., :, 1:] * m[..., :, :-1]
    my = m[..., 1:, :] * m[..., :-1, :]
    return (torch.abs(pred_dx - target_dx) * mx).sum() + (torch.abs(pred_dy - target_dy) * my).sum()


def frequency_l1_sum(pred, target, mask=None):
    m = _mask_like(mask, pred)
    pred_fft = torch.fft.rfft2(pred * m, norm="ortho")
    target_fft = torch.fft.rfft2(target * m, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft), reduction="sum")


def _kspace_degrade_batch(x, lowres_half, window="hamming"):
    b, c, h, w = x.shape
    lowres_half = lowres_half.to(device=x.device).view(-1).long()
    k = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
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
    cy, cx = h // 2, w // 2
    for i in range(b):
        div = int(lowres_half[i].item())
        if div <= 0:
            k_small[i] = k[i]
            continue
        y0, y1 = max(0, cy - div), min(h, cy + div)
        x0, x1 = max(0, cx - div), min(w, cx + div)
        k_small[i, :, y0:y1, x0:x1] = k[i, :, y0:y1, x0:x1]
    out = torch.fft.ifft2(torch.fft.ifftshift(k_small, dim=(-2, -1))).abs()
    return out.to(dtype=x.dtype)


def degradation_l1_sum(pred_x0, lr, lowres_half=None, mask=None, window="hamming"):
    pred_01 = ((pred_x0 + 1.0) * 0.5).clamp(0.0, 1.0)
    lr_01 = ((lr + 1.0) * 0.5).clamp(0.0, 1.0)
    if lowres_half is None:
        degraded = pred_01
    else:
        degraded = _kspace_degrade_batch(pred_01, lowres_half, window=window).clamp(0.0, 1.0)
    m = _mask_like(mask, pred_x0)
    return (torch.abs(degraded - lr_01) * m).sum()
