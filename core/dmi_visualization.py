"""Shared display helpers for DMI/MRSI pseudo-colour figures.

The simulation pipeline under ``simulated_with_lesion`` displays metabolite
maps with a brain-aware 1st--99th percentile window.  Validation PNG files do
not carry the brain mask, so this module estimates the foreground from the HR
reference border/background and applies the same robust-window principle.

These helpers are intentionally display-only.  Metric computation must keep
using the original tensors rather than clipped or colour-mapped images.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


DEFAULT_PERCENTILES: Tuple[float, float] = (1.0, 99.0)
DEFAULT_ROTATE_K = 1


def validate_percentiles(percentiles: Sequence[float]) -> Tuple[float, float]:
    """Return a validated ``(low, high)`` percentile pair."""
    if len(percentiles) != 2:
        raise ValueError("percentiles must contain exactly two values")
    low, high = float(percentiles[0]), float(percentiles[1])
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            f"percentiles must satisfy 0 <= low < high <= 100, got {(low, high)}"
        )
    return low, high


def orient_for_display(array: np.ndarray, rotate_k: int = DEFAULT_ROTATE_K) -> np.ndarray:
    """Rotate a 2-D map counter-clockwise in 90-degree increments."""
    data = np.asarray(array)
    if data.ndim != 2:
        raise ValueError(f"display array must be 2-D, got shape={data.shape}")
    return np.rot90(data, k=int(rotate_k) % 4)


def estimate_foreground_mask(reference: np.ndarray) -> np.ndarray:
    """Estimate a brain/foreground mask from an HR map.

    Validation images have a spatially constant background.  Its value is
    estimated from the image border, while a half-quantisation-step tolerance
    prevents tiny encoding noise from turning the entire field of view into
    foreground.
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError(f"reference must be 2-D, got shape={ref.shape}")
    finite = np.isfinite(ref)
    if not finite.any():
        return finite

    border = np.concatenate((ref[0, :], ref[-1, :], ref[:, 0], ref[:, -1]))
    border = border[np.isfinite(border)]
    background = float(np.median(border)) if border.size else float(np.nanmin(ref))
    finite_values = ref[finite]
    span = float(np.max(finite_values) - np.min(finite_values))
    tolerance = max(span / 510.0, np.finfo(np.float64).eps * 32.0)
    mask = finite & (np.abs(ref - background) > tolerance)

    # A nearly empty estimate is less useful than the finite support itself.
    min_pixels = max(4, int(np.ceil(0.01 * ref.size)))
    if int(mask.sum()) < min_pixels:
        return finite
    return mask


def robust_display_limits(
    arrays: Iterable[np.ndarray],
    reference: np.ndarray | None = None,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> Tuple[float, float]:
    """Compute shared robust limits for comparable metabolite maps."""
    low_pct, high_pct = validate_percentiles(percentiles)
    foreground = estimate_foreground_mask(reference) if reference is not None else None
    values = []
    for array in arrays:
        data = np.asarray(array, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"display array must be 2-D, got shape={data.shape}")
        valid = np.isfinite(data)
        if foreground is not None and foreground.shape == data.shape:
            valid &= foreground
        selected = data[valid]
        if selected.size:
            values.append(selected)

    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    vmin, vmax = np.percentile(merged, (low_pct, high_pct))
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmax <= vmin:
        scale = max(abs(vmin), 1.0)
        vmax = vmin + np.finfo(np.float64).eps * scale * 32.0
    return vmin, vmax


def signed_error_limit(
    difference: np.ndarray,
    reference: np.ndarray | None = None,
    percentile: float = 99.0,
) -> float:
    """Return a symmetric robust limit for a signed error map."""
    pct = float(percentile)
    if not (0.0 < pct <= 100.0):
        raise ValueError(f"percentile must satisfy 0 < p <= 100, got {pct}")
    diff = np.asarray(difference, dtype=np.float64)
    valid = np.isfinite(diff)
    if reference is not None:
        foreground = estimate_foreground_mask(reference)
        if foreground.shape == diff.shape:
            valid &= foreground
    selected = np.abs(diff[valid])
    if not selected.size:
        return 1.0
    limit = float(np.percentile(selected, pct))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(selected)) if selected.size else 1.0
    return max(limit, np.finfo(np.float64).eps * 32.0)
