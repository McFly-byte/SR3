"""Reversible, zero-preserving normalization for quantitative MRSI/DMI maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_normalization_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scales = payload.get("normalization_scales")
    if not isinstance(scales, dict) or not scales:
        raise ValueError(f"{contract_path} has no normalization_scales mapping")
    quantity = payload.get("quantity_contract", {}) or {}
    return {
        "normalization_scales": {str(key): float(value) for key, value in scales.items()},
        "quantity_name": str(quantity.get("quantity_name", "concentration_like_signal")),
        "quantity_unit": str(quantity.get("unit", "simulation_arbitrary_unit")),
        "voxel_semantics": str(quantity.get("voxel_semantics", "intensive")),
        "source": str(contract_path.resolve()),
    }


def normalize_quantity(
    values: np.ndarray,
    scale: float,
    *,
    clip: bool = True,
) -> np.ndarray:
    scale_f = float(scale)
    if not np.isfinite(scale_f) or scale_f <= 0.0:
        raise ValueError(f"normalization scale must be finite and positive, got {scale}")
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("quantity array contains non-finite values")
    normalized = array / scale_f
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)


def denormalize_quantity(values: np.ndarray, scale: float) -> np.ndarray:
    scale_f = float(scale)
    if not np.isfinite(scale_f) or scale_f <= 0.0:
        raise ValueError(f"normalization scale must be finite and positive, got {scale}")
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("normalized array contains non-finite values")
    return (array * scale_f).astype(np.float32, copy=False)

