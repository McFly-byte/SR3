"""Dataset field / shape / range tests for ImprovedWaterfilmDataset."""
from pathlib import Path

import pytest
import torch

from data.improved_waterfilm_dataset import ImprovedWaterfilmDataset

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "dataset_mrsi" / "mrsi_sr3_64" / "train"

pytestmark = pytest.mark.skipif(not TRAIN_DIR.exists(),
                                reason="training npz not present")


def _make_ds():
    return ImprovedWaterfilmDataset(
        str(TRAIN_DIR), split="train", data_len=8,
        hflip=False, min_block=8, max_block=32, struct_dropout_prob=0.15,
    )


def test_fields_shapes_and_ranges():
    ds = _make_ds()
    out = ds[0]
    H = 64
    # core condition tensors
    assert out["HR"].shape == (1, H, H)
    assert out["SR"].shape[0] == 7  # lr+t1+flair+4 met (healthy, no mask)
    assert out["SR"].shape[1:] == (H, H)
    assert out["LR"].shape == (1, H, H)
    assert out["MASK"].shape == (1, H, H)
    # [-1,1] domain for HR/SR/LR
    assert out["HR"].min() >= -1.001 and out["HR"].max() <= 1.001
    assert out["SR"].min() >= -1.001 and out["SR"].max() <= 1.001
    assert out["LR"].min() >= -1.001 and out["LR"].max() <= 1.001
    # native LR in [0,1], center-padded to 64
    assert "LR_NATIVE" in out
    assert out["LR_NATIVE"].shape == (1, H, H)
    assert out["LR_NATIVE"].min() >= -1e-6 and out["LR_NATIVE"].max() <= 1.001
    # HAS_NATIVE_LR flag
    assert "HAS_NATIVE_LR" in out
    assert float(out["HAS_NATIVE_LR"]) == pytest.approx(1.0)
    # LR_MATRIX is an even block in [8,32]
    assert "LR_MATRIX" in out
    b = int(out["LR_MATRIX"])
    assert 8 <= b <= 32 and b % 2 == 0
    # the center block of LR_NATIVE must match the block size (nonzero support)
    cy = H // 2
    native = out["LR_NATIVE"][0]
    ring = np_outer = None
    # outside the native n x n region must be ~0
    outside = torch.cat([native[:cy - b // 2].reshape(-1),
                         native[cy + b // 2:].reshape(-1)])
    assert outside.abs().max() < 1e-4, f"native padding not zero: {outside.abs().max()}"
