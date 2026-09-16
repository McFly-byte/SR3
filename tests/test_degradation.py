"""Tests for core/improved_degradation.py."""
import numpy as np
import torch

from core.improved_degradation import (
    data_consistency,
    degrade_hr_to_lr_condition,
    hf_fraction,
    sample_kspace_block,
    zeropad_lr_to_canvas,
)


def test_sample_block_even_and_in_range():
    rng = np.random.RandomState(0)
    for _ in range(50):
        b = sample_kspace_block(rng, 8, 32)
        assert 8 <= b <= 32
        assert b % 2 == 0


def test_degrade_shape_and_nonneg():
    hr = torch.rand(2, 1, 64, 64)
    for block in (8, 12, 16, 24, 32):
        lr = degrade_hr_to_lr_condition(hr, block, window="hamming")
        assert lr.shape == (2, 1, 64, 64)
        assert torch.isfinite(lr).all()
        assert lr.min() >= -1e-5


def test_amplitude_conservation_constant():
    """A constant HR maps to a constant value inside the native n x n support."""
    for value in (0.3, 0.7, 1.0):
        hr = torch.full((1, 1, 64, 64), value)
        for block in (8, 12, 16, 32):
            lr = degrade_hr_to_lr_condition(hr, block, window="none")
            cy = 64 // 2
            native = lr[0, 0, cy - block // 2:cy + block // 2,
                        cy - block // 2:cy + block // 2]
            assert torch.allclose(native, torch.full_like(native, value), atol=2e-3), \
                f"block={block} value={value} mean={native.mean().item():.4f}"


def test_zeropad_and_dc_roundtrip_energy():
    lr_small = np.random.RandomState(0).rand(12, 12) * 0.8 + 0.1
    lr64 = zeropad_lr_to_canvas(lr_small, 64)
    assert lr64.shape == (64, 64)
    # almost all energy must stay in the central 12x12 support
    prof = np.abs(np.fft.fftshift(np.fft.fft2(lr64))) ** 2
    cy, cx = 32, 32
    center = prof[cy - 6:cy + 6, cx - 6:cx + 6].sum()
    assert center / prof.sum() > 0.99

    # DC applied to a zero SR must recover the exact zero-pad baseline
    zero_sr = np.zeros((64, 64), dtype=np.float32)
    dc = data_consistency(zero_sr, lr_small, 64)
    assert np.allclose(dc, lr64, atol=1e-3)


def test_hf_fraction_low():
    lr_small = np.ones((8, 8))
    lr64 = zeropad_lr_to_canvas(lr_small, 64)
    assert hf_fraction(lr64) < 0.05
