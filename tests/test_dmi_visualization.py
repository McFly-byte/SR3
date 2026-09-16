import numpy as np
from pathlib import Path

from core.dmi_visualization import (
    estimate_foreground_mask,
    orient_for_display,
    robust_display_limits,
    signed_error_limit,
)
from scripts.visualize_sr3_mrsi_run import _stem_kind


def test_foreground_mask_ignores_constant_border_background():
    image = np.zeros((8, 8), dtype=np.float32)
    image[2:6, 2:6] = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
    mask = estimate_foreground_mask(image)
    assert not mask[0, 0]
    assert mask[3, 3]


def test_robust_limits_use_reference_foreground_and_shared_arrays():
    hr = np.zeros((10, 10), dtype=np.float32)
    hr[2:8, 2:8] = 2.0
    sr = hr.copy()
    lr = hr.copy()
    sr[4, 4] = 1000.0
    vmin, vmax = robust_display_limits((lr, sr, hr), reference=hr, percentiles=(1, 99))
    assert vmin == 2.0
    assert vmax < 1000.0


def test_signed_error_limit_is_positive_and_robust():
    hr = np.zeros((8, 8), dtype=np.float32)
    hr[1:7, 1:7] = 1.0
    diff = np.zeros_like(hr)
    diff[1:7, 1:7] = 0.1
    diff[3, 3] = 10.0
    limit = signed_error_limit(diff, reference=hr, percentile=95)
    assert 0.09 <= limit < 10.0


def test_orient_for_display_matches_simulation_rot90():
    image = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(orient_for_display(image, 1), np.array([[2, 4], [1, 3]]))


def test_validation_sr_zero_suffix_is_recognised_as_sr():
    assert _stem_kind(Path("50500_1_sr_0.png")) == ("50500_1", "sr")
