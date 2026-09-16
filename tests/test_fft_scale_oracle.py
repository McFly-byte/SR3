"""Independent FFT-scale / amplitude oracle test.

Written in plain numpy, NOT calling the code under test except to compare the
implementation.  It pins the bookkeeping:

  * torch/numpy FFT default normalization is "backward" (fft unnormalized,
    ifft divides by N).
  * HR -> native crops the central n x n k-space block and multiplies by
    n^2/N^2, so a constant HR stays constant.
  * The matching *inverse* used at inference zero-pads the native n x n
    k-space to N x N and multiplies by (N/n)^2.
  * Composition inverse(forward(HR)) must recover HR (and vice-versa).
"""
import numpy as np
import torch

from core.mrsi_physics import mrsi_native_forward_batch, crop_padded_native


def _numpy_forward(hr_N, n):
    """Independent HR(N x N) -> native(n x n), window none, correct scale."""
    N = hr_N.shape[0]
    K = np.fft.fftshift(np.fft.fft2(hr_N))
    c = N // 2
    ck = K[c - n // 2:c + n // 2, c - n // 2:c + n // 2]
    native = np.fft.ifft2(np.fft.ifftshift(ck * (n * n) / (N * N)))
    return np.real(native)


def _numpy_inverse(native_n, N):
    """Independent native(n x n) -> HR(N x N), zero-pad + (N/n)^2."""
    n = native_n.shape[0]
    kn = np.fft.fftshift(np.fft.fft2(native_n))
    big = np.zeros((N, N), dtype=np.complex128)
    c = N // 2
    big[c - n // 2:c + n // 2, c - n // 2:c + n // 2] = kn
    hr = np.fft.ifft2(np.fft.ifftshift(big)) * (N / n) ** 2
    return np.real(hr)


def test_constant_preserved_any_matrix():
    N = 64
    for n in (8, 12, 16, 32):
        hr = np.full((N, N), 0.7)
        native = _numpy_forward(hr, n)
        assert np.abs(native - 0.7).max() < 1e-9, f"n={n} maxerr={np.abs(native-0.7).max()}"


def test_roundtrip_forward_backward_recovers_native():
    """Start from a smooth native n x n image -> zero-pad to HR -> forward -> recover.

    The native image is a smooth Gaussian bump (its k-space energy is
    concentrated at the centre, so the even-size corner Hermitian edge
    leakage is negligible).
    """
    N, n = 64, 12
    yy, xx = np.mgrid[0:n, 0:n]
    native0 = np.exp(-(((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 2.0 ** 2)))
    hr = _numpy_inverse(native0, N)               # independent zero-pad + (N/n)^2
    out = mrsi_native_forward_batch(
        torch.from_numpy(hr).float()[None, None], n,
        window="none", clamp_nonnegative=False)
    native_rec = crop_padded_native(out, n)[0, 0].numpy()
    err = np.abs(native_rec - native0).max()
    # The scale bookkeeping itself is exact: constant HR -> 1e-9 and
    # impl-vs-oracle < 1e-5.  The residual here (~6e-4) is the even-size
    # corner Hermitian edge leakage of embedding an n x n block into N x N,
    # not a scale error.
    assert err < 1e-3, f"round-trip max err={err:.2e}"


def test_implementation_matches_independent_forward():
    """torch forward central n x n must equal the independent numpy forward."""
    N, n = 64, 12
    rng = np.random.RandomState(1)
    native0 = rng.rand(n, n) * 0.8 + 0.1
    hr = _numpy_inverse(native0, N)
    native_np = _numpy_forward(hr, n)
    out = mrsi_native_forward_batch(
        torch.from_numpy(hr).float()[None, None], n,
        window="none", clamp_nonnegative=False)
    native_t = crop_padded_native(out, n)[0, 0].numpy()
    err = np.abs(native_t - native_np).max()
    assert err < 1e-5, f"impl vs oracle max err={err:.2e}"


def test_hamming_constant_approx_conserving():
    """With a window, a constant must stay finite and ~constant inside support."""
    N, n = 64, 16
    hr = torch.full((1, 1, N, N), 0.7)
    out = mrsi_native_forward_batch(hr, n, window="hamming",
                                    clamp_nonnegative=True)[0, 0].numpy()
    c = N // 2
    inner = out[c - n // 2:c + n // 2, c - n // 2:c + n // 2]
    assert np.isfinite(out).all()
    # hamming tapers to ~0.54 at the edge of the retained block; center stays ~0.7
    assert inner[n // 2, n // 2] > 0.6
