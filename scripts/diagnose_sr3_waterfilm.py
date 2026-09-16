#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical diagnosis: why the local SR3 models under-perform on the two
water-film datasets.  This script is READ-ONLY with respect to the input
.mat / .npz evaluation data; the held-out HR is only used to compute spectra
and a few descriptive numbers, never to train or calibrate anything.

Outputs:
  reports/model_diagnosis.json
  reports/figures/diagnosis_truncation.png
  reports/figures/diagnosis_spectrum.png
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement")
REPORT_DIR = REPO / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 64
# Training degradation distribution, from data/prepare_mrsi_sr3_pairs.py
#   lowres_half = rng.randint(8, 16)  -> retained k-space block = 2*half in [16,32]
TRAIN_HALF_MIN, TRAIN_HALF_MAX = 8, 16


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def load_ygh():
    p = Path(r"D:\LMC\data\phantom_ygh\最终处理结果\results\reprocessing_results.mat")
    with h5py.File(str(p), "r") as f:
        rec = f["records"]

        def deref(field, idx):
            return np.squeeze(f[rec[field][idx, 0]][()]).astype(np.float64)

        return {
            "lr": deref("volNormMap", 0),   # 12x12
            "hr": deref("volNormMap", 1),   # 24x24
        }


def load_waterfilm():
    p = Path(r"D:\LMC\data\水膜数据处理\最终处理结果\results\all_results.mat")
    with h5py.File(str(p), "r") as f:
        lr = np.array(f["allResults"]["scan29"]["peakAreaVolNorm"]).astype(np.float64)  # 8x8
        hr = np.array(f["allResults"]["scan30"]["peakAreaVolNorm"]).astype(np.float64)  # 12x12
    return {"lr": lr, "hr": hr}


# ----------------------------------------------------------------------------
# physics helpers (mirror scripts/infer_waterfilm_phantom_v5.py)
# ----------------------------------------------------------------------------
def kspace_zeropad_lr(lr_small, out=TARGET):
    h, w = lr_small.shape
    k = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    big = np.zeros((out, out), dtype=np.complex64)
    cy, cx = out // 2, out // 2
    y0, x0 = cy - h // 2, cx - w // 2
    big[y0:y0 + h, x0:x0 + w] = k
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(big))).astype(np.float32)
    return img * (out / h) * (out / w)


def radial_power(img, out=TARGET):
    """Mean energy per radial k-shell, normalized to [0,1]."""
    k = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))) ** 2
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    Y, X = np.ogrid[: k.shape[0], : k.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    shells = np.arange(0, min(cy, cx) + 1)
    prof = np.zeros(len(shells))
    for s in shells:
        m = (r >= s - 0.5) & (r < s + 0.5)
        prof[s] = k[m].mean() if m.any() else 0.0
    tot = prof.sum()
    return prof / tot if tot > 0 else prof


def support_fraction(lr_small, out=TARGET):
    """Fraction of total k-space energy contained within the measured support."""
    h, w = lr_small.shape
    k = np.abs(np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float64)))) ** 2
    big = np.zeros((out, out))
    cy, cx = out // 2, out // 2
    y0, x0 = cy - h // 2, cx - w // 2
    # measured support on the 64 grid is exactly the central h x w block
    big[y0:y0 + h, x0:x0 + w] = 1.0
    k64 = np.zeros((out, out), dtype=np.complex128)
    k64[y0:y0 + h, x0:x0 + w] = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float64)))
    pw = np.abs(k64) ** 2
    return float(pw[big > 0.5].sum() / pw.sum())


# ----------------------------------------------------------------------------
def analyze(name, d, v5_npz):
    lr, hr = d["lr"], d["hr"]
    h, w = lr.shape
    hs, ws = hr.shape
    block = h  # square
    half = block // 2

    lr64 = kspace_zeropad_lr(lr / lr.max())
    hr_big = np.kron(hr / hr.max(), np.ones((TARGET // hs, TARGET // ws))) if (TARGET % hs == 0) else None

    out = {
        "dataset": name,
        "lr_shape": [int(h), int(w)],
        "hr_shape": [int(hs), int(ws)],
        "lr_max": float(lr.max()),
        "hr_max": float(hr.max()),
        "measured_kspace_block": int(block),
        "measured_kspace_half": int(half),
        "training_kspace_block_range": [2 * TRAIN_HALF_MIN, 2 * TRAIN_HALF_MAX],
        "training_kspace_half_range": [TRAIN_HALF_MIN, TRAIN_HALF_MAX],
        "in_oov_beyond_min_training": bool(half < TRAIN_HALF_MIN),
        "ratio_to_min_training_block": float(block / (2 * TRAIN_HALF_MIN)),
        "effective_downsample_vs_64": float(TARGET / block),
        "effective_downsample_vs_64_train_range": [TARGET / (2 * TRAIN_HALF_MAX), TARGET / (2 * TRAIN_HALF_MIN)],
        "kspace_energy_in_support": support_fraction(lr / lr.max()),
    }

    # radial spectra on the 64 grid
    spec = {
        "lr_kspace64": radial_power(lr64),
    }
    if v5_npz is not None:
        for kk in ["healthy_ema_raw_sr_best", "healthy_ema_dc_sr_best",
                   "current_mixed_raw_raw_sr_best", "current_mixed_raw_dc_sr_best"]:
            if kk in v5_npz.files:
                spec[kk] = radial_power(np.asarray(v5_npz[kk]))
        if "hr_bicubic_64" in v5_npz.files:
            spec["hr_bicubic64"] = radial_power(np.asarray(v5_npz["hr_bicubic_64"]))
    out["spectra_keys"] = list(spec.keys())

    # high-frequency energy ratio (> outer half of radius) for raw vs DC
    def hf_ratio(profile):
        n = len(profile)
        return float(profile[n // 2:].sum())

    if v5_npz is not None:
        out["hf_energy_ratios"] = {}
        for kk in spec:
            out["hf_energy_ratios"][kk] = hf_ratio(spec[kk])

    return out, spec, lr64, hr_big


def main():
    ygh = load_ygh()
    wf = load_waterfilm()

    v5_ygh = np.load(r"D:\LMC\data\phantom_ygh\最终处理结果\推理\sr_results_v5.npz")
    v5_wf = np.load(r"D:\LMC\data\水膜数据处理\最终处理结果\推理\sr_results_v5.npz")

    results = {}
    specs = {}
    res64 = {}
    for name, d, v5 in [("ygh_phantom", ygh, v5_ygh), ("waterfilm_user", wf, v5_wf)]:
        r, sp, lr64, _ = analyze(name, d, v5)
        results[name] = r
        specs[name] = sp
        res64[name] = lr64

    # ---- Figure 1: truncation support comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(["train\n(min 16)", "train\n(median 24)", "train\n(max 32)",
            "ygh\n(12)", "waterfilm\n(8)"],
           [16, 24, 32, 12, 8],
           color=["#999", "#bbb", "#ccc", "#d62728", "#1f77b4"])
    ax.set_ylabel("retained k-space block (px)")
    ax.set_title("k-space truncation: training vs real")
    for i, v in enumerate([16, 24, 32, 12, 8]):
        ax.text(i, v + 0.3, str(v), ha="center")
    ax = axes[1]
    eff = [TARGET / (2 * TRAIN_HALF_MAX), TARGET / (2 * TRAIN_HALF_MIN),
           TARGET / 12, TARGET / 8]
    ax.bar(["train low\n(2x)", "train high\n(4x)", "ygh\n(5.3x)", "waterfilm\n(8x)"],
           eff, color=["#999", "#bbb", "#d62728", "#1f77b4"])
    ax.set_ylabel("effective downsample vs 64")
    ax.set_title("degradation severity vs 64 canvas")
    for i, v in enumerate(eff):
        ax.text(i, v + 0.1, f"{v:.1f}x", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnosis_truncation.png", dpi=130)
    plt.close(fig)

    # ---- Figure 2: radial spectra ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, ["ygh_phantom", "waterfilm_user"]):
        sp = specs[name]
        r = np.arange(len(sp["lr_kspace64"])) / (len(sp["lr_kspace64"]) - 1)
        for kk, col in [("lr_kspace64", "#2ca02c"),
                        ("hr_bicubic64", "#7f7f7f"),
                        ("healthy_ema_raw_sr_best", "#d62728"),
                        ("healthy_ema_dc_sr_best", "#9467bd"),
                        ("current_mixed_raw_raw_sr_best", "#ff7f0e"),
                        ("current_mixed_raw_dc_sr_best", "#17becf")]:
            if kk in sp:
                ax.plot(r, sp[kk], label=kk, lw=1.6)
        ax.axvline(0.5, color="k", ls="--", alpha=0.4)
        ax.set_xlabel("normalized k radius")
        ax.set_ylabel("fractional power / shell")
        ax.set_title(f"radial k-power: {name}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnosis_spectrum.png", dpi=130)
    plt.close(fig)

    # ---- Figure 3: grid of actual images (ygh) ----
    v5 = v5_ygh
    panels = [("lr_kspace_64", v5["lr_kspace_64"]),
              ("hr_bicubic_64", v5["hr_bicubic_64"]),
              ("healthy_ema_raw", v5["healthy_ema_raw_sr_best"]),
              ("healthy_ema_dc", v5["healthy_ema_dc_sr_best"]),
              ("current_raw", v5["current_mixed_raw_raw_sr_best"]),
              ("current_dc", v5["current_mixed_raw_dc_sr_best"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for ax, (t, im) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle("ygh: LR zero-pad / HR(64) / raw SR / DC SR", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnosis_images_ygh.png", dpi=130)
    plt.close(fig)

    diag = {
        "training_degradation": {
            "source": "data/prepare_mrsi_sr3_pairs.py:47-64,254-256",
            "hr_canvas": TARGET,
            "lowres_half_range_used": [TRAIN_HALF_MIN, TRAIN_HALF_MAX],
            "retained_block_range": [2 * TRAIN_HALF_MIN, 2 * TRAIN_HALF_MAX],
            "window": "hamming applied on full 64-grid k-space before central-block extraction",
            "normalization": "hr /= per-patient-per-metabolite max (config met_norm); then [0,1]->[-1,1]",
            "target": "epsilon (noise) prediction, mask-weighted L1 (diffusion.py:537-542)",
            "condition_dropout_prob": 0.0,
        },
        "datasets": results,
        "held_out_note": "HR of both datasets used ONLY for spectrum/metrics here, never enters training.",
    }
    with (REPORT_DIR / "model_diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    print(json.dumps(diag, indent=2, ensure_ascii=False))
    print("\nFigures written to", FIG_DIR)


if __name__ == "__main__":
    main()
