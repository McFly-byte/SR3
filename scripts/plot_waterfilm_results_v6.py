#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot v6: display-space figures with axis direction labels. Does NOT overwrite v4/v5."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resize_np(arr, size):
    import torch
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


def psnr_native(pred_64, hr_native):
    p = resize_np(pred_64, hr_native.shape[0])
    pu = np.clip(np.round(p * 255), 0, 255).astype(np.float64)
    ru = np.clip(np.round(hr_native * 255), 0, 255).astype(np.float64)
    mse = np.mean((pu - ru) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0


def add_axis_labels(ax, axis_labels):
    ax.set_xlabel(f"{axis_labels['x']} (mm) →", fontsize=8)
    ax.set_ylabel(f"{axis_labels['y']} (mm) ↓", fontsize=8)


def plot_dataset(out_dir, cmap="turbo"):
    out = Path(out_dir)
    npz_f = out / "sr_results_v6.npz"
    meta_f = out / "inference_meta_v6.json"
    if not npz_f.is_file() or not meta_f.is_file():
        print(f"  SKIP: {out_dir}")
        return

    d = np.load(str(npz_f))
    with open(str(meta_f), "r", encoding="utf-8") as f:
        meta = json.load(f)

    axis_labels = meta["axis_labels"]
    hr_64 = d["display_hr_bicubic_64"]
    hr_native = d["display_hr_native"]
    native_size = hr_native.shape[0]

    models_info = meta.get("models", {})
    best_mk = max(models_info.keys(), key=lambda k: models_info[k]["final_native_psnr"])

    # --- comparison_best_v6.png: raw / dc / dc_denoised / final / HR + diff ---
    panels = []
    for v, lbl in [("raw", "Raw SR"), ("dc", "Data Consistency"),
                   ("dc_denoised", "DC + Denoise(σ=0.5)"), ("final", "Final (best variant)")]:
        key = f"display_{best_mk}_{v}_sr_best" if v != "final" else f"display_{best_mk}_final_sr"
        if key in d:
            arr = d[key]
            p = psnr_native(arr, hr_native)
            panels.append((arr, f"{lbl}\nPSNR_n={p:.2f} dB", arr))
    panels.append((hr_64, "HR reference", None))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4), constrained_layout=True)
    if n == 1: axes = [axes]
    im = None
    for ax, (arr, title, _) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        add_axis_labels(ax, axis_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"{best_mk} — {meta['dataset']} (display space)", fontsize=10)
    fig.savefig(str(out / "comparison_best_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_best_v6.png")

    # --- comparison_all_models_v6.png ---
    row = [
        ("display_lr_kspace_64", "LR (k-space pad)"),
        ("display_baseline_bicubic_64", "Bicubic LR"),
        ("display_healthy_ema_dc_denoised_sr_best", "healthy_ema DC+denoise"),
        ("display_healthy_raw_dc_denoised_sr_best", "healthy_raw DC+denoise"),
        ("display_current_mixed_raw_final_sr", "current_mixed best"),
        ("display_hr_bicubic_64", "HR"),
    ]
    fig, axes = plt.subplots(1, len(row), figsize=(2.4 * len(row), 3.0), constrained_layout=True)
    im = None
    for ax, (k, lbl) in zip(axes, row):
        arr = d[k]
        p = psnr_native(arr, hr_native) if k != "display_hr_bicubic_64" else None
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        title = f"{lbl}" + (f"\nPSNR_n={p:.2f}" if p else "")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        add_axis_labels(ax, axis_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"All models — {meta['dataset']} (display space)", fontsize=10)
    fig.savefig(str(out / "comparison_all_models_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_all_models_v6.png")

    # --- comparison_native_resolution_v6.png ---
    fig, axes = plt.subplots(1, len(row), figsize=(2.4 * len(row), 2.8), constrained_layout=True)
    for ax, (k, lbl) in zip(axes, row):
        arr = resize_np(d[k], native_size)
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(lbl, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        add_axis_labels(ax, axis_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"Native resolution ({native_size}x{native_size}) — {meta['dataset']}", fontsize=10)
    fig.savefig(str(out / "comparison_native_resolution_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_native_resolution_v6.png")

    # --- comparison_variants_v6.png ---
    variants = []
    for v in ["raw", "dc", "dc_denoised"]:
        key = f"display_{best_mk}_{v}_sr_best"
        if key in d:
            variants.append((v, d[key]))
    if f"display_{best_mk}_final_sr" in d:
        variants.append(("final", d[f"display_{best_mk}_final_sr"]))
    variants.append(("HR", hr_64))
    labels = {"raw": "Raw", "dc": "DC", "dc_denoised": "DC+denoise", "final": "Final", "HR": "HR"}
    fig, axes = plt.subplots(1, len(variants), figsize=(2.4 * len(variants), 3.0), constrained_layout=True)
    for ax, (v, arr) in zip(axes, variants):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        p = psnr_native(arr, hr_native) if v != "HR" else 0
        ax.set_title(f"{labels[v]}\nPSNR_n={p:.2f}" if v != "HR" else "HR", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        add_axis_labels(ax, axis_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"Variants — {best_mk} — {meta['dataset']}", fontsize=10)
    fig.savefig(str(out / "comparison_variants_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_variants_v6.png")

    # --- comparison_variants_native_v6.png ---
    fig, axes = plt.subplots(1, len(variants), figsize=(2.4 * len(variants), 2.8), constrained_layout=True)
    for ax, (v, arr) in zip(axes, variants):
        nat = resize_np(arr, native_size)
        im = ax.imshow(nat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(labels[v], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        add_axis_labels(ax, axis_labels)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"Native variants ({native_size}x{native_size}) — {meta['dataset']}", fontsize=10)
    fig.savefig(str(out / "comparison_variants_native_v6.png"), dpi=150)
    plt.close(fig)
    print("  Saved: comparison_variants_native_v6.png")

    # --- orientation_explanation_v6.png ---
    ref_path = meta.get("reference_image_path", "")
    ref_base = Path(ref_path).name if ref_path else "(n/a)"
    # ASCII-only fallback: strip non-ASCII chars from basename just in case
    ref_base = "".join(ch if ord(ch) < 128 else "_" for ch in ref_base)

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    m_hr = d["model_hr_bicubic_64"]
    ax0.imshow(m_hr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax0.set_title("Model space (identity)\nrows=Phase, cols=Read", fontsize=9)
    ax0.set_xlabel("Read ->"); ax0.set_ylabel("Phase (down)")
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1.imshow(hr_64, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title(f"Display space (transform={meta['display_transform']})\nx={axis_labels['x']}, y={axis_labels['y']}",
                  fontsize=9)
    ax1.set_xlabel(f"{axis_labels['x']} ->"); ax1.set_ylabel(f"{axis_labels['y']} (down)")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2.axis("off")
    note = (
        f"Reference image (basename):\n  {ref_base}\n\n"
        f"Model space: identity (raw h5py)\n"
        f"Display transform: {meta['display_transform']}\n"
        f"Axis (display): x={axis_labels['x']}, y={axis_labels['y']}\n"
        f"\nD4 def fix: anti_transpose = flip(a.T)\n"
        f"  (was flipud(rot90(a)) == a.T, a code duplication)\n"
        f"\nMetric invariance (model vs display):\n"
        f"  pass = {meta['metric_invariance_check']['pass']}\n"
        f"  max|dPSNR| = {meta['metric_invariance_check']['max_abs_diff']:.2e}\n"
        f"  tolerance  = {meta['metric_invariance_check']['tolerance']:.0e}"
    )
    ax2.text(0.0, 1.0, note, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(f"Model vs Display space - {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "orientation_explanation_v6.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: orientation_explanation_v6.png")


def main():
    datasets = [
        ("ygh", r"D:\LMC\data\phantom_ygh\最终处理结果\推理"),
        ("user", r"D:\LMC\data\水膜数据处理\最终处理结果\推理"),
    ]
    print("Generating v6 display-space plots...")
    for n, od in datasets:
        print(f"\n{n}:")
        plot_dataset(od)
    print("\nDone.")


if __name__ == "__main__":
    main()
