#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot water-film SR3 inference results.

Reads sr_results.npz and inference_meta.json from each output directory
and generates comparison figures. Must run as a separate process from
inference to avoid OpenMP/matplotlib conflicts.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(output_dir: str, cmap: str = "turbo"):
    out = Path(output_dir)
    npz_path = out / "sr_results.npz"
    meta_path = out / "inference_meta.json"

    if not npz_path.is_file():
        print(f"  SKIP: {npz_path} not found")
        return
    if not meta_path.is_file():
        print(f"  SKIP: {meta_path} not found")
        return

    d = np.load(str(npz_path))
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)

    lr_nearest = d["lr_nearest_64"]
    bicubic = d["bicubic_64"]
    hr = d["hr_64"]
    mask = d["mask_64"]

    model_keys = ["healthy_ema_sr", "healthy_raw_sr", "current_mixed_raw_sr"]
    model_labels = ["healthy_ema SR", "healthy_raw SR", "current_mixed SR"]

    # Get PSNR for each model from meta
    model_psnrs = {}
    for mk in ["healthy_ema", "healthy_raw", "current_mixed_raw"]:
        if mk in meta.get("models", {}):
            model_psnrs[mk] = meta["models"][mk]["best_psnr"]

    bicubic_psnr = meta.get("bicubic_baseline", {}).get("psnr", 0.0)

    # --- Figure 1: comparison_all_models.png ---
    panels = [
        (lr_nearest, "LR (nearest)"),
        (bicubic, f"Bicubic\nPSNR={bicubic_psnr:.2f}"),
    ]
    for mk, ml in zip(["healthy_ema", "healthy_raw", "current_mixed_raw"], model_labels):
        key = f"{mk}_sr"
        if key in d:
            psnr_val = model_psnrs.get(mk, 0.0)
            panels.append((d[key], f"{ml}\nPSNR={psnr_val:.2f}"))
    panels.append((hr, "HR reference"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Normalized signal")
    fig.suptitle(f"Water-film MRSI SR3 comparison — {meta['dataset']}  "
                 f"(orientation={meta['confirmed_orientation']})", fontsize=11)
    fig.savefig(str(out / "comparison_all_models.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_all_models.png")

    # --- Figure 2: comparison_best.png ---
    # Find best model by PSNR
    best_mk = max(model_psnrs, key=model_psnrs.get)
    best_sr = d[f"{best_mk}_sr"]

    diff = np.abs(best_sr - hr)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
    im0 = axes[0].imshow(best_sr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Best SR: {best_mk}\nPSNR={model_psnrs[best_mk]:.2f} dB", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(hr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("HR reference", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max(), interpolation="nearest")
    axes[2].set_title(f"|SR - HR|\nmax={diff.max():.4f}, mean={diff.mean():.5f}", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"Best model vs HR — {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "comparison_best.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_best.png (best model: {best_mk})")

    # --- Figure 3: orientation search bar chart ---
    orient_data = meta.get("orientation_search", {})
    if orient_data:
        names = list(orient_data.keys())
        vals = [orient_data[n] for n in names]
        selected = meta.get("confirmed_orientation", "")
        colors = ["#d62728" if n == selected else "#1f77b4" for n in names]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"D4 Orientation Search — {meta['dataset']}\n(red = selected: {selected})")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(str(out / "orientation_search.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: orientation_search.png")


def main():
    datasets = [
        ("ygh_phantom", r"D:\LMC\data\phantom_ygh\最终处理结果\推理"),
        ("waterfilm_user", r"D:\LMC\data\水膜数据处理\最终处理结果\推理"),
    ]
    print("Generating comparison plots...")
    for name, out_dir in datasets:
        print(f"\n{name}:")
        plot_dataset(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
