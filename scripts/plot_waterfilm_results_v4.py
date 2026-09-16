#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot water-film SR3 inference v4 results.

Generates:
  comparison_all_models.png   - LR(kspace) | bicubic | kspace_baseline | healthy_ema DC | healthy_raw DC | current_mixed DC | HR
  comparison_best.png         - best result (raw/DC/denoised if applicable) vs HR + diff
  comparison_native_resolution.png - native resolution comparison
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resize_np(arr, size, mode="bicubic"):
    import torch
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    if mode == "nearest":
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    else:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


def psnr_native(pred_64, hr_native):
    p = resize_np(pred_64, hr_native.shape[0])
    p_u8 = np.clip(np.round(p * 255), 0, 255).astype(np.float64)
    r_u8 = np.clip(np.round(hr_native * 255), 0, 255).astype(np.float64)
    mse = np.mean((p_u8 - r_u8) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0


def plot_dataset(output_dir: str, cmap: str = "turbo"):
    out = Path(output_dir)
    npz_path = out / "sr_results.npz"
    meta_path = out / "inference_meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        print(f"  SKIP: {output_dir}")
        return

    d = np.load(str(npz_path))
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)

    lr_kspace = d["lr_kspace_64"]
    lr_bicubic = d["lr_bicubic_64"]
    hr_64 = d["hr_bicubic_64"]
    hr_native = d["hr_native"]
    native_size = hr_native.shape[0]

    # --- Figure 1: comparison_all_models.png ---
    panels = [
        (lr_kspace, f"LR k-space zero-pad\n(PSNR_n={meta['baselines']['kspace']['psnr_native']:.2f})"),
        (lr_bicubic, f"LR bicubic\n(PSNR_n={meta['baselines']['bicubic']['psnr_native']:.2f})"),
    ]
    for mkey in ["healthy_ema", "healthy_raw", "current_mixed_raw"]:
        fin_key = f"{mkey}_final_sr"
        if fin_key in d:
            sr = d[fin_key]
            p = psnr_native(sr, hr_native)
            variant = meta["models"][mkey]["best_variant"]
            panels.append((sr, f"{mkey}\n({variant}, PSNR_n={p:.2f})"))
    panels.append((hr_64, "HR reference"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Signal")
    fig.suptitle(f"Water-film MRSI SR3 v4 — {meta['dataset']}  (orientation={meta['confirmed_orientation']})",
                 fontsize=10)
    fig.savefig(str(out / "comparison_all_models.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_all_models.png")

    # --- Figure 2: comparison_best.png ---
    # Find best model
    models_info = meta.get("models", {})
    best_mk = max(models_info.keys(), key=lambda k: models_info[k]["final_native_psnr"])
    best_sr = d[f"{best_mk}_final_sr"]
    best_variant = models_info[best_mk]["best_variant"]

    diff = np.abs(best_sr - hr_64)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    im0 = axes[0].imshow(best_sr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Best: {best_mk} ({best_variant})\nPSNR_n={models_info[best_mk]['final_native_psnr']:.2f}", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(hr_64, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("HR reference (bicubic 64)", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max(), interpolation="nearest")
    axes[2].set_title(f"|SR-HR|\nmax={diff.max():.4f}, mean={diff.mean():.5f}", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    fig.suptitle(f"Best model vs HR — {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "comparison_best.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_best.png (best: {best_mk}, variant: {best_variant})")

    # --- Figure 3: native resolution comparison ---
    kspace_native = resize_np(lr_kspace, native_size)
    bic_native = resize_np(lr_bicubic, native_size)
    best_native = resize_np(best_sr, native_size)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), constrained_layout=True)
    im0 = axes[0].imshow(kspace_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"k-space LR ({native_size}x{native_size})", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(bic_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Bicubic LR ({native_size}x{native_size})", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(best_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Best SR: {best_mk}\n({native_size}x{native_size})", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    im3 = axes[3].imshow(hr_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"Native HR ({native_size}x{native_size})", fontsize=9)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.colorbar(im3, ax=axes[3], shrink=0.7)

    fig.suptitle(f"Native-resolution comparison — {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "comparison_native_resolution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_native_resolution.png")

    # --- Figure 4: orientation search ---
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
        ax.set_ylabel("Native PSNR (dB)")
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
    print("Generating v4 comparison plots...")
    for name, out_dir in datasets:
        print(f"\n{name}:")
        plot_dataset(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
