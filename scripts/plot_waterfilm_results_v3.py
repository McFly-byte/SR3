#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot water-film SR3 inference v3 results.

Generates:
  comparison_all_models.png   - LR(kspace) | bicubic | healthy_ema raw/denoised | current_mixed | HR
  comparison_best.png         - best model raw/denoised/final vs HR + diff
  comparison_native_resolution.png - SR downsampled to native HR size vs native HR
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def resize_np(arr, size, mode="bicubic"):
    import torch
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    if mode == "nearest":
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    else:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


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
    mask_64 = d["mask_64"]
    native_size = hr_native.shape[0]

    # --- Figure 1: comparison_all_models.png ---
    panels = [
        (lr_kspace, "LR (k-space zero-pad)"),
        (lr_bicubic, f"Bicubic\nPSNR={meta['bicubic_baseline_native']['psnr']:.2f}"),
    ]
    for mkey in ["healthy_ema", "healthy_raw", "current_mixed_raw"]:
        raw_key = f"{mkey}_raw_sr"
        den_key = f"{mkey}_denoised_sr"
        if raw_key in d:
            raw = d[raw_key]
            den = d[den_key] if den_key in d else raw
            # Evaluate native PSNR
            den_native = resize_np(den, native_size)
            from skimage.metrics import structural_similarity as sk_ssim
            m = mask_64.astype(bool)
            p = den_native
            r = hr_native
            p_u8 = np.clip(np.round(p * 255), 0, 255).astype(np.float64)
            r_u8 = np.clip(np.round(r * 255), 0, 255).astype(np.float64)
            mse = np.mean((p_u8 - r_u8) ** 2)
            psnr_v = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0
            panels.append((den, f"{mkey}\n(denoised, native PSNR={psnr_v:.2f})"))
    panels.append((hr_64, "HR reference"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Normalized signal")
    fig.suptitle(f"Water-film MRSI SR3 v3 — {meta['dataset']}  (orientation={meta['confirmed_orientation']})",
                 fontsize=10)
    fig.savefig(str(out / "comparison_all_models.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_all_models.png")

    # --- Figure 2: comparison_best.png ---
    # Find best model by final_native_psnr
    models_info = meta.get("models", {})
    best_mk = max(models_info.keys(), key=lambda k: models_info[k]["final_native_psnr"])

    raw_key = f"{best_mk}_raw_sr"
    den_key = f"{best_mk}_denoised_sr"
    fin_key = f"{best_mk}_final_sr"
    sr_raw = d[raw_key]
    sr_den = d[den_key]
    sr_fin = d[fin_key]

    diff_raw = np.abs(sr_raw - hr_64)
    diff_fin = np.abs(sr_fin - hr_64)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    row_labels = ["Raw", "Final (denoised+blend)"]
    for row, (sr, diff, label) in enumerate([(sr_raw, diff_raw, "Raw"), (sr_fin, diff_fin, "Final")]):
        im0 = axes[row, 0].imshow(sr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].set_title(f"{label} SR: {best_mk}", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        fig.colorbar(im0, ax=axes[row, 0], shrink=0.7)

        im1 = axes[row, 1].imshow(hr_64, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].set_title("HR reference", fontsize=9)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        fig.colorbar(im1, ax=axes[row, 1], shrink=0.7)

        im2 = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max(), interpolation="nearest")
        axes[row, 2].set_title(f"|SR-HR| max={diff.max():.4f}", fontsize=9)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.7)

    fig.suptitle(f"Best model: {best_mk} — {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "comparison_best.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_best.png (best: {best_mk})")

    # --- Figure 3: comparison_native_resolution.png ---
    bic_native = resize_np(lr_bicubic, native_size)
    sr_fin_native = resize_np(sr_fin, native_size)
    sr_den_native = resize_np(sr_den, native_size)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    im0 = axes[0].imshow(bic_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Bicubic ({native_size}x{native_size})", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(sr_den_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"{best_mk} denoised ({native_size}x{native_size})", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(sr_fin_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"{best_mk} final ({native_size}x{native_size})", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    im3 = axes[3].imshow(hr_native, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"Native HR ({native_size}x{native_size})", fontsize=9)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im3, ax=axes[3], shrink=0.7)

    fig.suptitle(f"Native-resolution comparison — {meta['dataset']}", fontsize=11)
    fig.savefig(str(out / "comparison_native_resolution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_native_resolution.png")

    # --- Figure 4: orientation search bar chart ---
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
    print("Generating v3 comparison plots...")
    for name, out_dir in datasets:
        print(f"\n{name}:")
        plot_dataset(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
