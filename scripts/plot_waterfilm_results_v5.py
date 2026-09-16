#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot v5: variants comparison figure. Does NOT overwrite v4 plots."""

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


def plot_dataset(out_dir, cmap="turbo"):
    out = Path(out_dir)
    npz_f = out / "sr_results_v5.npz"
    meta_f = out / "inference_meta_v5.json"
    if not npz_f.is_file() or not meta_f.is_file():
        print(f"  SKIP: {out_dir}")
        return

    d = np.load(str(npz_f))
    with open(str(meta_f), "r", encoding="utf-8") as f:
        meta = json.load(f)

    hr_64 = d["hr_bicubic_64"]
    hr_native = d["hr_native"]
    native_size = hr_native.shape[0]

    # Best model
    models_info = meta.get("models", {})
    best_mk = max(models_info.keys(), key=lambda k: models_info[k]["final_native_psnr"])

    # Variants for best model
    variants = {}
    for v in ["raw", "dc", "dc_denoised"]:
        key = f"{best_mk}_{v}_sr_best"
        if key in d:
            variants[v] = d[key]
    if f"{best_mk}_final_sr" in d:
        variants["final"] = d[f"{best_mk}_final_sr"]

    # --- comparison_variants_v5.png ---
    labels_map = {
        "raw": "Raw SR",
        "dc": "Data Consistency",
        "dc_denoised": "DC + Denoise(σ=0.5)",
        "final": "Final (best variant)",
    }
    order = ["raw", "dc", "dc_denoised", "final"]
    panels = []
    for v in order:
        if v in variants:
            p = psnr_native(variants[v], hr_native)
            panels.append((variants[v], f"{labels_map[v]}\nPSNR_n={p:.2f}"))
    panels.append((hr_64, "HR reference"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Signal")
    fig.suptitle(f"Post-processing variants: {best_mk} — {meta['dataset']}", fontsize=10)
    fig.savefig(str(out / "comparison_variants_v5.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_variants_v5.png (best model: {best_mk})")

    # --- Native resolution variants ---
    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 3.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (arr, title) in zip(axes, panels):
        nat_arr = resize_np(arr, native_size)
        im = ax.imshow(nat_arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title.replace("\n", " "), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"Native-resolution variants ({native_size}x{native_size}) — {meta['dataset']}", fontsize=10)
    fig.savefig(str(out / "comparison_variants_native_v5.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: comparison_variants_native_v5.png")


def main():
    datasets = [
        ("ygh", r"D:\LMC\data\phantom_ygh\最终处理结果\推理"),
        ("user", r"D:\LMC\data\水膜数据处理\最终处理结果\推理"),
    ]
    print("Generating v5 variant plots...")
    for n, od in datasets:
        print(f"\n{n}:")
        plot_dataset(od)
    print("\nDone.")


if __name__ == "__main__":
    main()
