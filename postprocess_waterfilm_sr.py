# -*- coding: utf-8 -*-
"""
水膜 SR3 后处理脚本（不泄漏参考 / reference-free）

输入（每个推理目录）:
  sr3_ema_seed0_ddim50.npy   — 原始纯模型输出（只读，不修改）
  lr_normalized_native.npy    — LR 归一化原生分辨率（用于确定降采样目标尺寸）
  bicubic_baseline_64.npy     — bicubic baseline（只读，用于对比）
  measured_reference_64.npy   — 较高分辨率实测参考（仅用于事后评价指标，不参与任何滤波参数）

输出（每个推理目录）:
  sr3_post_lowpass_lrgrid.npy   — 降到 LR 分辨率再 bicubic 升回
  sr3_post_denoise_robust.npy   — 中值 + 双边去噪（参数来自噪声估计 + LR 像素足迹）
  postprocess_comparison.png    — 对比图（共享色标，明确标注）
  postprocess_metadata.json      — 算法 / 参数 / 是否使用参考 / 原始 SR hash
  postprocess_metrics.json       — 事后评价指标（PSNR/SSIM，未参与调参）

约束:
  - 不修改原始 sr3_ema_seed0_ddim50.npy
  - 所有滤波参数来自 LR 尺寸 / 输出尺寸 / 图像自身噪声估计，不依赖参考
  - 不把 bicubic 与 SR 混合后称为模型结果
"""

import os
import json
import hashlib
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_bilateral
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "key": "phantom_ygh",
        "dir": r"D:\LMC\data\phantom_ygh\最终处理结果\推理",
        "title": "Yang water phantom: scan 17 (12x12) -> scan 18 reference",
        "lr_label": "LR scan17 (12x12)",
        "ref_label": "Reference scan18 (24x24)",
    },
    {
        "key": "waterfilm_double_chamber",
        "dir": r"D:\LMC\data\水膜数据处理\最终处理结果\推理",
        "title": "Double-chamber water phantom: scan 29 (8x8) -> scan 30 reference",
        "lr_label": "LR scan29 (8x8)",
        "ref_label": "Reference scan30 (12x12)",
    },
]

OUT_SIZE = 64  # 输出分辨率


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_noise_sigma(img):
    """
    鲁棒噪声估计：图像自身高通残差（原图 - sigma=1 高斯）的 MAD。
    不依赖参考。
    """
    hp = img.astype(np.float64) - ndimage.gaussian_filter(img.astype(np.float64), sigma=1.0)
    mad = np.median(np.abs(hp - np.median(hp)))
    return float(1.4826 * mad)


def lowpass_to_lr_grid(img, lr_size):
    """
    将 64x64 图像降到 lr_size x lr_size（面积平均），再 bicubic 升回 64x64。
    参数仅来自 LR 尺寸。
    """
    # 面积平均降采样
    h, w = img.shape
    block_h = h // lr_size
    block_w = w // lr_size
    # 裁剪到整数倍
    crop_h = block_h * lr_size
    crop_w = block_w * lr_size
    cropped = img[:crop_h, :crop_w]
    down = cropped.reshape(lr_size, block_h, lr_size, block_w).mean(axis=(1, 3))
    # bicubic 升采样
    from skimage.transform import resize
    up = resize(down, (OUT_SIZE, OUT_SIZE), order=3, mode="edge", anti_aliasing=False)
    return up.astype(np.float32)


def robust_denoise(img, lr_size):
    """
    中值滤波 + 双边滤波。
    - 中值 kernel = 3（固定小核，去除脉冲散斑）
    - 双边 sigma_spatial = 64 / lr_size（LR 像素在输出空间的足迹）
    - 双边 sigma_color = 2.0 * noise_sigma（噪声由图像自身高通残差 MAD 估计）
    所有参数不依赖参考。
    """
    # Step 1: 中值滤波
    med = ndimage.median_filter(img.astype(np.float64), size=3)

    # Step 2: 噪声估计（在中值滤波后估计，减少散斑对估计的污染）
    noise_sigma = estimate_noise_sigma(med)

    # Step 3: 双边滤波
    sigma_spatial = OUT_SIZE / lr_size
    sigma_color = 2.0 * noise_sigma
    # 防止 sigma_color 为 0 或极小
    sigma_color = max(sigma_color, 1e-4)

    denoised = denoise_bilateral(
        med,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        mode="edge",
        channel_axis=None,
    )
    return denoised.astype(np.float32), {
        "median_kernel": 3,
        "bilateral_sigma_spatial": float(sigma_spatial),
        "bilateral_sigma_color": float(sigma_color),
        "noise_sigma_estimated": float(noise_sigma),
        "noise_estimator": "MAD of high-pass residual (img - gaussian sigma=1), x1.4826",
    }


def compute_metrics(img, ref):
    """事后评价指标，仅用于报告，不参与调参。"""
    data_range = float(max(ref.max(), img.max()) - min(ref.min(), img.min()))
    data_range = max(data_range, 1e-8)
    p = psnr(ref, img, data_range=data_range)
    s = ssim(ref, img, data_range=data_range)
    return float(p), float(s)


def make_comparison_figure(ds, lr_up, bicubic, sr_orig, sr_lowpass, sr_denoise, ref, out_path):
    """
    对比图：LR(升采样) | Bicubic | 原始SR3 | SR+低通 | SR+去噪 | 参考
    共享色标 [0, vmax]，明确标注每个面板。
    """
    panels = [
        (lr_up, f"{ds['lr_label']}\nupsampled to 64x64"),
        (bicubic, "Bicubic baseline\n(LR -> 64x64)"),
        (sr_orig, "Original SR3\n(pure model output)"),
        (sr_lowpass, "SR3 + lowpass\n(downsample to LR grid)"),
        (sr_denoise, "SR3 + robust denoise\n(median + bilateral)"),
        (ref, f"{ds['ref_label']}\n(resized, eval only)"),
    ]

    vmax = max(p[0].max() for p in panels)
    vmin = 0.0

    fig, axes = plt.subplots(1, 6, figsize=(20, 4.2))
    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Phase", fontsize=8)
        ax.set_ylabel("Read", fontsize=8)
        ax.tick_params(labelsize=7)

    # 共享色标
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("normalized intensity [0,1]", fontsize=8)

    fig.suptitle(
        f"{ds['title']}\n"
        f"Post-processing is reference-free; reference shown for evaluation only. "
        f"Original SR3 is preserved unchanged.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_dataset(ds):
    d = ds["dir"]
    print(f"\n{'='*70}")
    print(f"Processing: {ds['key']}  ->  {d}")
    print(f"{'='*70}")

    # 加载输入
    sr_path = os.path.join(d, "sr3_ema_seed0_ddim50.npy")
    lr_path = os.path.join(d, "lr_normalized_native.npy")
    bicubic_path = os.path.join(d, "bicubic_baseline_64.npy")
    ref_path = os.path.join(d, "measured_reference_64.npy")

    sr_orig = np.load(sr_path)
    lr_native = np.load(lr_path)
    bicubic = np.load(bicubic_path)
    ref = np.load(ref_path)

    lr_size = lr_native.shape[0]
    assert lr_native.shape[0] == lr_native.shape[1], "LR must be square"
    assert sr_orig.shape == (OUT_SIZE, OUT_SIZE), f"SR must be {OUT_SIZE}x{OUT_SIZE}"

    print(f"  LR size: {lr_size}x{lr_size}")
    print(f"  SR shape: {sr_orig.shape}, dtype={sr_orig.dtype}")
    print(f"  SR stats: min={sr_orig.min():.4f} max={sr_orig.max():.4f} "
          f"mean={sr_orig.mean():.4f} std={sr_orig.std():.4f}")

    # 原始 SR hash（处理前）
    sr_hash_before = sha256_file(sr_path)
    print(f"  Original SR sha256: {sr_hash_before[:16]}...")

    # LR 升采样（用于对比图显示）
    from skimage.transform import resize
    lr_up = resize(lr_native, (OUT_SIZE, OUT_SIZE), order=3, mode="edge", anti_aliasing=False)

    # ---- 后处理 Variant 1: 低通到 LR 网格 ----
    print("\n  [Variant 1] Lowpass to LR grid...")
    sr_lowpass = lowpass_to_lr_grid(sr_orig, lr_size)
    print(f"    lowpass stats: min={sr_lowpass.min():.4f} max={sr_lowpass.max():.4f} "
          f"mean={sr_lowpass.mean():.4f} std={sr_lowpass.std():.4f}")

    # ---- 后处理 Variant 2: 稳健去噪 ----
    print("\n  [Variant 2] Robust denoise (median + bilateral)...")
    sr_denoise, denoise_params = robust_denoise(sr_orig, lr_size)
    print(f"    noise_sigma estimated: {denoise_params['noise_sigma_estimated']:.6f}")
    print(f"    bilateral sigma_spatial: {denoise_params['bilateral_sigma_spatial']:.4f}")
    print(f"    bilateral sigma_color: {denoise_params['bilateral_sigma_color']:.6f}")
    print(f"    denoise stats: min={sr_denoise.min():.4f} max={sr_denoise.max():.4f} "
          f"mean={sr_denoise.mean():.4f} std={sr_denoise.std():.4f}")

    # ---- 保存后处理数组 ----
    lowpass_path = os.path.join(d, "sr3_post_lowpass_lrgrid.npy")
    denoise_path = os.path.join(d, "sr3_post_denoise_robust.npy")
    np.save(lowpass_path, sr_lowpass)
    np.save(denoise_path, sr_denoise)
    print(f"\n  Saved: {lowpass_path}")
    print(f"  Saved: {denoise_path}")

    # ---- 验证原始 SR 未被修改 ----
    sr_hash_after = sha256_file(sr_path)
    sr_unchanged = (sr_hash_before == sr_hash_after)
    print(f"\n  Original SR hash after processing: {sr_hash_after[:16]}...")
    print(f"  Original SR unchanged: {sr_unchanged}")
    assert sr_unchanged, "Original SR file was modified!"

    # ---- 事后评价指标（不参与调参） ----
    print("\n  [Post-hoc metrics vs reference (evaluation only, not used for tuning)]:")
    metrics = {}
    for name, img in [("bicubic", bicubic), ("sr_original", sr_orig),
                       ("sr_lowpass_lrgrid", sr_lowpass), ("sr_denoise_robust", sr_denoise)]:
        p, s = compute_metrics(img, ref)
        metrics[name] = {"psnr_db": p, "ssim": s}
        print(f"    {name:24s}: PSNR={p:.4f} dB, SSIM={s:.6f}")

    # ---- 生成对比图 ----
    fig_path = os.path.join(d, "postprocess_comparison.png")
    make_comparison_figure(ds, lr_up, bicubic, sr_orig, sr_lowpass, sr_denoise, ref, fig_path)
    print(f"\n  Saved comparison figure: {fig_path}")

    # ---- 元数据 ----
    metadata = {
        "dataset_key": ds["key"],
        "dataset_dir": d,
        "postprocessing_reference_free": True,
        "reference_used_for_filter_parameters": False,
        "reference_used_for_metrics_only": True,
        "original_sr_file": "sr3_ema_seed0_ddim50.npy",
        "original_sr_sha256": sr_hash_before,
        "original_sr_unchanged": sr_unchanged,
        "lr_size": lr_size,
        "output_size": OUT_SIZE,
        "variants": {
            "lowpass_lrgrid": {
                "description": "Downsample SR to LR native resolution via area averaging, then bicubic upsample to 64x64. Preserves only frequency content supported by LR sampling grid.",
                "algorithm": "area-average downsample + bicubic (order=3) upsample",
                "parameters": {
                    "downsample_target": [lr_size, lr_size],
                    "upsample_target": [OUT_SIZE, OUT_SIZE],
                    "upsample_order": 3,
                },
                "output_file": "sr3_post_lowpass_lrgrid.npy",
                "parameter_source": "LR native size (known input), output size",
            },
            "denoise_robust": {
                "description": "Median filter (kernel=3) followed by bilateral filter. Removes high-frequency speckle while preserving broad edges.",
                "algorithm": "median_filter(size=3) -> denoise_bilateral",
                "parameters": denoise_params,
                "output_file": "sr3_post_denoise_robust.npy",
                "parameter_source": "median kernel fixed=3; sigma_spatial=64/LR_size; sigma_color=2*noise_sigma where noise_sigma from MAD of image's own high-pass residual (img - gaussian sigma=1)",
            },
        },
        "posthoc_metrics_vs_reference": metrics,
        "metrics_warning": "Reference is another measured scan (not ideal HR ground truth). Metrics are post-hoc evaluation only and did NOT influence any filter parameter selection.",
        "constraints": [
            "Original sr3_ema_seed0_ddim50.npy is preserved unchanged (hash verified).",
            "No filter parameter is derived from the reference image.",
            "Bicubic baseline is not mixed with SR output; post-processed arrays are SR-only transformations.",
            "Post-processed results are labeled as post-processing, not model output.",
        ],
    }
    meta_path = os.path.join(d, "postprocess_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {meta_path}")

    metrics_path = os.path.join(d, "postprocess_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Saved metrics: {metrics_path}")

    return metadata


if __name__ == "__main__":
    all_meta = {}
    for ds in DATASETS:
        meta = process_dataset(ds)
        all_meta[ds["key"]] = meta
    print("\n\nDONE. Both datasets processed.")
