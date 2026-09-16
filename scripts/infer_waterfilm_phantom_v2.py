#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-film MRSI SR3 super-resolution inference (v2).

Runs on two datasets (ygh phantom + user water-film) with three model configs:
  - healthy_ema  (network="ema", 8 channels, no mask)
  - healthy_raw  (network="raw", 8 channels, no mask)
  - current_mixed_raw (network="raw", 9 channels, with mask)

Per dataset:
  1. D4 orientation search with healthy_ema, seed=0, 50 steps
  2. 3 models x 5 seeds (0..4), 50-step DDIM sampling at confirmed orientation
  3. Quantitative metrics (PSNR, SSIM, MAE, RMSE, Pearson, TV, HF energy, BG leakage)
  4. Saves npz / mat / csv / json  (NO matplotlib import here)

Plotting is done separately by plot_waterfilm_results.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

_REPO_ROOT = Path(r"D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement")
os.chdir(str(_REPO_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core.logger as Logger  # noqa: E402
import model as Model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE = 64
MET_ID = 1  # Glc: HDO=0, Glc=1, Glx=2, Lac=3
SEEDS = [0, 1, 2, 3, 4]
SAMPLE_STEPS = 50
MODEL_CONFIGS = [
    {
        "key": "healthy_ema",
        "config_path": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "config_resolved.json"),
        "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "checkpoint" / "I300000_E2522"),
        "model_dir": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema"),
        "network": "ema",
    },
    {
        "key": "healthy_raw",
        "config_path": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "config_resolved.json"),
        "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "checkpoint" / "I300000_E2522"),
        "model_dir": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema"),
        "network": "raw",
    },
    {
        "key": "current_mixed_raw",
        "config_path": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw" / "config_resolved.json"),
        "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw" / "checkpoint" / "I50500_E1931"),
        "model_dir": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw"),
        "network": "raw",
    },
]

# ---------------------------------------------------------------------------
# D4 orientation transforms (numpy)
# ---------------------------------------------------------------------------
D4_TRANSFORMS: Dict[str, callable] = {
    "identity": lambda a: a.copy(),
    "rot90": lambda a: np.rot90(a, k=1),
    "rot180": lambda a: np.rot90(a, k=2),
    "rot270": lambda a: np.rot90(a, k=3),
    "hflip": lambda a: np.fliplr(a).copy(),
    "vflip": lambda a: np.flipud(a).copy(),
    "transpose": lambda a: np.transpose(a).copy(),       # rot90 + hflip
    "anti_transpose": lambda a: np.flipud(np.rot90(a, k=1)).copy(),  # rot90 + vflip
}

D4_INVERSES: Dict[str, str] = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "hflip": "hflip",
    "vflip": "vflip",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    return D4_TRANSFORMS[name](arr)


def invert_transform(arr: np.ndarray, name: str) -> np.ndarray:
    inv_name = D4_INVERSES[name]
    return D4_TRANSFORMS[inv_name](arr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ygh_data() -> Dict[str, np.ndarray]:
    """Load ygh phantom data from reprocessing_results.mat (HDF5)."""
    mat_path = Path(r"D:\LMC\data\phantom_ygh\最终处理结果\results\reprocessing_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        records = f["records"]

        # Each field is (2,1) object array with h5py.Reference elements
        def deref_field(field_name: str, idx: int) -> np.ndarray:
            obj_arr = records[field_name]
            ref = obj_arr[idx, 0]
            data = f[ref][()]
            return np.squeeze(data)

        lr_map = deref_field("volNormMap", 0).astype(np.float64)   # (12,12)
        hr_map = deref_field("volNormMap", 1).astype(np.float64)   # (24,24)
        lr_mask = deref_field("supportMask", 0).astype(np.uint8)   # (12,12)
        hr_mask = deref_field("supportMask", 1).astype(np.uint8)   # (24,24)
        struct_img = deref_field("structImg", 0).astype(np.float64)  # (192,192)

    return {
        "lr": lr_map,
        "hr": hr_map,
        "lr_mask": lr_mask,
        "hr_mask": hr_mask,
        "struct": struct_img,
        "mat_path": str(mat_path),
        "struct_path": str(mat_path) + " (records/structImg)",
    }


def load_waterfilm_data() -> Dict[str, np.ndarray]:
    """Load user water-film data from all_results.mat (HDF5)."""
    mat_path = Path(r"D:\LMC\data\水膜数据处理\最终处理结果\results\all_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        all_results = f["allResults"]
        scan29 = all_results["scan29"]
        scan30 = all_results["scan30"]

        lr_map = np.array(scan29["peakAreaVolNorm"]).astype(np.float64)  # (8,8)
        hr_map = np.array(scan30["peakAreaVolNorm"]).astype(np.float64)  # (12,12)
        lr_mask = np.array(scan29["signalMask"]).astype(np.uint8)       # (8,8)
        hr_mask = np.array(scan30["signalMask"]).astype(np.uint8)       # (12,12)

    # Structural image from 2dseq: uint16, 32768 = 128*128*2, take first echo
    d2seq_path = Path(r"D:\LMC\data\水膜数据处理\20260528\20260528_103023_20260528_Phantom01_1_1\28\pdata\1\2dseq")
    raw = np.fromfile(str(d2seq_path), dtype=np.uint16)
    total = raw.size
    n_echo = 2
    echo_len = total // n_echo
    echo0 = raw[:echo_len]
    # reshape 128x128
    struct_img = echo0.reshape(128, 128).astype(np.float64)

    return {
        "lr": lr_map,
        "hr": hr_map,
        "lr_mask": lr_mask,
        "hr_mask": hr_mask,
        "struct": struct_img,
        "mat_path": str(mat_path),
        "struct_path": str(d2seq_path),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _resize(arr: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size), mode="nearest")
    elif mode == "bicubic":
        t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    else:
        raise ValueError(mode)
    return t.squeeze().numpy().astype(np.float32)


def build_inputs(
    lr: np.ndarray,
    hr: np.ndarray,
    struct: np.ndarray,
    hr_mask: np.ndarray,
    orientation: str,
) -> Dict[str, np.ndarray]:
    """Build all 64x64 inputs for a given orientation.

    Returns dict with lr_64, hr_64, struct_64, mask_64, bicubic_64, scale, lr_nearest_64.
    """
    # Apply D4 transform to all raw arrays
    lr_t = apply_transform(lr, orientation)
    hr_t = apply_transform(hr, orientation)
    struct_t = apply_transform(struct, orientation)
    mask_t = apply_transform(hr_mask, orientation)

    # Common scale from HR max
    hr_max = float(hr_t.max())
    scale = hr_max if hr_max > 1e-12 else 1.0

    lr_norm = (lr_t / scale).astype(np.float32)
    hr_norm = (hr_t / scale).astype(np.float32)
    struct_norm = _norm01(struct_t)
    mask_bin = (mask_t > 0.5).astype(np.float32)

    # Resize to 64
    lr_64 = _resize(lr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    hr_64 = _resize(hr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    struct_64 = _resize(struct_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    mask_64 = (_resize(mask_bin, TARGET_SIZE, "nearest") >= 0.5).astype(np.float32)
    bicubic_64 = lr_64.copy()  # bicubic upsampled LR is the baseline
    lr_nearest_64 = _resize(lr_norm, TARGET_SIZE, "nearest").clip(0.0, 1.0)

    return {
        "lr_64": lr_64,
        "hr_64": hr_64,
        "struct_64": struct_64,
        "mask_64": mask_64,
        "bicubic_64": bicubic_64,
        "lr_nearest_64": lr_nearest_64,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
def _load_json_config(path: str) -> OrderedDict:
    json_str = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def build_model(config_path: str, checkpoint_prefix: str, model_dir: str, gpu_id: int = 0):
    """Create and return a loaded diffusion model."""
    opt = _load_json_config(config_path)
    opt["phase"] = "val"
    opt["gpu_ids"] = [gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set paths to model dir
    opt["path"]["experiments_root"] = model_dir
    for key in ("log", "tb_logger", "results", "checkpoint"):
        if key in opt["path"]:
            opt["path"][key] = str(Path(model_dir) / opt["path"][key])
    opt["path"]["resume_state"] = checkpoint_prefix

    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["distributed"] = False
    opt["is_main_process"] = True
    opt["enable_wandb"] = False

    opt = Logger.dict_to_nonedict(opt)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    return diffusion, opt


def build_batch(
    lr_64: np.ndarray,
    struct_64: np.ndarray,
    mask_64: np.ndarray,
    use_mask_channel: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build the batch dict for model inference."""
    lr_t = torch.from_numpy(lr_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    struct_t = torch.from_numpy(struct_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    mask_t = torch.from_numpy(mask_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)

    # Condition: lr(1) + t1(1) + flair(1) + met_onehot(4) + mask(optional)
    parts = [lr_t, struct_t, struct_t]
    met_onehot = torch.zeros(1, 4, TARGET_SIZE, TARGET_SIZE, device=device, dtype=lr_t.dtype)
    met_onehot[:, MET_ID] = 1.0
    parts.append(met_onehot)
    if use_mask_channel:
        parts.append(mask_t)
    cond = torch.cat(parts, dim=1)

    def to_m1_1(x):
        return x * 2.0 - 1.0

    batch = {
        "HR": to_m1_1(lr_t),
        "SR": to_m1_1(cond),
        "LR": to_m1_1(lr_t),
        "MASK": mask_t,
    }
    return batch


def run_inference(
    diffusion,
    batch: Dict[str, torch.Tensor],
    seed: int,
    network: str,
) -> np.ndarray:
    """Run one inference pass, return SR in [0,1] as (64,64) float32."""
    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=int(seed), sample_num_steps=SAMPLE_STEPS, network=network)
    visuals = diffusion.get_current_visuals(need_LR=True)
    sr_tensor = visuals["SR"]
    if sr_tensor.dim() == 4 and sr_tensor.shape[0] > 1:
        sr_tensor = sr_tensor[-1:]
    sr = sr_tensor.detach().float().cpu().squeeze().clamp(-1, 1)
    sr = (sr + 1.0) * 0.5
    return sr.numpy().astype(np.float32).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute all metrics in the masked region. Both pred/ref in [0,1]."""
    m = mask.astype(bool)
    p = pred[m].astype(np.float64)
    r = ref[m].astype(np.float64)

    # PSNR (on full image, [0,255] scale like core.metrics)
    p_u8 = np.clip(np.round(pred * 255.0), 0, 255).astype(np.float64)
    r_u8 = np.clip(np.round(ref * 255.0), 0, 255).astype(np.float64)
    mse_full = np.mean((p_u8 - r_u8) ** 2)
    psnr_val = float(20 * np.log10(255.0 / np.sqrt(mse_full))) if mse_full > 0 else 99.0

    # SSIM (full image)
    ssim_val = float(sk_ssim(pred.astype(np.float64), ref.astype(np.float64),
                            data_range=1.0))

    # MAE, RMSE (masked)
    mae = float(np.mean(np.abs(p - r)))
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))

    # Pearson correlation (masked)
    if len(p) > 1 and np.std(p) > 1e-12 and np.std(r) > 1e-12:
        pearson = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson = 0.0

    # TV (total variation, full image)
    dy, dx = np.gradient(pred.astype(np.float64))
    tv = float(np.mean(np.hypot(dx, dy)))

    # High-frequency energy ratio via FFT
    fft = np.fft.fftshift(np.fft.fft2(pred.astype(np.float64)))
    power = np.abs(fft) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    Y, X = np.ogrid[:power.shape[0], :power.shape[1]]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)
    hf_threshold = 0.25 * max_r
    total_power = float(power.sum())
    hf_power = float(power[dist > hf_threshold].sum())
    hf_ratio = hf_power / total_power if total_power > 0 else 0.0

    # Background leakage: mean outside mask / mean inside mask
    inside_mean = float(pred[m].mean()) if m.sum() > 0 else 0.0
    outside = ~m
    outside_mean = float(pred[outside].mean()) if outside.sum() > 0 else 0.0
    bg_leak = outside_mean / inside_mean if inside_mean > 1e-12 else 0.0

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "tv": tv,
        "hf_ratio": hf_ratio,
        "bg_leakage": bg_leak,
    }


# ---------------------------------------------------------------------------
# Main inference pipeline for one dataset
# ---------------------------------------------------------------------------
def process_dataset(
    dataset_name: str,
    data: Dict[str, np.ndarray],
    output_dir: str,
) -> Dict[str, Any]:
    """Run full inference pipeline for one dataset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lr = data["lr"]
    hr = data["hr"]
    hr_mask = data["hr_mask"]
    struct = data["struct"]

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"  LR shape: {lr.shape}, HR shape: {hr.shape}, struct shape: {struct.shape}")
    print(f"  Output: {out}")
    print(f"{'='*70}")

    # --- Step 1: D4 orientation search ---
    print("\n[Step 1] D4 orientation search (healthy_ema, seed=0, 50 steps)...")
    # Build healthy_ema model once for orientation search
    hcfg = MODEL_CONFIGS[0]  # healthy_ema
    diffusion, opt = build_model(hcfg["config_path"], hcfg["checkpoint_prefix"], hcfg["model_dir"])
    ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
    use_mask = bool(ds_opt.get("use_mask_channel", False))
    device = diffusion.device

    orientation_results: Dict[str, float] = {}
    orientation_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for orient_name in D4_TRANSFORMS.keys():
        inputs = build_inputs(lr, hr, struct, hr_mask, orient_name)
        batch = build_batch(inputs["lr_64"], inputs["struct_64"], inputs["mask_64"],
                           use_mask, device)
        sr_01 = run_inference(diffusion, batch, seed=0, network="ema")
        m = compute_metrics(sr_01, inputs["hr_64"], inputs["mask_64"])
        orientation_results[orient_name] = m["psnr"]
        orientation_arrays[orient_name] = inputs
        print(f"  {orient_name:20s}  PSNR={m['psnr']:.3f} dB")

    # Select best orientation
    best_orient = max(orientation_results, key=orientation_results.get)
    print(f"\n  >> Best orientation: {best_orient} (PSNR={orientation_results[best_orient]:.3f} dB)")

    # Free orientation-search model
    del diffusion
    torch.cuda.empty_cache()

    # --- Step 2: Run all models x seeds at confirmed orientation ---
    print(f"\n[Step 2] Multi-seed inference at orientation={best_orient}...")
    inputs = orientation_arrays[best_orient]
    hr_64 = inputs["hr_64"]
    mask_64 = inputs["mask_64"]
    lr_64 = inputs["lr_64"]
    struct_64 = inputs["struct_64"]
    bicubic_64 = inputs["bicubic_64"]
    lr_nearest_64 = inputs["lr_nearest_64"]
    scale = inputs["scale"]

    # Bicubic baseline
    bicubic_metrics = compute_metrics(bicubic_64, hr_64, mask_64)
    print(f"  Bicubic baseline: PSNR={bicubic_metrics['psnr']:.3f} dB, SSIM={bicubic_metrics['ssim']:.4f}")

    all_rows: List[Dict[str, Any]] = []
    model_best_sr: Dict[str, np.ndarray] = {}
    model_best_info: Dict[str, Dict[str, Any]] = {}

    for mcfg in MODEL_CONFIGS:
        mkey = mcfg["key"]
        print(f"\n  Loading model: {mkey} (network={mcfg['network']})...")
        diffusion, opt = build_model(mcfg["config_path"], mcfg["checkpoint_prefix"], mcfg["model_dir"])
        ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
        use_mask_m = bool(ds_opt.get("use_mask_channel", False))
        device_m = diffusion.device

        batch = build_batch(lr_64, struct_64, mask_64, use_mask_m, device_m)

        best_seed = None
        best_psnr = -1e18
        best_sr = None

        for seed in SEEDS:
            t0 = time.time()
            sr_01 = run_inference(diffusion, batch, seed=seed, network=mcfg["network"])
            elapsed = time.time() - t0
            mets = compute_metrics(sr_01, hr_64, mask_64)
            row = {
                "model": mkey,
                "seed": seed,
                "network": mcfg["network"],
                **mets,
                "time_sec": round(elapsed, 2),
            }
            all_rows.append(row)
            print(f"    seed={seed}  PSNR={mets['psnr']:.3f}  SSIM={mets['ssim']:.4f}  "
                  f"MAE={mets['mae']:.5f}  RMSE={mets['rmse']:.5f}  ({elapsed:.1f}s)")

            if mets["psnr"] > best_psnr:
                best_psnr = mets["psnr"]
                best_seed = seed
                best_sr = sr_01.copy()

        # Invert orientation on best SR to get back to original orientation
        sr_original = invert_transform(best_sr, best_orient)
        model_best_sr[mkey] = sr_original
        model_best_info[mkey] = {
            "best_seed": best_seed,
            "best_psnr": best_psnr,
            "metrics": compute_metrics(best_sr, hr_64, mask_64),
        }

        del diffusion
        torch.cuda.empty_cache()

    # --- Step 3: Save outputs ---
    print(f"\n[Step 3] Saving outputs to {out}...")

    # 3a. sr_results.npz
    npz_dict: Dict[str, Any] = {
        "lr_64": lr_64.astype(np.float32),
        "hr_64": hr_64.astype(np.float32),
        "struct_64": struct_64.astype(np.float32),
        "mask_64": mask_64.astype(np.float32),
        "bicubic_64": bicubic_64.astype(np.float32),
        "lr_nearest_64": lr_nearest_64.astype(np.float32),
        "orientation": best_orient,
        "scale_factor": np.float32(scale),
    }
    for mkey, sr_arr in model_best_sr.items():
        npz_dict[f"{mkey}_sr"] = sr_arr.astype(np.float32)
    np.savez(str(out / "sr_results.npz"), **npz_dict)

    # 3b. sr_quantitative.mat (MATLAB v5)
    mat_dict = {k: v for k, v in npz_dict.items() if isinstance(v, np.ndarray)}
    mat_dict["orientation"] = best_orient
    mat_dict["scale_factor"] = scale
    sio.savemat(str(out / "sr_quantitative.mat"), mat_dict, do_compression=True)

    # 3c. metrics.csv
    import csv
    csv_path = out / "metrics.csv"
    fieldnames = ["model", "seed", "network", "psnr", "ssim", "mae", "rmse",
                  "pearson", "tv", "hf_ratio", "bg_leakage", "time_sec"]
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # 3d. metrics_summary.csv
    summary_path = out / "metrics_summary.csv"
    with open(str(summary_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "best_seed", "psnr", "ssim", "mae", "rmse",
                         "pearson", "tv", "hf_ratio", "bg_leakage",
                         "psnr_vs_bicubic", "ssim_vs_bicubic"])
        # Bicubic baseline row
        writer.writerow([
            "bicubic", "-",
            f"{bicubic_metrics['psnr']:.6f}", f"{bicubic_metrics['ssim']:.6f}",
            f"{bicubic_metrics['mae']:.6f}", f"{bicubic_metrics['rmse']:.6f}",
            f"{bicubic_metrics['pearson']:.6f}", f"{bicubic_metrics['tv']:.6f}",
            f"{bicubic_metrics['hf_ratio']:.6f}", f"{bicubic_metrics['bg_leakage']:.6f}",
            "0.000000", "0.000000",
        ])
        for mkey in [c["key"] for c in MODEL_CONFIGS]:
            info = model_best_info[mkey]
            m = info["metrics"]
            writer.writerow([
                mkey, info["best_seed"],
                f"{m['psnr']:.6f}", f"{m['ssim']:.6f}",
                f"{m['mae']:.6f}", f"{m['rmse']:.6f}",
                f"{m['pearson']:.6f}", f"{m['tv']:.6f}",
                f"{m['hf_ratio']:.6f}", f"{m['bg_leakage']:.6f}",
                f"{m['psnr'] - bicubic_metrics['psnr']:.6f}",
                f"{m['ssim'] - bicubic_metrics['ssim']:.6f}",
            ])

    # 3e. orientation_search.csv
    orient_path = out / "orientation_search.csv"
    with open(str(orient_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orientation", "psnr_db"])
        for oname, opsnr in orientation_results.items():
            writer.writerow([oname, f"{opsnr:.6f}"])
        writer.writerow(["SELECTED", best_orient])

    # 3f. inference_meta.json
    meta = {
        "dataset": dataset_name,
        "mat_path": data["mat_path"],
        "struct_path": data["struct_path"],
        "lr_shape_raw": list(lr.shape),
        "hr_shape_raw": list(hr.shape),
        "struct_shape_raw": list(struct.shape),
        "target_size": TARGET_SIZE,
        "metabolite": "Glc",
        "met_id": MET_ID,
        "normalization_scale": scale,
        "normalization_rule": "LR and HR divided by HR.max(); struct min-max to [0,1]",
        "confirmed_orientation": best_orient,
        "orientation_search": orientation_results,
        "seeds": SEEDS,
        "sample_num_steps": SAMPLE_STEPS,
        "models": {},
        "bicubic_baseline": bicubic_metrics,
        "output_files": {
            "npz": "sr_results.npz",
            "mat": "sr_quantitative.mat",
            "metrics_csv": "metrics.csv",
            "metrics_summary_csv": "metrics_summary.csv",
            "orientation_csv": "orientation_search.csv",
        },
    }
    for mcfg in MODEL_CONFIGS:
        mkey = mcfg["key"]
        info = model_best_info[mkey]
        meta["models"][mkey] = {
            "config_path": mcfg["config_path"],
            "checkpoint_prefix": mcfg["checkpoint_prefix"],
            "model_dir": mcfg["model_dir"],
            "network": mcfg["network"],
            "best_seed": info["best_seed"],
            "best_psnr": info["best_psnr"],
            "best_metrics": info["metrics"],
        }

    meta_path = out / "inference_meta.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} - DONE")
    print(f"  Best orientation: {best_orient}")
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        info = model_best_info[mkey]
        print(f"  {mkey:25s}  best_seed={info['best_seed']}  PSNR={info['best_psnr']:.3f} dB")
    print(f"  Bicubic baseline PSNR: {bicubic_metrics['psnr']:.3f} dB")
    print(f"{'='*70}")

    return meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Water-film MRSI SR3 Inference v2")
    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    datasets = [
        ("ygh_phantom", load_ygh_data(),
         r"D:\LMC\data\phantom_ygh\最终处理结果\推理"),
        ("waterfilm_user", load_waterfilm_data(),
         r"D:\LMC\data\水膜数据处理\最终处理结果\推理"),
    ]

    all_summaries = []
    for name, data, out_dir in datasets:
        try:
            summary = process_dataset(name, data, out_dir)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n!!! ERROR processing {name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL DATASETS COMPLETE")
    print("=" * 70)
    for s in all_summaries:
        best_model = max(s["models"].keys(),
                         key=lambda k: s["models"][k]["best_psnr"])
        print(f"  {s['dataset']:20s}  best_model={best_model:25s}  "
              f"best_PSNR={s['models'][best_model]['best_psnr']:.3f} dB  "
              f"orientation={s['confirmed_orientation']}  "
              f"struct={s['struct_path']}")


if __name__ == "__main__":
    main()
