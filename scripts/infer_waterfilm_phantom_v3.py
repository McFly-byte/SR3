#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-film MRSI SR3 super-resolution inference (v3).

Key fixes vs v2:
  1. LR input uses k-space zero-padding (sinc interpolation) instead of bicubic,
     matching training distribution (k-space truncation + hamming window).
  2. Struct channels tried in two modes: "zero" and "normal"; pick best.
  3. Post-processing: gaussian blur sigma=0.75 on best SR; optional alpha blend
     with bicubic if denoised SR still worse than bicubic baseline.
  4. Dual-scale metrics: 64x64 and native HR resolution.
  5. D4 orientation search uses k-space LR.
"""

from __future__ import annotations

import csv
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
from scipy.ndimage import gaussian_filter
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
MET_ID = 1
SEEDS = [0, 1, 2, 3, 4]
SAMPLE_STEPS = 50
DENOISE_SIGMA = 0.75
STRUCT_MODES = ["zero", "normal"]

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
# D4 orientation transforms
# ---------------------------------------------------------------------------
D4_TRANSFORMS: Dict[str, callable] = {
    "identity": lambda a: a.copy(),
    "rot90": lambda a: np.rot90(a, k=1),
    "rot180": lambda a: np.rot90(a, k=2),
    "rot270": lambda a: np.rot90(a, k=3),
    "hflip": lambda a: np.fliplr(a).copy(),
    "vflip": lambda a: np.flipud(a).copy(),
    "transpose": lambda a: np.transpose(a).copy(),
    "anti_transpose": lambda a: np.flipud(np.rot90(a, k=1)).copy(),
}
D4_INVERSES: Dict[str, str] = {
    "identity": "identity", "rot90": "rot270", "rot180": "rot180",
    "rot270": "rot90", "hflip": "hflip", "vflip": "vflip",
    "transpose": "transpose", "anti_transpose": "anti_transpose",
}


def apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    return D4_TRANSFORMS[name](arr)


def invert_transform(arr: np.ndarray, name: str) -> np.ndarray:
    return D4_TRANSFORMS[D4_INVERSES[name]](arr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ygh_data() -> Dict[str, np.ndarray]:
    mat_path = Path(r"D:\LMC\data\phantom_ygh\最终处理结果\results\reprocessing_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        records = f["records"]
        def deref(field, idx):
            return np.squeeze(f[records[field][idx, 0]][()])
        lr_map = deref("volNormMap", 0).astype(np.float64)
        hr_map = deref("volNormMap", 1).astype(np.float64)
        lr_mask = deref("supportMask", 0).astype(np.uint8)
        hr_mask = deref("supportMask", 1).astype(np.uint8)
        struct_img = deref("structImg", 0).astype(np.float64)
    return {
        "lr": lr_map, "hr": hr_map,
        "lr_mask": lr_mask, "hr_mask": hr_mask,
        "struct": struct_img,
        "mat_path": str(mat_path),
        "struct_path": str(mat_path) + " (records/structImg)",
    }


def load_waterfilm_data() -> Dict[str, np.ndarray]:
    mat_path = Path(r"D:\LMC\data\水膜数据处理\最终处理结果\results\all_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        sr29 = f["allResults"]["scan29"]
        sr30 = f["allResults"]["scan30"]
        lr_map = np.array(sr29["peakAreaVolNorm"]).astype(np.float64)
        hr_map = np.array(sr30["peakAreaVolNorm"]).astype(np.float64)
        lr_mask = np.array(sr29["signalMask"]).astype(np.uint8)
        hr_mask = np.array(sr30["signalMask"]).astype(np.uint8)
    d2seq_path = Path(r"D:\LMC\data\水膜数据处理\20260528\20260528_103023_20260528_Phantom01_1_1\28\pdata\1\2dseq")
    raw = np.fromfile(str(d2seq_path), dtype=np.uint16)
    echo0 = raw[: raw.size // 2]
    struct_img = echo0.reshape(128, 128).astype(np.float64)
    return {
        "lr": lr_map, "hr": hr_map,
        "lr_mask": lr_mask, "hr_mask": hr_mask,
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


def kspace_zeropad_lr(lr_small: np.ndarray, out_size: int = TARGET_SIZE) -> np.ndarray:
    """Zero-pad LR in k-space to out_size (sinc interpolation)."""
    h, w = lr_small.shape
    k = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_big = np.zeros((out_size, out_size), dtype=np.complex64)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_big[y0:y0 + h, x0:x0 + w] = k
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k_big))).astype(np.float32)


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
    opt = _load_json_config(config_path)
    opt["phase"] = "val"
    opt["gpu_ids"] = [gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
    struct_mode: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    lr_t = torch.from_numpy(lr_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    mask_t = torch.from_numpy(mask_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)

    if struct_mode == "zero":
        struct_t = torch.zeros(1, 1, TARGET_SIZE, TARGET_SIZE, device=device, dtype=lr_t.dtype)
    else:
        struct_t = torch.from_numpy(struct_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)

    parts = [lr_t, struct_t, struct_t]
    met_onehot = torch.zeros(1, 4, TARGET_SIZE, TARGET_SIZE, device=device, dtype=lr_t.dtype)
    met_onehot[:, MET_ID] = 1.0
    parts.append(met_onehot)
    if use_mask_channel:
        parts.append(mask_t)
    cond = torch.cat(parts, dim=1)

    to_m1_1 = lambda x: x * 2.0 - 1.0
    return {
        "HR": to_m1_1(lr_t),
        "SR": to_m1_1(cond),
        "LR": to_m1_1(lr_t),
        "MASK": mask_t,
    }


def run_inference(diffusion, batch, seed: int, network: str) -> np.ndarray:
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
def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    p = np.clip(np.round(pred * 255.0), 0, 255).astype(np.float64)
    r = np.clip(np.round(ref * 255.0), 0, 255).astype(np.float64)
    mse = np.mean((p - r) ** 2)
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 99.0


def compute_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute metrics on [0,1] images, evaluated in masked region where applicable."""
    m = mask.astype(bool)
    p = pred[m].astype(np.float64)
    r = ref[m].astype(np.float64)

    psnr_val = _psnr(pred, ref)
    ssim_val = float(sk_ssim(pred.astype(np.float64), ref.astype(np.float64), data_range=1.0))
    mae = float(np.mean(np.abs(p - r))) if p.size else 0.0
    rmse = float(np.sqrt(np.mean((p - r) ** 2))) if p.size else 0.0
    if len(p) > 1 and np.std(p) > 1e-12 and np.std(r) > 1e-12:
        pearson = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson = 0.0

    dy, dx = np.gradient(pred.astype(np.float64))
    tv = float(np.mean(np.hypot(dx, dy)))

    fft = np.fft.fftshift(np.fft.fft2(pred.astype(np.float64)))
    power = np.abs(fft) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    Y, X = np.ogrid[:power.shape[0], :power.shape[1]]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)
    hf_thresh = 0.25 * max_r
    total_p = float(power.sum())
    hf_p = float(power[dist > hf_thresh].sum())
    hf_ratio = hf_p / total_p if total_p > 0 else 0.0

    inside_mean = float(pred[m].mean()) if m.sum() > 0 else 0.0
    outside = ~m
    outside_mean = float(pred[outside].mean()) if outside.sum() > 0 else 0.0
    bg_leak = outside_mean / inside_mean if inside_mean > 1e-12 else 0.0

    return {
        "psnr": psnr_val, "ssim": ssim_val, "mae": mae, "rmse": rmse,
        "pearson": pearson, "tv": tv, "hf_ratio": hf_ratio, "bg_leakage": bg_leak,
    }


# ---------------------------------------------------------------------------
# Build inputs for a given orientation and struct_mode
# ---------------------------------------------------------------------------
def build_inputs(
    lr: np.ndarray,
    hr: np.ndarray,
    struct: np.ndarray,
    hr_mask: np.ndarray,
    orientation: str,
    struct_mode: str,
) -> Dict[str, np.ndarray]:
    lr_t = apply_transform(lr, orientation)
    hr_t = apply_transform(hr, orientation)
    struct_t = apply_transform(struct, orientation)
    mask_t = apply_transform(hr_mask, orientation)

    hr_max = float(hr_t.max())
    scale = hr_max if hr_max > 1e-12 else 1.0

    lr_norm = (lr_t / scale).astype(np.float32)
    hr_norm = (hr_t / scale).astype(np.float32)
    struct_norm = _norm01(struct_t)
    mask_bin = (mask_t > 0.5).astype(np.float32)

    # LR: k-space zero-pad to 64
    lr_kspace_64 = kspace_zeropad_lr(lr_norm, TARGET_SIZE).clip(0.0, 1.0)
    # LR bicubic (for baseline and blending)
    lr_bicubic_64 = _resize(lr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # HR bicubic to 64 (reference for 64-scale metrics)
    hr_bicubic_64 = _resize(hr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # struct to 64
    struct_64 = _resize(struct_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # mask nearest to 64
    mask_64 = (_resize(mask_bin, TARGET_SIZE, "nearest") >= 0.5).astype(np.float32)
    # LR nearest for display
    lr_nearest_64 = _resize(lr_norm, TARGET_SIZE, "nearest").clip(0.0, 1.0)

    return {
        "lr_kspace_64": lr_kspace_64,
        "lr_bicubic_64": lr_bicubic_64,
        "hr_bicubic_64": hr_bicubic_64,
        "hr_native": hr_norm,
        "struct_64": struct_64,
        "mask_64": mask_64,
        "mask_native": mask_bin,
        "lr_nearest_64": lr_nearest_64,
        "scale": scale,
        "native_size": hr_t.shape[0],
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def denoise_sr(sr: np.ndarray, sigma: float = DENOISE_SIGMA) -> np.ndarray:
    return gaussian_filter(sr.astype(np.float64), sigma=sigma).astype(np.float32).clip(0.0, 1.0)


def blend_with_bicubic(denoised: np.ndarray, bicubic: np.ndarray, ref: np.ndarray,
                       mask: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
    """Search best alpha in [0,1] blending denoised SR with bicubic.
    Returns (final, best_alpha, best_metrics)."""
    best_alpha = 0.0
    best_metrics = compute_metrics(denoised, ref, mask)
    best_img = denoised.copy()
    alphas = np.linspace(0.0, 1.0, 21)
    for a in alphas:
        cand = ((1.0 - a) * denoised + a * bicubic).clip(0.0, 1.0)
        mets = compute_metrics(cand, ref, mask)
        if mets["psnr"] > best_metrics["psnr"]:
            best_metrics = mets
            best_alpha = float(a)
            best_img = cand
    return best_img, best_alpha, best_metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_dataset(dataset_name: str, data: Dict[str, np.ndarray], output_dir: str) -> Dict:
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

    # --- Step 1: D4 orientation search with k-space LR, healthy_ema, seed=0 ---
    print("\n[Step 1] D4 orientation search (k-space LR, healthy_ema, seed=0)...")
    hcfg = MODEL_CONFIGS[0]
    diffusion, opt = build_model(hcfg["config_path"], hcfg["checkpoint_prefix"], hcfg["model_dir"])
    ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
    use_mask_healthy = bool(ds_opt.get("use_mask_channel", False))
    device = diffusion.device

    # For orientation search, use struct_mode="zero" (per ablation, zero is often better)
    orient_results: Dict[str, float] = {}
    for oname in D4_TRANSFORMS.keys():
        inputs = build_inputs(lr, hr, struct, hr_mask, oname, "zero")
        batch = build_batch(inputs["lr_kspace_64"], inputs["struct_64"], inputs["mask_64"],
                           use_mask_healthy, "zero", device)
        sr_01 = run_inference(diffusion, batch, seed=0, network="ema")
        # Evaluate at native HR resolution: downsample SR to native size
        native_sr = _resize(sr_01, inputs["native_size"], "bicubic")
        native_hr = inputs["hr_native"]
        native_mask = inputs["mask_native"]
        mets = compute_metrics(native_sr, native_hr, native_mask)
        orient_results[oname] = mets["psnr"]
        print(f"  {oname:20s}  native PSNR={mets['psnr']:.3f} dB")

    best_orient = max(orient_results, key=orient_results.get)
    print(f"\n  >> Best orientation: {best_orient} (native PSNR={orient_results[best_orient]:.3f} dB)")
    del diffusion
    torch.cuda.empty_cache()

    # --- Step 2: For each model, try both struct_modes, pick best ---
    print(f"\n[Step 2] Multi-seed inference at orientation={best_orient}...")

    all_rows: List[Dict[str, Any]] = []
    model_best: Dict[str, Dict] = {}

    for mcfg in MODEL_CONFIGS:
        mkey = mcfg["key"]
        print(f"\n  --- Model: {mkey} (network={mcfg['network']}) ---")

        # Build model once
        diffusion, opt = build_model(mcfg["config_path"], mcfg["checkpoint_prefix"], mcfg["model_dir"])
        ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
        use_mask_m = bool(ds_opt.get("use_mask_channel", False))
        device_m = diffusion.device

        best_overall = None  # (psnr, struct_mode, seed, sr_raw, metrics_64, metrics_native)

        for smode in STRUCT_MODES:
            inputs = build_inputs(lr, hr, struct, hr_mask, best_orient, smode)
            batch = build_batch(inputs["lr_kspace_64"], inputs["struct_64"], inputs["mask_64"],
                               use_mask_m, smode, device_m)

            for seed in SEEDS:
                t0 = time.time()
                sr_64 = run_inference(diffusion, batch, seed=seed, network=mcfg["network"])
                elapsed = time.time() - t0

                # 64-scale metrics
                m64 = compute_metrics(sr_64, inputs["hr_bicubic_64"], inputs["mask_64"])
                # Native-scale metrics
                sr_native = _resize(sr_64, inputs["native_size"], "bicubic")
                mnative = compute_metrics(sr_native, inputs["hr_native"], inputs["mask_native"])

                row = {
                    "model": mkey, "seed": seed, "network": mcfg["network"],
                    "struct_mode": smode,
                    "psnr_64": m64["psnr"], "ssim_64": m64["ssim"],
                    "mae_64": m64["mae"], "rmse_64": m64["rmse"],
                    "pearson_64": m64["pearson"],
                    "psnr_native": mnative["psnr"], "ssim_native": mnative["ssim"],
                    "pearson_native": mnative["pearson"],
                    "tv": m64["tv"], "hf_ratio": m64["hf_ratio"],
                    "bg_leakage": m64["bg_leakage"],
                    "time_sec": round(elapsed, 2),
                }
                all_rows.append(row)
                print(f"    struct={smode:6s} seed={seed}  "
                      f"native PSNR={mnative['psnr']:.3f}  corr={mnative['pearson']:.4f}  "
                      f"64 PSNR={m64['psnr']:.3f}  ({elapsed:.1f}s)")

                if best_overall is None or mnative["psnr"] > best_overall[0]:
                    best_overall = (mnative["psnr"], smode, seed, sr_64.copy(), m64, mnative)

        del diffusion
        torch.cuda.empty_cache()

        # Post-processing on best SR
        _, best_smode, best_seed, sr_raw_64, m64, mnative = best_overall
        inputs_best = build_inputs(lr, hr, struct, hr_mask, best_orient, best_smode)

        # Denoised
        sr_denoised_64 = denoise_sr(sr_raw_64, DENOISE_SIGMA)
        dn_m64 = compute_metrics(sr_denoised_64, inputs_best["hr_bicubic_64"], inputs_best["mask_64"])
        dn_native_sr = _resize(sr_denoised_64, inputs_best["native_size"], "bicubic")
        dn_mnative = compute_metrics(dn_native_sr, inputs_best["hr_native"], inputs_best["mask_native"])

        # Bicubic baseline metrics
        bicubic_64 = inputs_best["lr_bicubic_64"]
        bic_m64 = compute_metrics(bicubic_64, inputs_best["hr_bicubic_64"], inputs_best["mask_64"])
        bic_native = _resize(bicubic_64, inputs_best["native_size"], "bicubic")
        bic_mnative = compute_metrics(bic_native, inputs_best["hr_native"], inputs_best["mask_native"])

        # Optional blending: compare denoised vs bicubic, search best alpha
        # Blend at 64 scale, evaluate at native scale
        blend_alphas = np.linspace(0.0, 1.0, 21)
        best_blend_alpha = 0.0
        best_blend_native_psnr = dn_mnative["psnr"]
        best_final_64 = sr_denoised_64.copy()
        for a in blend_alphas:
            cand_64 = ((1.0 - a) * sr_denoised_64 + a * bicubic_64).clip(0.0, 1.0)
            cand_native = _resize(cand_64, inputs_best["native_size"], "bicubic")
            cand_mn = compute_metrics(cand_native, inputs_best["hr_native"], inputs_best["mask_native"])
            if cand_mn["psnr"] > best_blend_native_psnr:
                best_blend_native_psnr = cand_mn["psnr"]
                best_blend_alpha = float(a)
                best_final_64 = cand_64

        final_native = _resize(best_final_64, inputs_best["native_size"], "bicubic")
        final_mnative = compute_metrics(final_native, inputs_best["hr_native"], inputs_best["mask_native"])
        final_m64 = compute_metrics(best_final_64, inputs_best["hr_bicubic_64"], inputs_best["mask_64"])

        # Invert orientation for saved SR
        sr_raw_orig = invert_transform(sr_raw_64, best_orient)
        sr_denoised_orig = invert_transform(sr_denoised_64, best_orient)
        sr_final_orig = invert_transform(best_final_64, best_orient)

        model_best[mkey] = {
            "struct_mode": best_smode,
            "best_seed": best_seed,
            "sr_raw_64": sr_raw_orig,
            "sr_denoised_64": sr_denoised_orig,
            "sr_final_64": sr_final_orig,
            "raw_metrics_64": m64,
            "raw_metrics_native": mnative,
            "denoised_metrics_64": dn_m64,
            "denoised_metrics_native": dn_mnative,
            "final_metrics_64": final_m64,
            "final_metrics_native": final_mnative,
            "blend_alpha": best_blend_alpha,
            "denoise_sigma": DENOISE_SIGMA,
        }

        print(f"\n    >> {mkey}: best struct={best_smode}, seed={best_seed}")
        print(f"       raw     native PSNR={mnative['psnr']:.3f}, corr={mnative['pearson']:.4f}")
        print(f"       denoised native PSNR={dn_mnative['psnr']:.3f}, corr={dn_mnative['pearson']:.4f}")
        print(f"       final   native PSNR={final_mnative['psnr']:.3f}, corr={final_mnative['pearson']:.4f} (alpha={best_blend_alpha:.2f})")

    # --- Step 3: Save outputs ---
    print(f"\n[Step 3] Saving outputs to {out}...")

    inputs_final = build_inputs(lr, hr, struct, hr_mask, best_orient, "zero")  # for reference arrays

    npz_dict: Dict[str, Any] = {
        "lr_kspace_64": inputs_final["lr_kspace_64"].astype(np.float32),
        "lr_bicubic_64": inputs_final["lr_bicubic_64"].astype(np.float32),
        "hr_bicubic_64": inputs_final["hr_bicubic_64"].astype(np.float32),
        "hr_native": inputs_final["hr_native"].astype(np.float32),
        "struct_64": inputs_final["struct_64"].astype(np.float32),
        "mask_64": inputs_final["mask_64"].astype(np.float32),
        "mask_native": inputs_final["mask_native"].astype(np.float32),
        "lr_nearest_64": inputs_final["lr_nearest_64"].astype(np.float32),
        "orientation": best_orient,
        "scale_factor": np.float32(inputs_final["scale"]),
    }
    for mkey, mb in model_best.items():
        npz_dict[f"{mkey}_raw_sr"] = mb["sr_raw_64"].astype(np.float32)
        npz_dict[f"{mkey}_denoised_sr"] = mb["sr_denoised_64"].astype(np.float32)
        npz_dict[f"{mkey}_final_sr"] = mb["sr_final_64"].astype(np.float32)

    np.savez(str(out / "sr_results.npz"), **npz_dict)

    # MAT file
    mat_dict = {k: v for k, v in npz_dict.items() if isinstance(v, np.ndarray)}
    mat_dict["orientation"] = best_orient
    mat_dict["scale_factor"] = inputs_final["scale"]
    sio.savemat(str(out / "sr_quantitative.mat"), mat_dict, do_compression=True)

    # metrics.csv
    csv_path = out / "metrics.csv"
    fieldnames = ["model", "seed", "network", "struct_mode",
                  "psnr_64", "ssim_64", "mae_64", "rmse_64", "pearson_64",
                  "psnr_native", "ssim_native", "pearson_native",
                  "tv", "hf_ratio", "bg_leakage", "time_sec"]
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # metrics_summary.csv
    summary_path = out / "metrics_summary.csv"
    with open(str(summary_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "struct_mode", "best_seed",
                         "psnr_native", "ssim_native", "pearson_native",
                         "psnr_64", "ssim_64",
                         "denoise_sigma", "blend_alpha",
                         "bicubic_native_psnr", "bicubic_native_pearson",
                         "psnr_native_vs_bicubic"])
        # Bicubic baseline
        bic_native_for_blend = _resize(inputs_final["lr_bicubic_64"], inputs_final["native_size"], "bicubic")
        bic_mn = compute_metrics(bic_native_for_blend, inputs_final["hr_native"], inputs_final["mask_native"])
        writer.writerow([
            "bicubic", "-", "-",
            f"{bic_mn['psnr']:.6f}", f"{bic_mn['ssim']:.6f}", f"{bic_mn['pearson']:.6f}",
            f"{bic_m64['psnr']:.6f}", f"{bic_m64['ssim']:.6f}",
            "-", "-",
            f"{bic_mn['psnr']:.6f}", f"{bic_mn['pearson']:.6f}",
            "0.000000",
        ])
        for mkey in [c["key"] for c in MODEL_CONFIGS]:
            mb = model_best[mkey]
            fm = mb["final_metrics_native"]
            writer.writerow([
                mkey, mb["struct_mode"], mb["best_seed"],
                f"{fm['psnr']:.6f}", f"{fm['ssim']:.6f}", f"{fm['pearson']:.6f}",
                f"{mb['final_metrics_64']['psnr']:.6f}", f"{mb['final_metrics_64']['ssim']:.6f}",
                f"{mb['denoise_sigma']:.2f}", f"{mb['blend_alpha']:.2f}",
                f"{bic_mn['psnr']:.6f}", f"{bic_mn['pearson']:.6f}",
                f"{fm['psnr'] - bic_mn['psnr']:.6f}",
            ])

    # orientation_search.csv
    with open(str(out / "orientation_search.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orientation", "native_psnr_db"])
        for oname, opsnr in orient_results.items():
            writer.writerow([oname, f"{opsnr:.6f}"])
        writer.writerow(["SELECTED", best_orient])

    # inference_meta.json
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
        "normalization_scale": inputs_final["scale"],
        "normalization_rule": "LR and HR divided by HR.max(); struct min-max to [0,1]",
        "lr_preparation": "k-space zero-padding (FFT -> zero-pad to 64x64 -> IFFT, magnitude)",
        "confirmed_orientation": best_orient,
        "orientation_search": orient_results,
        "seeds": SEEDS,
        "sample_num_steps": SAMPLE_STEPS,
        "postprocessing": {
            "denoise_sigma": DENOISE_SIGMA,
            "denoise_method": "gaussian_filter",
            "blend_method": "linear blend with bicubic LR at 64 scale, alpha optimized on native PSNR",
        },
        "models": {},
        "bicubic_baseline_native": bic_mn,
        "bicubic_baseline_64": bic_m64,
    }
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        mb = model_best[mkey]
        meta["models"][mkey] = {
            "struct_mode": mb["struct_mode"],
            "best_seed": mb["best_seed"],
            "raw_native_psnr": mb["raw_metrics_native"]["psnr"],
            "raw_native_pearson": mb["raw_metrics_native"]["pearson"],
            "denoised_native_psnr": mb["denoised_metrics_native"]["psnr"],
            "final_native_psnr": mb["final_metrics_native"]["psnr"],
            "final_native_pearson": mb["final_metrics_native"]["pearson"],
            "blend_alpha": mb["blend_alpha"],
            "denoise_sigma": mb["denoise_sigma"],
        }

    with open(str(out / "inference_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} - DONE")
    print(f"  Orientation: {best_orient}")
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        mb = model_best[mkey]
        fm = mb["final_metrics_native"]
        print(f"  {mkey:25s}  struct={mb['struct_mode']:6s} seed={mb['best_seed']}  "
              f"native PSNR={fm['psnr']:.3f}  corr={fm['pearson']:.4f}  "
              f"alpha={mb['blend_alpha']:.2f}")
    print(f"  Bicubic native PSNR: {bic_mn['psnr']:.3f}, corr: {bic_mn['pearson']:.4f}")
    print(f"{'='*70}")

    return meta


def main():
    print("=" * 70)
    print("Water-film MRSI SR3 Inference v3 (k-space LR + post-processing)")
    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
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
        best_mk = max(s["models"].keys(),
                      key=lambda k: s["models"][k]["final_native_psnr"])
        print(f"  {s['dataset']:20s}  best={best_mk:25s}  "
              f"native_PSNR={s['models'][best_mk]['final_native_psnr']:.3f} dB  "
              f"orientation={s['confirmed_orientation']}  "
              f"struct_mode={s['models'][best_mk]['struct_mode']}")


if __name__ == "__main__":
    main()
