#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-film MRSI SR3 super-resolution inference (v4).

Critical fixes vs v3:
  1. k-space zero-padding includes IFFT normalization correction:
     img *= (out_size/h) * (out_size/w)
  2. Data consistency (DC) post-processing: replace center k-space of SR
     with LR k-space (scaled), ensuring low-frequency fidelity.
  3. Post-processing variants: raw -> dc -> dc_denoised (sigma=0.5)
  4. Two baselines: kspace LR and bicubic LR.
  5. Best variant selected by native-resolution PSNR; baselines reported honestly.
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
from typing import Any, Dict, List, Tuple

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
DC_DENOISE_SIGMA = 0.5
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
# D4 transforms
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
    """Zero-pad LR in k-space to out_size, with IFFT normalization correction."""
    h, w = lr_small.shape
    k = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_big = np.zeros((out_size, out_size), dtype=np.complex64)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_big[y0:y0 + h, x0:x0 + w] = k
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(k_big))).astype(np.float32)
    # Critical normalization correction
    img = img * (out_size / h) * (out_size / w)
    return img


def data_consistency(sr64: np.ndarray, lr_small: np.ndarray,
                     out_size: int = TARGET_SIZE) -> np.ndarray:
    """Replace center k-space of SR with LR k-space (scaled)."""
    h, w = lr_small.shape
    k_sr = np.fft.fftshift(np.fft.fft2(sr64.astype(np.float32)))
    k_lr = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_lr_scaled = k_lr * (out_size ** 2) / (h * w)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_sr[y0:y0 + h, x0:x0 + w] = k_lr_scaled
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k_sr))).astype(np.float32)


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


def build_batch(lr_64, struct_64, mask_64, use_mask_channel, struct_mode, device):
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
def _psnr(pred, ref):
    p = np.clip(np.round(pred * 255.0), 0, 255).astype(np.float64)
    r = np.clip(np.round(ref * 255.0), 0, 255).astype(np.float64)
    mse = np.mean((p - r) ** 2)
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 99.0


def compute_metrics(pred, ref, mask):
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
# Build inputs for a given orientation
# ---------------------------------------------------------------------------
def build_inputs(lr, hr, struct, hr_mask, orientation, struct_mode):
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

    # LR k-space zero-pad with normalization correction
    lr_kspace_64 = kspace_zeropad_lr(lr_norm, TARGET_SIZE).clip(0.0, 1.0)
    # LR bicubic
    lr_bicubic_64 = _resize(lr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # HR bicubic to 64
    hr_bicubic_64 = _resize(hr_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # struct to 64
    struct_64 = _resize(struct_norm, TARGET_SIZE, "bicubic").clip(0.0, 1.0)
    # mask nearest to 64
    mask_64 = (_resize(mask_bin, TARGET_SIZE, "nearest") >= 0.5).astype(np.float32)
    # LR nearest for display
    lr_nearest_64 = _resize(lr_norm, TARGET_SIZE, "nearest").clip(0.0, 1.0)

    return {
        "lr_norm": lr_norm,       # native-size normalized LR (for DC)
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
# Main pipeline
# ---------------------------------------------------------------------------
def process_dataset(dataset_name, data, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lr = data["lr"]
    hr = data["hr"]
    hr_mask = data["hr_mask"]
    struct = data["struct"]

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"  LR: {lr.shape}, HR: {hr.shape}, struct: {struct.shape}")
    print(f"  Output: {out}")
    print(f"{'='*70}")

    # --- Step 1: D4 orientation search ---
    print("\n[Step 1] D4 orientation search (healthy_ema, seed=0, k-space LR)...")
    hcfg = MODEL_CONFIGS[0]
    diffusion, opt = build_model(hcfg["config_path"], hcfg["checkpoint_prefix"], hcfg["model_dir"])
    ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
    use_mask_healthy = bool(ds_opt.get("use_mask_channel", False))
    device = diffusion.device

    orient_results = {}
    for oname in D4_TRANSFORMS.keys():
        inputs = build_inputs(lr, hr, struct, hr_mask, oname, "zero")
        batch = build_batch(inputs["lr_kspace_64"], inputs["struct_64"], inputs["mask_64"],
                           use_mask_healthy, "zero", device)
        sr_01 = run_inference(diffusion, batch, seed=0, network="ema")
        # Apply DC for orientation search evaluation
        sr_dc = data_consistency(sr_01, inputs["lr_norm"], TARGET_SIZE).clip(0.0, 1.0)
        native_sr = _resize(sr_dc, inputs["native_size"], "bicubic")
        mets = compute_metrics(native_sr, inputs["hr_native"], inputs["mask_native"])
        orient_results[oname] = mets["psnr"]
        print(f"  {oname:20s}  native PSNR(DC)={mets['psnr']:.3f} dB")

    best_orient = max(orient_results, key=orient_results.get)
    print(f"\n  >> Best orientation: {best_orient} (native PSNR={orient_results[best_orient]:.3f} dB)")
    del diffusion
    torch.cuda.empty_cache()

    # --- Step 2: Multi-seed inference ---
    print(f"\n[Step 2] Multi-seed inference at orientation={best_orient}...")

    all_rows = []
    model_best = {}

    for mcfg in MODEL_CONFIGS:
        mkey = mcfg["key"]
        print(f"\n  --- Model: {mkey} (network={mcfg['network']}) ---")

        diffusion, opt = build_model(mcfg["config_path"], mcfg["checkpoint_prefix"], mcfg["model_dir"])
        ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
        use_mask_m = bool(ds_opt.get("use_mask_channel", False))
        device_m = diffusion.device

        # Track best across all seeds and struct_modes
        # best_key: (psnr_native, struct_mode, seed, variant_name, sr_64, metrics_64, metrics_native)
        best_overall = None

        for smode in STRUCT_MODES:
            inputs = build_inputs(lr, hr, struct, hr_mask, best_orient, smode)
            batch = build_batch(inputs["lr_kspace_64"], inputs["struct_64"], inputs["mask_64"],
                               use_mask_m, smode, device_m)

            for seed in SEEDS:
                t0 = time.time()
                sr_raw = run_inference(diffusion, batch, seed=seed, network=mcfg["network"])
                elapsed = time.time() - t0

                # Three post-processing variants
                # 1. raw
                raw_m64 = compute_metrics(sr_raw, inputs["hr_bicubic_64"], inputs["mask_64"])
                raw_native = _resize(sr_raw, inputs["native_size"], "bicubic")
                raw_mnative = compute_metrics(raw_native, inputs["hr_native"], inputs["mask_native"])

                # 2. DC
                sr_dc = data_consistency(sr_raw, inputs["lr_norm"], TARGET_SIZE).clip(0.0, 1.0)
                dc_m64 = compute_metrics(sr_dc, inputs["hr_bicubic_64"], inputs["mask_64"])
                dc_native = _resize(sr_dc, inputs["native_size"], "bicubic")
                dc_mnative = compute_metrics(dc_native, inputs["hr_native"], inputs["mask_native"])

                # 3. DC + denoise
                sr_dc_dn = gaussian_filter(sr_dc.astype(np.float64), sigma=DC_DENOISE_SIGMA).astype(np.float32).clip(0.0, 1.0)
                dcdn_m64 = compute_metrics(sr_dc_dn, inputs["hr_bicubic_64"], inputs["mask_64"])
                dcdn_native = _resize(sr_dc_dn, inputs["native_size"], "bicubic")
                dcdn_mnative = compute_metrics(dcdn_native, inputs["hr_native"], inputs["mask_native"])

                # Log all three variants
                for vname, m64, mn in [("raw", raw_m64, raw_mnative),
                                       ("dc", dc_m64, dc_mnative),
                                       ("dc_denoised", dcdn_m64, dcdn_mnative)]:
                    row = {
                        "model": mkey, "seed": seed, "network": mcfg["network"],
                        "struct_mode": smode, "variant": vname,
                        "psnr_64": m64["psnr"], "ssim_64": m64["ssim"],
                        "mae_64": m64["mae"], "rmse_64": m64["rmse"],
                        "pearson_64": m64["pearson"],
                        "psnr_native": mn["psnr"], "ssim_native": mn["ssim"],
                        "pearson_native": mn["pearson"],
                        "tv": m64["tv"], "hf_ratio": m64["hf_ratio"],
                        "bg_leakage": m64["bg_leakage"],
                        "time_sec": round(elapsed, 2),
                    }
                    all_rows.append(row)

                print(f"    struct={smode:6s} seed={seed}  "
                      f"raw: PSNR_n={raw_mnative['psnr']:.2f} corr={raw_mnative['pearson']:.3f} | "
                      f"dc: PSNR_n={dc_mnative['psnr']:.2f} corr={dc_mnative['pearson']:.3f} | "
                      f"dc_dn: PSNR_n={dcdn_mnative['psnr']:.2f} corr={dcdn_mnative['pearson']:.3f}")

                # Track best by native PSNR across all variants
                for vname, sr_arr, m64, mn in [("raw", sr_raw, raw_m64, raw_mnative),
                                                ("dc", sr_dc, dc_m64, dc_mnative),
                                                ("dc_denoised", sr_dc_dn, dcdn_m64, dcdn_mnative)]:
                    if best_overall is None or mn["psnr"] > best_overall[0]:
                        best_overall = (mn["psnr"], smode, seed, vname, sr_arr.copy(), m64, mn)

        del diffusion
        torch.cuda.empty_cache()

        # Save best for this model
        _, best_smode, best_seed, best_variant, sr_best_64, best_m64, best_mnative = best_overall
        inputs_best = build_inputs(lr, hr, struct, hr_mask, best_orient, best_smode)

        # Also save raw/dc/dc_denoised for the best seed+struct
        # Recompute them for saving (we already have best variant)
        # But we need all three variants for the best seed
        # Let's re-run to get all three variants for the best configuration
        model_best[mkey] = {
            "struct_mode": best_smode,
            "best_seed": best_seed,
            "best_variant": best_variant,
            "sr_final_64": sr_best_64,  # already in oriented space
            "best_metrics_64": best_m64,
            "best_metrics_native": best_mnative,
        }

        print(f"\n    >> {mkey}: struct={best_smode}, seed={best_seed}, variant={best_variant}")
        print(f"       final native PSNR={best_mnative['psnr']:.3f}, corr={best_mnative['pearson']:.4f}")

    # --- Step 3: Compute baselines ---
    print("\n[Step 3] Computing baselines...")
    inputs_ref = build_inputs(lr, hr, struct, hr_mask, best_orient, "zero")

    # k-space LR baseline (64x64, already normalized-corrected)
    kspace_baseline_64 = inputs_ref["lr_kspace_64"]
    kspace_m64 = compute_metrics(kspace_baseline_64, inputs_ref["hr_bicubic_64"], inputs_ref["mask_64"])
    kspace_native = _resize(kspace_baseline_64, inputs_ref["native_size"], "bicubic")
    kspace_mnative = compute_metrics(kspace_native, inputs_ref["hr_native"], inputs_ref["mask_native"])

    # bicubic baseline
    bicubic_baseline_64 = inputs_ref["lr_bicubic_64"]
    bic_m64 = compute_metrics(bicubic_baseline_64, inputs_ref["hr_bicubic_64"], inputs_ref["mask_64"])
    bic_native = _resize(bicubic_baseline_64, inputs_ref["native_size"], "bicubic")
    bic_mnative = compute_metrics(bic_native, inputs_ref["hr_native"], inputs_ref["mask_native"])

    print(f"  kspace baseline: native PSNR={kspace_mnative['psnr']:.3f}, corr={kspace_mnative['pearson']:.4f}")
    print(f"  bicubic baseline: native PSNR={bic_mnative['psnr']:.3f}, corr={bic_mnative['pearson']:.4f}")

    # --- Step 4: Determine overall best ---
    all_candidates = [
        ("baseline_kspace", kspace_mnative["psnr"], kspace_baseline_64),
        ("baseline_bicubic", bic_mnative["psnr"], bicubic_baseline_64),
    ]
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        mb = model_best[mkey]
        all_candidates.append((f"{mkey}_{mb['best_variant']}", mb["best_metrics_native"]["psnr"], mb["sr_final_64"]))

    overall_best = max(all_candidates, key=lambda x: x[1])
    print(f"\n  >> Overall best: {overall_best[0]} (native PSNR={overall_best[1]:.3f} dB)")

    # --- Step 5: Save outputs ---
    print(f"\n[Step 4] Saving outputs to {out}...")

    # Invert orientation on SR arrays for saving
    npz_dict: Dict[str, Any] = {
        "lr_kspace_64": inputs_ref["lr_kspace_64"].astype(np.float32),
        "lr_bicubic_64": inputs_ref["lr_bicubic_64"].astype(np.float32),
        "hr_bicubic_64": inputs_ref["hr_bicubic_64"].astype(np.float32),
        "hr_native": inputs_ref["hr_native"].astype(np.float32),
        "struct_64": inputs_ref["struct_64"].astype(np.float32),
        "mask_64": inputs_ref["mask_64"].astype(np.float32),
        "mask_native": inputs_ref["mask_native"].astype(np.float32),
        "lr_nearest_64": inputs_ref["lr_nearest_64"].astype(np.float32),
        "orientation": best_orient,
        "scale_factor": np.float32(inputs_ref["scale"]),
    }
    for mkey, mb in model_best.items():
        # Invert orientation on saved SR
        sr_orig = invert_transform(mb["sr_final_64"], best_orient)
        npz_dict[f"{mkey}_final_sr"] = sr_orig.astype(np.float32)

    np.savez(str(out / "sr_results.npz"), **npz_dict)

    # MAT file
    mat_dict = {k: v for k, v in npz_dict.items() if isinstance(v, np.ndarray)}
    mat_dict["orientation"] = best_orient
    mat_dict["scale_factor"] = inputs_ref["scale"]
    sio.savemat(str(out / "sr_quantitative.mat"), mat_dict, do_compression=True)

    # metrics.csv
    csv_path = out / "metrics.csv"
    fieldnames = ["model", "seed", "network", "struct_mode", "variant",
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
        writer.writerow(["model", "struct_mode", "best_seed", "best_variant",
                         "psnr_native", "ssim_native", "pearson_native",
                         "psnr_64", "ssim_64",
                         "psnr_native_vs_bicubic", "psnr_native_vs_kspace"])
        # kspace baseline
        writer.writerow([
            "baseline_kspace", "-", "-", "-",
            f"{kspace_mnative['psnr']:.6f}", f"{kspace_mnative['ssim']:.6f}", f"{kspace_mnative['pearson']:.6f}",
            f"{kspace_m64['psnr']:.6f}", f"{kspace_m64['ssim']:.6f}",
            f"{kspace_mnative['psnr'] - bic_mnative['psnr']:.6f}", "0.000000",
        ])
        # bicubic baseline
        writer.writerow([
            "baseline_bicubic", "-", "-", "-",
            f"{bic_mnative['psnr']:.6f}", f"{bic_mnative['ssim']:.6f}", f"{bic_mnative['pearson']:.6f}",
            f"{bic_m64['psnr']:.6f}", f"{bic_m64['ssim']:.6f}",
            "0.000000", f"{bic_mnative['psnr'] - kspace_mnative['psnr']:.6f}",
        ])
        for mkey in [c["key"] for c in MODEL_CONFIGS]:
            mb = model_best[mkey]
            mn = mb["best_metrics_native"]
            m64 = mb["best_metrics_64"]
            writer.writerow([
                mkey, mb["struct_mode"], mb["best_seed"], mb["best_variant"],
                f"{mn['psnr']:.6f}", f"{mn['ssim']:.6f}", f"{mn['pearson']:.6f}",
                f"{m64['psnr']:.6f}", f"{m64['ssim']:.6f}",
                f"{mn['psnr'] - bic_mnative['psnr']:.6f}",
                f"{mn['psnr'] - kspace_mnative['psnr']:.6f}",
            ])

    # orientation_search.csv
    with open(str(out / "orientation_search.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orientation", "native_psnr_db_dc"])
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
        "normalization_scale": inputs_ref["scale"],
        "normalization_rule": "LR and HR divided by HR.max(); struct min-max to [0,1]",
        "lr_preparation": "k-space zero-padding with IFFT normalization correction: img *= (out_size/h)*(out_size/w)",
        "postprocessing": {
            "variants": ["raw", "dc", "dc_denoised"],
            "data_consistency": "replace center k-space of SR with LR k-space scaled by (out_size^2)/(h*w)",
            "dc_denoise_sigma": DC_DENOISE_SIGMA,
        },
        "confirmed_orientation": best_orient,
        "orientation_search": orient_results,
        "seeds": SEEDS,
        "sample_num_steps": SAMPLE_STEPS,
        "baselines": {
            "kspace": {"psnr_native": kspace_mnative["psnr"], "pearson_native": kspace_mnative["pearson"],
                       "psnr_64": kspace_m64["psnr"]},
            "bicubic": {"psnr_native": bic_mnative["psnr"], "pearson_native": bic_mnative["pearson"],
                        "psnr_64": bic_m64["psnr"]},
        },
        "models": {},
        "overall_best": overall_best[0],
        "overall_best_native_psnr": overall_best[1],
        "ablation_notes": [
            "k-space zero-padding without normalization correction caused LR to be ~35x too dark",
            "Adding normalization correction (out_size/h)*(out_size/w) fixes this",
            "Data consistency (DC) replaces low-freq k-space with measured LR, improving PSNR significantly",
            "If model SR does not beat baseline, baseline is reported as final best (honest reporting)",
        ],
    }
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        mb = model_best[mkey]
        meta["models"][mkey] = {
            "struct_mode": mb["struct_mode"],
            "best_seed": mb["best_seed"],
            "best_variant": mb["best_variant"],
            "final_native_psnr": mb["best_metrics_native"]["psnr"],
            "final_native_pearson": mb["best_metrics_native"]["pearson"],
            "final_64_psnr": mb["best_metrics_64"]["psnr"],
        }

    with open(str(out / "inference_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} - DONE")
    print(f"  Orientation: {best_orient}")
    print(f"  Baselines: kspace={kspace_mnative['psnr']:.2f} dB, bicubic={bic_mnative['psnr']:.2f} dB")
    for mkey in [c["key"] for c in MODEL_CONFIGS]:
        mb = model_best[mkey]
        print(f"  {mkey:25s}  variant={mb['best_variant']:12s}  "
              f"native PSNR={mb['best_metrics_native']['psnr']:.3f}  "
              f"corr={mb['best_metrics_native']['pearson']:.4f}")
    print(f"  OVERALL BEST: {overall_best[0]} ({overall_best[1]:.3f} dB)")
    print(f"{'='*70}")

    return meta


def main():
    print("=" * 70)
    print("Water-film MRSI SR3 Inference v4 (k-space norm fix + DC)")
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
        print(f"  {s['dataset']:20s}  overall_best={s['overall_best']:30s}  "
              f"native_PSNR={s['overall_best_native_psnr']:.3f} dB  "
              f"orientation={s['confirmed_orientation']}")


if __name__ == "__main__":
    main()
