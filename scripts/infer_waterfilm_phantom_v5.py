#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-film MRSI SR3 super-resolution inference (v5).

Based on v4 with fixes:
  1. Saves all per-model raw/dc/dc_denoised arrays (best + all-5-seed stacks)
  2. Meta JSON includes full array manifest
  3. Orientation tie detection
  4. New filenames (_v5 suffix) — does NOT overwrite v4 files
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

TARGET_SIZE = 64
MET_ID = 1
SEEDS = [0, 1, 2, 3, 4]
SAMPLE_STEPS = 50
DC_DENOISE_SIGMA = 0.5
STRUCT_MODES = ["zero", "normal"]
PSNR_TIE_TOL = 1e-6

MODEL_CONFIGS = [
    {"key": "healthy_ema",
     "config_path": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "config_resolved.json"),
     "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "checkpoint" / "I300000_E2522"),
     "model_dir": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema"),
     "network": "ema"},
    {"key": "healthy_raw",
     "config_path": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "config_resolved.json"),
     "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "checkpoint" / "I300000_E2522"),
     "model_dir": str(_REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema"),
     "network": "raw"},
    {"key": "current_mixed_raw",
     "config_path": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw" / "config_resolved.json"),
     "checkpoint_prefix": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw" / "checkpoint" / "I50500_E1931"),
     "model_dir": str(_REPO_ROOT / "experiments" / "models" / "current_mixed_i50500_raw"),
     "network": "raw"},
]

D4_TRANSFORMS = {
    "identity": lambda a: a.copy(),
    "rot90": lambda a: np.rot90(a, k=1),
    "rot180": lambda a: np.rot90(a, k=2),
    "rot270": lambda a: np.rot90(a, k=3),
    "hflip": lambda a: np.fliplr(a).copy(),
    "vflip": lambda a: np.flipud(a).copy(),
    "transpose": lambda a: np.transpose(a).copy(),
    "anti_transpose": lambda a: np.flipud(np.rot90(a, k=1)).copy(),
}
D4_INVERSES = {
    "identity": "identity", "rot90": "rot270", "rot180": "rot180",
    "rot270": "rot90", "hflip": "hflip", "vflip": "vflip",
    "transpose": "transpose", "anti_transpose": "anti_transpose",
}


def apply_transform(arr, name):
    return D4_TRANSFORMS[name](arr)


def invert_transform(arr, name):
    return D4_TRANSFORMS[D4_INVERSES[name]](arr)


# ---------- data loading (same as v4) ----------
def load_ygh_data():
    mat_path = Path(r"D:\LMC\data\phantom_ygh\最终处理结果\results\reprocessing_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        records = f["records"]
        def deref(field, idx):
            return np.squeeze(f[records[field][idx, 0]][()])
        return {
            "lr": deref("volNormMap", 0).astype(np.float64),
            "hr": deref("volNormMap", 1).astype(np.float64),
            "lr_mask": deref("supportMask", 0).astype(np.uint8),
            "hr_mask": deref("supportMask", 1).astype(np.uint8),
            "struct": deref("structImg", 0).astype(np.float64),
            "mat_path": str(mat_path),
            "struct_path": str(mat_path) + " (records/structImg)",
        }


def load_waterfilm_data():
    mat_path = Path(r"D:\LMC\data\水膜数据处理\最终处理结果\results\all_results.mat")
    with h5py.File(str(mat_path), "r") as f:
        sr29 = f["allResults"]["scan29"]
        sr30 = f["allResults"]["scan30"]
        d2 = {
            "lr": np.array(sr29["peakAreaVolNorm"]).astype(np.float64),
            "hr": np.array(sr30["peakAreaVolNorm"]).astype(np.float64),
            "lr_mask": np.array(sr29["signalMask"]).astype(np.uint8),
            "hr_mask": np.array(sr30["signalMask"]).astype(np.uint8),
        }
    d2seq = Path(r"D:\LMC\data\水膜数据处理\20260528\20260528_103023_20260528_Phantom01_1_1\28\pdata\1\2dseq")
    raw = np.fromfile(str(d2seq), dtype=np.uint16)
    echo0 = raw[: raw.size // 2]
    d2["struct"] = echo0.reshape(128, 128).astype(np.float64)
    d2["mat_path"] = str(mat_path)
    d2["struct_path"] = str(d2seq)
    return d2


# ---------- preprocessing ----------
def _norm01(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo)).astype(np.float32) if hi - lo > 1e-12 else np.zeros_like(a, dtype=np.float32)


def _resize(arr, size, mode="bicubic"):
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size), mode="nearest")
    else:
        t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


def kspace_zeropad_lr(lr_small, out_size=TARGET_SIZE):
    h, w = lr_small.shape
    k = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_big = np.zeros((out_size, out_size), dtype=np.complex64)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_big[y0:y0+h, x0:x0+w] = k
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(k_big))).astype(np.float32)
    return img * (out_size / h) * (out_size / w)


def data_consistency(sr64, lr_small, out_size=TARGET_SIZE):
    h, w = lr_small.shape
    k_sr = np.fft.fftshift(np.fft.fft2(sr64.astype(np.float32)))
    k_lr = np.fft.fftshift(np.fft.fft2(lr_small.astype(np.float32)))
    k_lr_scaled = k_lr * (out_size ** 2) / (h * w)
    cy, cx = out_size // 2, out_size // 2
    y0, x0 = cy - h // 2, cx - w // 2
    k_sr[y0:y0+h, x0:x0+w] = k_lr_scaled
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k_sr))).astype(np.float32)


# ---------- model ----------
def _load_json_config(path):
    s = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s += line.split("//")[0] + "\n"
    return json.loads(s, object_pairs_hook=OrderedDict)


def build_model(config_path, checkpoint_prefix, model_dir, gpu_id=0):
    opt = _load_json_config(config_path)
    opt["phase"] = "val"
    opt["gpu_ids"] = [gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    opt["path"]["experiments_root"] = model_dir
    for k in ("log", "tb_logger", "results", "checkpoint"):
        if k in opt["path"]:
            opt["path"][k] = str(Path(model_dir) / opt["path"][k])
    opt["path"]["resume_state"] = checkpoint_prefix
    opt["rank"] = 0; opt["world_size"] = 1; opt["local_rank"] = 0
    opt["distributed"] = False; opt["is_main_process"] = True; opt["enable_wandb"] = False
    opt = Logger.dict_to_nonedict(opt)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    return diffusion, opt


def build_batch(lr_64, struct_64, mask_64, use_mask, struct_mode, device):
    lr_t = torch.from_numpy(lr_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    mask_t = torch.from_numpy(mask_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    struct_t = torch.zeros(1, 1, TARGET_SIZE, TARGET_SIZE, device=device, dtype=lr_t.dtype) if struct_mode == "zero" \
        else torch.from_numpy(struct_64).view(1, 1, TARGET_SIZE, TARGET_SIZE).to(device)
    parts = [lr_t, struct_t, struct_t]
    met_oh = torch.zeros(1, 4, TARGET_SIZE, TARGET_SIZE, device=device, dtype=lr_t.dtype)
    met_oh[:, MET_ID] = 1.0
    parts.append(met_oh)
    if use_mask:
        parts.append(mask_t)
    cond = torch.cat(parts, dim=1)
    to_m1 = lambda x: x * 2.0 - 1.0
    return {"HR": to_m1(lr_t), "SR": to_m1(cond), "LR": to_m1(lr_t), "MASK": mask_t}


def run_inference(diffusion, batch, seed, network):
    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=int(seed), sample_num_steps=SAMPLE_STEPS, network=network)
    vis = diffusion.get_current_visuals(need_LR=True)
    sr = vis["SR"]
    if sr.dim() == 4 and sr.shape[0] > 1:
        sr = sr[-1:]
    sr = sr.detach().float().cpu().squeeze().clamp(-1, 1)
    return ((sr + 1.0) * 0.5).numpy().astype(np.float32).clip(0.0, 1.0)


# ---------- metrics ----------
def _psnr(p, r):
    pu = np.clip(np.round(p * 255), 0, 255).astype(np.float64)
    ru = np.clip(np.round(r * 255), 0, 255).astype(np.float64)
    mse = np.mean((pu - ru) ** 2)
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 99.0


def compute_metrics(pred, ref, mask):
    m = mask.astype(bool)
    p = pred[m].astype(np.float64); r = ref[m].astype(np.float64)
    fft_pow = np.abs(np.fft.fftshift(np.fft.fft2(pred.astype(np.float64)))) ** 2
    cy, cx = fft_pow.shape[0] // 2, fft_pow.shape[1] // 2
    Y, X = np.ogrid[:fft_pow.shape[0], :fft_pow.shape[1]]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)
    hf_mask = dist > (0.25 * max_r)
    hf_ratio = float(fft_pow[hf_mask].sum() / fft_pow.sum()) if fft_pow.sum() > 0 else 0.0

    return {
        "psnr": _psnr(pred, ref),
        "ssim": float(sk_ssim(pred.astype(np.float64), ref.astype(np.float64), data_range=1.0)),
        "mae": float(np.mean(np.abs(p - r))) if p.size else 0.0,
        "rmse": float(np.sqrt(np.mean((p - r) ** 2))) if p.size else 0.0,
        "pearson": float(np.corrcoef(p, r)[0, 1]) if len(p) > 1 and np.std(p) > 1e-12 and np.std(r) > 1e-12 else 0.0,
        "tv": float(np.mean(np.hypot(*np.gradient(pred.astype(np.float64))))),
        "hf_ratio": hf_ratio,
        "bg_leakage": float(pred[~m].mean()) / (float(pred[m].mean()) if m.sum() > 0 else 1.0)
            if m.sum() > 0 and pred[m].mean() > 1e-12 else 0.0,
    }


def build_inputs(lr, hr, struct, hr_mask, orientation, struct_mode):
    lr_t = apply_transform(lr, orientation)
    hr_t = apply_transform(hr, orientation)
    st_t = apply_transform(struct, orientation)
    mk_t = apply_transform(hr_mask, orientation)
    hr_max = float(hr_t.max())
    scale = hr_max if hr_max > 1e-12 else 1.0
    lr_n = (lr_t / scale).astype(np.float32)
    hr_n = (hr_t / scale).astype(np.float32)
    st_n = _norm01(st_t)
    mk_b = (mk_t > 0.5).astype(np.float32)
    return {
        "lr_norm": lr_n,
        "lr_kspace_64": kspace_zeropad_lr(lr_n, TARGET_SIZE).clip(0, 1),
        "lr_bicubic_64": _resize(lr_n, TARGET_SIZE).clip(0, 1),
        "hr_bicubic_64": _resize(hr_n, TARGET_SIZE).clip(0, 1),
        "hr_native": hr_n,
        "struct_64": _resize(st_n, TARGET_SIZE).clip(0, 1),
        "mask_64": (_resize(mk_b, TARGET_SIZE, "nearest") >= 0.5).astype(np.float32),
        "mask_native": mk_b,
        "lr_nearest_64": _resize(lr_n, TARGET_SIZE, "nearest").clip(0, 1),
        "scale": scale,
        "native_size": hr_t.shape[0],
    }


# ---------- main ----------
def process_dataset(name, data, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lr, hr, hr_mask, struct = data["lr"], data["hr"], data["hr_mask"], data["struct"]

    print(f"\n{'='*70}\nDataset: {name}\n{'='*70}")

    # Step 1: D4 search
    print("[Step 1] D4 orientation search...")
    hcfg = MODEL_CONFIGS[0]
    diffusion, opt = build_model(hcfg["config_path"], hcfg["checkpoint_prefix"], hcfg["model_dir"])
    use_mask_h = bool(dict(opt.get("datasets", {}).get("val", {}) or {}).get("use_mask_channel", False))
    dev = diffusion.device
    orient_results = {}
    for oname in D4_TRANSFORMS:
        inp = build_inputs(lr, hr, struct, hr_mask, oname, "zero")
        b = build_batch(inp["lr_kspace_64"], inp["struct_64"], inp["mask_64"], use_mask_h, "zero", dev)
        sr = run_inference(diffusion, b, 0, "ema")
        dc = data_consistency(sr, inp["lr_norm"]).clip(0, 1)
        nat = _resize(dc, inp["native_size"])
        m = compute_metrics(nat, inp["hr_native"], inp["mask_native"])
        orient_results[oname] = m["psnr"]
        print(f"  {oname:20s}  PSNR_n(DC)={m['psnr']:.4f}")
    del diffusion; torch.cuda.empty_cache()

    # Tie detection
    best_val = max(orient_results.values())
    tied = [k for k, v in orient_results.items() if abs(v - best_val) < PSNR_TIE_TOL]
    selected = tied[0]
    tie_detected = len(tied) > 1
    print(f"  Selected: {selected} (tie={tie_detected}, tied={tied})")

    # Step 2: Multi-seed inference
    print("[Step 2] Multi-seed inference...")
    all_rows = []
    # Store all SR arrays for saving: model -> struct_mode -> seed -> {variant: array_64}
    all_sr_arrays: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {}
    model_best = {}

    for mcfg in MODEL_CONFIGS:
        mk = mcfg["key"]
        print(f"\n  --- {mk} ---")
        diffusion, opt = build_model(mcfg["config_path"], mcfg["checkpoint_prefix"], mcfg["model_dir"])
        um = bool(dict(opt.get("datasets", {}).get("val", {}) or {}).get("use_mask_channel", False))
        dm = diffusion.device
        all_sr_arrays[mk] = {}
        best_overall = None

        for smode in STRUCT_MODES:
            inp = build_inputs(lr, hr, struct, hr_mask, selected, smode)
            b = build_batch(inp["lr_kspace_64"], inp["struct_64"], inp["mask_64"], um, smode, dm)
            all_sr_arrays[mk][smode] = {}

            for seed in SEEDS:
                t0 = time.time()
                raw = run_inference(diffusion, b, seed, mcfg["network"])
                elapsed = time.time() - t0
                dc = data_consistency(raw, inp["lr_norm"]).clip(0, 1)
                dcdn = gaussian_filter(dc.astype(np.float64), sigma=DC_DENOISE_SIGMA).astype(np.float32).clip(0, 1)

                all_sr_arrays[mk][smode][seed] = {"raw": raw, "dc": dc, "dc_denoised": dcdn}

                for vname, arr64 in [("raw", raw), ("dc", dc), ("dc_denoised", dcdn)]:
                    m64 = compute_metrics(arr64, inp["hr_bicubic_64"], inp["mask_64"])
                    nat = _resize(arr64, inp["native_size"])
                    mn = compute_metrics(nat, inp["hr_native"], inp["mask_native"])
                    all_rows.append({
                        "model": mk, "seed": seed, "network": mcfg["network"],
                        "struct_mode": smode, "variant": vname,
                        "psnr_64": m64["psnr"], "ssim_64": m64["ssim"],
                        "mae_64": m64["mae"], "rmse_64": m64["rmse"], "pearson_64": m64["pearson"],
                        "psnr_native": mn["psnr"], "ssim_native": mn["ssim"], "pearson_native": mn["pearson"],
                        "tv": m64["tv"], "hf_ratio": m64["hf_ratio"], "bg_leakage": m64["bg_leakage"],
                        "time_sec": round(elapsed, 2),
                    })
                    if best_overall is None or mn["psnr"] > best_overall[0]:
                        best_overall = (mn["psnr"], smode, seed, vname, arr64.copy(), m64, mn)

                print(f"    {smode:6s} s={seed}  raw:{compute_metrics(_resize(raw,inp['native_size']),inp['hr_native'],inp['mask_native'])['psnr']:.2f}  "
                      f"dc:{compute_metrics(_resize(dc,inp['native_size']),inp['hr_native'],inp['mask_native'])['psnr']:.2f}  "
                      f"dc_dn:{compute_metrics(_resize(dcdn,inp['native_size']),inp['hr_native'],inp['mask_native'])['psnr']:.2f}")

        del diffusion; torch.cuda.empty_cache()
        _, b_smode, b_seed, b_variant, _, b_m64, b_mn = best_overall
        model_best[mk] = {"struct_mode": b_smode, "best_seed": b_seed, "best_variant": b_variant,
                          "metrics_64": b_m64, "metrics_native": b_mn}
        print(f"    >> best: {b_smode} seed={b_seed} variant={b_variant} PSNR_n={b_mn['psnr']:.3f}")

    # Baselines
    inp_ref = build_inputs(lr, hr, struct, hr_mask, selected, "zero")
    kspace_64 = inp_ref["lr_kspace_64"]
    bicubic_64 = inp_ref["lr_bicubic_64"]
    k_m64 = compute_metrics(kspace_64, inp_ref["hr_bicubic_64"], inp_ref["mask_64"])
    k_nat = _resize(kspace_64, inp_ref["native_size"])
    k_mn = compute_metrics(k_nat, inp_ref["hr_native"], inp_ref["mask_native"])
    b_m64 = compute_metrics(bicubic_64, inp_ref["hr_bicubic_64"], inp_ref["mask_64"])
    b_nat = _resize(bicubic_64, inp_ref["native_size"])
    b_mn = compute_metrics(b_nat, inp_ref["hr_native"], inp_ref["mask_native"])

    # Overall best
    cands = [("baseline_kspace", k_mn["psnr"]), ("baseline_bicubic", b_mn["psnr"])]
    for mk in MODEL_CONFIGS:
        cands.append((f"{mk['key']}_{model_best[mk['key']]['best_variant']}", model_best[mk['key']]["metrics_native"]["psnr"]))
    overall_best = max(cands, key=lambda x: x[1])

    # ---- Build NPZ ----
    print("\n[Step 3] Saving v5 NPZ/MAT...")
    npz: Dict[str, np.ndarray] = {
        "lr_kspace_64": inp_ref["lr_kspace_64"].astype(np.float32),
        "lr_bicubic_64": inp_ref["lr_bicubic_64"].astype(np.float32),
        "hr_bicubic_64": inp_ref["hr_bicubic_64"].astype(np.float32),
        "hr_native": inp_ref["hr_native"].astype(np.float32),
        "struct_64": inp_ref["struct_64"].astype(np.float32),
        "mask_64": inp_ref["mask_64"].astype(np.float32),
        "mask_native": inp_ref["mask_native"].astype(np.float32),
        "lr_nearest_64": inp_ref["lr_nearest_64"].astype(np.float32),
        "baseline_kspace_64": kspace_64.astype(np.float32),
        "baseline_bicubic_64": bicubic_64.astype(np.float32),
        "orientation": selected,
        "scale_factor": np.float32(inp_ref["scale"]),
    }

    array_manifest = []  # for meta JSON

    for mcfg in MODEL_CONFIGS:
        mk = mcfg["key"]
        mb = model_best[mk]
        b_sm = mb["struct_mode"]
        b_sd = mb["best_seed"]

        # Save best-seed arrays (in oriented space)
        for vname in ["raw", "dc", "dc_denoised"]:
            arr = all_sr_arrays[mk][b_sm][b_sd][vname]
            key = f"{mk}_{vname}_sr_best"
            npz[key] = arr.astype(np.float32)
            # Metrics for this variant
            m64 = compute_metrics(arr, inp_ref["hr_bicubic_64"], inp_ref["mask_64"])
            nat = _resize(arr, inp_ref["native_size"])
            mn = compute_metrics(nat, inp_ref["hr_native"], inp_ref["mask_native"])
            array_manifest.append({
                "field": key, "model": mk, "variant": vname, "seed": b_sd,
                "struct_mode": b_sm, "shape": list(arr.shape),
                "psnr_native": mn["psnr"], "psnr_64": m64["psnr"],
            })

        # Final = best variant
        final_arr = all_sr_arrays[mk][b_sm][b_sd][mb["best_variant"]]
        npz[f"{mk}_final_sr"] = final_arr.astype(np.float32)
        array_manifest.append({
            "field": f"{mk}_final_sr", "model": mk, "variant": mb["best_variant"], "seed": b_sd,
            "struct_mode": b_sm, "shape": list(final_arr.shape),
            "psnr_native": mb["metrics_native"]["psnr"], "psnr_64": mb["metrics_64"]["psnr"],
        })

        # Save all-5-seed stacks for best struct_mode
        for vname in ["raw", "dc", "dc_denoised"]:
            stack = np.stack([all_sr_arrays[mk][b_sm][s][vname] for s in SEEDS], axis=0).astype(np.float32)
            key = f"{mk}_{vname}_sr_all"
            npz[key] = stack
            array_manifest.append({
                "field": key, "model": mk, "variant": vname, "seed": "all",
                "struct_mode": b_sm, "shape": list(stack.shape),
                "psnr_native": None, "psnr_64": None,
            })

    np.savez(str(out / "sr_results_v5.npz"), **npz)

    # MAT file
    mat_d = {k: v for k, v in npz.items() if isinstance(v, np.ndarray)}
    mat_d["orientation"] = selected
    mat_d["scale_factor"] = inp_ref["scale"]
    sio.savemat(str(out / "sr_quantitative_v5.mat"), mat_d, do_compression=True)

    # metrics.csv (v5, same as v4 but with _v5 name to not overwrite)
    with open(str(out / "metrics_v5.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","seed","network","struct_mode","variant",
            "psnr_64","ssim_64","mae_64","rmse_64","pearson_64",
            "psnr_native","ssim_native","pearson_native","tv","hf_ratio","bg_leakage","time_sec"])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    # Summary CSV
    with open(str(out / "metrics_summary_v5.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","struct_mode","best_seed","best_variant",
                     "psnr_native","ssim_native","pearson_native","psnr_64","ssim_64",
                     "vs_bicubic_native","vs_kspace_native"])
        w.writerow(["baseline_kspace","-","-","-",
                    f"{k_mn['psnr']:.6f}",f"{k_mn['ssim']:.6f}",f"{k_mn['pearson']:.6f}",
                    f"{k_m64['psnr']:.6f}",f"{k_m64['ssim']:.6f}",
                    f"{k_mn['psnr']-b_mn['psnr']:.6f}","0.000000"])
        w.writerow(["baseline_bicubic","-","-","-",
                    f"{b_mn['psnr']:.6f}",f"{b_mn['ssim']:.6f}",f"{b_mn['pearson']:.6f}",
                    f"{b_m64['psnr']:.6f}",f"{b_m64['ssim']:.6f}",
                    "0.000000",f"{b_mn['psnr']-k_mn['psnr']:.6f}"])
        for mcfg in MODEL_CONFIGS:
            mk = mcfg["key"]; mb = model_best[mk]
            mn = mb["metrics_native"]; m64 = mb["metrics_64"]
            w.writerow([mk, mb["struct_mode"], mb["best_seed"], mb["best_variant"],
                        f"{mn['psnr']:.6f}",f"{mn['ssim']:.6f}",f"{mn['pearson']:.6f}",
                        f"{m64['psnr']:.6f}",f"{m64['ssim']:.6f}",
                        f"{mn['psnr']-b_mn['psnr']:.6f}",f"{mn['psnr']-k_mn['psnr']:.6f}"])

    # Orientation CSV
    with open(str(out / "orientation_search_v5.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orientation", "native_psnr_db_dc"])
        for on, op in orient_results.items():
            w.writerow([on, f"{op:.10f}"])
        w.writerow(["SELECTED", selected])
        if tie_detected:
            w.writerow(["TIE", ",".join(tied)])

    # Meta JSON
    meta = {
        "dataset": name,
        "version": "v5",
        "mat_path": data["mat_path"],
        "struct_path": data["struct_path"],
        "lr_shape_raw": list(lr.shape), "hr_shape_raw": list(hr.shape), "struct_shape_raw": list(struct.shape),
        "target_size": TARGET_SIZE, "metabolite": "Glc", "met_id": MET_ID,
        "normalization_scale": inp_ref["scale"],
        "lr_preparation": "k-space zero-padding with IFFT normalization correction: img *= (out_size/h)*(out_size/w)",
        "postprocessing": {
            "variants": ["raw", "dc", "dc_denoised"],
            "data_consistency": "replace center k-space with LR k-space scaled by (out_size^2)/(h*w)",
            "dc_denoise_sigma": DC_DENOISE_SIGMA,
        },
        "selected_orientation": selected,
        "orientation_tie": tie_detected,
        "tied_orientations": tied if tie_detected else None,
        "orientation_note": (
            "Multiple orientations have numerically equivalent PSNR due to data symmetry; "
            f"selected '{selected}' arbitrarily among tied orientations."
            if tie_detected else
            "Unique best orientation found."
        ),
        "orientation_search": orient_results,
        "seeds": SEEDS,
        "sample_num_steps": SAMPLE_STEPS,
        "baselines": {
            "kspace": {"psnr_native": k_mn["psnr"], "pearson_native": k_mn["pearson"], "psnr_64": k_m64["psnr"]},
            "bicubic": {"psnr_native": b_mn["psnr"], "pearson_native": b_mn["pearson"], "psnr_64": b_m64["psnr"]},
        },
        "models": {},
        "overall_best": overall_best[0],
        "overall_best_native_psnr": overall_best[1],
        "arrays": array_manifest,
        "file_list": [
            "sr_results_v5.npz", "sr_quantitative_v5.mat",
            "metrics_v5.csv", "metrics_summary_v5.csv",
            "orientation_search_v5.csv", "inference_meta_v5.json",
        ],
    }
    for mcfg in MODEL_CONFIGS:
        mk = mcfg["key"]; mb = model_best[mk]
        meta["models"][mk] = {
            "struct_mode": mb["struct_mode"], "best_seed": mb["best_seed"],
            "best_variant": mb["best_variant"],
            "final_native_psnr": mb["metrics_native"]["psnr"],
            "final_native_pearson": mb["metrics_native"]["pearson"],
            "final_64_psnr": mb["metrics_64"]["psnr"],
        }

    with open(str(out / "inference_meta_v5.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\n{name} DONE: overall_best={overall_best[0]} ({overall_best[1]:.3f} dB)\n{'='*70}")
    return meta


def main():
    print("="*70 + "\nWater-film MRSI SR3 v5\n" + "="*70)
    datasets = [
        ("ygh_phantom", load_ygh_data(), r"D:\LMC\data\phantom_ygh\最终处理结果\推理"),
        ("waterfilm_user", load_waterfilm_data(), r"D:\LMC\data\水膜数据处理\最终处理结果\推理"),
    ]
    summaries = []
    for n, d, od in datasets:
        try:
            summaries.append(process_dataset(n, d, od))
        except Exception as e:
            print(f"ERROR {n}: {e}")
            traceback.print_exc()
    print("\nALL DONE")
    for s in summaries:
        print(f"  {s['dataset']:20s}  best={s['overall_best']:35s}  PSNR={s['overall_best_native_psnr']:.3f} dB")


if __name__ == "__main__":
    main()
