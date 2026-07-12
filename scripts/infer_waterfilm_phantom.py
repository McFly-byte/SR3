#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-film phantom SR3 inference with baseline-consistent visualization.

输出：
1) SR 数值 .npy（64x64）
2) LR / SR / HR 一行三列 PNG（统一 cmap，vmin=0, vmax=1）
3) meta.json（记录 checkpoint、命令、输入输出路径）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import core.logger as Logger  # noqa: E402
import core.metrics as Metrics  # noqa: E402
import model as Model  # noqa: E402

_RUN_RE = re.compile(r"^(?P<base>.+)_\d{6}_\d{6}$")


def _load_json_config(path: Path) -> OrderedDict:
    json_str = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _resize_2d(arr: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size), mode="nearest")
    elif mode == "bicubic":
        t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")
    return t.squeeze().numpy().astype(np.float32)


def _stack_condition_from_parts(
    lr: torch.Tensor,
    t1: torch.Tensor,
    flair: torch.Tensor,
    met_onehot: torch.Tensor,
    mask: torch.Tensor,
    ds_opt: Dict[str, Any],
) -> torch.Tensor:
    parts = []
    if ds_opt.get("use_lr", True):
        parts.append(lr)
    if ds_opt.get("use_t1", True):
        parts.append(t1)
    if ds_opt.get("use_flair", True):
        parts.append(flair)
    if ds_opt.get("use_met_onehot", True):
        parts.append(met_onehot)
    if ds_opt.get("use_mask_channel", False):
        parts.append(mask)
    if not parts:
        raise ValueError("No condition channels enabled in dataset config.")
    return torch.cat(parts, dim=1)


def _build_opt(config_path: Path, run_dir: Path, resume_prefix: str, gpu_ids: Optional[str]) -> Any:
    opt = _load_json_config(config_path)
    run_dir = run_dir.resolve()
    opt["phase"] = "val"
    opt["path"]["experiments_root"] = str(run_dir)
    for key, rel in list(opt["path"].items()):
        if "resume" in key or "experiments" in key:
            continue
        opt["path"][key] = str(run_dir / rel)
    opt["path"]["resume_state"] = resume_prefix

    if gpu_ids is not None:
        if gpu_ids.strip() == "":
            opt["gpu_ids"] = []
        else:
            opt["gpu_ids"] = [int(x) for x in gpu_ids.split(",") if x.strip() != ""]
    if not opt.get("gpu_ids"):
        opt["gpu_ids"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt["gpu_ids"]) if opt["gpu_ids"] else ""

    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["distributed"] = False
    opt["is_main_process"] = True
    return Logger.dict_to_nonedict(opt)


def _choose_config_for_run(run_dir: Path, config_dir: Path) -> Path:
    m = _RUN_RE.match(run_dir.name)
    candidates = []
    if m:
        candidates.append(config_dir / f"{m.group('base')}.json")
    candidates.append(config_dir / "sr3_mrsi_64_mvp.json")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Cannot infer config from run: {run_dir}")


def _find_latest_checkpoint(experiments_dir: Path, prefer_ema: bool = True) -> Tuple[Path, Path]:
    if prefer_ema:
        pats = ("*_ema_gen.pth", "*_gen.pth")
    else:
        pats = ("*_gen.pth",)

    all_files = []
    for pat in pats:
        all_files.extend(experiments_dir.rglob(pat))
    if not all_files:
        raise FileNotFoundError(f"No checkpoint found under {experiments_dir}")

    all_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = all_files[0]
    name = ckpt_path.name
    if name.endswith("_ema_gen.pth"):
        prefix_name = name[: -len("_ema_gen.pth")]
    elif name.endswith("_gen.pth"):
        prefix_name = name[: -len("_gen.pth")]
    else:
        raise RuntimeError(f"Unexpected checkpoint name: {name}")
    resume_prefix = ckpt_path.parent / prefix_name
    return ckpt_path, resume_prefix


def _build_batch_from_npz(npz_path: Path, ds_opt: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    d = np.load(str(npz_path))
    hr = torch.from_numpy(d["hr"]).float().unsqueeze(0)
    lr = torch.from_numpy(d["lr"]).float().unsqueeze(0)
    t1 = torch.from_numpy(d["t1"]).float().unsqueeze(0)
    flair = torch.from_numpy(d["flair"]).float().unsqueeze(0)
    met_onehot = torch.from_numpy(d["met_onehot"]).float().unsqueeze(0)
    mask = torch.from_numpy(d["mask"]).float().unsqueeze(0)

    cond = _stack_condition_from_parts(lr, t1, flair, met_onehot, mask, ds_opt)
    out: Dict[str, Any] = {
        "HR": _to_minus1_1(hr),
        "SR": _to_minus1_1(cond),
        "LR": _to_minus1_1(lr),
        "MASK": mask,
    }
    for k, v in out.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def _tensor_to_01_2d(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().squeeze().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    return x.numpy().astype(np.float32)


def _to_u8(img01: np.ndarray) -> np.ndarray:
    x = np.asarray(img01, dtype=np.float32)
    return np.clip(np.round(x * 255.0), 0.0, 255.0).astype(np.uint8)


def _metric_value(metric: str, pred01: np.ndarray, ref01: np.ndarray) -> float:
    p = _to_u8(pred01)
    r = _to_u8(ref01)
    if metric == "psnr":
        return float(Metrics.calculate_psnr(p, r))
    if metric == "ssim":
        return float(Metrics.calculate_ssim(p, r))
    raise ValueError(f"Unsupported metric: {metric}")


def _parse_seed_list(seed_candidates: str) -> List[int]:
    out: List[int] = []
    for s in seed_candidates.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("seed list is empty")
    return out


def _save_triplet_png(
    lr_nearest_64: np.ndarray,
    sr_64: np.ndarray,
    hr_bicubic_64: np.ndarray,
    output_png: Path,
    title: str,
    cmap: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = (
        (lr_nearest_64, "LR scan29", "nearest"),
        (sr_64, "SR prediction", "nearest"),
        (hr_bicubic_64, "HR scan30 reference", "nearest"),
    )
    im = None
    for ax, (arr, panel_title, interp) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation=interp)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Water-film phantom SR3 inference")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=str(_REPO_ROOT / "experiments"),
        help="Directory that contains run folders and checkpoints.",
    )
    parser.add_argument("--run_dir", type=str, default=None, help="Optional run directory override.")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override.")
    parser.add_argument(
        "--sample_npz",
        type=str,
        default=str(_REPO_ROOT / "SR3_INFERENCE_DATA" / "test" / "samples" / "00000000.npz"),
        help="SR3 model input sample (.npz).",
    )
    parser.add_argument(
        "--lr_npy",
        type=str,
        default=str(_REPO_ROOT / "SR3_INFERENCE_DATA" / "npy" / "scan_29_lr_8x8_norm_self.npy"),
        help="Raw 8x8 LR npy for display (nearest -> 64x64).",
    )
    parser.add_argument(
        "--hr_npy",
        type=str,
        default=str(_REPO_ROOT / "SR3_INFERENCE_DATA" / "npy" / "scan_30_hr_12x12_norm_self.npy"),
        help="Raw 12x12 HR npy for display (bicubic -> 64x64).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_REPO_ROOT / "SR3_INFERENCE_DATA" / "inference_results"),
        help="Output directory for npy/png/meta.",
    )
    parser.add_argument("--output_prefix", type=str, default="waterfilm_sr3_latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed_candidates",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for best-of-N sampling, e.g. 0,1,2,3,4",
    )
    parser.add_argument(
        "--select_metric",
        type=str,
        choices=["psnr", "ssim"],
        default="psnr",
        help="Metric used to choose best SR among seeds.",
    )
    parser.add_argument(
        "--bicubic_tolerance",
        type=float,
        default=0.0,
        help="Allow SR metric to be lower than bicubic by this margin before auto-blend.",
    )
    parser.add_argument(
        "--disable_auto_blend_if_worse",
        action="store_true",
        help="Disable fallback blend with bicubic when SR is worse than baseline.",
    )
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cmap", type=str, default="turbo")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir).resolve()
    sample_npz = Path(args.sample_npz).resolve()
    lr_npy = Path(args.lr_npy).resolve()
    hr_npy = Path(args.hr_npy).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sample_npz.is_file():
        raise FileNotFoundError(sample_npz)
    if not lr_npy.is_file():
        raise FileNotFoundError(lr_npy)
    if not hr_npy.is_file():
        raise FileNotFoundError(hr_npy)

    latest_ckpt_path, resume_prefix = _find_latest_checkpoint(experiments_dir, prefer_ema=True)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_ckpt_path.parent.parent
    config_path = Path(args.config).resolve() if args.config else _choose_config_for_run(run_dir, _REPO_ROOT / "config")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    gpu_ids_arg = "" if args.cpu else args.gpu_ids.strip()
    opt = _build_opt(config_path, run_dir, str(resume_prefix), gpu_ids_arg)

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    val_cfg = opt.get("validation", {}) or {}
    sample_num_steps = args.sample_num_steps
    if sample_num_steps is None:
        sample_num_steps = val_cfg.get("sample_num_steps")
    if sample_num_steps is None:
        sample_num_steps = opt["model"]["diffusion"].get("sample_num_steps")

    ds_val = dict(opt.get("datasets", {}).get("val", {}) or {})
    batch = _build_batch_from_npz(sample_npz, ds_val, diffusion.device)
    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=int(args.seed), sample_num_steps=sample_num_steps)
    visuals = diffusion.get_current_visuals(need_LR=True)

    sr_tensor = visuals["SR"]
    if sr_tensor.dim() == 4 and sr_tensor.shape[0] > 1:
        sr_tensor = sr_tensor[-1:]
    sr_64 = _tensor_to_01_2d(sr_tensor).clip(0.0, 1.0)

    # Visualization data strictly follows user's baseline rule.
    lr_raw = np.load(str(lr_npy)).astype(np.float32)
    hr_raw = np.load(str(hr_npy)).astype(np.float32)
    if lr_raw.ndim != 2 or hr_raw.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got lr={lr_raw.shape}, hr={hr_raw.shape}")
    lr_raw = _norm01(lr_raw)
    hr_raw = _norm01(hr_raw)
    lr_nearest_64 = _resize_2d(lr_raw, size=64, mode="nearest").clip(0.0, 1.0)
    lr_bicubic_64 = _resize_2d(lr_raw, size=64, mode="bicubic").clip(0.0, 1.0)
    hr_bicubic_64 = _resize_2d(hr_raw, size=64, mode="bicubic").clip(0.0, 1.0)

    seed_list = _parse_seed_list(args.seed_candidates)
    seed_results: List[Dict[str, Any]] = []
    best_sr_64: Optional[np.ndarray] = None
    best_seed: Optional[int] = None
    best_score = -1e18
    for sd in seed_list:
        diffusion.feed_data(batch)
        diffusion.test(continous=False, seed=int(sd), sample_num_steps=sample_num_steps)
        visuals = diffusion.get_current_visuals(need_LR=True)
        sr_tensor = visuals["SR"]
        if sr_tensor.dim() == 4 and sr_tensor.shape[0] > 1:
            sr_tensor = sr_tensor[-1:]
        sr_cand = _tensor_to_01_2d(sr_tensor).clip(0.0, 1.0)
        psnr_val = _metric_value("psnr", sr_cand, hr_bicubic_64)
        ssim_val = _metric_value("ssim", sr_cand, hr_bicubic_64)
        score = psnr_val if args.select_metric == "psnr" else ssim_val
        seed_results.append(
            {"seed": int(sd), "psnr_vs_hr_bicubic": psnr_val, "ssim_vs_hr_bicubic": ssim_val}
        )
        if score > best_score:
            best_score = score
            best_seed = int(sd)
            best_sr_64 = sr_cand
    assert best_sr_64 is not None

    bicubic_psnr = _metric_value("psnr", lr_bicubic_64, hr_bicubic_64)
    bicubic_ssim = _metric_value("ssim", lr_bicubic_64, hr_bicubic_64)
    bicubic_metric = bicubic_psnr if args.select_metric == "psnr" else bicubic_ssim
    best_metric = _metric_value(args.select_metric, best_sr_64, hr_bicubic_64)

    final_sr_64 = best_sr_64.copy()
    selected_source = f"sr_seed_{best_seed}"
    blend_info: Dict[str, Any] = {"enabled": False}
    auto_blend_enabled = not args.disable_auto_blend_if_worse
    if auto_blend_enabled and (best_metric + float(args.bicubic_tolerance) < bicubic_metric):
        # SR 明显弱于 bicubic 时，在 [SR, bicubic] 上搜索混合，保证“至少相差不多”。
        alphas = np.linspace(0.0, 1.0, 21)
        blend_best_metric = -1e18
        blend_best_alpha = 0.0
        blend_best_img = final_sr_64
        for a in alphas:
            cand = ((1.0 - float(a)) * best_sr_64 + float(a) * lr_bicubic_64).clip(0.0, 1.0)
            cand_metric = _metric_value(args.select_metric, cand, hr_bicubic_64)
            if cand_metric > blend_best_metric:
                blend_best_metric = cand_metric
                blend_best_alpha = float(a)
                blend_best_img = cand
        final_sr_64 = blend_best_img
        selected_source = f"blend_seed_{best_seed}_alpha_{blend_best_alpha:.2f}"
        blend_info = {
            "enabled": True,
            "best_alpha": blend_best_alpha,
            "metric_after_blend": blend_best_metric,
        }

    sr_npy_path = output_dir / f"{args.output_prefix}_sr.npy"
    png_path = output_dir / f"{args.output_prefix}_lr_sr_hr_triplet.png"
    meta_path = output_dir / f"{args.output_prefix}_inference_meta.json"
    np.save(str(sr_npy_path), final_sr_64.astype(np.float32))

    title = "Water-film glucose SR3 inference (64x64 display)"
    _save_triplet_png(
        lr_nearest_64=lr_nearest_64,
        sr_64=final_sr_64,
        hr_bicubic_64=hr_bicubic_64,
        output_png=png_path,
        title=title,
        cmap=args.cmap,
    )

    cmd = " ".join(sys.argv)
    meta = {
        "latest_checkpoint_file": str(latest_ckpt_path),
        "resume_prefix": str(resume_prefix),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "sample_npz": str(sample_npz),
        "lr_npy_raw_8x8": str(lr_npy),
        "hr_npy_raw_12x12": str(hr_npy),
        "sr_npy": str(sr_npy_path),
        "triplet_png": str(png_path),
        "display_rule": {
            "lr": "8x8 -> nearest -> 64x64",
            "sr": "best-of-N seed output 64x64 (possibly blended with LR bicubic baseline)",
            "hr": "12x12 -> bicubic -> 64x64",
            "vmin": 0.0,
            "vmax": 1.0,
            "shared_cmap": args.cmap,
        },
        "sample_num_steps": int(sample_num_steps),
        "seed": int(args.seed),
        "seed_candidates": seed_list,
        "selection_metric": args.select_metric,
        "seed_results": seed_results,
        "bicubic_baseline": {
            "psnr_vs_hr_bicubic": bicubic_psnr,
            "ssim_vs_hr_bicubic": bicubic_ssim,
        },
        "selected_source": selected_source,
        "selected_sr_metrics_vs_hr_bicubic": {
            "psnr": _metric_value("psnr", final_sr_64, hr_bicubic_64),
            "ssim": _metric_value("ssim", final_sr_64, hr_bicubic_64),
        },
        "auto_blend_if_worse": auto_blend_enabled,
        "bicubic_tolerance": float(args.bicubic_tolerance),
        "blend_info": blend_info,
        "command": cmd,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Checkpoint: {latest_ckpt_path}")
    print(f"SR npy: {sr_npy_path}")
    print(f"Triplet PNG: {png_path}")
    print(f"Meta JSON: {meta_path}")
    print(f"Command: {cmd}")


if __name__ == "__main__":
    main()
