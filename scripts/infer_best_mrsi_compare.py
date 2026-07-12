#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用某次训练实验中「验证集平均 PSNR 最优」的 checkpoint，对单张样本做 SR3 推理，
并保存 LR / SR / HR 三联对比图。

典型用法（在仓库根目录执行）::

    python scripts/infer_best_mrsi_compare.py \\
        --run_dir experiments/sr3_mrsi_64_mvp_260428_231833 \\
        --config config/sr3_mrsi_64_mvp.json \\
        --input_npz dataset_mrsi/mrsi_sr3_64/val/data/xxx.npz \\
        --output compare.png

若实验目录下没有 ``*_metrics.json``，则自动选用 checkpoint 目录中迭代步最大的 ``I*_gen.pth``。

输入说明
--------
- **推荐**：``--input_npz`` 使用与训练一致的 .npz（含 hr/lr/t1/flair/met_onehot/mask 等），HR 为真值。
- **备选**：``--lr`` 与 ``--hr`` 各一张灰度图或 .npy；会双三次插值到模型 ``image_size``，
  并按配置用 LR 复制占位 T1/FLAIR、met_onehot 默认全图 Glc（通道 0），仅适合快速可视化，
  与训练分布可能不一致。

依赖：torch、numpy、matplotlib、Pillow；项目内 model/data/core 模块。
"""

from __future__ import annotations

import argparse
import json
import logging
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

# 以仓库根为工作目录，便于 import model / core
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import core.logger as Logger  # noqa: E402
import model as Model  # noqa: E402
import core.metrics as Metrics  # noqa: E402

_METRICS_STEM_RE = re.compile(r"^(?P<iter>\d+)_metrics$", re.IGNORECASE)
_CKPT_GEN_RE = re.compile(r"^I(?P<it>\d+)_E(?P<ep>\d+)_gen\.pth$", re.IGNORECASE)


def _load_json_config(path: Path) -> OrderedDict:
    json_str = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def _pick_best_iter_from_metrics(run_dir: Path) -> Optional[Tuple[int, Path, float]]:
    best_psnr = float("-inf")
    best_it = -1
    best_path: Optional[Path] = None
    for p in sorted(run_dir.rglob("*.json"), key=lambda x: str(x).lower()):
        if not _METRICS_STEM_RE.match(p.stem):
            continue
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        it = data.get("iter")
        psnr = data.get("psnr")
        if it is None or psnr is None:
            continue
        try:
            it_i = int(it)
            psnr_f = float(psnr)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(psnr_f):
            continue
        if psnr_f > best_psnr or (psnr_f == best_psnr and it_i > best_it):
            best_psnr = psnr_f
            best_it = it_i
            best_path = p
    if best_path is None:
        return None
    return best_it, best_path, best_psnr


def _list_checkpoint_iters(checkpoint_dir: Path) -> List[Tuple[int, int, Path]]:
    """返回 (iter, epoch, path_prefix_without__gen.pth) 列表。"""
    out: List[Tuple[int, int, Path]] = []
    if not checkpoint_dir.is_dir():
        return out
    for name in os.listdir(checkpoint_dir):
        m = _CKPT_GEN_RE.match(name)
        if not m:
            continue
        it_i = int(m.group("it"))
        ep_i = int(m.group("ep"))
        prefix = checkpoint_dir / name[: -len("_gen.pth")]
        out.append((it_i, ep_i, prefix))
    out.sort(key=lambda x: x[0])
    return out


def _select_checkpoint_prefix(
    run_dir: Path,
    metrics_best_iter: Optional[int],
) -> Tuple[str, str]:
    """
    返回 (resume_state 前缀路径, 说明字符串)。
    resume_state 格式与训练一致：``.../checkpoint/I123_E4``（无后缀）。
    """
    ckpt_dir = run_dir / "checkpoint"
    cands = _list_checkpoint_iters(ckpt_dir)
    if not cands:
        raise FileNotFoundError(f"在 {ckpt_dir} 下未找到 I*_E*_gen.pth 权重文件。")

    if metrics_best_iter is None:
        it, ep, pref = cands[-1]
        return str(pref), f"无 metrics，使用最大 iter 检查点 I{it}_E{ep}"

    # 优先：iter <= 验证最佳步 的最大检查点（与 val 频率、存盘频率错开时更稳）
    chosen = None
    for it, ep, pref in cands:
        if it <= metrics_best_iter:
            chosen = (it, ep, pref)
    if chosen is None:
        it, ep, pref = cands[0]
        return str(pref), f"警告: 无 iter≤{metrics_best_iter} 的检查点，退回最小 iter I{it}_E{ep}"
    it, ep, pref = chosen
    return str(pref), f"验证最佳 iter={metrics_best_iter}，选用检查点 I{it}_E{ep} (≤ 该 iter 的最大存盘步)"


def _build_opt(
    config_path: Path,
    run_dir: Path,
    resume_prefix: str,
    gpu_ids: Optional[str],
) -> Any:
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
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        ",".join(str(x) for x in opt["gpu_ids"]) if opt["gpu_ids"] else ""
    )
    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["distributed"] = False
    opt["is_main_process"] = True
    return Logger.dict_to_nonedict(opt)


def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _load_image_or_npy(path: Path) -> np.ndarray:
    """返回 2D float32 数组，值域约 [0,1]（按数据 min-max 若超出则先 min-max）。"""
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    else:
        from PIL import Image

        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path} 期望 2D 灰度或 .npy 二维数组，得到 shape={arr.shape}")
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max <= 1.0 + 1e-3 and arr_min >= -1e-3:
        return np.clip(arr, 0.0, 1.0)
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)


def _resize_hw01(x2d: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(x2d).float().view(1, 1, x2d.shape[0], x2d.shape[1])
    t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.clamp(0.0, 1.0)


def _stack_condition_from_parts(
    lr: torch.Tensor,
    t1: torch.Tensor,
    flair: torch.Tensor,
    met_onehot: torch.Tensor,
    mask: torch.Tensor,
    ds_opt: Dict[str, Any],
) -> torch.Tensor:
    """与 ``data/MRSI_SR3_dataset.py`` 中 ``MRSISR3Dataset`` 的通道顺序一致。"""
    parts: List[torch.Tensor] = []
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
        raise ValueError("数据集配置中至少需要一路条件通道（use_lr/use_t1/...）。")
    return torch.cat(parts, dim=1)


def _batch_from_npz(npz_path: Path, device: torch.device, ds_opt: Dict[str, Any]) -> Dict[str, Any]:
    data = np.load(str(npz_path))
    hr = torch.from_numpy(data["hr"]).float().unsqueeze(0)
    lr = torch.from_numpy(data["lr"]).float().unsqueeze(0)
    t1 = torch.from_numpy(data["t1"]).float().unsqueeze(0)
    flair = torch.from_numpy(data["flair"]).float().unsqueeze(0)
    met_onehot = torch.from_numpy(data["met_onehot"]).float().unsqueeze(0)
    mask = torch.from_numpy(data["mask"]).float().unsqueeze(0)
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


def _norm_minmax_01(arr: np.ndarray) -> np.ndarray:
    """物理数值矩阵 → [0,1] float32（逐幅 min-max）。"""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _batch_from_lr_hr_numpy(
    lr_np: np.ndarray,
    hr_np: np.ndarray,
    image_size: int,
    device: torch.device,
    ds_opt: Dict[str, Any],
) -> Dict[str, Any]:
    """与 ``_batch_from_lr_hr_images`` 相同逻辑，输入已为二维物理或 [0,1] 数组。"""
    if lr_np.ndim != 2 or hr_np.ndim != 2:
        raise ValueError(f"LR/HR 须为二维数组，得到 lr={lr_np.shape}, hr={hr_np.shape}")
    lr_u8 = _norm_minmax_01(lr_np)
    hr_u8 = _norm_minmax_01(hr_np)
    lr = _resize_hw01(lr_u8, image_size)
    hr = _resize_hw01(hr_u8, image_size)
    t1 = lr.clone()
    flair = lr.clone()
    _, _, h, w = lr.shape
    met = torch.zeros(1, 4, h, w, device=lr.device, dtype=lr.dtype)
    met[:, 0, :, :] = 1.0
    mask = torch.ones_like(lr)
    cond = _stack_condition_from_parts(lr, t1, flair, met, mask, ds_opt)
    return {
        "HR": _to_minus1_1(hr).to(device),
        "SR": _to_minus1_1(cond).to(device),
        "LR": _to_minus1_1(lr).to(device),
        "MASK": mask.to(device),
    }


def _batch_from_lr_hr_images(
    lr_path: Path,
    hr_path: Path,
    image_size: int,
    device: torch.device,
    ds_opt: Dict[str, Any],
) -> Dict[str, Any]:
    lr_np = _load_image_or_npy(lr_path)
    hr_np = _load_image_or_npy(hr_path)
    return _batch_from_lr_hr_numpy(lr_np, hr_np, image_size, device, ds_opt)


def _tensor01_for_plot(t: torch.Tensor) -> np.ndarray:
    """[-1,1] -> [0,1] 的 HxW numpy。"""
    x = t.detach().float().cpu().squeeze().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    return x.numpy()


def _save_triplet_figure(
    lr01: np.ndarray,
    sr01: np.ndarray,
    hr01: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    dpi: float,
    panel_labels: Tuple[str, str, str] = ("LR", "SR", "HR"),
) -> None:
    vmin = float(min(lr01.min(), sr01.min(), hr01.min()))
    vmax = float(max(lr01.max(), sr01.max(), hr01.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, arr, lab in zip(axes, (lr01, sr01, hr01), panel_labels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(lab)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="[0,1] 归一化幅度")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SR3 MRSI 最优 checkpoint 单样本推理与三联图")
    parser.add_argument("--run_dir", type=str, required=True, help="实验目录（含 checkpoint/ 与 results/）")
    parser.add_argument("--config", type=str, required=True, help="训练用 JSON 配置（与 run 一致）")
    parser.add_argument(
        "--input_npz",
        type=str,
        default=None,
        help="单个 .npz 样本路径（与 MRSI_SR3_dataset 字段一致）",
    )
    parser.add_argument("--lr", type=str, default=None, help="低分图路径（与 --hr 联用）")
    parser.add_argument("--hr", type=str, default=None, help="真值 HR 图路径（与 --lr 联用）")
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default=None,
        help="显式指定权重前缀（同 resume_state），不含 _gen.pth；指定后忽略 metrics 自动选择",
    )
    parser.add_argument("--output", type=str, default="infer_compare_lr_sr_hr.png", help="输出对比图路径")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="例如 0 或 0,1；传空字符串 ``--gpu_ids \"\"`` 或配合 --cpu 使用 CPU",
    )
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（忽略配置文件中的 gpu_ids）")
    parser.add_argument("--seed", type=int, default=0, help="扩散采样随机种子")
    parser.add_argument(
        "--sample_num_steps",
        type=int,
        default=None,
        help="覆盖配置中的 DDIM 步数；默认读 validation.sample_num_steps 或 model.diffusion",
    )
    parser.add_argument("--cmap", type=str, default="turbo", help="matplotlib 色图名")
    parser.add_argument("--dpi", type=float, default=150.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("infer_compare")

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    metrics_best_iter: Optional[int] = None
    metrics_note = ""
    picked = _pick_best_iter_from_metrics(run_dir)
    if picked:
        metrics_best_iter, metrics_json, psnr = picked
        metrics_note = f"metrics 最佳: iter={metrics_best_iter}, PSNR={psnr:.4f} ({metrics_json.name})"
        log.info(metrics_note)
    else:
        log.info("未找到有效的 *_metrics.json，将按 checkpoint 最大 iter 选取。")

    if args.checkpoint_prefix:
        resume_prefix = args.checkpoint_prefix
        ckpt_msg = "使用用户指定 --checkpoint_prefix"
    else:
        resume_prefix, ckpt_msg = _select_checkpoint_prefix(run_dir, metrics_best_iter)
    log.info(ckpt_msg)
    log.info("加载权重前缀: %s", resume_prefix)

    if args.cpu:
        gpu_ids_arg = ""
    else:
        gpu_ids_arg = args.gpu_ids.strip() if args.gpu_ids is not None else "0"
    opt = _build_opt(config_path, run_dir, resume_prefix, gpu_ids_arg)

    image_size = int(opt["model"]["diffusion"]["image_size"])
    val_cfg = opt.get("validation", {}) or {}
    sample_num_steps = args.sample_num_steps
    if sample_num_steps is None:
        sample_num_steps = val_cfg.get("sample_num_steps")
    if sample_num_steps is None:
        sample_num_steps = opt["model"]["diffusion"].get("sample_num_steps")

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")

    val_ds = dict(opt.get("datasets", {}).get("val", {}) or {})

    dev = diffusion.device
    if args.input_npz:
        batch = _batch_from_npz(Path(args.input_npz), dev, val_ds)
    elif args.lr and args.hr:
        batch = _batch_from_lr_hr_images(Path(args.lr), Path(args.hr), image_size, dev, val_ds)
    else:
        raise SystemExit("请提供 --input_npz，或同时提供 --lr 与 --hr。")

    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=int(args.seed), sample_num_steps=sample_num_steps)
    visuals = diffusion.get_current_visuals(need_LR=True)

    sr_t = visuals["SR"]
    if sr_t.dim() == 4 and sr_t.shape[0] > 1:
        sr_t = sr_t[-1:]
    lr01 = _tensor01_for_plot(visuals["LR"])
    sr01 = _tensor01_for_plot(sr_t)
    hr01 = _tensor01_for_plot(visuals["HR"])

    out_path = Path(args.output)
    title_parts = [ckpt_msg]
    if metrics_note:
        title_parts.append(metrics_note)
    title = " | ".join(title_parts)
    _save_triplet_figure(lr01, sr01, hr01, out_path, title=title, cmap=args.cmap, dpi=args.dpi)
    log.info("已保存: %s", out_path.resolve())


if __name__ == "__main__":
    main()
