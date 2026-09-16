#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水膜 CSI 输出目录中的 ``peak_area_map.csv``（或同类方阵 CSV）与 SR3 训练权重的衔接脚本。

结论（默认行为）
----------------
若 **仅** 提供 Scan 26 的 8×8 CSV，而没有任何与之一一对应的 **同场、同网格定义下的高分真值**，
则脚本会向 stderr 打印原因并 ``sys.exit(1)``，**不生成**「LR / SR / HR」三联图。

原因简述
--------
- ``peak_area_map.csv`` 只有在提供固定、可复用的定量尺度后才可送入网络；脚本不再逐图 min-max。
- Scan 27 的 12×12 与 Scan 26 的 8×8 为 **不同扫描/不同 CSI 矩阵**，体素划分与 FOV 未必一致，
  **不能**把 27 的图当作 26 的 HR 真值做定量对比。

可选：``--ref_bicubic`` 生成 **LR（最近邻展示）| SR | 双三次上采样** 三联图，第三幅明确 **不是 HR**，
仅供组会定性展示扩散模型相对传统上采样的观感。

用法（仓库根目录）::

    # 若有自仿真或其它来源的配对准 HR 方阵 CSV（与 LR 同一物理场景、可配准到同一网格）：
    python scripts/phantom_peakmap_sr3_compare.py \\
        --lr_csv \"D:/.../scan_26_8x8_spatial_ifft_peak_area_map.csv\" \\
        --hr_csv \"D:/path/to/paired_hr.csv\" \\
        --run_dir experiments/models/healthy_phantom_i300000_ema \\
        --config config/sr3_mrsi_64_mvp.json \\
        --output graph/phantom_compare.png

    # 仅定性对比（第三幅为双三次，非真值 HR）：
    python scripts/phantom_peakmap_sr3_compare.py \\
        --lr_csv \"D:/.../scan_26_8x8_spatial_ifft_peak_area_map.csv\" \\
        --ref_bicubic \\
        --run_dir experiments/models/healthy_phantom_i300000_ema \\
        --config config/sr3_mrsi_64_mvp.json \\
        --output graph/phantom_lr_sr_bicubic.png
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_infer_module() -> Any:
    p = _REPO_ROOT / "scripts" / "infer_best_mrsi_compare.py"
    spec = importlib.util.spec_from_file_location("_sr3_infer_compare", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_square_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.float64, delimiter=",")
    if arr.ndim != 2:
        raise ValueError(f"{path} 期望二维矩阵，得到 shape={arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path} 期望方阵，得到 shape={arr.shape}")
    return arr.astype(np.float32)


def _up01_nearest(img01: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(img01).float().view(1, 1, img01.shape[0], img01.shape[1])
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze().numpy().clip(0.0, 1.0)


def _up01_bicubic(img01: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(img01).float().view(1, 1, img01.shape[0], img01.shape[1])
    t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.squeeze().numpy().clip(0.0, 1.0)


PROBLEM_ZH = """
【无法生成「LR / SR / HR」三联对比图】

已查看你给出的两类输出（以 peak_area_map.csv 为例）：
  • Scan 26：8×8 方阵 — 可作为「低分/稀疏采样」强度图做推理输入。
  • Scan 27：12×12 方阵 — 属于另一次采集、另一套 CSI 空间矩阵，与 26 的 8×8 在几何上
    **不是**「同一视场、同一网格上的高分真值」关系，因此 **不能**作为 Scan 26 的 HR 真值
    与扩散模型输出做严格的 LR/SR/HR 对比。

当前目录下也未发现与 8×8 同尺寸或可配准到同一网格的「同次扫描 HR」CSV。

你可以任选其一继续：
  1) 使用仿真/课题管线生成的 **成对** HR/LR（例如 .npz），用
     ``scripts/infer_best_mrsi_compare.py --input_npz ...``；
  2) 若已有 **配准好的** HR 方阵 CSV，对本脚本同时传 ``--lr_csv`` 与 ``--hr_csv``；
  3) 若仅需组会 **定性** 展示，可加 ``--ref_bicubic``：第三幅为双三次上采样（非 HR），
     第一幅 LR 为最近邻放大以体现原始 8×8 网格。

本消息输出后进程将退出（exit code 1）。
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="水膜 peak_area CSV 与 SR3 对比图（默认无 HR 则退出）")
    parser.add_argument("--lr_csv", type=str, required=True, help="例如 scan_26_8x8_spatial_ifft_peak_area_map.csv")
    parser.add_argument("--hr_csv", type=str, default=None, help="与 LR 配对准的高分方阵 CSV（可选）")
    parser.add_argument(
        "--ref_bicubic",
        action="store_true",
        help="无 HR 时仍出图：第三幅为双三次上采样（非真值），第一幅 LR 用最近邻上采样展示",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_prefix", type=str, default=None)
    parser.add_argument("--output", type=str, default="phantom_peakmap_compare.png")
    parser.add_argument("--normalization_contract", type=str, default=None)
    parser.add_argument("--metabolite", choices=("HDO", "Glc", "Glx", "Lac"), default="HDO")
    parser.add_argument("--normalization_scale", type=float, default=None)
    parser.add_argument("--input_normalized", action="store_true")
    parser.add_argument("--quantity_unit", type=str, default=None)
    parser.add_argument("--numeric_output", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_num_steps", type=int, default=None)
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--dpi", type=float, default=150.0)
    args = parser.parse_args()

    if not args.hr_csv and not args.ref_bicubic:
        print(PROBLEM_ZH, file=sys.stderr)
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("phantom_peakmap")

    ibc = _load_infer_module()
    import model as Model

    lr_path = Path(args.lr_csv)
    if not lr_path.is_file():
        raise FileNotFoundError(lr_path)
    lr_np = _load_square_csv(lr_path)

    hr_np: Optional[np.ndarray] = None
    if args.hr_csv:
        hr_p = Path(args.hr_csv)
        if not hr_p.is_file():
            raise FileNotFoundError(hr_p)
        hr_np = _load_square_csv(hr_p)

    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    metrics_best_iter: Optional[int] = None
    metrics_note = ""
    picked = ibc._pick_best_iter_from_metrics(run_dir)
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
        resume_prefix, ckpt_msg = ibc._select_checkpoint_prefix(run_dir, metrics_best_iter)
    log.info(ckpt_msg)

    if args.cpu:
        gpu_ids_arg = ""
    else:
        gpu_ids_arg = args.gpu_ids.strip() if args.gpu_ids is not None else "0"
    opt = ibc._build_opt(config_path, run_dir, resume_prefix, gpu_ids_arg)

    image_size = int(opt["model"]["diffusion"]["image_size"])
    val_cfg = opt.get("validation", {}) or {}
    sample_num_steps = args.sample_num_steps
    if sample_num_steps is None:
        sample_num_steps = val_cfg.get("sample_num_steps")
    if sample_num_steps is None:
        sample_num_steps = opt["model"]["diffusion"].get("sample_num_steps")

    # 送入网络的 HR 张量：有真值 HR 用 hr_np；否则用 LR 自身（推理不读取 HR，仅占位）
    hr_for_model = hr_np if hr_np is not None else lr_np
    val_ds = dict(opt.get("datasets", {}).get("val", {}) or {})

    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    dev = diffusion.device

    contract = ibc.load_normalization_contract(args.normalization_contract) if args.normalization_contract else None
    scale = args.normalization_scale
    if scale is None and contract is not None:
        scale = contract["normalization_scales"][args.metabolite]
    if scale is None:
        if args.input_normalized:
            scale = 1.0
        else:
            raise SystemExit(
                "定量 CSV 禁止逐图 min-max。请提供固定 --normalization_scale（实测峰面积通常需先做标定），"
                "或对已归一化数据显式加 --input_normalized。"
            )
    quantity_unit = args.quantity_unit
    if quantity_unit is None and contract is not None:
        quantity_unit = contract["quantity_unit"]
    quantity_unit = quantity_unit or "quantity_unit"
    batch = ibc._batch_from_lr_hr_numpy(
        lr_np,
        hr_for_model,
        image_size,
        dev,
        val_ds,
        normalization_scale=float(scale),
        input_normalized=bool(args.input_normalized),
        quantity_unit=quantity_unit,
        met_id=ibc._MET_TO_ID[args.metabolite],
    )
    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=int(args.seed), sample_num_steps=sample_num_steps)
    visuals = diffusion.get_current_visuals(need_LR=True)

    sr_t = visuals["SR"]
    if sr_t.dim() == 4 and sr_t.shape[0] > 1:
        sr_t = sr_t[-1:]

    lr01_small = ibc._as_normalized_01(
        lr_np, scale=float(scale), input_normalized=bool(args.input_normalized)
    )
    lr_plot = _up01_nearest(lr01_small, image_size)
    sr_plot = ibc._tensor01_for_plot(sr_t)

    if hr_np is not None:
        hr_plot = ibc._tensor01_for_plot(visuals["HR"])
        labels: Tuple[str, str, str] = ("LR (最近邻展示)", "SR (扩散)", "HR (配对准 CSV)")
        title_extra = "三联图含用户提供的 HR CSV"
    else:
        hr_plot = _up01_bicubic(lr01_small, image_size)
        labels = ("LR (最近邻展示)", "SR (扩散)", "参考: 双三次 (非 HR)")
        title_extra = "第三幅非真值 HR，仅与双三次上采样定性对比"

    title_parts = [ckpt_msg, title_extra]
    if metrics_note:
        title_parts.append(metrics_note)
    title = " | ".join(title_parts)

    out_path = Path(args.output)
    lr_quantity = ibc.denormalize_quantity(lr_plot, float(scale))
    sr_quantity = ibc.denormalize_quantity(sr_plot, float(scale))
    ref_quantity = ibc.denormalize_quantity(hr_plot, float(scale))
    ibc._save_triplet_figure(
        lr_quantity,
        sr_quantity,
        ref_quantity,
        out_path,
        title=title,
        cmap=args.cmap,
        dpi=args.dpi,
        panel_labels=labels,
        colorbar_label=quantity_unit,
    )
    numeric_path = (
        Path(args.numeric_output)
        if args.numeric_output
        else out_path.with_name(f"{out_path.stem}_quantitative.npz")
    )
    numeric_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        numeric_path,
        lr_quantity=lr_quantity.astype(np.float32),
        sr_quantity=sr_quantity.astype(np.float32),
        reference_quantity=ref_quantity.astype(np.float32),
        normalization_scale=np.float32(scale),
        quantity_unit=np.array(quantity_unit),
        voxel_semantics=np.array("intensive"),
        reference_is_hr=np.bool_(hr_np is not None),
        checkpoint_prefix=np.array(str(resume_prefix)),
    )
    log.info("已保存: %s", out_path.resolve())
    log.info("已保存浮点定量结果: %s", numeric_path.resolve())


if __name__ == "__main__":
    main()
