#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize SR3 / MRSI super-resolution training outputs as pseudo-color comparison grids.

Scans a run directory for image/array files whose stems match: <base>_lr, <base>_sr, <base>_hr
(e.g. 50000_0_lr.png, 50000_0_sr.png, 50000_0_hr.png from sr.py validation dumps).

迭代选择（自动）:
  - 递归查找各子目录下的 ``{iter}_metrics.json``（与 sr.py 验证阶段一致），按 **PSNR 最大** 取对应 iter；
  - PSNR 相同时取 **iter 更大**（更靠后的验证）；
  - 若无 metrics 文件，则在所有候选图中根据文件名前缀 ``{iter}_`` 取 **最大 iter**。

默认输出: ``<run_dir>/graph/comparison_grid.png`` 。
图顶标题仅为 iter 与验证指标；仅当 LR/HR 像素尺寸不同，或能在 run_dir 内 json
配置中读到 ``l_resolution`` ≠ ``r_resolution`` 时，才追加简短 SR 倍率说明。

Dependencies: numpy, matplotlib, Pillow (PIL), json (stdlib).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except ImportError as e:
    print("错误: 需要安装 Pillow。请执行: pip install Pillow", file=sys.stderr)
    raise SystemExit(1) from e

# 支持的扩展名（小写，含点）
_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}

# 匹配 stem：公共前缀 + _lr / _sr / _hr（不区分大小写）
_TRIPLET_STEM_RE = re.compile(r"^(?P<base>.+)_(?P<kind>lr|sr|hr)$", re.IGNORECASE)

# sr.py 写入的验证指标：{iter}_metrics.json
_METRICS_STEM_RE = re.compile(r"^(?P<iter>\d+)_metrics$", re.IGNORECASE)

# base 形如 ``50000_0`` → 迭代步 50000
_BASE_ITER_RE = re.compile(r"^(?P<iter>\d+)_(?P<rest>.+)$")

# 禁止使用的灰度色图（matplotlib 内置名）
_GRAY_NAMES = frozenset(
    {
        "gray",
        "grey",
        "Greys",
        "gist_gray",
        "gist_yerg",
    }
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="将 SR3 训练结果目录中的 LR/SR/HR 三联样本拼成伪彩对比总图并保存。"
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="训练结果目录（例如实验文件夹或含 png/npy 的子目录）。",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 PNG 路径；默认写入 <run_dir>/graph/comparison_grid.png。",
    )
    p.add_argument(
        "--iter",
        type=int,
        default=None,
        metavar="STEP",
        help="强制只可视化该训练迭代步（覆盖自动选最佳/最后）。文件名前缀须为 {STEP}_* 。",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help="最多可视化多少组完整三元组（默认: 8）。",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="matplotlib 色图名称（默认: turbo）。不可使用 gray/grey 等灰度色图。",
    )
    p.add_argument(
        "--norm_mode",
        type=str,
        choices=("per_triplet", "global"),
        default="per_triplet",
        help="归一化: per_triplet=每组 lr/sr/hr 联合 min-max; global=所有选中样本联合 min-max。",
    )
    p.add_argument(
        "--dpi",
        type=float,
        default=150.0,
        help="保存图像 DPI（默认: 150）。",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="与 --iter 联用：若在全树递归后仍未找到文件，再只扫描 run_dir 顶层目录。",
    )
    p.add_argument(
        "--colorbar",
        action="store_true",
        help="为每个子图显示 colorbar（默认关闭以保持紧凑）。",
    )
    p.add_argument(
        "--fig_width",
        type=float,
        default=12.0,
        help="总图宽度（英寸），默认 12。",
    )
    return p.parse_args()


def _find_metrics_json_files(run_dir: Path) -> List[Path]:
    """递归查找 ``<iter>_metrics.json``。"""
    out: List[Path] = []
    for p in run_dir.rglob("*.json"):
        if _METRICS_STEM_RE.match(p.stem):
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def _pick_best_iter_from_metrics(
    json_paths: List[Path],
) -> Optional[Tuple[int, Path, float]]:
    """
    按验证集平均 PSNR 最大选取 iter；PSNR 相同则取 iter 更大者。
    返回 (iter, metrics_json_path, psnr) ；无法解析时返回 None。
    """
    best_psnr = float("-inf")
    best_it = -1
    best_path: Optional[Path] = None
    for p in json_paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"警告: 无法读取 metrics 文件 {p}: {e}", file=sys.stderr)
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


def _iter_from_base(base: str) -> Optional[int]:
    m = _BASE_ITER_RE.match(base)
    if not m:
        return None
    return int(m.group("iter"))


def _try_load_metrics_dict(run_dir: Path, it: int) -> Optional[Dict[str, Any]]:
    """在 run_dir 下递归查找 ``{it}_metrics.json`` 并解析。"""
    name = f"{it}_metrics.json"
    for p in sorted(run_dir.rglob(name), key=lambda x: str(x).lower()):
        if p.name != name:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


# (json_key, 显示标签, format)
_METRIC_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("psnr", "PSNR", "{:.4f}"),
    ("ssim", "SSIM", "{:.4f}"),
    ("lpips", "LPIPS", "{:.4f}"),
    ("hfen", "HFEN", "{:.4f}"),
    ("frc_aucw", "FRC_AUCw", "{:.4f}"),
    ("frc_hf", "FRC_HF", "{:.4f}"),
    ("frc_cut", "FRC_cut", "{:.4f}"),
)


def _pixel_sr_suffix_if_different(lr: np.ndarray, hr: np.ndarray) -> str:
    """
    仅当 LR/HR 的 H×W 不一致时返回简短倍率说明（同尺寸则返回空串）。
    """
    h0, w0 = int(lr.shape[0]), int(lr.shape[1])
    h1, w1 = int(hr.shape[0]), int(hr.shape[1])
    if (h0, w0) == (h1, w1):
        return ""
    if h0 <= 0 or w0 <= 0:
        return f"  SR {h0}×{w0}→{h1}×{w1}"
    sx = h1 / float(h0)
    sy = w1 / float(w0)
    if abs(sx - sy) < 1e-2 and sx > 1e-6:
        rn = round(sx)
        if abs(sx - rn) < 1e-2 and rn >= 2:
            return f"  SR ×{int(rn)} ({h0}×{w0}→{h1}×{w1})"
    return f"  SR {h0}×{w0}→{h1}×{w1}"


def _config_lr_hr_scale_suffix(run_dir: Path) -> str:
    """
    在 run_dir 下尝试读取训练 json（跳过 metrics），若 datasets.* 中
    l_resolution != r_resolution 则给出像素级倍率说明；无法判断则返回空串。
    """
    for p in sorted(run_dir.rglob("*.json"), key=lambda x: str(x).lower())[:48]:
        if _METRICS_STEM_RE.match(p.stem):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        dsets = data.get("datasets")
        if not isinstance(dsets, dict):
            continue
        for phase in ("val", "train"):
            ds = dsets.get(phase)
            if not isinstance(ds, dict):
                continue
            lo = ds.get("l_resolution")
            hi = ds.get("r_resolution")
            if not isinstance(lo, int) or not isinstance(hi, int):
                continue
            if lo <= 0 or hi <= 0 or lo == hi:
                continue
            if hi % lo == 0:
                return f"  SR ×{hi // lo} ({lo}→{hi}px, config)"
            return f"  SR {lo}→{hi}px (config)"
    return ""


def _figure_header_text(
    chosen_iter: int,
    metrics: Optional[Dict[str, Any]],
    lr: np.ndarray,
    hr: np.ndarray,
    run_dir: Path,
) -> str:
    """图顶单行：iter + 各指标；仅当能推断像素倍率时追加 SR 片段。"""
    parts: List[str] = [f"iter={chosen_iter}"]
    if metrics:
        for key, label, fmt in _METRIC_FIELDS:
            v = metrics.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            parts.append(f"{label}=" + fmt.format(fv))
    extra = _pixel_sr_suffix_if_different(lr, hr)
    if not extra:
        extra = _config_lr_hr_scale_suffix(run_dir)
    if extra:
        parts.append(extra.strip())
    return "    ".join(parts)


def _collect_files(scan_root: Path, recursive: bool) -> List[Path]:
    if not scan_root.is_dir():
        raise FileNotFoundError(f"目录不存在或不是文件夹: {scan_root.resolve()}")

    files: List[Path] = []
    if recursive:
        for ext in _EXTENSIONS:
            files.extend(scan_root.rglob(f"*{ext}"))
    else:
        for ext in _EXTENSIONS:
            files.extend(scan_root.glob(f"*{ext}"))

    # 去重、排序
    return sorted(set(files), key=lambda p: str(p).lower())


def _max_iter_from_triplet_files(paths: List[Path]) -> Optional[int]:
    """从已匹配为三元组候选的文件名中推断出现的最大迭代步。"""
    steps: List[int] = []
    for p in paths:
        sk = _stem_kind(p)
        if sk is None:
            continue
        base, _ = sk
        it = _iter_from_base(base)
        if it is not None:
            steps.append(it)
    return max(steps) if steps else None


def _filter_paths_by_iter(paths: List[Path], chosen_iter: int) -> List[Path]:
    """只保留 stem 对应 base 以 ``{chosen_iter}_`` 开头的文件。"""
    prefix = f"{chosen_iter}_"
    out: List[Path] = []
    for p in paths:
        sk = _stem_kind(p)
        if sk is None:
            continue
        base, _ = sk
        if base.startswith(prefix):
            out.append(p)
    return out


def _resolve_iter_and_files(
    run_dir: Path,
    recursive: bool,
    force_iter: Optional[int],
) -> Tuple[int, List[Path], str, Optional[Dict[str, Any]]]:
    """
    确定要可视化的迭代步，并返回对应文件列表、日志说明、以及该步的 metrics 字典（若存在）。
    """
    if force_iter is not None:
        it = int(force_iter)
        all_p = _collect_files(run_dir, recursive=True)
        filtered = _filter_paths_by_iter(all_p, it)
        if not filtered:
            all_p = _collect_files(run_dir, recursive=recursive)
            filtered = _filter_paths_by_iter(all_p, it)
        if not filtered:
            raise FileNotFoundError(
                f"未找到迭代 {it} 的 *_lr/_sr/_hr 文件；可尝试 --recursive 或检查目录。"
            )
        mdict = _try_load_metrics_dict(run_dir, it)
        return it, filtered, f"用户指定 --iter {it}", mdict

    metrics_files = _find_metrics_json_files(run_dir)
    picked = _pick_best_iter_from_metrics(metrics_files)
    if picked is not None:
        it, mpath, psnr = picked
        mdict: Optional[Dict[str, Any]] = None
        try:
            with open(mpath, encoding="utf-8") as f:
                mdict = json.load(f)
        except (OSError, json.JSONDecodeError):
            mdict = None
        scan_root = mpath.parent
        files = _collect_files(scan_root, recursive=False)
        files = _filter_paths_by_iter(files, it)
        note = f"metrics 最佳 PSNR={psnr:.6g} @ iter={it}（{mpath.name}）"
        if not files:
            files = _collect_files(run_dir, recursive=True)
            files = _filter_paths_by_iter(files, it)
            note += "；已在整棵 run_dir 下回退搜索图像"
        if not files:
            raise FileNotFoundError(
                f"metrics 指向 iter={it}，但未找到对应三联图文件。"
            )
        return it, files, note, mdict

    # 无 metrics：用文件名中最大 iter（尽量“最后”的验证图）
    all_p = _collect_files(run_dir, recursive=True)
    mx = _max_iter_from_triplet_files(all_p)
    if mx is None:
        raise FileNotFoundError(
            "未找到任何符合 *_lr/_sr/_hr 的文件，且没有可用的 *_metrics.json。"
        )
    filtered = _filter_paths_by_iter(all_p, mx)
    mdict = _try_load_metrics_dict(run_dir, mx)
    return (
        mx,
        filtered,
        f"无 metrics json，使用文件名中最大 iter={mx}（已 recursive 扫描）",
        mdict,
    )


def _stem_kind(path: Path) -> Optional[Tuple[str, str]]:
    """若文件名符合 <base>_(lr|sr|hr)，返回 (base, kind)；否则 None。"""
    stem = path.stem
    m = _TRIPLET_STEM_RE.match(stem)
    if not m:
        return None
    return m.group("base"), m.group("kind").lower()


def _group_triplets(paths: List[Path]) -> Dict[str, Dict[str, Path]]:
    """
    base -> {'lr': path, 'sr': path, 'hr': path}（可能缺键）
    同一 base 若同一 kind 出现多个文件，保留路径字典序第一个并警告。
    """
    groups: Dict[str, Dict[str, Path]] = {}
    for p in paths:
        sk = _stem_kind(p)
        if sk is None:
            continue
        base, kind = sk
        g = groups.setdefault(base, {})
        if kind in g:
            warnings.warn(
                f"重复样本键: base={base!r}, kind={kind!r}，保留 {g[kind]!s}，忽略 {p!s}",
                UserWarning,
                stacklevel=2,
            )
            continue
        g[kind] = p
    return groups


def _to_2d_float(arr: np.ndarray, src: Path) -> np.ndarray:
    """将数组压成 2D float64，便于 imshow + colormap。"""
    a = np.asarray(arr)
    if a.dtype == np.object_:
        raise ValueError(f"无法将对象数组转为图像: {src}")

    a = np.squeeze(a)
    if a.ndim == 0:
        raise ValueError(f"标量数组无法可视化: {src}")
    if a.ndim == 1:
        raise ValueError(f"一维数组无法作为 2D 热图显示: {src}")

    if a.ndim == 2:
        return a.astype(np.float64, copy=False)

    if a.ndim == 3:
        c0, c1, c2 = a.shape
        # PIL / 常见保存: (H, W, C)，C 较小
        if c2 in (1, 3, 4) and c2 <= 8 and c0 >= c2 and c1 >= c2:
            plane = a[..., :3] if c2 >= 3 else a[..., 0]
            if plane.ndim == 3:
                plane = np.mean(plane.astype(np.float64), axis=-1)
            return plane
        # (C, H, W)，C 为通道
        if c0 in (1, 3, 4) and c0 <= 8 and c1 > c0 and c2 > c0:
            plane = a[:3] if c0 >= 3 else a[0:1]
            if plane.ndim == 3:
                plane = np.mean(plane.astype(np.float64), axis=0)
            else:
                plane = plane[0]
            return plane.astype(np.float64, copy=False)
        # 回退：沿最短轴若为通道维则压缩
        axis = int(np.argmin(a.shape))
        if a.shape[axis] <= 8:
            return np.mean(a.astype(np.float64), axis=axis)
        raise ValueError(f"无法推断 3D 数组的显示平面 (shape={a.shape}): {src}")

    while a.ndim > 2:
        a = a[0]
    return a.astype(np.float64, copy=False)


def _load_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        data = np.load(path, allow_pickle=False)
        return _to_2d_float(data, path)
    try:
        im = Image.open(path)
        arr = np.asarray(im)
    except Exception as e:
        raise RuntimeError(f"无法读取图像: {path}") from e
    return _to_2d_float(arr, path)


def _load_triplet(
    base: str, paths: Dict[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr = _load_image(paths["lr"])
    sr = _load_image(paths["sr"])
    hr = _load_image(paths["hr"])
    return lr, sr, hr


def _global_vmin_vmax(
    triplets: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[float, float]:
    stack = []
    for _, lr, sr, hr in triplets:
        stack.extend([lr.min(), lr.max(), sr.min(), sr.max(), hr.min(), hr.max()])
    vmin, vmax = float(np.min(stack)), float(np.max(stack))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("全局 vmin/vmax 非有限值，请检查数据是否含 NaN/Inf。")
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def _per_triplet_vmin_vmax(
    lr: np.ndarray, sr: np.ndarray, hr: np.ndarray
) -> Tuple[float, float]:
    vmin = float(min(lr.min(), sr.min(), hr.min()))
    vmax = float(max(lr.max(), sr.max(), hr.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("vmin/vmax 非有限值（NaN/Inf）。")
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def main() -> None:
    args = _parse_args()
    cmap_name = args.cmap.strip()
    if cmap_name.lower() in {x.lower() for x in _GRAY_NAMES}:
        print(
            f"错误: 色图 '{cmap_name}' 为灰度图，本脚本要求伪彩色。请改用 turbo、viridis 等。",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"错误: matplotlib 无法识别色图名称: {cmap_name!r}", file=sys.stderr)
        sys.exit(2)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"错误: 目录不存在: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        out_path = run_dir / "graph" / "comparison_grid.png"
    else:
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path

    try:
        chosen_iter, all_files, iter_note, metrics_dict = _resolve_iter_and_files(
            run_dir, recursive=args.recursive, force_iter=args.iter
        )
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"选用迭代 iter={chosen_iter}  |  {iter_note}")

    if not all_files:
        print(
            f"错误: iter={chosen_iter} 下没有匹配的图像文件。",
            file=sys.stderr,
        )
        sys.exit(1)

    groups = _group_triplets(all_files)
    complete: List[Tuple[str, Dict[str, Path]]] = []
    for base in sorted(groups.keys()):
        g = groups[base]
        if "lr" in g and "sr" in g and "hr" in g:
            complete.append((base, g))
        else:
            missing = [k for k in ("lr", "sr", "hr") if k not in g]
            print(
                f"警告: 跳过不完整三元组 base={base!r}，缺少: {', '.join(missing)}",
                file=sys.stderr,
            )

    if not complete:
        print(
            "错误: 没有找到任何完整的 lr+sr+hr 三元组。请检查文件名是否包含 _lr / _sr / _hr 后缀。",
            file=sys.stderr,
        )
        sys.exit(1)

    complete = complete[: max(0, args.max_samples)]
    n = len(complete)
    if n == 0:
        print("错误: max_samples=0，无可视化内容。", file=sys.stderr)
        sys.exit(1)

    # 加载数据
    loaded: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for base, g in complete:
        try:
            lr, sr, hr = _load_triplet(base, g)
        except Exception as e:
            print(f"警告: 加载失败，跳过 base={base!r}: {e}", file=sys.stderr)
            continue
        loaded.append((base, lr, sr, hr))

    if not loaded:
        print("错误: 所有候选三元组在加载时均失败。", file=sys.stderr)
        sys.exit(1)

    g_vmin: Optional[float] = None
    g_vmax: Optional[float] = None
    if args.norm_mode == "global":
        g_vmin, g_vmax = _global_vmin_vmax(loaded)

    _, lr0, _, hr0 = loaded[0]
    header_text = _figure_header_text(
        chosen_iter, metrics_dict, lr0, hr0, run_dir
    )

    n_rows = len(loaded)
    n_cols = 3
    fig_w = float(args.fig_width)
    # 与 subplots_adjust 一致；先定边距再算 fig_h，使每个子图格接近正方形。
    # 否则宽扁格子 + imshow 默认 aspect=equal 会在左右留大量空白，看起来像列间距过大。
    if args.colorbar:
        _L, _R, _T, _B = 0.002, 0.90, 0.92, 0.002
    else:
        _L, _R, _T, _B = 0.002, 0.998, 0.92, 0.002
    _aw, _ah = _R - _L, _T - _B
    fig_h = fig_w * (_aw / max(_ah, 1e-6)) * (n_rows / float(n_cols)) * 1.02
    fig_h = max(2.0, fig_h)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=False,
    )
    # wspace/hspace 同值；格子已为近似正方形时，列间/行间视觉留白一致。
    _gap = 0.006 if not args.colorbar else 0.05
    fig.subplots_adjust(
        left=_L,
        right=_R,
        top=_T,
        bottom=_B,
        wspace=_gap,
        hspace=_gap,
    )

    for r, (_, lr, sr, hr) in enumerate(loaded):
        if args.norm_mode == "per_triplet":
            vmin, vmax = _per_triplet_vmin_vmax(lr, sr, hr)
        else:
            vmin, vmax = g_vmin, g_vmax  # type: ignore

        panels = (lr, sr, hr)
        for c, data in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_axis_off()
            if args.colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(header_text, fontsize=9, y=0.995, va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
