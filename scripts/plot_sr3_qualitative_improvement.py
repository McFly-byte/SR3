#!/usr/bin/env python3
"""Create a matched baseline-vs-improved SR3 qualitative comparison.

The main metabolite panels follow the ``simulated_with_lesion`` display
convention: viridis, 90-degree counter-clockwise orientation, and a shared
brain-aware 1st--99th percentile window within each sample row.  Signed error
panels use a shared, zero-centred coolwarm scale.

Raw validation PNGs remain untouched; this script only creates presentation
figures and a JSON sidecar describing the display transformation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.dmi_visualization import (  # noqa: E402
    DEFAULT_PERCENTILES,
    DEFAULT_ROTATE_K,
    orient_for_display,
    robust_display_limits,
    signed_error_limit,
    validate_percentiles,
)


_IMAGE_RE = re.compile(
    r"^(?P<iteration>\d+)_(?P<index>\d+)_(?P<kind>lr|hr|sr(?:_\d+)?)$",
    re.IGNORECASE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成原模型/改进模型的同样本、同色标 DMI 超分效果对比图。"
    )
    parser.add_argument("--baseline_run", required=True, help="原模型完整验证实验目录")
    parser.add_argument("--selected_run", required=True, help="改进模型完整验证实验目录")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    parser.add_argument("--baseline_iter", type=int, default=None)
    parser.add_argument("--selected_iter", type=int, default=None)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="指定 validation 行号；默认取前 max_samples 个共同样本。",
    )
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=DEFAULT_PERCENTILES,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--rotate_k", type=int, choices=(0, 1, 2, 3), default=DEFAULT_ROTATE_K
    )
    parser.add_argument("--dpi", type=float, default=180.0)
    parser.add_argument("--fig_width", type=float, default=18.0)
    parser.add_argument(
        "--no_colorbar",
        action="store_true",
        help="隐藏 colorbar；默认每行显示一个主图色标和一个误差色标。",
    )
    return parser.parse_args()


def _normalise_kind(kind: str) -> str:
    kind = kind.lower()
    return "sr" if kind.startswith("sr_") else kind


def _discover_images(
    run_dir: Path, forced_iteration: int | None
) -> Tuple[int, Dict[int, Dict[str, Path]]]:
    by_iteration: Dict[int, Dict[int, Dict[str, Path]]] = {}
    for path in sorted(run_dir.rglob("*.png"), key=lambda item: str(item).lower()):
        match = _IMAGE_RE.match(path.stem)
        if not match:
            continue
        iteration = int(match.group("iteration"))
        index = int(match.group("index"))
        kind = _normalise_kind(match.group("kind"))
        sample = by_iteration.setdefault(iteration, {}).setdefault(index, {})
        sample.setdefault(kind, path)

    if not by_iteration:
        raise FileNotFoundError(f"{run_dir} 中未找到 validation LR/SR/HR PNG")
    iteration = forced_iteration if forced_iteration is not None else max(by_iteration)
    if iteration not in by_iteration:
        raise FileNotFoundError(
            f"{run_dir} 中没有 iter={iteration}，可用迭代为 {sorted(by_iteration)}"
        )
    complete = {
        index: paths
        for index, paths in by_iteration[iteration].items()
        if {"lr", "sr", "hr"}.issubset(paths)
    }
    if not complete:
        raise FileNotFoundError(f"{run_dir} 的 iter={iteration} 没有完整 LR/SR/HR 样本")
    return iteration, complete


def _load_normalised_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("F"), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) > 1.5:
        array = array / 255.0
    return array.astype(np.float64, copy=False)


def _find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("metrics.csv"), key=lambda path: str(path).lower())
    return candidates[-1] if candidates else None


def _load_metrics(run_dir: Path) -> Tuple[Path | None, Dict[int, Dict[str, str]]]:
    path = _find_metrics_csv(run_dir)
    if path is None:
        return None, {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return path, {int(row["index"]): row for row in rows if row.get("index")}


def _safe_float(row: Mapping[str, str] | None, key: str) -> float | None:
    if not row or row.get(key) in (None, ""):
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _row_label(
    index: int,
    baseline_row: Mapping[str, str] | None,
    selected_row: Mapping[str, str] | None,
) -> str:
    row = selected_row or baseline_row or {}
    parts = [f"Sample {index}"]
    context = []
    if row.get("met_name"):
        context.append(row["met_name"])
    if row.get("slice_idx") not in (None, ""):
        context.append(f"slice {row['slice_idx']}")
    if row.get("lowres") not in (None, "", "-1"):
        context.append(f"LR {2 * int(float(row['lowres']))}")
    if context:
        parts.append(" | ".join(context))
    baseline_psnr = _safe_float(baseline_row, "psnr")
    selected_psnr = _safe_float(selected_row, "psnr")
    if baseline_psnr is not None and selected_psnr is not None:
        parts.append(f"PSNR change {selected_psnr - baseline_psnr:+.2f} dB")
    return "\n".join(parts)


def _choose_indices(
    baseline: Mapping[int, object],
    selected: Mapping[int, object],
    requested: Iterable[int] | None,
    max_samples: int,
) -> list[int]:
    common = sorted(set(baseline).intersection(selected))
    if requested:
        requested_list = list(dict.fromkeys(int(value) for value in requested))
        missing = [value for value in requested_list if value not in common]
        if missing:
            raise ValueError(f"这些样本行在两组结果中不完整: {missing}")
        return requested_list
    return common[: max(0, int(max_samples))]


def _add_inset_colorbar(fig, ax, image, label: str) -> None:
    color_axis = inset_axes(ax, width="4%", height="58%", loc="lower right", borderpad=0.7)
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    colorbar.ax.set_title(label, fontsize=6, pad=2)


def main() -> None:
    args = _parse_args()
    percentiles = validate_percentiles(args.percentiles)
    baseline_run = Path(args.baseline_run).expanduser().resolve()
    selected_run = Path(args.selected_run).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    baseline_iter, baseline_images = _discover_images(baseline_run, args.baseline_iter)
    selected_iter, selected_images = _discover_images(selected_run, args.selected_iter)
    baseline_metrics_path, baseline_metrics = _load_metrics(baseline_run)
    selected_metrics_path, selected_metrics = _load_metrics(selected_run)
    indices = _choose_indices(
        baseline_images, selected_images, args.indices, args.max_samples
    )
    if not indices:
        raise SystemExit("没有可展示的共同样本")

    nrows, ncols = len(indices), 6
    fig_height = max(3.1, float(args.fig_width) * nrows / ncols * 0.90)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(float(args.fig_width), fig_height),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.025, wspace=0.025, hspace=0.06)
    titles = (
        "LR input",
        f"Baseline SR ({baseline_iter})",
        f"Improved SR ({selected_iter})",
        "HR target",
        "Baseline - HR",
        "Improved - HR",
    )

    sidecar_rows = []
    for row_number, index in enumerate(indices):
        baseline_paths = baseline_images[index]
        selected_paths = selected_images[index]
        lr = _load_normalised_gray(selected_paths["lr"])
        baseline_sr = _load_normalised_gray(baseline_paths["sr"])
        selected_sr = _load_normalised_gray(selected_paths["sr"])
        hr = _load_normalised_gray(selected_paths["hr"])

        vmin, vmax = robust_display_limits(
            (lr, baseline_sr, selected_sr, hr), reference=hr, percentiles=percentiles
        )
        baseline_error = baseline_sr - hr
        selected_error = selected_sr - hr
        error_limit = max(
            signed_error_limit(baseline_error, reference=hr, percentile=percentiles[1]),
            signed_error_limit(selected_error, reference=hr, percentile=percentiles[1]),
        )

        panels = (lr, baseline_sr, selected_sr, hr, baseline_error, selected_error)
        for column, panel in enumerate(panels):
            axis = axes[row_number, column]
            if column < 4:
                panel_image = axis.imshow(
                    orient_for_display(panel, args.rotate_k),
                    cmap=args.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
            else:
                panel_image = axis.imshow(
                    orient_for_display(panel, args.rotate_k),
                    cmap="coolwarm",
                    vmin=-error_limit,
                    vmax=error_limit,
                    interpolation="nearest",
                )
            axis.set_axis_off()
            if row_number == 0:
                axis.set_title(titles[column], fontsize=11)
            if not args.no_colorbar and column == 3:
                _add_inset_colorbar(fig, axis, panel_image, "value")
            elif not args.no_colorbar and column == 5:
                _add_inset_colorbar(fig, axis, panel_image, "error")

        baseline_row = baseline_metrics.get(index)
        selected_row = selected_metrics.get(index)
        axes[row_number, 0].text(
            -0.08,
            0.5,
            _row_label(index, baseline_row, selected_row),
            transform=axes[row_number, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color="#222222",
        )
        sidecar_rows.append(
            {
                "index": index,
                "baseline_psnr": _safe_float(baseline_row, "psnr"),
                "selected_psnr": _safe_float(selected_row, "psnr"),
                "vmin": vmin,
                "vmax": vmax,
                "signed_error_limit": error_limit,
                "met_name": (selected_row or baseline_row or {}).get("met_name"),
                "slice_idx": (selected_row or baseline_row or {}).get("slice_idx"),
            }
        )

    fig.suptitle(
        "SR3-DMI qualitative effect comparison\n"
        f"{args.cmap} | brain-aware {percentiles[0]:g}--{percentiles[1]:g}% shared row scale | "
        f"signed errors share a zero-centred scale | rot90 x{args.rotate_k}",
        fontsize=13,
        y=0.985,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    sidecar = {
        "output": str(output),
        "baseline_run": str(baseline_run),
        "selected_run": str(selected_run),
        "baseline_iter": baseline_iter,
        "selected_iter": selected_iter,
        "baseline_metrics_csv": str(baseline_metrics_path) if baseline_metrics_path else None,
        "selected_metrics_csv": str(selected_metrics_path) if selected_metrics_path else None,
        "cmap": args.cmap,
        "percentiles": list(percentiles),
        "rotate_k": args.rotate_k,
        "colorbar": not args.no_colorbar,
        "rows": sidecar_rows,
    }
    sidecar_path = output.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {output}")
    print(f"显示参数: {sidecar_path}")


if __name__ == "__main__":
    main()
