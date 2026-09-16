from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
MEETING = ROOT / "paper_material" / "20260814组会" / "课题当前进展"
MATERIAL = MEETING / "推理对比材料"
SUMMARY = MATERIAL / "汇总"

SIM_DATA = Path(r"D:\LMC\data\simulated_with_lesion\最终结果\sr3训练数据_msr_mrsi_npz\val")
SIM_MANIFEST = SIM_DATA / "manifest.csv"
SIM_PRE_METRICS = (
    ROOT
    / "experiments"
    / "evaluations"
    / "raw_ema_comparison_20260802"
    / "baseline_i50000_raw"
    / "results"
    / "metrics.csv"
)
SIM_POST_METRICS = (
    ROOT
    / "experiments"
    / "evaluations"
    / "raw_ema_comparison_20260802"
    / "current_i50500_raw"
    / "results"
    / "metrics.csv"
)
SIM_NUMERIC = MATERIAL / "仿真微调前后"

INVIVO_PRE = MATERIAL / "I50000_活体"
INVIVO_POST = ROOT / "paper_material" / "20260814组会" / "活体带病灶小鼠"

METABOLITES = ("HDO", "Glc", "Glx", "Lac")
METRIC_FIELDS = (
    "psnr",
    "ssim",
    "masked_psnr",
    "masked_mae",
    "roi_mean_rel_err",
    "false_hotspot_rate",
    "degradation_l1",
)

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid metric values for {field}")
    return float(values.mean())


def summarize_simulation() -> dict[str, object]:
    manifest = {row["sample_id"]: row for row in _read_csv(SIM_MANIFEST)}
    pre_rows = _read_csv(SIM_PRE_METRICS)
    post_rows = _read_csv(SIM_POST_METRICS)
    if len(pre_rows) != 408 or len(post_rows) != 408:
        raise AssertionError(f"Expected 408 simulation rows, got {len(pre_rows)} and {len(post_rows)}")

    pre_by_id = {row["sample_id"]: row for row in pre_rows}
    post_by_id = {row["sample_id"]: row for row in post_rows}
    if pre_by_id.keys() != post_by_id.keys():
        raise AssertionError("Pre/post simulation sample IDs differ")
    if not pre_by_id.keys() <= manifest.keys():
        raise AssertionError("Simulation metrics contain sample IDs absent from manifest")

    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in pre_by_id:
        grouped[manifest[sample_id]["scenario_type"]].append(sample_id)
    if {key: len(value) for key, value in grouped.items()} != {"healthy": 204, "lesion": 204}:
        raise AssertionError(f"Unexpected scenario counts: {dict((k, len(v)) for k, v in grouped.items())}")

    output_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"sample_count": 408, "scenario_counts": {k: len(v) for k, v in grouped.items()}}
    for scenario, ids in [("overall", list(pre_by_id)), ("healthy", grouped["healthy"]), ("lesion", grouped["lesion"])]:
        pre_subset = [pre_by_id[sample_id] for sample_id in ids]
        post_subset = [post_by_id[sample_id] for sample_id in ids]
        scenario_summary: dict[str, object] = {"count": len(ids)}
        for field in METRIC_FIELDS:
            before = _mean(pre_subset, field)
            after = _mean(post_subset, field)
            delta = after - before
            relative_change = delta / before * 100.0 if before != 0 else float("nan")
            output_rows.append(
                {
                    "scenario": scenario,
                    "count": len(ids),
                    "metric": field,
                    "i50000": before,
                    "i50500": after,
                    "absolute_change": delta,
                    "relative_change_pct": relative_change,
                }
            )
            scenario_summary[field] = {
                "i50000": before,
                "i50500": after,
                "absolute_change": delta,
                "relative_change_pct": relative_change,
            }
        summary[scenario] = scenario_summary

    _write_csv(SUMMARY / "仿真微调前后_分场景指标.csv", output_rows)
    return summary


def _rot(array: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(array, dtype=np.float32), 1)


def make_simulation_figure() -> None:
    pairs = [("healthy", "健康反事实"), ("lesion", "病灶反事实")]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), constrained_layout=True)
    for row_idx, (stem, label) in enumerate(pairs):
        before = np.load(SIM_NUMERIC / f"{stem}_pre.npz")
        after = np.load(SIM_NUMERIC / f"{stem}_post.npz")
        for key in ("lr_normalized", "hr_normalized"):
            if not np.array_equal(before[key], after[key]):
                raise AssertionError(f"Simulation {stem} pre/post {key} differs")
        if int(before["seed"]) != int(after["seed"]):
            raise AssertionError(f"Simulation {stem} seed differs")

        arrays = [
            _rot(before["lr_normalized"]),
            _rot(before["sr_normalized"]),
            _rot(after["sr_normalized"]),
            _rot(before["hr_normalized"]),
        ]
        titles = ["LR 输入", "I50000 SR", "I50500 SR", "HR 真值"]
        for col_idx, (array, title) in enumerate(zip(arrays, titles)):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(array, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nLac · 40 min · LR16", fontsize=12, fontweight="bold")
            if col_idx == 3:
                fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    fig.suptitle("同一样本、同一 seed 的微调前后仿真超分对比", fontsize=17, fontweight="bold")
    fig.savefig(SUMMARY / "仿真微调前后_健康与病灶对比.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _invivo_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    rows = _read_csv(path / "metrics" / "per_slice_sr_metrics.csv")
    return {(row["case_id"], int(row["slice_idx"]), row["metabolite"]): row for row in rows}


def summarize_invivo() -> dict[str, object]:
    pre = _invivo_rows(INVIVO_PRE)
    post = _invivo_rows(INVIVO_POST)
    if len(pre) != 32 or len(post) != 32 or pre.keys() != post.keys():
        raise AssertionError(f"Expected matched 32-row in-vivo results, got {len(pre)} and {len(post)}")

    fields = (
        "sr_ensemble_native_acquisition_rel_l1",
        "seed_mean_cv",
        "sr_ensemble_background_leakage_ratio",
        "sr_vs_bicubic_gradient_ratio",
    )
    output_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"sample_count": len(pre), "cases": {}}
    for case_id in ("ALL", "mouse_11.7T", "mouse_9.4T"):
        keys = list(pre) if case_id == "ALL" else [key for key in pre if key[0] == case_id]
        case_summary: dict[str, object] = {"count": len(keys)}
        for field in fields:
            before = float(np.mean([float(pre[key][field]) for key in keys]))
            after = float(np.mean([float(post[key][field]) for key in keys]))
            delta = after - before
            rel = delta / before * 100.0 if before != 0 else float("nan")
            output_rows.append(
                {
                    "case_id": case_id,
                    "count": len(keys),
                    "metric": field,
                    "i50000": before,
                    "i50500": after,
                    "absolute_change": delta,
                    "relative_change_pct": rel,
                }
            )
            case_summary[field] = {
                "i50000": before,
                "i50500": after,
                "absolute_change": delta,
                "relative_change_pct": rel,
            }
        summary["cases"][case_id] = case_summary
    _write_csv(SUMMARY / "活体微调前后_观测一致性指标.csv", output_rows)
    return summary


def make_invivo_figure(case_id: str, slice_idx: int, field_label: str) -> None:
    pre_metrics = _invivo_rows(INVIVO_PRE)
    post_metrics = _invivo_rows(INVIVO_POST)
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 8.3), constrained_layout=True)
    for row_idx, metabolite in enumerate(METABOLITES):
        stem = f"slice_{slice_idx:02d}_{metabolite}_sr_result.npz"
        pre = np.load(INVIVO_PRE / "numeric" / case_id / stem)
        post = np.load(INVIVO_POST / "numeric" / case_id / stem)
        if not np.array_equal(pre["seeds"], post["seeds"]):
            raise AssertionError(f"Seed mismatch: {case_id} {metabolite}")
        if not np.array_equal(pre["native_normalized"], post["native_normalized"]):
            raise AssertionError(f"Native input mismatch: {case_id} {metabolite}")
        if float(pre["normalization_scale_relative_au"]) != float(post["normalization_scale_relative_au"]):
            raise AssertionError(f"Scale mismatch: {case_id} {metabolite}")

        native = np.asarray(pre["native_relative_au"], dtype=np.float32)
        sr_pre = np.asarray(pre["sr_ensemble_mean_relative_au"], dtype=np.float32)
        sr_post = np.asarray(post["sr_ensemble_mean_relative_au"], dtype=np.float32)
        native_mask = np.asarray(pre["anatomy_mask_native"], dtype=bool)
        mask64 = np.asarray(pre["anatomy_mask_64"], dtype=bool)
        display_values = np.concatenate([native[native_mask], sr_pre[mask64], sr_post[mask64]])
        display_values = display_values[np.isfinite(display_values) & (display_values >= 0)]
        vmax = float(np.percentile(display_values, 99.0)) if display_values.size else 1.0
        vmax = max(vmax, 1e-8)

        panels = [native, sr_pre, sr_post]
        titles = [f"原始 LR {native.shape[0]}×{native.shape[1]}", "I50000 · 5-seed mean", "I50500 · 5-seed mean"]
        for col_idx, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(panel, cmap="turbo", vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metabolite, fontsize=12.5, fontweight="bold")
            if col_idx == 2:
                fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)

        key = (case_id, slice_idx, metabolite)
        before_error = float(pre_metrics[key]["sr_ensemble_native_acquisition_rel_l1"])
        after_error = float(post_metrics[key]["sr_ensemble_native_acquisition_rel_l1"])
        axes[row_idx, 2].text(
            0.98,
            0.02,
            f"回算相对 L1\n{before_error:.3f} → {after_error:.3f}",
            transform=axes[row_idx, 2].transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.58, "edgecolor": "none"},
        )

    fig.suptitle(
        f"{field_label} · 中间切片 {slice_idx + 1} · 微调前后活体超分对比",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(SUMMARY / f"活体_{case_id}_四代谢物_微调前后.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    simulation = summarize_simulation()
    invivo = summarize_invivo()
    make_invivo_figure("mouse_11.7T", 2, "11.7 T · 9×9 · 5 层")
    make_invivo_figure("mouse_9.4T", 1, "9.4 T · 7×7 · 3 层")
    report = {
        "simulation": simulation,
        "invivo": invivo,
        "comparison_contract": {
            "simulation": "paired 408-sample full-validation CSV; qualitative source is the retained full-validation comparison figure and sidecar",
            "invivo": "same native input, same normalization scale, same five deterministic seeds, 50 DDIM steps",
            "invivo_no_ground_truth": True,
            "invivo_quantity_unit": "within-case relative signal a.u.; not mM",
        },
    }
    (SUMMARY / "证据汇总.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(SUMMARY)


if __name__ == "__main__":
    main()
