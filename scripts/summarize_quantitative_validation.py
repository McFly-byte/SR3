from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT_DIR = EXPERIMENTS / "evaluations/model_selection_20260802" / "validation_summary"
NAMES = (
    "sr3_dmi_quant_full_previous_raw",
    "sr3_dmi_quant_full_selected_raw",
    "sr3_dmi_quant_full_selected_ema",
)
METRICS = (
    "psnr",
    "ssim",
    "masked_psnr",
    "masked_mae",
    "roi_mean_rel_err",
    "roi_mean_abs_err_quantity",
    "concentration_integral_rel_err",
    "native_acquisition_l1",
    "native_acquisition_rel_l1",
    "false_hotspot_rate",
)


def _latest_run(name: str) -> Path:
    runs = []
    for run in EXPERIMENTS.glob(f"{name}_*"):
        config_path = run / "config_resolved.json"
        if not config_path.is_file() or not (run / "results" / "metrics.json").is_file():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("name") == name:
            runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No completed run for {name}")
    return max(runs, key=lambda path: (path / "results" / "metrics.json").stat().st_mtime)


def _float(row: dict, key: str):
    value = row.get(key)
    return None if value in (None, "", "None") else float(value)


def _mean(rows: list[dict], key: str):
    values = [_float(row, key) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def main() -> int:
    runs = {name: _latest_run(name) for name in NAMES}
    payloads = {}
    sample_tables = {}
    for name, run in runs.items():
        with (run / "results" / "metrics.json").open("r", encoding="utf-8") as handle:
            payloads[name] = json.load(handle)
        with (run / "results" / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            sample_tables[name] = list(csv.DictReader(handle))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_rows = []
    for name in NAMES:
        config = json.loads((runs[name] / "config_resolved.json").read_text(encoding="utf-8"))
        metrics = payloads[name]
        overall_rows.append(
            {
                "name": name,
                "run_dir": str(runs[name]),
                "checkpoint": config["path"]["resume_state"],
                "eval_network": metrics.get("eval_network"),
                "count": metrics.get("count"),
                **{key: metrics.get(key) for key in METRICS},
            }
        )
    with (OUT_DIR / "overall.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0]))
        writer.writeheader()
        writer.writerows(overall_rows)

    grouped_rows = []
    for name, rows in sample_tables.items():
        groups = defaultdict(list)
        for row in rows:
            groups[(row.get("met_name", "unknown"), row.get("lr_matrix", "unknown"))].append(row)
        for (metabolite, lr_matrix), group in sorted(groups.items()):
            grouped_rows.append(
                {
                    "name": name,
                    "metabolite": metabolite,
                    "lr_matrix": lr_matrix,
                    "count": len(group),
                    **{key: _mean(group, key) for key in METRICS},
                }
            )
    with (OUT_DIR / "by_metabolite_lr_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped_rows[0]))
        writer.writeheader()
        writer.writerows(grouped_rows)

    baseline_name = "sr3_dmi_quant_full_previous_raw"
    baseline = payloads[baseline_name]
    baseline_samples = {row["sample_id"]: row for row in sample_tables[baseline_name]}
    paired_rows = []
    paired_summary = {}
    for candidate_name in NAMES[1:]:
        candidate_samples = {row["sample_id"]: row for row in sample_tables[candidate_name]}
        common_ids = sorted(set(baseline_samples) & set(candidate_samples), key=int)
        deltas = defaultdict(list)
        for sample_id in common_ids:
            old = baseline_samples[sample_id]
            new = candidate_samples[sample_id]
            record = {
                "candidate": candidate_name,
                "sample_id": sample_id,
                "metabolite": new.get("met_name"),
                "lr_matrix": new.get("lr_matrix"),
            }
            for key in METRICS:
                old_value, new_value = _float(old, key), _float(new, key)
                delta = None if old_value is None or new_value is None else new_value - old_value
                record[f"{key}_delta"] = delta
                if delta is not None:
                    deltas[key].append(delta)
            paired_rows.append(record)
        paired_summary[candidate_name] = {
            key: {
                "mean_delta": statistics.fmean(values),
                "median_delta": statistics.median(values),
                "lower_is_better_win_rate": sum(value < 0 for value in values) / len(values),
                "higher_is_better_win_rate": sum(value > 0 for value in values) / len(values),
            }
            for key, values in deltas.items()
        }
    with (OUT_DIR / "paired_sample_deltas.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    candidates = list(NAMES[1:])
    eligible = [
        name for name in candidates
        if float(payloads[name]["masked_psnr"]) >= float(baseline["masked_psnr"]) - 0.15
        and float(payloads[name]["false_hotspot_rate"]) <= float(baseline["false_hotspot_rate"]) + 0.002
        and float(payloads[name]["roi_mean_rel_err"]) <= float(baseline["roi_mean_rel_err"]) * 1.05
        and paired_summary[name]["masked_psnr"]["higher_is_better_win_rate"] >= 0.5
        and paired_summary[name]["roi_mean_rel_err"]["lower_is_better_win_rate"] >= 0.5
        and paired_summary[name]["native_acquisition_rel_l1"]["lower_is_better_win_rate"] >= 0.5
    ]
    if eligible:
        selected_name = max(
            eligible,
            key=lambda name: (float(payloads[name]["psnr"]), float(payloads[name]["ssim"])),
        )
        selection_reason = "Candidate passed aggregate and per-sample quantitative guardrails."
    else:
        selected_name = baseline_name
        selection_reason = (
            "Retained previous checkpoint: no candidate improved masked PSNR, ROI mean error "
            "and native acquisition residual on at least half of paired samples."
        )
    selected_payload = payloads[selected_name]
    summary = {
        "selection": selected_name,
        "selection_reason": selection_reason,
        "guardrails": {
            "masked_psnr_min": float(baseline["masked_psnr"]) - 0.15,
            "false_hotspot_rate_max": float(baseline["false_hotspot_rate"]) + 0.002,
            "roi_mean_rel_err_max": float(baseline["roi_mean_rel_err"]) * 1.05,
            "paired_win_rate_min": 0.5,
        },
        "runs": {name: str(path) for name, path in runs.items()},
        "baseline": baseline,
        "selected": selected_payload,
        "improvement": {
            key: float(selected_payload[key]) - float(baseline[key])
            for key in METRICS if baseline.get(key) is not None and selected_payload.get(key) is not None
        },
        "paired_summary": paired_summary,
    }
    with (OUT_DIR / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary["improvement"], ensure_ascii=False, indent=2))
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
