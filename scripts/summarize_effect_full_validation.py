from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT_DIR = ROOT / "experiments" / "analyses/training_diagnostics_20260730" / "improvement_runs"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    values = [float(value) for value in values if value not in (None, "", "None")]
    return sum(values) / len(values) if values else None


def _label(name: str):
    if "baseline_raw" in name:
        return "baseline_raw"
    if name.endswith("_ema"):
        return "selected_ema"
    if name.endswith("_raw"):
        return "selected_raw"
    return name


def main() -> int:
    latest = {}
    for run in EXPERIMENTS.glob("sr3_dmi_effect_full_*"):
        config_path = run / "config_resolved.json"
        metrics_path = run / "results" / "metrics.json"
        samples_path = run / "results" / "metrics.csv"
        if not (config_path.exists() and metrics_path.exists() and samples_path.exists()):
            continue
        config = _load_json(config_path)
        name = config.get("name", run.name)
        previous = latest.get(name)
        if previous is None or metrics_path.stat().st_mtime > previous["metrics_path"].stat().st_mtime:
            latest[name] = {
                "run": run,
                "config": config,
                "metrics_path": metrics_path,
                "samples_path": samples_path,
            }

    records = []
    per_metabolite = []
    sample_tables = {}
    for name, item in sorted(latest.items()):
        metrics = _load_json(item["metrics_path"])
        label = _label(name)
        records.append(
            {
                "label": label,
                "name": name,
                "network": metrics.get("eval_network"),
                "count": metrics.get("count"),
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "hfen": metrics.get("hfen"),
                "frc_aucw": metrics.get("frc_aucw"),
                "masked_psnr": metrics.get("masked_psnr"),
                "masked_mae": metrics.get("masked_mae"),
                "roi_mean_rel_err": metrics.get("roi_mean_rel_err"),
                "roi_std_rel_err": metrics.get("roi_std_rel_err"),
                "false_hotspot_rate": metrics.get("false_hotspot_rate"),
                "degradation_l1": metrics.get("degradation_l1"),
                "metrics_json": str(item["metrics_path"]),
                "run_dir": str(item["run"]),
                "checkpoint_base": item["config"]["path"]["resume_state"],
            }
        )

        groups = defaultdict(list)
        sample_rows = []
        with item["samples_path"].open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                groups[row.get("met_name", "unknown")].append(row)
                sample_rows.append(row)
        sample_tables[label] = sample_rows
        for metabolite, rows in sorted(groups.items()):
            per_metabolite.append(
                {
                    "label": label,
                    "metabolite": metabolite,
                    "count": len(rows),
                    "psnr": _mean(row.get("psnr") for row in rows),
                    "ssim": _mean(row.get("ssim") for row in rows),
                    "masked_psnr": _mean(row.get("masked_psnr") for row in rows),
                    "masked_mae": _mean(row.get("masked_mae") for row in rows),
                    "roi_mean_rel_err": _mean(row.get("roi_mean_rel_err") for row in rows),
                    "false_hotspot_rate": _mean(row.get("false_hotspot_rate") for row in rows),
                }
            )

    labels = {record["label"] for record in records}
    required = {"baseline_raw", "selected_raw", "selected_ema"}
    if not required.issubset(labels):
        raise FileNotFoundError(
            "Full validation is incomplete; found labels: {}".format(sorted(labels))
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_path = OUT_DIR / "full_validation_comparison.csv"
    with overall_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metabolite_path = OUT_DIR / "full_validation_by_metabolite.csv"
    with metabolite_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_metabolite[0]))
        writer.writeheader()
        writer.writerows(per_metabolite)

    by_label = {record["label"]: record for record in records}
    baseline = by_label["baseline_raw"]
    candidates = [by_label["selected_raw"], by_label["selected_ema"]]

    paired_rows = []
    paired_summary = {}
    baseline_samples = {row["sample_id"]: row for row in sample_tables["baseline_raw"]}
    for candidate_label in ("selected_raw", "selected_ema"):
        diffs = []
        candidate_samples = {row["sample_id"]: row for row in sample_tables[candidate_label]}
        for sample_id in sorted(set(baseline_samples) & set(candidate_samples), key=int):
            baseline_row = baseline_samples[sample_id]
            candidate_row = candidate_samples[sample_id]
            psnr_diff = float(candidate_row["psnr"]) - float(baseline_row["psnr"])
            masked_diff = float(candidate_row["masked_psnr"]) - float(baseline_row["masked_psnr"])
            hotspot_diff = float(candidate_row["false_hotspot_rate"]) - float(baseline_row["false_hotspot_rate"])
            diffs.append(psnr_diff)
            paired_rows.append(
                {
                    "candidate": candidate_label,
                    "sample_id": sample_id,
                    "metabolite": candidate_row.get("met_name"),
                    "psnr_diff_db": psnr_diff,
                    "masked_psnr_diff_db": masked_diff,
                    "false_hotspot_rate_diff": hotspot_diff,
                }
            )
        mean_diff = statistics.fmean(diffs)
        standard_error = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        paired_summary[candidate_label] = {
            "count": len(diffs),
            "mean_psnr_diff_db": mean_diff,
            "median_psnr_diff_db": statistics.median(diffs),
            "psnr_win_rate": sum(diff > 0.0 for diff in diffs) / len(diffs),
            "mean_psnr_diff_95ci_normal": [
                mean_diff - 1.96 * standard_error,
                mean_diff + 1.96 * standard_error,
            ],
        }

    paired_path = OUT_DIR / "full_validation_paired_differences.csv"
    with paired_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    hotspot_limit = max(
        float(baseline["false_hotspot_rate"]) * 1.10,
        float(baseline["false_hotspot_rate"]) + 0.005,
    )
    eligible = [
        record for record in candidates
        if float(record["masked_psnr"]) >= float(baseline["masked_psnr"]) - 0.10
        and float(record["false_hotspot_rate"]) <= hotspot_limit
    ]
    final = max(eligible or candidates, key=lambda record: (float(record["psnr"]), float(record["ssim"])))
    summary = {
        "baseline": baseline,
        "selected": final,
        "improvement": {
            "psnr_db": float(final["psnr"]) - float(baseline["psnr"]),
            "ssim": float(final["ssim"]) - float(baseline["ssim"]),
            "masked_psnr_db": float(final["masked_psnr"]) - float(baseline["masked_psnr"]),
            "masked_mae": float(final["masked_mae"]) - float(baseline["masked_mae"]),
            "roi_mean_rel_err": float(final["roi_mean_rel_err"]) - float(baseline["roi_mean_rel_err"]),
            "false_hotspot_rate": float(final["false_hotspot_rate"]) - float(baseline["false_hotspot_rate"]),
        },
        "guardrails": {"false_hotspot_rate_max": hotspot_limit},
        "paired_comparison": paired_summary,
    }
    summary_path = OUT_DIR / "final_selection.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(overall_path)
    print(metabolite_path)
    print(paired_path)
    print(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
