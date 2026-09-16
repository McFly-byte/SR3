from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT_DIR = ROOT / "experiments" / "analyses/training_diagnostics_20260730" / "improvement_runs"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_for(run: Path, iteration: int):
    matches = sorted((run / "checkpoint").glob(f"I{iteration}_E*_gen.pth"))
    return str(matches[-1].with_name(matches[-1].name[:-8])) if matches else None


def main() -> int:
    baseline_runs = sorted(
        EXPERIMENTS.glob("sr3_dmi_64_effect_eval_*/results/metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not baseline_runs:
        raise FileNotFoundError("No fixed-subset baseline metrics.json was found.")
    baseline_path = baseline_runs[-1]
    baseline = _load_json(baseline_path)

    rows = []
    for run in sorted(EXPERIMENTS.glob("sr3_dmi_effect_ft_*_*")):
        config_path = run / "config_resolved.json"
        if not config_path.exists():
            continue
        config = _load_json(config_path)
        candidate = config.get("name", run.name)
        if candidate.endswith("_smoke"):
            continue
        for metrics_path in sorted((run / "results").glob("*/*_metrics.json")):
            metrics = _load_json(metrics_path)
            iteration = int(metrics.get("iter", 0))
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "psnr": metrics.get("psnr"),
                    "ssim": metrics.get("ssim"),
                    "masked_psnr": metrics.get("masked_psnr"),
                    "roi_mean_rel_err": metrics.get("roi_mean_rel_err"),
                    "false_hotspot_rate": metrics.get("false_hotspot_rate"),
                    "count": metrics.get("count"),
                    "eval_network": metrics.get("eval_network"),
                    "checkpoint_base": _checkpoint_for(run, iteration),
                    "metrics_json": str(metrics_path),
                    "run_dir": str(run),
                }
            )

    if not rows:
        raise FileNotFoundError("Pilot validation metrics are not available yet.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pilot_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["candidate"], row["iteration"])))

    baseline_hotspot = baseline.get("false_hotspot_rate")
    hotspot_limit = None if baseline_hotspot is None else max(
        float(baseline_hotspot) * 1.10,
        float(baseline_hotspot) + 0.005,
    )
    eligible = [
        row for row in rows
        if row["psnr"] is not None
        and row["masked_psnr"] is not None
        and float(row["masked_psnr"]) >= float(baseline.get("masked_psnr", 0.0)) - 0.10
        and (
            hotspot_limit is None
            or row["false_hotspot_rate"] is None
            or float(row["false_hotspot_rate"]) <= hotspot_limit
        )
    ]
    pool = eligible or rows
    selected = max(pool, key=lambda row: (float(row["psnr"]), float(row.get("ssim") or 0.0)))

    selection = {
        "baseline": {"metrics_json": str(baseline_path), **baseline},
        "guardrails": {
            "masked_psnr_min": float(baseline.get("masked_psnr", 0.0)) - 0.10,
            "false_hotspot_rate_max": hotspot_limit,
        },
        "selected_for_full_validation": selected,
        "eligible_candidates": len(eligible),
        "all_candidates": len(rows),
    }
    selection_path = OUT_DIR / "pilot_selection.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)

    print(csv_path)
    print(selection_path)
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
