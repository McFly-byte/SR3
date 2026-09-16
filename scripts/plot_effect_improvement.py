from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "analyses/training_diagnostics_20260730" / "improvement_runs"
OUT = DATA / "final_effect_comparison.png"


def main() -> int:
    overall = pd.read_csv(DATA / "full_validation_comparison.csv").set_index("label")
    per_met = pd.read_csv(DATA / "full_validation_by_metabolite.csv")
    order = ["baseline_raw", "selected_ema", "selected_raw"]
    display = ["Baseline raw", "Selected EMA", "Selected raw"]
    colors = ["#9CA3AF", "#60A5FA", "#F97316"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.24
    for i, (label, name, color) in enumerate(zip(order, display, colors)):
        vals = [overall.loc[label, "psnr"], overall.loc[label, "masked_psnr"]]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Overall PSNR", "Masked PSNR"])
    ax.set_ylabel("dB (higher is better)")
    ax.set_ylim(28, 38)
    ax.set_title("Full validation fidelity (n=408)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    x = np.arange(2)
    for i, (label, name, color) in enumerate(zip(order, display, colors)):
        vals = [
            overall.loc[label, "roi_mean_rel_err"] * 100,
            overall.loc[label, "false_hotspot_rate"] * 100,
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=2)
    ax.set_xticks(x, ["ROI mean relative error", "False-hotspot rate"])
    ax.set_ylabel("% (lower is better)")
    ax.set_title("Quantitative error and hotspot guardrail")

    pivot_psnr = per_met.pivot(index="metabolite", columns="label", values="masked_psnr")
    met_order = [met for met in ["Glc", "Glx", "HDO", "Lac"] if met in pivot_psnr.index]
    delta = pivot_psnr.loc[met_order, "selected_raw"] - pivot_psnr.loc[met_order, "baseline_raw"]
    ax = axes[1, 0]
    bars = ax.bar(met_order, delta.values, color="#F97316")
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.bar_label(bars, fmt="%+.2f", fontsize=9, padding=2)
    ax.set_ylabel("Masked PSNR change (dB)")
    ax.set_title("Selected raw improvement by metabolite")

    hotspot = per_met.pivot(index="metabolite", columns="label", values="false_hotspot_rate") * 100
    ax = axes[1, 1]
    x = np.arange(len(met_order))
    for i, (label, name, color) in enumerate(
        [("baseline_raw", "Baseline raw", "#9CA3AF"), ("selected_raw", "Selected raw", "#F97316")]
    ):
        vals = hotspot.loc[met_order, label].values
        bars = ax.bar(x + (i - 0.5) * 0.34, vals, 0.34, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=2)
    ax.set_xticks(x, met_order)
    ax.set_ylabel("False-hotspot rate (%)")
    ax.set_title("Hotspot guardrail by metabolite")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("SR3-DMI effect-first improvement: selected 50.5k raw checkpoint", fontsize=15)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
