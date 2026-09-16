#!/usr/bin/env python
"""Validate the retained Bruker fid_proc.64 -> 2dseq stage for all audited scans.

Evidence basis (per scan):
- reco: RecoFileSink(fid_proc.64, shuffle=true), spectral FT shift=0.5,
  spectral FT exponent=1, magnitude, RECO_map_slope, int16 output.
- visu_pars: VisuCoreSize=[256,9,9,5], little endian, magnitude image.

The script is read-only with respect to source data and writes a CSV/JSON report.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(x, y)[0, 1])


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)) / (np.mean(np.abs(y)) + 1e-12))


def decode_fid(path: Path) -> np.ndarray:
    # shuffle=true is empirically verified against 2dseq as interleaved
    # little-endian complex128 with spectral dimension fastest on disk.
    raw = np.fromfile(path, dtype="<c16")
    if raw.size != 256 * 9 * 9 * 5:
        raise ValueError(f"Unexpected fid_proc.64 complex count {raw.size}: {path}")
    return np.moveaxis(raw.reshape(5, 9, 9, 256), -1, 0)


def decode_2dseq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i2")
    if raw.size != 256 * 9 * 9 * 5:
        raise ValueError(f"Unexpected 2dseq int16 count {raw.size}: {path}")
    return np.moveaxis(raw.reshape(5, 9, 9, 256), -1, 0).astype(np.float64)


def reconstruct(fid: np.ndarray) -> np.ndarray:
    # RecoFTFilter exponent=1 corresponds to inverse-sign convention relative
    # to NumPy FFT; NumPy ifft includes 1/N normalization. The half-FOV shift
    # is represented by a circular roll of 128 spectral samples.
    return np.roll(np.abs(np.fft.ifft(fid, axis=0)), 128, axis=0)


def find_scan_dir(root: Path, animal: str, scan_id: str) -> Path:
    matches = [p.parent.parent.parent for p in (root / animal).rglob("fid_proc.64")
               if p.parent.parent.parent.name == str(scan_id) and p.parent.name == "1"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one scan directory for {animal}/{scan_id}, got {matches}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--audit-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as f:
        audit_rows = list(csv.DictReader(f))
    if len(audit_rows) != 16:
        raise AssertionError(f"Expected 16 audited scans, got {len(audit_rows)}")

    rows: list[dict[str, object]] = []
    for source in audit_rows:
        animal = source["animal_id"]
        scan_id = source["scan_id"]
        scan = find_scan_dir(args.data_root, animal, scan_id)
        pdata = scan / "pdata" / "1"
        fid = decode_fid(pdata / "fid_proc.64")
        seq = decode_2dseq(pdata / "2dseq")
        magnitude = reconstruct(fid)
        slope = float(source["reco_map_slope"])
        visu_slope = float(source["visu_data_slope"])
        mapped = magnitude * slope

        quantizers = {
            "round": np.rint(mapped),
            "floor": np.floor(mapped),
            "trunc": np.trunc(mapped),
            "ceil": np.ceil(mapped),
        }
        q_metrics = {}
        for name, arr in quantizers.items():
            clipped = np.clip(arr, -32766, 32766)
            diff = clipped - seq
            q_metrics[name] = {
                "exact_fraction": float(np.mean(diff == 0)),
                "mae_intensity_units": float(np.mean(np.abs(diff))),
                "max_abs_diff": float(np.max(np.abs(diff))),
            }
        best_q = min(q_metrics, key=lambda k: q_metrics[k]["mae_intensity_units"])
        physical = seq * visu_slope
        row = {
            "animal_id": animal,
            "scan_id": scan_id,
            "scan_dir": str(scan),
            "reco_map_slope": slope,
            "visu_data_slope": visu_slope,
            "slope_reciprocal_product": slope * visu_slope,
            "correlation_mapped_vs_2dseq": corrcoef(mapped, seq),
            "nrmse_mapped_vs_2dseq": nrmse(mapped, seq),
            "nrmse_physical_vs_ifft_magnitude": nrmse(physical, magnitude),
            "best_quantizer": best_q,
            "quantizer_metrics": q_metrics,
            "mapped_min": float(mapped.min()),
            "mapped_max": float(mapped.max()),
            "seq_min": float(seq.min()),
            "seq_max": float(seq.max()),
        }
        row["status"] = (
            "PASS_FIDPROC_TO_2DSEQ"
            if row["correlation_mapped_vs_2dseq"] > 0.999999
            and row["nrmse_mapped_vs_2dseq"] < 5e-4
            and abs(row["slope_reciprocal_product"] - 1.0) < 1e-9
            else "FAIL_FIDPROC_TO_2DSEQ"
        )
        rows.append(row)

    summary = {
        "evidence_scope": "local Bruker reco/visu_pars + numerical reproduction",
        "formula": "2dseq ~= quantize(RECO_map_slope * roll(abs(ifft(fid_proc.64, spectral_axis)), 128))",
        "scan_count": len(rows),
        "pass_count": sum(r["status"] == "PASS_FIDPROC_TO_2DSEQ" for r in rows),
        "min_correlation": min(float(r["correlation_mapped_vs_2dseq"]) for r in rows),
        "max_nrmse": max(float(r["nrmse_mapped_vs_2dseq"]) for r in rows),
        "max_slope_reciprocal_error": max(abs(float(r["slope_reciprocal_product"]) - 1.0) for r in rows),
        "rows": rows,
        "limitations": [
            "This validates only the retained fid_proc.64 -> 2dseq stage.",
            "It does not validate ser/pv2tsdata as an input to the vendor reco pipeline.",
            "It does not yet reproduce AverageList, RecoSortMaps, spatial windows, fractional RECO_rotate, or rawdata.job0.",
        ],
    }
    if summary["pass_count"] != 16:
        raise AssertionError(json.dumps(summary, ensure_ascii=False, indent=2))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_keys = [
        "animal_id", "scan_id", "scan_dir", "reco_map_slope", "visu_data_slope",
        "slope_reciprocal_product", "correlation_mapped_vs_2dseq",
        "nrmse_mapped_vs_2dseq", "nrmse_physical_vs_ifft_magnitude",
        "best_quantizer", "mapped_min", "mapped_max", "seq_min", "seq_max", "status",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in flat_keys} for r in rows)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
