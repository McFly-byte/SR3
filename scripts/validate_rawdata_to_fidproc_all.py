#!/usr/bin/env python3
"""Batch validation of the parameter-derived Bruker rawdata -> fid_proc chain."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_rawdata_to_fidproc as v  # noqa: E402


SCANS = {
    "R001": [57, 58, 59, 60],
    "R002": [30, 31, 32, 33],
    "R003": [35, 36, 37, 38],
    "R004": [59, 60, 61, 62],
}

THRESHOLDS = {
    "magnitude_corr_min": 0.999999999,
    "magnitude_rel_l2_max": 1e-10,
    "complex_rel_l2_max": 1e-10,
    "complex_scale_abs_error_max": 1e-10,
}


def validate_scan(scan: Path, animal_id: str, scan_id: int) -> dict:
    method = v.read_text(scan / "method")
    reco = v.read_text(scan / "pdata" / "1" / "reco")
    counts = np.asarray(v.parse_numbers(v.bruker_raw(method, "AverageList")), dtype=np.int64)
    maps = np.asarray(v.parse_numbers(v.bruker_raw(reco, "RecoSortMaps"))[:305], dtype=np.int64)
    enc0 = np.asarray(v.parse_numbers(v.bruker_raw(method, "PVM_EncGenSteps0")), dtype=np.int64)
    enc1 = np.asarray(v.parse_numbers(v.bruker_raw(method, "PVM_EncGenSteps1")), dtype=np.int64)
    enc2 = np.asarray(v.parse_numbers(v.bruker_raw(method, "PVM_EncGenSteps2")), dtype=np.int64)
    coordinate_maps = (enc2 + 2) * 81 + (enc1 + 4) * 9 + (enc0 + 4)
    if not np.array_equal(maps, coordinate_maps):
        raise AssertionError(f"{animal_id}/{scan_id}: RecoSortMaps mismatch")

    user_window = np.asarray(v.parse_numbers(v.bruker_raw(reco, "RECO_usr_wdw")), dtype=np.float64)
    if user_window.size != 1024:
        raise AssertionError(f"{animal_id}/{scan_id}: RECO_usr_wdw has {user_window.size} entries")
    user_window = user_window.reshape(4, 256) / 2147483647.0
    windows = (user_window[3, :5], user_window[2, :9], user_window[1, :9])

    rotate_values = v.parse_numbers(v.bruker_raw(reco, "RECO_rotate"))
    rotates = (rotate_values[3], rotate_values[2], rotate_values[1])
    pc_values = v.parse_numbers(v.bruker_raw(reco, "RECO_pc_lin"))
    if len(pc_values) != 8:
        raise AssertionError(f"{animal_id}/{scan_id}: unexpected RECO_pc_lin {pc_values}")
    pc_first_order = pc_values[1::2]
    phase_correction = (pc_first_order[3], pc_first_order[2], pc_first_order[1])

    raw = v.decode_raw(scan / "rawdata.job0", "ri")
    encoded = v.aggregate_contiguous(raw, counts, "mean")
    pred = v.reconstruct(
        encoded,
        maps,
        windows,
        rotates,
        phase_correction,
        use_window=True,
        fft_direction="ifft",
        shift_convention="full_rotate",
        shift_sign=-1,
        apply_phase_correction=True,
    )
    ref = v.decode_vendor_fid(scan / "pdata" / "1" / "fid_proc.64")
    scale_abs, complex_rel_l2 = v.best_scale_nrmse(pred, ref)
    _, magnitude_rel_l2 = v.best_scale_nrmse(np.abs(pred), np.abs(ref))
    magnitude_corr = v.pearson(np.abs(pred), np.abs(ref))
    passed = bool(
        magnitude_corr >= THRESHOLDS["magnitude_corr_min"]
        and magnitude_rel_l2 <= THRESHOLDS["magnitude_rel_l2_max"]
        and complex_rel_l2 <= THRESHOLDS["complex_rel_l2_max"]
        and abs(scale_abs - 1.0) <= THRESHOLDS["complex_scale_abs_error_max"]
    )
    return {
        "animal_id": animal_id,
        "scan_id": scan_id,
        "scan_dir": str(scan),
        "average_count": int(counts.size),
        "average_list_sum": int(counts.sum()),
        "reco_rotate_z": rotates[0],
        "reco_rotate_y": rotates[1],
        "reco_rotate_x": rotates[2],
        "phase_correction_z_deg": phase_correction[0],
        "phase_correction_y_deg": phase_correction[1],
        "phase_correction_x_deg": phase_correction[2],
        "magnitude_corr": magnitude_corr,
        "magnitude_rel_l2_after_scale": magnitude_rel_l2,
        "complex_rel_l2_after_scalar": complex_rel_l2,
        "abs_complex_scale": scale_abs,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for animal_id, scan_ids in SCANS.items():
        for scan_id in scan_ids:
            row = validate_scan(args.data_root / animal_id / "RAW" / str(scan_id), animal_id, scan_id)
            rows.append(row)
            print(f"{animal_id}/{scan_id}: {'PASS' if row['passed'] else 'FAIL'} complex_rel_l2={row['complex_rel_l2_after_scalar']:.3e}")

    summary = {
        "schema_version": "1.0",
        "data_root": str(args.data_root),
        "scan_count": len(rows),
        "pass_count": sum(bool(row["passed"]) for row in rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "thresholds": THRESHOLDS,
        "reconstruction_contract": {
            "raw_complex_order": "real_then_imaginary",
            "average_order": "contiguous_groups_in_AverageList_order",
            "average_normalization": "arithmetic_mean_per_encoding",
            "sort": "RecoSortMaps_0_based_to_405_grid",
            "spatial_window": "per_scan_RECO_usr_wdw",
            "fractional_shift": "exp(-i*2*pi*RECO_rotate*n)_before_each_axis_ifft",
            "spatial_transform": "numpy_ifft_default_1_over_N_per_axis",
            "phase_correction": "per_scan_RECO_pc_lin_first_order_on_zero_based_output_coordinates",
            "output_layout": "complex128_[spectral,z,y,x]",
        },
        "aggregate_metrics": {
            "min_magnitude_corr": min(row["magnitude_corr"] for row in rows),
            "max_magnitude_rel_l2_after_scale": max(row["magnitude_rel_l2_after_scale"] for row in rows),
            "max_complex_rel_l2_after_scalar": max(row["complex_rel_l2_after_scalar"] for row in rows),
            "max_complex_scale_abs_error": max(abs(row["abs_complex_scale"] - 1.0) for row in rows),
        },
        "formal_vendor_frontend_reproduced": all(bool(row["passed"]) for row in rows),
        "training_permission_changed": False,
        "rows": rows,
        "limitations": [
            "This validates the retained vendor reconstruction from rawdata.job0 to fid_proc.64 for the 16 audited scans.",
            "It does not validate metabolite fitting, ppm direction, normalization, T2 registration, an LR-only loss, or an HR DMI/MRSI target.",
            "Training permission remains controlled by healthy_invivo_training_gate.json and is not changed here.",
        ],
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ("scan_count", "pass_count", "all_passed", "aggregate_metrics")}, indent=2))


if __name__ == "__main__":
    main()
