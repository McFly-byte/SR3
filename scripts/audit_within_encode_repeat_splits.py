#!/usr/bin/env python3
"""Audit within-encode repeat splitting as a candidate self-supervised signal.

This is a read-only evidence script. It does not train a model or grant training
permission. Each spatial encode's contiguous accumulations are split into even
and odd repeat indices. The two means observe the same encoded location within
the same acquisition group, while their count-weighted recombination must equal
the vendor full mean. Noise independence is explicitly not inferred from this
algebraic and empirical consistency audit.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


SELECTED_SCANS = {
    "R001": (57, 58, 59, 60),
    "R002": (30, 31, 32, 33),
    "R003": (35, 36, 37, 38),
    "R004": (59, 60, 61, 62),
}


def load_validator(repo: Path):
    path = repo / "scripts" / "validate_rawdata_to_fidproc.py"
    spec = importlib.util.spec_from_file_location("validate_rawdata_to_fidproc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_contiguous(raw: np.ndarray, counts: np.ndarray):
    a = np.empty((counts.size, raw.shape[1]), dtype=np.complex128)
    b = np.empty_like(a)
    na = np.empty(counts.size, dtype=np.int64)
    nb = np.empty_like(na)
    full = np.empty_like(a)
    cursor = 0
    for index, count in enumerate(counts.tolist()):
        group = raw[cursor : cursor + count]
        group_a = group[0::2]
        group_b = group[1::2]
        if group_a.size == 0 or group_b.size == 0:
            raise ValueError(f"Encode {index} has fewer than two accumulations")
        a[index] = group_a.mean(axis=0)
        b[index] = group_b.mean(axis=0)
        na[index] = group_a.shape[0]
        nb[index] = group_b.shape[0]
        full[index] = group.mean(axis=0)
        cursor += count
    if cursor != raw.shape[0]:
        raise ValueError(f"Consumed {cursor} raw blocks, expected {raw.shape[0]}")
    recombined = (na[:, None] * a + nb[:, None] * b) / counts[:, None]
    return a, b, full, recombined, na, nb


def load_contract(validator, scan: Path) -> dict[str, Any]:
    method = validator.read_text(scan / "method")
    reco = validator.read_text(scan / "pdata" / "1" / "reco")
    counts = np.asarray(
        validator.parse_numbers(validator.bruker_raw(method, "AverageList")), dtype=np.int64
    )
    maps = np.asarray(
        validator.parse_numbers(validator.bruker_raw(reco, "RecoSortMaps"))[:305], dtype=np.int64
    )
    rotates_all = validator.parse_numbers(validator.bruker_raw(reco, "RECO_rotate"))
    user_window = np.asarray(
        validator.parse_numbers(validator.bruker_raw(reco, "RECO_usr_wdw")), dtype=np.float64
    ).reshape(4, 256) / 2147483647.0
    pc = validator.parse_numbers(validator.bruker_raw(reco, "RECO_pc_lin"))[1::2]
    return {
        "counts": counts,
        "maps": maps,
        "windows": (user_window[3, :5], user_window[2, :9], user_window[1, :9]),
        "rotates": (rotates_all[3], rotates_all[2], rotates_all[1]),
        "phase_correction": (pc[3], pc[2], pc[1]),
    }


def reconstruct(validator, encoded: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    return validator.reconstruct(
        encoded,
        contract["maps"],
        contract["windows"],
        contract["rotates"],
        contract["phase_correction"],
        use_window=True,
        fft_direction="ifft",
        shift_convention="full_rotate",
        shift_sign=-1,
        apply_phase_correction=True,
    )


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    root = args.data_root.resolve()
    validator = load_validator(repo)
    rows: list[dict[str, Any]] = []

    for animal, scan_numbers in SELECTED_SCANS.items():
        for scan_number in scan_numbers:
            scan = root / animal / "RAW" / str(scan_number)
            contract = load_contract(validator, scan)
            raw = validator.decode_raw(scan / "rawdata.job0", "ri")
            a, b, full, recombined, na, nb = split_contiguous(raw, contract["counts"])
            recon_a = reconstruct(validator, a, contract)
            recon_b = reconstruct(validator, b, contract)
            recon_full = reconstruct(validator, full, contract)
            recon_recombined = reconstruct(validator, recombined, contract)
            vendor = validator.decode_vendor_fid(scan / "pdata" / "1" / "fid_proc.64")
            rows.append(
                {
                    "animal": animal,
                    "scan": scan_number,
                    "encode_count": int(contract["counts"].size),
                    "accumulation_count": int(contract["counts"].sum()),
                    "min_accumulations_per_encode": int(contract["counts"].min()),
                    "max_accumulations_per_encode": int(contract["counts"].max()),
                    "split_a_count_min": int(na.min()),
                    "split_b_count_min": int(nb.min()),
                    "encoded_recombination_rel_l2": rel_l2(recombined, full),
                    "voxel_recombination_rel_l2": rel_l2(recon_recombined, recon_full),
                    "full_reconstruction_vs_vendor_rel_l2": rel_l2(recon_full, vendor),
                    "split_pair_relative_difference": float(
                        np.linalg.norm(recon_a - recon_b)
                        / (0.5 * (np.linalg.norm(recon_a) + np.linalg.norm(recon_b)) + 1e-30)
                    ),
                    "split_pair_magnitude_correlation": validator.pearson(
                        np.abs(recon_a), np.abs(recon_b)
                    ),
                }
            )

    payload = {
        "schema_version": "1.0",
        "purpose": "candidate_within_encode_repeat_split_audit",
        "split_definition": "alternating accumulation indices within each contiguous AverageList group",
        "scan_count": len(rows),
        "all_encodes_have_two_nonempty_splits": all(
            row["split_a_count_min"] >= 1 and row["split_b_count_min"] >= 1 for row in rows
        ),
        "max_encoded_recombination_rel_l2": max(
            row["encoded_recombination_rel_l2"] for row in rows
        ),
        "max_voxel_recombination_rel_l2": max(
            row["voxel_recombination_rel_l2"] for row in rows
        ),
        "max_full_reconstruction_vs_vendor_rel_l2": max(
            row["full_reconstruction_vs_vendor_rel_l2"] for row in rows
        ),
        "min_split_pair_magnitude_correlation": min(
            row["split_pair_magnitude_correlation"] for row in rows
        ),
        "max_split_pair_relative_difference": max(
            row["split_pair_relative_difference"] for row in rows
        ),
        "candidate_pair_construction_numerically_valid": all(
            row["encoded_recombination_rel_l2"] <= 1e-12
            and row["voxel_recombination_rel_l2"] <= 1e-12
            and row["full_reconstruction_vs_vendor_rel_l2"] <= 1e-10
            for row in rows
        ),
        "noise_independence_proven": False,
        "physiological_stationarity_proven": False,
        "formal_real_data_training_allowed": False,
        "optimizer_step_executed": False,
        "network_parameters_updated": False,
        "interpretation": (
            "The retained raw accumulations can be split within each encoded location into two nonempty "
            "same-location observations whose count-weighted recombination exactly recovers the vendor full mean. "
            "This is a stronger candidate than treating consecutive scans as technical repeats, but it does not "
            "by itself prove independent zero-mean noise, define an HR target, or authorize training."
        ),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in (
        "scan_count",
        "candidate_pair_construction_numerically_valid",
        "max_encoded_recombination_rel_l2",
        "max_full_reconstruction_vs_vendor_rel_l2",
        "min_split_pair_magnitude_correlation",
        "noise_independence_proven",
        "formal_real_data_training_allowed",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
