#!/usr/bin/env python3
"""Audit retained Bruker fid_proc.64 spectra without inventing preprocessing.

The script is intentionally conservative: it applies only the spectral transform
already supported by each scan's vendor reco graph and the previously validated
fid_proc.64 -> 2dseq numerical chain. It does not perform delay correction,
apodization, phase/frequency correction, baseline subtraction, peak assignment,
metabolite integration, masking, registration, or normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


EXPECTED_SHAPE = (256, 9, 9, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1")


def scalar(text: str, key: str) -> float:
    match = re.search(rf"^##\${re.escape(key)}=\s*([^\r\n]+)", text, re.MULTILINE)
    if not match:
        raise KeyError(key)
    value = match.group(1).strip()
    if value.startswith("("):
        tail = text[match.end():].lstrip("\r\n ")
        value = tail.split()[0]
    return float(value.strip("<>"))


def first_array_value(text: str, key: str) -> float:
    match = re.search(rf"^##\${re.escape(key)}=\s*\([^\r\n]*\)\s*[\r\n]+([^\r\n]+)", text, re.MULTILINE)
    if not match:
        raise KeyError(key)
    return float(match.group(1).split()[0].strip("<>"))


def decode_fid(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<c16")
    if raw.size != int(np.prod(EXPECTED_SHAPE)):
        raise ValueError(f"Unexpected fid_proc.64 complex count {raw.size}: {path}")
    return np.moveaxis(raw.reshape(5, 9, 9, 256), -1, 0)


def decode_2dseq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i2")
    if raw.size != int(np.prod(EXPECTED_SHAPE)):
        raise ValueError(f"Unexpected 2dseq int16 count {raw.size}: {path}")
    return np.moveaxis(raw.reshape(5, 9, 9, 256), -1, 0).astype(np.float64)


def vendor_magnitude(fid: np.ndarray) -> np.ndarray:
    return np.roll(np.abs(np.fft.ifft(fid, axis=0)), 128, axis=0)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a64 - b64) ** 2)) / (np.mean(np.abs(b64)) + 1e-12))


def local_peak_indices(values: np.ndarray, count: int = 8, min_distance: int = 3) -> List[int]:
    candidates = [
        idx for idx in range(1, values.size - 1)
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]
    ]
    selected: List[int] = []
    for idx in sorted(candidates, key=lambda i: float(values[i]), reverse=True):
        if all(abs(idx - existing) >= min_distance for existing in selected):
            selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reco-validation-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    source_path = Path(args.reco_validation_json).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("pass_count") != 16:
        raise AssertionError("Requires the existing 16/16 fid_proc.64 -> 2dseq validation")

    rows: List[Dict[str, Any]] = []
    archive: Dict[str, np.ndarray] = {}
    for validated in source["rows"]:
        animal = str(validated["animal_id"])
        scan_id = str(validated["scan_id"])
        scan_dir = Path(validated["scan_dir"])
        method_path = scan_dir / "method"
        reco_path = scan_dir / "pdata" / "1" / "reco"
        visu_path = scan_dir / "pdata" / "1" / "visu_pars"
        fid_path = scan_dir / "pdata" / "1" / "fid_proc.64"
        seq_path = scan_dir / "pdata" / "1" / "2dseq"

        method = read_text(method_path)
        reco = read_text(reco_path)
        visu = read_text(visu_path)
        npoints = int(first_array_value(method, "PVM_SpecMatrix"))
        sw_hz = first_array_value(method, "PVM_SpecSWH")
        sw_ppm = first_array_value(method, "PVM_SpecSW")
        center_ppm = first_array_value(method, "PVM_FrqWorkPpm")
        transmitter_mhz = first_array_value(method, "PVM_FrqWork")
        acquisition_ms = scalar(method, "PVM_SpecAcquisitionTime")
        dwell_us = first_array_value(method, "PVM_SpecDwellTime")
        reco_rotate_spectral = first_array_value(reco, "RECO_rotate")
        visu_slope = first_array_value(visu, "VisuCoreDataSlope")
        reco_slope = first_array_value(reco, "RECO_map_slope")

        if npoints != 256:
            raise AssertionError(f"Unexpected spectral points for {animal}/{scan_id}: {npoints}")
        if abs(reco_rotate_spectral - 0.5) > 1e-12:
            raise AssertionError(f"Unsupported spectral RECO_rotate for {animal}/{scan_id}: {reco_rotate_spectral}")

        fid = decode_fid(fid_path)
        magnitude = vendor_magnitude(fid)
        seq_physical = decode_2dseq(seq_path) * visu_slope
        mean_magnitude = magnitude.mean(axis=(1, 2, 3))

        # The sampled frequency offsets are determined, but the mapping from
        # Bruker's exponent/sign convention to increasing/decreasing ppm is not
        # claimed here. Both candidates are retained until independently verified.
        frequency_hz = np.fft.fftshift(np.fft.fftfreq(npoints, d=1.0 / sw_hz))
        ppm_plus = center_ppm + frequency_hz / transmitter_mhz
        ppm_minus = center_ppm - frequency_hz / transmitter_mhz
        peaks = local_peak_indices(mean_magnitude)

        key = f"{animal}_{scan_id}"
        archive[f"{key}_mean_magnitude"] = mean_magnitude.astype(np.float64)
        archive[f"{key}_frequency_offset_hz"] = frequency_hz.astype(np.float64)
        archive[f"{key}_ppm_candidate_plus"] = ppm_plus.astype(np.float64)
        archive[f"{key}_ppm_candidate_minus"] = ppm_minus.astype(np.float64)

        row: Dict[str, Any] = {
            "animal_id": animal,
            "scan_id": scan_id,
            "scan_dir": str(scan_dir),
            "fid_proc_sha256": sha256(fid_path),
            "method_sha256": sha256(method_path),
            "reco_sha256": sha256(reco_path),
            "visu_pars_sha256": sha256(visu_path),
            "spectral_points": npoints,
            "spectral_width_hz": sw_hz,
            "spectral_width_ppm": sw_ppm,
            "center_ppm": center_ppm,
            "transmitter_frequency_mhz": transmitter_mhz,
            "acquisition_time_ms": acquisition_ms,
            "dwell_time_us_header": dwell_us,
            "dwell_time_us_from_width": 1e6 / sw_hz,
            "nominal_ppm_step": sw_ppm / npoints,
            "reco_rotate_spectral": reco_rotate_spectral,
            "reco_map_slope": reco_slope,
            "visu_data_slope": visu_slope,
            "reconstruction_correlation": corrcoef(magnitude, seq_physical),
            "reconstruction_nrmse": nrmse(magnitude, seq_physical),
            "mean_spectrum_min": float(mean_magnitude.min()),
            "mean_spectrum_max": float(mean_magnitude.max()),
            "strongest_observed_index": int(np.argmax(mean_magnitude)),
            "strongest_observed_frequency_offset_hz": float(frequency_hz[np.argmax(mean_magnitude)]),
            "strongest_ppm_candidate_plus": float(ppm_plus[np.argmax(mean_magnitude)]),
            "strongest_ppm_candidate_minus": float(ppm_minus[np.argmax(mean_magnitude)]),
            "unassigned_local_peak_indices": peaks,
            "unassigned_peak_frequency_offsets_hz": [float(frequency_hz[idx]) for idx in peaks],
            "unassigned_peak_ppm_candidate_plus": [float(ppm_plus[idx]) for idx in peaks],
            "unassigned_peak_ppm_candidate_minus": [float(ppm_minus[idx]) for idx in peaks],
            "status": "PASS_OBSERVATION_ONLY" if corrcoef(magnitude, seq_physical) > 0.999999 else "FAIL_RECONSTRUCTION",
        }
        rows.append(row)

    scalar_fields = [
        "animal_id", "scan_id", "scan_dir", "fid_proc_sha256", "spectral_points",
        "spectral_width_hz", "spectral_width_ppm", "center_ppm", "transmitter_frequency_mhz",
        "acquisition_time_ms", "dwell_time_us_header", "dwell_time_us_from_width",
        "nominal_ppm_step", "reco_rotate_spectral", "reco_map_slope", "visu_data_slope",
        "reconstruction_correlation", "reconstruction_nrmse", "mean_spectrum_min",
        "mean_spectrum_max", "strongest_observed_index", "strongest_observed_frequency_offset_hz",
        "strongest_ppm_candidate_plus", "strongest_ppm_candidate_minus", "status",
    ]
    write_csv(Path(args.output_csv), ({k: row[k] for k in scalar_fields} for row in rows), scalar_fields)
    np.savez_compressed(Path(args.output_npz), **archive)

    summary = {
        "schema_version": 1,
        "evidence_scope": "local Bruker method/reco/visu_pars, fid_proc.64, 2dseq, and reproducible NumPy calculation",
        "source_reco_validation": {"path": str(source_path), "sha256": sha256(source_path)},
        "scan_count": len(rows),
        "pass_count": sum(row["status"] == "PASS_OBSERVATION_ONLY" for row in rows),
        "transform": "roll(abs(ifft(fid_proc.64, spectral_axis)), 128), exactly matching the previously validated vendor reco tail",
        "processing_applied": ["vendor-supported spectral transform", "spatial arithmetic mean for an audit spectrum"],
        "processing_not_applied": [
            "receiver/average reconstruction before fid_proc.64", "delay correction", "line broadening or other apodization",
            "zero filling", "phase correction", "frequency referencing", "baseline subtraction", "denoising",
            "peak assignment", "metabolite fitting or integration", "spatial masking", "T2 registration",
            "intensity normalization", "interpolation", "network inference", "network parameter update",
        ],
        "ppm_axis_status": {
            "center_ppm_from_method": True,
            "width_from_method_and_visu": True,
            "frequency_offsets_reproducible": True,
            "ppm_direction": "unresolved; both +/- candidates are stored",
            "reason": "The retained headers establish center and width, but this audit has not independently verified the ppm sign against an assigned spectral reference.",
        },
        "metabolite_assignment_status": "unresolved; reported peaks are deliberately unassigned",
        "formal_preprocessing_validated": False,
        "training_permission_changed": False,
        "minimum_reconstruction_correlation": min(row["reconstruction_correlation"] for row in rows),
        "maximum_reconstruction_nrmse": max(row["reconstruction_nrmse"] for row in rows),
        "rows": rows,
        "outputs": {"csv": str(Path(args.output_csv).resolve()), "npz": str(Path(args.output_npz).resolve())},
    }
    if summary["pass_count"] != 16:
        raise AssertionError(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
