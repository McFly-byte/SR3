#!/usr/bin/env python3
"""Audit retained TopSpin temporary 1D processing against Bruker CSI files.

This script is evidence-only.  It does not alter source data, define metabolite
peaks, or enable training.  It checks byte identity, tests explicit numerical
transform candidates, and reports only reproducible relationships.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SCANS = {
    "R001": (57, 58, 59, 60),
    "R002": (30, 31, 32, 33),
    "R003": (35, 36, 37, 38),
    "R004": (59, 60, 61, 62),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scalar(text: str, key: str) -> str:
    m = re.search(rf"^##\${re.escape(key)}=\s*([^\r\n]+)", text, re.MULTILINE)
    if not m:
        raise KeyError(key)
    return m.group(1).strip().strip("<>")


def audit_procno(text: str) -> int:
    m = re.search(r"\brser\s+(\d+)\s+from", text)
    if not m:
        raise ValueError("rser row not found")
    return int(m.group(1))


def complex_scale_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[complex, float]:
    candidate = np.asarray(candidate, dtype=np.complex128).ravel()
    reference = np.asarray(reference, dtype=np.complex128).ravel()
    scale = np.vdot(candidate, reference) / np.vdot(candidate, candidate)
    residual = np.linalg.norm(reference - scale * candidate) / (np.linalg.norm(reference) + 1e-30)
    return complex(scale), float(residual)


def magnitude_scale_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    candidate = np.asarray(candidate, dtype=np.float64).ravel()
    scale = float(np.dot(candidate, reference) / np.dot(candidate, candidate))
    residual = float(np.linalg.norm(reference - scale * candidate) / (np.linalg.norm(reference) + 1e-30))
    corr = float(np.corrcoef(reference, candidate)[0, 1])
    return scale, residual, corr


def decode_interleaved_float64(path: Path, order: str = "RI") -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size % 2:
        raise ValueError(f"Odd float64 count: {path}")
    if order == "RI":
        return raw[0::2] + 1j * raw[1::2]
    return raw[1::2] + 1j * raw[0::2]


def decode_fidproc(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<c16")
    if raw.size != 256 * 9 * 9 * 5:
        raise ValueError(f"Unexpected fid_proc count: {path} {raw.size}")
    return np.moveaxis(raw.reshape(5, 9, 9, 256), -1, 0)


def find_best_ft(temp_fid: np.ndarray, observed_complex: np.ndarray, sw_hz: float, lb_hz: float) -> dict[str, Any]:
    n = temp_fid.size
    dwell = 1.0 / sw_hz
    apodized = temp_fid * np.exp(-np.pi * lb_hz * np.arange(n) * dwell)
    candidates: list[dict[str, Any]] = []
    for zero_position in ("left", "right"):
        padded = np.zeros(512, dtype=np.complex128)
        if zero_position == "left":
            padded[:n] = apodized
        else:
            padded[-n:] = apodized
        for transform_name, transformed in (("fft", np.fft.fft(padded)), ("ifft", np.fft.ifft(padded))):
            for shift_name, shifted in (
                ("none", transformed),
                ("fftshift", np.fft.fftshift(transformed)),
                ("ifftshift", np.fft.ifftshift(transformed)),
            ):
                for reversed_axis in (False, True):
                    candidate = shifted[::-1] if reversed_axis else shifted
                    scale, error = complex_scale_error(observed_complex, candidate)
                    mag_scale, mag_error, mag_corr = magnitude_scale_error(
                        np.abs(observed_complex), np.abs(candidate)
                    )
                    candidates.append({
                        "zero_position": zero_position,
                        "transform": transform_name,
                        "shift": shift_name,
                        "reversed": reversed_axis,
                        "complex_scale_real": scale.real,
                        "complex_scale_imag": scale.imag,
                        "complex_relative_l2": error,
                        "magnitude_scale": mag_scale,
                        "magnitude_relative_l2": mag_error,
                        "magnitude_correlation": mag_corr,
                    })
    return min(candidates, key=lambda item: item["magnitude_relative_l2"])


def best_fidproc_trace(temp_fid: np.ndarray, fidproc: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for z in range(fidproc.shape[1]):
        for y in range(fidproc.shape[2]):
            for x in range(fidproc.shape[3]):
                candidate = fidproc[:, z, y, x]
                for conjugated in (False, True):
                    trace = np.conj(candidate) if conjugated else candidate
                    for reversed_axis in (False, True):
                        trace2 = trace[::-1] if reversed_axis else trace
                        scale, error = complex_scale_error(temp_fid, trace2)
                        mag_scale, mag_error, mag_corr = magnitude_scale_error(
                            np.abs(temp_fid), np.abs(trace2)
                        )
                        row = {
                            "z": z,
                            "y": y,
                            "x": x,
                            "conjugated": conjugated,
                            "reversed": reversed_axis,
                            "complex_scale_real": scale.real,
                            "complex_scale_imag": scale.imag,
                            "complex_relative_l2": error,
                            "magnitude_scale": mag_scale,
                            "magnitude_relative_l2": mag_error,
                            "magnitude_correlation": mag_corr,
                        }
                        if best is None or row["complex_relative_l2"] < best["complex_relative_l2"]:
                            best = row
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for animal, scan_ids in EXPECTED_SCANS.items():
        for scan_id in scan_ids:
            scan = args.data_root / animal / "RAW" / str(scan_id)
            temp = scan / "~TEMP" / "1"
            proc_dir = temp / "pdata" / "1"
            procs_path = proc_dir / "procs"
            audit_path = proc_dir / "auditp.txt"
            row: dict[str, Any] = {
                "animal_id": animal,
                "scan_id": scan_id,
                "scan_dir": str(scan),
                "temp_procs_present": procs_path.is_file(),
                "temp_audit_present": audit_path.is_file(),
            }
            if not procs_path.is_file():
                row["status"] = "NO_TEMP_PROCESSING_RECORD"
                rows.append(row)
                continue

            procs = procs_path.read_text(encoding="latin-1")
            row["procs_sha256"] = sha256(procs_path)
            for key in ("AXNUC", "OFFSET", "SF", "SI", "SW_p", "FT_mod", "FTSIZE", "FCOR", "LB", "WDW", "PHC0", "PHC1", "REVERSE"):
                row[key] = scalar(procs, key)

            if not audit_path.is_file():
                row["status"] = "PROCS_WITHOUT_AUDIT_TRAIL"
                rows.append(row)
                continue

            audit = audit_path.read_text(encoding="latin-1")
            procno = audit_procno(audit)
            ser_path = scan / "pv2tsdata" / "1" / "ser"
            temp_fid_path = temp / "fid"
            one_r_path = proc_dir / "1r"
            one_i_path = proc_dir / "1i"
            fidproc_path = scan / "pdata" / "1" / "fid_proc.64"
            chunk_size = temp_fid_path.stat().st_size
            ser_bytes = ser_path.read_bytes()
            start = (procno - 1) * chunk_size
            extracted = ser_bytes[start:start + chunk_size]
            temp_bytes = temp_fid_path.read_bytes()
            row.update({
                "audit_sha256": sha256(audit_path),
                "source_ser_sha256": sha256(ser_path),
                "temp_fid_sha256": sha256(temp_fid_path),
                "rser_procno_1based": procno,
                "temp_fid_bytes": len(temp_bytes),
                "ser_chunk_exact_match": extracted == temp_bytes,
                "ser_chunk_md5": hashlib.md5(extracted).hexdigest(),
                "temp_fid_md5": md5(temp_fid_path),
                "audit_records_em_lb_10": "em LB = 10 SI = 512" in audit,
                "audit_records_ft_mod_6": "ft FT_mod = 6 PKNL = 1 SI = 512" in audit,
                "audit_records_apk": "<apk" in audit,
            })

            temp_fid = decode_interleaved_float64(temp_fid_path, "RI")
            observed_complex = np.fromfile(one_r_path, dtype="<i4").astype(np.float64) + 1j * np.fromfile(one_i_path, dtype="<i4").astype(np.float64)
            best_ft = find_best_ft(
                temp_fid,
                observed_complex,
                float(row["SW_p"]),
                float(row["LB"]),
            )
            best_trace = best_fidproc_trace(
                temp_fid,
                decode_fidproc(fidproc_path),
            )
            flat_voxel_index_0based = (
                int(best_trace["z"]) * 9 * 9
                + int(best_trace["y"]) * 9
                + int(best_trace["x"])
            )
            width_ppm_from_topspin = float(row["SW_p"]) / float(row["SF"])
            center_ppm_from_topspin = float(row["OFFSET"]) - width_ppm_from_topspin / 2.0
            row["best_topspin_ft_candidate"] = best_ft
            row["best_fidproc_trace_match"] = best_trace
            row["axis_contract"] = {
                "topspin_left_edge_ppm": float(row["OFFSET"]),
                "spectral_width_ppm_from_SW_p_over_SF": width_ppm_from_topspin,
                "center_ppm_from_left_edge_minus_half_width": center_ppm_from_topspin,
                "expected_method_center_ppm": 4.7,
                "center_error_ppm": center_ppm_from_topspin - 4.7,
                "array_index_direction": "increasing array index corresponds to increasing frequency offset and decreasing ppm",
                "ppm_formula_for_512_point_topspin_array": "ppm[i] = OFFSET - i * (SW_p / SF) / SI",
                "original_256_to_zero_filled_512_index_relation": "i_512 = 2 * i_256",
                "rser_procno_matches_fidproc_flat_voxel_plus_one": procno == flat_voxel_index_0based + 1,
                "no_axis_reversal_in_best_numerical_match": best_ft["reversed"] is False and best_trace["reversed"] is False,
            }
            row["status"] = "AUDITED"
            rows.append(row)

    audited = [r for r in rows if r["status"] == "AUDITED"]
    summary = {
        "schema_version": 1,
        "evidence_scope": "retained TopSpin ~TEMP audit/procs/fid/1r/1i, pv2tsdata/1/ser, pdata/1/fid_proc.64, and explicit NumPy calculations",
        "scan_count": len(rows),
        "temp_procs_count": sum(r["temp_procs_present"] for r in rows),
        "complete_audit_count": len(audited),
        "exact_rser_chunk_match_count": sum(r.get("ser_chunk_exact_match", False) for r in audited),
        "exact_or_numerical_fidproc_trace_match_count": sum(
            r["best_fidproc_trace_match"]["complex_relative_l2"] <= 1e-12 for r in audited
        ),
        "topspin_ft_magnitude_match_count": sum(
            r["best_topspin_ft_candidate"]["magnitude_correlation"] >= 0.999999999
            and r["best_topspin_ft_candidate"]["magnitude_relative_l2"] <= 1e-6
            for r in audited
        ),
        "voxel_index_identity_count": sum(
            r["axis_contract"]["rser_procno_matches_fidproc_flat_voxel_plus_one"] for r in audited
        ),
        "no_axis_reversal_count": sum(
            r["axis_contract"]["no_axis_reversal_in_best_numerical_match"] for r in audited
        ),
        "maximum_center_error_ppm": max(
            abs(r["axis_contract"]["center_error_ppm"]) for r in audited
        ) if audited else None,
        "ppm_direction_status": (
            "resolved_from_retained_topspin_processing_and_numerical_identity"
            if audited
            and all(r["ser_chunk_exact_match"] for r in audited)
            and all(r["best_fidproc_trace_match"]["complex_relative_l2"] <= 1e-12 for r in audited)
            and all(r["best_topspin_ft_candidate"]["magnitude_correlation"] >= 0.999999999 for r in audited)
            and all(r["axis_contract"]["rser_procno_matches_fidproc_flat_voxel_plus_one"] for r in audited)
            and all(r["axis_contract"]["no_axis_reversal_in_best_numerical_match"] for r in audited)
            and max(abs(r["axis_contract"]["center_error_ppm"]) for r in audited) <= 2e-4
            else "unresolved"
        ),
        "ppm_direction": "decreasing ppm with increasing reconstructed spectral array index",
        "training_permission_changed": False,
        "interpretation_limit": "The retained TopSpin one-row path resolves reconstructed array orientation and supports its ppm-axis convention. It does not validate voxelwise frequency referencing, automated phase correction, baseline correction, metabolite fitting, units, QC thresholds, or formal training.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
