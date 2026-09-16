#!/usr/bin/env python3
"""Validate Bruker rawdata.job0 -> fid_proc.64 without modifying source data.

The candidate space is deliberately constrained by the local ParaVision reco graph:
RecoAverageFilter(AverageList) -> RecoSortFilter(RecoSortMaps) -> 3-D reformat ->
per-axis vendor user window -> fractional FT shift -> complex FT -> first-order
phase correction -> fid_proc.64.

Unknown numerical conventions (complex order, averaging normalization, FT sign and
FT-shift convention) are compared against the retained vendor fid_proc.64.  The
script reports candidates and does not grant training permission by itself.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="replace")


def bruker_raw(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^##\${re.escape(key)}=(.*)$", text)
    if not match:
        return None
    lines = [match.group(1).strip()]
    for line in text[match.end():].splitlines():
        if line.startswith("##"):
            break
        if not line.startswith("$$"):
            lines.append(line.strip())
    return " ".join(x for x in lines if x).strip()


def strip_shape(raw: str | None) -> str:
    if raw is None:
        return ""
    return re.sub(r"^\(\s*[^)]*\)\s*", "", raw).strip()


def expand_repetitions(value: str) -> str:
    pattern = re.compile(r"@(\d+)\*\(([^()]*)\)")
    while True:
        match = pattern.search(value)
        if not match:
            return value
        value = value[:match.start()] + " ".join([match.group(2).strip()] * int(match.group(1))) + value[match.end():]


def parse_numbers(raw: str | None) -> list[float]:
    value = expand_repetitions(strip_shape(raw))
    out: list[float] = []
    for token in value.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den else float("nan")


def best_scale_nrmse(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    x = np.asarray(pred).ravel()
    y = np.asarray(ref).ravel()
    alpha = np.vdot(x, y) / (np.vdot(x, x) + 1e-30)
    rel = float(np.linalg.norm(alpha * x - y) / (np.linalg.norm(y) + 1e-30))
    return float(abs(alpha)), rel


def decode_vendor_fid(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    expected = 2 * 256 * 5 * 9 * 9
    if values.size != expected:
        raise ValueError(f"Unexpected fid_proc.64 float64 count: {values.size} != {expected}")
    data = values[0::2] + 1j * values[1::2]
    return np.moveaxis(data.reshape(5, 9, 9, 256), -1, 0)


def decode_raw(path: Path, complex_order: str) -> np.ndarray:
    values = np.fromfile(path, dtype="<i4")
    if values.size % (2 * 256):
        raise ValueError(f"rawdata.job0 int32 count is not divisible by 512: {values.size}")
    blocks = values.reshape(-1, 256, 2)
    if complex_order == "ri":
        return blocks[..., 0].astype(np.float64) + 1j * blocks[..., 1]
    if complex_order == "ir":
        return blocks[..., 1].astype(np.float64) + 1j * blocks[..., 0]
    raise ValueError(complex_order)


def aggregate_contiguous(raw: np.ndarray, counts: np.ndarray, mode: str) -> np.ndarray:
    if int(counts.sum()) != raw.shape[0]:
        raise ValueError(f"AverageList sum {counts.sum()} != raw blocks {raw.shape[0]}")
    out = np.empty((counts.size, raw.shape[1]), dtype=np.complex128)
    start = 0
    for idx, count in enumerate(counts.tolist()):
        group = raw[start:start + count]
        out[idx] = group.mean(axis=0) if mode == "mean" else group.sum(axis=0)
        start += count
    return out


def aggregate_cycle(raw: np.ndarray, counts: np.ndarray, mode: str) -> np.ndarray:
    """Alternative ordering: one accumulation for each still-active encode per cycle."""
    buckets: list[list[np.ndarray]] = [[] for _ in range(counts.size)]
    cursor = 0
    for cycle in range(1, int(counts.max()) + 1):
        for idx, count in enumerate(counts.tolist()):
            if count >= cycle:
                buckets[idx].append(raw[cursor])
                cursor += 1
    if cursor != raw.shape[0]:
        raise ValueError(f"Cycle aggregation consumed {cursor} of {raw.shape[0]} blocks")
    out = np.empty((counts.size, raw.shape[1]), dtype=np.complex128)
    for idx, bucket in enumerate(buckets):
        group = np.stack(bucket)
        out[idx] = group.mean(axis=0) if mode == "mean" else group.sum(axis=0)
    return out


def phase_grid(length: int, shift: float, convention: str, sign: int) -> np.ndarray:
    if convention == "none":
        return np.ones(length, dtype=np.complex128)
    coordinate = np.arange(length, dtype=np.float64)
    amount = shift if convention == "full_rotate" else shift - 0.5
    return np.exp(sign * 2j * np.pi * amount * coordinate)


def reconstruct(
    encoded: np.ndarray,
    maps: np.ndarray,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    rotates: tuple[float, float, float],
    phase_correction_deg: tuple[float, float, float],
    *,
    use_window: bool,
    fft_direction: str,
    shift_convention: str,
    shift_sign: int,
    apply_phase_correction: bool,
) -> np.ndarray:
    grid = np.zeros((256, 5 * 9 * 9), dtype=np.complex128)
    grid[:, maps] = encoded.T
    grid = grid.reshape(256, 5, 9, 9)
    for axis, (window, rotate) in enumerate(zip(windows, rotates), start=1):
        if use_window:
            shape = [1, 1, 1, 1]
            shape[axis] = window.size
            grid = grid * window.reshape(shape)
        ramp = phase_grid(grid.shape[axis], rotate, shift_convention, shift_sign)
        shape = [1, 1, 1, 1]
        shape[axis] = ramp.size
        grid = grid * ramp.reshape(shape)
        if fft_direction == "fft":
            grid = np.fft.fft(grid, axis=axis)
        else:
            grid = np.fft.ifft(grid, axis=axis)
    if apply_phase_correction:
        # The local reco graph declares first-order 180 degree phase corrections
        # in all three spatial dimensions.  With zero-based output coordinates,
        # exp(i*pi*n) is the observed checkerboard sign.  The sign convention is
        # immaterial for exactly 180 degrees.
        indices = np.indices(grid.shape[1:])
        phase = sum(
            np.deg2rad(deg) * indices[axis]
            for axis, deg in enumerate(phase_correction_deg)
        )
        grid = grid * np.exp(1j * phase)[None]
    return grid


def fit_separable_phase(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Fit ref ~= pred * exp(i*(b0+bz*z+by*y+bx*x)) by weighted LS."""
    ratio = ref * np.conj(pred)
    weight = np.abs(ref) * np.abs(pred)
    # Average over spectral points before phase unwrapping; the missing vendor
    # operation is spatial and should be shared across all 256 FID samples.
    spatial = ratio.sum(axis=0)
    phase = np.angle(spatial)
    phase = np.unwrap(np.unwrap(np.unwrap(phase, axis=0), axis=1), axis=2)
    z, y, x = np.meshgrid(np.arange(5), np.arange(9), np.arange(9), indexing="ij")
    design = np.column_stack([np.ones(405), z.ravel(), y.ravel(), x.ravel()])
    w = np.sqrt(np.maximum(weight.sum(axis=0).ravel(), 0.0))
    beta, *_ = np.linalg.lstsq(design * w[:, None], phase.ravel() * w, rcond=None)
    correction = np.exp(1j * (beta[0] + beta[1] * z + beta[2] * y + beta[3] * x))[None]
    corrected = pred * correction
    alpha = np.vdot(corrected.ravel(), ref.ravel()) / (np.vdot(corrected.ravel(), corrected.ravel()) + 1e-30)
    rel = float(np.linalg.norm(alpha * corrected - ref) / (np.linalg.norm(ref) + 1e-30))
    return {
        "phase_intercept_rad": float(beta[0]),
        "phase_slope_z_rad_per_index": float(beta[1]),
        "phase_slope_y_rad_per_index": float(beta[2]),
        "phase_slope_x_rad_per_index": float(beta[3]),
        "phase_slopes_deg_per_index": [float(v * 180 / np.pi) for v in beta[1:]],
        "complex_rel_l2_after_phase_plane_and_scalar": rel,
        "complex_scalar_real": float(alpha.real),
        "complex_scalar_imag": float(alpha.imag),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    scan = args.scan_dir
    method = read_text(scan / "method")
    acqp = read_text(scan / "acqp")
    reco = read_text(scan / "pdata" / "1" / "reco")
    raw_path = scan / "rawdata.job0"
    fid_path = scan / "pdata" / "1" / "fid_proc.64"

    counts = np.asarray(parse_numbers(bruker_raw(method, "AverageList")), dtype=np.int64)
    sort_maps_all = np.asarray(parse_numbers(bruker_raw(reco, "RecoSortMaps")), dtype=np.int64)
    sort_size = np.asarray(parse_numbers(bruker_raw(reco, "RecoSortSize")), dtype=np.int64)
    sort_range = np.asarray(parse_numbers(bruker_raw(reco, "RecoSortRange")), dtype=np.int64)
    rotates_all = parse_numbers(bruker_raw(reco, "RECO_rotate"))
    user_window = np.asarray(parse_numbers(bruker_raw(reco, "RECO_usr_wdw")), dtype=np.float64)
    enc0 = np.asarray(parse_numbers(bruker_raw(method, "PVM_EncGenSteps0")), dtype=np.int64)
    enc1 = np.asarray(parse_numbers(bruker_raw(method, "PVM_EncGenSteps1")), dtype=np.int64)
    enc2 = np.asarray(parse_numbers(bruker_raw(method, "PVM_EncGenSteps2")), dtype=np.int64)

    if counts.size != 305 or int(counts.sum()) != 2570:
        raise AssertionError(f"Unexpected AverageList: size={counts.size}, sum={counts.sum()}")
    if sort_size.tolist() != [305, 1, 1] or sort_range.tolist() != [405, 1, 1]:
        raise AssertionError(f"Unexpected sort contract: size={sort_size}, range={sort_range}")
    maps = sort_maps_all[:305]
    coordinate_maps = (enc2 + 2) * 81 + (enc1 + 4) * 9 + (enc0 + 4)
    if not np.array_equal(maps, coordinate_maps):
        raise AssertionError("RecoSortMaps does not match PVM_EncGenSteps 0-based z,y,x flattening")
    if user_window.size != 4 * 256:
        raise AssertionError(f"Unexpected RECO_usr_wdw size: {user_window.size}")
    user_window = user_window.reshape(4, 256) / 2147483647.0
    windows = (user_window[3, :5], user_window[2, :9], user_window[1, :9])
    rotates = (rotates_all[3], rotates_all[2], rotates_all[1])
    phase_corr_pairs = parse_numbers(bruker_raw(reco, "RECO_pc_lin"))
    if len(phase_corr_pairs) != 8:
        raise AssertionError(f"Unexpected RECO_pc_lin pair encoding: {phase_corr_pairs}")
    # The JCAMP value is four (constant, first-order) pairs for
    # [spectral, x, y, z].  Only the second member of each pair is the
    # first-order phase used by RecoPhasCorrFilter.
    phase_corr_first_order = phase_corr_pairs[1::2]
    phase_correction = (
        phase_corr_first_order[3],
        phase_corr_first_order[2],
        phase_corr_first_order[1],
    )
    ref = decode_vendor_fid(fid_path)

    candidates: list[dict[str, Any]] = []
    for complex_order, aggregate_order, aggregate_mode in itertools.product(
        ("ri", "ir"), ("contiguous", "cycle"), ("mean", "sum")
    ):
        raw = decode_raw(raw_path, complex_order)
        encoded = (
            aggregate_contiguous(raw, counts, aggregate_mode)
            if aggregate_order == "contiguous"
            else aggregate_cycle(raw, counts, aggregate_mode)
        )
        for use_window, fft_direction, shift_convention, shift_sign, apply_phase_correction in itertools.product(
            (True, False), ("fft", "ifft"), ("full_rotate", "delta_rotate", "none"), (-1, 1), (True, False)
        ):
            pred = reconstruct(
                encoded,
                maps,
                windows,
                rotates,
                phase_correction,
                use_window=use_window,
                fft_direction=fft_direction,
                shift_convention=shift_convention,
                shift_sign=shift_sign,
                apply_phase_correction=apply_phase_correction,
            )
            scale, rel = best_scale_nrmse(pred, ref)
            candidates.append({
                "complex_order": complex_order,
                "aggregate_order": aggregate_order,
                "aggregate_mode": aggregate_mode,
                "use_vendor_window": use_window,
                "fft_direction": fft_direction,
                "shift_convention": shift_convention,
                "shift_sign": shift_sign,
                "apply_declared_phase_correction": apply_phase_correction,
                "magnitude_corr": pearson(np.abs(pred), np.abs(ref)),
                "magnitude_rel_l2_after_scale": best_scale_nrmse(np.abs(pred), np.abs(ref))[1],
                "complex_rel_l2_after_scalar": rel,
                "abs_complex_scale": scale,
            })
    candidates.sort(key=lambda row: (row["complex_rel_l2_after_scalar"], -row["magnitude_corr"]))

    top = candidates[:args.top]
    # Recompute the best complex candidate.  A valid vendor-equivalent branch
    # must agree in both complex values and magnitude, not merely in appearance.
    best = top[0]
    raw = decode_raw(raw_path, best["complex_order"])
    encoded = (
        aggregate_contiguous(raw, counts, best["aggregate_mode"])
        if best["aggregate_order"] == "contiguous"
        else aggregate_cycle(raw, counts, best["aggregate_mode"])
    )
    pred = reconstruct(
        encoded,
        maps,
        windows,
        rotates,
        phase_correction,
        use_window=best["use_vendor_window"],
        fft_direction=best["fft_direction"],
        shift_convention=best["shift_convention"],
        shift_sign=best["shift_sign"],
        apply_phase_correction=best["apply_declared_phase_correction"],
    )

    pass_thresholds = {
        "magnitude_corr_min": 0.999999999,
        "magnitude_rel_l2_after_scale_max": 1e-10,
        "complex_rel_l2_after_scalar_max": 1e-10,
    }
    passed = bool(
        best["magnitude_corr"] >= pass_thresholds["magnitude_corr_min"]
        and best["magnitude_rel_l2_after_scale"] <= pass_thresholds["magnitude_rel_l2_after_scale_max"]
        and best["complex_rel_l2_after_scalar"] <= pass_thresholds["complex_rel_l2_after_scalar_max"]
    )

    payload = {
        "schema_version": "1.0",
        "scan_dir": str(scan),
        "source_files": {
            "rawdata_job0": {"path": str(raw_path), "bytes": raw_path.stat().st_size},
            "fid_proc_64": {"path": str(fid_path), "bytes": fid_path.stat().st_size},
        },
        "first_party_contract": {
            "raw_complex_samples": raw_path.stat().st_size // 8,
            "spectral_points": 256,
            "average_list_size": int(counts.size),
            "average_list_sum": int(counts.sum()),
            "sort_size": sort_size.tolist(),
            "sort_range": sort_range.tolist(),
            "sort_maps_match_encoding_coordinates": True,
            "spatial_shape_zyx": [5, 9, 9],
            "vendor_window_zyx": [w.tolist() for w in windows],
            "reco_rotate_zyx": list(rotates),
            "reco_pc_lin_pairs_spectral_xyz_deg": np.asarray(phase_corr_pairs).reshape(4, 2).tolist(),
            "phase_correction_first_order_zyx_deg": list(phase_correction),
        },
        "candidate_count": len(candidates),
        "pass_thresholds": pass_thresholds,
        "best_complex_candidate": best,
        "best_candidate_phase_plane_diagnostic": fit_separable_phase(pred, ref),
        "top_candidates": top,
        "formal_vendor_frontend_reproduced": passed,
        "training_permission_changed": False,
        "limitations": [
            "Candidate ranking uses retained vendor fid_proc.64 as a read-only numerical reference.",
            "A magnitude match alone is insufficient; complex agreement and declared phase-correction equivalence are required.",
            "This script does not update network parameters and does not create an HR target.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
