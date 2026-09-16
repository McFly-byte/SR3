#!/usr/bin/env python
"""Read-only numerical audit of Bruker ser -> fid_proc.64 -> 2dseq layout.

This script does not modify source data. It tests explicit candidate storage layouts
against vendor-generated intermediate/final files and records reproducible metrics.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


SPATIAL_SHAPES = [(5, 9, 9), (9, 9, 5)]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den else float("nan")


def best_scale_rmse(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    x = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(ref, dtype=np.float64).ravel()
    den = float(np.dot(x, x))
    scale = float(np.dot(x, y) / den) if den else float("nan")
    rmse = float(np.sqrt(np.mean((scale * x - y) ** 2)))
    nrmse = rmse / (float(np.mean(np.abs(y))) + 1e-12)
    return scale, nrmse


def load_ser(path: Path, endian: str, interleave: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(endian + "i4"))
    if raw.size != 405 * 1024:
        raise ValueError(f"Unexpected ser int32 count: {raw.size}")
    blocks = raw.reshape(405, 1024)[:, :512]
    if interleave == "ri":
        return blocks[:, 0::2].astype(np.float64) + 1j * blocks[:, 1::2]
    if interleave == "ir":
        return blocks[:, 1::2].astype(np.float64) + 1j * blocks[:, 0::2]
    raise ValueError(interleave)


def load_complex128(
    path: Path,
    endian: str,
    storage: str,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Decode candidate RecoFileSink(shuffle=true) layouts.

    The returned array follows VisuCoreSize order [spectroscopic, spatial...],
    with the spectroscopic dimension first in memory interpretation.
    """
    values = np.fromfile(path, dtype=np.dtype(endian + "f8"))
    if values.size != 2 * 256 * 405:
        raise ValueError(f"Unexpected fid_proc float64 count: {values.size}")
    if storage == "interleaved_spectral_fast":
        c = values[0::2] + 1j * values[1::2]
        return np.moveaxis(c.reshape(*shape, 256, order="C"), -1, 0)
    if storage == "global_split_spectral_fast":
        c = values[: 256 * 405] + 1j * values[256 * 405 :]
        return np.moveaxis(c.reshape(*shape, 256, order="C"), -1, 0)
    if storage == "voxel_block_ri":
        blocks = values.reshape(405, 512)
        c = blocks[:, :256] + 1j * blocks[:, 256:]
        return np.moveaxis(c.reshape(*shape, 256, order="C"), -1, 0)
    if storage == "voxel_block_ir":
        blocks = values.reshape(405, 512)
        c = blocks[:, 256:] + 1j * blocks[:, :256]
        return np.moveaxis(c.reshape(*shape, 256, order="C"), -1, 0)
    if storage == "spectral_block_ri":
        blocks = values.reshape(256, 810)
        c = blocks[:, :405] + 1j * blocks[:, 405:]
        return c.reshape(256, *shape, order="C")
    if storage == "spectral_block_ir":
        blocks = values.reshape(256, 810)
        c = blocks[:, 405:] + 1j * blocks[:, :405]
        return c.reshape(256, *shape, order="C")
    raise ValueError(storage)


def load_2dseq(path: Path, endian: str, order: str, shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(endian + "i2"))
    if raw.size != 256 * 405:
        raise ValueError(f"Unexpected 2dseq int16 count: {raw.size}")
    if order == "spectral_first":
        return raw.reshape(256, *shape, order="C").astype(np.float64)
    if order == "spectral_last":
        return np.moveaxis(raw.reshape(*shape, 256, order="C"), -1, 0).astype(np.float64)
    raise ValueError(order)


def fft_axis(x: np.ndarray, axis: int, direction: str, shift: str) -> np.ndarray:
    y = x
    if shift == "ifftshift_before":
        y = np.fft.ifftshift(y, axes=axis)
    elif shift == "fftshift_before":
        y = np.fft.fftshift(y, axes=axis)
    elif shift != "none":
        raise ValueError(shift)
    y = np.fft.fft(y, axis=axis) if direction == "fft" else np.fft.ifft(y, axis=axis)
    return y


def compare_complex(pred: np.ndarray, ref: np.ndarray) -> dict:
    p = pred.ravel()
    r = ref.ravel()
    alpha = np.vdot(p, r) / (np.vdot(p, p) + 1e-30)
    rel = float(np.linalg.norm(alpha * p - r) / (np.linalg.norm(r) + 1e-30))
    return {
        "complex_rel_l2_after_scalar": rel,
        "magnitude_corr": pearson(np.abs(pred), np.abs(ref)),
        "complex_scalar_real": float(alpha.real),
        "complex_scalar_imag": float(alpha.imag),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scan = args.scan_dir
    ser_path = scan / "pv2tsdata" / "1" / "ser"
    fid_path = scan / "pdata" / "1" / "fid_proc.64"
    seq_path = scan / "pdata" / "1" / "2dseq"

    results: dict[str, object] = {
        "scan_dir": str(scan),
        "sizes": {p.name: p.stat().st_size for p in (ser_path, fid_path, seq_path)},
        "ser_to_fid_candidates": [],
        "fid_to_2dseq_candidates": [],
    }

    # ser is a pv2tsdata export, whereas reco consumes rawdata.job0. This
    # comparison is diagnostic only and must not be interpreted as a complete
    # vendor reconstruction because AverageList, RecoSortMaps, user windows and
    # fractional RECO_rotate are deliberately not approximated here.
    fid_storages = (
        "interleaved_spectral_fast",
        "global_split_spectral_fast",
        "voxel_block_ri",
        "voxel_block_ir",
        "spectral_block_ri",
        "spectral_block_ir",
    )
    for ser_endian, interleave, shape, fid_endian, fid_storage in itertools.product(
        ("<", ">"), ("ri", "ir"), SPATIAL_SHAPES, ("<", ">"), fid_storages
    ):
        ser = load_ser(ser_path, ser_endian, interleave)
        ref = load_complex128(fid_path, fid_endian, fid_storage, shape)
        base = ser.T.reshape(256, *shape)
        for direction, shift in itertools.product(("fft", "ifft"), ("none", "ifftshift_before", "fftshift_before")):
            pred = base
            for axis in (1, 2, 3):
                pred = fft_axis(pred, axis, direction, shift)
            metrics = compare_complex(pred, ref)
            results["ser_to_fid_candidates"].append({
                "ser_endian": ser_endian,
                "interleave": interleave,
                "shape": list(shape),
                "fid_endian": fid_endian,
                "fid_storage": fid_storage,
                "direction": direction,
                "shift": shift,
                **metrics,
            })

    results["ser_to_fid_candidates"].sort(key=lambda x: x["complex_rel_l2_after_scalar"])

    # Audit vendor spectral FFT/magnitude/mapping stage. VisuCoreSize declares
    # [256, 9, 9, 5] with the spectroscopic dimension first (fastest on disk).
    # We also search a circular spectral roll because RecoFTShiftFilter(shift=0.5)
    # may express its output placement differently from NumPy's shift helpers.
    # Hard-constrained by VisuCoreByteOrder=littleEndian and
    # VisuCoreSize=[256,9,9,5] with the first dimension fastest on disk.
    shape = (5, 9, 9)
    seq_endian = "<"
    seq_order = "spectral_last"
    for fid_storage in fid_storages:
        fid = load_complex128(fid_path, "<", fid_storage, shape)
        ref = load_2dseq(seq_path, seq_endian, seq_order, shape)
        for direction, reverse in itertools.product(("fft", "ifft"), (False, True)):
            spec = fft_axis(fid, 0, direction, "none")
            pred0 = np.abs(spec[::-1] if reverse else spec)
            best = None
            for roll in range(256):
                pred = np.roll(pred0, roll, axis=0)
                corr = pearson(pred, ref)
                scale, nrmse = best_scale_rmse(pred, ref)
                candidate = (corr, -nrmse, roll, scale, nrmse)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            assert best is not None
            corr, _, roll, scale, nrmse = best
            results["fid_to_2dseq_candidates"].append({
                "shape": list(shape),
                "fid_endian": "<",
                "fid_storage": fid_storage,
                "seq_endian": seq_endian,
                "seq_order": seq_order,
                "direction": direction,
                "shift": "none",
                "spectral_reverse": reverse,
                "spectral_roll": roll,
                "corr": corr,
                "best_scale": scale,
                "nrmse_after_scale": nrmse,
            })

    results["fid_to_2dseq_candidates"].sort(key=lambda x: (-x["corr"], x["nrmse_after_scale"]))
    results["best_ser_to_fid"] = results["ser_to_fid_candidates"][:10]
    results["best_fid_to_2dseq"] = results["fid_to_2dseq_candidates"][:10]
    # Avoid an unwieldy audit file while retaining the decisive candidates.
    results.pop("ser_to_fid_candidates")
    results.pop("fid_to_2dseq_candidates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
