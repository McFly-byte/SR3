#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process two in-vivo 2H MRSI mouse datasets and run the selected SR3 model.

The workflow intentionally keeps all reported metabolite maps in relative
signal units.  Neither dataset includes an external concentration reference or
paired high-resolution MRSI, so this script does not calculate mM, PSNR, or
SSIM.  Instead it saves native-acquisition consistency, stochastic stability,
background leakage, and edge/detail diagnostics.

Primary inputs
--------------
* Bruker 11.7 T: ParaVision ``pdata/1/fid_proc.64`` (9 x 9 x 5 x 256)
* UIH 9.4 T: standard DICOM MR Spectroscopy Storage (7 x 7 x 3 x 512)

The Bruker file is treated as image-domain voxel FIDs.  No spatial IFFT is
applied, matching the corrected water-film workflow.  Only the spectral/time
axis is transformed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_dilation, disk

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import core.logger as Logger  # noqa: E402
from core.mrsi_physics import crop_padded_native, mrsi_native_forward_batch  # noqa: E402
from core.sr_metrics import native_acquisition_consistency_2d  # noqa: E402
import model as Model  # noqa: E402


METABOLITE_WINDOWS: "OrderedDict[str, Tuple[float, float]]" = OrderedDict(
    [
        ("HDO", (4.35, 5.05)),
        ("Glc", (3.25, 4.15)),
        ("Glx", (1.95, 2.85)),
        ("Lac", (0.95, 1.65)),
    ]
)
METABOLITE_IDS = {name: idx for idx, name in enumerate(METABOLITE_WINDOWS)}
METABOLITE_COLORS = {
    "HDO": "#2563eb",
    "Glc": "#d97706",
    "Glx": "#7c3aed",
    "Lac": "#be185d",
}


@dataclass
class CaseData:
    case_id: str
    field_strength_t: float
    vendor: str
    fid: np.ndarray  # (slice, row, col, time), complex
    t2: np.ndarray  # (slice, row, col), float
    spectral_width_hz: float
    transmitter_frequency_mhz: float
    center_ppm: float
    ppm_sign: int
    native_fov_mm: Tuple[float, float, float]
    native_voxel_mm: Tuple[float, float, float]
    source_files: List[Path]
    metadata: Dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="replace")


def _jcamp_values(text: str, key: str) -> List[float]:
    pattern = re.compile(
        rf"^##\${re.escape(key)}\s*=\s*(.*?)(?=^##\$|^##END=|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    body = match.group(1)
    body = re.sub(r"^\s*\([^\n]*\)\s*", "", body)
    return [float(token) for token in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", body)]


def _jcamp_scalar(text: str, key: str, default: float | None = None) -> float:
    values = _jcamp_values(text, key)
    if values:
        return float(values[0])
    if default is None:
        raise KeyError(f"Missing Bruker parameter {key}")
    return float(default)


def _load_json_config(path: Path) -> OrderedDict:
    json_str = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def _build_model_opt(
    config_path: Path,
    run_dir: Path,
    resume_prefix: Path,
    output_dir: Path,
    gpu_id: str,
) -> Any:
    opt = _load_json_config(config_path)
    opt["phase"] = "val"
    opt["gpu_ids"] = [] if gpu_id.strip() == "" else [int(gpu_id)]
    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["distributed"] = False
    opt["is_main_process"] = True
    opt["path"]["experiments_root"] = str(output_dir.resolve())
    for key, rel in list(opt["path"].items()):
        if key in ("resume_state", "experiments_root"):
            continue
        opt["path"][key] = str((output_dir / str(rel)).resolve())
    opt["path"]["resume_state"] = str(resume_prefix.resolve())
    opt.setdefault("validation", {})["eval_network"] = "raw"
    opt["validation"]["native_data_consistency"] = {"enabled": False}
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.strip()
    return Logger.dict_to_nonedict(opt)


def _extract_enhanced_t2(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    slope, intercept = 1.0, 0.0
    try:
        transform = ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]
        slope = float(transform.RescaleSlope)
        intercept = float(transform.RescaleIntercept)
    except (AttributeError, IndexError, TypeError):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    frame_positions: List[List[float]] = []
    try:
        for frame in ds.PerFrameFunctionalGroupsSequence:
            frame_positions.append([float(x) for x in frame.PlanePositionSequence[0].ImagePositionPatient])
    except (AttributeError, IndexError):
        pass
    info = {
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "protocol_name": str(getattr(ds, "ProtocolName", "")),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "frames": int(arr.shape[0]),
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "frame_positions_patient_mm": frame_positions,
    }
    return arr, info


def _load_bruker_case(root: Path) -> CaseData:
    scan = root / "mouse_11.7T" / "30"
    method_path = scan / "method"
    acqp_path = scan / "acqp"
    reco_path = scan / "pdata" / "1" / "reco"
    fid_path = scan / "pdata" / "1" / "fid_proc.64"
    t2_path = root / "mouse_11.7T" / "T2_Axial2D_0.1W_EnIm1.dcm"
    method = _read_text(method_path)
    acqp = _read_text(acqp_path)
    matrix = [int(round(x)) for x in _jcamp_values(method, "PVM_Matrix")[:3]]
    if len(matrix) != 3:
        raise ValueError(f"Expected 3-D Bruker matrix, got {matrix}")
    nt_declared = int(round(_jcamp_scalar(method, "PVM_SpecMatrix")))
    fov = tuple(float(x) for x in _jcamp_values(method, "PVM_Fov")[:3])
    voxel = tuple(float(x) for x in _jcamp_values(method, "PVM_SpatResol")[:3])
    sw = _jcamp_scalar(method, "PVM_SpecSWH")
    bf1 = _jcamp_scalar(acqp, "BF1")
    sf0 = _jcamp_scalar(acqp, "SF0ppm")
    center_ppm = (bf1 - sf0) / sf0 * 1e6

    raw = np.fromfile(fid_path, dtype="<f8")
    if raw.size % 2:
        raise ValueError(f"Odd number of float64 values in {fid_path}")
    complex_values = raw[0::2] + 1j * raw[1::2]
    nvox = int(np.prod(matrix))
    if complex_values.size % nvox:
        raise ValueError("fid_proc.64 size is incompatible with PVM_Matrix")
    nt = complex_values.size // nvox
    if nt != nt_declared:
        logging.warning("Bruker declared Nt=%d but fid_proc gives Nt=%d", nt_declared, nt)
    # MATLAB/ParaVision ordering: [time, read, phase, slice].  Transpose to
    # conventional [slice, row(phase), column(read), time] for display/model use.
    fid_matlab = complex_values.reshape((nt, matrix[0], matrix[1], matrix[2]), order="F")
    fid = np.transpose(fid_matlab, (3, 2, 1, 0)).astype(np.complex64)
    t2, t2_info = _extract_enhanced_t2(t2_path)
    if t2.shape[0] != matrix[2]:
        raise ValueError(f"Bruker T2 frames {t2.shape[0]} != CSI slices {matrix[2]}")
    metadata = {
        "scan": 30,
        "nucleus": "2H",
        "matrix_read_phase_slice": matrix,
        "spectral_points_declared": nt_declared,
        "spectral_points_file": nt,
        "echo_time_ms": _jcamp_scalar(method, "PVM_EchoTime"),
        "repetition_time_ms": _jcamp_scalar(method, "PVM_RepetitionTime"),
        "averages": int(round(_jcamp_scalar(method, "PVM_NAverages"))),
        "spectral_acquisition_time_ms": _jcamp_scalar(method, "PVM_SpecAcquisitionTime"),
        "source_domain": "ParaVision image-domain voxel FID; no spatial IFFT",
        "center_ppm_formula": "(BF1-SF0ppm)/SF0ppm*1e6",
        "t2": t2_info,
        "reco_sha256": _sha256(reco_path),
    }
    return CaseData(
        case_id="mouse_11.7T",
        field_strength_t=11.7522625795724,
        vendor="Bruker BioSpin",
        fid=fid,
        t2=t2,
        spectral_width_hz=float(sw),
        transmitter_frequency_mhz=float(sf0),
        center_ppm=float(center_ppm),
        ppm_sign=1,
        native_fov_mm=fov,
        native_voxel_mm=voxel,
        source_files=[fid_path, method_path, acqp_path, reco_path, t2_path],
        metadata=metadata,
    )


def _repair_ppm_reference(value: float) -> Tuple[float, str]:
    original = float(value)
    repaired = original
    if not np.isfinite(repaired) or repaired == 0:
        return 4.7, "invalid value replaced with 4.7"
    # A deuterium chemical-shift reference should be on the order of a few
    # ppm.  The UIH export contains the correct mantissa with a corrupted
    # decimal exponent (4.616...e18), so keep shifting the decimal point until
    # it falls inside a deliberately broad spectroscopy range.
    while abs(repaired) > 20:
        repaired /= 10.0
    return float(repaired), f"decimal exponent repaired from {original:.12g}"


def _load_uih_t2_coronal(
    t2_dir: Path,
    target_y_positions: Sequence[float],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Path]]:
    records = []
    for path in sorted(t2_dir.glob("*.dcm")):
        ds = pydicom.dcmread(str(path), force=True)
        pos = [float(x) for x in ds.ImagePositionPatient]
        records.append((path, int(getattr(ds, "InstanceNumber", 0)), pos, ds.pixel_array.astype(np.float32)))
    if not records:
        raise FileNotFoundError(f"No DICOM files under {t2_dir}")
    selected = []
    images = []
    paths = []
    for target_y in target_y_positions:
        path, instance, pos, image = min(records, key=lambda item: abs(item[2][1] - target_y))
        images.append(image)
        paths.append(path)
        selected.append(
            {
                "target_y_mm": float(target_y),
                "selected_instance": int(instance),
                "selected_file": path.name,
                "selected_y_mm": float(pos[1]),
                "absolute_position_error_mm": float(abs(pos[1] - target_y)),
            }
        )
    return np.stack(images).astype(np.float32), selected, [item[0] for item in records]


def _load_uih_case(root: Path) -> CaseData:
    csi_path = root / "3601_csi_fid__3d_TR750" / "00000001.dcm"
    ds = pydicom.dcmread(str(csi_path), force=True)
    rows = int(ds.Rows)
    cols = int(ds.Columns)
    frames = int(ds.NumberOfFrames)
    points = int(ds.DataPointColumns)
    raw = np.frombuffer(ds.SpectroscopyData, dtype="<f4")
    expected = frames * rows * cols * points * 2
    if raw.size != expected:
        raise ValueError(f"DICOM spectroscopy floats={raw.size}, expected={expected}")
    complex_values = raw[0::2] + 1j * raw[1::2]
    fid = complex_values.reshape((frames, rows, cols, points), order="C").astype(np.complex64)

    center_ppm, reference_repair = _repair_ppm_reference(float(ds.ChemicalShiftReference))
    position = [float(x) for x in ds.ImagePositionPatient]
    slice_step_mm = float(ds.SliceThickness)
    # UIH stores the out-of-plane direction in private geometry as -Y here.
    target_y = [position[1] - idx * slice_step_mm for idx in range(frames)]
    t2, t2_selection, all_t2_paths = _load_uih_t2_coronal(
        root / "701_t2_fse_cor__0.15x1mm",
        target_y,
    )
    fov_xy = (float(ds.PixelSpacing[1]) * cols, float(ds.PixelSpacing[0]) * rows)
    fov_z = slice_step_mm * frames
    metadata = {
        "series_description": str(ds.SeriesDescription),
        "protocol_name": str(ds.ProtocolName),
        "nucleus": str(ds.ResonantNucleus),
        "matrix_row_col_frame": [rows, cols, frames],
        "spectral_points": points,
        "echo_time_ms": float(ds.EchoTime),
        "repetition_time_ms": float(ds.RepetitionTime),
        "averages": float(ds.NumberOfAverages),
        "signal_domain": str(ds.SignalDomainColumns),
        "data_representation": str(ds.DataRepresentation),
        "source_domain": "DICOM image-domain voxel FID; no spatial IFFT",
        "chemical_shift_reference_raw": float(ds.ChemicalShiftReference),
        "chemical_shift_reference_repair": reference_repair,
        "t2_slice_selection": t2_selection,
        "image_position_patient_mm": position,
        "image_orientation_patient_raw": [float(x) for x in ds.ImageOrientationPatient],
    }
    return CaseData(
        case_id="mouse_9.4T",
        field_strength_t=float(ds.MagneticFieldStrength),
        vendor=str(ds.Manufacturer),
        fid=fid,
        t2=t2,
        spectral_width_hz=float(ds.SpectralWidth),
        transmitter_frequency_mhz=float(ds.TransmitterFrequency),
        center_ppm=float(center_ppm),
        ppm_sign=-1,
        native_fov_mm=(fov_xy[0], fov_xy[1], fov_z),
        native_voxel_mm=(float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), slice_step_mm),
        source_files=[csi_path, *all_t2_paths],
        metadata=metadata,
    )


def _normalize_t2_volume(volume: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(volume, dtype=np.float32)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size == 0:
        return np.zeros_like(arr), 0.0, 1.0
    lo, hi = np.percentile(positive, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _anatomy_mask(image01: np.ndarray) -> np.ndarray:
    image = np.asarray(image01, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0 or float(finite.max()) <= 0:
        return np.ones_like(image, dtype=bool)
    nonzero = finite[finite > 0]
    try:
        threshold = float(threshold_otsu(nonzero if nonzero.size > 16 else finite))
    except ValueError:
        threshold = 0.08
    threshold = max(0.03, min(threshold * 0.55, 0.35))
    mask = image > threshold
    mask = _largest_component(mask)
    radius = max(1, int(round(min(image.shape) / 80)))
    mask = binary_closing(mask, disk(radius * 2))
    mask = ndi.binary_fill_holes(mask)
    mask = binary_dilation(mask, disk(radius))
    ratio = float(mask.mean())
    if ratio < 0.03 or ratio > 0.90:
        mask = _largest_component(image > 0.03)
        mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def _resize_2d(array: np.ndarray, size: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).view(1, 1, *array.shape)
    if mode in ("bilinear", "bicubic"):
        out = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False)
    else:
        out = F.interpolate(tensor, size=(size, size), mode=mode)
    return out.squeeze().numpy().astype(np.float32)


def _resize_to_native(array: np.ndarray, shape: Tuple[int, int], mode: str = "area") -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).view(1, 1, *array.shape)
    if mode in ("bilinear", "bicubic"):
        out = F.interpolate(tensor, size=shape, mode=mode, align_corners=False)
    else:
        out = F.interpolate(tensor, size=shape, mode=mode)
    return out.squeeze().numpy().astype(np.float32)


def _prepare_spectra(
    case: CaseData,
    line_broadening_hz: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    fid = np.asarray(case.fid, dtype=np.complex64).copy()
    nt = fid.shape[-1]
    envelope = np.mean(np.abs(fid), axis=(0, 1, 2))
    delay = int(np.argmax(envelope[: max(4, nt // 2)]))
    if delay >= 3:
        fid = np.roll(fid, -delay, axis=-1)
        fid[..., -delay:] = 0
    else:
        delay = 0
    time_s = np.arange(nt, dtype=np.float32) / float(case.spectral_width_hz)
    apodization = np.exp(-np.pi * float(line_broadening_hz) * time_s).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(fid * apodization, axis=-1), axes=-1)
    magnitude = np.abs(spectra).astype(np.float32)
    hz = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / float(case.spectral_width_hz)))
    ppm = case.center_ppm + int(case.ppm_sign) * hz / float(case.transmitter_frequency_mhz)

    reference_shift = 0.0
    detected_hdo = None
    if case.case_id == "mouse_9.4T":
        mean_spectrum = magnitude.reshape(-1, nt).mean(axis=0)
        candidates = np.where(np.abs(ppm - 4.7) <= 0.9)[0]
        if candidates.size:
            peak_idx = int(candidates[np.argmax(mean_spectrum[candidates])])
            detected_hdo = float(ppm[peak_idx])
            reference_shift = 4.7 - detected_hdo
            ppm = ppm + reference_shift

    info = {
        "detected_delay_points": delay,
        "delay_rule": "argmax of spatially averaged FID envelope in first half; cyclic shift then zero wrapped tail",
        "line_broadening_hz": float(line_broadening_hz),
        "ppm_axis_sign": int(case.ppm_sign),
        "metadata_center_ppm": float(case.center_ppm),
        "detected_hdo_ppm_before_shift": detected_hdo,
        "water_reference_shift_ppm": float(reference_shift),
        "final_hdo_reference_ppm": 4.7 if detected_hdo is not None else None,
        "fid_envelope": envelope,
    }
    return magnitude, ppm.astype(np.float32), info


def _extract_relative_maps(
    magnitude: np.ndarray,
    ppm: np.ndarray,
    native_masks: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    nt = magnitude.shape[-1]
    noise_sel = ((ppm < 0.0) | (ppm > 8.0)) & (np.abs(ppm) < 0.9 * np.max(np.abs(ppm)))
    if int(noise_sel.sum()) < max(12, nt // 12):
        noise_sel = np.ones(nt, dtype=bool)
        for lo, hi in METABOLITE_WINDOWS.values():
            noise_sel &= ~((ppm >= lo) & (ppm <= hi))
    noise_values = magnitude[..., noise_sel]
    baseline = np.median(noise_values, axis=-1)
    noise_mad = np.median(np.abs(noise_values - baseline[..., None]), axis=-1)
    noise_sigma = np.maximum(1.4826 * noise_mad, np.finfo(np.float32).eps)
    corrected = np.maximum(magnitude - baseline[..., None], 0.0)
    ppm_step = float(np.median(np.abs(np.diff(ppm))))

    maps: Dict[str, np.ndarray] = {}
    snr_maps: Dict[str, np.ndarray] = {}
    window_info = {}
    for metabolite, (lo, hi) in METABOLITE_WINDOWS.items():
        sel = (ppm >= lo) & (ppm <= hi)
        if int(sel.sum()) < 2:
            raise ValueError(f"Too few spectral points for {metabolite}: {int(sel.sum())}")
        area = corrected[..., sel].sum(axis=-1) * ppm_step
        height = corrected[..., sel].max(axis=-1)
        area = np.where(native_masks, area, 0.0).astype(np.float32)
        snr = np.where(native_masks, height / noise_sigma, np.nan).astype(np.float32)
        maps[metabolite] = area
        snr_maps[metabolite] = snr
        window_info[metabolite] = {
            "ppm_low": lo,
            "ppm_high": hi,
            "spectral_points": int(sel.sum()),
            "ppm_step": ppm_step,
        }
    return maps, snr_maps, {
        "noise_ppm_point_count": int(noise_sel.sum()),
        "windows": window_info,
        "baseline_method": "per-voxel median magnitude in off-metabolite noise region",
        "peak_measure": "positive magnitude above baseline integrated over fixed ppm window",
    }


def _build_native_masks(t2_01: np.ndarray, native_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    masks64 = []
    native = []
    for image in t2_01:
        mask = _anatomy_mask(image)
        mask64 = _resize_2d(mask.astype(np.float32), 64, "nearest") >= 0.5
        occupancy = _resize_to_native(mask.astype(np.float32), native_hw, "area")
        native_mask = occupancy >= 0.20
        if native_mask.mean() < 0.10:
            native_mask = occupancy > 0.01
        masks64.append(mask64)
        native.append(native_mask)
    return np.stack(masks64), np.stack(native)


def _stack_condition(
    lr: torch.Tensor,
    t2: torch.Tensor,
    met_id: int,
    mask: torch.Tensor,
    dataset_opt: Mapping[str, Any],
) -> torch.Tensor:
    parts = []
    if dataset_opt.get("use_lr", True):
        parts.append(lr)
    if dataset_opt.get("use_t1", True):
        parts.append(t2)
    if dataset_opt.get("use_flair", True):
        parts.append(t2)
    if dataset_opt.get("use_met_onehot", True):
        met = torch.zeros((1, 4, *lr.shape[-2:]), device=lr.device, dtype=lr.dtype)
        met[:, int(met_id)] = 1.0
        parts.append(met)
    if dataset_opt.get("use_mask_channel", False):
        parts.append(mask)
    if not parts:
        raise ValueError("No condition channels are enabled")
    return torch.cat(parts, dim=1)


def _to_minus1_1(value: torch.Tensor) -> torch.Tensor:
    return value * 2.0 - 1.0


def _sr_to_01(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.dim() == 4 and tensor.shape[0] > 1:
        tensor = tensor[-1:]
    tensor = ((tensor.squeeze() + 1.0) * 0.5).clamp(0.0, 1.0)
    return tensor.numpy().astype(np.float32)


def _sample_seeds(sample_key: str, count: int) -> List[int]:
    base = zlib.crc32(sample_key.encode("utf-8")) & 0x7FFFFFFF
    return [int((base + idx * 104729) % 0x7FFFFFFF) for idx in range(count)]


def _run_sr_samples(
    diffusion: Any,
    dataset_opt: Mapping[str, Any],
    native_norm: np.ndarray,
    t2_64: np.ndarray,
    mask64: np.ndarray,
    metabolite: str,
    seeds: Sequence[int],
    sample_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    device = diffusion.device
    lr64 = _resize_2d(native_norm, 64, "bicubic").clip(0.0, 1.0)
    lr_t = torch.from_numpy(lr64).view(1, 1, 64, 64).to(device)
    t2_t = torch.from_numpy(t2_64.astype(np.float32)).view(1, 1, 64, 64).to(device)
    mask_t = torch.from_numpy(mask64.astype(np.float32)).view(1, 1, 64, 64).to(device)
    cond = _stack_condition(lr_t, t2_t, METABOLITE_IDS[metabolite], mask_t, dataset_opt)
    batch = {
        "HR": _to_minus1_1(lr_t),
        "SR": _to_minus1_1(cond),
        "LR": _to_minus1_1(lr_t),
        "MASK": mask_t,
    }
    outputs = []
    for seed in seeds:
        diffusion.feed_data(batch)
        diffusion.test(
            continous=False,
            seed=int(seed),
            sample_num_steps=int(sample_steps),
            network="raw",
        )
        outputs.append(_sr_to_01(diffusion.get_current_visuals(need_LR=True)["SR"]))
    stack = np.stack(outputs).astype(np.float32)
    return lr64, stack


def _gradient_energy(image: np.ndarray, mask: np.ndarray) -> float:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float32))
    grad = np.hypot(gx, gy)
    selected = grad[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else float(grad.mean())


def _high_frequency_fraction(image: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(image, dtype=np.float32) * np.asarray(mask, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices(x.shape)
    cy, cx = (np.asarray(x.shape) - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = radius >= (0.25 * min(x.shape))
    return float(power[high].sum() / max(power.sum(), np.finfo(np.float64).eps))


def _background_leakage(image: np.ndarray, mask: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    inside = float(np.mean(np.abs(image[mask]))) if mask.any() else float(np.mean(np.abs(image)))
    outside = float(np.mean(np.abs(image[~mask]))) if (~mask).any() else 0.0
    return outside / max(inside, 1e-8)


def _hotspot_extrapolation(image: np.ndarray, lr64: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    lr_values = lr64[mask]
    threshold = float(np.quantile(lr_values, 0.95))
    native_hot = ndi.binary_dilation((lr64 >= threshold) & mask, iterations=3)
    sr_threshold = float(np.quantile(image[mask], 0.95))
    sr_hot = (image >= sr_threshold) & mask
    return float(np.sum(sr_hot & ~native_hot) / max(np.sum(mask), 1))


def _forward_native_residual(
    image: np.ndarray,
    native: np.ndarray,
    native_mask: np.ndarray,
    window: str = "hamming",
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = int(native.shape[0])
    pred = torch.from_numpy(image).view(1, 1, *image.shape)
    padded = mrsi_native_forward_batch(pred, [n], window=window, clamp_nonnegative=True)
    projected = crop_padded_native(padded, n).squeeze().numpy()
    residual = (projected - native).astype(np.float32)
    values = residual[native_mask]
    target = native[native_mask]
    if values.size == 0:
        values = residual.reshape(-1)
        target = native.reshape(-1)
    metrics = {
        "native_projection_l1_manual": float(np.mean(np.abs(values))),
        "native_projection_rmse_manual": float(np.sqrt(np.mean(values**2))),
        "native_projection_bias_manual": float(np.mean(values)),
        "native_projection_rel_l1_manual": float(
            np.mean(np.abs(values)) / max(float(np.mean(np.abs(target))), 1e-8)
        ),
    }
    return residual, metrics


def _metrics_for_sample(
    native: np.ndarray,
    lr64: np.ndarray,
    sr_primary: np.ndarray,
    sr_mean: np.ndarray,
    sr_std: np.ndarray,
    seed_stack: np.ndarray,
    mask64: np.ndarray,
    native_mask: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray]:
    n = int(native.shape[0])
    mask_t = torch.from_numpy(mask64.astype(np.float32))
    native_t = torch.from_numpy(native.astype(np.float32))
    variants = {"bicubic": lr64, "sr_primary": sr_primary, "sr_ensemble": sr_mean}
    out: Dict[str, Any] = {}
    for label, image in variants.items():
        metrics = native_acquisition_consistency_2d(
            torch.from_numpy(image),
            native_t,
            n,
            mask=mask_t,
            window="hamming",
        )
        for key, value in metrics.items():
            out[f"{label}_{key}"] = float(value)
        out[f"{label}_gradient_mean"] = _gradient_energy(image, mask64)
        out[f"{label}_high_frequency_fraction"] = _high_frequency_fraction(image, mask64)
        out[f"{label}_background_leakage_ratio"] = _background_leakage(image, mask64)
        out[f"{label}_hotspot_extrapolation_rate"] = _hotspot_extrapolation(image, lr64, mask64)
    residual, residual_metrics = _forward_native_residual(sr_mean, native, native_mask)
    out.update({f"sr_ensemble_{key}": value for key, value in residual_metrics.items()})
    masked_std = sr_std[mask64]
    masked_mean = sr_mean[mask64]
    out["seed_pixel_std_mean"] = float(masked_std.mean()) if masked_std.size else float(sr_std.mean())
    out["seed_mean_cv"] = float(
        (masked_std.mean() if masked_std.size else sr_std.mean())
        / max(float(masked_mean.mean() if masked_mean.size else sr_mean.mean()), 1e-8)
    )
    per_seed_means = np.asarray([sample[mask64].mean() for sample in seed_stack], dtype=np.float64)
    out["seed_roi_mean_cv"] = float(per_seed_means.std() / max(abs(per_seed_means.mean()), 1e-8))
    out["sr_vs_bicubic_gradient_ratio"] = float(
        out["sr_ensemble_gradient_mean"] / max(out["bicubic_gradient_mean"], 1e-8)
    )
    out["sr_vs_bicubic_hf_ratio"] = float(
        out["sr_ensemble_high_frequency_fraction"]
        / max(out["bicubic_high_frequency_fraction"], 1e-8)
    )
    out["native_rel_l1_change_vs_bicubic"] = float(
        out["sr_ensemble_native_acquisition_rel_l1"]
        - out["bicubic_native_acquisition_rel_l1"]
    )
    return out, residual


def _plot_spectral_qc(
    case: CaseData,
    magnitude: np.ndarray,
    ppm: np.ndarray,
    spectral_info: Mapping[str, Any],
    output: Path,
) -> None:
    mean_spectrum = magnitude.reshape(-1, magnitude.shape[-1]).mean(axis=0)
    envelope = np.asarray(spectral_info["fid_envelope"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axes[0].plot(np.arange(envelope.size), envelope, color="#1f2937", linewidth=1.5)
    delay = int(spectral_info["detected_delay_points"])
    axes[0].axvline(delay, color="#d97706", linestyle="--", label=f"delay={delay}")
    axes[0].set(title=f"{case.case_id}: spatial mean FID envelope", xlabel="sample", ylabel="mean |FID|")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(ppm, mean_spectrum / max(float(mean_spectrum.max()), 1e-8), color="#111827")
    for metabolite, (lo, hi) in METABOLITE_WINDOWS.items():
        axes[1].axvspan(lo, hi, color=METABOLITE_COLORS[metabolite], alpha=0.17, label=metabolite)
    axes[1].set_xlim(8.0, 0.0)
    axes[1].set(
        title=f"{case.case_id}: mean magnitude spectrum",
        xlabel="chemical shift (ppm)",
        ylabel="normalized magnitude",
    )
    axes[1].legend(ncol=4, frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_t2_grid_qc(
    case: CaseData,
    t2_01: np.ndarray,
    native_masks: np.ndarray,
    output: Path,
) -> None:
    slices = t2_01.shape[0]
    fig, axes = plt.subplots(1, slices, figsize=(3.4 * slices, 3.5), squeeze=False, constrained_layout=True)
    rows, cols = native_masks.shape[-2:]
    for idx, ax in enumerate(axes[0]):
        ax.imshow(t2_01[idx], cmap="gray", vmin=0, vmax=1)
        h, w = t2_01[idx].shape
        for y in np.linspace(0, h, rows + 1):
            ax.axhline(y - 0.5, color="#f59e0b", linewidth=0.35, alpha=0.75)
        for x in np.linspace(0, w, cols + 1):
            ax.axvline(x - 0.5, color="#f59e0b", linewidth=0.35, alpha=0.75)
        ax.set_title(f"slice {idx} | {rows}x{cols} grid")
        ax.axis("off")
    fig.suptitle(f"{case.case_id}: metadata-matched T2 and native MRSI grid", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_sample_comparison(
    case_id: str,
    slice_idx: int,
    metabolite: str,
    native_au: np.ndarray,
    scale_au: float,
    t2_64: np.ndarray,
    lr64: np.ndarray,
    sr_primary: np.ndarray,
    sr_mean: np.ndarray,
    sr_std: np.ndarray,
    residual_native: np.ndarray,
    output: Path,
) -> None:
    lr_au = lr64 * scale_au
    sr_primary_au = sr_primary * scale_au
    sr_mean_au = sr_mean * scale_au
    std_au = sr_std * scale_au
    native64 = _resize_2d(native_au, 64, "nearest")
    vmax = float(np.percentile(np.concatenate([native64.ravel(), lr_au.ravel(), sr_mean_au.ravel()]), 99.5))
    vmax = max(vmax, 1e-8)
    diff = sr_mean_au - lr_au
    dlim = max(float(np.percentile(np.abs(diff), 99.0)), 1e-8)
    residual64 = _resize_2d(residual_native * scale_au, 64, "nearest")
    rlim = max(float(np.percentile(np.abs(residual64), 99.0)), 1e-8)
    panels = [
        (t2_64, "T2 anatomy", "gray", 0.0, 1.0),
        (native64, f"native {native_au.shape[0]}x{native_au.shape[1]}", "turbo", 0.0, vmax),
        (lr_au, "bicubic 64x64", "turbo", 0.0, vmax),
        (sr_primary_au, "SR primary seed", "turbo", 0.0, vmax),
        (sr_mean_au, "SR 5-seed mean", "turbo", 0.0, vmax),
        (std_au, "seed uncertainty (SD)", "magma", 0.0, max(float(np.percentile(std_au, 99)), 1e-8)),
        (residual64, "forward residual at native grid", "coolwarm", -rlim, rlim),
        (diff, "SR mean - bicubic", "coolwarm", -dlim, dlim),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), constrained_layout=True)
    for ax, (image, title, cmap, vmin, vmax_panel) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax_panel, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{case_id} | slice {slice_idx} | {metabolite} | relative signal (a.u.; within-case only)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_metabolite_montage(
    case_id: str,
    metabolite: str,
    records: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    count = len(records)
    fig, axes = plt.subplots(count, 4, figsize=(12.5, 3.0 * count), squeeze=False, constrained_layout=True)
    all_signal = np.concatenate(
        [
            np.asarray(record["native64_au"]).ravel()
            for record in records
        ]
        + [np.asarray(record["sr_mean_au"]).ravel() for record in records]
    )
    vmax = max(float(np.percentile(all_signal, 99.5)), 1e-8)
    for row, record in enumerate(records):
        images = [
            (record["t2_64"], "T2", "gray", 0.0, 1.0),
            (record["native64_au"], "native nearest", "turbo", 0.0, vmax),
            (record["lr64_au"], "bicubic", "turbo", 0.0, vmax),
            (record["sr_mean_au"], "SR 5-seed mean", "turbo", 0.0, vmax),
        ]
        for col, (image, title, cmap, vmin, vmax_panel) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax_panel, interpolation="nearest")
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"slice {record['slice_idx']}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col in (2, 3):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{case_id} | {metabolite} | common within-case scale (relative a.u.)", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["case_id"]), str(row["metabolite"])), []).append(row)
    keys = [
        "bicubic_native_acquisition_rel_l1",
        "sr_primary_native_acquisition_rel_l1",
        "sr_ensemble_native_acquisition_rel_l1",
        "native_rel_l1_change_vs_bicubic",
        "seed_pixel_std_mean",
        "seed_mean_cv",
        "seed_roi_mean_cv",
        "sr_vs_bicubic_gradient_ratio",
        "sr_vs_bicubic_hf_ratio",
        "sr_ensemble_background_leakage_ratio",
        "sr_ensemble_hotspot_extrapolation_rate",
        "native_peak_snr_median",
    ]
    output = []
    for (case_id, metabolite), items in sorted(groups.items()):
        record: Dict[str, Any] = {"case_id": case_id, "metabolite": metabolite, "slice_count": len(items)}
        for key in keys:
            values = np.asarray([float(item[key]) for item in items], dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
            record[f"{key}_std"] = float(finite.std()) if finite.size else float("nan")
        output.append(record)
    return output


def _plot_metric_summary(summary: Sequence[Mapping[str, Any]], output: Path) -> None:
    labels = [f"{row['case_id']}\n{row['metabolite']}" for row in summary]
    x = np.arange(len(summary))
    bic = np.asarray([row["bicubic_native_acquisition_rel_l1_mean"] for row in summary], dtype=float)
    sr = np.asarray([row["sr_ensemble_native_acquisition_rel_l1_mean"] for row in summary], dtype=float)
    seed_cv = np.asarray([row["seed_mean_cv_mean"] for row in summary], dtype=float)
    detail = np.asarray([row["sr_vs_bicubic_gradient_ratio_mean"] for row in summary], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, bic, width, label="bicubic", color="#94a3b8")
    axes[0].bar(x + width / 2, sr, width, label="SR 5-seed mean", color="#2563eb")
    axes[0].set_ylabel("native acquisition relative L1 (lower is better)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_title("Forward consistency with observed native metabolite map")
    axes[1].bar(x - width / 2, seed_cv, width, label="seed mean CV", color="#d97706")
    axes[1].bar(x + width / 2, detail, width, label="gradient ratio vs bicubic", color="#7c3aed")
    axes[1].axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_title("Stochastic stability and added spatial detail")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _environment_info() -> Dict[str, Any]:
    def run(command: Sequence[str]) -> str:
        try:
            return subprocess.check_output(command, cwd=_REPO_ROOT, text=True, encoding="utf-8", errors="replace").strip()
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"unavailable: {exc}"

    gpu = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpu.append({"index": idx, "name": torch.cuda.get_device_name(idx)})
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pydicom": pydicom.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": gpu,
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_status_short": run(["git", "status", "--short"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process 11.7T/9.4T in-vivo mouse 2H MRSI and run SR3")
    parser.add_argument("--data_root", default=r"D:\LMC\data\invivo_zlx")
    parser.add_argument(
        "--output_root",
        default=str(_REPO_ROOT / "paper_material" / "20260814组会" / "活体带病灶小鼠"),
    )
    parser.add_argument(
        "--run_dir",
        default=str(_REPO_ROOT / "experiments" / "models/current_mixed_i50500_raw"),
    )
    parser.add_argument(
        "--checkpoint_prefix",
        default=str(
            _REPO_ROOT
            / "experiments"
            / "models/current_mixed_i50500_raw"
            / "checkpoint"
            / "I50500_E1931"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            _REPO_ROOT
            / "experiments"
            / "models/current_mixed_i50500_raw"
            / "config_resolved.json"
        ),
    )
    parser.add_argument("--gpu_id", default="1")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed_count", type=int, default=5)
    parser.add_argument("--line_broadening_hz", type=float, default=10.0)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1, help="Debug only; -1 processes all")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_prefix = Path(args.checkpoint_prefix).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_file = Path(str(checkpoint_prefix) + "_gen.pth")
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ["data_info", "methods", "metrics", "numeric", "images/qc", "images/comparisons", "images/montages", "logs"]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    log_path = output_root / "logs" / "process_invivo_mice_sr.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    warnings.filterwarnings("once", category=UserWarning)
    logging.info("Output root: %s", output_root)
    logging.info("Checkpoint: %s", checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cases = [
        _load_bruker_case(data_root / "mouse_11.7T"),
        _load_uih_case(data_root / "mouse_9.4T"),
    ]
    logging.info("Loaded cases: %s", [case.case_id for case in cases])

    diffusion = None
    dataset_opt: Mapping[str, Any] = {}
    if not args.skip_inference:
        opt = _build_model_opt(config_path, run_dir, checkpoint_prefix, output_root / "model_runtime", args.gpu_id)
        diffusion = Model.create_model(opt)
        diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
        dataset_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
        logging.info("Loaded model on %s; eval network=%s", diffusion.device, diffusion.get_eval_network_name("raw"))

    inventory = []
    acquisition_rows = []
    hash_rows = []
    metric_rows: List[Dict[str, Any]] = []
    spectral_rows: List[Dict[str, Any]] = []
    montage_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    processed_count = 0

    for case in cases:
        logging.info("Processing %s", case.case_id)
        t2_01, t2_lo, t2_hi = _normalize_t2_volume(case.t2)
        native_hw = tuple(int(x) for x in case.fid.shape[1:3])
        masks64, native_masks = _build_native_masks(t2_01, native_hw)
        magnitude, ppm, spectral_info = _prepare_spectra(case, args.line_broadening_hz)
        maps, snr_maps, extraction_info = _extract_relative_maps(magnitude, ppm, native_masks)
        case.metadata["t2_normalization_p1"] = t2_lo
        case.metadata["t2_normalization_p99_5"] = t2_hi
        case.metadata["spectral_preprocessing"] = {
            key: value for key, value in spectral_info.items() if key != "fid_envelope"
        }
        case.metadata["metabolite_extraction"] = extraction_info
        case.metadata["native_anatomy_mask_voxel_counts"] = [int(mask.sum()) for mask in native_masks]

        _plot_spectral_qc(
            case,
            magnitude,
            ppm,
            spectral_info,
            output_root / "images" / "qc" / f"{case.case_id}_spectral_qc.png",
        )
        _plot_t2_grid_qc(
            case,
            t2_01,
            native_masks,
            output_root / "images" / "qc" / f"{case.case_id}_t2_grid_alignment.png",
        )

        scales = {}
        for metabolite, values in maps.items():
            selected = values[native_masks]
            selected = selected[np.isfinite(selected) & (selected > 0)]
            scale = float(np.percentile(selected, 99.0)) if selected.size else float(np.max(values))
            scales[metabolite] = max(scale, 1e-8)
        case.metadata["relative_normalization_scales_au"] = scales
        case_dir = output_root / "numeric" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "preprocessed_native_maps.npz",
            ppm=ppm,
            t2_raw=case.t2.astype(np.float32),
            t2_normalized=t2_01.astype(np.float32),
            mask_native=native_masks.astype(np.uint8),
            mask_64=masks64.astype(np.uint8),
            **{f"{key}_relative_peak_area_au": value for key, value in maps.items()},
            **{f"{key}_peak_snr": value for key, value in snr_maps.items()},
        )

        for slice_idx in range(case.fid.shape[0]):
            t2_64 = _resize_2d(t2_01[slice_idx], 64, "bicubic").clip(0, 1)
            for metabolite in METABOLITE_WINDOWS:
                if args.max_samples >= 0 and processed_count >= args.max_samples:
                    break
                native_au = maps[metabolite][slice_idx]
                scale = scales[metabolite]
                native_norm = np.clip(native_au / scale, 0.0, 1.0).astype(np.float32)
                seeds = _sample_seeds(f"{case.case_id}|{slice_idx}|{metabolite}", int(args.seed_count))
                if diffusion is None:
                    lr64 = _resize_2d(native_norm, 64, "bicubic").clip(0, 1)
                    seed_stack = np.repeat(lr64[None], len(seeds), axis=0)
                    sr_primary, sr_mean, sr_std = lr64.copy(), lr64.copy(), np.zeros_like(lr64)
                else:
                    lr64, seed_stack = _run_sr_samples(
                        diffusion,
                        dataset_opt,
                        native_norm,
                        t2_64,
                        masks64[slice_idx],
                        metabolite,
                        seeds,
                        int(args.sample_steps),
                    )
                    sr_primary = seed_stack[0]
                    sr_mean = seed_stack.mean(axis=0).astype(np.float32)
                    sr_std = seed_stack.std(axis=0).astype(np.float32)

                sample_metrics, residual_native = _metrics_for_sample(
                    native_norm,
                    lr64,
                    sr_primary,
                    sr_mean,
                    sr_std,
                    seed_stack,
                    masks64[slice_idx],
                    native_masks[slice_idx],
                )
                snr_values = snr_maps[metabolite][slice_idx][native_masks[slice_idx]]
                finite_snr = snr_values[np.isfinite(snr_values)]
                sample_metrics.update(
                    {
                        "case_id": case.case_id,
                        "field_strength_t": case.field_strength_t,
                        "slice_idx": slice_idx,
                        "metabolite": metabolite,
                        "native_rows": native_au.shape[0],
                        "native_columns": native_au.shape[1],
                        "normalization_scale_relative_au": scale,
                        "quantity_unit": "relative_signal_au",
                        "native_mask_voxels": int(native_masks[slice_idx].sum()),
                        "native_peak_snr_median": float(np.median(finite_snr)) if finite_snr.size else float("nan"),
                        "native_peak_snr_p25": float(np.percentile(finite_snr, 25)) if finite_snr.size else float("nan"),
                        "seed_count": len(seeds),
                        "primary_seed": seeds[0],
                        "sample_steps": int(args.sample_steps),
                        "eval_network": "raw",
                    }
                )
                metric_rows.append(sample_metrics)
                spectral_rows.append(
                    {
                        "case_id": case.case_id,
                        "slice_idx": slice_idx,
                        "metabolite": metabolite,
                        "ppm_low": METABOLITE_WINDOWS[metabolite][0],
                        "ppm_high": METABOLITE_WINDOWS[metabolite][1],
                        "relative_scale_au": scale,
                        "native_mask_voxels": int(native_masks[slice_idx].sum()),
                        "peak_snr_median": sample_metrics["native_peak_snr_median"],
                        "peak_snr_p25": sample_metrics["native_peak_snr_p25"],
                    }
                )

                stem = f"slice_{slice_idx:02d}_{metabolite}"
                np.savez_compressed(
                    case_dir / f"{stem}_sr_result.npz",
                    native_relative_au=native_au.astype(np.float32),
                    native_normalized=native_norm,
                    lr_bicubic_normalized=lr64,
                    lr_bicubic_relative_au=(lr64 * scale).astype(np.float32),
                    sr_primary_normalized=sr_primary,
                    sr_primary_relative_au=(sr_primary * scale).astype(np.float32),
                    sr_seed_stack_normalized=seed_stack,
                    sr_ensemble_mean_normalized=sr_mean,
                    sr_ensemble_mean_relative_au=(sr_mean * scale).astype(np.float32),
                    sr_seed_std_normalized=sr_std,
                    sr_seed_std_relative_au=(sr_std * scale).astype(np.float32),
                    t2_normalized_64=t2_64,
                    anatomy_mask_64=masks64[slice_idx].astype(np.uint8),
                    anatomy_mask_native=native_masks[slice_idx].astype(np.uint8),
                    forward_residual_native_normalized=residual_native,
                    normalization_scale_relative_au=np.float32(scale),
                    quantity_unit=np.array("relative_signal_au"),
                    metabolite=np.array(metabolite),
                    slice_idx=np.int32(slice_idx),
                    seeds=np.asarray(seeds, dtype=np.int64),
                    sample_steps=np.int32(args.sample_steps),
                    checkpoint=np.array(str(checkpoint_file)),
                )
                _plot_sample_comparison(
                    case.case_id,
                    slice_idx,
                    metabolite,
                    native_au,
                    scale,
                    t2_64,
                    lr64,
                    sr_primary,
                    sr_mean,
                    sr_std,
                    residual_native,
                    output_root / "images" / "comparisons" / f"{case.case_id}_{stem}_comparison.png",
                )
                montage_records.setdefault((case.case_id, metabolite), []).append(
                    {
                        "slice_idx": slice_idx,
                        "t2_64": t2_64,
                        "native64_au": _resize_2d(native_au, 64, "nearest"),
                        "lr64_au": lr64 * scale,
                        "sr_mean_au": sr_mean * scale,
                    }
                )
                logging.info(
                    "%s slice=%d met=%s native_relL1 bicubic=%.4f SRmean=%.4f seedCV=%.4f",
                    case.case_id,
                    slice_idx,
                    metabolite,
                    sample_metrics["bicubic_native_acquisition_rel_l1"],
                    sample_metrics["sr_ensemble_native_acquisition_rel_l1"],
                    sample_metrics["seed_mean_cv"],
                )
                processed_count += 1
            if args.max_samples >= 0 and processed_count >= args.max_samples:
                break

        for metabolite in METABOLITE_WINDOWS:
            records = montage_records.get((case.case_id, metabolite), [])
            if records:
                _plot_metabolite_montage(
                    case.case_id,
                    metabolite,
                    records,
                    output_root / "images" / "montages" / f"{case.case_id}_{metabolite}_all_slices.png",
                )

        inventory.append(
            {
                "case_id": case.case_id,
                "vendor": case.vendor,
                "field_strength_t": case.field_strength_t,
                "fid_shape_slice_row_col_time": list(case.fid.shape),
                "t2_shape_slice_row_col": list(case.t2.shape),
                "spectral_width_hz": case.spectral_width_hz,
                "transmitter_frequency_mhz": case.transmitter_frequency_mhz,
                "center_ppm_before_optional_water_reference": case.center_ppm,
                "ppm_sign": case.ppm_sign,
                "native_fov_mm": list(case.native_fov_mm),
                "native_voxel_mm": list(case.native_voxel_mm),
                "metadata": case.metadata,
            }
        )
        acquisition_rows.append(
            {
                "case_id": case.case_id,
                "vendor": case.vendor,
                "field_strength_T": case.field_strength_t,
                "nucleus": "2H",
                "native_matrix": "x".join(str(x) for x in case.fid.shape[1:3]),
                "slices": case.fid.shape[0],
                "spectral_points": case.fid.shape[-1],
                "spectral_width_Hz": case.spectral_width_hz,
                "transmitter_frequency_MHz": case.transmitter_frequency_mhz,
                "FOV_mm": "x".join(f"{x:g}" for x in case.native_fov_mm),
                "voxel_mm": "x".join(f"{x:g}" for x in case.native_voxel_mm),
                "TR_ms": case.metadata.get("repetition_time_ms"),
                "TE_ms": case.metadata.get("echo_time_ms"),
                "averages": case.metadata.get("averages"),
                "data_domain": case.metadata.get("source_domain"),
            }
        )
        for source in case.source_files:
            hash_rows.append(
                {
                    "case_id": case.case_id,
                    "source_file": str(source),
                    "bytes": source.stat().st_size,
                    "mtime": source.stat().st_mtime,
                    "sha256": _sha256(source),
                }
            )

    summary_rows = _summarize_metrics(metric_rows)
    _write_csv(output_root / "metrics" / "per_slice_sr_metrics.csv", metric_rows)
    _write_csv(output_root / "metrics" / "summary_by_case_metabolite.csv", summary_rows)
    _write_csv(output_root / "metrics" / "spectral_qc_by_slice.csv", spectral_rows)
    _write_csv(output_root / "data_info" / "acquisition_parameters.csv", acquisition_rows)
    _write_csv(output_root / "data_info" / "source_hashes.csv", hash_rows)
    _write_json(output_root / "data_info" / "dataset_inventory.json", {"cases": inventory})
    _write_json(output_root / "data_info" / "environment.json", _environment_info())
    _plot_metric_summary(summary_rows, output_root / "images" / "metrics_summary.png")

    checkpoint_hash = _sha256(checkpoint_file)
    run_metadata = {
        "command": " ".join(sys.argv),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_hash,
        "eval_network": "raw",
        "sample_steps": int(args.sample_steps),
        "seed_count": int(args.seed_count),
        "seed_rule": "CRC32(case|slice|metabolite) + 104729*k, deterministic",
        "line_broadening_hz": float(args.line_broadening_hz),
        "quantity_unit": "relative_signal_au",
        "important_limitations": [
            "no external concentration calibration; values are not mM",
            "no paired HR MRSI; PSNR/SSIM are intentionally not calculated",
            "T2 is duplicated into the model T1 and FLAIR condition channels",
            "native matrices 7x7 and 9x9 are below the training range 16x16/24x24/32x32",
            "no lesion annotation; no lesion-specific ROI metric is reported",
        ],
        "processed_sample_count": len(metric_rows),
    }
    _write_json(output_root / "run_metadata.json", run_metadata)
    shutil.copy2(config_path, output_root / "methods" / "model_config_resolved.json")
    shutil.copy2(Path(__file__), output_root / "methods" / Path(__file__).name)
    (output_root / "methods" / "command.txt").write_text(run_metadata["command"] + "\n", encoding="utf-8")
    logging.info("Completed %d slice-metabolite samples", len(metric_rows))
    logging.info("Checkpoint SHA256: %s", checkpoint_hash)
    logging.info("Metrics: %s", output_root / "metrics" / "summary_by_case_metabolite.csv")


if __name__ == "__main__":
    main()
