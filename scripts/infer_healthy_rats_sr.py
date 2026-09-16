#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run healthy-rat 2H MRSI through the healthy-phantom SR3 model (EMA).

Adapted from process_invivo_mice_sr.py for the ZLX healthy rat cohort.
Key differences:
  - Bruker RAW/<scan>/pdata/1/fid_proc.64 (9x9x5x256, image-domain voxel FID)
  - T2 axial high-res DICOM (36 slices x 180x180), proportionally matched to 5 CSI slices
  - Metabolite one-hot contract: Glx=0, Glc=1, Lac=2, Lipid=3 (NOT HDO)
  - Single seed 0, DDIM 50 steps, EMA weights
  - Output under the data root inference/ directory
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_dilation, disk

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import core.logger as Logger  # noqa: E402
from core.mrsi_physics import (  # noqa: E402
    center_pad_native, crop_padded_native, mrsi_native_forward_batch,
    refine_native_data_consistency,
)
from core.sr_metrics import native_acquisition_consistency_2d  # noqa: E402
import model as Model  # noqa: E402

# ── Metabolite contract (matches training DEFAULT_METS) ──────────────────────
METABOLITE_WINDOWS: "OrderedDict[str, Tuple[float, float]]" = OrderedDict([
    ("Glx",   (1.95, 2.85)),
    ("Glc",   (3.25, 4.15)),
    ("Lac",   (0.95, 1.65)),
    ("Lipid", (0.20, 0.90)),
])
METABOLITE_IDS = {name: idx for idx, name in enumerate(METABOLITE_WINDOWS)}
METABOLITE_COLORS = {
    "Glx": "#7c3aed", "Glc": "#d97706", "Lac": "#be185d", "Lipid": "#059669",
}

DATA_ROOT = Path(r"D:\LMC\data\invivo_zlx\zlx_healthy_rats_data")
OUTPUT_ROOT = DATA_ROOT / "inference"
CONFIG_PATH = _REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "config_resolved.json"
CKPT_PREFIX = _REPO_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema" / "checkpoint" / "I300000_E2522"

RAT_STUDY_DIRS = {
    "R001": "20260705_ZLX_brain_R001_E6_P1",
    "R002": "20260709_ZLX_brain_R002_E9_P1",
    "R003": "20260709_ZLX_brain_R003_E6_P1",
    "R004": "20260715_ZLX_brain_R004_E13_P1",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _json_default(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    raise TypeError(type(v))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            c = f.read(4 * 1024 * 1024)
            if not c: break
            d.update(c)
    return d.hexdigest().upper()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="replace")


def _jcamp_values(text: str, key: str) -> List[float]:
    pat = re.compile(rf"^##\${re.escape(key)}\s*=\s*(.*?)(?=^##\$|^##END=|\Z)", re.MULTILINE | re.DOTALL)
    m = pat.search(text)
    if not m: return []
    body = re.sub(r"^\s*\([^\n]*\)\s*", "", m.group(1))
    return [float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", body)]


def _jcamp_scalar(text: str, key: str, default=None) -> float:
    vals = _jcamp_values(text, key)
    if vals: return float(vals[0])
    if default is not None: return float(default)
    raise KeyError(key)


def _load_json_config(path: Path):
    s = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s += line.split("//")[0] + "\n"
    return json.loads(s, object_pairs_hook=OrderedDict)


# ── Data loading ─────────────────────────────────────────────────────────────
@dataclass
class ScanData:
    rat_id: str
    scan_num: int
    fid: np.ndarray          # (slice, row, col, time) complex64
    t2_volume: np.ndarray    # (n_t2_slices, H, W) float32
    t2_matched: np.ndarray   # (5, H, W) float32 — matched to CSI slices
    spectral_width_hz: float
    transmitter_frequency_mhz: float
    center_ppm: float
    ppm_sign: int
    native_fov_mm: Tuple[float, float, float]
    native_voxel_mm: Tuple[float, float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_files: List[Path] = field(default_factory=list)


def _extract_t2(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2: arr = arr[None]
    slope, intercept = 1.0, 0.0
    try:
        tr = ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]
        slope = float(tr.RescaleSlope); intercept = float(tr.RescaleIntercept)
    except (AttributeError, IndexError, TypeError):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    info = {
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "rows": int(ds.Rows), "columns": int(ds.Columns),
        "frames": int(arr.shape[0]),
        "slice_thickness_mm": float(getattr(ds, "SliceThickness", 0.5)),
    }
    return arr, info


def _match_t2_to_csi(t2_vol: np.ndarray, n_csi_slices: int = 5) -> np.ndarray:
    """Proportionally match high-res T2 slices to CSI slices.
    CSI: 5 slices x 3.6mm = 18mm; T2: 36 slices x 0.5mm = 18mm.
    Take the T2 slice whose center is closest to each CSI slice center."""
    n_t2 = t2_vol.shape[0]
    t2_thick = 18.0 / n_t2
    csi_thick = 18.0 / n_csi_slices
    matched = []
    for z in range(n_csi_slices):
        csi_center = z * csi_thick + csi_thick / 2.0
        t_idx = int(round((csi_center - t2_thick / 2.0) / t2_thick))
        t_idx = max(0, min(n_t2 - 1, t_idx))
        matched.append(t2_vol[t_idx])
    return np.stack(matched).astype(np.float32)


def _load_bruker_scan(rat_id: str, scan_num: int) -> ScanData:
    scan_dir = DATA_ROOT / rat_id / "RAW" / str(scan_num)
    method_path = scan_dir / "method"
    acqp_path = scan_dir / "acqp"
    reco_path = scan_dir / "pdata" / "1" / "reco"
    fid_path = scan_dir / "pdata" / "1" / "fid_proc.64"
    t2_path = DATA_ROOT / rat_id / RAT_STUDY_DIRS[rat_id] / "T2_Axial_AutoW_EnIm1.dcm"

    method = _read_text(method_path)
    acqp = _read_text(acqp_path)
    matrix = [int(round(x)) for x in _jcamp_values(method, "PVM_Matrix")[:3]]
    nt_decl = int(round(_jcamp_scalar(method, "PVM_SpecMatrix")))
    fov = tuple(float(x) for x in _jcamp_values(method, "PVM_Fov")[:3])
    voxel = tuple(float(x) for x in _jcamp_values(method, "PVM_SpatResol")[:3])
    sw = _jcamp_scalar(method, "PVM_SpecSWH")
    bf1 = _jcamp_scalar(acqp, "BF1")
    sf0 = _jcamp_scalar(acqp, "SF0ppm")
    center_ppm = (bf1 - sf0) / sf0 * 1e6

    raw = np.fromfile(fid_path, dtype="<f8")
    cvals = raw[0::2] + 1j * raw[1::2]
    nvox = int(np.prod(matrix))
    nt = cvals.size // nvox
    # MATLAB/ParaVision: [time, read, phase, slice] -> [slice, phase, read, time]
    fid_m = cvals.reshape((nt, matrix[0], matrix[1], matrix[2]), order="F")
    fid = np.transpose(fid_m, (3, 2, 1, 0)).astype(np.complex64)

    t2_vol, t2_info = _extract_t2(t2_path)
    t2_matched = _match_t2_to_csi(t2_vol, fid.shape[0])

    # Determine scan phase / FA from method
    fa = _jcamp_scalar(method, "PVM_ExcPulse1", default=0.0)
    # Try to get flip angle from method (variable names vary)
    scan_phase = "unknown"
    # We'll label by scan number ranges per rat in the caller

    meta = {
        "scan": scan_num,
        "nucleus": "2H",
        "matrix_read_phase_slice": matrix,
        "spectral_points_declared": nt_decl,
        "spectral_points_file": nt,
        "echo_time_ms": _jcamp_scalar(method, "PVM_EchoTime", default=0),
        "repetition_time_ms": _jcamp_scalar(method, "PVM_RepetitionTime", default=0),
        "averages": int(round(_jcamp_scalar(method, "PVM_NAverages", default=1))),
        "spectral_acquisition_time_ms": _jcamp_scalar(method, "PVM_SpecAcquisitionTime", default=0),
        "source_domain": "ParaVision image-domain voxel FID; no spatial IFFT",
        "center_ppm_formula": "(BF1-SF0ppm)/SF0ppm*1e6",
        "t2": t2_info,
        "t2_match_rule": "proportional center-matching (18mm total coverage)",
        "reco_sha256": _sha256(reco_path),
    }
    return ScanData(
        rat_id=rat_id, scan_num=scan_num, fid=fid,
        t2_volume=t2_vol, t2_matched=t2_matched,
        spectral_width_hz=float(sw), transmitter_frequency_mhz=float(sf0),
        center_ppm=float(center_ppm), ppm_sign=-1,
        native_fov_mm=fov, native_voxel_mm=voxel,
        metadata=meta,
        source_files=[fid_path, method_path, acqp_path, reco_path, t2_path],
    )


def _get_scan_phase(rat_id: str, scan_num: int) -> str:
    """Label baseline vs postinfusion based on README-known ranges.
    R004: 36-38 baseline, 39-62 postinfusion FA60, 63-65 postinfusion FA30.
    For other rats we only have 4 scans; label by order: first=baseline, rest=postinfusion."""
    if rat_id == "R004":
        if 36 <= scan_num <= 38: return "baseline_FA60"
        if 39 <= scan_num <= 62: return "postinfusion_FA60"
        if 63 <= scan_num <= 65: return "postinfusion_FA30"
    # For R001-R003, scans are numbered differently; use positional label
    return f"scan_{scan_num}"


# ── Spectral processing ──────────────────────────────────────────────────────
def _prepare_spectra(scan: ScanData, line_broadening_hz: float = 10.0):
    fid = np.asarray(scan.fid, dtype=np.complex64).copy()
    nt = fid.shape[-1]
    envelope = np.mean(np.abs(fid), axis=(0, 1, 2))
    delay = int(np.argmax(envelope[: max(4, nt // 2)]))
    if delay >= 3:
        fid = np.roll(fid, -delay, axis=-1)
        fid[..., -delay:] = 0
    else:
        delay = 0
    time_s = np.arange(nt, dtype=np.float32) / float(scan.spectral_width_hz)
    apod = np.exp(-np.pi * line_broadening_hz * time_s).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(fid * apod, axis=-1), axes=-1)
    magnitude = np.abs(spectra).astype(np.float32)
    hz = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / scan.spectral_width_hz))
    ppm = scan.center_ppm + scan.ppm_sign * hz / scan.transmitter_frequency_mhz
    info = {
        "detected_delay_points": delay,
        "line_broadening_hz": line_broadening_hz,
        "ppm_axis_sign": scan.ppm_sign,
        "metadata_center_ppm": scan.center_ppm,
    }
    return magnitude, ppm.astype(np.float32), info


def _extract_maps(magnitude, ppm, native_masks):
    nt = magnitude.shape[-1]
    noise_sel = ((ppm < 0.0) | (ppm > 8.0)) & (np.abs(ppm) < 0.9 * np.max(np.abs(ppm)))
    if int(noise_sel.sum()) < max(12, nt // 12):
        noise_sel = np.ones(nt, dtype=bool)
        for lo, hi in METABOLITE_WINDOWS.values():
            noise_sel &= ~((ppm >= lo) & (ppm <= hi))
    noise_vals = magnitude[..., noise_sel]
    baseline = np.median(noise_vals, axis=-1)
    noise_mad = np.median(np.abs(noise_vals - baseline[..., None]), axis=-1)
    noise_sigma = np.maximum(1.4826 * noise_mad, np.finfo(np.float32).eps)
    corrected = np.maximum(magnitude - baseline[..., None], 0.0)
    ppm_step = float(np.median(np.abs(np.diff(ppm))))
    maps = {}; snr_maps = {}; win_info = {}
    for met, (lo, hi) in METABOLITE_WINDOWS.items():
        sel = (ppm >= lo) & (ppm <= hi)
        area = corrected[..., sel].sum(axis=-1) * ppm_step
        height = corrected[..., sel].max(axis=-1)
        area = np.where(native_masks, area, 0.0).astype(np.float32)
        snr = np.where(native_masks, height / noise_sigma, np.nan).astype(np.float32)
        maps[met] = area; snr_maps[met] = snr
        win_info[met] = {"ppm_low": lo, "ppm_high": hi, "points": int(sel.sum()), "ppm_step": ppm_step}
    return maps, snr_maps, {
        "noise_points": int(noise_sel.sum()),
        "windows": win_info,
        "baseline_method": "per-voxel median in off-metabolite noise region",
    }


# ── Anatomy & masking ────────────────────────────────────────────────────────
def _normalize_t2(vol):
    arr = np.asarray(vol, dtype=np.float32)
    pos = arr[np.isfinite(arr) & (arr > 0)]
    if pos.size == 0: return np.zeros_like(arr), 0.0, 1.0
    lo, hi = np.percentile(pos, [1.0, 99.5])
    if hi <= lo: hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
    return out, float(lo), float(hi)


def _largest_component(mask):
    lbl, n = ndi.label(mask)
    if n == 0: return mask.astype(bool)
    sizes = ndi.sum(mask, lbl, index=np.arange(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _anatomy_mask(img01):
    img = np.asarray(img01, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0 or finite.max() <= 0: return np.ones_like(img, dtype=bool)
    nz = finite[finite > 0]
    try: thr = float(threshold_otsu(nz if nz.size > 16 else finite))
    except ValueError: thr = 0.08
    thr = max(0.03, min(thr * 0.55, 0.35))
    m = img > thr
    m = _largest_component(m)
    r = max(1, int(round(min(img.shape) / 80)))
    m = binary_closing(m, disk(r * 2))
    m = ndi.binary_fill_holes(m)
    m = binary_dilation(m, disk(r))
    if m.mean() < 0.03 or m.mean() > 0.90:
        m = _largest_component(img > 0.03)
        m = ndi.binary_fill_holes(m)
    return m.astype(bool)


def _resize_2d(arr, size, mode):
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).view(1, 1, *arr.shape)
    if mode in ("bilinear", "bicubic"):
        out = F.interpolate(t, size=(size, size), mode=mode, align_corners=False)
    else:
        out = F.interpolate(t, size=(size, size), mode=mode)
    return out.squeeze().numpy().astype(np.float32)


def _resize_to_native(arr, shape, mode="area"):
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).view(1, 1, *arr.shape)
    if mode in ("bilinear", "bicubic"):
        out = F.interpolate(t, size=shape, mode=mode, align_corners=False)
    else:
        out = F.interpolate(t, size=shape, mode=mode)
    return out.squeeze().numpy().astype(np.float32)


def _build_masks(t2_01, native_hw):
    m64, mnat = [], []
    for img in t2_01:
        m = _anatomy_mask(img)
        m64.append(_resize_2d(m.astype(np.float32), 64, "nearest") >= 0.5)
        occ = _resize_to_native(m.astype(np.float32), native_hw, "area")
        nm = occ >= 0.20
        if nm.mean() < 0.10: nm = occ > 0.01
        mnat.append(nm)
    return np.stack(m64), np.stack(mnat)


# ── Model ────────────────────────────────────────────────────────────────────
def _build_model_opt(gpu_id: str, output_root: Path):
    opt = _load_json_config(CONFIG_PATH)
    opt["phase"] = "val"
    opt["gpu_ids"] = [int(gpu_id)]
    opt["rank"] = 0; opt["world_size"] = 1; opt["local_rank"] = 0
    opt["distributed"] = False; opt["is_main_process"] = True
    runtime_dir = output_root / "_model_runtime"
    opt["path"]["experiments_root"] = str(runtime_dir.resolve())
    for k in list(opt["path"].keys()):
        if k in ("resume_state", "experiments_root"): continue
        opt["path"][k] = str((runtime_dir / str(opt["path"][k])).resolve())
    opt["path"]["resume_state"] = str(CKPT_PREFIX.resolve())
    opt.setdefault("validation", {})["eval_network"] = "ema"
    # DC is applied manually after multi-seed averaging (see process_scan),
    # so keep the model-internal DC disabled here.
    opt["validation"]["native_data_consistency"] = {"enabled": False}
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.strip()
    return Logger.dict_to_nonedict(opt)


def _stack_condition(lr_t, t2_t, met_id, mask_t, ds_opt):
    parts = []
    if ds_opt.get("use_lr", True): parts.append(lr_t)
    if ds_opt.get("use_t1", True): parts.append(t2_t)
    if ds_opt.get("use_flair", True): parts.append(t2_t)
    if ds_opt.get("use_met_onehot", True):
        met = torch.zeros((1, 4, *lr_t.shape[-2:]), device=lr_t.device, dtype=lr_t.dtype)
        met[:, int(met_id)] = 1.0
        parts.append(met)
    if ds_opt.get("use_mask_channel", False): parts.append(mask_t)
    return torch.cat(parts, dim=1)


def _to_m11(t): return t * 2.0 - 1.0


def _sr_to_01(v):
    t = v.detach().float().cpu()
    if t.dim() == 4 and t.shape[0] > 1: t = t[-1:]
    return ((t.squeeze() + 1.0) * 0.5).clamp(0, 1).numpy().astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────
def _grad_energy(img, mask):
    gy, gx = np.gradient(np.asarray(img, dtype=np.float32))
    g = np.hypot(gx, gy)
    sel = g[np.asarray(mask, dtype=bool)]
    return float(sel.mean()) if sel.size else float(g.mean())


def _bg_leakage(img, mask):
    img = np.asarray(img, dtype=np.float32); mask = np.asarray(mask, dtype=bool)
    inside = float(np.mean(np.abs(img[mask]))) if mask.any() else float(np.mean(np.abs(img)))
    outside = float(np.mean(np.abs(img[~mask]))) if (~mask).any() else 0.0
    return outside / max(inside, 1e-8)


def _forward_residual(sr_img, native, native_mask, window="hamming"):
    n = int(native.shape[0])
    pred = torch.from_numpy(sr_img).view(1, 1, *sr_img.shape)
    padded = mrsi_native_forward_batch(pred, [n], window=window, clamp_nonnegative=True)
    projected = crop_padded_native(padded, n).squeeze().numpy()
    residual = (projected - native).astype(np.float32)
    vals = residual[native_mask]; tgt = native[native_mask]
    if vals.size == 0: vals = residual.reshape(-1); tgt = native.reshape(-1)
    return residual, {
        "native_proj_l1": float(np.mean(np.abs(vals))),
        "native_proj_rmse": float(np.sqrt(np.mean(vals**2))),
        "native_proj_rel_l1": float(np.mean(np.abs(vals)) / max(float(np.mean(np.abs(tgt))), 1e-8)),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────
def _sr_panel_label(n_seeds: int, dc_enabled: bool, seed: int, sample_steps: int) -> str:
    """Build an accurate SR panel title reflecting the actual inference method.

    - Single seed, no DC  -> original label "SR (EMA, seed0, DDIM50)"
    - Multi-seed avg       -> "SR (EMA, N-seed avg, DDIM50)"
    - DC enabled           -> append " + DC"
    """
    if int(n_seeds) > 1:
        seed_part = f"{int(n_seeds)}-seed avg"
    else:
        seed_part = f"seed{int(seed)}"
    label = f"SR (EMA, {seed_part}, DDIM{int(sample_steps)})"
    if bool(dc_enabled):
        label += " + DC"
    return label


def _plot_comparison(rat_id, scan_num, slice_idx, metabolite, native_au, scale,
                     t2_64, lr64, sr, residual_native, output, *, sr_label: str = "SR (EMA, seed0, DDIM50)"):
    lr_au = lr64 * scale; sr_au = sr * scale
    native64 = _resize_2d(native_au, 64, "nearest")
    vmax = max(float(np.percentile(np.concatenate([native64.ravel(), lr_au.ravel(), sr_au.ravel()]), 99.5)), 1e-8)
    diff = sr_au - lr_au
    dlim = max(float(np.percentile(np.abs(diff), 99.0)), 1e-8)
    res64 = _resize_2d(residual_native * scale, 64, "nearest")
    rlim = max(float(np.percentile(np.abs(res64), 99.0)), 1e-8)
    panels = [
        (t2_64, "T2 anatomy", "gray", 0, 1),
        (native64, f"native {native_au.shape[0]}x{native_au.shape[1]}", "turbo", 0, vmax),
        (lr_au, "bicubic LR 64x64", "turbo", 0, vmax),
        (sr_au, sr_label, "turbo", 0, vmax),
        (res64, "forward residual", "coolwarm", -rlim, rlim),
        (diff, "SR - bicubic", "coolwarm", -dlim, dlim),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, (im, title, cmap, vmin, vmax_p) in zip(axes.ravel(), panels):
        cb = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax_p, interpolation="nearest")
        ax.set_title(title, fontsize=9); ax.axis("off")
        fig.colorbar(cb, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{rat_id} scan{scan_num} slice{slice_idx} | {metabolite} | relative a.u.", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170); plt.close(fig)


def _plot_spectral_qc(scan, magnitude, ppm, output):
    mean_spec = magnitude.reshape(-1, magnitude.shape[-1]).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ppm, mean_spec / max(mean_spec.max(), 1e-8), color="#111827")
    for met, (lo, hi) in METABOLITE_WINDOWS.items():
        ax.axvspan(lo, hi, color=METABOLITE_COLORS[met], alpha=0.17, label=met)
    ax.set_xlim(8.0, -0.5)
    ax.set(title=f"{scan.rat_id} scan{scan.scan_num}: mean magnitude spectrum",
           xlabel="ppm", ylabel="normalized magnitude")
    ax.legend(ncol=4, frameon=False, fontsize=8); ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


# ── Main per-scan processing ─────────────────────────────────────────────────
def process_scan(scan: ScanData, diffusion, ds_opt, sample_steps: int, seed: int,
                 output_base: Path, metric_rows: list, spectral_rows: list,
                 *, n_seeds: int = 1, dc_enabled: bool = False,
                 dc_iters: int = 20, dc_lr: float = 0.02, dc_anchor: float = 0.05):
    rat = scan.rat_id; sn = scan.scan_num
    logger = logging.getLogger("base")
    t2_01, t2_lo, t2_hi = _normalize_t2(scan.t2_matched)
    native_hw = tuple(int(x) for x in scan.fid.shape[1:3])
    masks64, native_masks = _build_masks(t2_01, native_hw)
    magnitude, ppm, spec_info = _prepare_spectra(scan)
    maps, snr_maps, extract_info = _extract_maps(magnitude, ppm, native_masks)
    scan.metadata["t2_norm_p1"] = t2_lo; scan.metadata["t2_norm_p99_5"] = t2_hi
    scan.metadata["spectral"] = spec_info
    scan.metadata["metabolite_extraction"] = extract_info

    scan_dir = output_base / rat / f"scan_{sn:03d}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    _plot_spectral_qc(scan, magnitude, ppm, scan_dir / "spectral_qc.png")

    scales = {}
    for met, vals in maps.items():
        sel = vals[native_masks]
        sel = sel[np.isfinite(sel) & (sel > 0)]
        scales[met] = max(float(np.percentile(sel, 99.0)) if sel.size else float(vals.max()), 1e-8)

    # Save preprocessed native maps
    np.savez_compressed(
        scan_dir / "preprocessed_native_maps.npz",
        ppm=ppm, t2_matched_normalized=t2_01.astype(np.float32),
        mask_native=native_masks.astype(np.uint8), mask_64=masks64.astype(np.uint8),
        **{f"{k}_relative_area_au": v for k, v in maps.items()},
        **{f"{k}_snr": v for k, v in snr_maps.items()},
    )

    device = diffusion.device
    for slice_idx in range(scan.fid.shape[0]):
        t2_64 = _resize_2d(t2_01[slice_idx], 64, "bicubic").clip(0, 1)
        for metabolite in METABOLITE_WINDOWS:
            native_au = maps[metabolite][slice_idx]
            scale = scales[metabolite]
            native_norm = np.clip(native_au / scale, 0, 1).astype(np.float32)
            lr64 = _resize_2d(native_norm, 64, "bicubic").clip(0, 1)

            lr_t = torch.from_numpy(lr64).view(1, 1, 64, 64).to(device)
            t2_t = torch.from_numpy(t2_64).view(1, 1, 64, 64).to(device)
            mask_t = torch.from_numpy(masks64[slice_idx].astype(np.float32)).view(1, 1, 64, 64).to(device)
            cond = _stack_condition(lr_t, t2_t, METABOLITE_IDS[metabolite], mask_t, ds_opt)
            batch = {"HR": _to_m11(lr_t), "SR": _to_m11(cond), "LR": _to_m11(lr_t), "MASK": mask_t}

            # ── Multi-seed averaging ──────────────────────────────────────
            # Diffusion sampling is stochastic; averaging N independent seeds
            # suppresses salt-and-pepper speckle without blurring structure.
            sr_accum = np.zeros((64, 64), dtype=np.float64)
            seeds_used = list(range(int(seed), int(seed) + int(n_seeds)))
            for s in seeds_used:
                diffusion.feed_data(batch)
                diffusion.test(continous=False, seed=s,
                                sample_num_steps=sample_steps, network="ema")
                sr_accum += _sr_to_01(
                    diffusion.get_current_visuals(need_LR=True)["SR"]
                ).astype(np.float64)
            sr = (sr_accum / max(int(n_seeds), 1)).astype(np.float32)

            # ── Data-consistency (DC) post-processing ─────────────────────
            # Project the averaged SR back to the native 9×9 acquisition and
            # gently pull it toward the measured map.  This directly attacks
            # the out-of-distribution speckle: the model was trained on
            # phantom 16–32 matrices and hallucinates high-frequency content
            # for real 9×9 in-vivo input.  DC enforces physical consistency
            # with the actual measurement while an anchor term preserves
            # plausible structure.
            dc_applied = False
            if bool(dc_enabled):
                native_matrix = int(native_norm.shape[0])
                sr_m11 = torch.from_numpy(sr * 2.0 - 1.0).view(1, 1, 64, 64).to(device)
                native_padded = center_pad_native(
                    torch.from_numpy(native_norm).view(1, 1, native_matrix, native_matrix),
                    (64, 64),
                ).to(device)
                sr_refined_m11 = refine_native_data_consistency(
                    sr_m11, native_padded, native_matrix,
                    hr_mask=mask_t,
                    window="hamming",
                    iterations=int(dc_iters),
                    learning_rate=float(dc_lr),
                    anchor_weight=float(dc_anchor),
                )
                sr = ((sr_refined_m11.detach().squeeze().float().cpu().numpy() + 1.0) * 0.5
                      ).clip(0, 1).astype(np.float32)
                dc_applied = True

            # Metrics
            native_t = torch.from_numpy(native_norm.astype(np.float32))
            mask_t_cpu = torch.from_numpy(masks64[slice_idx].astype(np.float32))
            metrics = {}
            for label, img in [("bicubic", lr64), ("sr", sr)]:
                m = native_acquisition_consistency_2d(
                    torch.from_numpy(img), native_t, int(native_norm.shape[0]),
                    mask=mask_t_cpu, window="hamming")
                for k, v in m.items(): metrics[f"{label}_{k}"] = float(v)
                metrics[f"{label}_gradient_mean"] = _grad_energy(img, masks64[slice_idx])
                metrics[f"{label}_bg_leakage"] = _bg_leakage(img, masks64[slice_idx])
            residual, res_m = _forward_residual(sr, native_norm, native_masks[slice_idx])
            metrics.update({f"sr_{k}": v for k, v in res_m.items()})
            metrics["sr_vs_bicubic_grad_ratio"] = metrics["sr_gradient_mean"] / max(metrics["bicubic_gradient_mean"], 1e-8)
            metrics["native_rel_l1_change"] = metrics["sr_native_acquisition_rel_l1"] - metrics["bicubic_native_acquisition_rel_l1"]

            snr_vals = snr_maps[metabolite][slice_idx][native_masks[slice_idx]]
            fs = snr_vals[np.isfinite(snr_vals)]
            metrics.update({
                "rat_id": rat, "scan_num": sn, "slice_idx": slice_idx,
                "metabolite": metabolite, "met_id": METABOLITE_IDS[metabolite],
                "native_rows": native_au.shape[0], "native_cols": native_au.shape[1],
                "normalization_scale_au": scale,
                "native_mask_voxels": int(native_masks[slice_idx].sum()),
                "native_peak_snr_median": float(np.median(fs)) if fs.size else float("nan"),
                "seed": seed, "sample_steps": sample_steps, "network": "ema",
                "n_seeds": int(n_seeds), "dc_enabled": bool(dc_enabled),
                "dc_applied": bool(dc_applied), "dc_iters": int(dc_iters),
                "dc_lr": float(dc_lr), "dc_anchor": float(dc_anchor),
            })
            metric_rows.append(metrics)
            spectral_rows.append({
                "rat_id": rat, "scan_num": sn, "slice_idx": slice_idx,
                "metabolite": metabolite, "met_id": METABOLITE_IDS[metabolite],
                "ppm_low": METABOLITE_WINDOWS[metabolite][0],
                "ppm_high": METABOLITE_WINDOWS[metabolite][1],
                "scale_au": scale, "snr_median": metrics["native_peak_snr_median"],
            })

            # Save per-sample result
            stem = f"slice_{slice_idx:02d}_{metabolite}"
            np.savez_compressed(
                scan_dir / f"{stem}_sr_result.npz",
                model_input_lr_normalized=lr64,
                model_input_t2_normalized=t2_64,
                model_input_met_id=np.int32(METABOLITE_IDS[metabolite]),
                sr_normalized=sr,
                sr_relative_au=(sr * scale).astype(np.float32),
                bicubic_lr_normalized=lr64,
                bicubic_lr_relative_au=(lr64 * scale).astype(np.float32),
                native_relative_au=native_au.astype(np.float32),
                native_normalized=native_norm,
                anatomy_mask_64=masks64[slice_idx].astype(np.uint8),
                anatomy_mask_native=native_masks[slice_idx].astype(np.uint8),
                forward_residual_native=residual,
                normalization_scale_au=np.float32(scale),
                metabolite=np.array(metabolite),
                slice_idx=np.int32(slice_idx),
                seed=np.int32(seed), sample_steps=np.int32(sample_steps),
                n_seeds=np.int32(n_seeds), dc_enabled=np.uint8(dc_enabled),
                dc_applied=np.uint8(dc_applied), dc_iters=np.int32(dc_iters),
                dc_lr=np.float32(dc_lr), dc_anchor=np.float32(dc_anchor),
                checkpoint=np.array(str(CKPT_PREFIX) + "_ema_gen.pth"),
            )
            _plot_comparison(
                rat, sn, slice_idx, metabolite, native_au, scale, t2_64, lr64, sr, residual,
                scan_dir / f"{stem}_comparison.png",
                sr_label=_sr_panel_label(n_seeds, dc_enabled, seed, sample_steps),
            )
            logger.info("%s scan%d slice%d %s: bicubic_relL1=%.4f sr_relL1=%.4f grad_ratio=%.2f",
                        rat, sn, slice_idx, metabolite,
                        metrics["bicubic_native_acquisition_rel_l1"],
                        metrics["sr_native_acquisition_rel_l1"],
                        metrics["sr_vs_bicubic_grad_ratio"])

    # Scan metadata
    _write_json(scan_dir / "scan_metadata.json", {
        "rat_id": rat, "scan_num": sn, "scan_phase": _get_scan_phase(rat, sn),
        "fid_shape": list(scan.fid.shape),
        "spectral_width_hz": scan.spectral_width_hz,
        "transmitter_frequency_mhz": scan.transmitter_frequency_mhz,
        "center_ppm": scan.center_ppm,
        "native_fov_mm": list(scan.native_fov_mm),
        "native_voxel_mm": list(scan.native_voxel_mm),
        "metadata": scan.metadata,
        "source_files": [str(p) for p in scan.source_files],
        "source_sha256": {p.name: _sha256(p) for p in scan.source_files if p.is_file()},
    })


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Healthy rat MRSI SR3 inference (EMA)")
    parser.add_argument("--rats", type=str, default="R001,R002,R003,R004",
                        help="Comma-separated rat IDs to process")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line_broadening_hz", type=float, default=10.0)
    # ── Improvement: multi-seed averaging ──
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of independent diffusion seeds to average "
                             "(reduces stochastic speckle).  Default 1 = original behaviour.")
    # ── Improvement: data-consistency post-processing ──
    parser.add_argument("--dc", action="store_true",
                        help="Enable native data-consistency refinement after sampling.")
    parser.add_argument("--dc_iters", type=int, default=20,
                        help="DC gradient-descent iterations (default 20).")
    parser.add_argument("--dc_lr", type=float, default=0.02,
                        help="DC Adam learning rate (default 0.02).")
    parser.add_argument("--dc_anchor", type=float, default=0.05,
                        help="DC anchor weight — lower lets data consistency dominate "
                             "more (default 0.05).")
    # ── Output ──
    parser.add_argument("--out_name", type=str, default="inference",
                        help="Output directory name under DATA_ROOT (default 'inference'). "
                             "Use a different name to avoid overwriting baseline results.")
    args = parser.parse_args()

    rat_ids = [r.strip() for r in args.rats.split(",") if r.strip()]
    output_root = DATA_ROOT / args.out_name
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"infer_{'_'.join(rat_ids)}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    warnings.filterwarnings("once", category=UserWarning)
    logger = logging.getLogger("base")
    method_tag = (f"n_seeds={args.n_seeds}"
                  + (f", dc(iters={args.dc_iters},lr={args.dc_lr},anchor={args.dc_anchor})"
                     if args.dc else ", dc=off"))
    logger.info("Rats: %s | GPU: %s | seed=%d steps=%d | %s",
                rat_ids, args.gpu_id, args.seed, args.sample_steps, method_tag)
    logger.info("Output directory: %s", output_root)

    # Load model
    opt = _build_model_opt(args.gpu_id, output_root)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    ds_opt = dict(opt.get("datasets", {}).get("val", {}) or {})
    logger.info("Model loaded on %s, eval network=%s", diffusion.device,
                diffusion.get_eval_network_name("ema"))

    metric_rows = []; spectral_rows = []
    for rat in rat_ids:
        raw_dir = DATA_ROOT / rat / "RAW"
        scan_nums = sorted(int(d.name) for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit())
        logger.info("%s: found scans %s", rat, scan_nums)
        for sn in scan_nums:
            logger.info("Loading %s scan %d ...", rat, sn)
            scan = _load_bruker_scan(rat, sn)
            process_scan(scan, diffusion, ds_opt, args.sample_steps, args.seed,
                         output_root, metric_rows, spectral_rows,
                         n_seeds=args.n_seeds, dc_enabled=args.dc,
                         dc_iters=args.dc_iters, dc_lr=args.dc_lr, dc_anchor=args.dc_anchor)

    # Write aggregate CSVs
    _write_csv(output_root / f"metrics_{'_'.join(rat_ids)}.csv", metric_rows)
    _write_csv(output_root / f"spectral_qc_{'_'.join(rat_ids)}.csv", spectral_rows)

    # Summary
    summary = {}
    for rat in rat_ids:
        rat_rows = [r for r in metric_rows if r["rat_id"] == rat]
        for met in METABOLITE_WINDOWS:
            mr = [r for r in rat_rows if r["metabolite"] == met]
            if not mr: continue
            key = f"{rat}_{met}"
            summary[key] = {
                "n_samples": len(mr),
                "bicubic_rel_l1_mean": float(np.mean([r["bicubic_native_acquisition_rel_l1"] for r in mr])),
                "sr_rel_l1_mean": float(np.mean([r["sr_native_acquisition_rel_l1"] for r in mr])),
                "sr_grad_ratio_mean": float(np.mean([r["sr_vs_bicubic_grad_ratio"] for r in mr])),
                "sr_bg_leakage_mean": float(np.mean([r["sr_bg_leakage"] for r in mr])),
                "snr_median_mean": float(np.nanmean([r["native_peak_snr_median"] for r in mr])),
            }
    summary["_method"] = {
        "n_seeds": int(args.n_seeds),
        "dc_enabled": bool(args.dc),
        "dc_iters": int(args.dc_iters),
        "dc_lr": float(args.dc_lr),
        "dc_anchor": float(args.dc_anchor),
        "sample_steps": int(args.sample_steps),
        "seed": int(args.seed),
        "network": "ema",
    }
    _write_json(output_root / f"summary_{'_'.join(rat_ids)}.json", summary)
    logger.info("Done. %d samples processed.", len(metric_rows))
    logger.info("Summary: %s", output_root / f"summary_{'_'.join(rat_ids)}.json")


if __name__ == "__main__":
    main()
