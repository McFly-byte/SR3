import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_METS = ["Glx", "Glc", "Lac", "Lipid"]


def _resize_2d(
    img: np.ndarray,
    size_hw: Tuple[int, int],
    mode: str = "bicubic",
) -> np.ndarray:
    t = torch.from_numpy(img.astype(np.float32))[None, None, :, :]
    out = F.interpolate(t, size=size_hw, mode=mode, align_corners=True if mode in ("bicubic", "bilinear") else None)
    return out[0, 0].cpu().numpy().astype(np.float32)


def _resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    out = F.interpolate(t, size=size_hw, mode="nearest")
    return out[0, 0].cpu().numpy().astype(np.float32)


def _kspace_window_2d(h: int, w: int, kind: str) -> np.ndarray:
    kind = (kind or "none").lower()
    if kind in ("none", "rect", "boxcar"):
        return np.ones((h, w), dtype=np.float32)
    if kind == "hamming":
        wy = np.hamming(h).astype(np.float32)
        wx = np.hamming(w).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)
    if kind == "hann":
        wy = np.hanning(h).astype(np.float32)
        wx = np.hanning(w).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)
    raise ValueError(f"Unsupported k-space window: {kind}")


def _kspace_degrade(hr: np.ndarray, lowres_half: int, window_kind: str = "hamming") -> np.ndarray:
    h, w = hr.shape
    div = int(lowres_half)
    if div <= 0:
        raise ValueError(f"lowres_half must be > 0, got {div}")
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - div), min(h, cy + div)
    x0, x1 = max(0, cx - div), min(w, cx + div)

    k = np.fft.fftshift(np.fft.fft2(hr.astype(np.float32)))
    if window_kind is not None:
        k = k * _kspace_window_2d(h, w, window_kind)

    k_small = np.zeros_like(k, dtype=np.complex64)
    k_small[y0:y1, x0:x1] = k[y0:y1, x0:x1]
    lr_c = np.fft.ifft2(np.fft.ifftshift(k_small))
    lr = np.abs(lr_c).astype(np.float32)
    return lr


def _parse_patient_id(patient_dir_name: str) -> int:
    if not patient_dir_name.lower().startswith("patient"):
        raise ValueError(f"Invalid patient folder name: {patient_dir_name}")
    return int(patient_dir_name.lower().replace("patient", ""))


def _collect_patient_dirs(data_root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        try:
            pid = _parse_patient_id(p.name)
        except Exception:
            continue
        out[pid] = p
    return out


def _build_mask_and_structural(
    patient_dir: Path,
    out_hw: Tuple[int, int],
    apply_brain_mask: bool,
    brain_mask_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1_vol = np.load(str(patient_dir / "MRI_sliced" / "T1_sliced.npy"), mmap_mode="r")
    fl_vol = np.load(str(patient_dir / "MRI_sliced" / "flair_sliced.npy"), mmap_mode="r")
    if t1_vol.shape[0] != fl_vol.shape[0]:
        raise ValueError(f"T1/flair slice count mismatch in {patient_dir}")
    num_slices = t1_vol.shape[0] // 3
    masks = np.zeros((num_slices, out_hw[0], out_hw[1]), dtype=np.float32)
    t1_all = np.zeros_like(masks)
    fl_all = np.zeros_like(masks)

    for s in range(num_slices):
        t1_chunk = t1_vol[s * 3:(s + 1) * 3, :, :]
        fl_chunk = fl_vol[s * 3:(s + 1) * 3, :, :]
        if t1_chunk.shape[0] < 3 or fl_chunk.shape[0] < 3:
            continue
        t1_den = float(np.max(t1_chunk))
        fl_den = float(np.max(fl_chunk))
        if t1_den <= 0:
            t1_den = 1.0
        if fl_den <= 0:
            fl_den = 1.0
        t1_mid = (t1_chunk[1] / t1_den).astype(np.float32)
        fl_mid = (fl_chunk[1] / fl_den).astype(np.float32)
        if apply_brain_mask:
            mask_mri = (np.maximum(t1_mid, fl_mid) > float(brain_mask_thr)).astype(np.float32)
        else:
            mask_mri = np.ones_like(t1_mid, dtype=np.float32)

        mask64 = _resize_mask(mask_mri, out_hw)
        t1_64 = _resize_2d(t1_mid, out_hw, mode="bicubic")
        fl_64 = _resize_2d(fl_mid, out_hw, mode="bicubic")

        # Keep background aligned with mask for stable training/evaluation.
        t1_64 = t1_64 * mask64
        fl_64 = fl_64 * mask64

        masks[s] = mask64
        t1_all[s] = np.clip(t1_64, 0.0, 1.0)
        fl_all[s] = np.clip(fl_64, 0.0, 1.0)
    return masks, t1_all, fl_all


def _compute_met_scales(
    patient_dir: Path,
    mets: List[str],
    masks: np.ndarray,
    met_norm: str,
) -> Dict[str, float]:
    if met_norm == "per_slice":
        return {m: 1.0 for m in mets}
    scales: Dict[str, float] = {}
    num_slices = masks.shape[0]
    for met in mets:
        arr = np.load(str(patient_dir / "Met_filtered" / f"{met}.npy"), mmap_mode="r")
        vmax = 0.0
        for s in range(num_slices):
            m = masks[s] > 0.5
            if not np.any(m):
                continue
            v = float(np.max(arr[s][m]))
            vmax = max(vmax, v)
        scales[met] = vmax if vmax > 0.0 else 1.0
    return scales


def _valid_slice_mask(
    patient_dir: Path,
    mets: List[str],
    masks: np.ndarray,
    min_brain_mask_ratio: float,
    skip_constant_slices: bool,
    constant_slice_tol: float,
) -> np.ndarray:
    num_slices = masks.shape[0]
    valid = np.ones((num_slices,), dtype=bool)
    if min_brain_mask_ratio > 0.0:
        ratios = masks.reshape(num_slices, -1).mean(axis=1)
        valid &= (ratios >= float(min_brain_mask_ratio))

    if skip_constant_slices:
        for met in mets:
            arr = np.load(str(patient_dir / "Met_filtered" / f"{met}.npy"), mmap_mode="r")
            if arr.shape[0] != num_slices:
                valid[:] = False
                break
            for s in range(num_slices):
                if not valid[s]:
                    continue
                m = masks[s] > 0.5
                if not np.any(m):
                    valid[s] = False
                    continue
                region = arr[s][m]
                if float(np.max(region) - np.min(region)) <= float(constant_slice_tol):
                    valid[s] = False
    return valid


def _write_split(
    split_name: str,
    patient_ids: List[int],
    patient_dirs: Dict[int, Path],
    out_root: Path,
    rng: random.Random,
    mets: List[str],
    lowres_min: int,
    lowres_max: int,
    window_kind: str,
    out_hw: Tuple[int, int],
    apply_brain_mask: bool,
    brain_mask_thr: float,
    met_norm: str,
    min_brain_mask_ratio: float,
    skip_constant_slices: bool,
    constant_slice_tol: float,
) -> Dict[str, int]:
    split_dir = out_root / split_name
    sample_dir = split_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.csv"

    rows = []
    sample_idx = 0
    kept_patients = 0
    met_to_id = {m: i for i, m in enumerate(DEFAULT_METS)}

    for pid in patient_ids:
        patient_dir = patient_dirs.get(pid)
        if patient_dir is None:
            continue
        kept_patients += 1
        masks, t1_all, fl_all = _build_mask_and_structural(
            patient_dir, out_hw=out_hw, apply_brain_mask=apply_brain_mask, brain_mask_thr=brain_mask_thr
        )
        num_slices = masks.shape[0]
        valid = _valid_slice_mask(
            patient_dir=patient_dir,
            mets=mets,
            masks=masks,
            min_brain_mask_ratio=min_brain_mask_ratio,
            skip_constant_slices=skip_constant_slices,
            constant_slice_tol=constant_slice_tol,
        )
        scales = _compute_met_scales(patient_dir=patient_dir, mets=mets, masks=masks, met_norm=met_norm)

        met_arrays = {
            m: np.load(str(patient_dir / "Met_filtered" / f"{m}.npy"), mmap_mode="r")
            for m in mets
        }
        for s in range(num_slices):
            if not valid[s]:
                continue
            mask = masks[s].astype(np.float32)
            for met in mets:
                arr = met_arrays[met]
                hr = np.array(arr[s], copy=True).astype(np.float32) * mask
                if met_norm == "per_slice":
                    scale = float(np.max(hr)) if float(np.max(hr)) > 0 else 1.0
                else:
                    scale = float(scales[met])
                hr = hr / (scale + 1e-8)
                hr = np.clip(hr, 0.0, 1.0).astype(np.float32)

                lowres_half = rng.randint(int(lowres_min), int(lowres_max))
                lr = _kspace_degrade(hr, lowres_half=lowres_half, window_kind=window_kind)
                lr = np.clip(lr * mask, 0.0, 1.0).astype(np.float32)

                t1 = t1_all[s].astype(np.float32)
                fl = fl_all[s].astype(np.float32)
                met_onehot = np.zeros((len(DEFAULT_METS), out_hw[0], out_hw[1]), dtype=np.float32)
                met_onehot[met_to_id[met], :, :] = 1.0

                out_file = sample_dir / f"{sample_idx:08d}.npz"
                np.savez_compressed(
                    out_file,
                    hr=hr[None, :, :],
                    lr=lr[None, :, :],
                    t1=t1[None, :, :],
                    flair=fl[None, :, :],
                    met_onehot=met_onehot,
                    mask=mask[None, :, :],
                    met_id=np.int16(met_to_id[met]),
                    patient_id=np.int16(pid),
                    slice_idx=np.int16(s),
                    lowres=np.int16(lowres_half),
                )
                rows.append(
                    {
                        "sample_id": sample_idx,
                        "npz": str(Path("samples") / out_file.name),
                        "split": split_name,
                        "patient_id": pid,
                        "slice_idx": s,
                        "met_name": met,
                        "met_id": met_to_id[met],
                        "lowres": lowres_half,
                    }
                )
                sample_idx += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "npz", "split", "patient_id", "slice_idx", "met_name", "met_id", "lowres"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"patients": kept_patients, "samples": len(rows)}


def _as_int_list(vals: List[int]) -> List[int]:
    return [int(x) for x in vals]


def main():
    parser = argparse.ArgumentParser(description="Prepare offline MRSI SR3 pairs (.npz + manifest).")
    parser.add_argument("--data-root", type=str, required=True, help="Path to data_processed root.")
    parser.add_argument("--out-root", type=str, required=True, help="Output directory for SR3 npz dataset.")
    parser.add_argument("--train-patients", nargs="+", type=int, required=True)
    parser.add_argument("--val-patients", nargs="+", type=int, required=True)
    parser.add_argument("--test-patients", nargs="+", type=int, required=True)
    parser.add_argument("--metabolites", nargs="+", type=str, default=DEFAULT_METS)
    parser.add_argument("--resolution", type=int, default=64, help="Output HR/LR resolution (default: 64).")
    parser.add_argument("--lowres-min", type=int, default=8, help="Min low-res half-size for k-space truncation.")
    parser.add_argument("--lowres-max", type=int, default=16, help="Max low-res half-size for k-space truncation.")
    parser.add_argument("--kspace-window", type=str, default="hamming", choices=["none", "hamming", "hann"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply-brain-mask", action="store_true", help="Use MRI-derived mask for metabolite background.")
    parser.add_argument("--brain-mask-thr", type=float, default=0.08)
    parser.add_argument("--met-norm", type=str, default="per_patient_met", choices=["per_patient_met", "per_slice"])
    parser.add_argument("--min-brain-mask-ratio", type=float, default=0.0)
    parser.add_argument("--skip-constant-slices", action="store_true")
    parser.add_argument("--constant-slice-tol", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_hw = (int(args.resolution), int(args.resolution))
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if out_root.exists() and not args.overwrite:
        raise FileExistsError(f"out_root exists: {out_root}. Use --overwrite to allow reuse.")
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    train_ids = _as_int_list(args.train_patients)
    val_ids = _as_int_list(args.val_patients)
    test_ids = _as_int_list(args.test_patients)
    mets = [m for m in args.metabolites if m in DEFAULT_METS]
    if len(mets) == 0:
        raise ValueError("No valid metabolites selected.")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("train/val/test patient splits must be disjoint.")
    if args.lowres_min > args.lowres_max:
        raise ValueError("--lowres-min must be <= --lowres-max.")

    patient_dirs = _collect_patient_dirs(data_root)

    stats = {
        "config": {
            "data_root": str(data_root),
            "out_root": str(out_root),
            "resolution": int(args.resolution),
            "lowres_min": int(args.lowres_min),
            "lowres_max": int(args.lowres_max),
            "kspace_window": args.kspace_window,
            "seed": int(args.seed),
            "metabolites": mets,
            "apply_brain_mask": bool(args.apply_brain_mask),
            "brain_mask_thr": float(args.brain_mask_thr),
            "met_norm": args.met_norm,
            "min_brain_mask_ratio": float(args.min_brain_mask_ratio),
            "skip_constant_slices": bool(args.skip_constant_slices),
            "constant_slice_tol": float(args.constant_slice_tol),
        }
    }

    stats["train"] = _write_split(
        split_name="train",
        patient_ids=train_ids,
        patient_dirs=patient_dirs,
        out_root=out_root,
        rng=rng,
        mets=mets,
        lowres_min=args.lowres_min,
        lowres_max=args.lowres_max,
        window_kind=args.kspace_window,
        out_hw=out_hw,
        apply_brain_mask=args.apply_brain_mask,
        brain_mask_thr=args.brain_mask_thr,
        met_norm=args.met_norm,
        min_brain_mask_ratio=args.min_brain_mask_ratio,
        skip_constant_slices=args.skip_constant_slices,
        constant_slice_tol=args.constant_slice_tol,
    )
    stats["val"] = _write_split(
        split_name="val",
        patient_ids=val_ids,
        patient_dirs=patient_dirs,
        out_root=out_root,
        rng=rng,
        mets=mets,
        lowres_min=args.lowres_min,
        lowres_max=args.lowres_max,
        window_kind=args.kspace_window,
        out_hw=out_hw,
        apply_brain_mask=args.apply_brain_mask,
        brain_mask_thr=args.brain_mask_thr,
        met_norm=args.met_norm,
        min_brain_mask_ratio=args.min_brain_mask_ratio,
        skip_constant_slices=args.skip_constant_slices,
        constant_slice_tol=args.constant_slice_tol,
    )
    stats["test"] = _write_split(
        split_name="test",
        patient_ids=test_ids,
        patient_dirs=patient_dirs,
        out_root=out_root,
        rng=rng,
        mets=mets,
        lowres_min=args.lowres_min,
        lowres_max=args.lowres_max,
        window_kind=args.kspace_window,
        out_hw=out_hw,
        apply_brain_mask=args.apply_brain_mask,
        brain_mask_thr=args.brain_mask_thr,
        met_norm=args.met_norm,
        min_brain_mask_ratio=args.min_brain_mask_ratio,
        skip_constant_slices=args.skip_constant_slices,
        constant_slice_tol=args.constant_slice_tol,
    )

    with (out_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


