import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


REQUIRED_KEYS = ["hr", "lr", "t1", "flair", "met_onehot", "mask", "met_id", "patient_id", "slice_idx", "lowres"]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_range(name: str, arr: np.ndarray, errors: List[str], tol: float = 1e-4) -> None:
    if not np.all(np.isfinite(arr)):
        errors.append(f"{name}: non-finite values")
        return
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_min < -tol or arr_max > 1.0 + tol:
        errors.append(f"{name}: range [{arr_min:.6g}, {arr_max:.6g}] outside [0,1]")


def _summarize_split(split_dir: Path, strict: bool, max_errors: int) -> Dict:
    manifest = split_dir / "manifest.csv"
    if not manifest.exists():
        return {"exists": False, "error": f"manifest.csv not found: {manifest}"}

    with manifest.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors: List[str] = []
    patients = set()
    met_counter = Counter()
    lowres_counter = Counter()
    shapes = Counter()
    ranges = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})

    for idx, row in enumerate(rows):
        if len(errors) >= max_errors:
            break
        rel = row.get("npz")
        if not rel:
            errors.append(f"row {idx}: missing npz column")
            continue
        npz_path = split_dir / rel
        if not npz_path.exists():
            errors.append(f"row {idx}: npz not found: {npz_path}")
            continue
        try:
            data = np.load(str(npz_path))
        except Exception as exc:
            errors.append(f"row {idx}: failed to load {npz_path}: {exc}")
            continue

        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            errors.append(f"{npz_path}: missing fields {missing}")
            continue

        pid = int(data["patient_id"])
        met_id = int(data["met_id"])
        lowres = int(data["lowres"])
        patients.add(pid)
        met_counter[str(row.get("met_name", met_id))] += 1
        lowres_counter[str(lowres)] += 1

        ref_shape = tuple(data["hr"].shape)
        shapes[str(ref_shape)] += 1
        if strict:
            if len(ref_shape) != 3 or ref_shape[0] != 1:
                errors.append(f"{npz_path}: hr must be (1,H,W), got {ref_shape}")
            for key in ["lr", "t1", "flair", "mask"]:
                if tuple(data[key].shape) != ref_shape:
                    errors.append(f"{npz_path}: {key} shape {tuple(data[key].shape)} != hr {ref_shape}")
            if len(ref_shape) == 3 and tuple(data["met_onehot"].shape) != (4, ref_shape[1], ref_shape[2]):
                errors.append(f"{npz_path}: met_onehot shape {tuple(data['met_onehot'].shape)} invalid")
            for key in ["hr", "lr", "t1", "flair", "mask", "met_onehot"]:
                _check_range(key, data[key], errors)
            if "met_onehot" in data.files and not np.allclose(data["met_onehot"].sum(axis=0), 1.0, atol=1e-4):
                errors.append(f"{npz_path}: met_onehot does not sum to 1 per pixel")

        for key in ["hr", "lr", "t1", "flair", "mask"]:
            arr = data[key]
            ranges[key]["min"] = min(ranges[key]["min"], float(np.min(arr)))
            ranges[key]["max"] = max(ranges[key]["max"], float(np.max(arr)))

    return {
        "exists": True,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "num_rows": len(rows),
        "num_patients": len(patients),
        "patient_ids": sorted(int(x) for x in patients),
        "metabolites": dict(met_counter),
        "lowres": dict(lowres_counter),
        "shapes": dict(shapes),
        "ranges": dict(ranges),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MRSI SR3 manifest/npz dataset integrity.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing train/val/test split directories.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--strict", action="store_true", help="Enable shape/range/one-hot validation for each npz.")
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    root = Path(args.root)
    summary = {"root": str(root), "strict": bool(args.strict), "splits": {}}
    patient_sets = {}
    for split in args.splits:
        split_summary = _summarize_split(root / split, strict=bool(args.strict), max_errors=int(args.max_errors))
        summary["splits"][split] = split_summary
        if split_summary.get("exists"):
            patient_sets[split] = set(split_summary.get("patient_ids", []))

    leaks = []
    split_names = sorted(patient_sets)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = sorted(patient_sets[left] & patient_sets[right])
            if overlap:
                leaks.append({"left": left, "right": right, "patient_ids": overlap})
    summary["patient_overlap"] = leaks
    summary["ok"] = not leaks and all(not s.get("errors") for s in summary["splits"].values() if s.get("exists"))

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
