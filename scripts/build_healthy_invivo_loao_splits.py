#!/usr/bin/env python3
"""Generate four leave-one-animal-out manifests and assert zero leakage.

The output carries only acquisition identities and source paths. It does not
create surrogate HR targets, compute fold statistics from validation animals,
or modify source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED = {
    "R001": [57, 58, 59, 60],
    "R002": [30, 31, 32, 33],
    "R003": [35, 36, 37, 38],
    "R004": [59, 60, 61, 62],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(audit_json: Path) -> List[Dict[str, Any]]:
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    data_root = Path(payload["source_data_root"])
    rows = payload.get("scans", [])
    selected = []
    for row in rows:
        animal = str(row.get("animal_id"))
        scan = int(row.get("scan_id"))
        if animal not in EXPECTED or scan not in EXPECTED[animal]:
            continue
        if not str(row.get("qc_status", row.get("status", ""))).startswith("PASS"):
            raise ValueError(f"Audit row is not PASS: {animal}/{scan}")
        scan_dir = data_root / animal / "RAW" / str(scan)
        paths = {
            "scan_dir": scan_dir,
            "method_path": scan_dir / "method",
            "acqp_path": scan_dir / "acqp",
            "reco_path": scan_dir / "pdata" / "1" / "reco",
            "visu_pars_path": scan_dir / "pdata" / "1" / "visu_pars",
            "fid_proc_path": scan_dir / "pdata" / "1" / "fid_proc.64",
            "two_dseq_path": scan_dir / "pdata" / "1" / "2dseq",
            "ser_path": scan_dir / "pv2tsdata" / "1" / "ser",
        }
        missing = [name for name, path in paths.items() if name != "scan_dir" and not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required source files for {animal}/{scan}: {missing}")
        selected.append({
            "animal_id": animal,
            "scan_id": scan,
            "acquisition_type": row.get("acquisition_type", "2H_CSI"),
            "timepoint_status": "ordered_scan_subset_only; absolute postinfusion time unconfirmed",
            **{name: str(path) for name, path in paths.items()},
            "t2_usage": "withheld_from_training_loss_until_registration_and_texture_leakage_validation",
        })
    selected.sort(key=lambda item: (item["animal_id"], item["scan_id"]))
    found = {animal: [row["scan_id"] for row in selected if row["animal_id"] == animal] for animal in EXPECTED}
    if found != EXPECTED:
        raise ValueError(f"Selected scan identities differ from frozen protocol: {found}")
    return selected


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["animal_id", "scan_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    audit_path = Path(args.audit_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = load_rows(audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    all_animals = sorted(EXPECTED)
    for fold_index, held_out in enumerate(all_animals, start=1):
        train_animals = [animal for animal in all_animals if animal != held_out]
        train_rows = [row for row in rows if row["animal_id"] in train_animals]
        val_rows = [row for row in rows if row["animal_id"] == held_out]
        overlap = sorted(set(r["animal_id"] for r in train_rows) & set(r["animal_id"] for r in val_rows))
        if overlap:
            raise AssertionError(f"Animal leakage in fold {fold_index}: {overlap}")
        fold_dir = output_dir / f"fold_{fold_index}_holdout_{held_out}"
        train_csv = fold_dir / "train_scans.csv"
        val_csv = fold_dir / "validation_scans.csv"
        write_csv(train_csv, train_rows)
        write_csv(val_csv, val_rows)
        split_json = fold_dir / "split.json"
        split_payload = {
            "fold": fold_index,
            "held_out_animal": held_out,
            "training_animals": train_animals,
            "validation_animals": [held_out],
            "training_scan_count": len(train_rows),
            "validation_scan_count": len(val_rows),
            "animal_overlap": overlap,
            "normalization_fit_scope": "training_animals_only",
            "degradation_fit_scope": "training_animals_only",
            "early_stopping_scope": "training-side inner split only; held-out animal is not used for tuning",
            "timepoint_interpretation": "scan order only; do not treat as pure technical repeats until timing semantics are confirmed",
            "t2_loss_enabled": False,
            "hr_target_present": False,
            "files": {
                "train_csv": {"path": str(train_csv), "sha256": sha256(train_csv)},
                "validation_csv": {"path": str(val_csv), "sha256": sha256(val_csv)},
            },
        }
        split_json.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        folds.append({
            "fold": fold_index,
            "held_out_animal": held_out,
            "training_animals": train_animals,
            "validation_animals": [held_out],
            "training_scan_count": len(train_rows),
            "validation_scan_count": len(val_rows),
            "animal_overlap": overlap,
            "split_json": str(split_json),
            "split_json_sha256": sha256(split_json),
        })

    validation_animals = [fold["held_out_animal"] for fold in folds]
    passed = (
        len(rows) == 16
        and len(folds) == 4
        and sorted(validation_animals) == all_animals
        and all(fold["training_scan_count"] == 12 and fold["validation_scan_count"] == 4 for fold in folds)
        and all(not fold["animal_overlap"] for fold in folds)
    )
    summary = {
        "schema_version": 1,
        "protocol": "four-fold leave-one-animal-out",
        "source_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "selected_scan_count": len(rows),
        "animal_ids": all_animals,
        "folds": folds,
        "global_invariants": {
            "animal_is_minimum_independent_unit": True,
            "held_out_animal_used_for_hyperparameter_selection": False,
            "fold_statistics_fit_on_training_animals_only": True,
            "surrogate_hr_targets_created": False,
            "t2_used_as_hr_target": False,
        },
        "passed": passed,
    }
    summary_path = output_dir / "loao_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "summary": str(summary_path)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
