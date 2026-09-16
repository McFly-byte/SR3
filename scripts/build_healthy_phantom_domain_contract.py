#!/usr/bin/env python3
"""Build a machine-readable contract for the frozen healthy phantom SR3 domain.

This is an audit-only utility. It does not modify datasets or checkpoints and it
never infers real-data metabolite labels that are not supported by source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_METABOLITES = ["Glx", "Glc", "Lac", "Lipid"]
EXPECTED_MAPPING = {name: idx for idx, name in enumerate(EXPECTED_METABOLITES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--generator-script", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    generator_script = Path(args.generator_script).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_metabolites = list(config["config"]["metabolites"])
    if configured_metabolites != EXPECTED_METABOLITES:
        raise AssertionError(
            f"Configured metabolite order changed: {configured_metabolites}; expected {EXPECTED_METABOLITES}"
        )

    splits: Dict[str, Any] = {}
    manifest_mapping: Dict[str, set[int]] = {name: set() for name in EXPECTED_METABOLITES}
    total_rows = 0
    for split in ("train", "val", "test"):
        manifest = dataset_root / split / "manifest.csv"
        rows = load_manifest(manifest)
        total_rows += len(rows)
        for row in rows:
            name = row["met_name"]
            if name not in manifest_mapping:
                raise AssertionError(f"Unexpected metabolite in {manifest}: {name}")
            manifest_mapping[name].add(int(row["met_id"]))
        expected_count = int(config[split]["samples"])
        if len(rows) != expected_count:
            raise AssertionError(f"{split} manifest rows={len(rows)} != config samples={expected_count}")
        splits[split] = {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "row_count": len(rows),
            "patient_count_from_config": int(config[split]["patients"]),
        }

    observed_mapping = {name: sorted(values) for name, values in manifest_mapping.items()}
    if observed_mapping != {name: [idx] for name, idx in EXPECTED_MAPPING.items()}:
        raise AssertionError(f"Manifest metabolite mapping mismatch: {observed_mapping}")

    sample_checks = []
    for metabolite, met_id in EXPECTED_MAPPING.items():
        match = None
        for split in ("train", "val", "test"):
            manifest = dataset_root / split / "manifest.csv"
            for row in load_manifest(manifest):
                if row["met_name"] == metabolite and int(row["met_id"]) == met_id:
                    match = (split, row)
                    break
            if match:
                break
        if not match:
            raise AssertionError(f"No sample found for {metabolite}/{met_id}")
        split, row = match
        npz_path = dataset_root / split / row["npz"]
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        import numpy as np

        with np.load(npz_path, allow_pickle=False) as data:
            onehot = data["met_onehot"]
            npz_met_id = int(data["met_id"])
            active = np.flatnonzero(np.max(onehot, axis=(1, 2)) > 0.5).tolist()
            if npz_met_id != met_id or active != [met_id]:
                raise AssertionError(
                    f"NPZ semantic mismatch {npz_path}: met_id={npz_met_id}, active={active}, expected={met_id}"
                )
            sample_checks.append({
                "metabolite": metabolite,
                "met_id": met_id,
                "split": split,
                "npz": str(npz_path),
                "npz_sha256": sha256(npz_path),
                "hr_shape": list(data["hr"].shape),
                "lr_shape": list(data["lr"].shape),
                "met_onehot_shape": list(onehot.shape),
                "active_onehot_channels": active,
                "value_ranges": {
                    "hr": [float(data["hr"].min()), float(data["hr"].max())],
                    "lr": [float(data["lr"].min()), float(data["lr"].max())],
                },
            })

    payload = {
        "schema_version": 1,
        "evidence_scope": "local generator source, resolved run config, manifests, and sampled NPZ contents",
        "dataset_root": str(dataset_root),
        "generator_script": {"path": str(generator_script), "sha256": sha256(generator_script)},
        "run_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "metabolite_id_contract": EXPECTED_MAPPING,
        "condition_contract": {
            "lr_channels": 1,
            "structural_channels": ["T1", "FLAIR"],
            "metabolite_onehot_channels": 4,
            "mask_channel_in_registered_baseline": False,
            "total_condition_channels": 7,
            "diffusion_target_channels": 1,
            "denoiser_input_channels": 8,
            "denoiser_output_channels": 1,
        },
        "normalization_contract": {
            "metabolite": "per_patient_met: per patient and metabolite, maximum inside the MRI-derived brain mask across retained slices; divide then clip to [0,1]",
            "structural": "T1 and FLAIR are normalized independently per three-slice chunk by their own maximum before selecting the middle slice",
            "physical_concentration_scale": False,
            "cross_domain_scale_compatibility_with_real_DMI": "unresolved",
        },
        "simulation_degradation_contract": {
            "domain": "2D 64x64 metabolite image",
            "operation": "FFT2 -> separable full-grid Hamming window -> retain a centered square support -> zero elsewhere -> IFFT2 -> magnitude",
            "lowres_half_range": [int(config["config"]["lowres_min"]), int(config["config"]["lowres_max"])],
            "retained_side_range": [2 * int(config["config"]["lowres_min"]), 2 * int(config["config"]["lowres_max"])],
            "vendor_matched_to_current_305_of_405_Bruker_CSI": False,
        },
        "splits": splits,
        "total_manifest_rows": total_rows,
        "observed_manifest_mapping": observed_mapping,
        "sample_npz_checks": sample_checks,
        "real_data_mapping_policy": {
            "allowed_without_new_evidence": ["Glx->0", "Glc->1", "Lac->2", "Lipid->3"],
            "HDO_supported_by_pretrained_onehot": False,
            "historical_HDO_Glc_Glx_Lac_mapping_compatible": False,
            "required_action": "Do not run the pretrained model on HDO. Do not remap Glx or Lac to historical IDs. Lipid use requires a validated real-data measurement definition; otherwise exclude the class from real adaptation/evaluation rather than substituting another metabolite.",
        },
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
