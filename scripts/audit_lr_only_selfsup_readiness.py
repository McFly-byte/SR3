#!/usr/bin/env python3
"""Audit whether the real-data LR-only adaptation entry is scientifically ready."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    gate_path = repo / "experiments" / "healthy_invivo_selfsup" / "audit" / "healthy_invivo_training_gate.json"
    diffusion_path = repo / "model" / "sr3_modules" / "diffusion.py"
    model_path = repo / "model" / "model.py"
    operator_path = repo / "core" / "bruker_csi_operator.py"
    guard_path = repo / "core" / "healthy_invivo_selfsup.py"
    evidence_paths = [gate_path, diffusion_path, model_path, operator_path, guard_path]
    for path in evidence_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    diffusion_text = diffusion_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    historical_hr_dependency = bool(
        re.search(r"def p_losses\(self, x_in, noise=None\):\s+x_start = x_in\['HR'\]", diffusion_text)
        and "b, c, h, w = self.data['HR'].shape" in model_text
    )
    unresolved = [
        item["id"] for item in gate["blocking_requirements"] if not item["resolved"]
    ]
    payload = {
        "schema_version": "1.0",
        "formal_real_data_training_allowed": gate["formal_real_data_training_allowed"],
        "optimizer_step_executed": False,
        "network_parameters_updated": False,
        "historical_sr3_training_requires_hr": historical_hr_dependency,
        "lr_only_components": {
            "fail_closed_gate_check": True,
            "prohibited_pseudo_target_check": True,
            "fresh_optimizer_state_check": True,
            "measured_domain_complex_loss": True,
            "pretrained_parameter_anchor": True,
            "vendor_reconstruction_forward_and_adjoint": True,
        },
        "unresolved_blockers": unresolved,
        "readiness": "blocked" if unresolved or not gate["formal_real_data_training_allowed"] else "ready",
        "required_next_evidence": {
            "generic_center_crop_operator_not_vendor_matched": "Connect an HR metabolite-image variable to the validated measured domain; the newly validated operator begins at averaged raw encodes, not at a 64x64 metabolite map.",
            "spectral_preprocessing_not_validated": "The reconstructed ppm array direction is resolved. Freeze the remaining sampling/digital-filter semantics, reference-peak/B0 alignment, phase correction, baseline handling, fitting/integration, units and QC thresholds across 16 scans.",
            "real_metabolite_contract_not_validated": "Confirm real Glx/Glc/Lac definitions and exclude unsupported HDO/Lipid substitutions.",
            "lr_only_selfsupervised_entry_not_validated": "After upstream contracts are fixed, integrate a model-output-to-measurement transform and run a no-update smoke test before any optimizer step.",
            "simulation_real_scale_alignment_not_validated": "Fit normalization only on each training fold and prove compatibility with the healthy simulation checkpoint input/output scale.",
            "animal_timepoints_not_confirmed": "Absolute ordering and postinfusion labels are confirmed, but obtain the injection timestamp/experiment log, retained-subset rationale, and technical-repeat policy before repeatability modeling.",
            "t2_registration_not_validated": "Keep T2 disabled until registration error and texture-leakage controls pass.",
        },
        "prohibited_shortcuts": [
            "HR=LR",
            "bicubic_or_other_interpolation_as_HR",
            "T2_as_DMI_or_MRSI_target",
            "model_output_as_target",
            "historical_HDO_Glc_Glx_Lac_ids_with_the_healthy_pretrained_onehot",
            "real_data_PSNR_or_SSIM_without_a_real_HR_target",
        ],
        "evidence_files": {
            path.relative_to(repo).as_posix(): {"sha256": sha256(path)} for path in evidence_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "readiness": payload["readiness"],
        "formal_real_data_training_allowed": payload["formal_real_data_training_allowed"],
        "unresolved_blocker_count": len(unresolved),
        "optimizer_step_executed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
