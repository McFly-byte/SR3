#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a machine-readable training gate for healthy in-vivo self-supervision.

The gate is deliberately conservative.  The retained vendor reconstruction is
now checked in two independently reported stages: ``rawdata.job0 -> fid_proc.64``
and ``fid_proc.64 -> 2dseq``.  Closing those stages does not by itself validate
metabolite fitting, cross-domain normalization, an LR-only optimization loss, or
T2 registration.  Formal real-data parameter updates remain disabled until every
required gate has evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


DEFAULT_REQUIRED_SCAN_COUNT = 16


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(check_id: str, passed: bool, evidence: str, consequence: str) -> Dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "evidence": evidence,
        "consequence": consequence,
    }


def build_gate(
    audit: Mapping[str, Any],
    raw_to_fid_validation: Mapping[str, Any],
    fid_validation: Mapping[str, Any],
    spectral_observation: Mapping[str, Any],
    topspin_spectral_audit: Mapping[str, Any],
    spectral_metabolite_contract: Mapping[str, Any],
    phantom_domain_contract: Mapping[str, Any],
    reco_chain: Mapping[str, Any],
    checkpoint_resume: Mapping[str, Any],
    loao_summary: Mapping[str, Any],
    scan_timing: Mapping[str, Any],
    within_encode_repeat_split: Mapping[str, Any],
    t2_csi_geometry: Mapping[str, Any],
    *,
    required_scan_count: int = DEFAULT_REQUIRED_SCAN_COUNT,
) -> Dict[str, Any]:
    scans = list(audit.get("scans", []))
    t2_rows = list(audit.get("t2", audit.get("t2_scans", [])))
    checkpoints = list(audit.get("checkpoints", []))

    header_pass = (
        len(scans) == required_scan_count
        and all(str(row.get("qc_status", row.get("status", ""))).startswith("PASS") for row in scans)
    )
    t2_geometry_pass = len(t2_rows) == 4
    raw_to_fid_metrics = raw_to_fid_validation.get("aggregate_metrics", {})
    raw_to_fid_pass = (
        int(raw_to_fid_validation.get("scan_count", -1)) == required_scan_count
        and int(raw_to_fid_validation.get("pass_count", -1)) == required_scan_count
        and raw_to_fid_validation.get("formal_vendor_frontend_reproduced") is True
        and raw_to_fid_validation.get("training_permission_changed") is False
        and float(raw_to_fid_metrics.get("min_magnitude_corr", 0.0)) >= 0.999999999
        and float(raw_to_fid_metrics.get("max_complex_rel_l2_after_scalar", 1.0)) <= 1e-10
        and float(raw_to_fid_metrics.get("max_complex_scale_abs_error", 1.0)) <= 1e-10
    )
    fid_pass = (
        int(fid_validation.get("scan_count", -1)) == required_scan_count
        and int(fid_validation.get("pass_count", -1)) == required_scan_count
        and float(fid_validation.get("min_correlation", 0.0)) >= 0.99999
        and float(fid_validation.get("max_nrmse", 1.0)) <= 5e-4
    )

    best_ser = list(reco_chain.get("best_ser_to_fid", []))
    best_ser_l2 = min(
        (float(item.get("complex_rel_l2_after_scalar", float("inf"))) for item in best_ser),
        default=float("inf"),
    )
    simple_ser_fft_rejected = best_ser_l2 > 0.9

    registered_checkpoint_hashes = {
        Path(str(item.get("path", ""))).name: str(item.get("sha256", ""))
        for item in checkpoints
        if item.get("exists") and item.get("load_status") == "ok" and item.get("sha256")
    }
    required_checkpoint_names = {
        "I300000_E2522_ema_gen.pth",
        "I300000_E2522_gen.pth",
        "I300000_E2522_opt.pth",
    }
    checkpoint_registered = required_checkpoint_names.issubset(registered_checkpoint_hashes)
    checkpoint_resume_pass = bool(checkpoint_resume.get("passed"))
    spectral_observation_pass = (
        int(spectral_observation.get("scan_count", -1)) == required_scan_count
        and int(spectral_observation.get("pass_count", -1)) == required_scan_count
        and spectral_observation.get("formal_preprocessing_validated") is False
        and spectral_observation.get("training_permission_changed") is False
        and spectral_observation.get("metabolite_assignment_status", "").startswith("unresolved")
    )
    spectral_contract_audit_pass = (
        int(spectral_metabolite_contract.get("scan_count", -1)) == required_scan_count
        and spectral_metabolite_contract.get("formal_preprocessing_validated") is False
        and spectral_metabolite_contract.get("training_permission_changed") is False
        and len(spectral_metabolite_contract.get("unresolved_contract", {})) >= 1
        and "ppm_direction" not in spectral_metabolite_contract.get("unresolved_contract", {})
        and spectral_metabolite_contract.get("confirmed_from_local_first_party_files", {}).get(
            "reconstructed_ppm_axis_direction", {}
        ).get("status") == "resolved"
        and topspin_spectral_audit.get("ppm_direction_status")
        == "resolved_from_retained_topspin_processing_and_numerical_identity"
        and topspin_spectral_audit.get("training_permission_changed") is False
        and spectral_metabolite_contract.get("confirmed_from_local_first_party_files", {}).get(
            "frequency_field_semantics", {}
        ).get("acq_o1b_list_unique_hz") == [0.0]
        and abs(
            float(
                spectral_metabolite_contract.get("confirmed_from_local_first_party_files", {}).get(
                    "frequency_field_semantics", {}
                ).get("expected_strongest_peak_shift_bins_if_tx_offset_were_misapplied_to_receiver_axis", 0.0)
            )
        ) > 20.0
        and abs(
            float(
                spectral_metabolite_contract.get("confirmed_from_local_first_party_files", {}).get(
                    "frequency_field_semantics", {}
                ).get("observed_strongest_index_group_mean_difference", 999.0)
            )
        ) < 2.0
        and spectral_metabolite_contract.get("not_valid_for_real_preprocessing", {}).get(
            "phantom_peak_model", {}
        ).get("classification") == "simulation_only"
        and spectral_metabolite_contract.get("not_valid_for_real_preprocessing", {}).get(
            "generic_baseline_functions", {}
        ).get("classification") == "generic_unbound_code"
    )
    phantom_contract_pass = (
        bool(phantom_domain_contract.get("passed"))
        and phantom_domain_contract.get("metabolite_id_contract")
        == {"Glx": 0, "Glc": 1, "Lac": 2, "Lipid": 3}
        and phantom_domain_contract.get("real_data_mapping_policy", {}).get("HDO_supported_by_pretrained_onehot") is False
        and phantom_domain_contract.get("simulation_degradation_contract", {}).get(
            "vendor_matched_to_current_305_of_405_Bruker_CSI"
        ) is False
    )
    loao_pass = bool(loao_summary.get("passed")) and all(
        not fold.get("animal_overlap")
        and int(fold.get("training_scan_count", -1)) == 12
        and int(fold.get("validation_scan_count", -1)) == 4
        for fold in loao_summary.get("folds", [])
    ) and len(loao_summary.get("folds", [])) == 4
    timing_confirmed = scan_timing.get("first_party_confirmed", {})
    timing_not_confirmed = scan_timing.get("not_confirmed", {})
    timing_audit_pass = (
        int(scan_timing.get("scan_count", -1)) == required_scan_count
        and timing_confirmed.get("absolute_scan_start_time") is True
        and timing_confirmed.get("within_animal_absolute_order") is True
        and timing_confirmed.get("postinfusion_label_for_all_selected_scans") is True
        and bool(timing_not_confirmed.get("injection_timestamp"))
        and bool(timing_not_confirmed.get("minutes_post_injection"))
        and scan_timing.get("policy", {}).get("do_not_treat_as_technical_repeats") is True
        and scan_timing.get("policy", {}).get("training_permission_changed") is False
    )
    within_encode_split_pass = (
        int(within_encode_repeat_split.get("scan_count", -1)) == required_scan_count
        and within_encode_repeat_split.get("all_encodes_have_two_nonempty_splits") is True
        and within_encode_repeat_split.get("candidate_pair_construction_numerically_valid") is True
        and float(within_encode_repeat_split.get("max_encoded_recombination_rel_l2", 1.0)) <= 1e-12
        and float(within_encode_repeat_split.get("max_full_reconstruction_vs_vendor_rel_l2", 1.0)) <= 1e-10
        and within_encode_repeat_split.get("noise_independence_proven") is False
        and within_encode_repeat_split.get("formal_real_data_training_allowed") is False
    )
    t2_confirmed = t2_csi_geometry.get("confirmed", {})
    t2_limits = t2_csi_geometry.get("interpretation_limits", {})
    t2_geometry_initialization_pass = (
        int(t2_csi_geometry.get("animal_count", -1)) == 4
        and int(t2_csi_geometry.get("scan_count", -1)) == required_scan_count
        and t2_confirmed.get("same_nominal_orientation") is True
        and t2_confirmed.get("same_nominal_spatial_fov") is True
        and t2_confirmed.get("same_frame_of_reference_uid") is True
        and t2_confirmed.get("same_study_uid") is True
        and t2_confirmed.get("within_animal_csi_geometry_constant") is True
        and t2_limits.get("header_geometry_is_not_registration") is True
        and t2_limits.get("motion_between_T2_and_CSI_not_quantified") is True
        and t2_limits.get("no_intensity_registration_or_resampling") is True
        and t2_limits.get("no_registration_error_metric") is True
        and t2_limits.get("no_texture_leakage_test") is True
        and t2_limits.get("t2_loss_allowed") is False
        and t2_limits.get("training_permission_changed") is False
        and all(
            animal.get("registration_status") == "geometry_initialization_only"
            and animal.get("intensity_registration_executed") is False
            and animal.get("resampling_executed") is False
            and animal.get("t2_loss_allowed") is False
            and float(animal.get("candidate_relation", {}).get("t2_to_first_selected_csi_hours", 0.0)) > 0.0
            for animal in t2_csi_geometry.get("animals", {}).values()
        )
    )

    checks = [
        _check(
            "headers_and_required_files",
            header_pass,
            f"{len(scans)}/{required_scan_count} scan rows pass header/file audit",
            "All selected scans must have complete first-party metadata and files.",
        ),
        _check(
            "t2_geometry_available_not_registered",
            t2_geometry_pass,
            f"{len(t2_rows)}/4 T2 volumes have parsed geometry; registration is not asserted",
            "T2 can be retained for later registration work but cannot yet enter a loss.",
        ),
        _check(
            "t2_csi_header_geometry_initialization_audit",
            t2_geometry_initialization_pass,
            (
                f"animals={t2_csi_geometry.get('animal_count')}; scans={t2_csi_geometry.get('scan_count')}; "
                "nominal orientation/FOV and FrameOfReference/Study UID match; "
                "T2-to-first-selected-CSI elapsed time is nonzero and motion remains unquantified; "
                "registration_status=geometry_initialization_only; t2_loss_allowed=false"
            ),
            "Header geometry may initialize future registration, but it is not an intensity registration and cannot enable a T2 loss.",
        ),
        _check(
            "rawdata_to_fidproc_vendor_front_end",
            raw_to_fid_pass,
            (
                f"pass_count={raw_to_fid_validation.get('pass_count')}; "
                f"min_magnitude_corr={raw_to_fid_metrics.get('min_magnitude_corr')}; "
                f"max_complex_rel_l2={raw_to_fid_metrics.get('max_complex_rel_l2_after_scalar')}; "
                f"max_scale_error={raw_to_fid_metrics.get('max_complex_scale_abs_error')}"
            ),
            "The retained vendor raw encoding-to-complex voxel-FID reconstruction is numerically reproduced per scan.",
        ),
        _check(
            "fidproc_to_2dseq_back_end",
            fid_pass,
            (
                f"pass_count={fid_validation.get('pass_count')}; "
                f"min_corr={fid_validation.get('min_correlation')}; "
                f"max_nrmse={fid_validation.get('max_nrmse')}"
            ),
            "The retained image-domain complex voxel FID can be used for reproducible spectral processing.",
        ),
        _check(
            "fidproc_observation_only_spectral_audit",
            spectral_observation_pass,
            (
                f"pass_count={spectral_observation.get('pass_count')}; "
                f"formal_preprocessing_validated={spectral_observation.get('formal_preprocessing_validated')}; "
                f"ppm_direction={spectral_observation.get('ppm_axis_status', {}).get('ppm_direction')}"
            ),
            "Raw observation spectra are reproducible, while peak assignment and formal preprocessing remain explicitly unresolved.",
        ),
        _check(
            "real_spectral_metabolite_contract_audit",
            spectral_contract_audit_pass,
            (
                f"scan_count={spectral_metabolite_contract.get('scan_count')}; "
                f"formal_preprocessing_validated={spectral_metabolite_contract.get('formal_preprocessing_validated')}; "
                f"unresolved_items={len(spectral_metabolite_contract.get('unresolved_contract', {}))}; "
                "the retained TopSpin rser/FT audit resolves the reconstructed ppm direction as decreasing with increasing array index while leaving reference-peak alignment and phase policy unresolved; "
                "method/acqu sampling equalities are retained without inferring a digital-filter delay policy; "
                "the retained pulse program binds ACQ_O1_list to transmit and ACQ_O1B_list=0 to receive, so the transmit offset is not added to the reconstructed spectral axis; "
                "simulation peak defaults and unbound baseline helpers are excluded from the real contract"
            ),
            "Available local evidence is source-qualified without promoting simulation defaults or generic helpers into real-data preprocessing.",
        ),
        _check(
            "healthy_phantom_domain_contract",
            phantom_contract_pass,
            (
                f"metabolite_mapping={phantom_domain_contract.get('metabolite_id_contract')}; "
                f"HDO_supported={phantom_domain_contract.get('real_data_mapping_policy', {}).get('HDO_supported_by_pretrained_onehot')}; "
                f"vendor_matched={phantom_domain_contract.get('simulation_degradation_contract', {}).get('vendor_matched_to_current_305_of_405_Bruker_CSI')}"
            ),
            "Real-data conditions must preserve 0=Glx, 1=Glc, 2=Lac, 3=Lipid; HDO cannot reuse a pretrained class ID.",
        ),
        _check(
            "reject_simple_ser_fft_model",
            simple_ser_fft_rejected,
            f"best tested ser-to-fid complex relative L2={best_ser_l2:.12g}",
            "A direct ser-to-spatial-FFT operator must not be used as acquisition consistency.",
        ),
        _check(
            "baseline_checkpoint_registered",
            checkpoint_registered,
            (
                "Registered checkpoint hashes: "
                + ", ".join(f"{name}={registered_checkpoint_hashes[name]}" for name in sorted(registered_checkpoint_hashes))
                if registered_checkpoint_hashes else "Checkpoint audit missing"
            ),
            "The initialization checkpoint must be immutable and traceable.",
        ),
        _check(
            "checkpoint_resume_smoke_test",
            checkpoint_resume_pass,
            (
                f"raw_exact={checkpoint_resume.get('raw_restore', {}).get('exact_match')}; "
                f"optimizer_exact={checkpoint_resume.get('optimizer_restore', {}).get('exact_match')}; "
                f"ema_exact={checkpoint_resume.get('ema_restore', {}).get('exact_match')}"
            ),
            "Raw weights, matching Adam state, and the separate EMA shadow can be restored without changing registered files.",
        ),
        _check(
            "animal_level_loao_dry_run",
            loao_pass,
            (
                f"folds={len(loao_summary.get('folds', []))}; "
                f"selected_scans={loao_summary.get('selected_scan_count')}; "
                "each fold has 12 training scans, 4 validation scans, and zero animal overlap"
            ),
            "Animal is the minimum independent unit and each animal is held out exactly once.",
        ),
        _check(
            "scan_timing_semantics_audit",
            timing_audit_pass,
            (
                f"scan_count={scan_timing.get('scan_count')}; absolute order and postinfusion labels are confirmed; "
                "injection timestamp/minutes-post-injection and technical-repeat equivalence remain unresolved"
            ),
            "Absolute ordering may be used, but scan number must not be converted to elapsed post-injection time or treated as a technical repeat label.",
        ),
        _check(
            "within_encode_repeat_split_candidate",
            within_encode_split_pass,
            (
                f"scan_count={within_encode_repeat_split.get('scan_count')}; "
                f"max_encoded_recombination_rel_l2={within_encode_repeat_split.get('max_encoded_recombination_rel_l2')}; "
                f"max_full_reconstruction_vs_vendor_rel_l2={within_encode_repeat_split.get('max_full_reconstruction_vs_vendor_rel_l2')}; "
                f"min_split_pair_magnitude_correlation={within_encode_repeat_split.get('min_split_pair_magnitude_correlation')}; "
                "noise_independence_proven=false"
            ),
            "Each encoding's retained accumulations admit two same-location nonempty splits that recombine to the vendor mean, but statistical noise independence and a training objective remain unproven.",
        ),
    ]

    blockers = [
        {
            "id": "bruker_encoding_forward_reproduced",
            "resolved": raw_to_fid_pass,
            "required_evidence": (
                "Numerically reproduce rawdata.job0 -> AverageList -> RecoSortMaps -> "
                "scan-specific spatial windows -> fractional RECO_rotate -> spatial IFFT -> "
                "declared phase correction -> fid_proc.64 for all selected scans."
            ),
        },
        {
            "id": "generic_center_crop_operator_not_vendor_matched",
            "resolved": False,
            "required_evidence": (
                "Show that core/mrsi_physics.py matches the scan-specific 305/405 weighted Bruker encoding; "
                "current center-crop plus generic Hamming implementation is simulation-only."
            ),
        },
        {
            "id": "spectral_preprocessing_not_validated",
            "resolved": False,
            "required_evidence": (
                "Freeze and test the remaining sampling/digital-filter semantics, phase/frequency-reference correction, baseline handling, "
                "metabolite fitting/integration, spectral units, and QC thresholds on all 16 scans. The reconstructed ppm direction is resolved."
            ),
        },
        {
            "id": "real_metabolite_contract_not_validated",
            "resolved": False,
            "required_evidence": (
                "Validate the real-data Glx/Glc/Lac measurement definitions against the pretrained "
                "0=Glx, 1=Glc, 2=Lac, 3=Lipid contract; HDO must remain unsupported and Lipid cannot be substituted."
            ),
        },
        {
            "id": "lr_only_selfsupervised_entry_not_validated",
            "resolved": False,
            "required_evidence": (
                "The fail-closed LR-only optimizer guard is implemented and tested, but integration remains blocked: "
                "define and validate a model-output-to-measured-LR transform after the spectral/metabolite and scale "
                "contracts are frozen; then run no-update and gradient smoke tests without reading or fabricating HR, "
                "T2, interpolation, or model-output labels."
            ),
        },
        {
            "id": "simulation_real_scale_alignment_not_validated",
            "resolved": False,
            "required_evidence": "Derive fold-local normalization and confirm compatibility with the healthy simulation training scale.",
        },
        {
            "id": "animal_timepoints_not_confirmed",
            "resolved": False,
            "required_evidence": (
                "Absolute scan starts, within-animal order, approximately five-minute spacing and postinfusion labels are now confirmed. "
                "Still obtain a first-party injection timestamp/experiment log, explain the retained four-scan subset, and determine whether "
                "biological dynamics preclude technical-repeat treatment before repeatability modeling."
            ),
        },
        {
            "id": "checkpoint_resume_strategy_smoke_tested",
            "resolved": checkpoint_resume_pass,
            "required_evidence": "Run a minimal raw+optimizer resume test and separately define the EMA initialization policy.",
        },
        {
            "id": "t2_registration_not_validated",
            "resolved": False,
            "required_evidence": (
                "Header geometry now confirms matched nominal orientation/FOV and matching FrameOfReference/Study UID, "
                "with a candidate shared-coverage relation. However T2 precedes the first selected CSI by about 1.56-2.57 hours, "
                "VisuCorePosition semantics still need vendor confirmation, and motion/intensity registration error and texture leakage "
                "remain unquantified. Complete and validate those items before any T2 loss is enabled."
            ),
        },
        {
            "id": "animal_level_cv_pipeline_dry_run",
            "resolved": loao_pass,
            "required_evidence": "Generate four leave-one-animal-out splits and assert fold-local statistics and zero animal leakage.",
        },
    ]

    formal_training_allowed = all(item["passed"] for item in checks) and all(
        item["resolved"] for item in blockers
    )
    return {
        "schema_version": 1,
        "research_scope": "healthy-rat real-data self-supervised DMI/MRSI adaptation without HR target labels",
        "formal_real_data_training_allowed": formal_training_allowed,
        "allowed_now": [
            "read-only auditing",
            "parameter-derived Bruker forward-operator implementation and adjoint/unit testing",
            "observation-only fid_proc.64 spectral QC without peak assignment or normalization",
            "formal spectral preprocessing development with explicit unresolved-parameter gates",
            "simulation-only operator tests",
            "checkpoint resume smoke tests without overwriting registered checkpoints",
            "animal-level split generation and leakage assertions",
            "absolute scan-timing audit without inferring injection-relative time or technical repeats",
            "read-only within-encode repeat-split auditing without assuming independent noise or using it as a training target",
            "T2-CSI header-geometry initialization audit without resampling, registration claims, or T2 loss",
        ],
        "prohibited_now": [
            "formal real-data network parameter updates",
            "claiming the generic center-crop/Hamming operator represents the Bruker acquisition",
            "using T2, interpolation, or model outputs as HR DMI/MRSI labels",
            "reporting real-data PSNR/SSIM against a nonexistent HR target",
            "enabling a T2 loss before registration and leakage validation",
        ],
        "passed_evidence_checks": checks,
        "blocking_requirements": blockers,
        "interpretation": (
            "The retained rawdata.job0-to-fid_proc.64 vendor front end and fid_proc.64-to-2dseq back end "
            "are numerically closed for all selected scans. A new LR-only optimization guard rejects pseudo-HR/T2 "
            "targets and blocks all parameter updates while this gate is closed, but no model-output-to-measurement "
            "loss is scientifically defined yet. The source-qualified spectral audit confirms the 512-real/256-complex "
            "sampling relations, and a retained TopSpin rser/FT path is numerically identical to fid_proc.64 voxel traces and resolves "
            "the reconstructed ppm direction as decreasing with increasing spectral array index. The pulse program proves that ACQ_O1_list controls transmit while "
            "ACQ_O1B_list=0 controls receive, so ACQ_O1_list is not added to the reconstructed spectral axis; the acqu/acqp "
            "digital-filter delay semantics and validated reference-peak alignment remain unresolved. Local phantom peak "
            "defaults and generic baseline helpers are not valid real-data contracts. The existing core/mrsi_physics.py "
            "generic center-crop/Hamming operator is still not this scan-specific operator, and formal spectral "
            "preprocessing, metabolite semantics, simulation-real scale alignment, and injection-relative timing/technical-repeat semantics remain unresolved. "
            "The retained raw accumulations can be split within each encoding into two nonempty same-location groups whose weighted recombination matches the vendor mean, "
            "but their noise independence is not established and this candidate does not authorize Noise2Noise training. "
            "T2/CSI headers share nominal orientation/FOV and FrameOfReference/Study UID and therefore provide a candidate registration initialization, "
            "but the 1.56-2.57 hour elapsed time leaves motion unquantified and no intensity-registration error or texture-leakage control exists; T2 loss remains prohibited."
        ),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    status = "OPEN" if payload["formal_real_data_training_allowed"] else "CLOSED"
    lines = [
        "# Healthy in-vivo self-supervision training gate",
        "",
        f"**Formal real-data training gate: {status}**",
        "",
        payload["interpretation"],
        "",
        "## Passed evidence checks",
        "",
    ]
    for item in payload["passed_evidence_checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- [{mark}] `{item['id']}` — {item['evidence']}")
    lines.extend(["", "## Blocking requirements", ""])
    for item in payload["blocking_requirements"]:
        state = "resolved" if item["resolved"] else "unresolved"
        lines.append(f"- [{state}] `{item['id']}` — {item['required_evidence']}")
    lines.extend(["", "## Allowed now", ""])
    lines.extend(f"- {value}" for value in payload["allowed_now"])
    lines.extend(["", "## Prohibited now", ""])
    lines.extend(f"- {value}" for value in payload["prohibited_now"])
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-json", default="healthy_invivo_training_gate.json")
    parser.add_argument("--output-md", default="HEALTHY_INVIVO_TRAINING_GATE.md")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    sources = {
        "healthy_rats_audit": audit_dir / "healthy_rats_audit.json",
        "rawdata_to_fidproc_validation": audit_dir / "rawdata_to_fidproc_validation.json",
        "fidproc_to_2dseq_validation": audit_dir / "fidproc_to_2dseq_validation.json",
        "fidproc_spectra_observation_only": audit_dir / "fidproc_spectra_observation_only.json",
        "topspin_temp_spectral_processing_audit": audit_dir / "topspin_temp_spectral_processing_audit.json",
        "real_spectral_metabolite_contract": audit_dir / "real_spectral_metabolite_contract.json",
        "healthy_phantom_domain_contract": audit_dir / "healthy_phantom_domain_contract.json",
        "R001_57_bruker_reco_chain": audit_dir / "R001_57_bruker_reco_chain.json",
        "checkpoint_resume_smoke_test": audit_dir / "checkpoint_resume_smoke_test.json",
        "loao_summary": audit_dir.parent / "splits" / "loao_summary.json",
        "healthy_invivo_scan_timing": audit_dir / "healthy_invivo_scan_timing.json",
        "within_encode_repeat_split_audit": audit_dir / "within_encode_repeat_split_audit.json",
        "t2_csi_geometry_audit": audit_dir / "t2_csi_geometry_audit.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing gate evidence: {missing}")

    payload = build_gate(
        _read_json(sources["healthy_rats_audit"]),
        _read_json(sources["rawdata_to_fidproc_validation"]),
        _read_json(sources["fidproc_to_2dseq_validation"]),
        _read_json(sources["fidproc_spectra_observation_only"]),
        _read_json(sources["topspin_temp_spectral_processing_audit"]),
        _read_json(sources["real_spectral_metabolite_contract"]),
        _read_json(sources["healthy_phantom_domain_contract"]),
        _read_json(sources["R001_57_bruker_reco_chain"]),
        _read_json(sources["checkpoint_resume_smoke_test"]),
        _read_json(sources["loao_summary"]),
        _read_json(sources["healthy_invivo_scan_timing"]),
        _read_json(sources["within_encode_repeat_split_audit"]),
        _read_json(sources["t2_csi_geometry_audit"]),
    )
    payload["evidence_files"] = {
        key: {"path": str(path), "sha256": _sha256(path)} for key, path in sources.items()
    }

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "formal_real_data_training_allowed": payload["formal_real_data_training_allowed"],
        "output_json": str(output_json),
        "output_md": str(output_md),
        "blocker_count": sum(not item["resolved"] for item in payload["blocking_requirements"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
