#!/usr/bin/env python3
"""Build a source-qualified spectral/metabolite contract for healthy rat scans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_SCANS = {
    "R001": (57, 58, 59, 60),
    "R002": (30, 31, 32, 33),
    "R003": (35, 36, 37, 38),
    "R004": (59, 60, 61, 62),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(path: Path) -> str:
    return path.read_text(encoding="latin-1")


def array_values(content: str, key: str) -> list[str]:
    match = re.search(rf"^##\${re.escape(key)}=\s*\([^\r\n]*\)\s*[\r\n]+([^\r\n]+)", content, re.MULTILINE)
    if not match:
        raise KeyError(key)
    return match.group(1).split()


def scalar(content: str, key: str) -> str:
    match = re.search(rf"^##\${re.escape(key)}=\s*([^\r\n]+)", content, re.MULTILINE)
    if not match:
        raise KeyError(key)
    return match.group(1).strip().strip("<>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--phantom-script", type=Path, required=True)
    parser.add_argument("--baseline-script", type=Path, required=True)
    parser.add_argument("--observation-audit", type=Path, required=True)
    parser.add_argument("--topspin-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for animal, scans in EXPECTED_SCANS.items():
        for scan in scans:
            scan_dir = args.data_root / animal / "RAW" / str(scan)
            method_path = scan_dir / "method"
            acqp_path = scan_dir / "acqp"
            acqu_path = scan_dir / "acqu"
            reco_path = scan_dir / "pdata" / "1" / "reco"
            pulseprogram_path = scan_dir / "lists" / "pp" / "zouCSI.ppg"
            method = text(method_path)
            acqp = text(acqp_path)
            acqu = text(acqu_path)
            reco = text(reco_path)
            pulseprogram = text(pulseprogram_path)
            spectral_points = int(float(array_values(method, "PVM_SpecMatrix")[0]))
            spectral_width_hz = float(array_values(method, "PVM_SpecSWH")[0])
            spectral_acquisition_time_ms = float(scalar(method, "PVM_SpecAcquisitionTime"))
            method_dwell_time_us = float(array_values(method, "PVM_SpecDwellTime")[0])
            acqu_td_real_points = int(float(scalar(acqu, "TD")))
            complex_sample_interval_us = 1000.0 * spectral_acquisition_time_ms / spectral_points
            working_frequency_mhz = float(array_values(method, "PVM_FrqWork")[0])
            reference_frequency_mhz = float(array_values(method, "PVM_FrqRef")[0])
            working_offset_hz = float(array_values(method, "PVM_FrqWorkOffset")[0])
            working_ppm = float(array_values(method, "PVM_FrqWorkPpm")[0])
            frequency_difference_hz = (working_frequency_mhz - reference_frequency_mhz) * 1e6
            ppm_from_frequency_difference = frequency_difference_hz / reference_frequency_mhz
            acqu_base_frequency_mhz = float(scalar(acqu, "BF1"))
            acqu_working_frequency_mhz = float(scalar(acqu, "SFO1"))
            acqp_base_frequency_mhz = float(scalar(acqp, "BF1"))
            acqp_working_frequency_mhz = float(scalar(acqp, "SFO1"))
            acqp_reference_frequency_mhz = float(scalar(acqp, "SF0ppm"))
            row = {
                "animal_id": animal,
                "scan_id": scan,
                "nucleus": scalar(method, "PVM_Nucleus1Enum"),
                "reference_frequency_mhz": reference_frequency_mhz,
                "working_frequency_mhz": working_frequency_mhz,
                "working_ppm": working_ppm,
                "working_offset_hz": working_offset_hz,
                "working_minus_reference_hz": frequency_difference_hz,
                "working_offset_matches_frequency_difference": abs(
                    working_offset_hz - frequency_difference_hz
                ) < 1e-6,
                "working_ppm_from_frequency_difference": ppm_from_frequency_difference,
                "working_ppm_matches_frequency_difference": abs(
                    working_ppm - ppm_from_frequency_difference
                ) < 1e-6,
                "spectral_offset_hz": float(scalar(method, "PVM_SpecOffsetHz")),
                "spectral_offset_ppm": float(scalar(method, "PVM_SpecOffsetppm")),
                "spectral_points": spectral_points,
                "spectral_width_hz": spectral_width_hz,
                "spectral_width_ppm": float(array_values(method, "PVM_SpecSW")[0]),
                "spectral_acquisition_time_ms": spectral_acquisition_time_ms,
                "method_dwell_time_us": method_dwell_time_us,
                "acqu_td_real_points": acqu_td_real_points,
                "complex_sample_interval_us": complex_sample_interval_us,
                "inverse_bandwidth_us": 1e6 / spectral_width_hz,
                "method_dwell_times_two_equals_complex_interval": abs(
                    2.0 * method_dwell_time_us - complex_sample_interval_us
                ) < 1e-9,
                "acquisition_time_equals_td_times_method_dwell": abs(
                    spectral_acquisition_time_ms - acqu_td_real_points * method_dwell_time_us / 1000.0
                ) < 1e-9,
                "acquisition_mode": int(float(scalar(acqu, "AQ_mod"))),
                "decimation": int(float(scalar(acqu, "DECIM"))),
                "digital_mode": int(float(scalar(acqu, "DIGMOD"))),
                "digital_filter_version": int(float(scalar(acqu, "DSPFVS"))),
                "acqu_digital_filter_group_delay": float(scalar(acqu, "GRPDLY")),
                "acqu_base_frequency_mhz": acqu_base_frequency_mhz,
                "acqu_working_frequency_mhz": acqu_working_frequency_mhz,
                "acqp_base_frequency_mhz": acqp_base_frequency_mhz,
                "acqp_working_frequency_mhz": acqp_working_frequency_mhz,
                "acqp_reference_frequency_mhz": acqp_reference_frequency_mhz,
                "acqu_o1_hz": float(scalar(acqu, "O1")),
                "frequency_fields_match_method": (
                    abs(acqu_base_frequency_mhz - working_frequency_mhz) < 1e-12
                    and abs(acqu_working_frequency_mhz - working_frequency_mhz) < 1e-12
                    and abs(acqp_base_frequency_mhz - working_frequency_mhz) < 1e-12
                    and abs(acqp_working_frequency_mhz - working_frequency_mhz) < 1e-12
                    and abs(acqp_reference_frequency_mhz - reference_frequency_mhz) < 1e-12
                ),
                "acqu_spectral_width_hz": float(scalar(acqu, "SW_h")),
                "vendor_spectral_reco_rotate": float(array_values(reco, "RECO_rotate")[0]),
                "vendor_reco_b0_demod_delay": float(scalar(reco, "RecoB0DemodDelay")),
                "vendor_reco_graph_contains_spectral_fft": (
                    "RecoFTShiftFilter FTS0{shift=0.5; winDirection=0; exponent=1}" in reco
                    and "RecoFTFilter FT0{direction=0; exponent=1}" in reco
                ),
                "vendor_reco_text_mentions_grpdly": "GRPDLY" in reco.upper(),
                "acq_o1_hz": float(scalar(acqp, "O1")),
                "acq_o1_list_hz": float(array_values(acqp, "ACQ_O1_list")[0]),
                "acq_o1b_list_hz": float(array_values(acqp, "ACQ_O1B_list")[0]),
                "pulseprogram_binds_acq_o1_list_to_tx1": (
                    "define list<frequency> freqTx1={$ACQ_O1_list}" in pulseprogram
                    and "freqTx1:f1" in pulseprogram
                ),
                "pulseprogram_binds_acq_o1b_list_to_receiver": (
                    "define list<frequency> freqRx={$ACQ_O1B_list}" in pulseprogram
                    and "freqRx(receive):f1" in pulseprogram
                ),
                "acqp_digital_filter_group_delay": float(scalar(acqp, "GRPDLY")),
                "drift_compensation": scalar(acqp, "ACQ_DriftCompActive"),
                "method_sha256": sha256(method_path),
                "acqp_sha256": sha256(acqp_path),
                "acqu_sha256": sha256(acqu_path),
                "reco_sha256": sha256(reco_path),
                "pulseprogram_sha256": sha256(pulseprogram_path),
            }
            rows.append(row)

    stable_fields = {}
    candidate_fields = [
        "nucleus", "reference_frequency_mhz", "working_frequency_mhz", "working_ppm",
        "working_offset_hz", "working_minus_reference_hz",
        "working_offset_matches_frequency_difference", "working_ppm_from_frequency_difference",
        "working_ppm_matches_frequency_difference", "spectral_offset_hz", "spectral_offset_ppm",
        "spectral_points", "spectral_width_hz", "spectral_width_ppm",
        "spectral_acquisition_time_ms", "method_dwell_time_us", "acqu_td_real_points",
        "complex_sample_interval_us", "inverse_bandwidth_us",
        "method_dwell_times_two_equals_complex_interval",
        "acquisition_time_equals_td_times_method_dwell", "acquisition_mode", "decimation",
        "digital_mode", "digital_filter_version", "acqu_digital_filter_group_delay",
        "acqu_base_frequency_mhz", "acqu_working_frequency_mhz", "acqp_base_frequency_mhz",
        "acqp_working_frequency_mhz", "acqp_reference_frequency_mhz", "acqu_o1_hz",
        "frequency_fields_match_method", "acqu_spectral_width_hz", "vendor_spectral_reco_rotate",
        "vendor_reco_b0_demod_delay", "vendor_reco_graph_contains_spectral_fft",
        "vendor_reco_text_mentions_grpdly", "acq_o1_hz", "acq_o1_list_hz", "acq_o1b_list_hz",
        "pulseprogram_binds_acq_o1_list_to_tx1", "pulseprogram_binds_acq_o1b_list_to_receiver",
        "acqp_digital_filter_group_delay", "drift_compensation",
    ]
    for key in candidate_fields:
        values = [row[key] for row in rows]
        stable_fields[key] = {
            "all_equal": all(value == values[0] for value in values),
            "unique_values": sorted(set(values), key=str),
        }

    readme_text = args.readme.read_text(encoding="utf-8")
    phantom_text = args.phantom_script.read_text(encoding="utf-8")
    baseline_text = args.baseline_script.read_text(encoding="utf-8")
    observation = json.loads(args.observation_audit.read_text(encoding="utf-8"))
    if observation["pass_count"] != 16:
        raise AssertionError("The observation-only spectral audit is not 16/16 PASS")
    topspin = json.loads(args.topspin_audit.read_text(encoding="utf-8"))
    ppm_direction_resolved = (
        topspin.get("ppm_direction_status")
        == "resolved_from_retained_topspin_processing_and_numerical_identity"
        and topspin.get("complete_audit_count") == 8
        and topspin.get("exact_rser_chunk_match_count") == 8
        and topspin.get("exact_or_numerical_fidproc_trace_match_count") == 8
        and topspin.get("topspin_ft_magnitude_match_count") == 8
        and topspin.get("voxel_index_identity_count") == 8
        and topspin.get("no_axis_reversal_count") == 8
        and float(topspin.get("maximum_center_error_ppm", 1.0)) <= 2e-4
        and topspin.get("training_permission_changed") is False
    )
    if not ppm_direction_resolved:
        raise AssertionError("The retained TopSpin ppm-axis evidence did not pass its numerical identity gates")

    observation_by_scan = {
        (str(row["animal_id"]), int(row["scan_id"])): row for row in observation["rows"]
    }
    tx_offset_groups: dict[float, list[int]] = {}
    for row in rows:
        key = (row["animal_id"], int(row["scan_id"]))
        observed = observation_by_scan[key]
        tx_offset_groups.setdefault(row["acq_o1_list_hz"], []).append(
            int(observed["strongest_observed_index"])
        )
    if len(tx_offset_groups) != 2:
        raise AssertionError(f"Expected two ACQ_O1_list groups, got {sorted(tx_offset_groups)}")
    tx_offsets = sorted(tx_offset_groups)
    frequency_bin_hz = rows[0]["spectral_width_hz"] / rows[0]["spectral_points"]
    tx_offset_difference_hz = tx_offsets[1] - tx_offsets[0]
    expected_bins_if_misapplied_to_receiver_axis = tx_offset_difference_hz / frequency_bin_hz
    observed_group_means = {
        str(offset): sum(indices) / len(indices) for offset, indices in tx_offset_groups.items()
    }
    observed_strongest_index_difference = (
        observed_group_means[str(tx_offsets[1])] - observed_group_means[str(tx_offsets[0])]
    )

    payload = {
        "schema_version": "1.0",
        "evidence_scope": "local readme, all 16 Bruker method/acqp/acqu/pdata-1-reco files, local source, and reproducible parsing",
        "scan_count": len(rows),
        "confirmed_from_local_first_party_files": {
            "probe": {
                "value": "2H-Glc injected through tail vein",
                "evidence": "readme.md line 7",
                "limitation": "This identifies the administered probe, not a fitted metabolite map contract.",
            },
            "acquisition": stable_fields,
            "sampling_semantics": {
                "confirmed_relations": [
                    "The method/acqu files encode 512 real points at 166 us, giving 84.992 ms acquisition time.",
                    "The reconstructed fid_proc.64 contains 256 complex points; 84.992 ms / 256 = 332 us, equal to 1 / 3012.04819277108 Hz within floating-point tolerance.",
                    "Across the 16 scans, acqu records AQ_mod=3, DECIM=2, DIGMOD=1, DSPFVS=20 and GRPDLY=76.125, while acqp records GRPDLY=-1.",
                    "The retained pdata/1/reco graph explicitly applies spectral RECO_rotate=0.5 followed by the spectral FFT, records RecoB0DemodDelay=0, and does not mention GRPDLY by name.",
                ],
                "interpretation_limit": "These header and reconstruction-graph facts establish sample-count, interval, and explicit vendor-stage relations only. Absence of a GRPDLY token in reco does not prove that the delay was previously corrected or that no correction is needed; no additional correction is applied by this audit.",
            },
            "frequency_field_semantics": {
                "confirmed_relations": [
                    "For all 16 scans, PVM_FrqWork - PVM_FrqRef equals PVM_FrqWorkOffset in Hz within 1e-6 Hz.",
                    "For all 16 scans, (PVM_FrqWork - PVM_FrqRef) / PVM_FrqRef equals PVM_FrqWorkPpm within 1e-6 ppm.",
                    "For all 16 scans, acqu/acqp BF1 and SFO1 equal PVM_FrqWork, while acqp SF0ppm equals PVM_FrqRef; O1 is zero.",
                    "The retained zouCSI pulse program binds ACQ_O1_list to freqTx1 for RF excitation and binds the separate ACQ_O1B_list to freqRx(receive).",
                    "ACQ_O1B_list is zero in all 16 scans; therefore ACQ_O1_list is not applied again as a receiver-axis or post-reconstruction ppm offset.",
                ],
                "acq_o1_list_unique_hz": tx_offsets,
                "acq_o1b_list_unique_hz": stable_fields["acq_o1b_list_hz"]["unique_values"],
                "frequency_bin_hz": frequency_bin_hz,
                "tx_offset_group_difference_hz": tx_offset_difference_hz,
                "expected_strongest_peak_shift_bins_if_tx_offset_were_misapplied_to_receiver_axis": expected_bins_if_misapplied_to_receiver_axis,
                "observed_strongest_index_group_means": observed_group_means,
                "observed_strongest_index_group_mean_difference": observed_strongest_index_difference,
                "interpretation_limit": "This resolves the transmit-versus-receive role of ACQ_O1_list in the retained pulse program and excludes adding it to the reconstructed spectral axis. It does not independently assign a metabolite peak, determine ppm sign, or define a voxelwise B0 referencing procedure.",
            },
            "observation_only_vendor_spectral_transform": {
                "scan_count": observation["scan_count"],
                "pass_count": observation["pass_count"],
                "minimum_reconstruction_correlation": observation["minimum_reconstruction_correlation"],
                "maximum_reconstruction_nrmse": observation["maximum_reconstruction_nrmse"],
            },
            "reconstructed_ppm_axis_direction": {
                "status": "resolved",
                "direction": topspin["ppm_direction"],
                "confirmed_relations": [
                    "For all 8 scans retaining a complete TopSpin audit trail, rser extracted a 4096-byte row from pv2tsdata/1/ser that is byte-identical to the temporary fid.",
                    "For all 8 scans, that temporary fid is identical to one complex fid_proc.64 voxel trace within relative L2 <= 1e-12, with the 1-based rser row equal to the flattened voxel index plus one.",
                    "For all 8 scans, left-loading the 256-point trace, applying the recorded exponential window, zero-filling to 512, IFFT and centering reproduces the TopSpin processed magnitude with correlation >= 0.999999999 and no axis reversal.",
                    "For all 8 scans, OFFSET - (SW_p / SF) / 2 reproduces the 4.7 ppm method center within 2e-4 ppm.",
                ],
                "axis_formula": "ppm[i] = OFFSET - i * (SW_p / SF) / SI for the retained 512-point TopSpin processed array; original 256-point reconstructed indices map as i_512 = 2*i_256.",
                "coverage_limit": "Complete processing audit trails are retained for 8/16 scans. The axis convention is shared by the common acquisition/reconstruction contract, but this does not validate automatic phase correction or metabolite assignment.",
            },
        },
        "not_valid_for_real_preprocessing": {
            "phantom_peak_model": {
                "classification": "simulation_only",
                "observed_code": "ppm_peaks={HDO:4.8, Glc:3.8, Glx:2.4, Lac:1.2}; linewidth_hz=20; hdo_amp=1.0",
                "reason": "The source describes a phantom simulation, labels peak locations as reference values, and calls HDO amplitude a simplified treatment.",
                "source_sha256": sha256(args.phantom_script),
                "source_contains_expected_markers": all(
                    marker in phantom_text for marker in ("ppm_peaks", "hdo_amp = 1.0", "MRSI Phantom Simulation")
                ),
            },
            "generic_baseline_functions": {
                "classification": "generic_unbound_code",
                "observed_code": "Savitzky-Golay/polynomial, ALS, and edge-window cubic alternatives with defaults",
                "reason": "No call site or validation ties these alternatives or their defaults to the 16 healthy Bruker scans.",
                "source_sha256": sha256(args.baseline_script),
                "source_contains_expected_markers": all(
                    marker in baseline_text for marker in ("baseline_correction", "baseline_correction_als", "baseline_correction_st")
                ),
            },
        },
        "unresolved_contract": {
            "sampling_and_digital_filter_semantics": "partially resolved: method/acqu prove 512 real samples at 166 us and 256 complex samples at 332 us, and the retained TopSpin path reproduces a 256-to-512 zero-filled spectral transform; however no vendor documentation or validated numerical experiment establishes whether acqu GRPDLY=76.125 requires any additional correction after fid_proc.64 reconstruction",
            "frequency_reference": "partially resolved: the retained pulse program proves ACQ_O1_list is a transmit-frequency list and ACQ_O1B_list is the receiver-frequency list, so ACQ_O1_list must not be added to the reconstructed ppm axis; however no validated voxelwise or scanwise reference-peak alignment procedure has been identified",
            "phase_correction": "unresolved: no validated zero/first-order spectral phase correction procedure has been identified",
            "baseline": "unresolved: generic alternatives exist but none is selected or validated for these scans",
            "line_broadening_and_zero_fill": "unresolved: no real-data policy has been identified",
            "metabolite_model": "unresolved: no validated real-data peak basis, fitting model, integration windows, linewidth constraints, or uncertainty output",
            "metabolite_semantics": "unresolved: the pretrained contract is Glx/Glc/Lac/Lipid; historical HDO/Glc/Glx/Lac IDs are incompatible except Glc",
            "units": "unresolved: fid_proc.64 and 2dseq scaling are reproducible, but no physical metabolite concentration/relative-area unit contract is established",
            "quality_control": "unresolved: acceptance thresholds for SNR, linewidth, fit residual, baseline, phase and frequency drift are not frozen",
        },
        "formal_preprocessing_validated": False,
        "training_permission_changed": False,
        "rows": rows,
        "sources": {
            "readme": {"path": str(args.readme.resolve()), "sha256": sha256(args.readme)},
            "phantom_script": {"path": str(args.phantom_script.resolve()), "sha256": sha256(args.phantom_script)},
            "baseline_script": {"path": str(args.baseline_script.resolve()), "sha256": sha256(args.baseline_script)},
            "observation_audit": {"path": str(args.observation_audit.resolve()), "sha256": sha256(args.observation_audit)},
            "topspin_temp_spectral_processing_audit": {"path": str(args.topspin_audit.resolve()), "sha256": sha256(args.topspin_audit)},
        },
        "readme_contains_probe_statement": "2H-Glc" in readme_text and "尾静脉注射" in readme_text,
    }
    if len(rows) != 16 or not payload["readme_contains_probe_statement"]:
        raise AssertionError("Spectral contract evidence is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "scan_count": len(rows),
        "formal_preprocessing_validated": False,
        "training_permission_changed": False,
        "unresolved_contract_item_count": len(payload["unresolved_contract"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
