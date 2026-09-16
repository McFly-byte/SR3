import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON = Path(r"D:\code_software\Anaconda\envs\msr_mrsi\python.exe")


class HealthyInvivoDataContractsTests(unittest.TestCase):
    def test_loao_builder_populates_existing_source_paths(self):
        script = _REPO_ROOT / "scripts" / "build_healthy_invivo_loao_splits.py"
        audit = _REPO_ROOT / "experiments" / "healthy_invivo_selfsup" / "audit" / "healthy_rats_audit.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "splits"
            subprocess.check_call([
                str(_PYTHON), str(script), "--audit-json", str(audit), "--output-dir", str(output_dir)
            ], cwd=_REPO_ROOT)
            summary = json.loads((output_dir / "loao_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            for fold in summary["folds"]:
                fold_dir = Path(fold["split_json"]).parent
                for csv_name in ("train_scans.csv", "validation_scans.csv"):
                    with (fold_dir / csv_name).open("r", encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertTrue(rows)
                    for row in rows:
                        for field in (
                            "scan_dir", "method_path", "acqp_path", "reco_path", "visu_pars_path",
                            "fid_proc_path", "two_dseq_path", "ser_path",
                        ):
                            self.assertTrue(row[field], f"empty {field} in {fold_dir / csv_name}")
                            self.assertTrue(Path(row[field]).exists(), row[field])

    def test_phantom_domain_contract_is_reproducible_and_semantically_frozen(self):
        script = _REPO_ROOT / "scripts" / "build_healthy_phantom_domain_contract.py"
        dataset_root = _REPO_ROOT / "dataset_mrsi" / "mrsi_sr3_64"
        generator = _REPO_ROOT / "data" / "prepare_mrsi_sr3_pairs.py"
        config = dataset_root / "run_config.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "contract.json"
            subprocess.check_call([
                str(_PYTHON), str(script),
                "--dataset-root", str(dataset_root),
                "--generator-script", str(generator),
                "--config", str(config),
                "--output", str(output),
            ], cwd=_REPO_ROOT)
            contract = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(contract["passed"])
            self.assertEqual(contract["metabolite_id_contract"], {"Glx": 0, "Glc": 1, "Lac": 2, "Lipid": 3})
            self.assertEqual(contract["condition_contract"]["denoiser_input_channels"], 8)
            self.assertFalse(contract["real_data_mapping_policy"]["HDO_supported_by_pretrained_onehot"])
            self.assertFalse(contract["simulation_degradation_contract"]["vendor_matched_to_current_305_of_405_Bruker_CSI"])
            self.assertEqual(contract["total_manifest_rows"], 7616)
    def test_real_spectral_contract_separates_sampling_facts_from_unresolved_delay_policy(self):
        script = _REPO_ROOT / "scripts" / "build_real_spectral_metabolite_contract.py"
        data_root = Path(r"D:\LMC\data\invivo_zlx\zlx_healthy_rats_data")
        readme = data_root / "readme.md"
        phantom_script = Path(r"D:\LMC\projects\project\phantom_generation.py")
        baseline_script = Path(r"D:\LMC\projects\project\hkj\baseline_corr.py")
        observation = (
            _REPO_ROOT / "experiments" / "healthy_invivo_selfsup" / "audit"
            / "fidproc_spectra_observation_only.json"
        )
        topspin_audit = (
            _REPO_ROOT / "experiments" / "healthy_invivo_selfsup" / "audit"
            / "topspin_temp_spectral_processing_audit.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "contract.json"
            subprocess.check_call([
                str(_PYTHON), str(script),
                "--data-root", str(data_root),
                "--readme", str(readme),
                "--phantom-script", str(phantom_script),
                "--baseline-script", str(baseline_script),
                "--observation-audit", str(observation),
                "--topspin-audit", str(topspin_audit),
                "--output", str(output),
            ], cwd=_REPO_ROOT)
            contract = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(contract["scan_count"], 16)
            self.assertFalse(contract["formal_preprocessing_validated"])
            self.assertFalse(contract["training_permission_changed"])
            self.assertNotIn("ppm_direction", contract["unresolved_contract"])
            axis = contract["confirmed_from_local_first_party_files"]["reconstructed_ppm_axis_direction"]
            self.assertEqual(axis["status"], "resolved")
            self.assertIn("decreasing ppm", axis["direction"])
            self.assertIn("sampling_and_digital_filter_semantics", contract["unresolved_contract"])
            acquisition = contract["confirmed_from_local_first_party_files"]["acquisition"]
            self.assertEqual(acquisition["method_dwell_time_us"]["unique_values"], [166.0])
            self.assertEqual(acquisition["acqu_td_real_points"]["unique_values"], [512])
            self.assertEqual(acquisition["complex_sample_interval_us"]["unique_values"], [332.0])
            self.assertEqual(acquisition["acqu_digital_filter_group_delay"]["unique_values"], [76.125])
            self.assertEqual(acquisition["acqp_digital_filter_group_delay"]["unique_values"], [-1.0])
            self.assertEqual(acquisition["acquisition_mode"]["unique_values"], [3])
            self.assertEqual(acquisition["decimation"]["unique_values"], [2])
            self.assertEqual(acquisition["vendor_spectral_reco_rotate"]["unique_values"], [0.5])
            self.assertEqual(acquisition["vendor_reco_b0_demod_delay"]["unique_values"], [0.0])
            self.assertEqual(acquisition["vendor_reco_graph_contains_spectral_fft"]["unique_values"], [True])
            self.assertEqual(acquisition["vendor_reco_text_mentions_grpdly"]["unique_values"], [False])
            self.assertEqual(acquisition["working_offset_matches_frequency_difference"]["unique_values"], [True])
            self.assertEqual(acquisition["working_ppm_matches_frequency_difference"]["unique_values"], [True])
            self.assertEqual(acquisition["frequency_fields_match_method"]["unique_values"], [True])
            self.assertEqual(acquisition["acq_o1_hz"]["unique_values"], [0.0])
            self.assertEqual(acquisition["acqu_o1_hz"]["unique_values"], [0.0])
            self.assertEqual(acquisition["acq_o1_list_hz"]["unique_values"], [-211.795277777778, 43.0661111111111])
            self.assertEqual(acquisition["acq_o1b_list_hz"]["unique_values"], [0.0])
            self.assertEqual(acquisition["pulseprogram_binds_acq_o1_list_to_tx1"]["unique_values"], [True])
            self.assertEqual(acquisition["pulseprogram_binds_acq_o1b_list_to_receiver"]["unique_values"], [True])
            frequency_semantics = contract["confirmed_from_local_first_party_files"]["frequency_field_semantics"]
            self.assertGreater(
                abs(frequency_semantics["expected_strongest_peak_shift_bins_if_tx_offset_were_misapplied_to_receiver_axis"]),
                20.0,
            )
            self.assertLess(abs(frequency_semantics["observed_strongest_index_group_mean_difference"]), 2.0)
            self.assertIn("partially resolved", contract["unresolved_contract"]["frequency_reference"])
            self.assertTrue(all(row["method_dwell_times_two_equals_complex_interval"] for row in contract["rows"]))
            self.assertTrue(all(row["acquisition_time_equals_td_times_method_dwell"] for row in contract["rows"]))
            self.assertTrue(all(len(row["acqu_sha256"]) == 64 for row in contract["rows"]))
            self.assertTrue(all(len(row["reco_sha256"]) == 64 for row in contract["rows"]))

    def test_t2_csi_geometry_is_initialization_only_and_never_enables_t2_loss(self):
        script = _REPO_ROOT / "scripts" / "audit_t2_csi_geometry.py"
        data_root = Path(r"D:\LMC\data\invivo_zlx\zlx_healthy_rats_data")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "t2_csi_geometry.json"
            subprocess.check_call([
                str(_PYTHON), str(script),
                "--data-root", str(data_root),
                "--output", str(output),
            ], cwd=_REPO_ROOT)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["animal_count"], 4)
            self.assertEqual(audit["scan_count"], 16)
            self.assertTrue(audit["confirmed"]["same_nominal_orientation"])
            self.assertTrue(audit["confirmed"]["same_nominal_spatial_fov"])
            self.assertTrue(audit["confirmed"]["same_frame_of_reference_uid"])
            self.assertTrue(audit["confirmed"]["same_study_uid"])
            self.assertTrue(audit["interpretation_limits"]["header_geometry_is_not_registration"])
            self.assertTrue(audit["interpretation_limits"]["motion_between_T2_and_CSI_not_quantified"])
            self.assertFalse(audit["interpretation_limits"]["t2_loss_allowed"])
            elapsed = [
                animal["candidate_relation"]["t2_to_first_selected_csi_hours"]
                for animal in audit["animals"].values()
            ]
            self.assertGreaterEqual(min(elapsed), 1.5)
            self.assertLessEqual(max(elapsed), 2.6)
            for animal in audit["animals"].values():
                self.assertEqual(animal["registration_status"], "geometry_initialization_only")
                self.assertFalse(animal["intensity_registration_executed"])
                self.assertFalse(animal["resampling_executed"])
                self.assertFalse(animal["t2_loss_allowed"])


if __name__ == "__main__":
    unittest.main()
