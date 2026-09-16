import json
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "experiments" / "healthy_invivo_selfsup" / "audit" / "healthy_invivo_training_gate.json"


class HealthyInvivoTrainingGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _GATE_PATH.is_file():
            raise FileNotFoundError(
                f"Run scripts/build_healthy_invivo_training_gate.py first: {_GATE_PATH}"
            )
        cls.gate = json.loads(_GATE_PATH.read_text(encoding="utf-8"))

    def test_formal_training_is_closed_until_all_required_evidence_exists(self):
        self.assertFalse(self.gate["formal_real_data_training_allowed"])
        unresolved = {
            item["id"] for item in self.gate["blocking_requirements"] if not item["resolved"]
        }
        self.assertNotIn("bruker_encoding_forward_reproduced", unresolved)
        self.assertIn("generic_center_crop_operator_not_vendor_matched", unresolved)
        self.assertIn("spectral_preprocessing_not_validated", unresolved)
        self.assertIn("real_metabolite_contract_not_validated", unresolved)
        self.assertIn("lr_only_selfsupervised_entry_not_validated", unresolved)
        self.assertIn("animal_timepoints_not_confirmed", unresolved)
        self.assertIn("t2_registration_not_validated", unresolved)
        self.assertNotIn("animal_level_cv_pipeline_dry_run", unresolved)
        self.assertNotIn("checkpoint_resume_strategy_smoke_tested", unresolved)

    def test_fidproc_back_end_evidence_passes_without_overclaiming(self):
        checks = {item["id"]: item for item in self.gate["passed_evidence_checks"]}
        self.assertTrue(checks["rawdata_to_fidproc_vendor_front_end"]["passed"])
        self.assertTrue(checks["fidproc_to_2dseq_back_end"]["passed"])
        self.assertTrue(checks["fidproc_observation_only_spectral_audit"]["passed"])
        self.assertTrue(checks["real_spectral_metabolite_contract_audit"]["passed"])
        self.assertTrue(checks["healthy_phantom_domain_contract"]["passed"])
        self.assertTrue(checks["reject_simple_ser_fft_model"]["passed"])
        self.assertFalse(self.gate["formal_real_data_training_allowed"])
        unresolved = {item["id"] for item in self.gate["blocking_requirements"] if not item["resolved"]}
        self.assertIn("spectral_preprocessing_not_validated", unresolved)
        self.assertIn("rawdata.job0-to-fid_proc.64 vendor front end", self.gate["interpretation"])
        self.assertIn("fid_proc.64-to-2dseq back end", self.gate["interpretation"])
        self.assertIn("generic center-crop/Hamming operator is still not", self.gate["interpretation"])
        self.assertIn("LR-only optimization guard rejects pseudo-HR/T2", self.gate["interpretation"])
        self.assertIn("generic baseline helpers are not valid real-data contracts", self.gate["interpretation"])
        self.assertIn("ACQ_O1_list controls transmit", self.gate["interpretation"])
        self.assertIn("ACQ_O1B_list=0 controls receive", self.gate["interpretation"])
        self.assertIn("ppm direction as decreasing", self.gate["interpretation"])
        self.assertIn("validated reference-peak alignment remain unresolved", self.gate["interpretation"])
        contract_check = checks["real_spectral_metabolite_contract_audit"]["evidence"]
        self.assertIn("ACQ_O1_list to transmit", contract_check)
        self.assertIn("not added to the reconstructed spectral axis", contract_check)

    def test_high_risk_shortcuts_are_explicitly_prohibited(self):
        text = "\n".join(self.gate["prohibited_now"])
        self.assertIn("T2, interpolation, or model outputs as HR DMI/MRSI labels", text)
        self.assertIn("PSNR/SSIM", text)
        self.assertIn("generic center-crop/Hamming", text)

    def test_resume_and_loao_evidence_pass(self):
        checks = {item["id"]: item for item in self.gate["passed_evidence_checks"]}
        self.assertTrue(checks["checkpoint_resume_smoke_test"]["passed"])
        self.assertTrue(checks["animal_level_loao_dry_run"]["passed"])
        self.assertTrue(checks["scan_timing_semantics_audit"]["passed"])
        self.assertTrue(checks["within_encode_repeat_split_candidate"]["passed"])
        self.assertIn("noise_independence_proven=false", checks["within_encode_repeat_split_candidate"]["evidence"])
        self.assertTrue(checks["t2_csi_header_geometry_initialization_audit"]["passed"])
        blockers = {item["id"]: item for item in self.gate["blocking_requirements"]}
        self.assertTrue(blockers["checkpoint_resume_strategy_smoke_tested"]["resolved"])
        self.assertTrue(blockers["animal_level_cv_pipeline_dry_run"]["resolved"])
        self.assertFalse(blockers["t2_registration_not_validated"]["resolved"])
        self.assertIn("1.56-2.57 hours", blockers["t2_registration_not_validated"]["required_evidence"])

    def test_all_evidence_files_are_hashed(self):
        evidence = self.gate["evidence_files"]
        self.assertEqual(
            set(evidence),
            {
                "healthy_rats_audit",
                "rawdata_to_fidproc_validation",
                "fidproc_to_2dseq_validation",
                "fidproc_spectra_observation_only",
                "topspin_temp_spectral_processing_audit",
                "real_spectral_metabolite_contract",
                "healthy_phantom_domain_contract",
                "R001_57_bruker_reco_chain",
                "checkpoint_resume_smoke_test",
                "loao_summary",
                "healthy_invivo_scan_timing",
                "within_encode_repeat_split_audit",
                "t2_csi_geometry_audit",
            },
        )
        for item in evidence.values():
            self.assertTrue(Path(item["path"]).is_file())
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
