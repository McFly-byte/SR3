import json
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TIMING_PATH = (
    _REPO_ROOT
    / "experiments"
    / "healthy_invivo_selfsup"
    / "audit"
    / "healthy_invivo_scan_timing.json"
)


class HealthyInvivoScanTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _TIMING_PATH.is_file():
            raise FileNotFoundError(
                f"Run scripts/audit_healthy_invivo_scan_timing.py first: {_TIMING_PATH}"
            )
        cls.payload = json.loads(_TIMING_PATH.read_text(encoding="utf-8"))

    def test_all_selected_scans_have_first_party_absolute_order(self):
        self.assertEqual(self.payload["scan_count"], 16)
        confirmed = self.payload["first_party_confirmed"]
        self.assertTrue(confirmed["absolute_scan_start_time"])
        self.assertTrue(confirmed["within_animal_absolute_order"])
        self.assertTrue(confirmed["postinfusion_label_for_all_selected_scans"])
        self.assertTrue(all(row["acq_abs_matches_visu_acq_date"] for row in self.payload["rows"]))

    def test_injection_relative_time_and_repeat_equivalence_remain_unresolved(self):
        self.assertTrue(self.payload["not_confirmed"]["injection_timestamp"])
        self.assertTrue(self.payload["not_confirmed"]["minutes_post_injection"])
        self.assertTrue(self.payload["not_confirmed"]["technical_repeat_equivalence"])
        self.assertTrue(self.payload["policy"]["scan_number_is_not_elapsed_time"])
        self.assertTrue(self.payload["policy"]["do_not_treat_as_technical_repeats"])
        self.assertFalse(self.payload["policy"]["training_permission_changed"])
        self.assertTrue(all(row["injection_relative_time_minutes"] is None for row in self.payload["rows"]))
        self.assertTrue(all(row["technical_repeat_status"] == "unresolved" for row in self.payload["rows"]))

    def test_intervals_are_sequential_five_minute_acquisitions_not_inferred_repeats(self):
        intervals = []
        for animal in self.payload["per_animal"].values():
            self.assertTrue(animal["absolute_start_order_matches_scan_order"])
            self.assertTrue(animal["all_labeled_postinfusion"])
            self.assertFalse(animal["injection_timestamp_available"])
            self.assertFalse(animal["injection_relative_time_available"])
            self.assertEqual(animal["technical_repeat_status"], "unresolved")
            intervals.extend(animal["start_to_start_intervals_seconds"])
        self.assertEqual(len(intervals), 12)
        self.assertGreaterEqual(min(intervals), 303.0)
        self.assertLessEqual(max(intervals), 306.0)

    def test_all_source_files_are_hashed(self):
        for row in self.payload["rows"]:
            for key in ("acqp_sha256", "method_sha256", "visu_pars_sha256"):
                self.assertRegex(row[key], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
