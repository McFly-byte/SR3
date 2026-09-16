import copy
import json
import sys
import unittest
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from core.healthy_invivo_selfsup import (  # noqa: E402
    GuardedLRAdaptationOptimizer,
    LRMeasurementLossConfig,
    PretrainedParameterAnchor,
    ProhibitedTargetError,
    RealDataTrainingGateClosed,
    assert_fresh_optimizer,
    assert_formal_real_training_allowed,
    assert_lr_only_batch,
    load_training_gate,
    measured_domain_consistency_loss,
)

_GATE_PATH = (
    _REPO_ROOT
    / "experiments"
    / "healthy_invivo_selfsup"
    / "audit"
    / "healthy_invivo_training_gate.json"
)


class HealthyInvivoSelfSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_training_gate(_GATE_PATH)

    def test_current_authoritative_gate_refuses_parameter_updates(self):
        self.assertFalse(self.gate["formal_real_data_training_allowed"])
        with self.assertRaises(RealDataTrainingGateClosed):
            assert_formal_real_training_allowed(self.gate)

    def test_closed_gate_cannot_mutate_model_or_optimizer(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = copy.deepcopy(model.state_dict())
        predicted = model(torch.ones((2, 3)))
        loss = predicted.square().mean()
        with self.assertRaises(RealDataTrainingGateClosed):
            GuardedLRAdaptationOptimizer(
                gate=self.gate,
                optimizer=optimizer,
                required_measurement_keys=("MEASURED_LR",),
            )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertEqual(len(optimizer.state), 0)

    def test_pseudo_hr_and_t2_are_rejected(self):
        with self.assertRaises(ProhibitedTargetError):
            assert_lr_only_batch(
                {"MEASURED_LR": torch.ones(1), "HR": torch.ones(1)},
                required_measurement_keys=("MEASURED_LR",),
            )
        with self.assertRaises(ProhibitedTargetError):
            assert_lr_only_batch(
                {"MEASURED_LR": torch.ones(1), "T2": torch.ones(1)},
                required_measurement_keys=("MEASURED_LR",),
                t2_policy="disabled",
            )

    def test_measured_domain_complex_loss_is_zero_only_at_identity(self):
        measured = torch.tensor([1 + 2j, -3 + 0.5j], dtype=torch.complex128)
        zero = measured_domain_consistency_loss(
            measured,
            measured,
            config=LRMeasurementLossConfig(kind="l2"),
        )
        shifted = measured_domain_consistency_loss(
            measured + (1 - 1j),
            measured,
            config=LRMeasurementLossConfig(kind="l2"),
        )
        self.assertEqual(float(zero), 0.0)
        self.assertGreater(float(shifted), 0.0)

    def test_pretrained_anchor_detects_parameter_drift(self):
        model = torch.nn.Linear(2, 1, bias=False)
        anchor = PretrainedParameterAnchor(model)
        self.assertEqual(float(anchor.loss(model).detach()), 0.0)
        with torch.no_grad():
            model.weight.add_(0.25)
        self.assertGreater(float(anchor.loss(model).detach()), 0.0)

    def test_guarded_session_allows_multiple_steps_after_single_freshness_check(self):
        synthetic = json.loads(json.dumps(self.gate))
        synthetic["formal_real_data_training_allowed"] = True
        for blocker in synthetic["blocking_requirements"]:
            blocker["resolved"] = True
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        session = GuardedLRAdaptationOptimizer(
            gate=synthetic,
            optimizer=optimizer,
            required_measurement_keys=("MEASURED_LR",),
        )
        batch = {"MEASURED_LR": torch.ones((1, 1))}
        for _ in range(2):
            loss = model(torch.ones((1, 2))).square().mean()
            session.step(batch=batch, loss=loss)
        self.assertEqual(session.step_count, 2)
        self.assertGreater(len(optimizer.state), 0)

    def test_inherited_optimizer_state_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(torch.ones((1, 2))).sum().backward()
        optimizer.step()
        with self.assertRaisesRegex(ValueError, "fresh state"):
            assert_fresh_optimizer(optimizer)

    def test_synthetic_open_gate_requires_all_blockers_resolved(self):
        synthetic = json.loads(json.dumps(self.gate))
        synthetic["formal_real_data_training_allowed"] = True
        for blocker in synthetic["blocking_requirements"]:
            blocker["resolved"] = True
        assert_formal_real_training_allowed(synthetic)


if __name__ == "__main__":
    unittest.main()
