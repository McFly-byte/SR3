"""Fail-closed contracts for healthy in-vivo LR-only self-supervised adaptation.

This module is intentionally independent from the historical HR-supervised SR3
training path.  It prevents accidental parameter updates while the scientific
contracts for real metabolite maps, spectral preprocessing, scale alignment and
a differentiable HR-metabolite-to-measurement operator remain unresolved.

It does not create a pseudo target and it never interprets T2, interpolation or a
model prediction as a high-resolution DMI/MRSI label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class RealDataTrainingGateClosed(RuntimeError):
    """Raised before any optimizer step when formal real-data training is closed."""


class ProhibitedTargetError(ValueError):
    """Raised when a real-data batch contains a prohibited pseudo-HR target."""


PROHIBITED_TARGET_KEYS = frozenset(
    {
        "HR",
        "GT",
        "TARGET",
        "HIGH_RESOLUTION",
        "HR_DMI",
        "HR_MRSI",
        "INTERPOLATED_HR",
        "MODEL_OUTPUT_AS_TARGET",
        "T2_AS_TARGET",
    }
)


@dataclass(frozen=True)
class LRMeasurementLossConfig:
    """Configuration for a measured-domain robust consistency loss."""

    kind: str = "charbonnier"
    epsilon: float = 1e-6

    def validate(self) -> None:
        if self.kind not in {"l1", "l2", "charbonnier"}:
            raise ValueError(f"Unsupported measured-domain loss: {self.kind}")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def load_training_gate(path: str | Path) -> Mapping[str, Any]:
    gate_path = Path(path)
    with gate_path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if "formal_real_data_training_allowed" not in gate:
        raise ValueError(f"Missing formal_real_data_training_allowed in {gate_path}")
    if "blocking_requirements" not in gate:
        raise ValueError(f"Missing blocking_requirements in {gate_path}")
    return gate


def unresolved_blocker_ids(gate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get("id"))
        for item in gate.get("blocking_requirements", [])
        if not bool(item.get("resolved"))
    )


def assert_formal_real_training_allowed(gate: Mapping[str, Any]) -> None:
    """Fail before model/optimizer mutation unless every formal gate is open."""
    blockers = unresolved_blocker_ids(gate)
    allowed = gate.get("formal_real_data_training_allowed") is True
    if not allowed or blockers:
        raise RealDataTrainingGateClosed(
            "Formal healthy in-vivo parameter updates are disabled; unresolved blockers: "
            + (", ".join(blockers) if blockers else "gate flag is false")
        )


def assert_lr_only_batch(
    batch: Mapping[str, Any],
    *,
    required_measurement_keys: Sequence[str],
    t2_policy: str = "disabled",
) -> None:
    """Reject pseudo targets and require explicit measured-domain observations.

    ``required_measurement_keys`` must be chosen by the eventual validated data
    contract.  This function does not guess whether the network should consume
    complex voxel FIDs, fitted metabolite maps or another representation.
    """
    present_prohibited = sorted(PROHIBITED_TARGET_KEYS.intersection(batch))
    if present_prohibited:
        raise ProhibitedTargetError(
            "Real LR-only adaptation batch contains prohibited target keys: "
            + ", ".join(present_prohibited)
        )
    missing = [key for key in required_measurement_keys if key not in batch]
    if missing:
        raise KeyError(f"Missing measured-domain batch fields: {missing}")
    policy = str(t2_policy).lower()
    if policy not in {"disabled", "evaluation_only", "weak_regularizer"}:
        raise ValueError(f"Unknown T2 policy: {t2_policy}")
    t2_keys = {"T2", "T2_IMAGE", "T2_CONDITION", "T2_REGULARIZER"}
    present_t2 = sorted(t2_keys.intersection(batch))
    if present_t2 and policy == "disabled":
        raise ProhibitedTargetError(
            "T2 fields are disabled until registration and texture-leakage controls pass: "
            + ", ".join(present_t2)
        )


def measured_domain_consistency_loss(
    predicted: torch.Tensor,
    measured: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    config: LRMeasurementLossConfig = LRMeasurementLossConfig(),
) -> torch.Tensor:
    """Robust loss between like-for-like measured-domain complex tensors.

    This function does not perform an acquisition transform.  The caller must
    supply ``predicted`` and ``measured`` in the same validated domain, units,
    layout and normalization.  Complex residuals are handled through magnitude
    squared or magnitude penalties without discarding phase beforehand.
    """
    config.validate()
    if predicted.shape != measured.shape:
        raise ValueError(
            f"Predicted/measured shape mismatch: {tuple(predicted.shape)} vs {tuple(measured.shape)}"
        )
    if predicted.dtype != measured.dtype:
        raise TypeError(
            f"Predicted/measured dtype mismatch: {predicted.dtype} vs {measured.dtype}"
        )
    residual = predicted - measured
    magnitude = residual.abs() if torch.is_complex(residual) else residual.abs()
    if config.kind == "l1":
        pointwise = magnitude
    elif config.kind == "l2":
        pointwise = magnitude.square()
    else:
        pointwise = torch.sqrt(magnitude.square() + float(config.epsilon)) - float(
            config.epsilon
        ) ** 0.5
    if weight is None:
        return pointwise.mean()
    weight_t = weight.to(device=pointwise.device, dtype=pointwise.dtype)
    try:
        weighted = pointwise * weight_t
    except RuntimeError as exc:
        raise ValueError(
            f"Weight shape {tuple(weight_t.shape)} is not broadcastable to {tuple(pointwise.shape)}"
        ) from exc
    denominator = torch.broadcast_to(weight_t, pointwise.shape).sum().clamp_min(1.0)
    return weighted.sum() / denominator


class PretrainedParameterAnchor:
    """Frozen copy of selected parameters for catastrophic-forgetting control."""

    def __init__(self, module: nn.Module, trainable_only: bool = True) -> None:
        self._reference = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if (parameter.requires_grad or not trainable_only)
        }
        if not self._reference:
            raise ValueError("No parameters were selected for anchoring")

    def loss(self, module: nn.Module) -> torch.Tensor:
        parameters = dict(module.named_parameters())
        missing = sorted(set(self._reference).difference(parameters))
        if missing:
            raise KeyError(f"Anchored parameters missing from module: {missing}")
        terms = []
        for name, reference in self._reference.items():
            current = parameters[name]
            terms.append((current - reference.to(current.device, current.dtype)).square().mean())
        return torch.stack(terms).mean()


def assert_fresh_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Reject inherited Adam moments for a new real-domain adaptation run."""
    if optimizer.state:
        raise ValueError(
            "Real-domain adaptation optimizer must start with fresh state; inherited optimizer moments detected"
        )


class GuardedLRAdaptationOptimizer:
    """Fail-closed optimizer session for a future validated LR-only pilot.

    Freshness is checked exactly once when the adaptation session is created, so
    historical Adam moments cannot be inherited while legitimate later steps may
    accumulate their own optimizer state.  The authoritative gate and LR-only
    batch contract are rechecked before every parameter update.
    """

    def __init__(
        self,
        *,
        gate: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        required_measurement_keys: Sequence[str],
        t2_policy: str = "disabled",
    ) -> None:
        assert_formal_real_training_allowed(gate)
        assert_fresh_optimizer(optimizer)
        self._gate = gate
        self._optimizer = optimizer
        self._required_measurement_keys = tuple(required_measurement_keys)
        if not self._required_measurement_keys:
            raise ValueError("At least one measured-domain batch key is required")
        self._t2_policy = t2_policy
        self.step_count = 0

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def step(self, *, batch: Mapping[str, Any], loss: torch.Tensor) -> None:
        """Validate then perform one parameter update."""
        assert_formal_real_training_allowed(self._gate)
        assert_lr_only_batch(
            batch,
            required_measurement_keys=self._required_measurement_keys,
            t2_policy=self._t2_policy,
        )
        if loss.ndim != 0:
            raise ValueError(f"Expected scalar loss, got shape {tuple(loss.shape)}")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LR-only loss; optimizer step refused")
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        self.step_count += 1
