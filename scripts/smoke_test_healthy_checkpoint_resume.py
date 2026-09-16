#!/usr/bin/env python3
"""Smoke-test registered healthy SR3 raw/optimizer/EMA checkpoint restoration.

This script is deliberately read-only with respect to registered checkpoints. It
constructs the model on CPU, restores the raw network and Adam state through the
project's real resume path, verifies tensor/state equality, and records the EMA
policy without performing any training update.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import create_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def tensor_state_summary(actual: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    actual_keys = set(actual)
    expected_keys = set(expected)
    shared = sorted(actual_keys & expected_keys)
    unequal = []
    max_abs = 0.0
    for key in shared:
        left = actual[key].detach().cpu()
        right = expected[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            unequal.append(key)
            if left.shape == right.shape and torch.is_floating_point(left) and torch.is_floating_point(right):
                max_abs = max(max_abs, float((left - right).abs().max().item()))
    return {
        "actual_key_count": len(actual_keys),
        "expected_key_count": len(expected_keys),
        "missing_keys": sorted(expected_keys - actual_keys),
        "unexpected_keys": sorted(actual_keys - expected_keys),
        "unequal_keys": unequal,
        "max_abs_diff": max_abs,
        "exact_match": not (expected_keys - actual_keys or actual_keys - expected_keys or unequal),
    }


def optimizer_summary(model, saved: Dict[str, Any]) -> Dict[str, Any]:
    restored = model.optG.state_dict()
    expected = saved["optimizer"]
    restored_groups = restored.get("param_groups", [])
    expected_groups = expected.get("param_groups", [])
    group_match = restored_groups == expected_groups
    restored_state = restored.get("state", {})
    expected_state = expected.get("state", {})
    state_keys_match = set(restored_state) == set(expected_state)
    tensor_mismatch = []
    scalar_mismatch = []
    for param_id in sorted(set(restored_state) & set(expected_state)):
        left_state = restored_state[param_id]
        right_state = expected_state[param_id]
        if set(left_state) != set(right_state):
            scalar_mismatch.append({"param_id": int(param_id), "field_keys_differ": True})
            continue
        for key in left_state:
            left = left_state[key]
            right = right_state[key]
            if torch.is_tensor(left) and torch.is_tensor(right):
                if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                    tensor_mismatch.append({"param_id": int(param_id), "field": key})
            elif left != right:
                scalar_mismatch.append({"param_id": int(param_id), "field": key})
    return {
        "saved_epoch": int(saved["epoch"]),
        "saved_iter": int(saved["iter"]),
        "restored_begin_epoch": int(model.begin_epoch),
        "restored_begin_step": int(model.begin_step),
        "param_group_count": len(restored_groups),
        "state_entry_count": len(restored_state),
        "param_groups_exact_match": group_match,
        "state_keys_match": state_keys_match,
        "tensor_mismatch": tensor_mismatch,
        "scalar_mismatch": scalar_mismatch,
        "exact_match": group_match and state_keys_match and not tensor_mismatch and not scalar_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    prefix = Path(args.checkpoint_prefix).resolve()
    output_path = Path(args.output).resolve()
    raw_path = Path(str(prefix) + "_gen.pth")
    opt_path = Path(str(prefix) + "_opt.pth")
    ema_path = Path(str(prefix) + "_ema_gen.pth")
    for path in (config_path, raw_path, opt_path, ema_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    opt = json.loads(config_path.read_text(encoding="utf-8"))
    opt = copy.deepcopy(opt)
    opt["phase"] = "train"
    opt["gpu_ids"] = []
    opt["distributed"] = False
    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["is_main_process"] = True
    opt.setdefault("path", {})["resume_state"] = str(prefix)
    opt["path"]["checkpoint"] = str(output_path.parent / "unused_checkpoint_output")

    expected_raw = torch.load(raw_path, map_location="cpu")
    expected_opt = torch.load(opt_path, map_location="cpu")
    expected_ema = torch.load(ema_path, map_location="cpu")

    model = create_model(opt)
    raw_check = tensor_state_summary(unwrap(model.netG).state_dict(), expected_raw)
    ema_check = tensor_state_summary(unwrap(model.netG_EMA).state_dict(), expected_ema)
    opt_check = optimizer_summary(model, expected_opt)

    payload = {
        "schema_version": 1,
        "mode": "read_only_checkpoint_resume_smoke_test",
        "device": str(model.device),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "checkpoint_prefix": str(prefix),
        "checkpoints": {
            "raw": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "optimizer": {"path": str(opt_path), "sha256": sha256(opt_path)},
            "ema": {"path": str(ema_path), "sha256": sha256(ema_path)},
        },
        "raw_restore": raw_check,
        "optimizer_restore": opt_check,
        "ema_restore": ema_check,
        "ema_policy": {
            "resume_behavior": "load registered EMA checkpoint when present",
            "reset_on_resume": bool(opt.get("train", {}).get("ema_scheduler", {}).get("reset_on_resume", False)),
            "formal_adaptation_policy": "Initialize trainable raw network from raw checkpoint and restore its matching optimizer only for exact resume experiments; preserve registered EMA as a separate evaluation shadow. Do not combine EMA weights with the historical raw optimizer. For a new real-domain adaptation run, optimizer policy must be declared separately and must not silently inherit historical Adam moments.",
        },
    }
    payload["passed"] = bool(raw_check["exact_match"] and opt_check["exact_match"] and ema_check["exact_match"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "output": str(output_path)}, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
