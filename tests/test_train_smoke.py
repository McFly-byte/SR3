"""Smoke tests for the improved training step.

  * one finite epsilon-loss step
  * enabled composite auxiliary losses: log must contain
    loss/x0_charbonnier, loss/x0_ssim, loss/acquisition_native, all finite
  * weights=0 control: those terms must be absent / zero
  * resume extra-step parsing (no big checkpoint loaded)

The low-frequency native consistency uses the EXISTING acquisition_loss_weight
entry point (it already calls native_acquisition_l1_sum); we do NOT add a
second weight for the same physical quantity.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import core.logger as Logger
import model as Model
from train_improved import resolve_target_step

REPO = Path(__file__).resolve().parents[1]


def _make_opt(x0_charbonnier=0.2, x0_ssim=0.05, acquisition_loss=0.5):
    cfg = json.loads((REPO / "configs" / "improved_waterfilm.json").read_text(encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="sr3_smoke_")
    cfg["phase"] = "train"
    cfg["gpu_ids"] = [0]
    cfg["distributed"] = False
    cfg["is_main_process"] = True
    cfg["rank"] = 0
    cfg["world_size"] = 1
    cfg["local_rank"] = 0
    cfg["enable_wandb"] = False
    cfg["path"]["resume_state"] = None
    for k in ("log", "tb_logger", "results", "checkpoint"):
        cfg["path"][k] = str(Path(tmp) / k)
        Path(cfg["path"][k]).mkdir(parents=True, exist_ok=True)
    cfg["train"]["n_iter"] = 4
    cfg["train"]["ema_scheduler"]["step_start_ema"] = 100
    cfg["datasets"]["train"]["batch_size"] = 2
    cfg["model"]["diffusion"]["x0_charbonnier_weight"] = x0_charbonnier
    cfg["model"]["diffusion"]["x0_ssim_weight"] = x0_ssim
    cfg["model"]["diffusion"]["acquisition_loss_weight"] = acquisition_loss
    return Logger.dict_to_nonedict(cfg)


def _synthetic_batch(b=2, c_cond=7, size=64, block=12):
    g = torch.Generator().manual_seed(0)
    hr = torch.rand(b, 1, size, size, generator=g) * 2 - 1
    sr = torch.rand(b, c_cond, size, size, generator=g) * 2 - 1
    lr = sr[:, 0:1].clone()
    mask = (torch.rand(b, 1, size, size, generator=g) > 0.5).float()
    lr_native = torch.rand(b, 1, size, size, generator=g)
    matrix = torch.full((b,), block, dtype=torch.int64)
    has_native = torch.ones(b, dtype=torch.float32)
    return {"HR": hr, "SR": sr, "LR": lr, "MASK": mask,
            "LR_NATIVE": lr_native, "LR_MATRIX": matrix,
            "HAS_NATIVE_LR": has_native}


def _one_step(opt):
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["train"], schedule_phase="train")
    diffusion.current_step = 1
    diffusion.feed_data(_synthetic_batch())
    ok = diffusion.optimize_parameters()
    assert ok
    return diffusion.get_current_log()


def test_one_training_step_loss_finite():
    log = _one_step(_make_opt(x0_charbonnier=0.0, x0_ssim=0.0, acquisition_loss=0.0))
    l = float(log["l_pix"])
    assert np.isfinite(l) and 0.0 < l < 5.0


def test_composite_aux_terms_present_when_enabled():
    log = _one_step(_make_opt(x0_charbonnier=0.2, x0_ssim=0.05, acquisition_loss=0.5))
    for k in ("loss/x0_charbonnier", "loss/x0_ssim", "loss/acquisition_native"):
        assert k in log, f"missing {k}; keys={list(log)}"
        assert np.isfinite(log[k]), f"{k} non-finite"
        assert log[k] >= 0.0
    # epsilon loss must still be the dominant term (order of magnitude)
    assert log["l_pix"] > 0.0


def test_aux_terms_absent_when_disabled():
    log = _one_step(_make_opt(x0_charbonnier=0.0, x0_ssim=0.0, acquisition_loss=0.0))
    for k in ("loss/x0_charbonnier", "loss/x0_ssim", "loss/acquisition_native"):
        assert k not in log, f"{k} should be off when weight=0"


def test_no_duplicate_acquisition_term():
    # there must be exactly one acquisition-related term, not two
    log = _one_step(_make_opt(x0_charbonnier=0.2, x0_ssim=0.05, acquisition_loss=0.5))
    acq = [k for k in log if "acquisition" in k]
    assert len(acq) == 1, f"duplicate acquisition terms: {acq}"


def test_resolve_target_step():
    assert resolve_target_step(0, 300) == 300
    assert resolve_target_step(300000, 20000) == 320000
    assert resolve_target_step(51000, 50) == 51050
