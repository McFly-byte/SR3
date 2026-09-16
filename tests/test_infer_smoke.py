"""Smoke test: one inference pass (short DDIM) must yield a finite 64x64 SR.

Uses the same improved config and synthetic condition as the training smoke
test.  No checkpoint / real data required.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import core.logger as Logger
import model as Model

REPO = Path(__file__).resolve().parents[1]


def _make_opt():
    cfg = json.loads((REPO / "configs" / "improved_waterfilm.json").read_text(encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="sr3_infer_smoke_")
    cfg["phase"] = "val"
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
    cfg["validation"] = {"eval_network": "raw", "sample_num_steps": 3}
    return Logger.dict_to_nonedict(cfg)


def test_one_inference_step():
    opt = _make_opt()
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["val"], schedule_phase="val"
    )
    g = torch.Generator().manual_seed(0)
    b, c_cond, size = 1, 7, 64
    sr_cond = torch.rand(b, c_cond, size, size, generator=g) * 2 - 1
    batch = {"HR": torch.zeros(b, 1, size, size),
             "SR": sr_cond,
             "LR": sr_cond[:, 0:1].clone(),
             "MASK": torch.ones(b, 1, size, size)}
    diffusion.feed_data(batch)
    diffusion.test(continous=False, seed=0, sample_num_steps=3, network="raw")
    vis = diffusion.get_current_visuals(need_LR=False)
    sr = vis["SR"]
    assert sr.shape[-1] == size and sr.shape[-2] == size
    assert torch.isfinite(sr).all()
    assert sr.min() >= -1.001 and sr.max() <= 1.001
