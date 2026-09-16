#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal training entry for the *improved* water-film SR3 recipe.

This does NOT replace sr.py.  It reuses the existing model construction
(Model.create_model) but drives a lightweight loop on top of
``ImprovedWaterfilmDataset`` (runtime re-degradation with block in [8,32] +
structural dropout).  Outputs go to experiments/improved_waterfilm/ so no old
checkpoint is touched.

Examples
--------
# smoke overfit (a few hundred steps, tiny subset):
python train_improved.py --config configs/improved_waterfilm.json \
    --iters 300 --batch 4 --data-len 64 --out experiments/improved_waterfilm

# full fine-tune from the released healthy checkpoint:
python train_improved.py --config configs/improved_waterfilm.json \
    --resume experiments/models/healthy_phantom_i300000_ema/checkpoint/I300000_E2522 \
    --iters 20000 --batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
os.chdir(str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.utils.data import DataLoader

import core.logger as Logger
import model as Model
from data.improved_waterfilm_dataset import ImprovedWaterfilmDataset


def resolve_target_step(begin_step: int, extra_iters: int) -> int:
    """Resolve the absolute step the loop should reach.

    ``--iters`` / ``extra_iters`` means *additional* training steps added on top
    of the resumed checkpoint, so a resume at 300000 with --iters 20000 targets
    step 320000.  Pure function so it can be unit-tested without a checkpoint.
    """
    return int(begin_step) + int(extra_iters)


def _build_opt(config_path: str, resume: str | None, out_root: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        opt = json.load(f)
    opt["phase"] = "train"
    opt["gpu_ids"] = [0]
    opt["distributed"] = False
    opt["is_main_process"] = True
    opt["rank"] = 0
    opt["world_size"] = 1
    opt["local_rank"] = 0
    opt["enable_wandb"] = False
    opt["path"]["experiments_root"] = out_root
    for k in ("log", "tb_logger", "results", "checkpoint"):
        opt["path"][k] = str(Path(out_root) / k)
    opt["path"]["resume_state"] = resume
    for k in ("log", "tb_logger", "results", "checkpoint"):
        Path(opt["path"][k]).mkdir(parents=True, exist_ok=True)
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/improved_waterfilm.json")
    ap.add_argument("--resume", default=None,
                    help="checkpoint prefix without '_gen.pth' suffix (e.g. .../I300000_E2522)")
    ap.add_argument("--out", default="experiments/improved_waterfilm")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data-len", type=int, default=None,
                    help="cap training samples (use a small number for overfit smoke)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    opt = _build_opt(args.config, args.resume, args.out)
    if args.iters is not None:
        opt["train"]["n_iter"] = int(args.iters)
    train_cfg = opt["datasets"]["train"]
    if args.batch is not None:
        train_cfg["batch_size"] = int(args.batch)
    if args.data_len is not None:
        train_cfg["data_len"] = int(args.data_len)

    torch.manual_seed(int(opt["train"].get("seed", 0)))
    np.random.seed(int(opt["train"].get("seed", 0)))

    Logger.setup_logger(None, opt["path"]["log"], "train", level=Logger.logging.INFO, screen=True)
    logger = Logger.logging.getLogger("base")

    # dataset / loader -------------------------------------------------------
    train_set = ImprovedWaterfilmDataset(
        train_cfg["dataroot"],
        split="train",
        data_len=int(train_cfg.get("data_len", -1)),
        hflip=bool(train_cfg.get("hflip", True)),
        use_lr=bool(train_cfg.get("use_lr", True)),
        use_t1=bool(train_cfg.get("use_t1", True)),
        use_flair=bool(train_cfg.get("use_flair", True)),
        use_met_onehot=bool(train_cfg.get("use_met_onehot", True)),
        use_mask_channel=bool(train_cfg.get("use_mask_channel", False)),
        min_block=int(train_cfg.get("min_block", 8)),
        max_block=int(train_cfg.get("max_block", 32)),
        struct_dropout_prob=float(train_cfg.get("struct_dropout_prob", 0.1)),
        window=str(train_cfg.get("window", "hamming")),
    )
    loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
    )
    logger.info("Improved train set size: %d", len(train_set))

    # model -------------------------------------------------------------------
    opt = Logger.dict_to_nonedict(opt)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["train"], schedule_phase="train"
    )

    n_iter = int(opt["train"]["n_iter"])   # this is *additional* steps
    print_freq = max(1, int(opt["train"].get("print_freq", 20)))
    save_freq = max(1, int(opt["train"].get("save_checkpoint_freq", 500)))
    step = getattr(diffusion, "begin_step", 0)
    epoch = getattr(diffusion, "begin_epoch", 0)
    target_step = resolve_target_step(step, n_iter)
    logger.info("Starting improved training at step %s for %d extra iters (target=%d).",
                step, n_iter, target_step)

    losses = []
    data_iter = iter(loader)
    while step < target_step:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch = next(data_iter)
        step += 1
        diffusion.current_step = step
        diffusion.feed_data(batch)
        ok = diffusion.optimize_parameters()
        log = diffusion.get_current_log()
        losses.append(float(log.get("l_pix", float("nan"))))
        if step % print_freq == 0 or step == 1:
            logger.info(
                "<epoch:%d step:%d> l_pix=%.4e %s",
                epoch, step,
                float(log.get("l_pix", float("nan"))),
                " ".join("{}={:.3e}".format(k, v) for k, v in log.items()
                         if k not in ("l_pix",)),
            )
        if step % save_freq == 0:
            diffusion.save_network(epoch, step)
            logger.info("Saved checkpoint at step %d.", step)

    diffusion.save_network(epoch, step, label="final")
    logger.info("Improved training DONE. last losses tail: %s",
                np.round(losses[-10:], 5).tolist())
    print("DONE", "last_loss=", float(np.mean(losses[-10:])) if losses else float("nan"))


if __name__ == "__main__":
    main()
