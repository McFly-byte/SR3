import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.networks as networks


def load_config(path):
    json_str = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def make_dummy_batch(batch_size=2, image_size=64, device="cpu"):
    hr = torch.randn(batch_size, 1, image_size, image_size, device=device).clamp(-1, 1)
    lr = torch.randn(batch_size, 1, image_size, image_size, device=device).clamp(-1, 1)
    t1 = torch.randn(batch_size, 1, image_size, image_size, device=device).clamp(-1, 1)
    flair = torch.randn(batch_size, 1, image_size, image_size, device=device).clamp(-1, 1)
    met = torch.full((batch_size, 4, image_size, image_size), -1.0, device=device)
    met_ids = torch.arange(batch_size, device=device) % 4
    for b in range(batch_size):
        met[b, int(met_ids[b])] = 1.0
    mask = torch.ones(batch_size, 1, image_size, image_size, device=device)
    sr = torch.cat([lr, t1, flair, met], dim=1)
    return {"HR": hr, "SR": sr, "LR": lr, "MASK": mask}


def grad_norm(named_params, include):
    total_sq = 0.0
    count = 0
    max_abs = 0.0
    for name, param in named_params:
        if include not in name:
            continue
        count += param.numel()
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(grad.pow(2).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return total_sq ** 0.5, max_abs, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sr3_mrsi_64_phase2.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    opt = load_config(args.config)
    opt = copy.deepcopy(opt)
    opt["phase"] = "train"
    opt["gpu_ids"] = []
    opt.setdefault("model", {}).setdefault("diffusion", {})
    image_size = int(opt["model"]["diffusion"].get("image_size", 64))

    device = torch.device(args.device)
    net = networks.define_G(opt).to(device)
    net.set_loss(device)
    net.set_new_noise_schedule(opt["model"]["beta_schedule"]["train"], device)
    net.train()

    batch = make_dummy_batch(args.batch_size, image_size=image_size, device=device)
    loss = net(batch).sum() / float(batch["HR"].numel())
    loss.backward()

    params = list(net.named_parameters())
    groups = [
        "denoise_fn.t1_encoder",
        "denoise_fn.cross_attn_modules",
        "denoise_fn.downs.0",
        "denoise_fn.downs",
        "denoise_fn.mid",
        "denoise_fn.ups",
        "denoise_fn.lr_adapter",
        "denoise_fn.structure_adapter",
        "denoise_fn.fusion_blocks",
        "denoise_fn.met_embedding",
    ]

    print(f"config: {Path(args.config).as_posix()}")
    print(f"loss: {float(loss.detach().cpu()):.6g}")
    print("gradient norms:")
    for group in groups:
        norm, max_abs, count = grad_norm(params, group)
        if count == 0:
            print(f"  {group:<34} absent")
        else:
            flag = "OK"
            if norm == 0.0:
                flag = "ZERO"
            elif norm < 1e-10:
                flag = "TINY"
            print(f"  {group:<34} norm={norm:.6e} max={max_abs:.6e} params={count} [{flag}]")

    t1_norm, _, t1_count = grad_norm(params, "denoise_fn.t1_encoder")
    ca_norm, _, ca_count = grad_norm(params, "denoise_fn.cross_attn_modules")
    if t1_count > 0 or ca_count > 0:
        if t1_norm <= 1e-10 or ca_norm <= 1e-10:
            print("diagnosis: T1 cross-attention branch has zero/tiny gradient in at least one part.")
        else:
            print("diagnosis: T1 cross-attention branch receives non-zero gradients.")
    else:
        print("diagnosis: T1 cross-attention branch is disabled in this config.")


if __name__ == "__main__":
    main()
