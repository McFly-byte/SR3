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
    for b in range(batch_size):
        met[b, b % 4] = 1.0
    mask = torch.ones(batch_size, 1, image_size, image_size, device=device)
    sr = torch.cat([lr, t1, flair, met], dim=1)
    return {"HR": hr, "SR": sr, "LR": lr, "MASK": mask}


def count_params(module, predicate=None):
    total = 0
    for name, param in module.named_parameters():
        if predicate is None or predicate(name):
            total += param.numel()
    return total


def adapter_name(name):
    keys = [
        "condition_parser",
        "lr_adapter",
        "structure_adapter",
        "met_embedding",
        "fusion_blocks",
    ]
    return any(key in name for key in keys)


def grad_norm(module, predicate):
    total_sq = 0.0
    seen = 0
    with_grad = 0
    for name, param in module.named_parameters():
        if not predicate(name):
            continue
        seen += param.numel()
        if param.grad is None:
            continue
        with_grad += param.numel()
        total_sq += float(param.grad.detach().pow(2).sum().item())
    return total_sq ** 0.5, seen, with_grad


def build_net(opt, device):
    opt = copy.deepcopy(opt)
    opt["phase"] = "train"
    opt["gpu_ids"] = []
    net = networks.define_G(opt).to(device)
    net.set_loss(device)
    net.set_new_noise_schedule(opt["model"]["beta_schedule"]["train"], device)
    net.train()
    return net


def run_backward(net, batch):
    net.zero_grad(set_to_none=True)
    loss = net(batch).sum() / float(batch["HR"].numel())
    loss.backward()
    return float(loss.detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sr3_mrsi_64_phase3_sgda.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    opt = load_config(cfg_path)
    image_size = int(opt["model"]["diffusion"].get("image_size", 64))
    device = torch.device(args.device)
    batch = make_dummy_batch(args.batch_size, image_size=image_size, device=device)

    sgda_net = build_net(opt, device)
    sgda_loss = run_backward(sgda_net, batch)
    total_params = count_params(sgda_net)
    adapter_params = count_params(sgda_net, adapter_name)
    adapter_grad, adapter_seen, adapter_with_grad = grad_norm(sgda_net, adapter_name)

    disabled_opt = copy.deepcopy(opt)
    disabled_opt["model"]["condition_adapter"]["enabled"] = False
    disabled_opt["model"]["unet"]["t1_in_channel"] = 0
    baseline_net = build_net(disabled_opt, device)
    baseline_loss = run_backward(baseline_net, batch)
    baseline_params = count_params(baseline_net)

    print(f"config: {cfg_path.as_posix()}")
    print(f"sgda loss: {sgda_loss:.6g}")
    print(f"baseline-disabled loss: {baseline_loss:.6g}")
    print(f"total params with SGDA: {total_params:,}")
    print(f"total params with adapter disabled: {baseline_params:,}")
    print(f"adapter params: {adapter_params:,}")
    print(
        "adapter grad norm: "
        f"{adapter_grad:.6e} (params={adapter_seen:,}, params_with_grad={adapter_with_grad:,})"
    )
    if adapter_params <= 0:
        raise RuntimeError("SGDA adapter parameters were not created.")
    if adapter_grad <= 1e-12:
        raise RuntimeError("SGDA adapter gradient is zero or too small.")
    print("smoke test: OK")


if __name__ == "__main__":
    main()
