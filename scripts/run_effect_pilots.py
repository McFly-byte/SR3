from __future__ import annotations

import copy
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "sr3_dmi_64_effect_finetune.json"
RUN_DIR = ROOT / "experiments" / "analyses/training_diagnostics_20260730" / "improvement_runs"


CANDIDATES = [
    {
        "name": "sr3_dmi_effect_ft_control",
        "gpu": "0",
        "diffusion": {},
    },
    {
        "name": "sr3_dmi_effect_ft_minsnr5",
        "gpu": "1",
        "diffusion": {"min_snr_gamma": 5.0},
    },
    {
        "name": "sr3_dmi_effect_ft_fidelity",
        "gpu": "2",
        "diffusion": {
            "x0_loss_weight": 0.05,
            "roi_mean_loss_weight": 0.01,
            "grad_loss_weight": 0.02,
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run two optimizer steps per candidate.')
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_dir = RUN_DIR / "generated_configs"
    log_dir = RUN_DIR / "launcher_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    processes = []
    for candidate in CANDIDATES:
        config = copy.deepcopy(base)
        config["name"] = candidate["name"] + ("_smoke" if args.smoke else "")
        config["gpu_ids"] = [int(candidate["gpu"])]
        config["model"]["diffusion"].update(candidate["diffusion"])
        if args.smoke:
            config["train"]["n_iter"] = 50002
            config["train"]["val_freq"] = 5000
            config["train"]["save_checkpoint_freq"] = 5000
            config["train"]["print_freq"] = 1
            config["train"]["val_max_samples"] = 8
        config_path = config_dir / f"{config['name']}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

        log_path = log_dir / f"{config['name']}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        command = [
            sys.executable,
            str(ROOT / "sr.py"),
            "-p",
            "train",
            "-c",
            str(config_path),
            "-gpu",
            candidate["gpu"],
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((config["name"], process, log_handle, log_path))
        print(f"started {config['name']} pid={process.pid} gpu={candidate['gpu']}", flush=True)

    failed = []
    for name, process, log_handle, log_path in processes:
        code = process.wait()
        log_handle.close()
        print(f"finished {name} exit={code} log={log_path}", flush=True)
        if code != 0:
            failed.append((name, code))

    if failed:
        print("failed candidates: " + ", ".join(f"{name}({code})" for name, code in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
