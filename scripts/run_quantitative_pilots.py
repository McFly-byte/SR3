from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "experiments"
    / "models/current_mixed_i50500_raw"
    / "config_resolved.json"
)
BEST_CHECKPOINT = (
    "experiments/models/current_mixed_i50500_raw/"
    "checkpoint/I50500_E1931"
)
DATA_ROOT = Path(
    "D:/LMC/data/simulated_with_lesion/最终结果/"
    "sr3训练数据_msr_mrsi_npz_quant_v2"
)
ARTIFACT_ROOT = ROOT / "experiments" / "evaluations/model_selection_20260802"


CANDIDATES = (
    ("sr3_dmi_quant_v2_control", "0", 0.0),
    ("sr3_dmi_quant_v2_acq001", "1", 0.01),
    ("sr3_dmi_quant_v2_acq003", "2", 0.03),
)


def _make_config(base: dict, name: str, gpu: str, weight: float, smoke: bool) -> dict:
    config = copy.deepcopy(base)
    config["name"] = name + ("_smoke" if smoke else "")
    config["gpu_ids"] = [int(gpu)]
    config["path"]["resume_state"] = BEST_CHECKPOINT
    for split in ("train", "val"):
        config["datasets"][split]["dataroot"] = str(DATA_ROOT / split).replace("\\", "/")
        config["datasets"][split]["use_native_lr_consistency"] = True
    diffusion = config["model"]["diffusion"]
    diffusion["degradation_loss_weight"] = 0.0
    diffusion["acquisition_loss_weight"] = float(weight)
    diffusion["acquisition_window"] = "hamming"
    train = config["train"]
    train["n_iter"] = 50502 if smoke else 51000
    train["val_freq"] = 5000 if smoke else 500
    train["save_checkpoint_freq"] = 5000 if smoke else 500
    train["print_freq"] = 1 if smoke else 50
    train["val_max_samples"] = 8 if smoke else 64
    config["validation"]["eval_network"] = "raw"
    config["validation"]["save_image_count"] = 4
    config["wandb"]["project"] = "sr3_dmi_quantitative_v2"
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(BASE_CONFIG)
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    config_dir = ARTIFACT_ROOT / "generated_configs"
    log_dir = ARTIFACT_ROOT / "launcher_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name, gpu, weight in CANDIDATES:
        config = _make_config(base, name, gpu, weight, bool(args.smoke))
        config_path = config_dir / f"{config['name']}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        log_path = log_dir / f"{config['name']}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "sr.py"),
                "-p",
                "train",
                "-c",
                str(config_path),
                "-gpu",
                gpu,
            ],
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        jobs.append((config["name"], process, log_handle, log_path))
        print(f"started {config['name']} pid={process.pid} gpu={gpu} weight={weight}", flush=True)

    failures = []
    for name, process, log_handle, log_path in jobs:
        return_code = process.wait()
        log_handle.close()
        print(f"finished {name} exit={return_code} log={log_path}", flush=True)
        if return_code != 0:
            failures.append((name, return_code))
    if failures:
        print("failed: " + ", ".join(f"{name}({code})" for name, code in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

