from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "sr3_dmi_64_effect_eval.json"
DATA_ROOT = Path(
    "D:/LMC/data/simulated_with_lesion/最终结果/"
    "sr3训练数据_msr_mrsi_npz_quant_v2"
)
ARTIFACT_ROOT = ROOT / "experiments" / "evaluations/model_selection_20260802"
PREVIOUS_BEST = (
    ROOT
    / "experiments"
    / "models/current_mixed_i50500_raw"
    / "checkpoint"
    / "I50500_E1931"
)


def _selected_checkpoint() -> Path:
    candidates = []
    for run in (ROOT / "experiments").glob("sr3_dmi_quant_v2_control_*"):
        config_path = run / "config_resolved.json"
        if not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("name") != "sr3_dmi_quant_v2_control":
                continue
        matches = [
            path for path in run.rglob("I51000_E*_gen.pth")
            if not path.name.endswith("_ema_gen.pth")
        ]
        if matches:
            candidates.append(max(matches, key=lambda path: path.stat().st_mtime))
    if not candidates:
        raise FileNotFoundError("No completed quantitative-v2 control checkpoint was found.")
    gen_path = max(candidates, key=lambda path: path.stat().st_mtime)
    return gen_path.with_name(gen_path.name[: -len("_gen.pth")])


def main() -> int:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    selected = _selected_checkpoint()
    jobs = (
        ("sr3_dmi_quant_full_previous_raw", PREVIOUS_BEST, "raw", "0"),
        ("sr3_dmi_quant_full_selected_raw", selected, "raw", "1"),
        ("sr3_dmi_quant_full_selected_ema", selected, "ema", "2"),
    )

    config_dir = ARTIFACT_ROOT / "full_validation_configs"
    log_dir = ARTIFACT_ROOT / "full_validation_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for name, checkpoint, network, gpu in jobs:
        config = copy.deepcopy(base)
        config["name"] = name
        config["gpu_ids"] = [int(gpu)]
        config["path"]["resume_state"] = str(checkpoint)
        config["datasets"]["val"]["dataroot"] = str(DATA_ROOT / "val").replace("\\", "/")
        config["datasets"]["val"]["use_native_lr_consistency"] = True
        config["validation"]["eval_network"] = network
        config["validation"]["max_samples"] = -1
        config["validation"]["save_image_count"] = 8
        config["model"]["diffusion"]["acquisition_loss_weight"] = 0.0
        config["model"]["diffusion"]["acquisition_window"] = "hamming"
        config_path = config_dir / f"{name}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        log_path = log_dir / f"{name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "sr.py"),
                "-p",
                "val",
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
        processes.append((name, process, log_handle, log_path))
        print(f"started {name} pid={process.pid} gpu={gpu} checkpoint={checkpoint}", flush=True)

    failures = []
    for name, process, log_handle, log_path in processes:
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
