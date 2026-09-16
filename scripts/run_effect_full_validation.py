from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config" / "sr3_dmi_64_effect_eval.json"
RUN_DIR = ROOT / "experiments" / "analyses/training_diagnostics_20260730" / "improvement_runs"
SELECTION = RUN_DIR / "pilot_selection.json"


def main() -> int:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    with SELECTION.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)

    selected = selection["selected_for_full_validation"]
    selected_checkpoint = selected.get("checkpoint_base")
    if not selected_checkpoint:
        raise FileNotFoundError("The selected pilot checkpoint is missing.")

    baseline_checkpoint = base["path"]["resume_state"]
    suffix = f"{selected['candidate']}_i{selected['iteration']}"
    jobs = [
        ("sr3_dmi_effect_full_baseline_raw", baseline_checkpoint, "raw", "0"),
        (f"sr3_dmi_effect_full_{suffix}_raw", selected_checkpoint, "raw", "1"),
        (f"sr3_dmi_effect_full_{suffix}_ema", selected_checkpoint, "ema", "2"),
    ]

    config_dir = RUN_DIR / "generated_configs"
    log_dir = RUN_DIR / "full_eval_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []

    for name, checkpoint, network, gpu in jobs:
        config = copy.deepcopy(base)
        config["name"] = name
        config["gpu_ids"] = [int(gpu)]
        config["path"]["resume_state"] = checkpoint
        config["validation"]["eval_network"] = network
        config["validation"]["max_samples"] = -1
        config["validation"]["save_image_count"] = 8
        config["metrics"]["enable_hfen"] = True
        config["metrics"]["enable_frc"] = True
        config["metrics"]["enable_dmi_metrics"] = True
        config_path = config_dir / f"{name}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

        log_path = log_dir / f"{name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
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
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((name, process, log_handle, log_path))
        print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)

    failed = []
    for name, process, log_handle, log_path in processes:
        code = process.wait()
        log_handle.close()
        print(f"finished {name} exit={code} log={log_path}", flush=True)
        if code != 0:
            failed.append((name, code))
    if failed:
        print("failed evaluations: " + ", ".join(f"{name}({code})" for name, code in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
