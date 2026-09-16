from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_quantitative_full_validation import DATA_ROOT, ROOT, _selected_checkpoint


BASE_CONFIG = ROOT / "config" / "sr3_dmi_64_effect_eval.json"
ARTIFACT_ROOT = ROOT / "experiments" / "evaluations/model_selection_20260802"
CANDIDATES = (
    ("sr3_dmi_quant_dc_weak", "0", 5, 0.005, 1.0),
    ("sr3_dmi_quant_dc_mid", "1", 8, 0.01, 1.0),
    ("sr3_dmi_quant_dc_strong", "2", 10, 0.02, 0.5),
)


def main() -> int:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    checkpoint = _selected_checkpoint()
    config_dir = ARTIFACT_ROOT / "data_consistency_configs"
    log_dir = ARTIFACT_ROOT / "data_consistency_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, iterations, learning_rate, anchor_weight in CANDIDATES:
        config = copy.deepcopy(base)
        config["name"] = name
        config["gpu_ids"] = [int(gpu)]
        config["path"]["resume_state"] = str(checkpoint)
        config["datasets"]["val"]["dataroot"] = str(DATA_ROOT / "val").replace("\\", "/")
        config["datasets"]["val"]["use_native_lr_consistency"] = True
        config["validation"]["eval_network"] = "raw"
        config["validation"]["max_samples"] = 64
        config["validation"]["save_image_count"] = 4
        config["validation"]["native_data_consistency"] = {
            "enabled": True,
            "window": "hamming",
            "iterations": iterations,
            "learning_rate": learning_rate,
            "anchor_weight": anchor_weight,
        }
        config_path = config_dir / f"{name}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        log_path = log_dir / f"{name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "sr.py"), "-p", "val", "-c", str(config_path), "-gpu", gpu],
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        jobs.append((name, process, log_handle, log_path))
        print(
            f"started {name} pid={process.pid} gpu={gpu} "
            f"iterations={iterations} lr={learning_rate} anchor={anchor_weight}",
            flush=True,
        )
    failed = []
    for name, process, log_handle, log_path in jobs:
        code = process.wait()
        log_handle.close()
        print(f"finished {name} exit={code} log={log_path}", flush=True)
        if code != 0:
            failed.append((name, code))
    if failed:
        print("failed: " + ", ".join(f"{name}({code})" for name, code in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
