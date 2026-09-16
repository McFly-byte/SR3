# -*- coding: utf-8 -*-
"""
水膜 SR3 后处理 — 独立验证脚本
从磁盘重新读取所有文件，不依赖处理脚本的内部状态。
检查项：
  1. 后处理输出形状 (64,64) float32，所有值有限
  2. 原始 SR 文件 hash 未改变（与 postprocess_metadata.json 中记录对比，并重新计算）
  3. 后处理数组 != 原始 SR（确实经过处理）
  4. 后处理数组 != bicubic（没有把 bicubic 混入或冒充）
  5. 元数据声明 reference_free=True，参考未用于滤波参数
  6. 后处理脚本源码审计：滤波参数计算不引用参考数组
"""
import os
import json
import hashlib
import numpy as np

DATASETS = [
    {"key": "phantom_ygh", "dir": r"D:\LMC\data\phantom_ygh\最终处理结果\推理"},
    {"key": "waterfilm_double_chamber", "dir": r"D:\LMC\data\水膜数据处理\最终处理结果\推理"},
]
SCRIPT_PATH = r"D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement\postprocess_waterfilm_sr.py"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_script_reference_leakage(path):
    """
    审计后处理脚本：检查滤波参数计算部分是否引用了参考数据。
    这是静态源码审计，不保证运行时行为，但能捕获明显的参考泄漏。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    findings = []
    # 找到滤波参数计算的函数
    in_denoise_func = False
    in_lowpass_func = False
    for i, line in enumerate(lines, 1):
        if "def robust_denoise" in line:
            in_denoise_func = True
            in_lowpass_func = False
        elif "def lowpass_to_lr_grid" in line:
            in_lowpass_func = True
            in_denoise_func = False
        elif line.startswith("def ") and "robust_denoise" not in line and "lowpass_to_lr_grid" not in line:
            in_denoise_func = False
            in_lowpass_func = False

        if in_denoise_func or in_lowpass_func:
            # 检查是否引用了 ref / reference / measured_reference
            if any(kw in line for kw in ["ref", "reference", "measured_ref"]):
                # 排除注释和文档字符串中的提及
                stripped = line.strip()
                if not stripped.startswith("#") and '"""' not in stripped:
                    findings.append(f"Line {i}: potential reference usage in filter function: {stripped[:80]}")

    return findings


def verify_dataset(ds):
    d = ds["dir"]
    key = ds["key"]
    results = {"dataset": key, "checks": {}, "passed": True}

    def check(name, condition, detail=""):
        results["checks"][name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            results["passed"] = False

    # 加载所有文件
    sr_path = os.path.join(d, "sr3_ema_seed0_ddim50.npy")
    lowpass_path = os.path.join(d, "sr3_post_lowpass_lrgrid.npy")
    denoise_path = os.path.join(d, "sr3_post_denoise_robust.npy")
    bicubic_path = os.path.join(d, "bicubic_baseline_64.npy")
    meta_path = os.path.join(d, "postprocess_metadata.json")
    metrics_path = os.path.join(d, "postprocess_metrics.json")
    fig_path = os.path.join(d, "postprocess_comparison.png")

    # 检查文件存在
    for name, p in [("sr_original", sr_path), ("sr_lowpass", lowpass_path),
                     ("sr_denoise", denoise_path), ("bicubic", bicubic_path),
                     ("metadata", meta_path), ("metrics", metrics_path), ("figure", fig_path)]:
        check(f"file_exists_{name}", os.path.isfile(p), p)

    if not os.path.isfile(sr_path) or not os.path.isfile(lowpass_path) or not os.path.isfile(denoise_path):
        return results

    sr = np.load(sr_path)
    lowpass = np.load(lowpass_path)
    denoise = np.load(denoise_path)
    bicubic = np.load(bicubic_path)

    # 1. 形状和 dtype
    for name, arr in [("lowpass", lowpass), ("denoise", denoise)]:
        check(f"shape_{name}", arr.shape == (64, 64), f"got {arr.shape}")
        check(f"dtype_{name}", arr.dtype == np.float32, f"got {arr.dtype}")
        check(f"finite_{name}", np.isfinite(arr).all(),
              f"nan={np.isnan(arr).sum()}, inf={np.isinf(arr).sum()}")

    # 2. 原始 SR hash 未改变
    sr_hash_now = sha256_file(sr_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    sr_hash_recorded = meta.get("original_sr_sha256", "")
    check("original_sr_hash_matches_metadata", sr_hash_now == sr_hash_recorded,
          f"now={sr_hash_now[:16]}..., recorded={sr_hash_recorded[:16]}...")
    check("original_sr_unchanged_flag", meta.get("original_sr_unchanged", False) is True, "")

    # 3. 后处理数组 != 原始 SR
    check("lowpass_differs_from_original", not np.array_equal(lowpass, sr),
          f"max_abs_diff={np.abs(lowpass - sr).max():.6f}")
    check("denoise_differs_from_original", not np.array_equal(denoise, sr),
          f"max_abs_diff={np.abs(denoise - sr).max():.6f}")

    # 4. 后处理数组 != bicubic
    check("lowpass_differs_from_bicubic", not np.array_equal(lowpass, bicubic),
          f"max_abs_diff={np.abs(lowpass - bicubic).max():.6f}")
    check("denoise_differs_from_bicubic", not np.array_equal(denoise, bicubic),
          f"max_abs_diff={np.abs(denoise - bicubic).max():.6f}")

    # 5. 元数据声明参考未用于滤波
    check("metadata_reference_free", meta.get("postprocessing_reference_free", False) is True, "")
    check("metadata_ref_not_for_filter_params",
          meta.get("reference_used_for_filter_parameters", None) is False,
          str(meta.get("reference_used_for_filter_parameters")))
    check("metadata_ref_for_metrics_only",
          meta.get("reference_used_for_metrics_only", None) is True,
          str(meta.get("reference_used_for_metrics_only")))

    # 6. 检查参数来源记录
    for variant in ["lowpass_lrgrid", "denoise_robust"]:
        v = meta.get("variants", {}).get(variant, {})
        check(f"param_source_recorded_{variant}", "parameter_source" in v, v.get("parameter_source", ""))

    # 7. 值范围合理（[0,1] 附近，滤波不应产生极端值）
    for name, arr in [("lowpass", lowpass), ("denoise", denoise)]:
        check(f"range_nonnegative_{name}", arr.min() >= -1e-6, f"min={arr.min():.6f}")
        check(f"range_not_too_high_{name}", arr.max() <= 1.0 + 1e-6, f"max={arr.max():.6f}")

    return results


def main():
    print("=" * 70)
    print("独立验证：水膜 SR3 后处理")
    print("=" * 70)

    all_passed = True

    # 脚本源码审计
    print("\n[脚本源码审计] 检查滤波函数中是否引用参考数据...")
    findings = audit_script_reference_leakage(SCRIPT_PATH)
    if findings:
        print("  ⚠ 发现潜在引用:")
        for f in findings:
            print(f"    {f}")
        all_passed = False
    else:
        print("  ✓ 滤波函数中未发现参考数据引用")

    # 逐数据集验证
    for ds in DATASETS:
        print(f"\n{'─' * 70}")
        print(f"数据集: {ds['key']}")
        print(f"{'─' * 70}")
        r = verify_dataset(ds)
        for name, c in r["checks"].items():
            status = "✓" if c["passed"] else "✗ FAIL"
            detail = f" — {c['detail']}" if c["detail"] else ""
            print(f"  {status}  {name}{detail}")
        if not r["passed"]:
            all_passed = False
            print(f"\n  ✗ 数据集 {ds['key']} 验证未通过")
        else:
            print(f"\n  ✓ 数据集 {ds['key']} 全部通过")

    # 保存验证报告
    report = {
        "script_audit_findings": findings,
        "datasets": {},
        "all_passed": all_passed,
    }
    for ds in DATASETS:
        r = verify_dataset(ds)
        report["datasets"][ds["key"]] = r
        report_path = os.path.join(ds["dir"], "postprocess_independent_verification.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"\n  验证报告已保存: {report_path}")

    print(f"\n{'=' * 70}")
    if all_passed:
        print("✓ 全部验证通过")
    else:
        print("✗ 存在未通过项，请检查上方报告")
    print(f"{'=' * 70}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
