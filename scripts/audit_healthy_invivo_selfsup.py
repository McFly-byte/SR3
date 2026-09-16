#!/usr/bin/env python3
"""Audit the four healthy-rat Bruker CSI studies without modifying source data.

All fields are extracted from local files. Missing or ambiguous fields are emitted as
"待确认" rather than filled with defaults.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

DATA_ROOT = Path(r"D:\LMC\data\invivo_zlx\zlx_healthy_rats_data")
PROJECT_ROOT = Path(r"D:\LMC\projects\Image-Super-Resolution-via-Iterative-Refinement")
OUT_DIR = PROJECT_ROOT / "experiments" / "healthy_invivo_selfsup" / "audit"
MODEL_DIR = PROJECT_ROOT / "experiments" / "models" / "healthy_phantom_i300000_ema"
SCANS = {
    "R001": [57, 58, 59, 60],
    "R002": [30, 31, 32, 33],
    "R003": [35, 36, 37, 38],
    "R004": [59, 60, 61, 62],
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="latin-1", errors="replace")


def bruker_raw(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^##\${re.escape(key)}=(.*)$", text)
    if not match:
        return None
    first = match.group(1).strip()
    lines = [first]
    cursor = match.end()
    for line in text[cursor:].splitlines():
        if line.startswith("##"):
            break
        if line.startswith("$$"):
            continue
        lines.append(line.strip())
    return " ".join(part for part in lines if part).strip()


def strip_shape(raw: str | None) -> str | None:
    if raw is None:
        return None
    return re.sub(r"^\(\s*[^)]*\)\s*", "", raw).strip()


def expand_repetitions(value: str) -> str:
    pattern = re.compile(r"@(\d+)\*\(([^()]*)\)")
    while True:
        match = pattern.search(value)
        if not match:
            return value
        count = int(match.group(1))
        token = match.group(2).strip()
        value = value[: match.start()] + " ".join([token] * count) + value[match.end() :]


def parse_tokens(raw: str | None) -> list[str]:
    value = strip_shape(raw)
    if not value:
        return []
    value = expand_repetitions(value)
    angle = re.findall(r"<([^>]*)>", value)
    if angle and re.sub(r"<[^>]*>", "", value).strip() == "":
        return angle
    return value.replace("(", " ").replace(")", " ").replace(",", " ").split()


def parse_numbers(raw: str | None) -> list[float]:
    out: list[float] = []
    for token in parse_tokens(raw):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def scalar(raw: str | None) -> Any:
    tokens = parse_tokens(raw)
    if not tokens:
        return "待确认"
    if len(tokens) == 1:
        token = tokens[0]
        try:
            number = float(token)
            return int(number) if number.is_integer() else number
        except ValueError:
            return token
    return tokens


def find_study_dir(animal_dir: Path) -> Path | None:
    candidates = [p for p in animal_dir.iterdir() if p.is_dir() and p.name != "RAW"]
    return candidates[0] if len(candidates) == 1 else None


def json_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def audit_dicom(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "t2_path": str(path),
        "t2_exists": path.is_file(),
        "t2_bytes": path.stat().st_size if path.is_file() else 0,
        "t2_parser": "待确认",
        "t2_rows": "待确认",
        "t2_columns": "待确认",
        "t2_frames": "待确认",
        "t2_pixel_spacing_mm": "待确认",
        "t2_slice_thickness_mm": "待确认",
        "t2_spacing_between_slices_mm": "待确认",
        "t2_fov_mm": "待确认",
        "t2_image_orientation_patient": "待确认",
        "t2_first_image_position_patient": "待确认",
        "t2_last_image_position_patient": "待确认",
        "t2_series_description": "待确认",
    }
    if not path.is_file():
        return result
    try:
        import pydicom  # type: ignore

        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        rows = int(getattr(ds, "Rows", 0)) or "待确认"
        columns = int(getattr(ds, "Columns", 0)) or "待确认"
        frames = int(getattr(ds, "NumberOfFrames", 1))
        pixel_spacing: Any = "待确认"
        slice_thickness: Any = "待确认"
        orientation: Any = "待确认"
        first_position: Any = "待确认"
        last_position: Any = "待确认"
        spacing_between: Any = "待确认"
        fov: Any = "待确认"

        if hasattr(ds, "SharedFunctionalGroupsSequence"):
            shared = ds.SharedFunctionalGroupsSequence[0]
            if hasattr(shared, "PixelMeasuresSequence"):
                measures = shared.PixelMeasuresSequence[0]
                if hasattr(measures, "PixelSpacing"):
                    pixel_spacing = [float(x) for x in measures.PixelSpacing]
                if hasattr(measures, "SliceThickness"):
                    slice_thickness = float(measures.SliceThickness)
            if hasattr(shared, "PlaneOrientationSequence"):
                orientation = [float(x) for x in shared.PlaneOrientationSequence[0].ImageOrientationPatient]
        else:
            if hasattr(ds, "PixelSpacing"):
                pixel_spacing = [float(x) for x in ds.PixelSpacing]
            if hasattr(ds, "SliceThickness"):
                slice_thickness = float(ds.SliceThickness)
            if hasattr(ds, "ImageOrientationPatient"):
                orientation = [float(x) for x in ds.ImageOrientationPatient]

        positions: list[list[float]] = []
        if hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            for frame in ds.PerFrameFunctionalGroupsSequence:
                if hasattr(frame, "PlanePositionSequence"):
                    positions.append([float(x) for x in frame.PlanePositionSequence[0].ImagePositionPatient])
        elif hasattr(ds, "ImagePositionPatient"):
            positions.append([float(x) for x in ds.ImagePositionPatient])
        if positions:
            first_position = positions[0]
            last_position = positions[-1]
        if len(positions) > 1:
            distances = []
            for p0, p1 in zip(positions[:-1], positions[1:]):
                distances.append(math.sqrt(sum((b - a) ** 2 for a, b in zip(p0, p1))))
            distances.sort()
            spacing_between = distances[len(distances) // 2]
        if isinstance(rows, int) and isinstance(columns, int) and isinstance(pixel_spacing, list):
            fov = [rows * pixel_spacing[0], columns * pixel_spacing[1], frames * float(slice_thickness)]

        result.update(
            {
                "t2_parser": "pydicom_enhanced_multiframe" if hasattr(ds, "SharedFunctionalGroupsSequence") else "pydicom",
                "t2_rows": rows,
                "t2_columns": columns,
                "t2_frames": frames,
                "t2_pixel_spacing_mm": pixel_spacing,
                "t2_slice_thickness_mm": slice_thickness,
                "t2_spacing_between_slices_mm": spacing_between,
                "t2_fov_mm": fov,
                "t2_image_orientation_patient": orientation,
                "t2_first_image_position_patient": first_position,
                "t2_last_image_position_patient": last_position,
                "t2_series_description": str(getattr(ds, "SeriesDescription", "待确认")),
            }
        )
    except Exception as exc:  # preserve the exact parser failure for auditability
        result["t2_parser"] = f"unreadable:{type(exc).__name__}:{exc}"
    return result


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else "待确认",
        "load_status": "not_attempted",
    }
    if not path.is_file():
        return result
    try:
        import torch

        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(str(path), map_location="cpu")
        result["load_status"] = "ok"
        result["top_level_type"] = type(obj).__name__
        if isinstance(obj, dict):
            result["top_level_keys"] = list(obj.keys())[:50]
            tensor_items = [(k, v) for k, v in obj.items() if hasattr(v, "shape")]
            result["tensor_count"] = len(tensor_items)
            result["parameter_count"] = int(sum(math.prod(v.shape) for _, v in tensor_items))
            result["first_tensors"] = [
                {"key": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in tensor_items[:12]
            ]
            for k, v in tensor_items:
                if k.endswith("noise_level_mlp.0.weight") or k.endswith("final_conv.2.weight"):
                    result.setdefault("semantic_tensors", []).append(
                        {"key": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                    )
            if "optimizer" in obj and isinstance(obj["optimizer"], dict):
                opt = obj["optimizer"]
                result["optimizer_param_groups"] = len(opt.get("param_groups", []))
                result["optimizer_state_entries"] = len(opt.get("state", {}))
                result["epoch"] = obj.get("epoch", "待确认")
                result["iter"] = obj.get("iter", "待确认")
        del obj
    except Exception as exc:
        result["load_status"] = f"failed:{type(exc).__name__}:{exc}"
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t2_rows: list[dict[str, Any]] = []

    for animal_id, scan_ids in SCANS.items():
        animal_dir = DATA_ROOT / animal_id
        raw_root = animal_dir / "RAW"
        study_dir = find_study_dir(animal_dir)
        t2_path = (study_dir / "T2_Axial_AutoW_EnIm1.dcm") if study_dir else animal_dir / "T2_Axial_AutoW_EnIm1.dcm"
        t2 = audit_dicom(t2_path)
        t2_rows.append({"animal_id": animal_id, **t2})
        for order, scan_id in enumerate(scan_ids, start=1):
            scan_dir = raw_root / str(scan_id)
            method_path = scan_dir / "method"
            acqp_path = scan_dir / "acqp"
            reco_path = scan_dir / "pdata" / "1" / "reco"
            visu_path = scan_dir / "pdata" / "1" / "visu_pars"
            seq_path = scan_dir / "pdata" / "1" / "2dseq"
            ser_path = scan_dir / "pv2tsdata" / "1" / "ser"
            method = read_text(method_path) if method_path.is_file() else ""
            acqp = read_text(acqp_path) if acqp_path.is_file() else ""
            reco = read_text(reco_path) if reco_path.is_file() else ""
            visu = read_text(visu_path) if visu_path.is_file() else ""

            matrix = [int(x) for x in parse_numbers(bruker_raw(method, "PVM_Matrix"))]
            spec_matrix = [int(x) for x in parse_numbers(bruker_raw(method, "PVM_SpecMatrix"))]
            fov = parse_numbers(bruker_raw(method, "PVM_Fov"))
            spatial_resolution = parse_numbers(bruker_raw(method, "PVM_SpatResol"))
            reco_size = [int(x) for x in parse_numbers(bruker_raw(reco, "RECO_size"))]
            visu_size = [int(x) for x in parse_numbers(bruker_raw(visu, "VisuCoreSize"))]
            phase1 = parse_numbers(bruker_raw(acqp, "ACQ_spatial_phase_1"))
            phase2 = parse_numbers(bruker_raw(acqp, "ACQ_spatial_phase_2"))
            full_grid = math.prod(matrix) if matrix else None
            sampled_encodes = len(phase1) if phase1 and len(phase1) == len(phase2) else None
            acq_size = [int(x) for x in parse_numbers(bruker_raw(acqp, "ACQ_size"))]
            word_size = str(scalar(bruker_raw(acqp, "ACQ_word_size")))
            storage_block_bytes = acq_size[0] * 4 if acq_size and word_size == "_32_BIT" else None
            ser_storage_blocks = (
                ser_path.stat().st_size // storage_block_bytes
                if ser_path.is_file() and storage_block_bytes
                else None
            )

            required = [method_path, acqp_path, reco_path, visu_path, seq_path, ser_path]
            missing = [str(p) for p in required if not p.is_file()]
            scan_name = scalar(bruker_raw(acqp, "ACQ_scan_name"))
            row = {
                "animal_id": animal_id,
                "scan_id": scan_id,
                "dynamic_order_in_current_subset": order,
                "timepoint_minutes": "待确认",
                "acquisition_type": scan_name,
                "method": scalar(bruker_raw(method, "Method")),
                "nucleus": scalar(bruker_raw(acqp, "NUC1")),
                "bf1_mhz": scalar(bruker_raw(acqp, "BF1")),
                "flip_angle_deg": scalar(bruker_raw(acqp, "ACQ_flip_angle")),
                "csi_matrix": matrix or "待确认",
                "spectral_points": spec_matrix[0] if spec_matrix else "待确认",
                "fov_mm": fov or "待确认",
                "voxel_mm": spatial_resolution or "待确认",
                "tr_ms": scalar(bruker_raw(method, "PVM_RepetitionTime")),
                "te_ms": scalar(bruker_raw(method, "PVM_EchoTime")),
                "averages_method": scalar(bruker_raw(method, "PVM_NAverages")),
                "acqp_na": scalar(bruker_raw(acqp, "NA")),
                "spectral_bw_hz": scalar(bruker_raw(method, "PVM_SpecSWH")),
                "spectral_bw_ppm": scalar(bruker_raw(method, "PVM_SpecSW")),
                "spectral_acquisition_time_ms": scalar(bruker_raw(method, "PVM_SpecAcquisitionTime")),
                "scan_time_ms": scalar(bruker_raw(method, "PVM_ScanTime")),
                "receiver_count": scalar(bruker_raw(method, "PVM_EncNReceivers")),
                "available_receivers": scalar(bruker_raw(method, "PVM_EncAvailReceivers")),
                "slice_orientation": scalar(bruker_raw(method, "PVM_SPackArrSliceOrient")),
                "read_orientation": scalar(bruker_raw(method, "PVM_SPackArrReadOrient")),
                "grad_orientation": parse_numbers(bruker_raw(method, "PVM_SPackArrGradOrient")) or "待确认",
                "read_offset_mm": scalar(bruker_raw(method, "PVM_SPackArrReadOffset")),
                "phase1_offset_mm": scalar(bruker_raw(method, "PVM_SPackArrPhase1Offset")),
                "phase2_offset_mm": scalar(bruker_raw(method, "PVM_SPackArrPhase2Offset")),
                "slice_offset_mm": scalar(bruker_raw(method, "PVM_SPackArrSliceOffset")),
                "encoding_steps_1": parse_numbers(bruker_raw(method, "PVM_EncSteps1")) or "待确认",
                "encoding_steps_2": parse_numbers(bruker_raw(method, "PVM_EncSteps2")) or "待确认",
                "full_cartesian_encode_count": full_grid or "待确认",
                "sampled_encode_count": sampled_encodes or "待确认",
                "sampling_fraction": (sampled_encodes / full_grid) if sampled_encodes and full_grid else "待确认",
                "acq_size": acq_size or "待确认",
                "acq_word_size": word_size,
                "acq_byte_order": scalar(bruker_raw(acqp, "BYTORDA")),
                "reco_size": reco_size or "待确认",
                "reco_fov_cm": parse_numbers(bruker_raw(reco, "RECO_fov")) or "待确认",
                "reco_word_type": scalar(bruker_raw(reco, "RECO_wordtype")),
                "reco_byte_order": scalar(bruker_raw(reco, "RECO_byte_order")),
                "reco_image_type": scalar(bruker_raw(reco, "RECO_image_type")),
                "reco_map_slope": scalar(bruker_raw(reco, "RECO_map_slope")),
                "reco_map_offset": scalar(bruker_raw(reco, "RECO_map_offset")),
                "reco_rotate": parse_numbers(bruker_raw(reco, "RECO_rotate")) or "待确认",
                "reco_transposition": scalar(bruker_raw(reco, "RECO_transposition")),
                "visu_size": visu_size or "待确认",
                "visu_extent": parse_numbers(bruker_raw(visu, "VisuCoreExtent")) or "待确认",
                "visu_orientation": parse_numbers(bruker_raw(visu, "VisuCoreOrientation")) or "待确认",
                "visu_position": parse_numbers(bruker_raw(visu, "VisuCorePosition")) or "待确认",
                "visu_data_slope": scalar(bruker_raw(visu, "VisuCoreDataSlope")),
                "visu_data_offset": scalar(bruker_raw(visu, "VisuCoreDataOffs")),
                "method_exists": method_path.is_file(),
                "acqp_exists": acqp_path.is_file(),
                "reco_exists": reco_path.is_file(),
                "visu_pars_exists": visu_path.is_file(),
                "seq2d_exists": seq_path.is_file(),
                "seq2d_bytes": seq_path.stat().st_size if seq_path.is_file() else 0,
                "ser_exists": ser_path.is_file(),
                "ser_bytes": ser_path.stat().st_size if ser_path.is_file() else 0,
                "ser_storage_block_bytes": storage_block_bytes or "待确认",
                "ser_storage_block_count": ser_storage_blocks or "待确认",
                "ser_storage_matches_spatial_voxel_count": (
                    ser_storage_blocks == full_grid
                    if ser_storage_blocks is not None and full_grid is not None
                    else "待确认"
                ),
                "ser_interpretation": "pv2tsdata export; appears to contain one 1024-int32 block per 9x9x5 spatial voxel; complex/time layout still requires validation",
                "t2_available": t2["t2_exists"],
                "qc_status": "PASS_HEADER_AUDIT" if not missing else "FAIL_MISSING_FILES",
                "exclusion_reason": "" if not missing else ";".join(missing),
                "evidence_scope": "local Bruker method/acqp/reco/visu_pars and file metadata",
            }
            rows.append(row)

    checkpoints = []
    ckpt_dir = MODEL_DIR / "checkpoint"
    for filename in [
        "I300000_E2522_ema_gen.pth",
        "I300000_E2522_gen.pth",
        "I300000_E2522_opt.pth",
    ]:
        checkpoints.append(inspect_checkpoint(ckpt_dir / filename))

    csv_path = OUT_DIR / "healthy_rats_scan_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(value) for key, value in row.items()})

    t2_csv = OUT_DIR / "healthy_rats_t2_audit.csv"
    with t2_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(t2_rows[0].keys()))
        writer.writeheader()
        for row in t2_rows:
            writer.writerow({key: json_value(value) for key, value in row.items()})

    payload = {
        "schema_version": "1.0",
        "source_data_root": str(DATA_ROOT),
        "evidence_policy": "Only local files and reproducible calculations; unresolved fields are 待确认.",
        "scan_count": len(rows),
        "animal_count": len(SCANS),
        "scans": rows,
        "t2": t2_rows,
        "checkpoints": checkpoints,
        "known_limitations": [
            "Current disk subset contains four scans per animal; original complete dynamic series is not established.",
            "Scan number defines order within the current subset but not elapsed post-infusion minutes.",
            "T2-to-CSI registration has not been performed or validated.",
            "The 2dseq is a reconstructed magnitude spectroscopic image, not an HR DMI/MRSI label.",
            "The pv2tsdata ser export contains 405 non-empty storage blocks of 1024 int32 values, matching 9x9x5 spatial voxels; complex/time interleaving still requires validation.",
        ],
    }
    json_path = OUT_DIR / "healthy_rats_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    sampled_counts = sorted({r["sampled_encode_count"] for r in rows}, key=str)
    sampling_fractions = sorted({r["sampling_fraction"] for r in rows}, key=str)
    markdown = [
        "# 健康大鼠真实数据与健康基线首轮审计",
        "",
        "## 证据范围",
        "",
        "本报告仅汇总本地 Bruker `method/acqp/reco/visu_pars`、文件元数据、DICOM 头（若可解析）、checkpoint 内容与可复现计算。未确认项不填默认值。",
        "",
        "## 已核实",
        "",
        f"- 动物数：{len(SCANS)}；当前磁盘 CSI 扫描数：{len(rows)}。",
        f"- 所有必需文件完整的扫描数：{sum(r['qc_status'] == 'PASS_HEADER_AUDIT' for r in rows)}/{len(rows)}。",
        f"- 实际空间编码计数集合：{sampled_counts}；相对 9×9×5 完整网格的采样比例集合：{sampling_fractions}。",
        "- 该采样计数意味着前向算子必须尊重实际 phase-encoding 列表与顺序，不能仅以任意插值或普通缩放代替。",
        "- `2dseq` 为厂商重建的 magnitude spectroscopic image；它不是 HR 代谢图标签。",
        "",
        "## 尚不能确认/停止条件",
        "",
        "- `pv2tsdata/1/ser` 文件大小对应 405 个非空的 1024×int32 存储块，与 9×9×5 空间体素数一致；它不是可直接按 305 条原始编码线读取的 k-space。",
        "- `ser` 的复数/时间交织及与厂商 `2dseq` 的数值对应尚未通过单元测试确认。",
        "- 4次扫描的真实注射后分钟数尚未确认；当前仅能记录 scan 顺序。",
        "- T2 与 CSI 的配准关系尚未验证，因此不得启用 T2 正则或以 T2 评价边界一致性。",
        "- 在采集算子和谱处理正确性完成测试前，不启动正式自监督微调。",
        "",
        "## 文件",
        "",
        f"- `{csv_path}`",
        f"- `{t2_csv}`",
        f"- `{json_path}`",
    ]
    (OUT_DIR / "INITIAL_AUDIT_SUMMARY.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "scan_csv": str(csv_path),
        "t2_csv": str(t2_csv),
        "json": str(json_path),
        "summary": str(OUT_DIR / "INITIAL_AUDIT_SUMMARY.md"),
        "scan_count": len(rows),
        "pass_count": sum(r["qc_status"] == "PASS_HEADER_AUDIT" for r in rows),
        "sampled_encode_counts": sampled_counts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
