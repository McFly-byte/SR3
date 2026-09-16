#!/usr/bin/env python3
"""Audit T2/CSI physical geometry without claiming image registration.

This script reads first-party DICOM and Bruker metadata only. It does not resample,
interpolate, optimize a transform, or enable T2 use in training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_SCANS = {
    "R001": (57, 58, 59, 60),
    "R002": (30, 31, 32, 33),
    "R003": (35, 36, 37, 38),
    "R004": (59, 60, 61, 62),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_latin1(path: Path) -> str:
    return path.read_text(encoding="latin-1")


def raw(content: str, key: str) -> str:
    match = re.search(rf"(?m)^##\${re.escape(key)}=(.*)$", content)
    if not match:
        raise KeyError(key)
    lines = [match.group(1).strip()]
    for line in content[match.end():].splitlines():
        if line.startswith("##"):
            break
        if not line.startswith("$$"):
            lines.append(line.strip())
    return " ".join(part for part in lines if part).strip()


def numbers(content: str, key: str) -> list[float]:
    value = re.sub(r"^\(\s*[^)]*\)\s*", "", raw(content, key))
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", value)]


def text_value(content: str, key: str) -> str:
    value = re.sub(r"^\(\s*[^)]*\)\s*", "", raw(content, key)).strip()
    angle = re.findall(r"<([^>]*)>", value)
    return angle[0] if angle else value.split()[0]


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S,%f%z")


def parse_dicom_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M%S.%f%z")


def vec_close(left: list[float], right: list[float], atol: float = 1e-6) -> bool:
    return len(left) == len(right) and max(abs(a - b) for a, b in zip(left, right)) <= atol


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def find_t2(animal_dir: Path) -> Path:
    candidates = sorted(animal_dir.glob("*/*T2_Axial_AutoW*.dcm"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one T2 DICOM under {animal_dir}, got {candidates}")
    return candidates[0]


def dicom_geometry(path: Path) -> dict[str, Any]:
    import pydicom  # type: ignore

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    shared = ds.SharedFunctionalGroupsSequence[0]
    measures = shared.PixelMeasuresSequence[0]
    orientation = [float(value) for value in shared.PlaneOrientationSequence[0].ImageOrientationPatient]
    positions = [
        [float(value) for value in frame.PlanePositionSequence[0].ImagePositionPatient]
        for frame in ds.PerFrameFunctionalGroupsSequence
    ]
    pixel_spacing = [float(value) for value in measures.PixelSpacing]
    slice_thickness = float(measures.SliceThickness)
    frame_time = None
    if hasattr(ds, "AcquisitionDateTime"):
        frame_time = str(ds.AcquisitionDateTime)
    elif hasattr(ds, "SeriesDate") and hasattr(ds, "SeriesTime"):
        frame_time = f"{ds.SeriesDate}{ds.SeriesTime}"
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "frames": int(ds.NumberOfFrames),
        "pixel_spacing_mm": pixel_spacing,
        "slice_thickness_mm": slice_thickness,
        "orientation_6": orientation,
        "positions": positions,
        "first_position": positions[0],
        "last_position": positions[-1],
        "frame_of_reference_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "acquisition_datetime_raw": frame_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    animals: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for animal, scans in EXPECTED_SCANS.items():
        animal_dir = args.data_root / animal
        t2_path = find_t2(animal_dir)
        t2 = dicom_geometry(t2_path)
        t2_orientation_9 = t2["orientation_6"] + [0.0, 0.0, 1.0]
        t2_fov = [
            t2["columns"] * t2["pixel_spacing_mm"][1],
            t2["rows"] * t2["pixel_spacing_mm"][0],
            t2["frames"] * t2["slice_thickness_mm"],
        ]
        csi_rows: list[dict[str, Any]] = []
        for scan in scans:
            visu_path = animal_dir / "RAW" / str(scan) / "pdata" / "1" / "visu_pars"
            visu = read_latin1(visu_path)
            size = [int(value) for value in numbers(visu, "VisuCoreSize")]
            extent = numbers(visu, "VisuCoreExtent")
            orientation = numbers(visu, "VisuCoreOrientation")
            position = numbers(visu, "VisuCorePosition")
            acq_date = text_value(visu, "VisuAcqDate")
            acq_datetime = parse_iso(acq_date)
            t2_datetime = parse_dicom_datetime(t2["acquisition_datetime_raw"])
            row = {
                "animal_id": animal,
                "scan_id": scan,
                "csi_size_spec_xyz": size,
                "csi_extent_ppm_xyz_mm": extent,
                "csi_spatial_spacing_xyz_mm": [
                    extent[1] / size[1],
                    extent[2] / size[2],
                    extent[3] / size[3],
                ],
                "csi_orientation_9": orientation,
                "csi_position": position,
                "csi_frame_of_reference_uid": text_value(visu, "VisuSeriesFrameOfReferenceUid"),
                "csi_study_uid": text_value(visu, "VisuStudyUid"),
                "csi_acq_date_iso": acq_datetime.isoformat(timespec="milliseconds"),
                "visu_pars_sha256": sha256(visu_path),
                "orientation_matches_t2": vec_close(orientation, t2_orientation_9),
                "spatial_fov_matches_t2": vec_close(extent[1:], t2_fov),
                "frame_of_reference_matches_t2": text_value(visu, "VisuSeriesFrameOfReferenceUid") == t2["frame_of_reference_uid"],
                "study_uid_matches_t2": text_value(visu, "VisuStudyUid") == t2["study_instance_uid"],
                "csi_position_minus_t2_first_position_mm": subtract(position, t2["first_position"]),
                "csi_position_minus_t2_first_position_norm_mm": norm(subtract(position, t2["first_position"])),
                "t2_to_csi_hours": (acq_datetime - t2_datetime).total_seconds() / 3600.0,
            }
            csi_rows.append(row)
            all_rows.append(row)

        first_positions_equal = all(vec_close(row["csi_position"], csi_rows[0]["csi_position"]) for row in csi_rows)
        orientations_equal = all(vec_close(row["csi_orientation_9"], csi_rows[0]["csi_orientation_9"]) for row in csi_rows)
        fovs_equal = all(vec_close(row["csi_extent_ppm_xyz_mm"][1:], csi_rows[0]["csi_extent_ppm_xyz_mm"][1:]) for row in csi_rows)
        t2_first = t2["first_position"]
        csi_first = csi_rows[0]["csi_position"]
        z_delta = csi_first[2] - t2_first[2]
        expected_z_delta = (
            csi_rows[0]["csi_spatial_spacing_xyz_mm"][2]
            - t2["slice_thickness_mm"]
        ) / 2.0
        candidate_shared_coverage = (
            abs(csi_first[0] - t2_first[0]) <= 1e-6
            and abs(csi_first[1] - t2_first[1]) <= 1e-6
            and abs(z_delta - expected_z_delta) <= 1e-6
        )
        animals[animal] = {
            "t2": {key: value for key, value in t2.items() if key != "positions"},
            "t2_fov_xyz_mm": t2_fov,
            "csi_scans": csi_rows,
            "within_animal_csi_position_identical": first_positions_equal,
            "within_animal_csi_orientation_identical": orientations_equal,
            "within_animal_csi_fov_identical": fovs_equal,
            "candidate_common_physical_coverage_from_header_geometry": candidate_shared_coverage,
            "candidate_relation": {
                "x_y_positions_equal": abs(csi_first[0] - t2_first[0]) <= 1e-6 and abs(csi_first[1] - t2_first[1]) <= 1e-6,
                "observed_z_position_delta_mm": z_delta,
                "half_voxel_center_delta_expected_if_first_voxel_centers_share_edges_mm": expected_z_delta,
                "t2_to_first_selected_csi_hours": csi_rows[0]["t2_to_csi_hours"],
                "motion_risk_from_elapsed_time_not_quantified": True,
            },
            "registration_status": "geometry_initialization_only",
            "intensity_registration_executed": False,
            "resampling_executed": False,
            "t2_loss_allowed": False,
        }

    payload = {
        "schema_version": "1.0",
        "animal_count": len(animals),
        "scan_count": len(all_rows),
        "source_scope": "four enhanced multiframe T2 DICOM headers and 16 CSI pdata/1/visu_pars files",
        "confirmed": {
            "same_nominal_orientation": all(row["orientation_matches_t2"] for row in all_rows),
            "same_nominal_spatial_fov": all(row["spatial_fov_matches_t2"] for row in all_rows),
            "same_frame_of_reference_uid": all(row["frame_of_reference_matches_t2"] for row in all_rows),
            "same_study_uid": all(row["study_uid_matches_t2"] for row in all_rows),
            "within_animal_csi_geometry_constant": all(
                item["within_animal_csi_position_identical"]
                and item["within_animal_csi_orientation_identical"]
                and item["within_animal_csi_fov_identical"]
                for item in animals.values()
            ),
        },
        "interpretation_limits": {
            "header_geometry_is_not_registration": True,
            "visu_core_position_semantics_need_vendor_confirmation": True,
            "motion_between_T2_and_CSI_not_quantified": True,
            "no_intensity_registration_or_resampling": True,
            "no_registration_error_metric": True,
            "no_texture_leakage_test": True,
            "t2_loss_allowed": False,
            "training_permission_changed": False,
        },
        "animals": animals,
    }
    if len(all_rows) != 16:
        raise AssertionError(f"Expected 16 CSI rows, got {len(all_rows)}")
    if not payload["confirmed"]["same_nominal_orientation"]:
        raise AssertionError("T2/CSI nominal orientations do not all match")
    if not payload["confirmed"]["same_nominal_spatial_fov"]:
        raise AssertionError("T2/CSI nominal spatial FOVs do not all match")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "animal_count": len(animals),
        "scan_count": len(all_rows),
        "same_nominal_orientation": payload["confirmed"]["same_nominal_orientation"],
        "same_nominal_spatial_fov": payload["confirmed"]["same_nominal_spatial_fov"],
        "same_frame_of_reference_uid": payload["confirmed"]["same_frame_of_reference_uid"],
        "same_study_uid": payload["confirmed"]["same_study_uid"],
        "registration_status": "geometry_initialization_only",
        "t2_loss_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
