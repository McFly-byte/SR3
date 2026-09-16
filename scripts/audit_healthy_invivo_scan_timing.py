#!/usr/bin/env python3
"""Audit first-party acquisition timing without inferring injection-relative time."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EXPECTED_SCANS = {
    "R001": (57, 58, 59, 60),
    "R002": (30, 31, 32, 33),
    "R003": (35, 36, 37, 38),
    "R004": (59, 60, 61, 62),
}


def read_latin1(path: Path) -> str:
    return path.read_text(encoding="latin-1")


def scalar(content: str, key: str) -> str:
    match = re.search(rf"^##\${re.escape(key)}=\s*([^\r\n]+)", content, re.MULTILINE)
    if not match:
        raise KeyError(key)
    return match.group(1).strip().strip("<>")


def next_line_value(content: str, key: str) -> str:
    match = re.search(
        rf"^##\${re.escape(key)}=\s*\([^\r\n]*\)\s*[\r\n]+([^\r\n]+)",
        content,
        re.MULTILINE,
    )
    if not match:
        raise KeyError(key)
    return match.group(1).strip().strip("<>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_abs_time(value: str) -> tuple[datetime, int, int]:
    numbers = [int(part.strip()) for part in value.strip("()").split(",")]
    if len(numbers) != 3:
        raise ValueError(f"Unexpected ACQ_abs_time: {value}")
    epoch_s, millis, tz_minutes = numbers
    tz = timezone(timedelta(minutes=tz_minutes))
    dt = datetime.fromtimestamp(epoch_s + millis / 1000.0, tz=tz)
    return dt, millis, tz_minutes


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S,%f%z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    per_animal: dict[str, dict[str, Any]] = {}
    for animal, scans in EXPECTED_SCANS.items():
        animal_rows: list[dict[str, Any]] = []
        for retained_order, scan in enumerate(scans, start=1):
            scan_dir = args.data_root / animal / "RAW" / str(scan)
            acqp_path = scan_dir / "acqp"
            method_path = scan_dir / "method"
            visu_path = scan_dir / "pdata" / "1" / "visu_pars"
            acqp = read_latin1(acqp_path)
            method = read_latin1(method_path)
            visu = read_latin1(visu_path)

            acq_start, millis, tz_minutes = parse_abs_time(scalar(acqp, "ACQ_abs_time"))
            visu_acq = parse_iso(scalar(visu, "VisuAcqDate"))
            series_date = parse_iso(scalar(visu, "VisuSeriesDate"))
            scan_time_ms = float(scalar(method, "PVM_ScanTime"))
            acq_end = acq_start + timedelta(milliseconds=scan_time_ms)
            scan_name = next_line_value(acqp, "ACQ_scan_name")
            series_comment = next_line_value(visu, "VisuSeriesExperimentComment")
            row = {
                "animal_id": animal,
                "scan_id": scan,
                "retained_subset_order": retained_order,
                "acq_start_iso": acq_start.isoformat(timespec="milliseconds"),
                "acq_end_derived_iso": acq_end.isoformat(timespec="milliseconds"),
                "scan_time_ms": scan_time_ms,
                "timezone_offset_minutes": tz_minutes,
                "acq_abs_millisecond_component": millis,
                "visu_acq_date_iso": visu_acq.isoformat(timespec="milliseconds"),
                "visu_series_date_iso": series_date.isoformat(timespec="milliseconds"),
                "acq_abs_matches_visu_acq_date": abs((acq_start - visu_acq).total_seconds()) < 0.001,
                "series_minus_start_seconds": (series_date - acq_start).total_seconds(),
                "scan_name": scan_name,
                "series_comment": series_comment,
                "postinfusion_label_present": "postinfusion" in (scan_name + " " + series_comment).lower(),
                "injection_relative_time_minutes": None,
                "technical_repeat_status": "unresolved",
                "acqp_sha256": sha256(acqp_path),
                "method_sha256": sha256(method_path),
                "visu_pars_sha256": sha256(visu_path),
            }
            animal_rows.append(row)

        for index, row in enumerate(animal_rows):
            previous = animal_rows[index - 1] if index else None
            if previous is None:
                row["seconds_from_previous_start"] = None
                row["idle_seconds_after_previous_derived_end"] = None
            else:
                current_start = datetime.fromisoformat(row["acq_start_iso"])
                previous_start = datetime.fromisoformat(previous["acq_start_iso"])
                previous_end = datetime.fromisoformat(previous["acq_end_derived_iso"])
                row["seconds_from_previous_start"] = (current_start - previous_start).total_seconds()
                row["idle_seconds_after_previous_derived_end"] = (current_start - previous_end).total_seconds()

        starts = [datetime.fromisoformat(row["acq_start_iso"]) for row in animal_rows]
        monotonic = all(left < right for left, right in zip(starts, starts[1:]))
        per_animal[animal] = {
            "scan_ids": list(scans),
            "absolute_start_order_matches_scan_order": monotonic,
            "all_labeled_postinfusion": all(row["postinfusion_label_present"] for row in animal_rows),
            "start_to_start_intervals_seconds": [
                (right - left).total_seconds() for left, right in zip(starts, starts[1:])
            ],
            "injection_timestamp_available": False,
            "injection_relative_time_available": False,
            "technical_repeat_status": "unresolved",
        }
        rows.extend(animal_rows)

    payload = {
        "schema_version": "1.0",
        "scan_count": len(rows),
        "first_party_confirmed": {
            "absolute_scan_start_time": True,
            "derived_scan_end_from_PVM_ScanTime": True,
            "within_animal_absolute_order": all(
                item["absolute_start_order_matches_scan_order"] for item in per_animal.values()
            ),
            "postinfusion_label_for_all_selected_scans": all(
                item["all_labeled_postinfusion"] for item in per_animal.values()
            ),
        },
        "not_confirmed": {
            "injection_timestamp": "not found in the selected scan method/acqp/pdata/1/visu_pars files or readme",
            "minutes_post_injection": "cannot be derived without a first-party injection timestamp or experiment log",
            "technical_repeat_equivalence": "postinfusion scans are sequential approximately five-minute acquisitions and may contain biological dynamics",
            "reason_only_four_scans_retained": "not established by selected scan metadata",
        },
        "policy": {
            "scan_number_is_not_elapsed_time": True,
            "do_not_treat_as_technical_repeats": True,
            "allowed_repeatability_use": "descriptive longitudinal/within-animal consistency only until experimental timing semantics are supplied",
            "training_permission_changed": False,
        },
        "per_animal": per_animal,
        "rows": rows,
    }
    if len(rows) != 16:
        raise AssertionError(f"Expected 16 scans, got {len(rows)}")
    if not payload["first_party_confirmed"]["within_animal_absolute_order"]:
        raise AssertionError("Absolute acquisition order disagrees with scan ordering")
    if not all(row["acq_abs_matches_visu_acq_date"] for row in rows):
        raise AssertionError("ACQ_abs_time and VisuAcqDate disagree")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "scan_count": len(rows),
        "within_animal_absolute_order": payload["first_party_confirmed"]["within_animal_absolute_order"],
        "postinfusion_label_for_all_selected_scans": payload["first_party_confirmed"]["postinfusion_label_for_all_selected_scans"],
        "injection_relative_time_available": False,
        "technical_repeat_status": "unresolved",
        "training_permission_changed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
