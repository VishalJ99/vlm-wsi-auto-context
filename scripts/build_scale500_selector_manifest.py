#!/usr/bin/env python3
"""Build an all-case selector manifest from the scale500 worklist.

The selector probe expects a JSON file with a top-level ``records`` list.  The
scale500 production worklist is CSV, while detector provenance lives in the
per-group ``all_detections.json`` files.  This adapter joins those sources so
the selector can be run over exactly the selected 500 slides, in worklist order.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCALE500_ROOT = Path("runs/detector_pipeline_scale500_v1")
DEFAULT_OUTPUT = (
    DEFAULT_SCALE500_ROOT
    / "analysis"
    / "bbox_diversity_selection_all500_prohigh_v1"
    / "input_manifest.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_case_id(value: str) -> str:
    return Path(value).stem.replace("-", "_")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_detector_index(scale500_root: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for group in ("non_sv40", "sv40_skip_odd"):
        all_detections_path = scale500_root / group / "all_detections.json"
        rows = read_json(all_detections_path)
        if not isinstance(rows, list):
            raise SystemExit(f"Expected list in {all_detections_path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            case_id = normalize_case_id(str(row.get("case_id") or ""))
            if not case_id:
                continue
            if case_id in index:
                raise SystemExit(f"Duplicate detector case_id across groups: {case_id}")
            index[case_id] = (group, row)
    return index


def case_id_from_worklist(row: dict[str, str]) -> str:
    for key in ("Anon_Slide_ID", "source_filename", "dest_filename", "wsi_path"):
        value = row.get(key)
        if value:
            return normalize_case_id(value)
    raise SystemExit(f"Could not derive case id from worklist row: {row}")


def build_records(scale500_root: Path, worklist_csv: Path) -> list[dict[str, Any]]:
    detector_index = load_detector_index(scale500_root)
    with worklist_csv.open("r", encoding="utf-8", newline="") as f:
        worklist_rows = list(csv.DictReader(f))
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for offset, worklist_row in enumerate(worklist_rows, start=1):
        case_id = case_id_from_worklist(worklist_row)
        if case_id in seen:
            raise SystemExit(f"Duplicate worklist case_id: {case_id}")
        seen.add(case_id)
        detector_entry = detector_index.get(case_id)
        if detector_entry is None:
            missing.append(case_id)
            continue
        detector_group, detector_row = detector_entry
        case_dir = Path(str(detector_row.get("case_dir") or scale500_root / detector_group / case_id))
        detections_path = case_dir / "detections.json"
        records.append(
            {
                "selection_index": offset,
                "selection_index_within_stain": worklist_row.get("selection_index_within_stain"),
                "case_id": case_id,
                "worklist_case_id": worklist_row.get("case_id"),
                "anon_path_id": worklist_row.get("Anon_Path_ID"),
                "anon_patient_id": worklist_row.get("Anon_Patient_ID"),
                "anon_slide_id": worklist_row.get("Anon_Slide_ID"),
                "stain": worklist_row.get("stain"),
                "stain_label": worklist_row.get("stain"),
                "wsi_path": detector_row.get("wsi_path") or worklist_row.get("source_wsi_path") or worklist_row.get("source_path"),
                "worklist_wsi_path": worklist_row.get("wsi_path"),
                "source_wsi_path": worklist_row.get("source_wsi_path") or worklist_row.get("source_path"),
                "pipeline_status": detector_row.get("pipeline_status"),
                "final_boxes": len(detector_row.get("detections") or []),
                "case_dir": str(case_dir),
                "detections_path": str(detections_path),
                "final_overlay_png": str(case_dir / "final_detected_bboxes.png"),
                "source_detector_group": detector_group,
                "worklist_row": worklist_row,
            }
        )
    extra = sorted(set(detector_index) - seen)
    if missing:
        raise SystemExit(f"Missing detector records for {len(missing)} worklist cases: {missing[:20]}")
    if extra:
        raise SystemExit(f"Detector index has {len(extra)} cases outside the worklist: {extra[:20]}")
    return records


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale500-root", type=Path, default=DEFAULT_SCALE500_ROOT)
    parser.add_argument(
        "--worklist-csv",
        type=Path,
        default=DEFAULT_SCALE500_ROOT / "worklists" / "selected_wsis.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    scale500_root = args.scale500_root.expanduser().resolve()
    worklist_csv = args.worklist_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    records = build_records(scale500_root, worklist_csv)
    payload = {
        "created_at": utc_now(),
        "source_scale500_root": str(scale500_root),
        "source_worklist_csv": str(worklist_csv),
        "record_count": len(records),
        "stain_counts": dict(Counter(str(row.get("stain") or "") for row in records)),
        "detector_group_counts": dict(Counter(str(row.get("source_detector_group") or "") for row in records)),
        "records": records,
    }
    write_json(output, payload)
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
