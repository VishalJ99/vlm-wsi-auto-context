#!/usr/bin/env python3
"""Export detector-pipeline bbox outputs as supervised detector datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKET = "PER-248"
EXPORT_VERSION = "detector_training_dataset_export_v1_2026-05-31"
COORDINATE_SYSTEM = "normalized_0_1000_y_min_x_min_y_max_x_max"


@dataclass(frozen=True)
class BoxRecord:
    normalized_yxyx: tuple[float, float, float, float]
    pixel_xywh: tuple[float, float, float, float]
    yolo_xywh: tuple[float, float, float, float]
    source_detection: dict[str, Any]


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    case_display: str
    wsi_path: str
    source_thumbnail_path: Path
    output_image_relpath: str
    output_image_abspath: Path
    width: int
    height: int
    stain: str
    patient_id: str
    slide_id: str
    group_id: str
    split: str
    source_pipeline_record: dict[str, Any]
    boxes: tuple[BoxRecord, ...]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _repo_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _redacted_argv(argv: list[str]) -> str:
    return " ".join(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stage1_thumbnail_index(pipeline_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    path = pipeline_root / "intermediate_stage_artifacts" / "stage1_cases.jsonl"
    if not path.is_file():
        return index
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            thumbnail_path = row.get("thumbnail_path")
            if case_id and thumbnail_path:
                index[case_id] = Path(str(thumbnail_path))
    return index


def _load_detection_records(pipeline_root: Path) -> list[dict[str, Any]]:
    all_detections = pipeline_root / "all_detections.json"
    if all_detections.is_file():
        payload = _read_json(all_detections)
        if not isinstance(payload, list):
            raise ValueError(f"{all_detections} must contain a list of case records")
        records = payload
    else:
        per_case = sorted(pipeline_root.glob("*/detections.json"))
        if not per_case:
            raise FileNotFoundError(
                f"No all_detections.json or per-case */detections.json files found under {pipeline_root}"
            )
        records = [_read_json(path) for path in per_case]

    out: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Detection record under {pipeline_root} is not an object: {record!r}")
        copied = dict(record)
        copied["_source_pipeline_root"] = str(pipeline_root.resolve())
        out.append(copied)
    return out


def _load_detection_records_many(pipeline_roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pipeline_root in pipeline_roots:
        for record in _load_detection_records(pipeline_root):
            key = (str(record.get("case_id") or ""), str(record.get("wsi_path") or ""))
            if key in seen:
                raise ValueError(
                    f"Duplicate detector record across input roots: case_id={key[0]!r} wsi_path={key[1]!r}"
                )
            seen.add(key)
            records.append(record)
    return records


def _load_stage1_thumbnail_indices(pipeline_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for pipeline_root in pipeline_roots:
        index.update(_load_stage1_thumbnail_index(pipeline_root))
    return index


def _case_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("case_id") or ""), str(record.get("wsi_path") or ""))


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe or "case"


def _derive_stain(record: dict[str, Any]) -> str:
    wsi_path = str(record.get("wsi_path") or "")
    if wsi_path:
        parent = Path(wsi_path).parent.name
        if parent:
            return parent
    case_id = str(record.get("case_id") or "")
    match = re.match(r"(?P<stain>.+?)_patient_\d+_slide_\d+$", case_id)
    if match:
        return match.group("stain")
    return ""


def _derive_patient_slide(record: dict[str, Any]) -> tuple[str, str]:
    joined = " ".join(
        str(record.get(key) or "")
        for key in ("case_id", "case_display", "wsi_path")
    )
    patient = ""
    slide = ""
    patient_match = re.search(r"patient[_-](?P<patient>[A-Za-z0-9]+)", joined, re.IGNORECASE)
    slide_match = re.search(r"slide[_-](?P<slide>[A-Za-z0-9]+)", joined, re.IGNORECASE)
    if patient_match:
        patient = patient_match.group("patient")
    if slide_match:
        slide = slide_match.group("slide")
    return patient, slide


def _derive_group_id(record: dict[str, Any], strategy: str) -> str:
    case_id = str(record.get("case_id") or "")
    patient, slide = _derive_patient_slide(record)
    if strategy == "case":
        return case_id
    if strategy == "patient":
        return f"patient_{patient}" if patient else case_id
    if strategy in {"auto", "patient_slide"} and patient and slide:
        return f"patient_{patient}_slide_{slide}"
    return case_id


def _parse_split_fractions(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split-fractions must be TRAIN,VAL,TEST")
    if any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("split fractions must be non-negative")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("at least one split fraction must be positive")
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def _split_group_ids(
    group_ids: list[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    unique = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)

    train_frac, val_frac, test_frac = fractions
    n_groups = len(unique)
    if n_groups == 0:
        return {}

    requested_val = val_frac > 0
    requested_test = test_frac > 0
    n_test = int(round(n_groups * test_frac))
    n_val = int(round(n_groups * val_frac))
    if requested_test and n_groups >= 3:
        n_test = max(1, n_test)
    if requested_val and n_groups >= 3:
        n_val = max(1, n_val)
    if n_val + n_test >= n_groups:
        overflow = n_val + n_test - (n_groups - 1)
        while overflow > 0 and n_test >= n_val and n_test > 0:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
    n_train = n_groups - n_val - n_test
    if train_frac > 0 and n_train == 0:
        n_train = 1
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1

    split_by_group: dict[str, str] = {}
    train_groups = unique[:n_train]
    val_groups = unique[n_train : n_train + n_val]
    test_groups = unique[n_train + n_val :]
    for group in train_groups:
        split_by_group[group] = "train"
    for group in val_groups:
        split_by_group[group] = "val"
    for group in test_groups:
        split_by_group[group] = "test"
    return split_by_group


def _resolve_thumbnail(record: dict[str, Any], pipeline_root: Path, stage1_index: dict[str, Path]) -> Path:
    case_id = str(record.get("case_id") or "")
    candidates: list[Path] = []
    path_value = ((record.get("paths") or {}).get("thumbnail_path") if isinstance(record.get("paths"), dict) else "")
    if path_value:
        candidates.append(Path(str(path_value)))
    source_pipeline_root = record.get("_source_pipeline_root")
    if source_pipeline_root and case_id:
        candidates.append(
            Path(str(source_pipeline_root))
            / case_id
            / "intermediate_stage_artifacts"
            / "stage1_thumbnail_detection"
            / "thumbnail.png"
        )
    if case_id in stage1_index:
        candidates.append(stage1_index[case_id])
    if case_id:
        candidates.append(
            pipeline_root
            / case_id
            / "intermediate_stage_artifacts"
            / "stage1_thumbnail_detection"
            / "thumbnail.png"
        )
    case_dir = record.get("case_dir")
    if case_dir:
        candidates.append(
            Path(str(case_dir))
            / "intermediate_stage_artifacts"
            / "stage1_thumbnail_detection"
            / "thumbnail.png"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a clean Stage 1 thumbnail for case_id={case_id!r}. "
        "Rerun the detector pipeline with --save-all-stage-artifacts or provide an output root that has thumbnails."
    )


def normalized_yxyx_to_pixel_xywh(
    box: Iterable[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    y_min, x_min, y_max, x_max = [float(value) for value in box]
    y_min = min(1000.0, max(0.0, y_min))
    x_min = min(1000.0, max(0.0, x_min))
    y_max = min(1000.0, max(0.0, y_max))
    x_max = min(1000.0, max(0.0, x_max))
    if y_max <= y_min or x_max <= x_min:
        raise ValueError(f"Invalid normalized box after clamping: {[y_min, x_min, y_max, x_max]}")
    x = x_min / 1000.0 * width
    y = y_min / 1000.0 * height
    w = (x_max - x_min) / 1000.0 * width
    h = (y_max - y_min) / 1000.0 * height
    return (x, y, w, h)


def normalized_yxyx_to_yolo_xywh(box: Iterable[float]) -> tuple[float, float, float, float]:
    y_min, x_min, y_max, x_max = [float(value) for value in box]
    y_min = min(1000.0, max(0.0, y_min))
    x_min = min(1000.0, max(0.0, x_min))
    y_max = min(1000.0, max(0.0, y_max))
    x_max = min(1000.0, max(0.0, x_max))
    if y_max <= y_min or x_max <= x_min:
        raise ValueError(f"Invalid normalized box after clamping: {[y_min, x_min, y_max, x_max]}")
    return (
        (x_min + x_max) / 2000.0,
        (y_min + y_max) / 2000.0,
        (x_max - x_min) / 1000.0,
        (y_max - y_min) / 1000.0,
    )


def _build_box_records(record: dict[str, Any], width: int, height: int) -> tuple[BoxRecord, ...]:
    boxes: list[BoxRecord] = []
    for detection in record.get("detections") or []:
        raw_box = detection.get("box_2d")
        if not isinstance(raw_box, list | tuple) or len(raw_box) != 4:
            raise ValueError(f"Detection for {record.get('case_id')} lacks a 4-value box_2d: {detection}")
        normalized = tuple(float(value) for value in raw_box)
        pixel = normalized_yxyx_to_pixel_xywh(normalized, width, height)
        yolo = normalized_yxyx_to_yolo_xywh(normalized)
        boxes.append(
            BoxRecord(
                normalized_yxyx=normalized,
                pixel_xywh=pixel,
                yolo_xywh=yolo,
                source_detection=detection,
            )
        )
    return tuple(boxes)


def _materialize_image(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Image already exists: {destination}")
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _format_float(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _create_cases(
    records: list[dict[str, Any]],
    pipeline_roots: list[Path],
    output_dir: Path,
    image_mode: str,
    split_by_group: dict[str, str],
    group_strategy: str,
    overwrite: bool,
) -> list[CaseRecord]:
    if not pipeline_roots:
        raise ValueError("At least one pipeline root is required")
    default_pipeline_root = pipeline_roots[0]
    stage1_index = _load_stage1_thumbnail_indices(pipeline_roots)
    cases: list[CaseRecord] = []
    used_names: set[str] = set()
    for record in sorted(records, key=_case_sort_key):
        case_id = str(record.get("case_id") or Path(str(record.get("wsi_path") or "case")).stem)
        thumbnail = _resolve_thumbnail(record, default_pipeline_root, stage1_index)
        with Image.open(thumbnail) as image:
            width, height = image.size
        stain = _derive_stain(record)
        patient_id, slide_id = _derive_patient_slide(record)
        group_id = _derive_group_id(record, group_strategy)
        split = split_by_group[group_id]
        stem = _safe_filename(case_id)
        suffix = thumbnail.suffix.lower() if thumbnail.suffix else ".png"
        filename = f"{stem}{suffix}"
        if filename in used_names:
            digest = hashlib.sha1(str(record.get("wsi_path") or case_id).encode()).hexdigest()[:8]
            filename = f"{stem}_{digest}.png"
        used_names.add(filename)
        output_image_relpath = f"images/{split}/{filename}"
        output_image_abspath = output_dir / output_image_relpath
        _materialize_image(thumbnail, output_image_abspath, image_mode, overwrite=True)
        boxes = _build_box_records(record, width, height)
        cases.append(
            CaseRecord(
                case_id=case_id,
                case_display=str(record.get("case_display") or ""),
                wsi_path=str(record.get("wsi_path") or ""),
                source_thumbnail_path=thumbnail,
                output_image_relpath=output_image_relpath,
                output_image_abspath=output_image_abspath,
                width=width,
                height=height,
                stain=stain,
                patient_id=patient_id,
                slide_id=slide_id,
                group_id=group_id,
                split=split,
                source_pipeline_record=record,
                boxes=boxes,
            )
        )
    return cases


def _case_manifest_row(case: CaseRecord) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_display": case.case_display,
        "split": case.split,
        "group_id": case.group_id,
        "stain": case.stain,
        "patient_id": case.patient_id,
        "slide_id": case.slide_id,
        "wsi_path": case.wsi_path,
        "source_thumbnail_path": str(case.source_thumbnail_path),
        "image_path": str(case.output_image_abspath),
        "image_relpath": case.output_image_relpath,
        "width": case.width,
        "height": case.height,
        "box_count": len(case.boxes),
        "pipeline_ticket": case.source_pipeline_record.get("ticket", ""),
        "pipeline_git_commit": case.source_pipeline_record.get("git_commit", ""),
        "pipeline_version": case.source_pipeline_record.get("pipeline_version", ""),
    }


def _annotation_manifest_row(case: CaseRecord, box: BoxRecord, index: int) -> dict[str, Any]:
    y_min, x_min, y_max, x_max = box.normalized_yxyx
    x, y, w, h = box.pixel_xywh
    yolo_x, yolo_y, yolo_w, yolo_h = box.yolo_xywh
    return {
        **_case_manifest_row(case),
        "detection_index": index,
        "normalized_y_min": y_min,
        "normalized_x_min": x_min,
        "normalized_y_max": y_max,
        "normalized_x_max": x_max,
        "pixel_x_min": x,
        "pixel_y_min": y,
        "pixel_width": w,
        "pixel_height": h,
        "yolo_class": 0,
        "yolo_x_center": yolo_x,
        "yolo_y_center": yolo_y,
        "yolo_width": yolo_w,
        "yolo_height": yolo_h,
        "classification_decision": box.source_detection.get("classification_decision", ""),
        "odd_one_out_flagged": box.source_detection.get("odd_one_out_flagged", ""),
        "source_candidate_order": box.source_detection.get("source_candidate_order", ""),
    }


def _write_manifests(output_dir: Path, cases: list[CaseRecord]) -> None:
    case_rows = [_case_manifest_row(case) for case in cases]
    annotation_rows = [
        _annotation_manifest_row(case, box, index)
        for case in cases
        for index, box in enumerate(case.boxes, start=1)
    ]
    case_fields = [
        "case_id",
        "case_display",
        "split",
        "group_id",
        "stain",
        "patient_id",
        "slide_id",
        "wsi_path",
        "source_thumbnail_path",
        "image_path",
        "image_relpath",
        "width",
        "height",
        "box_count",
        "pipeline_ticket",
        "pipeline_git_commit",
        "pipeline_version",
    ]
    annotation_fields = [
        *case_fields,
        "detection_index",
        "normalized_y_min",
        "normalized_x_min",
        "normalized_y_max",
        "normalized_x_max",
        "pixel_x_min",
        "pixel_y_min",
        "pixel_width",
        "pixel_height",
        "yolo_class",
        "yolo_x_center",
        "yolo_y_center",
        "yolo_width",
        "yolo_height",
        "classification_decision",
        "odd_one_out_flagged",
        "source_candidate_order",
    ]
    jsonl_rows = []
    for case in cases:
        jsonl_rows.append(
            {
                **_case_manifest_row(case),
                "coordinate_system": COORDINATE_SYSTEM,
                "annotations": [
                    {
                        "detection_index": index,
                        "box_2d": list(box.normalized_yxyx),
                        "pixel_xywh": list(box.pixel_xywh),
                        "yolo_xywh": list(box.yolo_xywh),
                        "classification_decision": box.source_detection.get("classification_decision", ""),
                        "odd_one_out_flagged": box.source_detection.get("odd_one_out_flagged", ""),
                        "source_candidate_order": box.source_detection.get("source_candidate_order", ""),
                    }
                    for index, box in enumerate(case.boxes, start=1)
                ],
            }
        )
    _write_csv(output_dir / "manifests" / "cases.csv", case_rows, case_fields)
    _write_csv(output_dir / "manifests" / "manifest.csv", annotation_rows, annotation_fields)
    _write_jsonl(output_dir / "manifests" / "manifest.jsonl", jsonl_rows)


def _coco_for_cases(cases: list[CaseRecord], split_name: str, class_name: str, ticket: str) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, case in enumerate(cases, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": case.output_image_relpath,
                "width": case.width,
                "height": case.height,
                "case_id": case.case_id,
                "case_display": case.case_display,
                "wsi_path": case.wsi_path,
                "source_thumbnail_path": str(case.source_thumbnail_path),
                "split": case.split,
                "group_id": case.group_id,
                "stain": case.stain,
                "patient_id": case.patient_id,
                "slide_id": case.slide_id,
            }
        )
        for index, box in enumerate(case.boxes, start=1):
            x, y, w, h = box.pixel_xywh
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": {
                        "case_id": case.case_id,
                        "detection_index": index,
                        "box_2d": list(box.normalized_yxyx),
                        "coordinate_system": COORDINATE_SYSTEM,
                        "classification_decision": box.source_detection.get("classification_decision", ""),
                        "odd_one_out_flagged": box.source_detection.get("odd_one_out_flagged", ""),
                        "source_candidate_order": box.source_detection.get("source_candidate_order", ""),
                    },
                }
            )
            ann_id += 1
    return {
        "info": {
            "description": "Detector pipeline thumbnail bbox export",
            "version": EXPORT_VERSION,
            "ticket": ticket,
            "split": split_name,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": class_name, "supercategory": "tissue"}],
        "images": images,
        "annotations": annotations,
    }


def _write_coco(output_dir: Path, cases: list[CaseRecord], class_name: str, ticket: str) -> None:
    by_split = {split: [case for case in cases if case.split == split] for split in ("train", "val", "test")}
    for split, split_cases in by_split.items():
        _write_json(
            output_dir / "annotations" / f"instances_{split}.json",
            _coco_for_cases(split_cases, split, class_name, ticket),
        )
    _write_json(output_dir / "annotations" / "instances_all.json", _coco_for_cases(cases, "all", class_name, ticket))


def _write_yolo(output_dir: Path, cases: list[CaseRecord], class_name: str) -> None:
    for case in cases:
        label_path = output_dir / "labels" / case.split / f"{Path(case.output_image_relpath).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "0 "
            + " ".join(_format_float(value) for value in box.yolo_xywh)
            for box in case.boxes
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_text = textwrap.dedent(
        f"""\
        path: {output_dir.resolve()}
        train: images/train
        val: images/val
        test: images/test
        names:
          0: {class_name}
        """
    )
    (output_dir / "dataset.yaml").write_text(yaml_text)


def _split_counts(cases: list[CaseRecord]) -> dict[str, int]:
    return {split: sum(1 for case in cases if case.split == split) for split in ("train", "val", "test")}


def _box_counts(cases: list[CaseRecord]) -> dict[str, int]:
    return {split: sum(len(case.boxes) for case in cases if case.split == split) for split in ("train", "val", "test")}


def _write_summary(
    output_dir: Path,
    pipeline_roots: list[Path],
    cases: list[CaseRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_roots = [str(root.resolve()) for root in pipeline_roots]
    source_jsons = [str((root / "all_detections.json").resolve()) for root in pipeline_roots]
    summary = {
        "ticket": args.ticket,
        "export_version": EXPORT_VERSION,
        "git_commit": _repo_git_commit(),
        "source_pipeline_roots": source_roots,
        "output_dir": str(output_dir.resolve()),
        "source_all_detections_json": source_jsons,
        "image_mode": args.image_mode,
        "group_strategy": args.group_by,
        "split_fractions": {
            "train": args.split_fractions[0],
            "val": args.split_fractions[1],
            "test": args.split_fractions[2],
        },
        "split_seed": args.seed,
        "case_count": len(cases),
        "box_count": sum(len(case.boxes) for case in cases),
        "group_count": len({case.group_id for case in cases}),
        "split_image_counts": _split_counts(cases),
        "split_box_counts": _box_counts(cases),
        "coordinate_system": COORDINATE_SYSTEM,
        "class_names": [args.class_name],
        "files": {
            "coco_all": str((output_dir / "annotations" / "instances_all.json").resolve()),
            "coco_train": str((output_dir / "annotations" / "instances_train.json").resolve()),
            "coco_val": str((output_dir / "annotations" / "instances_val.json").resolve()),
            "coco_test": str((output_dir / "annotations" / "instances_test.json").resolve()),
            "yolo_dataset_yaml": str((output_dir / "dataset.yaml").resolve()),
            "manifest_jsonl": str((output_dir / "manifests" / "manifest.jsonl").resolve()),
            "manifest_csv": str((output_dir / "manifests" / "manifest.csv").resolve()),
            "cases_csv": str((output_dir / "manifests" / "cases.csv").resolve()),
        },
    }
    if len(source_roots) == 1:
        summary["source_pipeline_root"] = source_roots[0]
    _write_json(output_dir / "summary.json", summary)
    return summary


def _write_reproduction(
    output_dir: Path,
    pipeline_roots: list[Path],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    source_lines = []
    for pipeline_root in pipeline_roots:
        source_json = pipeline_root / "all_detections.json"
        source_hash = _sha256(source_json) if source_json.is_file() else "missing"
        source_lines.append(
            f"- root: {pipeline_root.resolve()}\n"
            f"  all_detections_json: {source_json.resolve()}\n"
            f"  all_detections_sha256: {source_hash}"
        )
    source_text = "\n".join(source_lines)
    text = f"""\
Detector Training Dataset Export
================================

Ticket: {args.ticket}
Export version: {EXPORT_VERSION}
Git commit: {_repo_git_commit()}

Command:
{_redacted_argv(sys.argv)}

Source detector pipeline root:
{source_text}

Output directory:
{output_dir.resolve()}

Export contract:
- X: clean Stage 1 thumbnail images, materialized under images/{{train,val,test}}.
- Y canonical: COCO detection JSON over those thumbnails with pixel-space xywh boxes.
- Y derived: Ultralytics-style YOLO text labels plus dataset.yaml.
- Audit manifest: JSONL and CSV preserving original normalized 0-1000 [y_min, x_min, y_max, x_max] boxes, converted pixel boxes, YOLO boxes, source WSI path, case metadata, and split.

Split policy:
- group_by: {args.group_by}
- split_fractions: train={args.split_fractions[0]}, val={args.split_fractions[1]}, test={args.split_fractions[2]}
- seed: {args.seed}
- rationale: keep serial stains from the same patient/slide group in the same split when patient/slide ids are derivable.

Validation:
python scripts/export_detector_training_dataset.py validate {output_dir.resolve()}

Summary:
{json.dumps(summary, indent=2, sort_keys=True)}
"""
    (output_dir / "reproduction.txt").write_text(text)


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_roots = [path.resolve() for path in args.pipeline_output_roots]
    output_dir = args.output_dir.resolve()
    records = _load_detection_records_many(pipeline_roots)
    records = sorted(records, key=_case_sort_key)
    group_ids = [_derive_group_id(record, args.group_by) for record in records]
    split_by_group = _split_group_ids(group_ids, args.split_fractions, args.seed)
    _prepare_output_dir(output_dir, args.overwrite)
    cases = _create_cases(
        records,
        pipeline_roots,
        output_dir,
        args.image_mode,
        split_by_group,
        args.group_by,
        args.overwrite,
    )
    _write_coco(output_dir, cases, args.class_name, args.ticket)
    _write_yolo(output_dir, cases, args.class_name)
    _write_manifests(output_dir, cases)
    summary = _write_summary(output_dir, pipeline_roots, cases, args)
    _write_reproduction(output_dir, pipeline_roots, summary, args)
    if not args.skip_validation:
        validation = validate_dataset(output_dir)
        summary["validation"] = validation
        _write_json(output_dir / "summary.json", summary)
        _write_reproduction(output_dir, pipeline_roots, summary, args)
    return summary


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    coco_path = dataset_dir / "annotations" / "instances_all.json"
    manifest_path = dataset_dir / "manifests" / "manifest.jsonl"
    yaml_path = dataset_dir / "dataset.yaml"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations: {coco_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest JSONL: {manifest_path}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing YOLO dataset.yaml: {yaml_path}")

    coco = _read_json(coco_path)
    images = coco.get("images") or []
    annotations = coco.get("annotations") or []
    image_by_id = {image["id"]: image for image in images}
    missing_images: list[str] = []
    for image in images:
        image_path = dataset_dir / image["file_name"]
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        with Image.open(image_path) as opened:
            if opened.size != (int(image["width"]), int(image["height"])):
                raise ValueError(
                    f"Image size mismatch for {image_path}: COCO says {(image['width'], image['height'])}, file is {opened.size}"
                )
    if missing_images:
        raise FileNotFoundError(f"Missing exported images: {missing_images[:5]}")

    for ann in annotations:
        image = image_by_id.get(ann["image_id"])
        if image is None:
            raise ValueError(f"Annotation {ann.get('id')} references missing image_id={ann.get('image_id')}")
        x, y, w, h = [float(value) for value in ann["bbox"]]
        if x < -1e-6 or y < -1e-6 or w <= 0 or h <= 0:
            raise ValueError(f"Annotation {ann.get('id')} has invalid bbox {ann['bbox']}")
        if x + w > float(image["width"]) + 1e-6 or y + h > float(image["height"]) + 1e-6:
            raise ValueError(f"Annotation {ann.get('id')} exceeds image bounds: {ann['bbox']} for {image}")

    manifest_rows = 0
    with manifest_path.open() as handle:
        for line in handle:
            if line.strip():
                manifest_rows += 1

    label_count = 0
    for label_path in sorted((dataset_dir / "labels").glob("*/*.txt")):
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"YOLO label row must have 5 fields in {label_path}: {line}")
            cls = int(parts[0])
            values = [float(value) for value in parts[1:]]
            if cls != 0:
                raise ValueError(f"Unexpected class id in {label_path}: {cls}")
            if any(value < -1e-6 or value > 1.0 + 1e-6 for value in values):
                raise ValueError(f"YOLO values out of [0,1] in {label_path}: {line}")
            if values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"YOLO width/height must be positive in {label_path}: {line}")
            label_count += 1

    if label_count != len(annotations):
        raise ValueError(f"YOLO label count {label_count} does not match COCO annotation count {len(annotations)}")

    validation = {
        "dataset_dir": str(dataset_dir),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "manifest_case_rows": manifest_rows,
        "yolo_label_count": label_count,
        "status": "ok",
    }
    _write_json(dataset_dir / "validation.json", validation)
    return validation


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export detector pipeline outputs to COCO, YOLO, and audit manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export one or more detector pipeline output roots.")
    export.add_argument(
        "pipeline_output_roots",
        type=Path,
        nargs="+",
        help="Detector pipeline root(s) with all_detections.json or per-case detections.json files.",
    )
    export.add_argument("--output-dir", type=Path, required=True, help="Destination dataset directory.")
    export.add_argument("--ticket", default=DEFAULT_TICKET, help="Ticket/provenance id to record in outputs.")
    export.add_argument(
        "--image-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Materialize thumbnails by copying or symlinking them.",
    )
    export.add_argument(
        "--group-by",
        choices=("auto", "patient_slide", "patient", "case"),
        default="auto",
        help="Grouping unit for split assignment. auto uses patient+slide when derivable.",
    )
    export.add_argument(
        "--split-fractions",
        type=_parse_split_fractions,
        default=(0.8, 0.1, 0.1),
        help="Comma-separated train,val,test fractions. Default: 0.8,0.1,0.1.",
    )
    export.add_argument("--seed", type=int, default=13, help="Deterministic split seed.")
    export.add_argument("--class-name", default="tissue_candidate", help="Single detector class name.")
    export.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    export.add_argument("--skip-validation", action="store_true", help="Do not run post-export validation.")
    export.set_defaults(func=export_dataset)

    validate = subparsers.add_parser("validate", help="Validate an exported dataset directory.")
    validate.add_argument("dataset_dir", type=Path, help="Dataset directory to validate.")
    validate.set_defaults(func=lambda args: validate_dataset(args.dataset_dir))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = args.func(args)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
