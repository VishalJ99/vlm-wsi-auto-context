#!/usr/bin/env python3
# ABOUTME: Seed run_auto_context.py resume runs from scale500 detector outputs.
# ABOUTME: Materializes Stage 1 bboxes and Stage 2 cached crops for bbox-seeded foreground runs.
"""
Export scale500 detector outputs into run_auto_context.py's canonical layout.

This seeds:

  <output_root>/<wsi_id>/<run_id>/stage1/
  <output_root>/<wsi_id>/<run_id>/bboxes/<x1_y1_x2_y2>/stage2/
  <output_root>/<wsi_id>/<run_id>/bboxes/<x1_y1_x2_y2>/stage3/  (optional)

Then run:

  python run_auto_context.py \
    --wsi-list <manifest>.wsi_list.txt \
    --output-root <output_root> \
    --run-id <run_id> \
    --resume \
    --skip-stage2 \
    --stage6-icl-k 0

Stage 1 and Stage 2 will be resume hits. With the default
--seed-stage3 all-foreground, Stage 3 is also a resume hit and Stage 4+
continue normally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


ARTIFACT_KEYS: Tuple[str, ...] = (
    "ink_or_pen_marks",
    "debris",
    "labels",
    "air_bubbles",
    "cracks",
    "tissue_folds",
    "paraffin_mounting_medium",
)
ROTATION_KEYS: Tuple[str, ...] = ("0", "90", "180", "270")


class ExportError(RuntimeError):
    """Fatal scale500 export error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ExportError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def bbox_to_str(bbox: Sequence[int]) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return f"{x1}_{y1}_{x2}_{y2}"


def normalize_case_id(value: str) -> str:
    stem = Path(value).stem
    return stem.replace("-", "_")


def yxyx_norm_to_xyxy_l0(
    box_2d: Sequence[float],
    *,
    wsi_width: int,
    wsi_height: int,
) -> List[int]:
    if len(box_2d) != 4:
        raise ExportError(f"Expected normalized yxyx box with four values, got {box_2d}")
    y1n, x1n, y2n, x2n = [float(v) for v in box_2d]
    x1 = int(math.floor((x1n / 1000.0) * wsi_width))
    y1 = int(math.floor((y1n / 1000.0) * wsi_height))
    x2 = int(math.ceil((x2n / 1000.0) * wsi_width))
    y2 = int(math.ceil((y2n / 1000.0) * wsi_height))
    return clamp_xyxy([x1, y1, x2, y2], wsi_width=wsi_width, wsi_height=wsi_height)


def xyxy_l0_to_yxyx_norm(
    bbox: Sequence[int],
    *,
    wsi_width: int,
    wsi_height: int,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    vals = [
        int(round((y1 / float(wsi_height)) * 1000.0)),
        int(round((x1 / float(wsi_width)) * 1000.0)),
        int(round((y2 / float(wsi_height)) * 1000.0)),
        int(round((x2 / float(wsi_width)) * 1000.0)),
    ]
    return [max(0, min(1000, v)) for v in vals]


def xyxy_l0_to_xyxy_thumb(
    bbox: Sequence[int],
    *,
    wsi_width: int,
    wsi_height: int,
    thumb_width: int,
    thumb_height: int,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    sx = thumb_width / float(wsi_width)
    sy = thumb_height / float(wsi_height)
    out = [
        int(math.floor(x1 * sx)),
        int(math.floor(y1 * sy)),
        int(math.ceil(x2 * sx)),
        int(math.ceil(y2 * sy)),
    ]
    return clamp_xyxy(out, wsi_width=thumb_width, wsi_height=thumb_height)


def clamp_xyxy(
    bbox: Sequence[int],
    *,
    wsi_width: int,
    wsi_height: int,
) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(wsi_width, x1))
    x2 = max(0, min(wsi_width, x2))
    y1 = max(0, min(wsi_height, y1))
    y2 = max(0, min(wsi_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ExportError(f"Invalid bbox after clamping: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def build_exclude_all_verdicts() -> Dict[str, Dict[str, object]]:
    return {
        key: {
            "votes": {rot: "SD" for rot in ROTATION_KEYS},
            "counts": {"SD": len(ROTATION_KEYS), "WA": 0, "SA": 0},
            "verdict": "EXCLUDE",
        }
        for key in ARTIFACT_KEYS
    }


def build_exclude_all_strength() -> Dict[str, Dict[str, str]]:
    return {rot: {key: "SD" for key in ARTIFACT_KEYS} for rot in ROTATION_KEYS}


def apply_prefix_maps(path: str, prefix_maps: Sequence[Tuple[str, str]]) -> str:
    for src, dst in prefix_maps:
        if path == src:
            return dst
        if path.startswith(src.rstrip("/") + "/"):
            rel = path[len(src.rstrip("/")) + 1 :]
            return str(Path(dst) / rel)
    return path


def parse_prefix_maps(raw_maps: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for raw in raw_maps:
        if "=" not in raw:
            raise ExportError(f"Invalid --wsi-prefix-map {raw!r}; expected FROM=TO")
        src, dst = raw.split("=", 1)
        src = src.strip().rstrip("/")
        dst = dst.strip().rstrip("/")
        if not src or not dst:
            raise ExportError(f"Invalid --wsi-prefix-map {raw!r}; empty FROM or TO")
        parsed.append((src, dst))
    return parsed


def iter_case_dirs(scale500_run_dir: Path) -> Iterable[Path]:
    for child in sorted(scale500_run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "detections.json").exists():
            yield child


def resolve_scale500_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "detections.json").exists():
        return path.parent
    if not path.exists():
        raise ExportError(f"Scale500 path not found: {path}")
    return path


def load_case_selection(args: argparse.Namespace) -> Optional[set]:
    selected = {normalize_case_id(case) for case in args.case}
    if args.case_list:
        for raw in args.case_list.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            selected.add(normalize_case_id(line.split(",")[0].strip()))
    return selected or None


def normalize_box_id_list(value: Any, *, field: str, case_id: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = [part.strip() for part in stripped.split(",") if part.strip()]
    if not isinstance(value, list):
        raise ExportError(f"Selection field {field!r} for {case_id} is not a list")
    out: List[int] = []
    for item in value:
        try:
            box_id = int(item)
        except Exception as exc:
            raise ExportError(f"Selection field {field!r} for {case_id} contains {item!r}") from exc
        if box_id < 1:
            raise ExportError(f"Selection field {field!r} for {case_id} contains non-positive id {box_id}")
        if box_id not in out:
            out.append(box_id)
    return sorted(out)


def choose_selection_for_row(row: Dict[str, Any], policy: str) -> Tuple[List[int], str]:
    case_id = normalize_case_id(str(row.get("case_id") or ""))
    if not case_id:
        raise ExportError(f"Selection row missing case_id: {row}")
    baseline = normalize_box_id_list(
        row.get("baseline_selected_box_ids"),
        field="baseline_selected_box_ids",
        case_id=case_id,
    )
    direct = normalize_box_id_list(
        row.get("direct_selected_box_ids"),
        field="direct_selected_box_ids",
        case_id=case_id,
    )
    verifier = normalize_box_id_list(
        row.get("verifier_selected_box_ids"),
        field="verifier_selected_box_ids",
        case_id=case_id,
    )
    verifier_added = normalize_box_id_list(
        row.get("verifier_added_vs_baseline_box_ids"),
        field="verifier_added_vs_baseline_box_ids",
        case_id=case_id,
    )
    verifier_dropped = normalize_box_id_list(
        row.get("verifier_dropped_vs_baseline_box_ids"),
        field="verifier_dropped_vs_baseline_box_ids",
        case_id=case_id,
    )
    verifier_ok = str(row.get("verifier_parse_status") or "").strip().lower() == "ok"

    if policy == "baseline":
        return baseline, "baseline_selected_box_ids"
    if policy == "direct":
        return direct, "direct_selected_box_ids"
    if policy == "verifier":
        if verifier_ok and verifier:
            return verifier, "verifier_selected_box_ids"
        raise ExportError(
            f"Selection row for {case_id} has no strict verifier selection "
            f"(parse_status={row.get('verifier_parse_status')!r}, ids={verifier})"
        )
    if policy == "verifier-or-baseline":
        if verifier_ok and verifier:
            return verifier, "verifier_selected_box_ids"
        return baseline, "baseline_fallback_verifier_not_ok_or_empty"
    if policy == "conservative-verifier-drop-only":
        if verifier_ok and verifier and verifier_dropped and not verifier_added:
            return verifier, "verifier_drop_only_revision"
        return baseline, "baseline_fallback_not_drop_only"
    raise ExportError(f"Unknown selection policy: {policy}")


def load_selection_map(selection_jsonl: Optional[Path], policy: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if selection_jsonl is None:
        return None
    path = selection_jsonl.expanduser().resolve()
    if not path.exists():
        raise ExportError(f"Selection JSONL not found: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ExportError(f"Selection row {line_no} in {path} is not an object")
            case_id = normalize_case_id(str(row.get("case_id") or ""))
            if not case_id:
                raise ExportError(f"Selection row {line_no} in {path} missing case_id")
            selected, source = choose_selection_for_row(row, policy)
            if not selected:
                raise ExportError(f"Selection row for {case_id} produced no selected boxes")
            out[case_id] = {
                "case_id": case_id,
                "selected_box_ids": selected,
                "selection_policy": policy,
                "selection_source": source,
                "selection_jsonl": str(path),
                "baseline_selected_box_ids": normalize_box_id_list(
                    row.get("baseline_selected_box_ids"),
                    field="baseline_selected_box_ids",
                    case_id=case_id,
                ),
                "direct_selected_box_ids": normalize_box_id_list(
                    row.get("direct_selected_box_ids"),
                    field="direct_selected_box_ids",
                    case_id=case_id,
                ),
                "verifier_selected_box_ids": normalize_box_id_list(
                    row.get("verifier_selected_box_ids"),
                    field="verifier_selected_box_ids",
                    case_id=case_id,
                ),
                "verifier_added_vs_baseline_box_ids": normalize_box_id_list(
                    row.get("verifier_added_vs_baseline_box_ids"),
                    field="verifier_added_vs_baseline_box_ids",
                    case_id=case_id,
                ),
                "verifier_dropped_vs_baseline_box_ids": normalize_box_id_list(
                    row.get("verifier_dropped_vs_baseline_box_ids"),
                    field="verifier_dropped_vs_baseline_box_ids",
                    case_id=case_id,
                ),
                "direct_parse_status": row.get("direct_parse_status"),
                "verifier_parse_status": row.get("verifier_parse_status"),
                "verifier_confidence": row.get("verifier_confidence"),
                "verifier_pairs": row.get("verifier_pairs"),
            }
    if not out:
        raise ExportError(f"No selection rows loaded from {path}")
    return out


def resolve_path_maybe_relative(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def case_dir_from_case_input(case_input_root: Path, case_id: str) -> Optional[Path]:
    case_input_path = case_input_root / case_id / "case_input.json"
    if not case_input_path.exists():
        case_input_path = case_input_root / "cases" / case_id / "case_input.json"
    if not case_input_path.exists():
        return None
    payload = load_json(case_input_path)
    detections_path = payload.get("detections_path")
    if not isinstance(detections_path, str) or not detections_path:
        raise ExportError(f"Missing detections_path in {case_input_path}")
    resolved = resolve_path_maybe_relative(detections_path)
    if not resolved.exists():
        raise ExportError(f"detections_path from {case_input_path} not found: {resolved}")
    return resolved.parent


def find_case_dirs_recursive(source_root: Path, wanted_case_ids: set[str]) -> List[Path]:
    found: Dict[str, Path] = {}
    for detections_path in sorted(source_root.rglob("detections.json")):
        if "analysis" in detections_path.parts:
            continue
        case_id = normalize_case_id(detections_path.parent.name)
        if case_id in wanted_case_ids and case_id not in found:
            found[case_id] = detections_path.parent
    return [found[case_id] for case_id in sorted(wanted_case_ids) if case_id in found]


def candidate_dir_for_detection(case_dir: Path, detection: Dict, index: int) -> Optional[Path]:
    candidates_root = (
        case_dir
        / "intermediate_stage_artifacts"
        / "stage5_post_redetect_merge_and_crop"
        / "candidates"
    )
    if not candidates_root.exists():
        return None
    order = detection.get("source_candidate_order")
    prefixes: List[str] = []
    if isinstance(order, int):
        prefixes.append(f"{order:02d}_")
    prefixes.append(f"{index + 1:02d}_")
    for prefix in prefixes:
        matches = sorted(p for p in candidates_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
        if matches:
            return matches[0]
    return None


def load_candidate_metadata(candidate_dir: Optional[Path]) -> Tuple[Optional[Dict], Optional[Path]]:
    if candidate_dir is None:
        return None, None
    meta_path = candidate_dir / "metadata.json"
    if not meta_path.exists():
        return None, candidate_dir
    return load_json(meta_path), candidate_dir


def bbox_from_candidate_or_detection(
    *,
    candidate_meta: Optional[Dict],
    detection: Dict,
    wsi_width: int,
    wsi_height: int,
) -> List[int]:
    if candidate_meta:
        read_info = candidate_meta.get("candidate", {}).get("read_info", {})
        bbox = read_info.get("source_bbox_level0")
        if isinstance(bbox, list) and len(bbox) == 4:
            return clamp_xyxy(bbox, wsi_width=wsi_width, wsi_height=wsi_height)
    box_2d = detection.get("box_2d")
    if not isinstance(box_2d, list):
        raise ExportError(f"Detection missing box_2d: {detection}")
    return yxyx_norm_to_xyxy_l0(box_2d, wsi_width=wsi_width, wsi_height=wsi_height)


def save_stage2_crop(
    *,
    dst_png: Path,
    candidate_meta: Optional[Dict],
    candidate_dir: Optional[Path],
    thumbnail_path: Path,
    bbox_thumb: Sequence[int],
    crop_mode: str,
) -> Dict[str, object]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    crop_path: Optional[Path] = None
    crop_candidates: List[Path] = []
    read_info: Dict = {}
    if candidate_meta:
        candidate = candidate_meta.get("candidate", {})
        read_info = candidate.get("read_info", {}) if isinstance(candidate.get("read_info"), dict) else {}
        raw_crop_path = candidate.get("crop_path")
        if isinstance(raw_crop_path, str) and raw_crop_path:
            crop_candidates.append(Path(raw_crop_path))
        if candidate_dir:
            crop_candidates.append(candidate_dir / "crop.png")
    for candidate_crop_path in crop_candidates:
        if candidate_crop_path.exists():
            crop_path = candidate_crop_path
            break
    if crop_path is not None:
        img = Image.open(crop_path).convert("RGB")
        crop_source = str(crop_path)
        saved_mode = crop_mode
        if crop_mode == "source-bbox":
            source_bbox_in_crop = read_info.get("source_bbox_in_crop")
            if isinstance(source_bbox_in_crop, list) and len(source_bbox_in_crop) == 4:
                cx1, cy1, cx2, cy2 = [int(round(float(v))) for v in source_bbox_in_crop]
                cx1 = max(0, min(img.width - 1, cx1))
                cy1 = max(0, min(img.height - 1, cy1))
                cx2 = max(cx1 + 1, min(img.width, cx2))
                cy2 = max(cy1 + 1, min(img.height, cy2))
                img = img.crop((cx1, cy1, cx2, cy2))
            else:
                saved_mode = "full-context-fallback-no-source-bbox-in-crop"
        img.save(dst_png)
        return {
            "bbox_region_source": "scale500_candidate_crop",
            "source_crop_path": crop_source,
            "crop_mode": saved_mode,
            "saved_size": [img.width, img.height],
        }

    thumb = Image.open(thumbnail_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox_thumb]
    img = thumb.crop((x1, y1, x2, y2))
    img.save(dst_png)
    return {
        "bbox_region_source": "stage1_thumbnail_fallback",
        "source_crop_path": str(thumbnail_path),
        "crop_mode": "thumbnail-fallback",
        "saved_size": [img.width, img.height],
    }


def save_all_foreground_stage3(
    *,
    stage3_dir: Path,
    stage2_crop_path: Path,
    bbox_l0: Sequence[int],
    crop_info: Dict[str, object],
) -> Dict[str, object]:
    stage3_dir.mkdir(parents=True, exist_ok=True)
    crop = Image.open(stage2_crop_path).convert("RGB")
    crop.save(stage3_dir / "crop.png")

    mask = Image.new("L", crop.size, color=255)
    mask.save(stage3_dir / "mask.png")

    orange = Image.new("RGB", crop.size, color=(255, 140, 0))
    overlay = Image.blend(crop, orange, alpha=0.35)
    overlay.save(stage3_dir / "overlay.png")

    x1, y1, x2, y2 = [int(v) for v in bbox_l0]
    level0_w = max(1, x2 - x1)
    level0_h = max(1, y2 - y1)
    metadata = {
        "bbox_level0": [x1, y1, x2, y2],
        "crop_dims": {"width": crop.width, "height": crop.height},
        "level0_dims": {"width": level0_w, "height": level0_h},
        "scale_factor": {
            "x": level0_w / crop.width if crop.width > 0 else 1.0,
            "y": level0_h / crop.height if crop.height > 0 else 1.0,
        },
        "method": "scale500_adapter_all_foreground",
        "foreground_pixel_ratio": 1.0,
        "mask_policy": "all_pixels_foreground",
        "source_stage2_crop": str(stage2_crop_path),
        "source_stage2_crop_info": crop_info,
        "created_at": utc_now(),
    }
    write_json(stage3_dir / "metadata.json", metadata)
    return {
        "stage3_dir": str(stage3_dir),
        "seed_stage3_mode": "all-foreground",
        "crop_size": [crop.width, crop.height],
    }


def export_case(
    *,
    case_dir: Path,
    output_root: Path,
    run_id: str,
    crop_mode: str,
    prefix_maps: Sequence[Tuple[str, str]],
    seed_stage3: str,
    overwrite: bool,
    dry_run: bool,
    selection_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    detections_path = case_dir / "detections.json"
    detections_payload = load_json(detections_path)
    raw_detections = detections_payload.get("detections")
    if not isinstance(raw_detections, list) or not raw_detections:
        raise ExportError(f"No final detections found in {detections_path}")
    source_detection_count = len(raw_detections)
    selected_detection_ids: Optional[set[int]] = None
    selection_metadata: Optional[Dict[str, Any]] = None
    if selection_entry is not None:
        selected_box_ids = normalize_box_id_list(
            selection_entry.get("selected_box_ids"),
            field="selected_box_ids",
            case_id=case_dir.name,
        )
        invalid_ids = [idx for idx in selected_box_ids if idx > source_detection_count]
        if invalid_ids:
            raise ExportError(
                f"Selection for {case_dir.name} references ids outside 1..{source_detection_count}: "
                f"{invalid_ids}"
            )
        selected_detection_ids = set(selected_box_ids)
        selection_metadata = {
            key: value
            for key, value in selection_entry.items()
            if key
            in {
                "selection_policy",
                "selection_source",
                "selection_jsonl",
                "selected_box_ids",
                "baseline_selected_box_ids",
                "direct_selected_box_ids",
                "verifier_selected_box_ids",
                "verifier_added_vs_baseline_box_ids",
                "verifier_dropped_vs_baseline_box_ids",
                "direct_parse_status",
                "verifier_parse_status",
                "verifier_confidence",
                "verifier_pairs",
            }
        }
        selection_metadata["source_detection_count"] = source_detection_count

    stage_artifacts_dir = case_dir / "intermediate_stage_artifacts"
    stage1_source_dir = stage_artifacts_dir / "stage1_thumbnail_detection"
    stage1_source_meta_path = stage1_source_dir / "metadata.json"
    thumbnail_src = stage1_source_dir / "thumbnail.png"
    if not stage1_source_meta_path.exists() or not thumbnail_src.exists():
        raise ExportError(f"Missing scale500 Stage 1 metadata/thumbnail under {stage1_source_dir}")

    stage1_source_meta = load_json(stage1_source_meta_path)
    wsi_path_original = str(
        detections_payload.get("wsi_path")
        or stage1_source_meta.get("wsi_path")
        or ""
    )
    if not wsi_path_original:
        raise ExportError(f"Could not resolve WSI path for {case_dir}")
    wsi_path = apply_prefix_maps(wsi_path_original, prefix_maps)

    wsi_dims = stage1_source_meta.get("wsi_dimensions") or {}
    wsi_width = int(wsi_dims.get("width") or 0)
    wsi_height = int(wsi_dims.get("height") or 0)
    if wsi_width <= 0 or wsi_height <= 0:
        stage4_summary = stage_artifacts_dir / "stage4_high_res_crop_redetect" / "stage4_high_res_crop_redetect.json"
        if stage4_summary.exists():
            stage4_payload = load_json(stage4_summary)
            wsi_size = stage4_payload.get("wsi_size")
            if isinstance(wsi_size, list) and len(wsi_size) == 2:
                wsi_width, wsi_height = [int(v) for v in wsi_size]
    if wsi_width <= 0 or wsi_height <= 0:
        raise ExportError(f"Could not resolve WSI dimensions for {case_dir}")

    with Image.open(thumbnail_src) as thumb:
        thumb_width, thumb_height = thumb.size

    wsi_id = Path(wsi_path_original).stem
    run_dir = output_root / wsi_id / run_id
    if run_dir.exists() and not overwrite:
        raise ExportError(f"Run dir already exists, pass --overwrite to replace: {run_dir}")
    if dry_run:
        return {
            "case_id": case_dir.name,
            "wsi_id": wsi_id,
            "wsi_path": wsi_path,
            "run_dir": str(run_dir),
            "bbox_count": (
                len(selected_detection_ids) if selected_detection_ids is not None else len(raw_detections)
            ),
            "source_detection_count": source_detection_count,
            "selection_policy": selection_metadata.get("selection_policy") if selection_metadata else None,
            "selected_box_ids": selection_metadata.get("selected_box_ids") if selection_metadata else None,
            "dry_run": True,
        }
    if run_dir.exists():
        shutil.rmtree(run_dir)

    stage1_dir = run_dir / "stage1"
    bboxes_root = run_dir / "bboxes"
    logs_dir = run_dir / "logs"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    bboxes_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(thumbnail_src, stage1_dir / "thumbnail.png")

    detected_regions: List[Dict[str, object]] = []
    stage2_exports: List[Dict[str, object]] = []
    stage3_exports: List[Dict[str, object]] = []
    seen_bbox_keys = set()
    for idx, detection in enumerate(raw_detections):
        source_detection_id = idx + 1
        if selected_detection_ids is not None and source_detection_id not in selected_detection_ids:
            continue
        if not isinstance(detection, dict):
            continue
        candidate_dir = candidate_dir_for_detection(case_dir, detection, idx)
        candidate_meta, candidate_dir = load_candidate_metadata(candidate_dir)
        bbox_l0 = bbox_from_candidate_or_detection(
            candidate_meta=candidate_meta,
            detection=detection,
            wsi_width=wsi_width,
            wsi_height=wsi_height,
        )
        bbox_key = bbox_to_str(bbox_l0)
        if bbox_key in seen_bbox_keys:
            continue
        seen_bbox_keys.add(bbox_key)

        bbox_thumb = xyxy_l0_to_xyxy_thumb(
            bbox_l0,
            wsi_width=wsi_width,
            wsi_height=wsi_height,
            thumb_width=thumb_width,
            thumb_height=thumb_height,
        )
        bbox_norm_yxyx = xyxy_l0_to_yxyx_norm(
            bbox_l0,
            wsi_width=wsi_width,
            wsi_height=wsi_height,
        )

        region = {
            "label": f"scale500_tissue_{len(detected_regions) + 1}",
            "bbox_level0": bbox_l0,
            "bbox_thumbnail": bbox_thumb,
            "bbox_normalized": [
                bbox_norm_yxyx[1],
                bbox_norm_yxyx[0],
                bbox_norm_yxyx[3],
                bbox_norm_yxyx[2],
            ],
            "bbox_normalized_yxyx": bbox_norm_yxyx,
            "source": "scale500_detector_final_detection",
            "source_case_dir": str(case_dir),
            "source_detection_id": source_detection_id,
            "source_candidate_order": detection.get("source_candidate_order"),
            "source_detection": detection,
        }
        detected_regions.append(region)

        stage2_dir = bboxes_root / bbox_key / "stage2"
        crop_info = save_stage2_crop(
            dst_png=stage2_dir / "bbox_region.png",
            candidate_meta=candidate_meta,
            candidate_dir=candidate_dir,
            thumbnail_path=thumbnail_src,
            bbox_thumb=bbox_thumb,
            crop_mode=crop_mode,
        )
        stage1_perception = {
            rot: (
                "SKIPPED: scale500 detector adapter seeded Stage 2 for "
                "run_auto_context.py resume."
            )
            for rot in ROTATION_KEYS
        }
        write_json(stage2_dir / "stage1_artifact_perception.json", stage1_perception)
        write_json(stage2_dir / "stage2_claim_evidence.json", {})
        write_json(stage2_dir / "stage3_strength.json", build_exclude_all_strength())
        verdicts = build_exclude_all_verdicts()
        write_json(stage2_dir / "stage4_verdicts.json", verdicts)
        write_json(stage2_dir / "verdicts.json", verdicts)
        stage2_meta = {
            "wsi_path": wsi_path,
            "wsi_path_original": wsi_path_original,
            "bbox_level0": bbox_l0,
            "stage2_mode": "scale500_detector_adapter",
            "artifact_verdict_policy": "exclude_all",
            "source_case_dir": str(case_dir),
            "source_detections_json": str(detections_path),
            "source_candidate_dir": str(candidate_dir) if candidate_dir else None,
            "source_detection_id": source_detection_id,
            "crop_export": crop_info,
            "created_at": utc_now(),
        }
        write_json(stage2_dir / "metadata.json", stage2_meta)
        stage2_exports.append(
            {
                "bbox": bbox_l0,
                "bbox_key": bbox_key,
                "stage2_dir": str(stage2_dir),
                **crop_info,
            }
        )
        if seed_stage3 == "all-foreground":
            stage3_info = save_all_foreground_stage3(
                stage3_dir=bboxes_root / bbox_key / "stage3",
                stage2_crop_path=stage2_dir / "bbox_region.png",
                bbox_l0=bbox_l0,
                crop_info=crop_info,
            )
            stage3_exports.append({"bbox": bbox_l0, "bbox_key": bbox_key, **stage3_info})

    if not detected_regions:
        raise ExportError(f"No usable detections exported from {detections_path}")

    stage1_meta = dict(stage1_source_meta)
    stage1_meta.update(
        {
            "wsi_path": wsi_path,
            "wsi_path_original": wsi_path_original,
            "wsi_dimensions": {"width": wsi_width, "height": wsi_height},
            "thumbnail_dimensions": {"width": thumb_width, "height": thumb_height},
            "model": "scale500_detector_adapter",
            "backend": "scale500_detector_adapter",
            "detected_regions": detected_regions,
            "regions_count": len(detected_regions),
            "adapter_source": {
                "source": "scale500_detector_final_detections",
                "source_case_dir": str(case_dir),
                "source_detections_json": str(detections_path),
                "source_stage1_metadata": str(stage1_source_meta_path),
                "crop_mode": crop_mode,
                "source_detection_count": source_detection_count,
                "selection": selection_metadata,
                "created_at": utc_now(),
            },
        }
    )
    write_json(stage1_dir / "metadata.json", stage1_meta)
    write_json(
        stage1_dir / "bboxes.json",
        {
            "detected_regions": detected_regions,
            "regions_count": len(detected_regions),
            "source": (
                "scale500_detector_selector_filtered_detections"
                if selection_metadata
                else "scale500_detector_final_detections"
            ),
            "source_detections_json": str(detections_path),
            "source_detection_count": source_detection_count,
            "selection": selection_metadata,
            "created_at": utc_now(),
        },
    )

    pipeline_metadata = {
        "wsi_input": wsi_path,
        "wsi_path": wsi_path,
        "wsi_path_original": wsi_path_original,
        "wsi_id": wsi_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": {
            "source_scale500_case_dir": str(case_dir),
            "crop_mode": crop_mode,
            "wsi_prefix_maps": list(prefix_maps),
            "selection": selection_metadata,
        },
        "stage_runs": {
            "stage1_mode": "scale500_detector_adapter",
            "stage2_mode": "scale500_detector_adapter",
            "stage3_mode": (
                "scale500_adapter_all_foreground" if seed_stage3 == "all-foreground" else "not_seeded"
            ),
            "resume_next_expected": "stage4" if seed_stage3 == "all-foreground" else "stage3",
        },
        "bboxes": [r["bbox_level0"] for r in detected_regions],
        "source_detection_count": source_detection_count,
        "selected_box_ids": selection_metadata.get("selected_box_ids") if selection_metadata else None,
        "stage2_exports": stage2_exports,
        "stage3_exports": stage3_exports,
        "created_at": utc_now(),
    }
    write_json(run_dir / "pipeline_metadata.json", pipeline_metadata)
    write_json(
        run_dir / "pipeline_status.json",
        {
            "ok": False,
            "adapter_seeded": True,
            "resume_next_expected": "stage4" if seed_stage3 == "all-foreground" else "stage3",
            "wsi_path": wsi_path,
            "wsi_id": wsi_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "bbox_count": len(detected_regions),
            "source_detection_count": source_detection_count,
            "selection_policy": selection_metadata.get("selection_policy") if selection_metadata else None,
            "selected_box_ids": selection_metadata.get("selected_box_ids") if selection_metadata else None,
            "created_at": utc_now(),
        },
    )
    (run_dir / "reproduction.txt").write_text(
        "\n".join(
            [
                "Scale500 detector to run_auto_context adapter export",
                f"created_at={utc_now()}",
                f"source_case_dir={case_dir}",
                f"source_detections_json={detections_path}",
                f"output_run_dir={run_dir}",
                f"crop_mode={crop_mode}",
                f"seed_stage3={seed_stage3}",
                f"source_detection_count={source_detection_count}",
                f"selection_policy={selection_metadata.get('selection_policy') if selection_metadata else 'none'}",
                f"selection_source={selection_metadata.get('selection_source') if selection_metadata else 'none'}",
                f"selected_box_ids={selection_metadata.get('selected_box_ids') if selection_metadata else 'all'}",
                "",
                "Resume command pattern:",
                "python run_auto_context.py \\",
                f"  --wsi {wsi_path} \\",
                f"  --output-root {output_root} \\",
                f"  --run-id {run_id} \\",
                "  --resume \\",
                "  --skip-stage2 \\",
                "  --stage6-icl-k 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "case_id": case_dir.name,
        "wsi_id": wsi_id,
        "wsi_path": wsi_path,
        "wsi_path_original": wsi_path_original,
        "run_dir": str(run_dir),
        "bbox_count": len(detected_regions),
        "source_detection_count": source_detection_count,
        "selection_policy": selection_metadata.get("selection_policy") if selection_metadata else None,
        "selection_source": selection_metadata.get("selection_source") if selection_metadata else None,
        "selected_box_ids": selection_metadata.get("selected_box_ids") if selection_metadata else None,
        "stage2_crop_modes": sorted({str(item["crop_mode"]) for item in stage2_exports}),
        "seed_stage3": seed_stage3,
        "dry_run": False,
    }


def write_manifest_files(output_root: Path, run_id: str, rows: Sequence[Dict[str, object]]) -> Dict[str, str]:
    manifest_dir = output_root / "_scale500_adapter_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = manifest_dir / f"{run_id}.jsonl"
    csv_path = manifest_dir / f"{run_id}.csv"
    wsi_list_path = manifest_dir / f"{run_id}.wsi_list.txt"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    fieldnames = [
        "case_id",
        "wsi_id",
        "wsi_path",
        "wsi_path_original",
        "run_dir",
        "bbox_count",
        "source_detection_count",
        "selection_policy",
        "selection_source",
        "selected_box_ids",
        "stage2_crop_modes",
        "seed_stage3",
        "dry_run",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            if isinstance(row_out.get("stage2_crop_modes"), list):
                row_out["stage2_crop_modes"] = ";".join(str(v) for v in row_out["stage2_crop_modes"])
            if isinstance(row_out.get("selected_box_ids"), list):
                row_out["selected_box_ids"] = json.dumps(row_out["selected_box_ids"])
            writer.writerow(row_out)
    with wsi_list_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["wsi_path"]) + "\n")
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "wsi_list": str(wsi_list_path),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed run_auto_context.py resume outputs from scale500 detector cases.",
    )
    parser.add_argument(
        "--scale500-run-dir",
        type=Path,
        required=True,
        help="Scale500 detector batch root or a single case dir containing detections.json.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID/stem to export. Can be repeated. Defaults to all cases.",
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        default=None,
        help="Optional text/CSV file whose first column is a case ID/stem.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max cases to export after filtering.")
    parser.add_argument(
        "--selection-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional artifact-redundancy/selector results JSONL. When provided, only "
            "selected box IDs for each case are exported as Stage 1 bboxes."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=[
            "baseline",
            "direct",
            "verifier",
            "verifier-or-baseline",
            "conservative-verifier-drop-only",
        ],
        default="verifier",
        help=(
            "Which selection ids to export from --selection-jsonl. The verifier policy "
            "requires verifier_selected_box_ids with parse_status ok. Use "
            "verifier-or-baseline for the previous fallback behavior."
        ),
    )
    parser.add_argument(
        "--selection-case-input-root",
        type=Path,
        default=None,
        help=(
            "Optional selection probe cases directory or its parent. It must contain "
            "<case_id>/case_input.json or cases/<case_id>/case_input.json and resolves "
            "mixed scale500 subroots exactly."
        ),
    )
    parser.add_argument(
        "--stage2-crop-mode",
        choices=["source-bbox", "full-context"],
        default="source-bbox",
        help=(
            "source-bbox crops the saved scale500 high-res candidate crop down to "
            "source_bbox_in_crop so Stage 4 coordinates align with the bbox."
        ),
    )
    parser.add_argument(
        "--seed-stage3",
        choices=["all-foreground", "none"],
        default="all-foreground",
        help=(
            "Default all-foreground writes Stage 3 crop/mask/metadata from the "
            "scale500 high-res crop so resume skips thumbnail KMeans and Stage 6 "
            "gates over the full seeded bbox."
        ),
    )
    parser.add_argument(
        "--wsi-prefix-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Rewrite WSI paths in seeded metadata/worklist, useful for Alex mirrors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    prefix_maps = parse_prefix_maps(args.wsi_prefix_map)
    source_root = resolve_scale500_root(args.scale500_run_dir)
    output_root = args.output_root.expanduser().resolve()
    selected_cases = load_case_selection(args)
    selection_map = load_selection_map(args.selection_jsonl, args.selection_policy)

    case_dirs: List[Path]
    if selection_map is not None and args.selection_case_input_root is not None:
        case_input_root = args.selection_case_input_root.expanduser()
        if not case_input_root.is_absolute():
            case_input_root = (Path.cwd() / case_input_root).resolve()
        wanted = set(selection_map)
        if selected_cases is not None:
            wanted &= selected_cases
        case_dirs = []
        missing_inputs: List[str] = []
        for case_id in sorted(wanted):
            case_dir = case_dir_from_case_input(case_input_root, case_id)
            if case_dir is None:
                missing_inputs.append(case_id)
            else:
                case_dirs.append(case_dir)
        if missing_inputs:
            raise ExportError(
                f"Missing case_input.json for selected cases under {case_input_root}: {missing_inputs}"
            )
    elif selection_map is not None:
        wanted = set(selection_map)
        if selected_cases is not None:
            wanted &= selected_cases
        case_dirs = find_case_dirs_recursive(source_root, wanted)
    else:
        case_dirs = list(iter_case_dirs(source_root))
    if selected_cases is not None:
        case_dirs = [p for p in case_dirs if normalize_case_id(p.name) in selected_cases]
    if selection_map is not None:
        case_dirs = [p for p in case_dirs if normalize_case_id(p.name) in selection_map]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        case_dirs = case_dirs[: args.limit]
    if not case_dirs:
        raise ExportError(f"No matching scale500 case dirs found under {source_root}")

    rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []
    for case_dir in case_dirs:
        try:
            row = export_case(
                case_dir=case_dir,
                output_root=output_root,
                run_id=args.run_id,
                crop_mode=args.stage2_crop_mode,
                prefix_maps=prefix_maps,
                seed_stage3=args.seed_stage3,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                selection_entry=selection_map.get(normalize_case_id(case_dir.name)) if selection_map else None,
            )
            rows.append(row)
            print(f"exported {row['case_id']}: {row['bbox_count']} bbox(es) -> {row['run_dir']}")
        except Exception as exc:
            failures.append((case_dir.name, str(exc)))
            print(f"failed {case_dir.name}: {exc}")

    if rows and not args.dry_run:
        manifest_paths = write_manifest_files(output_root, args.run_id, rows)
        print("manifest paths:")
        for key, value in manifest_paths.items():
            print(f"  {key}: {value}")

    if failures:
        print("failures:")
        for case_id, error in failures:
            print(f"  {case_id}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
