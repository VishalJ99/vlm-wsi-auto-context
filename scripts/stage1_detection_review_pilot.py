#!/usr/bin/env python3
"""Run a focused VLM review of Stage 1 foreground detections."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = REPO_ROOT / "runs" / "stage1_detector_pilot_v1"
DEFAULT_MANIFEST = DEFAULT_PILOT_ROOT / "review_packet" / "all_detections_manifest.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_PILOT_ROOT / "stage1_detection_review_v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_KNOWN_CASE_INDICES = (20, 45, 66, 70, 74, 78, 92, 96, 100)

KNOWN_CASE_NOTES = {
    20: "SV40 control-tissue example; useful as a clean control-tissue reference.",
    45: "Known failure mode: synthetic/full-slide fallback or near-full-thumbnail bbox.",
    66: "Known review target: large/loose bbox or multiple cores may be present.",
    70: "Known review target: missed tissue core.",
    74: "Known review target: crystalline artifact / false-positive artifact risk.",
    78: "Known review target: bbox may need splitting into multiple cores.",
    92: "Known review target: left-hand bbox contains two cores and may need splitting.",
    96: "Known review target: possible air bubble / artifact in or near detections.",
    100: "Known failure mode: tissue cores missed and noise detected.",
}


PROMPT_VERSION = "stage1_detection_review_v2_blind_2026-05-15"
FEEDBACK_REDETECT_PROMPT_VERSION = "stage1_feedback_redetect_v1_2026-05-15"
RAW_OVERLAY_REVIEW_PROMPT_VERSION = "stage1_raw_overlay_box_review_v1_2026-05-23"

DETECTION_REVIEW_PROMPT = """\
You are auditing object-detection bounding boxes for tissue-core foreground regions on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the same thumbnail with Stage 1 detection overlays and labels such as tissue_1, tissue_2.
- A text list gives the detected bboxes and their geometry.

Review at two levels:
1. Per bbox: decide whether each detection is usable and localized well.
2. Whole thumbnail: decide whether the detection set is safe to continue or needs refinement/rerun.

Important terminology:
- acceptable: bbox tightly encloses one intended tissue core/region with reasonable margin.
- too_loose: bbox contains the intended tissue but has excessive irrelevant background, glass, artifact, or neighboring tissue. This is poor localization / over-coverage / low box tightness.
- too_tight: bbox cuts off tissue or incompletely encloses the visible tissue.
- merged_multiple_cores: one bbox contains multiple separate tissue cores that should likely be separate detections.
- false_positive: bbox is mainly artifact, noise, glass mark, bubble, debris, or non-tissue.
- near-full-thumbnail box: severe/limit case of too_loose localization where the bbox covers most or all of the thumbnail. Mark localization_quality as too_loose and also set is_near_full_thumbnail_box true.
- Do not call a thin edge band a near-full-thumbnail box just because it touches several edges; if it is mainly an edge artifact/smudge, grade it as false_positive.
- Treat crystalline material, pen marks, air bubbles, mounting-media smudges, dust, and glass-edge marks as false positives unless the bbox clearly contains tissue architecture.
- Treat visible tissue-like cores or fragments outside all bboxes as missed_tissue_core. Do not dismiss faint tissue fragments as artifacts unless they lack tissue color/structure.

Use excess_background as the severity of loose localization:
- none: no meaningful extra background.
- mild: slightly loose but likely usable.
- moderate: too much background; refinement would help.
- severe: dominated by irrelevant background or near-full-thumbnail fallback.

Return only one JSON object with this exact shape:
{
  "slide_review": {
    "overall_pass": true,
    "missed_tissue_core": false,
    "all_significant_cores_missed": false,
    "has_false_positive_artifact": false,
    "has_full_thumbnail_or_degenerate_bbox": false,
    "needs_refinement": false,
    "priority": "low",
    "reasoning": "short explanation"
  },
  "bbox_reviews": [
    {
      "bbox_id": "tissue_1",
      "localization_quality": "acceptable",
      "excess_background": "none",
      "is_near_full_thumbnail_box": false,
      "cuts_off_tissue": false,
      "multiple_cores_in_bbox": false,
      "artifact_false_positive": false,
      "suggested_action": "accept",
      "reasoning": "short explanation"
    }
  ]
}

Allowed localization_quality values: acceptable, too_loose, too_tight, merged_multiple_cores, false_positive, uncertain.
Allowed excess_background values: none, mild, moderate, severe.
Allowed suggested_action values: accept, refine_tighter, expand, split, discard_artifact, rerun_detector.
Allowed priority values: low, medium, high.

Every detected bbox from the text list must appear exactly once in bbox_reviews.
Set slide_review.overall_pass to false whenever slide_review.needs_refinement is true.
Set slide_review.needs_refinement to true whenever any bbox suggested_action is not accept, any bbox is false_positive, a missed tissue core is present, a merged bbox needs splitting, or a near-full-thumbnail/degenerated bbox is present.
For near-full-thumbnail boxes, suggested_action should usually be rerun_detector rather than refine_tighter.
"""

FEEDBACK_REDETECT_PROMPT = """\
You are looking at a whole slide image containing tissue core biopsies at low magnification.

First, count how many separate tissue cores you see in the source thumbnail.
Then, draw a bounding box around each tissue core.

This is a second-pass redetection after a reviewer found an error in the first detection.
Use the reviewer feedback to pay special attention to subtle missed tissue, but rerun detection from the source thumbnail rather than merely copying the previous boxes.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the previous Stage 1 detection overlay.
- The text below includes the previous bbox geometry and reviewer feedback.

Output a JSON array of bounding boxes in normalized 0-1000 coordinates:
[{"box_2d": [y_min, x_min, y_max, x_max], "label": "tissue_1"}]

Rules:
- Each tissue core or distinct tissue fragment must have its own separate bounding box.
- Do not merge multiple separate cores into one box.
- Include faint tissue fragments if they have tissue-like color or structure.
- Ignore glass edges, pen marks, bubbles, dust, debris, and mounting-media smudges.
- Output JSON only. Do not include prose.
"""

FIRST_PASS_DETECT_PROMPT = """\
You are looking at a whole slide image containing tissue core biopsies at low magnification.

First, count how many separate tissue cores you see in the image.
Then, draw a bounding box around each tissue core.

Output a JSON array of bounding boxes in normalized 0-1000 coordinates:
[{"box_2d": [y_min, x_min, y_max, x_max], "label": "tissue_1"}]

Rules:
- Each tissue core or distinct tissue fragment must have its own separate bounding box.
- Do not merge multiple separate cores into one box.
- Include faint tissue fragments if they have tissue-like color or structure.
- Ignore glass edges, pen marks, bubbles, dust, debris, and mounting-media smudges.
- Output JSON only. Do not include prose.
"""

RAW_OVERLAY_REVIEW_PROMPT = """\
Your task is to review the quality of this tissue detection overlay.

Inputs:
- Image 1 is a whole-slide thumbnail with raw tissue-detection bounding boxes drawn on it.
- A text list gives every bbox id and its geometry.

For each bounding box, output:
1. tightness: whether the box is very_tight, very_loose, or ok.
2. detection_signal: whether the box is detecting signal, noise, or uncertain.

Definitions:
- signal: visible tissue-like foreground is present inside the bbox at thumbnail scale.
- noise: the bbox mainly covers background, glass marks, dust, pen, bubble, edge artifact, debris, or other non-tissue-like visual noise at thumbnail scale.
- uncertain: the thumbnail overlay is not enough to confidently decide signal versus noise.
- very_tight: the bbox appears to cut off visible tissue-like signal.
- very_loose: the bbox includes a large amount of irrelevant background compared with the tissue-like signal.
- ok: the bbox has reasonable coverage and margin for the visible signal at thumbnail scale.

Do not use pathology domain knowledge.
Do not infer control tissue, diagnosis, specimen type, or downstream handling.
Do not judge missed tissue outside the boxes.
Do not propose new boxes or refined coordinates.

Return only one JSON object with this exact shape:
{
  "overlay_review": {
    "bbox_count": 0,
    "overall_quality": "ok",
    "reasoning": "short summary"
  },
  "bbox_reviews": [
    {
      "bbox_id": "r0_01",
      "tightness": "ok",
      "detection_signal": "signal",
      "reasoning": "short visual reason"
    }
  ]
}

Allowed tightness values: very_tight, very_loose, ok, uncertain.
Allowed detection_signal values: signal, noise, uncertain.
Allowed overall_quality values: ok, mixed, poor, uncertain.

Every bbox id from the text list must appear exactly once in bbox_reviews.
"""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_text": text}


def _extract_json_payload(text: str) -> Any:
    text = text.strip()
    try:
        payload = json.loads(text)
        if _is_detection_payload(payload):
            return payload
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            if _is_detection_payload(payload):
                return payload
        except json.JSONDecodeError:
            pass
    recovered = _recover_detection_objects(text)
    if len(recovered) > 1:
        return recovered
    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for idx in starts:
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if _is_detection_payload(payload):
            if isinstance(payload, dict) and _detection_coords(payload) is not None:
                if len(recovered) > 1:
                    return recovered
            return payload
    if recovered:
        return recovered
    return {"raw_text": text}


def _detection_coords(item: dict[str, Any]) -> Any:
    for key, value in item.items():
        normalized_key = re.sub(r"\s+", "", str(key)).lower()
        if normalized_key in {"box_2d", "bbox_2d", "bbox"}:
            return value
        if not isinstance(value, list) or len(value) != 4:
            continue
        if re.fullmatch(r"(?:bbox|box)(?:[_-]?\d+)?", normalized_key):
            return value
        if re.fullmatch(r".*(?:bbox|box)[_-]?\d+", normalized_key):
            return value
    return None


def _is_detection_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return bool(
            _detection_coords(payload) is not None
            or isinstance(payload.get("detected_regions"), list)
            or isinstance(payload.get("bboxes"), list)
            or isinstance(payload.get("boxes"), list)
            or "raw_text" in payload
        )
    if isinstance(payload, list):
        return len(payload) == 0 or any(isinstance(item, dict) for item in payload)
    return False


def _recover_detection_objects(text: str) -> list[dict[str, Any]]:
    """Recover complete detection objects from malformed or truncated JSON arrays."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        idx = start + end
        if isinstance(payload, dict) and _detection_coords(payload) is not None:
            objects.append(payload)
    return objects


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    base_url = args.api_base or "https://openrouter.ai/api/v1"
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return base_url, api_key


def _selected_rows(manifest: Path, indices: list[int]) -> list[dict[str, str]]:
    rows = _read_csv(manifest)
    by_index = {int(row["index"]): row for row in rows}
    missing = [idx for idx in indices if idx not in by_index]
    if missing:
        raise SystemExit(f"Missing manifest indices: {missing}")
    return [by_index[idx] for idx in indices]


def _load_bboxes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload.get("detected_regions", []))


def _bbox_geometry(bbox: dict[str, Any], thumbnail_size: tuple[int, int]) -> dict[str, Any]:
    width, height = thumbnail_size
    x1, y1, x2, y2 = [float(v) for v in bbox.get("bbox_thumbnail", [0, 0, 0, 0])]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_ratio = (bw * bh) / float(max(1, width * height))
    width_ratio = bw / float(max(1, width))
    height_ratio = bh / float(max(1, height))
    touches_edges = {
        "left": x1 <= 2,
        "top": y1 <= 2,
        "right": x2 >= width - 3,
        "bottom": y2 >= height - 3,
    }
    edge_touch_count = sum(1 for value in touches_edges.values() if value)
    near_full = area_ratio >= 0.70 or (width_ratio >= 0.85 and height_ratio >= 0.85)
    edge_spanning = edge_touch_count >= 3
    return {
        "bbox_thumbnail": [round(x1), round(y1), round(x2), round(y2)],
        "thumbnail_size": [width, height],
        "area_ratio": round(area_ratio, 4),
        "width_ratio": round(width_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "touches_edges": touches_edges,
        "edge_touch_count": edge_touch_count,
        "geometry_near_full_thumbnail": near_full,
        "geometry_edge_spanning": edge_spanning,
    }


def _load_review_result(output_root: Path, index: int) -> dict[str, Any]:
    results_path = output_root / "reviews" / "detection_review_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Reviewer results JSONL does not exist: {results_path}")
    task_id = f"detection_review_{index:03d}"
    for row in _read_jsonl(results_path):
        if row.get("task_id") == task_id:
            return row
    raise SystemExit(f"Reviewer result not found for {task_id} in {results_path}")


def _review_feedback_text(review: dict[str, Any]) -> str:
    parsed = review.get("parsed_response") if isinstance(review.get("parsed_response"), dict) else {}
    slide = _slide_review(parsed)
    bbox_reviews = _bbox_reviews(parsed)
    lines = [
        f"Reviewer overall_pass: {slide.get('overall_pass')}",
        f"Reviewer missed_tissue_core: {slide.get('missed_tissue_core')}",
        f"Reviewer needs_refinement: {slide.get('needs_refinement')}",
        f"Reviewer priority: {slide.get('priority')}",
        f"Reviewer slide reasoning: {slide.get('reasoning', '')}",
        "Reviewer bbox findings:",
    ]
    for bbox in bbox_reviews:
        lines.append(
            "- "
            + json.dumps(
                {
                    "bbox_id": bbox.get("bbox_id", ""),
                    "localization_quality": bbox.get("localization_quality", ""),
                    "excess_background": bbox.get("excess_background", ""),
                    "is_near_full_thumbnail_box": bbox.get("is_near_full_thumbnail_box", ""),
                    "cuts_off_tissue": bbox.get("cuts_off_tissue", ""),
                    "multiple_cores_in_bbox": bbox.get("multiple_cores_in_bbox", ""),
                    "artifact_false_positive": bbox.get("artifact_false_positive", ""),
                    "suggested_action": bbox.get("suggested_action", ""),
                    "reasoning": bbox.get("reasoning", ""),
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def _normalised_detection_items(payload: Any, thumbnail_size: tuple[int, int]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if _detection_coords(payload) is not None:
            items = [payload]
        elif isinstance(payload.get("detected_regions"), list):
            items = payload["detected_regions"]
        elif isinstance(payload.get("bboxes"), list):
            items = payload["bboxes"]
        elif isinstance(payload.get("boxes"), list):
            items = payload["boxes"]
        else:
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    width, height = thumbnail_size
    detections: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        coords = _detection_coords(item)
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            raw_coords = [float(value) for value in coords]
        except (TypeError, ValueError):
            continue
        bbox_thumbnail, normalized_box, interpretation, warnings = _parse_detection_coords(raw_coords, thumbnail_size)
        detections.append(
            {
                "label": str(item.get("label") or f"tissue_{idx}"),
                "raw_box_2d": [round(value, 3) for value in raw_coords],
                "coordinate_interpretation": interpretation,
                "parser_warnings": warnings,
                "box_2d_yxyx_normalized": normalized_box,
                "bbox_thumbnail": bbox_thumbnail,
            }
        )
    return detections


def _parse_detection_coords(
    raw_coords: list[float],
    thumbnail_size: tuple[int, int],
) -> tuple[list[int], list[int], str, list[str]]:
    width, height = thumbnail_size
    warnings: list[str] = []

    def pixel_bbox(x1: float, y1: float, x2: float, y2: float) -> list[int]:
        sx1, sx2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        sy1, sy2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        return [round(sx1), round(sy1), round(sx2), round(sy2)]

    def normalized_from_pixel(bbox: list[int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            round(y1 / float(max(1, height)) * 1000.0),
            round(x1 / float(max(1, width)) * 1000.0),
            round(y2 / float(max(1, height)) * 1000.0),
            round(x2 / float(max(1, width)) * 1000.0),
        ]

    y1, x1, y2, x2 = raw_coords
    if all(0.0 <= value <= 1000.0 for value in raw_coords):
        cy1, cy2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        cx1, cx2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        if [cy1, cx1, cy2, cx2] != raw_coords:
            warnings.append("coords_sorted_or_clipped")
        bbox = [
            round(cx1 / 1000.0 * width),
            round(cy1 / 1000.0 * height),
            round(cx2 / 1000.0 * width),
            round(cy2 / 1000.0 * height),
        ]
        normalized = [round(cy1), round(cx1), round(cy2), round(cx2)]
        return bbox, normalized, "normalized_yxyx", warnings

    px1, py1, px2, py2 = raw_coords
    if 0 <= px1 <= width and 0 <= px2 <= width and 0 <= py1 <= height and 0 <= py2 <= height:
        bbox = pixel_bbox(px1, py1, px2, py2)
        warnings.append("interpreted_as_pixel_xyxy_outside_prompt_schema")
        return bbox, normalized_from_pixel(bbox), "pixel_xyxy", warnings

    if 0 <= y1 <= height and 0 <= y2 <= height and 0 <= x1 <= width and 0 <= x2 <= width:
        bbox = pixel_bbox(x1, y1, x2, y2)
        warnings.append("interpreted_as_pixel_yxyx_outside_prompt_schema")
        return bbox, normalized_from_pixel(bbox), "pixel_yxyx", warnings

    cy1, cy2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    cx1, cx2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    bbox = [
        round(cx1 / 1000.0 * width),
        round(cy1 / 1000.0 * height),
        round(cx2 / 1000.0 * width),
        round(cy2 / 1000.0 * height),
    ]
    if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
        warnings.append("degenerate_after_clipping")
    warnings.append("coords_outside_0_1000_clipped_as_normalized_yxyx")
    return bbox, [round(cy1), round(cx1), round(cy2), round(cx2)], "normalized_yxyx_clipped", warnings


def _transform_normalized_yxyx_to_rot0(coords: list[float], rotation: int) -> list[float]:
    """Transform a normalized yxyx bbox from the rotated detector view to the source thumbnail."""
    y1, x1, y2, x2 = coords
    if rotation == 0:
        return [y1, x1, y2, x2]
    if rotation == 90:
        return [1000 - x2, y1, 1000 - x1, y2]
    if rotation == 180:
        return [1000 - y2, 1000 - x2, 1000 - y1, 1000 - x1]
    if rotation == 270:
        return [x1, 1000 - y2, x2, 1000 - y1]
    raise ValueError(f"Unsupported rotation: {rotation}")


def _load_raw_orientation_bboxes(
    bboxes_path: Path,
    thumbnail_size: tuple[int, int],
    rotation: int,
) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(bboxes_path.read_text())
    per_orientation = payload.get("per_orientation_raw")
    if not isinstance(per_orientation, dict):
        return [], "missing_per_orientation_raw"
    raw_items = per_orientation.get(str(rotation))
    if not isinstance(raw_items, list):
        return [], f"missing_raw_rotation_{rotation}"

    bboxes: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        coords = _detection_coords(item)
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            raw_coords = [float(value) for value in coords]
        except (TypeError, ValueError):
            continue
        warnings: list[str] = []
        if all(0.0 <= value <= 1000.0 for value in raw_coords):
            parse_coords = _transform_normalized_yxyx_to_rot0(raw_coords, rotation)
            if rotation:
                warnings.append(f"transformed_from_rot{rotation}_to_rot0")
        else:
            parse_coords = raw_coords
            if rotation:
                warnings.append("rotation_transform_skipped_for_non_normalized_coords")
        bbox_thumbnail, normalized_box, interpretation, parse_warnings = _parse_detection_coords(
            parse_coords,
            thumbnail_size,
        )
        label = f"r{rotation}_{idx:02d}"
        bboxes.append(
            {
                "label": label,
                "source_label": str(item.get("label", "")),
                "rotation": rotation,
                "raw_box_2d": [round(value, 3) for value in raw_coords],
                "box_2d_yxyx_normalized": normalized_box,
                "coordinate_interpretation": interpretation,
                "parser_warnings": warnings + parse_warnings,
                "bbox_thumbnail": bbox_thumbnail,
            }
        )
    if not bboxes:
        return [], f"no_parseable_raw_rotation_{rotation}_bboxes"
    return bboxes, ""


def _raw_overlay_bbox_text(bboxes: list[dict[str, Any]], thumbnail_size: tuple[int, int]) -> str:
    lines = [f"Thumbnail size: {thumbnail_size[0]} x {thumbnail_size[1]} pixels."]
    for bbox in bboxes:
        geom = _bbox_geometry(bbox, thumbnail_size)
        lines.append(
            "- "
            + json.dumps(
                {
                    "bbox_id": bbox.get("label", ""),
                    "source_label": bbox.get("source_label", ""),
                    "rotation": bbox.get("rotation", ""),
                    "raw_box_2d": bbox.get("raw_box_2d", []),
                    "bbox_thumbnail": geom["bbox_thumbnail"],
                    "area_ratio": geom["area_ratio"],
                    "width_ratio": geom["width_ratio"],
                    "height_ratio": geom["height_ratio"],
                    "edge_touch_count": geom["edge_touch_count"],
                    "geometry_edge_spanning": geom["geometry_edge_spanning"],
                    "geometry_near_full_thumbnail": geom["geometry_near_full_thumbnail"],
                    "parser_warnings": bbox.get("parser_warnings", []),
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def _draw_redetect_overlay(thumbnail_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_thumbnail"]
        label = str(idx + 1)
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label_box = draw.textbbox((x1 + 4, y1 + 4), label, font=font)
        draw.rectangle(label_box, fill="white", outline=color, width=2)
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_feedback_redetect_pdf(output_dir: Path, record: dict[str, Any]) -> None:
    page = Image.new("RGB", (1800, 2200), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(15)
    y = 30
    draw.text((40, y), record["case_display"], font=title_font, fill="black")
    y += 46
    draw.text(
        (40, y),
        f"Second-pass detections: {len(record['detections'])} | model={record['model']}",
        font=body_font,
        fill="black",
    )
    y += 38
    source = _thumb(Path(record["thumbnail_path"]), (540, 360))
    original = _thumb(Path(record["original_overlay_path"]), (540, 360))
    redetect = _thumb(Path(record["redetect_overlay_path"]), (540, 360))
    for x, label, image in (
        (40, "Source thumbnail", source),
        (630, "Original overlay", original),
        (1220, "Feedback redetection", redetect),
    ):
        draw.text((x, y), label, font=body_font, fill="black")
        page.paste(image, (x, y + 30))
    y += 430
    draw.text((40, y), "Reviewer feedback supplied to detector", font=body_font, fill="black")
    y += 30
    y = _draw_wrapped(draw, (40, y), record["reviewer_feedback"], small_font, 160, "#111111")
    y += 10
    draw.text((40, y), "Second-pass parsed detections", font=body_font, fill="black")
    y += 30
    for detection in record["detections"]:
        y = _draw_wrapped(draw, (60, y), json.dumps(detection, sort_keys=True), small_font, 160, "#111111")
    pdf_path = output_dir / "feedback_redetect_report.pdf"
    page.save(pdf_path, "PDF", resolution=150)


def _chat_with_images(
    *,
    model: str,
    prompt_text: str,
    image_paths: list[Path],
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key: str,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    request_kwargs: dict[str, Any] = {}
    if reasoning_effort:
        request_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
                + [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}}
                    for path in image_paths
                ],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **request_kwargs,
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    response_model = getattr(response, "model", "")
    return raw, usage, response_model


def _detections_as_bboxes(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": detection.get("label", f"tissue_{idx}"),
            "bbox_thumbnail": detection.get("bbox_thumbnail", [0, 0, 0, 0]),
            "synthetic": False,
            "synthetic_source": "",
        }
        for idx, detection in enumerate(detections, start=1)
    ]


def _review_feedback_text_from_parsed(parsed: dict[str, Any]) -> str:
    return _review_feedback_text({"parsed_response": parsed})


def _raw_excerpt(value: Any, limit: int = 1800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated in PDF; full raw response is in JSON]"


def _write_model_loop_pdf(output_dir: Path, record: dict[str, Any]) -> None:
    page = Image.new("RGB", (2200, 3500), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(30)
    body_font = _font(18)
    small_font = _font(15)
    y = 35
    draw.text((45, y), f"Model loop: {record['model']}", font=title_font, fill="black")
    y += 44
    draw.text((45, y), record["case_display"], font=body_font, fill="#222222")
    y += 34
    slide = _slide_review(record.get("review_parsed", {}))
    draw.text(
        (45, y),
        f"first_pass={len(record['first_pass_detections'])} boxes | "
        f"review missed={slide.get('missed_tissue_core')} needs_refinement={slide.get('needs_refinement')} | "
        f"feedback_redetect={len(record['feedback_redetect_detections'])} boxes",
        font=body_font,
        fill="#111111",
    )
    y += 46

    cards = [
        (
            "Source thumbnail",
            Path(record["thumbnail_path"]),
            "Original input.",
        ),
        (
            "First-pass detection",
            Path(record["first_pass_overlay_path"]),
            record.get("first_pass_summary", ""),
        ),
        (
            "Same-model review",
            Path(record["first_pass_overlay_path"]),
            slide.get("reasoning", ""),
        ),
        (
            "Feedback redetection",
            Path(record["feedback_redetect_overlay_path"]),
            record.get("feedback_redetect_summary", ""),
        ),
    ]
    card_w, card_h = 1030, 610
    image_w, image_h = 940, 340
    for idx, (title, image_path, summary) in enumerate(cards):
        x = 45 + (idx % 2) * (card_w + 50)
        yy = y + (idx // 2) * (card_h + 45)
        draw.rectangle((x, yy, x + card_w, yy + card_h), outline="#bbbbbb", width=2)
        draw.text((x + 22, yy + 18), title, font=body_font, fill="black")
        image = _thumb(image_path, (image_w, image_h))
        page.paste(image, (x + 45, yy + 55))
        ty = yy + 415
        for line in textwrap.wrap(str(summary), 112):
            draw.text((x + 22, ty), line, font=small_font, fill="#111111")
            ty += 20

    y = y + 2 * (card_h + 45) + 25
    draw.text((45, y), "Parsed outputs", font=body_font, fill="black")
    y += 30
    parsed_text = (
        "First pass: "
        + _raw_excerpt(json.dumps(record["first_pass_detections"], sort_keys=True), 1800)
        + "\nFeedback redetect: "
        + _raw_excerpt(json.dumps(record["feedback_redetect_detections"], sort_keys=True), 1800)
    )
    y = _draw_wrapped(draw, (45, y), parsed_text, small_font, 180, "#111111")
    y += 20
    draw.text((45, y), "Raw model outputs", font=body_font, fill="black")
    y += 30
    raw_text = (
        "First-pass raw:\n"
        + _raw_excerpt(record.get("first_pass_raw", ""), 1300)
        + "\n\nSame-model review raw:\n"
        + _raw_excerpt(record.get("review_raw", ""), 1300)
        + "\n\nFeedback-redetect raw:\n"
        + _raw_excerpt(record.get("feedback_redetect_raw", ""), 1300)
    )
    _draw_wrapped(draw, (45, y), raw_text, small_font, 180, "#111111")
    page.save(output_dir / "model_loop_report.pdf", "PDF", resolution=150)


def _bbox_text(bboxes: list[dict[str, Any]], thumbnail_size: tuple[int, int]) -> str:
    lines = [f"Thumbnail size: {thumbnail_size[0]} x {thumbnail_size[1]} pixels."]
    for bbox in bboxes:
        geom = _bbox_geometry(bbox, thumbnail_size)
        label = bbox.get("label", "")
        lines.append(
            "- "
            + json.dumps(
                {
                    "bbox_id": label,
                    "bbox_thumbnail": geom["bbox_thumbnail"],
                    "area_ratio": geom["area_ratio"],
                    "width_ratio": geom["width_ratio"],
                    "height_ratio": geom["height_ratio"],
                    "edge_touch_count": geom["edge_touch_count"],
                    "geometry_edge_spanning": geom["geometry_edge_spanning"],
                    "geometry_near_full_thumbnail": geom["geometry_near_full_thumbnail"],
                    "synthetic": bool(bbox.get("synthetic", False)),
                    "synthetic_source": bbox.get("synthetic_source", ""),
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def _case_display(row: dict[str, str]) -> str:
    return (
        f"{row['index']}/100 | {row['stain']} | {row['case_id']} | "
        f"{row['Anon_Path_ID']} | {Path(row['wsi_path']).name}"
    )


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def build_detection_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = args.output_root.resolve()
    rows = _selected_rows(args.manifest.resolve(), args.indices)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        thumbnail_path = Path(row["thumbnail_path"])
        overlay_path = Path(row["overlay_path"])
        bboxes_json_path = Path(row["bboxes_json_path"])
        if not thumbnail_path.exists():
            raise SystemExit(f"Thumbnail does not exist: {thumbnail_path}")
        if not overlay_path.exists():
            raise SystemExit(f"Overlay does not exist: {overlay_path}")
        if not bboxes_json_path.exists():
            raise SystemExit(f"Bboxes JSON does not exist: {bboxes_json_path}")
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size
        bboxes = _load_bboxes(bboxes_json_path)
        task = {
            "task_id": f"detection_review_{int(row['index']):03d}",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "case_display": _case_display(row),
            "known_case_note": KNOWN_CASE_NOTES.get(int(row["index"]), ""),
            "manifest_row": row,
            "thumbnail_path": str(thumbnail_path),
            "overlay_path": str(overlay_path),
            "bboxes_json_path": str(bboxes_json_path),
            "bbox_count": len(bboxes),
            "bbox_text": _bbox_text(bboxes, thumbnail_size),
            "bboxes": [
                {
                    "label": bbox.get("label", ""),
                    **_bbox_geometry(bbox, thumbnail_size),
                }
                for bbox in bboxes
            ],
            "prompt": DETECTION_REVIEW_PROMPT,
            "created_at": _timestamp(),
        }
        tasks.append(task)
    tasks_path = output_root / "tasks" / "detection_review_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    return tasks


def _review_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    user_text = (
        task["prompt"]
        + "\n\nCase:\n"
        + task["case_display"]
        + "\n\nDetected bboxes:\n"
        + task["bbox_text"]
    )
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "case_display": task["case_display"],
        "prompt_version": task["prompt_version"],
        "model": args.model,
        "thumbnail_path": task["thumbnail_path"],
        "overlay_path": task["overlay_path"],
        "bboxes_json_path": task["bboxes_json_path"],
        "bbox_count": task["bbox_count"],
        "known_case_note": task.get("known_case_note", ""),
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(Path(task["thumbnail_path"]))}},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(Path(task["overlay_path"]))}},
                    ],
                }
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        parsed = _extract_json_object(raw)
        record["raw_response"] = raw
        record["parsed_response"] = parsed
        record["usage"] = response.usage.model_dump() if getattr(response, "usage", None) else {}
        record["response_model"] = getattr(response, "model", "")
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def run_detection_review(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    tasks = build_detection_tasks(args)
    tasks_path = output_root / "tasks" / "detection_review_tasks.jsonl"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks), "tasks_jsonl": str(tasks_path)}, indent=2))
        return 0

    base_url, api_key = _api_settings(args)
    results: list[dict[str, Any]] = []
    if args.max_concurrent > 1:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(_review_one, task, args, base_url, api_key) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [_review_one(task, args, base_url, api_key) for task in tasks]

    results.sort(key=lambda row: row["task_id"])
    results_path = output_root / "reviews" / "detection_review_results.jsonl"
    _write_jsonl(results_path, results)
    summarize_detection_review(output_root, results)
    write_reproduction(output_root, args, tasks_path, results_path)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "tasks_jsonl": str(tasks_path),
                "results_jsonl": str(results_path),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


def build_raw_overlay_review_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = args.output_root.resolve()
    rows = _selected_rows(args.manifest.resolve(), args.indices)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        thumbnail_path = Path(row["thumbnail_path"])
        bboxes_json_path = Path(row["bboxes_json_path"])
        if not thumbnail_path.exists():
            raise SystemExit(f"Thumbnail does not exist: {thumbnail_path}")
        if not bboxes_json_path.exists():
            raise SystemExit(f"Bboxes JSON does not exist: {bboxes_json_path}")
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size

        raw_bboxes, skip_reason = _load_raw_orientation_bboxes(
            bboxes_json_path,
            thumbnail_size,
            args.rotation,
        )
        case_slug = _safe_slug(f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}")
        overlay_path = output_root / "raw_overlays" / f"{case_slug}_rot{args.rotation}_raw_overlay.png"
        if raw_bboxes:
            _draw_redetect_overlay(thumbnail_path, raw_bboxes, overlay_path)

        task = {
            "task_id": f"raw_overlay_review_{int(row['index']):03d}_rot{args.rotation}",
            "prompt_version": RAW_OVERLAY_REVIEW_PROMPT_VERSION,
            "model": args.model,
            "case_display": _case_display(row),
            "manifest_row": row,
            "rotation": args.rotation,
            "thumbnail_path": str(thumbnail_path),
            "overlay_path": str(overlay_path) if raw_bboxes else "",
            "bboxes_json_path": str(bboxes_json_path),
            "bbox_count": len(raw_bboxes),
            "bbox_text": _raw_overlay_bbox_text(raw_bboxes, thumbnail_size) if raw_bboxes else "",
            "bboxes": [
                {
                    "label": bbox.get("label", ""),
                    "source_label": bbox.get("source_label", ""),
                    "rotation": bbox.get("rotation", ""),
                    "raw_box_2d": bbox.get("raw_box_2d", []),
                    "box_2d_yxyx_normalized": bbox.get("box_2d_yxyx_normalized", []),
                    "coordinate_interpretation": bbox.get("coordinate_interpretation", ""),
                    "parser_warnings": bbox.get("parser_warnings", []),
                    **_bbox_geometry(bbox, thumbnail_size),
                }
                for bbox in raw_bboxes
            ],
            "prompt": RAW_OVERLAY_REVIEW_PROMPT,
            "skip_reason": skip_reason,
            "created_at": _timestamp(),
        }
        tasks.append(task)
    tasks_path = output_root / "tasks" / f"raw_overlay_review_rot{args.rotation}_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    return tasks


def _review_one_raw_overlay(
    task: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "case_display": task["case_display"],
        "prompt_version": task["prompt_version"],
        "model": args.model,
        "rotation": task["rotation"],
        "thumbnail_path": task["thumbnail_path"],
        "overlay_path": task["overlay_path"],
        "bboxes_json_path": task["bboxes_json_path"],
        "bbox_count": task["bbox_count"],
        "bboxes": task["bboxes"],
        "created_at": _timestamp(),
        "error": "",
        "skip_reason": task.get("skip_reason", ""),
    }
    if task.get("skip_reason"):
        record["raw_response"] = ""
        record["parsed_response"] = {
            "overlay_review": {
                "bbox_count": 0,
                "overall_quality": "uncertain",
                "reasoning": task["skip_reason"],
            },
            "bbox_reviews": [],
        }
        record["usage"] = {}
        record["response_model"] = ""
        return record

    user_text = (
        task["prompt"]
        + "\n\nCase:\n"
        + task["case_display"]
        + f"\n\nReviewed detector orientation: rot{task['rotation']} only."
        + "\n\nDetected bboxes:\n"
        + task["bbox_text"]
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=user_text,
            image_paths=[Path(task["overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        parsed = _extract_json_object(raw)
        record["raw_response"] = raw
        record["parsed_response"] = parsed
        record["usage"] = usage
        record["response_model"] = response_model
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def run_raw_overlay_review(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    tasks = build_raw_overlay_review_tasks(args)
    tasks_path = output_root / "tasks" / f"raw_overlay_review_rot{args.rotation}_tasks.jsonl"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "reviewable_tasks": sum(1 for task in tasks if not task.get("skip_reason")),
                    "skipped_tasks": sum(1 for task in tasks if task.get("skip_reason")),
                    "tasks_jsonl": str(tasks_path),
                    "output_root": str(output_root),
                },
                indent=2,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    reviewable = [task for task in tasks if not task.get("skip_reason")]
    skipped = [
        _review_one_raw_overlay(task, args, base_url, api_key)
        for task in tasks
        if task.get("skip_reason")
    ]
    results: list[dict[str, Any]] = []
    if args.max_concurrent > 1:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [
                pool.submit(_review_one_raw_overlay, task, args, base_url, api_key)
                for task in reviewable
            ]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [_review_one_raw_overlay(task, args, base_url, api_key) for task in reviewable]
    results.extend(skipped)
    results.sort(key=lambda row: row["task_id"])

    results_path = output_root / "reviews" / f"raw_overlay_review_rot{args.rotation}_results.jsonl"
    _write_jsonl(results_path, results)
    summarize_raw_overlay_review(output_root, args.rotation, results)
    write_raw_overlay_review_reproduction(output_root, args, tasks_path, results_path)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "reviewable_tasks": len(reviewable),
                "skipped_tasks": len(skipped),
                "tasks_jsonl": str(tasks_path),
                "results_jsonl": str(results_path),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


def run_feedback_redetect(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    row = _selected_rows(args.manifest.resolve(), [args.index])[0]
    thumbnail_path = Path(row["thumbnail_path"])
    overlay_path = Path(row["overlay_path"])
    bboxes_json_path = Path(row["bboxes_json_path"])
    for path in (thumbnail_path, overlay_path, bboxes_json_path):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")

    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size
    original_bboxes = _load_bboxes(bboxes_json_path)
    review = _load_review_result(output_root, args.index)
    reviewer_feedback = _review_feedback_text(review)
    case_slug = f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}"
    run_label = args.run_label or _safe_slug(args.model)
    out_dir = output_root / "feedback_redetect" / case_slug / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = (
        FEEDBACK_REDETECT_PROMPT
        + "\n\nCase:\n"
        + _case_display(row)
        + "\n\nPrevious Stage 1 detected bboxes:\n"
        + _bbox_text(original_bboxes, thumbnail_size)
        + "\n\nReviewer feedback from previous blind review:\n"
        + reviewer_feedback
    )
    task = {
        "task_id": f"feedback_redetect_{int(row['index']):03d}",
        "prompt_version": FEEDBACK_REDETECT_PROMPT_VERSION,
        "model": args.model,
        "case_display": _case_display(row),
        "thumbnail_path": str(thumbnail_path),
        "original_overlay_path": str(overlay_path),
        "bboxes_json_path": str(bboxes_json_path),
        "reviewer_feedback": reviewer_feedback,
        "prompt": prompt_text,
        "created_at": _timestamp(),
    }
    _write_json(out_dir / "feedback_redetect_task.json", task)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "task_json": str(out_dir / "feedback_redetect_task.json")}, indent=2))
        return 0

    from openai import OpenAI

    base_url, api_key = _api_settings(args)
    client = OpenAI(base_url=base_url, api_key=api_key)
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "case_display": task["case_display"],
        "prompt_version": task["prompt_version"],
        "model": args.model,
        "thumbnail_path": str(thumbnail_path),
        "original_overlay_path": str(overlay_path),
        "bboxes_json_path": str(bboxes_json_path),
        "reviewer_feedback": reviewer_feedback,
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(thumbnail_path)}},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(overlay_path)}},
                    ],
                }
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        parsed = _extract_json_payload(raw)
        detections = _normalised_detection_items(parsed, thumbnail_size)
        redetect_overlay_path = out_dir / "feedback_redetect_overlay.png"
        _draw_redetect_overlay(thumbnail_path, detections, redetect_overlay_path)
        record.update(
            {
                "raw_response": raw,
                "parsed_response": parsed,
                "detections": detections,
                "redetect_overlay_path": str(redetect_overlay_path),
                "usage": response.usage.model_dump() if getattr(response, "usage", None) else {},
                "response_model": getattr(response, "model", ""),
            }
        )
        _write_feedback_redetect_pdf(out_dir, record)
    except Exception as exc:
        record.update(
            {
                "raw_response": "",
                "parsed_response": {},
                "detections": [],
                "redetect_overlay_path": "",
                "usage": {},
                "response_model": "",
                "error": repr(exc),
            }
        )

    _write_json(out_dir / "feedback_redetect_result.json", record)
    write_feedback_reproduction(out_dir, args, task)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "detections": len(record["detections"]),
                "error": record["error"],
                "result_json": str(out_dir / "feedback_redetect_result.json"),
                "overlay": record.get("redetect_overlay_path", ""),
                "pdf": str(out_dir / "feedback_redetect_report.pdf"),
            },
            indent=2,
        )
    )
    return 0


def run_model_loop(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    row = _selected_rows(args.manifest.resolve(), [args.index])[0]
    thumbnail_path = Path(row["thumbnail_path"])
    if not thumbnail_path.exists():
        raise SystemExit(f"Thumbnail does not exist: {thumbnail_path}")
    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size

    case_slug = f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}"
    run_label = args.run_label or _safe_slug(args.model)
    out_dir = output_root / "model_loops" / case_slug / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    first_prompt = FIRST_PASS_DETECT_PROMPT + "\n\nCase:\n" + _case_display(row)
    first_task = {
        "task_id": f"model_loop_first_detect_{int(row['index']):03d}",
        "model": args.model,
        "case_display": _case_display(row),
        "thumbnail_path": str(thumbnail_path),
        "prompt": first_prompt,
        "created_at": _timestamp(),
    }
    _write_json(out_dir / "first_pass_task.json", first_task)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output_dir": str(out_dir)}, indent=2))
        return 0
    base_url, api_key = _api_settings(args)

    record: dict[str, Any] = {
        "case_display": _case_display(row),
        "model": args.model,
        "run_label": run_label,
        "thumbnail_path": str(thumbnail_path),
        "created_at": _timestamp(),
        "errors": [],
    }
    try:
        first_raw, first_usage, first_response_model = _chat_with_images(
            model=args.model,
            prompt_text=first_prompt,
            image_paths=[thumbnail_path],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        first_parsed = _extract_json_payload(first_raw)
        first_detections = _normalised_detection_items(first_parsed, thumbnail_size)
        first_overlay = out_dir / "first_pass_overlay.png"
        _draw_redetect_overlay(thumbnail_path, first_detections, first_overlay)
        _write_json(
            out_dir / "first_pass_result.json",
            {
                "raw_response": first_raw,
                "parsed_response": first_parsed,
                "detections": first_detections,
                "usage": first_usage,
                "response_model": first_response_model,
                "overlay_path": str(first_overlay),
            },
        )
        record.update(
            {
                "first_pass_raw": first_raw,
                "first_pass_parsed": first_parsed,
                "first_pass_detections": first_detections,
                "first_pass_overlay_path": str(first_overlay),
                "first_pass_usage": first_usage,
                "first_pass_response_model": first_response_model,
                "first_pass_summary": f"{len(first_detections)} first-pass detections.",
            }
        )

        first_bboxes = _detections_as_bboxes(first_detections)
        review_prompt = (
            DETECTION_REVIEW_PROMPT
            + "\n\nCase:\n"
            + _case_display(row)
            + "\n\nDetected bboxes:\n"
            + _bbox_text(first_bboxes, thumbnail_size)
        )
        _write_json(
            out_dir / "review_task.json",
            {
                "task_id": f"model_loop_review_{int(row['index']):03d}",
                "model": args.model,
                "case_display": _case_display(row),
                "thumbnail_path": str(thumbnail_path),
                "overlay_path": str(first_overlay),
                "bbox_text": _bbox_text(first_bboxes, thumbnail_size),
                "prompt": review_prompt,
                "created_at": _timestamp(),
            },
        )
        review_raw, review_usage, review_response_model = _chat_with_images(
            model=args.model,
            prompt_text=review_prompt,
            image_paths=[thumbnail_path, first_overlay],
            temperature=args.temperature,
            max_tokens=args.review_max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        review_parsed = _extract_json_object(review_raw)
        _write_json(
            out_dir / "review_result.json",
            {
                "raw_response": review_raw,
                "parsed_response": review_parsed,
                "usage": review_usage,
                "response_model": review_response_model,
            },
        )
        record.update(
            {
                "review_raw": review_raw,
                "review_parsed": review_parsed,
                "review_usage": review_usage,
                "review_response_model": review_response_model,
            }
        )

        reviewer_feedback = _review_feedback_text_from_parsed(review_parsed)
        redetect_prompt = (
            FEEDBACK_REDETECT_PROMPT
            + "\n\nCase:\n"
            + _case_display(row)
            + "\n\nPrevious first-pass detected bboxes:\n"
            + _bbox_text(first_bboxes, thumbnail_size)
            + "\n\nReviewer feedback from same-model review:\n"
            + reviewer_feedback
        )
        _write_json(
            out_dir / "feedback_redetect_task.json",
            {
                "task_id": f"model_loop_feedback_redetect_{int(row['index']):03d}",
                "model": args.model,
                "case_display": _case_display(row),
                "thumbnail_path": str(thumbnail_path),
                "first_pass_overlay_path": str(first_overlay),
                "reviewer_feedback": reviewer_feedback,
                "prompt": redetect_prompt,
                "created_at": _timestamp(),
            },
        )
        redetect_raw, redetect_usage, redetect_response_model = _chat_with_images(
            model=args.model,
            prompt_text=redetect_prompt,
            image_paths=[thumbnail_path, first_overlay],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        redetect_parsed = _extract_json_payload(redetect_raw)
        redetect_detections = _normalised_detection_items(redetect_parsed, thumbnail_size)
        redetect_overlay = out_dir / "feedback_redetect_overlay.png"
        _draw_redetect_overlay(thumbnail_path, redetect_detections, redetect_overlay)
        _write_json(
            out_dir / "feedback_redetect_result.json",
            {
                "raw_response": redetect_raw,
                "parsed_response": redetect_parsed,
                "detections": redetect_detections,
                "usage": redetect_usage,
                "response_model": redetect_response_model,
                "overlay_path": str(redetect_overlay),
            },
        )
        record.update(
            {
                "reviewer_feedback": reviewer_feedback,
                "feedback_redetect_raw": redetect_raw,
                "feedback_redetect_parsed": redetect_parsed,
                "feedback_redetect_detections": redetect_detections,
                "feedback_redetect_overlay_path": str(redetect_overlay),
                "feedback_redetect_usage": redetect_usage,
                "feedback_redetect_response_model": redetect_response_model,
                "feedback_redetect_summary": f"{len(redetect_detections)} feedback-redetect detections.",
            }
        )
        _write_model_loop_pdf(out_dir, record)
        write_model_loop_reproduction(out_dir, args, record)
    except Exception as exc:
        record["errors"].append(repr(exc))

    _write_json(out_dir / "model_loop_result.json", record)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "model": args.model,
                "first_pass_detections": len(record.get("first_pass_detections", [])),
                "review_missed_tissue_core": _slide_review(record.get("review_parsed", {})).get("missed_tissue_core", ""),
                "feedback_redetect_detections": len(record.get("feedback_redetect_detections", [])),
                "errors": record.get("errors", []),
                "pdf": str(out_dir / "model_loop_report.pdf"),
            },
            indent=2,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _refresh_model_loop_record(run_dir: Path) -> dict[str, Any]:
    record_path = run_dir / "model_loop_result.json"
    if not record_path.exists():
        raise SystemExit(f"Missing model loop result: {record_path}")
    record = _read_json(record_path)
    thumbnail_path = Path(record["thumbnail_path"])
    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size

    first_result_path = run_dir / "first_pass_result.json"
    first_result = _read_json(first_result_path) if first_result_path.exists() else {}
    first_raw = str(record.get("first_pass_raw") or first_result.get("raw_response") or "")
    first_parsed = _extract_json_payload(first_raw)
    first_detections = _normalised_detection_items(first_parsed, thumbnail_size)
    first_overlay = run_dir / "first_pass_overlay.png"
    _draw_redetect_overlay(thumbnail_path, first_detections, first_overlay)
    first_result.update(
        {
            "raw_response": first_raw,
            "parsed_response": first_parsed,
            "detections": first_detections,
            "overlay_path": str(first_overlay),
        }
    )
    _write_json(first_result_path, first_result)

    review_result_path = run_dir / "review_result.json"
    review_result = _read_json(review_result_path) if review_result_path.exists() else {}
    review_raw = str(record.get("review_raw") or review_result.get("raw_response") or "")
    review_parsed = _extract_json_object(review_raw) if review_raw else {}
    if review_result_path.exists():
        review_result.update({"raw_response": review_raw, "parsed_response": review_parsed})
        _write_json(review_result_path, review_result)

    redetect_result_path = run_dir / "feedback_redetect_result.json"
    redetect_result = _read_json(redetect_result_path) if redetect_result_path.exists() else {}
    redetect_raw = str(record.get("feedback_redetect_raw") or redetect_result.get("raw_response") or "")
    redetect_parsed = _extract_json_payload(redetect_raw)
    redetect_detections = _normalised_detection_items(redetect_parsed, thumbnail_size)
    redetect_overlay = run_dir / "feedback_redetect_overlay.png"
    _draw_redetect_overlay(thumbnail_path, redetect_detections, redetect_overlay)
    redetect_result.update(
        {
            "raw_response": redetect_raw,
            "parsed_response": redetect_parsed,
            "detections": redetect_detections,
            "overlay_path": str(redetect_overlay),
        }
    )
    _write_json(redetect_result_path, redetect_result)

    record.update(
        {
            "first_pass_raw": first_raw,
            "first_pass_parsed": first_parsed,
            "first_pass_detections": first_detections,
            "first_pass_overlay_path": str(first_overlay),
            "first_pass_summary": f"{len(first_detections)} first-pass detections after parser refresh.",
            "review_raw": review_raw,
            "review_parsed": review_parsed,
            "feedback_redetect_raw": redetect_raw,
            "feedback_redetect_parsed": redetect_parsed,
            "feedback_redetect_detections": redetect_detections,
            "feedback_redetect_overlay_path": str(redetect_overlay),
            "feedback_redetect_summary": f"{len(redetect_detections)} feedback-redetect detections after parser refresh.",
            "parser_refreshed_at": _timestamp(),
        }
    )
    _write_model_loop_pdf(run_dir, record)
    _write_json(record_path, record)
    return record


def _detection_warnings_summary(detections: list[dict[str, Any]]) -> str:
    modes: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for detection in detections:
        mode = str(detection.get("coordinate_interpretation", ""))
        if mode:
            modes[mode] = modes.get(mode, 0) + 1
        for warning in detection.get("parser_warnings", []) or []:
            warnings[str(warning)] = warnings.get(str(warning), 0) + 1
    parts = []
    if modes:
        parts.append("modes=" + ", ".join(f"{key}:{value}" for key, value in sorted(modes.items())))
    if warnings:
        parts.append("warnings=" + ", ".join(f"{key}:{value}" for key, value in sorted(warnings.items())))
    return " | ".join(parts) if parts else "no parser warnings"


def _write_model_loop_comparison_pdf(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_font = _font(30)
    body_font = _font(18)
    small_font = _font(14)
    pages: list[Image.Image] = []

    summary_page = Image.new("RGB", (2200, 1350), "white")
    draw = ImageDraw.Draw(summary_page)
    y = 40
    draw.text((45, y), "Case 070 per-model detect-review-redetect comparison", font=title_font, fill="black")
    y += 45
    if records:
        draw.text((45, y), records[0].get("case_display", ""), font=body_font, fill="#222222")
    y += 55
    draw.text((45, y), "Summary", font=body_font, fill="black")
    y += 35
    for record in records:
        slide = _slide_review(record.get("review_parsed", {}))
        first_detections = record.get("first_pass_detections", [])
        redetect_detections = record.get("feedback_redetect_detections", [])
        line = (
            f"{record.get('model', '')} ({record.get('run_label', '')}) | "
            f"first_pass={len(first_detections)} | "
            f"review missed={slide.get('missed_tissue_core')} needs_refinement={slide.get('needs_refinement')} "
            f"pass={slide.get('overall_pass')} | "
            f"feedback_redetect={len(redetect_detections)}"
        )
        y = _draw_wrapped(draw, (65, y), line, small_font, 170, "#111111")
        y = _draw_wrapped(
            draw,
            (85, y + 5),
            "First-pass parser: " + _detection_warnings_summary(first_detections),
            small_font,
            165,
            "#222222",
        )
        y = _draw_wrapped(
            draw,
            (85, y + 5),
            "Feedback parser: " + _detection_warnings_summary(redetect_detections),
            small_font,
            165,
            "#222222",
        )
        y += 22
    pages.append(summary_page)

    for record in records:
        page = Image.new("RGB", (2600, 2600), "white")
        draw = ImageDraw.Draw(page)
        slide = _slide_review(record.get("review_parsed", {}))
        first_detections = record.get("first_pass_detections", [])
        redetect_detections = record.get("feedback_redetect_detections", [])

        y = 35
        draw.text((45, y), str(record.get("model", "")), font=title_font, fill="black")
        y += 42
        draw.text((45, y), f"label={record.get('run_label', '')}", font=body_font, fill="#222222")
        y += 32
        draw.text(
            (45, y),
            f"first_pass={len(first_detections)} | review missed={slide.get('missed_tissue_core')} "
            f"needs_refinement={slide.get('needs_refinement')} pass={slide.get('overall_pass')} | "
            f"feedback_redetect={len(redetect_detections)}",
            font=body_font,
            fill="#111111",
        )
        y += 48

        first_img = _thumb(Path(record["first_pass_overlay_path"]), (1150, 640))
        redetect_img = _thumb(Path(record["feedback_redetect_overlay_path"]), (1150, 640))
        draw.text((70, y), "First-pass raw output overlay", font=body_font, fill="black")
        draw.text((1380, y), "Feedback-redetect raw output overlay", font=body_font, fill="black")
        page.paste(first_img, (70, y + 35))
        page.paste(redetect_img, (1380, y + 35))
        y += 720

        draw.text((70, y), "Parsed first-pass detections", font=body_font, fill="black")
        draw.text((1380, y), "Parsed feedback-redetect detections", font=body_font, fill="black")
        _draw_wrapped(
            draw,
            (70, y + 32),
            _raw_excerpt(json.dumps(first_detections, sort_keys=True), 1500),
            small_font,
            105,
            "#111111",
        )
        _draw_wrapped(
            draw,
            (1380, y + 32),
            _raw_excerpt(json.dumps(redetect_detections, sort_keys=True), 1500),
            small_font,
            105,
            "#111111",
        )
        y += 520

        draw.text((70, y), "Raw model outputs", font=body_font, fill="black")
        y += 35
        raw = (
            "First raw:\n"
            + _raw_excerpt(record.get("first_pass_raw", ""), 900)
            + "\n\nReview raw:\n"
            + _raw_excerpt(record.get("review_raw", ""), 900)
            + "\n\nRedetect raw:\n"
            + _raw_excerpt(record.get("feedback_redetect_raw", ""), 900)
        )
        _draw_wrapped(draw, (70, y), raw, small_font, 210, "#222222")
        pages.append(page)

    if pages:
        pages[0].save(output_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_model_loop_comparison_reproduction(output_path: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    reproduction = f"""\
Stage 1 per-model loop comparison refresh
=========================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Comparison PDF: {output_path.resolve()}

Command:
python scripts/stage1_detection_review_pilot.py refresh-model-loops \\
  --comparison-pdf {output_path.resolve()} \\
"""
    for run_dir in args.run_dir:
        reproduction += f"  --run-dir {run_dir.resolve()} \\\n"
    reproduction = reproduction.rstrip(" \\\n") + "\n\nInputs:\n"
    for record in records:
        reproduction += (
            f"- {record.get('run_label', '')}: {record.get('model', '')}; "
            f"first_pass={len(record.get('first_pass_detections', []))}; "
            f"feedback_redetect={len(record.get('feedback_redetect_detections', []))}\n"
        )
    (output_path.parent / "reproduction.txt").write_text(reproduction)


def cmd_refresh_model_loops(args: argparse.Namespace) -> int:
    records = [_refresh_model_loop_record(path.resolve()) for path in args.run_dir]
    if args.comparison_pdf:
        _write_model_loop_comparison_pdf(args.comparison_pdf.resolve(), records)
        _write_model_loop_comparison_reproduction(args.comparison_pdf.resolve(), args, records)
    print(
        json.dumps(
            {
                "refreshed": [
                    {
                        "run_label": record.get("run_label", ""),
                        "model": record.get("model", ""),
                        "first_pass_detections": len(record.get("first_pass_detections", [])),
                        "feedback_redetect_detections": len(record.get("feedback_redetect_detections", [])),
                    }
                    for record in records
                ],
                "comparison_pdf": str(args.comparison_pdf.resolve()) if args.comparison_pdf else "",
            },
            indent=2,
        )
    )
    return 0


def _slide_review(parsed: dict[str, Any]) -> dict[str, Any]:
    slide = parsed.get("slide_review")
    return slide if isinstance(slide, dict) else {}


def _bbox_reviews(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = parsed.get("bbox_reviews")
    return reviews if isinstance(reviews, list) else []


def _raw_overlay_review(parsed: dict[str, Any]) -> dict[str, Any]:
    review = parsed.get("overlay_review")
    return review if isinstance(review, dict) else {}


def _flat_raw_overlay_review_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slide_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    for result in results:
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        overlay_review = _raw_overlay_review(parsed)
        bbox_reviews = _bbox_reviews(parsed)
        geom_by_id = {
            str(bbox.get("label", "")): bbox
            for bbox in result.get("bboxes", [])
            if isinstance(bbox, dict)
        }
        slide_rows.append(
            {
                "task_id": result.get("task_id", ""),
                "case_display": result.get("case_display", ""),
                "rotation": result.get("rotation", ""),
                "bbox_count": result.get("bbox_count", ""),
                "parse_ok": bool(overlay_review or bbox_reviews),
                "error": result.get("error", ""),
                "skip_reason": result.get("skip_reason", ""),
                "overall_quality": overlay_review.get("overall_quality", ""),
                "review_bbox_count": overlay_review.get("bbox_count", ""),
                "reasoning": overlay_review.get("reasoning", ""),
                "thumbnail_path": result.get("thumbnail_path", ""),
                "overlay_path": result.get("overlay_path", ""),
            }
        )
        for bbox in bbox_reviews:
            bbox_id = str(bbox.get("bbox_id", ""))
            geom = geom_by_id.get(bbox_id, {})
            bbox_rows.append(
                {
                    "task_id": result.get("task_id", ""),
                    "case_display": result.get("case_display", ""),
                    "rotation": result.get("rotation", ""),
                    "bbox_id": bbox_id,
                    "source_label": geom.get("source_label", ""),
                    "tightness": bbox.get("tightness", ""),
                    "detection_signal": bbox.get("detection_signal", ""),
                    "reasoning": bbox.get("reasoning", ""),
                    "bbox_thumbnail": json.dumps(geom.get("bbox_thumbnail", [])),
                    "raw_box_2d": json.dumps(geom.get("raw_box_2d", [])),
                    "area_ratio": geom.get("area_ratio", ""),
                    "width_ratio": geom.get("width_ratio", ""),
                    "height_ratio": geom.get("height_ratio", ""),
                    "edge_touch_count": geom.get("edge_touch_count", ""),
                    "geometry_near_full_thumbnail": geom.get("geometry_near_full_thumbnail", ""),
                }
            )
    return slide_rows, bbox_rows


def _flat_summary_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slide_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    for result in results:
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        slide = _slide_review(parsed)
        bbox_reviews = _bbox_reviews(parsed)
        slide_rows.append(
            {
                "task_id": result.get("task_id", ""),
                "case_display": result.get("case_display", ""),
                "bbox_count": result.get("bbox_count", ""),
                "parse_ok": bool(slide or bbox_reviews),
                "error": result.get("error", ""),
                "overall_pass": slide.get("overall_pass", ""),
                "missed_tissue_core": slide.get("missed_tissue_core", ""),
                "all_significant_cores_missed": slide.get("all_significant_cores_missed", ""),
                "has_false_positive_artifact": slide.get("has_false_positive_artifact", ""),
                "has_full_thumbnail_or_degenerate_bbox": slide.get("has_full_thumbnail_or_degenerate_bbox", ""),
                "needs_refinement": slide.get("needs_refinement", ""),
                "priority": slide.get("priority", ""),
                "reasoning": slide.get("reasoning", ""),
                "known_case_note": result.get("known_case_note", ""),
                "thumbnail_path": result.get("thumbnail_path", ""),
                "overlay_path": result.get("overlay_path", ""),
            }
        )
        for bbox in bbox_reviews:
            bbox_rows.append(
                {
                    "task_id": result.get("task_id", ""),
                    "case_display": result.get("case_display", ""),
                    "bbox_id": bbox.get("bbox_id", ""),
                    "localization_quality": bbox.get("localization_quality", ""),
                    "excess_background": bbox.get("excess_background", ""),
                    "is_near_full_thumbnail_box": bbox.get("is_near_full_thumbnail_box", ""),
                    "cuts_off_tissue": bbox.get("cuts_off_tissue", ""),
                    "multiple_cores_in_bbox": bbox.get("multiple_cores_in_bbox", ""),
                    "artifact_false_positive": bbox.get("artifact_false_positive", ""),
                    "suggested_action": bbox.get("suggested_action", ""),
                    "reasoning": bbox.get("reasoning", ""),
                }
            )
    return slide_rows, bbox_rows


def summarize_detection_review(output_root: Path, results: list[dict[str, Any]] | None = None) -> None:
    if results is None:
        results = _read_jsonl(output_root / "reviews" / "detection_review_results.jsonl")
    slide_rows, bbox_rows = _flat_summary_rows(results)
    _write_csv(
        output_root / "summary" / "detection_review_slides.csv",
        slide_rows,
        [
            "task_id",
            "case_display",
            "bbox_count",
            "parse_ok",
            "error",
            "overall_pass",
            "missed_tissue_core",
            "all_significant_cores_missed",
            "has_false_positive_artifact",
            "has_full_thumbnail_or_degenerate_bbox",
            "needs_refinement",
            "priority",
            "reasoning",
            "known_case_note",
            "thumbnail_path",
            "overlay_path",
        ],
    )
    _write_csv(
        output_root / "summary" / "detection_review_bboxes.csv",
        bbox_rows,
        [
            "task_id",
            "case_display",
            "bbox_id",
            "localization_quality",
            "excess_background",
            "is_near_full_thumbnail_box",
            "cuts_off_tissue",
            "multiple_cores_in_bbox",
            "artifact_false_positive",
            "suggested_action",
            "reasoning",
        ],
    )
    counts = {
        "results": len(results),
        "parse_ok": sum(1 for row in slide_rows if row["parse_ok"]),
        "errors": sum(1 for row in slide_rows if row["error"]),
        "needs_refinement": sum(1 for row in slide_rows if str(row["needs_refinement"]).lower() == "true"),
        "missed_tissue_core": sum(1 for row in slide_rows if str(row["missed_tissue_core"]).lower() == "true"),
        "full_thumbnail_or_degenerate": sum(
            1 for row in slide_rows if str(row["has_full_thumbnail_or_degenerate_bbox"]).lower() == "true"
        ),
        "bbox_reviews": len(bbox_rows),
    }
    _write_json(output_root / "summary" / "detection_review_summary.json", counts)
    write_review_pdf(output_root, results, slide_rows, bbox_rows)


def summarize_raw_overlay_review(
    output_root: Path,
    rotation: int,
    results: list[dict[str, Any]] | None = None,
) -> None:
    if results is None:
        results = _read_jsonl(output_root / "reviews" / f"raw_overlay_review_rot{rotation}_results.jsonl")
    slide_rows, bbox_rows = _flat_raw_overlay_review_rows(results)
    _write_csv(
        output_root / "summary" / f"raw_overlay_review_rot{rotation}_slides.csv",
        slide_rows,
        [
            "task_id",
            "case_display",
            "rotation",
            "bbox_count",
            "parse_ok",
            "error",
            "skip_reason",
            "overall_quality",
            "review_bbox_count",
            "reasoning",
            "thumbnail_path",
            "overlay_path",
        ],
    )
    _write_csv(
        output_root / "summary" / f"raw_overlay_review_rot{rotation}_bboxes.csv",
        bbox_rows,
        [
            "task_id",
            "case_display",
            "rotation",
            "bbox_id",
            "source_label",
            "tightness",
            "detection_signal",
            "reasoning",
            "bbox_thumbnail",
            "raw_box_2d",
            "area_ratio",
            "width_ratio",
            "height_ratio",
            "edge_touch_count",
            "geometry_near_full_thumbnail",
        ],
    )
    tightness_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    for row in bbox_rows:
        tightness = str(row.get("tightness", ""))
        signal = str(row.get("detection_signal", ""))
        tightness_counts[tightness] = tightness_counts.get(tightness, 0) + 1
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    counts = {
        "results": len(results),
        "parse_ok": sum(1 for row in slide_rows if row["parse_ok"]),
        "errors": sum(1 for row in slide_rows if row["error"]),
        "skipped": sum(1 for row in slide_rows if row["skip_reason"]),
        "bbox_reviews": len(bbox_rows),
        "tightness_counts": tightness_counts,
        "detection_signal_counts": signal_counts,
        "overall_quality_counts": {
            quality: sum(1 for row in slide_rows if row["overall_quality"] == quality)
            for quality in sorted({str(row["overall_quality"]) for row in slide_rows})
        },
    }
    _write_json(output_root / "summary" / f"raw_overlay_review_rot{rotation}_summary.json", counts)
    write_raw_overlay_review_pdf(output_root, rotation, results, slide_rows, bbox_rows)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, width: int, fill: str) -> int:
    x, y = xy
    line_height = int(font.size * 1.35) if hasattr(font, "size") else 18
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        y += 4
    return y


def _thumb(path: Path, max_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size)
    canvas = Image.new("RGB", max_size, "white")
    x = (max_size[0] - image.width) // 2
    y = (max_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def write_review_pdf(
    output_root: Path,
    results: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
    bbox_rows: list[dict[str, Any]],
) -> None:
    slide_by_task = {row["task_id"]: row for row in slide_rows}
    bboxes_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        bboxes_by_task.setdefault(row["task_id"], []).append(row)

    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(15)
    for result in results:
        task_id = result["task_id"]
        slide = slide_by_task.get(task_id, {})
        page = Image.new("RGB", (1800, 2200), "white")
        draw = ImageDraw.Draw(page)
        y = 30
        draw.text((40, y), result.get("case_display", task_id), font=title_font, fill="black")
        y += 48
        draw.text(
            (40, y),
            f"pass={slide.get('overall_pass')}  needs_refinement={slide.get('needs_refinement')}  "
            f"missed={slide.get('missed_tissue_core')}  priority={slide.get('priority')}",
            font=body_font,
            fill="black",
        )
        y += 36
        draw.text((40, y), f"Known note: {result.get('known_case_note', '')}", font=small_font, fill="#333333")
        y += 34
        source = _thumb(Path(result["thumbnail_path"]), (820, 420))
        overlay = _thumb(Path(result["overlay_path"]), (820, 420))
        page.paste(source, (40, y))
        page.paste(overlay, (930, y))
        y += 450
        draw.text((40, y), "Slide reasoning", font=body_font, fill="black")
        y += 30
        y = _draw_wrapped(draw, (40, y), slide.get("reasoning", ""), small_font, 150, "#111111")
        y += 10
        draw.text((40, y), "BBox reviews", font=body_font, fill="black")
        y += 30
        for bbox in bboxes_by_task.get(task_id, []):
            line = (
                f"{bbox.get('bbox_id')}: {bbox.get('localization_quality')} / "
                f"background={bbox.get('excess_background')} / "
                f"full_thumb={bbox.get('is_near_full_thumbnail_box')} / "
                f"action={bbox.get('suggested_action')} | {bbox.get('reasoning')}"
            )
            y = _draw_wrapped(draw, (60, y), line, small_font, 160, "#111111")
            if y > 2050:
                break
        pages.append(page)

    pdf_path = output_root / "visuals" / "detection_review_smoke.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def write_raw_overlay_review_pdf(
    output_root: Path,
    rotation: int,
    results: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
    bbox_rows: list[dict[str, Any]],
) -> None:
    slide_by_task = {row["task_id"]: row for row in slide_rows}
    bboxes_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        bboxes_by_task.setdefault(row["task_id"], []).append(row)

    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    for result in results:
        task_id = result["task_id"]
        slide = slide_by_task.get(task_id, {})
        rows = bboxes_by_task.get(task_id, [])
        page = Image.new("RGB", (1800, 2400), "white")
        draw = ImageDraw.Draw(page)
        y = 30
        draw.text((40, y), result.get("case_display", task_id), font=title_font, fill="black")
        y += 46
        header = (
            f"raw rot{rotation} | quality={slide.get('overall_quality')} | "
            f"boxes={slide.get('bbox_count')} | parsed={len(rows)} | "
            f"skip={slide.get('skip_reason')}"
        )
        draw.text((40, y), header, font=body_font, fill="black")
        y += 34
        reason = str(slide.get("reasoning", ""))
        y = _draw_wrapped(draw, (40, y), reason, small_font, 150, "#222222")
        y += 16

        overlay_raw = str(result.get("overlay_path", ""))
        overlay_path = Path(overlay_raw) if overlay_raw else None
        if overlay_path is not None and overlay_path.is_file():
            overlay = _thumb(overlay_path, (1680, 760))
            page.paste(overlay, (40, y))
            y += overlay.height + 24
        else:
            draw.text((40, y), "No raw orientation overlay available.", font=body_font, fill="#aa0000")
            y += 42

        tightness_counts: dict[str, int] = {}
        signal_counts: dict[str, int] = {}
        for row in rows:
            tightness_counts[str(row.get("tightness", ""))] = tightness_counts.get(str(row.get("tightness", "")), 0) + 1
            signal_counts[str(row.get("detection_signal", ""))] = signal_counts.get(
                str(row.get("detection_signal", "")),
                0,
            ) + 1
        draw.text(
            (40, y),
            f"tightness={tightness_counts}  detection_signal={signal_counts}",
            font=small_font,
            fill="#111111",
        )
        y += 32
        draw.text((40, y), "BBox reviews", font=body_font, fill="black")
        y += 28
        for bbox in rows:
            line = (
                f"{bbox.get('bbox_id')}: tightness={bbox.get('tightness')} / "
                f"signal={bbox.get('detection_signal')} / area={bbox.get('area_ratio')} | "
                f"{bbox.get('reasoning')}"
            )
            y = _draw_wrapped(draw, (60, y), line, small_font, 170, "#111111")
            if y > 2320:
                y = _draw_wrapped(
                    draw,
                    (60, y),
                    "... [truncated on PDF page; full bbox rows are in CSV]",
                    small_font,
                    170,
                    "#555555",
                )
                break
        pages.append(page)

    pdf_path = output_root / "visuals" / f"raw_overlay_review_rot{rotation}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def write_reproduction(output_root: Path, args: argparse.Namespace, tasks_path: Path, results_path: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    reproduction = f"""\
Stage 1 detection reviewer smoke test
====================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions
Manifest: {args.manifest.resolve()}
Task indices: {','.join(str(i) for i in args.indices)}

Command:
python scripts/stage1_detection_review_pilot.py run-detection-review \\
  --manifest {args.manifest.resolve()} \\
  --output-root {output_root} \\
  --indices {','.join(str(i) for i in args.indices)} \\
  --model {args.model} \\
  --max-concurrent {args.max_concurrent} \\
  --temperature {args.temperature}

Outputs:
- Tasks: {tasks_path}
- Raw/parsed results: {results_path}
- Slide summary: {output_root / 'summary' / 'detection_review_slides.csv'}
- Bbox summary: {output_root / 'summary' / 'detection_review_bboxes.csv'}
- PDF: {output_root / 'visuals' / 'detection_review_smoke.pdf'}

Notes:
- This is a flag-only reviewer test. No second-pass bbox refinement is run.
- Known-case notes are retained in local summaries only and are not sent to the VLM prompt.
- Full-thumbnail boxes are treated as the severe limit case of loose localization and also marked with is_near_full_thumbnail_box.
"""
    (output_root / "reproduction.txt").write_text(reproduction)


def write_raw_overlay_review_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    tasks_path: Path,
    results_path: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    reproduction = f"""\
Stage 1 raw-orientation overlay reviewer pass
============================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Prompt version: {RAW_OVERLAY_REVIEW_PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions
Manifest: {args.manifest.resolve()}
Detector orientation reviewed: rot{args.rotation}
Task indices: {','.join(str(i) for i in args.indices)}

Command:
python scripts/stage1_detection_review_pilot.py run-raw-overlay-review \\
  --manifest {args.manifest.resolve()} \\
  --output-root {output_root} \\
  --indices {','.join(str(i) for i in args.indices)} \\
  --rotation {args.rotation} \\
  --model {args.model} \\
  --max-concurrent {args.max_concurrent} \\
  --temperature {args.temperature}

Outputs:
- Tasks: {tasks_path}
- Raw/parsed results: {results_path}
- Slide summary: {output_root / 'summary' / f'raw_overlay_review_rot{args.rotation}_slides.csv'}
- Bbox summary: {output_root / 'summary' / f'raw_overlay_review_rot{args.rotation}_bboxes.csv'}
- Summary JSON: {output_root / 'summary' / f'raw_overlay_review_rot{args.rotation}_summary.json'}
- PDF: {output_root / 'visuals' / f'raw_overlay_review_rot{args.rotation}.pdf'}
- Raw overlays: {output_root / 'raw_overlays'}

Notes:
- This pass reviews only one raw detector orientation from `per_orientation_raw`.
- It intentionally does not use the postprocessed/merged Stage 1 bbox set.
- It does not review missed tissue, does not propose refined coordinates, and does not use pathology/control-tissue semantics.
- Each bbox is graded only for thumbnail-level tightness and signal-vs-noise.
"""
    (output_root / "reproduction.txt").write_text(reproduction)


def write_feedback_reproduction(output_dir: Path, args: argparse.Namespace, task: dict[str, Any]) -> None:
    reproduction = f"""\
Stage 1 feedback redetection experiment
=======================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Prompt version: {FEEDBACK_REDETECT_PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions
Manifest: {args.manifest.resolve()}
Case: {task['case_display']}

Command:
PYTHONPATH=/data2/vj724/python_deps/openai_py310:$PYTHONPATH \\
python scripts/stage1_detection_review_pilot.py run-feedback-redetect \\
  --manifest {args.manifest.resolve()} \\
  --output-root {args.output_root.resolve()} \\
  --index {args.index} \\
  --model {args.model} \\
  --run-label {args.run_label or _safe_slug(args.model)} \\
  --temperature {args.temperature}

Outputs:
- Task: {output_dir / 'feedback_redetect_task.json'}
- Result: {output_dir / 'feedback_redetect_result.json'}
- Overlay: {output_dir / 'feedback_redetect_overlay.png'}
- PDF: {output_dir / 'feedback_redetect_report.pdf'}

Notes:
- This is a second-pass detector call, not a reviewer call.
- The detector receives the source thumbnail, the original Stage 1 overlay, original bbox geometry, and the blind reviewer feedback text.
- The output coordinate convention requested from the model is normalized 0-1000 `[y_min, x_min, y_max, x_max]`.
"""
    (output_dir / "reproduction.txt").write_text(reproduction)


def write_model_loop_reproduction(output_dir: Path, args: argparse.Namespace, record: dict[str, Any]) -> None:
    reproduction = f"""\
Stage 1 per-model detect-review-redetect loop
=============================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Model: {args.model}
Manifest: {args.manifest.resolve()}
Case: {record['case_display']}

Command:
PYTHONPATH=/data2/vj724/python_deps/openai_py310:$PYTHONPATH \\
python scripts/stage1_detection_review_pilot.py run-model-loop \\
  --manifest {args.manifest.resolve()} \\
  --output-root {args.output_root.resolve()} \\
  --index {args.index} \\
  --model {args.model} \\
  --run-label {args.run_label or _safe_slug(args.model)} \\
  --temperature {args.temperature}

Outputs:
- First-pass task/result/overlay:
  {output_dir / 'first_pass_task.json'}
  {output_dir / 'first_pass_result.json'}
  {output_dir / 'first_pass_overlay.png'}
- Same-model review task/result:
  {output_dir / 'review_task.json'}
  {output_dir / 'review_result.json'}
- Same-model feedback redetect task/result/overlay:
  {output_dir / 'feedback_redetect_task.json'}
  {output_dir / 'feedback_redetect_result.json'}
  {output_dir / 'feedback_redetect_overlay.png'}
- Combined result:
  {output_dir / 'model_loop_result.json'}
- Report:
  {output_dir / 'model_loop_report.pdf'}

Summary at generation:
- First-pass detections: {len(record.get('first_pass_detections', []))}
- Review missed_tissue_core: {_slide_review(record.get('review_parsed', {})).get('missed_tissue_core', '')}
- Feedback-redetect detections: {len(record.get('feedback_redetect_detections', []))}
"""
    (output_dir / "reproduction.txt").write_text(reproduction)


def cmd_summarize_detection_review(args: argparse.Namespace) -> int:
    summarize_detection_review(args.output_root.resolve())
    print(json.dumps({"output_root": str(args.output_root.resolve())}, indent=2))
    return 0


def parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-detection-review", help="Run paid VLM review for selected Stage 1 detections.")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--indices", type=parse_indices, default=list(DEFAULT_KNOWN_CASE_INDICES))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--api-base", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=1800)
    run.add_argument("--max-concurrent", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=run_detection_review)

    raw = sub.add_parser(
        "run-raw-overlay-review",
        help="Run paid VLM review over one raw detector orientation without TTA merge postprocessing.",
    )
    raw.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    raw.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "raw_overlay_review_v1")
    raw.add_argument("--indices", type=parse_indices, default=list(range(1, 101)))
    raw.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    raw.add_argument("--model", default=DEFAULT_MODEL)
    raw.add_argument("--api-base", default=None)
    raw.add_argument("--api-key", default=None)
    raw.add_argument("--temperature", type=float, default=0.0)
    raw.add_argument("--max-tokens", type=int, default=1400)
    raw.add_argument("--max-concurrent", type=int, default=8)
    raw.add_argument("--dry-run", action="store_true")
    raw.set_defaults(func=run_raw_overlay_review)

    feedback = sub.add_parser(
        "run-feedback-redetect",
        help="Rerun detector on one case using thumbnail, prior overlay, and reviewer feedback.",
    )
    feedback.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    feedback.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    feedback.add_argument("--index", type=int, default=70)
    feedback.add_argument("--model", default=DEFAULT_MODEL)
    feedback.add_argument(
        "--run-label",
        default=None,
        help="Optional subdirectory label for comparing multiple models on the same case.",
    )
    feedback.add_argument("--api-base", default=None)
    feedback.add_argument("--api-key", default=None)
    feedback.add_argument("--temperature", type=float, default=0.0)
    feedback.add_argument("--max-tokens", type=int, default=1200)
    feedback.add_argument("--dry-run", action="store_true")
    feedback.set_defaults(func=run_feedback_redetect)

    loop = sub.add_parser(
        "run-model-loop",
        help="Run first-pass detection, same-model review, and same-model feedback redetection.",
    )
    loop.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    loop.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    loop.add_argument("--index", type=int, default=70)
    loop.add_argument("--model", default=DEFAULT_MODEL)
    loop.add_argument("--run-label", default=None)
    loop.add_argument("--api-base", default=None)
    loop.add_argument("--api-key", default=None)
    loop.add_argument("--temperature", type=float, default=0.0)
    loop.add_argument("--max-tokens", type=int, default=1200)
    loop.add_argument("--review-max-tokens", type=int, default=1800)
    loop.add_argument("--dry-run", action="store_true")
    loop.set_defaults(func=run_model_loop)

    refresh = sub.add_parser(
        "refresh-model-loops",
        help="Reparse saved model-loop raw outputs and rebuild overlays/PDFs without new VLM calls.",
    )
    refresh.add_argument("--run-dir", type=Path, action="append", required=True)
    refresh.add_argument("--comparison-pdf", type=Path, default=None)
    refresh.set_defaults(func=cmd_refresh_model_loops)

    summarize = sub.add_parser("summarize-detection-review", help="Regenerate summaries and PDF from existing results.")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    summarize.set_defaults(func=cmd_summarize_detection_review)

    summarize_raw = sub.add_parser(
        "summarize-raw-overlay-review",
        help="Regenerate raw-orientation overlay reviewer summaries and PDF from existing results.",
    )
    summarize_raw.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "raw_overlay_review_v1")
    summarize_raw.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    summarize_raw.set_defaults(
        func=lambda args: (
            summarize_raw_overlay_review(args.output_root.resolve(), args.rotation) or 0
        )
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
