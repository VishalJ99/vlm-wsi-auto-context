#!/usr/bin/env python3
"""Refine Stage 7 final boxes using high-thinking shrinkability text plus crop overlays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    _chat_with_images,
    _extract_json_object,
    _font,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHRINKABILITY_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage7_case99_final_box_shrinkability_any_sides_primary_flash_v1"
)
DEFAULT_HIGH_RESULTS = (
    DEFAULT_SHRINKABILITY_ROOT
    / "high_thinking/reviews/stage6_final_box_shrinkability_high_thinking.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage7_case99_bbox_refinement_from_high_shrinkability_flash_v1"
)
DEFAULT_FEEDBACK_NORMALIZED_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage7_case99_bbox_refinement_feedback_normalized_flash_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
TICKET = "PER-207"


BASE_PROMPT = """\
You are refining one red bounding box around primary tissue in a high-resolution whole-slide crop.

Inputs:
- Image 1 is the crop with the current bounding box drawn in red. There is no numeric label.
- The current box is in local crop pixel coordinates: [x_min, y_min, x_max, y_max] = {current_box_xyxy}
- The image size is width={image_width}, height={image_height} pixels.
- The previous high-thinking side-shrinkability review said:
<high_thinking_review>
{high_thinking_review}
</high_thinking_review>

Task:
Produce one refined bounding box for the same primary tissue target.
Use the high-thinking review as guidance about which sides can move inward, but verify visually from the image.
Preserve all visible extremities of the primary tissue the current box is focused on.
If a side is already tight or clipping, do not move it inward.
If the current box clips primary tissue, you may move that side outward only as much as needed.
Return coordinates in local crop pixel coordinates as integers using [x_min, y_min, x_max, y_max].
Coordinates must be within image bounds.

Output JSON only:
{{
  "current_box_xyxy": [x_min, y_min, x_max, y_max],
  "refined_box_xyxy": [x_min, y_min, x_max, y_max],
  "side_changes": {{"left": "keep|inward|outward", "top": "keep|inward|outward", "right": "keep|inward|outward", "bottom": "keep|inward|outward"}},
  "reason": "short visual reason"
}}
"""

FEEDBACK_NORMALIZED_PROMPT = """\
Update the bounding box shown in the image given the following feedback

{feedback}

Output the bounding box coordinates as JSON in normalized 0-1000 coordinates:

{{"box_2d": [y_min, x_min, y_max, x_max]}}
"""


def _prompt_version(prompt_mode: str) -> str:
    if prompt_mode == "feedback-normalized":
        return "stage7_bbox_refinement_feedback_normalized_2026-05-29"
    return "stage7_bbox_refinement_from_high_shrinkability_2026-05-29"


def _default_output_root_for_mode(prompt_mode: str) -> Path:
    if prompt_mode == "feedback-normalized":
        return DEFAULT_FEEDBACK_NORMALIZED_OUTPUT_ROOT
    return DEFAULT_OUTPUT_ROOT


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, --api-key-stdin, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return args.api_base or "https://openrouter.ai/api/v1", api_key


def _draw_current_box_no_enum(crop_path: Path, current_box: list[int], output_path: Path) -> None:
    image = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, max(image.size) // 180)
    draw.rectangle(tuple(current_box), outline="#e31a1c", width=line_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _save_jpeg(image_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92, optimize=True)


def _draw_refinement_overlay(
    crop_path: Path,
    current_box: list[int],
    refined_box: list[int] | None,
    output_path: Path,
) -> None:
    image = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(3, max(image.size) // 180)
    draw.rectangle(tuple(current_box), outline="#e31a1c", width=line_width)
    if refined_box:
        inset_width = max(2, line_width - 1)
        draw.rectangle(tuple(refined_box), outline="#00a65a", width=inset_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _coerce_box_xyxy(value: Any, width: int, height: int) -> tuple[list[int] | None, list[str], list[Any]]:
    warnings: list[str] = []
    raw: list[Any] = value if isinstance(value, list) else []
    if not isinstance(value, list) or len(value) != 4:
        return None, ["missing_or_invalid_box"], raw
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None, ["non_numeric_box"], raw
    clipped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    if clipped != [x1, y1, x2, y2]:
        warnings.append("box_clipped_to_image_bounds")
    ix1, iy1, ix2, iy2 = [int(round(v)) for v in clipped]
    if ix2 <= ix1 or iy2 <= iy1:
        warnings.append("degenerate_box")
        return None, warnings, raw
    return [ix1, iy1, ix2, iy2], warnings, raw


def _coerce_box_yxyx_normalized(value: Any) -> tuple[list[float] | None, list[str], list[Any]]:
    warnings: list[str] = []
    raw: list[Any] = value if isinstance(value, list) else []
    if not isinstance(value, list) or len(value) != 4:
        return None, ["missing_or_invalid_box_2d"], raw
    try:
        y1, x1, y2, x2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None, ["non_numeric_box_2d"], raw
    clipped = [
        max(0.0, min(1000.0, y1)),
        max(0.0, min(1000.0, x1)),
        max(0.0, min(1000.0, y2)),
        max(0.0, min(1000.0, x2)),
    ]
    if clipped != [y1, x1, y2, x2]:
        warnings.append("box_2d_clipped_to_0_1000")
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        warnings.append("degenerate_box_2d")
        return None, warnings, raw
    return [round(v, 3) for v in clipped], warnings, raw


def _normalized_yxyx_to_crop_xyxy(box_2d: list[float], width: int, height: int) -> list[int]:
    y1, x1, y2, x2 = [float(v) for v in box_2d]
    return [
        int(round(x1 / 1000.0 * width)),
        int(round(y1 / 1000.0 * height)),
        int(round(x2 / 1000.0 * width)),
        int(round(y2 / 1000.0 * height)),
    ]


def _find_box(parsed: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = parsed.get(key)
        if value is not None:
            return value
    return None


def _find_box_in_raw_text(raw: str, keys: tuple[str, ...]) -> list[float] | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*\[([^\]]+)'
        match = re.search(pattern, raw)
        if not match:
            continue
        values = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        if len(values) >= 4:
            return [float(value) for value in values[:4]]
    return None


def _side_change(current: list[int], refined: list[int]) -> dict[str, str]:
    cx1, cy1, cx2, cy2 = current
    rx1, ry1, rx2, ry2 = refined

    def compare(delta: int, inward_positive: bool) -> str:
        if abs(delta) <= 1:
            return "keep"
        moved_inward = delta > 0 if inward_positive else delta < 0
        return "inward" if moved_inward else "outward"

    return {
        "left": compare(rx1 - cx1, True),
        "top": compare(ry1 - cy1, True),
        "right": compare(rx2 - cx2, False),
        "bottom": compare(ry2 - cy2, False),
    }


def _box_metrics(current: list[int], refined: list[int]) -> dict[str, Any]:
    cx1, cy1, cx2, cy2 = current
    rx1, ry1, rx2, ry2 = refined
    current_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    refined_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    return {
        "current_width": cx2 - cx1,
        "current_height": cy2 - cy1,
        "current_area": current_area,
        "refined_width": rx2 - rx1,
        "refined_height": ry2 - ry1,
        "refined_area": refined_area,
        "area_ratio_refined_over_current": round(refined_area / current_area, 4) if current_area else None,
        "delta_left_px": rx1 - cx1,
        "delta_top_px": ry1 - cy1,
        "delta_right_px": rx2 - cx2,
        "delta_bottom_px": ry2 - cy2,
    }


def _load_metadata(task: dict[str, Any]) -> dict[str, Any]:
    metadata_path = _resolve_path(task["metadata_path"])
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return {}


def _wsi_size_from_metadata(metadata: dict[str, Any]) -> tuple[int, int] | None:
    pyramid = metadata.get("pyramid") if isinstance(metadata, dict) else {}
    dimensions = pyramid.get("level_dimensions") if isinstance(pyramid, dict) else None
    if isinstance(dimensions, list) and dimensions:
        first = dimensions[0]
        if isinstance(first, list) and len(first) == 2:
            return int(first[0]), int(first[1])
    return None


def _xyxy_crop_to_level0(task: dict[str, Any], crop_size: tuple[int, int], box: list[int]) -> list[int]:
    padded = task.get("padded_bbox_level0")
    if not isinstance(padded, list) or len(padded) != 4:
        raise ValueError(f"Missing padded_bbox_level0 for {task.get('task_id')}")
    px1, py1, px2, py2 = [float(v) for v in padded]
    width, height = crop_size
    x1, y1, x2, y2 = [float(v) for v in box]
    scale_x = (px2 - px1) / float(max(1, width))
    scale_y = (py2 - py1) / float(max(1, height))
    return [
        int(round(px1 + x1 * scale_x)),
        int(round(py1 + y1 * scale_y)),
        int(round(px1 + x2 * scale_x)),
        int(round(py1 + y2 * scale_y)),
    ]


def _level0_xyxy_to_normalized_yxyx(level0_box: list[int], wsi_size: tuple[int, int]) -> list[float]:
    width, height = wsi_size
    x1, y1, x2, y2 = [float(v) for v in level0_box]
    return [
        round(y1 / float(max(1, height)) * 1000.0, 3),
        round(x1 / float(max(1, width)) * 1000.0, 3),
        round(y2 / float(max(1, height)) * 1000.0, 3),
        round(x2 / float(max(1, width)) * 1000.0, 3),
    ]


def _parse_refinement(raw: str, task: dict[str, Any], crop_size: tuple[int, int]) -> dict[str, Any]:
    width, height = crop_size
    parsed = _extract_json_object(raw)
    valid_json_object = isinstance(parsed, dict) and "raw_text" not in parsed
    if not isinstance(parsed, dict):
        parsed = {"raw_text": raw}
    prompt_mode = task.get("prompt_mode", "xyxy")

    if prompt_mode == "feedback-normalized":
        box_2d_value = _find_box(parsed, ("box_2d", "bbox_2d", "box", "bbox")) if valid_json_object else None
        fallback_warnings: list[str] = []
        if box_2d_value is None:
            box_2d_value = _find_box_in_raw_text(raw, ("box_2d", "bbox_2d", "box", "bbox"))
        if not valid_json_object:
            fallback_warnings.append("json_object_not_found_or_truncated")
        box_2d, box_warnings, raw_box_2d = _coerce_box_yxyx_normalized(box_2d_value)
        warnings = fallback_warnings + box_warnings
        expected_current = [int(v) for v in task["source_bbox_in_crop"]]
        if box_2d is None:
            return {
                "parsed_response": parsed,
                "parse_status": "invalid_box_2d",
                "parse_warnings": warnings,
                "current_box_xyxy": None,
                "expected_current_box_xyxy": expected_current,
                "refined_box_xyxy": None,
                "refined_box_input_yxyx_normalized": None,
                "raw_refined_box_yxyx_normalized": raw_box_2d,
                "side_changes_reported": {},
                "side_changes_measured": {},
                "metrics": {},
            }
        refined_box = _normalized_yxyx_to_crop_xyxy(box_2d, width, height)
        measured = _side_change(expected_current, refined_box)
        if valid_json_object and not warnings:
            parse_status = "ok"
        elif valid_json_object:
            parse_status = "ok_with_warnings"
        else:
            parse_status = "coords_recovered_from_truncated_json"
        return {
            "parsed_response": parsed,
            "parse_status": parse_status,
            "parse_warnings": warnings,
            "current_box_xyxy": None,
            "expected_current_box_xyxy": expected_current,
            "refined_box_xyxy": refined_box,
            "refined_box_input_yxyx_normalized": box_2d,
            "raw_refined_box_yxyx_normalized": raw_box_2d,
            "side_changes_reported": {},
            "side_changes_measured": measured,
            "reason": parsed.get("reason", ""),
            "metrics": _box_metrics(expected_current, refined_box),
        }

    current_value = _find_box(parsed, ("current_box_xyxy", "current_bbox_xyxy")) if valid_json_object else None
    refined_value = (
        _find_box(parsed, ("refined_box_xyxy", "refined_bbox_xyxy", "refined_box", "bbox_xyxy"))
        if valid_json_object
        else None
    )
    fallback_warnings: list[str] = []
    if current_value is None:
        current_value = _find_box_in_raw_text(raw, ("current_box_xyxy", "current_bbox_xyxy"))
    if refined_value is None:
        refined_value = _find_box_in_raw_text(raw, ("refined_box_xyxy", "refined_bbox_xyxy", "refined_box", "bbox_xyxy"))
    if not valid_json_object:
        fallback_warnings.append("json_object_not_found_or_truncated")

    current_box, current_warnings, raw_current = _coerce_box_xyxy(current_value, width, height)
    refined_box, refined_warnings, raw_refined = _coerce_box_xyxy(refined_value, width, height)
    expected_current = [int(v) for v in task["source_bbox_in_crop"]]
    warnings = fallback_warnings + current_warnings + refined_warnings
    if current_box is not None and current_box != expected_current:
        warnings.append("reported_current_box_differs_from_task_current_box")
    if refined_box is None:
        return {
            "parsed_response": parsed,
            "parse_status": "invalid_refined_box",
            "parse_warnings": warnings,
            "current_box_xyxy": current_box,
            "expected_current_box_xyxy": expected_current,
            "refined_box_xyxy": None,
            "refined_box_input_yxyx_normalized": None,
            "raw_current_box_xyxy": raw_current,
            "raw_refined_box_xyxy": raw_refined,
            "side_changes_reported": parsed.get("side_changes", {}),
            "side_changes_measured": {},
            "metrics": {},
        }

    measured = _side_change(expected_current, refined_box)
    if valid_json_object and not warnings:
        parse_status = "ok"
    elif valid_json_object:
        parse_status = "ok_with_warnings"
    else:
        parse_status = "coords_recovered_from_truncated_json" if refined_box else "parse_error"
    return {
        "parsed_response": parsed,
        "parse_status": parse_status,
        "parse_warnings": warnings,
        "current_box_xyxy": current_box,
        "expected_current_box_xyxy": expected_current,
        "refined_box_xyxy": refined_box,
        "refined_box_input_yxyx_normalized": None,
        "raw_current_box_xyxy": raw_current,
        "raw_refined_box_xyxy": raw_refined,
        "side_changes_reported": parsed.get("side_changes", {}),
        "side_changes_measured": measured,
        "reason": parsed.get("reason", ""),
        "metrics": _box_metrics(expected_current, refined_box),
    }


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    high_rows = _read_jsonl(args.high_results.resolve())
    selected_rows = [
        row
        for row in high_rows
        if (not args.indices or int(row.get("case_index", -1)) in set(args.indices))
        and (not row.get("error"))
    ]
    if not selected_rows:
        raise SystemExit(f"No usable high-thinking shrinkability rows found in {args.high_results}")

    tasks: list[dict[str, Any]] = []
    for row in selected_rows:
        crop_path = _resolve_path(row["crop_path"])
        if not crop_path.exists():
            raise SystemExit(f"Crop does not exist: {crop_path}")
        with Image.open(crop_path) as image:
            crop_size = image.size
        current_box = [int(v) for v in row["source_bbox_in_crop"]]
        case_slug = row.get("case_slug") or _safe_slug(f"{int(row['case_index']):03d}_{row['case_display']}")
        final_box_index = int(row["final_box_index"])
        region_dir = args.output_root / "inputs/cases" / case_slug / "regions" / f"{final_box_index:02d}"
        no_enum_overlay = region_dir / "current_box_overlay_no_enum.png"
        vlm_image = region_dir / "current_box_overlay_no_enum_vlm.jpg"
        _draw_current_box_no_enum(crop_path, current_box, no_enum_overlay)
        _save_jpeg(no_enum_overlay, vlm_image)
        high_thinking_review = row.get("raw_response", "").strip()
        if args.prompt_mode == "feedback-normalized":
            prompt = FEEDBACK_NORMALIZED_PROMPT.format(feedback=high_thinking_review)
        else:
            prompt = BASE_PROMPT.format(
                current_box_xyxy=json.dumps(current_box),
                image_width=crop_size[0],
                image_height=crop_size[1],
                high_thinking_review=high_thinking_review,
            )
        task = {
            "task_id": f"stage7_bbox_refinement_{int(row['case_index']):03d}_{final_box_index:02d}",
            "case_index": int(row["case_index"]),
            "case_display": row["case_display"],
            "case_slug": case_slug,
            "final_box_index": final_box_index,
            "source_shrinkability_task_id": row.get("task_id", ""),
            "source_high_thinking_results": str(args.high_results.resolve()),
            "high_thinking_raw_response": row.get("raw_response", ""),
            "crop_path": str(crop_path),
            "current_overlay_no_enum_path": str(no_enum_overlay.resolve()),
            "vlm_image_path": str(vlm_image.resolve()),
            "source_bbox_in_crop": current_box,
            "crop_size": list(crop_size),
            "box_2d_yxyx_normalized": row.get("box_2d_yxyx_normalized", []),
            "source_bbox_level0": row.get("source_bbox_level0", []),
            "padded_bbox_level0": row.get("padded_bbox_level0", []),
            "metadata_path": str(_resolve_path(row["metadata_path"])),
            "wsi_path": row.get("wsi_path", ""),
            "wsi_reader": row.get("wsi_reader", ""),
            "final_overlay_path": row.get("final_overlay_path", ""),
            "prompt": prompt,
            "prompt_mode": args.prompt_mode,
            "prompt_version": _prompt_version(args.prompt_mode),
            "model": args.model,
            "created_at": _timestamp(),
        }
        _write_json(region_dir / "task.json", {key: value for key, value in task.items() if key != "prompt"})
        tasks.append(task)
    tasks.sort(key=lambda row: (int(row["case_index"]), int(row["final_box_index"])))
    _write_jsonl(args.output_root / "tasks/stage7_bbox_refinement_tasks.jsonl", tasks)
    return tasks


def _run_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    record = {
        key: task[key]
        for key in (
            "task_id",
            "case_index",
            "case_display",
            "case_slug",
            "final_box_index",
            "source_shrinkability_task_id",
            "source_high_thinking_results",
            "crop_path",
            "current_overlay_no_enum_path",
            "vlm_image_path",
            "source_bbox_in_crop",
            "crop_size",
            "box_2d_yxyx_normalized",
            "source_bbox_level0",
            "padded_bbox_level0",
            "metadata_path",
            "wsi_path",
            "wsi_reader",
            "final_overlay_path",
            "prompt_mode",
            "prompt_version",
            "model",
            "created_at",
        )
    }
    record.update(
        {
            "raw_response": "",
            "parsed_response": {},
            "error": "",
            "usage": {},
            "response_model": "",
            "parse_status": "not_run",
            "refined_box_xyxy": None,
            "refined_box_input_yxyx_normalized": None,
            "refined_box_level0_xyxy": None,
            "refined_box_yxyx_normalized": None,
            "refined_overlay_path": "",
        }
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=task["prompt"],
            image_paths=[Path(task["vlm_image_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        record.update({"raw_response": raw, "usage": usage, "response_model": response_model})
        _postprocess_record_from_raw(record, task, args)
    except Exception as exc:
        record["error"] = repr(exc)
        record["parse_status"] = "error"
    return record


def _postprocess_record_from_raw(record: dict[str, Any], task: dict[str, Any], args: argparse.Namespace) -> None:
    raw = record.get("raw_response") or ""
    parsed = _parse_refinement(raw, task, tuple(task["crop_size"]))
    record.update(parsed)
    record["source_bbox_in_crop"] = task["source_bbox_in_crop"]
    record["crop_size"] = task["crop_size"]
    record["current_overlay_no_enum_path"] = task["current_overlay_no_enum_path"]
    record["prompt_mode"] = task.get("prompt_mode", "xyxy")
    record["prompt_version"] = task.get("prompt_version", _prompt_version(record["prompt_mode"]))
    refined_box = parsed.get("refined_box_xyxy")
    metadata = _load_metadata(task)
    wsi_size = _wsi_size_from_metadata(metadata)
    overlay_path = (
        args.output_root
        / "outputs/cases"
        / task["case_slug"]
        / "regions"
        / f"{int(task['final_box_index']):02d}"
        / "current_vs_refined_overlay.png"
    )
    if refined_box:
        refined_level0 = _xyxy_crop_to_level0(task, tuple(task["crop_size"]), refined_box)
        record["refined_box_level0_xyxy"] = refined_level0
        if wsi_size:
            record["refined_box_yxyx_normalized"] = _level0_xyxy_to_normalized_yxyx(refined_level0, wsi_size)
        _draw_refinement_overlay(
            Path(task["crop_path"]),
            [int(v) for v in task["source_bbox_in_crop"]],
            refined_box,
            overlay_path,
        )
    else:
        record["refined_box_level0_xyxy"] = None
        record["refined_box_yxyx_normalized"] = None
        _draw_refinement_overlay(
            Path(task["crop_path"]),
            [int(v) for v in task["source_bbox_in_crop"]],
            None,
            overlay_path,
        )
    record["refined_overlay_path"] = str(overlay_path.resolve())


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_height: int | None = None,
) -> int:
    x, y = xy
    line_height = line_height or (int(font.size * 1.35) if hasattr(font, "size") else 22)
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        y += 4
    return y


def _raw_excerpt(value: Any, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated in PDF; full text is in JSONL]"


def _draw_cover(tasks: list[dict[str, Any]], args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(42)
    header = _font(29)
    body = _font(22)
    small = _font(18)
    y = 60
    draw.text((70, y), "Stage 7 Bbox Coordinate Refinement Probe", font=title, fill="black")
    y += 68
    draw.text((70, y), f"model={args.model} | reasoning={args.reasoning_effort} | boxes={len(tasks)}", font=body, fill="#222222")
    y += 42
    draw.text((70, y), f"prompt_mode={args.prompt_mode} | prompt_version={_prompt_version(args.prompt_mode)}", font=small, fill="#333333")
    y += 58
    draw.text((70, y), "Input Contract", font=header, fill="black")
    y += 38
    if args.prompt_mode == "feedback-normalized":
        text = (
            "Each call receives one no-enumeration high-resolution crop overlay: the current Stage 7 final box is drawn as a red rectangle only. "
            "The text prompt is the user's simplified feedback prompt, with the high-thinking shrinkability answer inserted as the feedback."
        )
    else:
        text = (
            "Each call receives one no-enumeration high-resolution crop overlay: the current Stage 7 final box is drawn as a red rectangle only. "
            "The text prompt includes the high-thinking shrinkability answer from the prior run and asks for one refined local crop-pixel xyxy box."
        )
    y = _draw_wrapped(draw, (95, y), text, 145, body)
    y += 36
    draw.text((70, y), "Output Contract", font=header, fill="black")
    y += 38
    if args.prompt_mode == "feedback-normalized":
        text = (
            "The model output is parsed as JSON with box_2d in normalized 0-1000 yxyx coordinates relative to the input crop image. "
            "The parsed box_2d is converted to local crop-pixel xyxy, then projected back to level-0 WSI xyxy and WSI-normalized yxyx."
        )
    else:
        text = (
            "The model output is parsed as JSON with current_box_xyxy, refined_box_xyxy, side_changes, and reason. "
            "Parsed local crop-pixel xyxy coordinates are mechanically projected back to level-0 WSI xyxy and normalized yxyx 0-1000 coordinates."
        )
    y = _draw_wrapped(draw, (95, y), text, 145, body)
    y += 36
    draw.text((70, y), "Boxes", font=header, fill="black")
    y += 38
    for task in tasks:
        y = _draw_wrapped(
            draw,
            (95, y),
            f"{task['case_index']:03d} final box {task['final_box_index']:02d} | {task['case_display']}",
            150,
            small,
            "#111111",
            24,
        )
    return page


def _draw_page(task: dict[str, Any], result: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(36)
    header = _font(26)
    body = _font(19)
    small = _font(16)
    y = 48
    draw.text(
        (60, y),
        f"{task['case_index']:03d} final box {task['final_box_index']:02d} | bbox refinement",
        font=title,
        fill="black",
    )
    y += 46
    y = _draw_wrapped(draw, (60, y), task["case_display"], 160, body, "#222222", 24)
    y += 14
    draw.text((60, y), "Input overlay sent to VLM (red, no enumeration)", font=header, fill="black")
    draw.text((1240, y), "Parsed refinement overlay (red=current, green=refined)", font=header, fill="black")
    y += 36
    input_overlay = _thumb(Path(task["current_overlay_no_enum_path"]), (1120, 880))
    refined_overlay = _thumb(Path(result["refined_overlay_path"]), (1120, 880)) if result.get("refined_overlay_path") else Image.new("RGB", (1120, 880), "white")
    page.paste(input_overlay, (60, y))
    page.paste(refined_overlay, (1240, y))
    y += 920
    draw.text((60, y), "Coordinate Output", font=header, fill="black")
    y += 34
    summary = {
        "parse_status": result.get("parse_status"),
        "error": result.get("error"),
        "current_box_xyxy": result.get("source_bbox_in_crop"),
        "refined_box_input_yxyx_normalized": result.get("refined_box_input_yxyx_normalized"),
        "refined_box_xyxy": result.get("refined_box_xyxy"),
        "refined_box_yxyx_normalized": result.get("refined_box_yxyx_normalized"),
        "side_changes_measured": result.get("side_changes_measured"),
        "metrics": result.get("metrics"),
    }
    y = _draw_wrapped(draw, (80, y), json.dumps(summary, sort_keys=True), 155, small, "#111111", 21)
    y += 24
    draw.text((60, y), "High-thinking shrinkability text supplied", font=header, fill="black")
    y += 34
    y = _draw_wrapped(draw, (80, y), _raw_excerpt(task.get("high_thinking_raw_response", ""), 1400), 155, small, "#111111", 21)
    y += 24
    draw.text((60, y), "Raw bbox-refinement model output", font=header, fill="black")
    y += 34
    _draw_wrapped(draw, (80, y), _raw_excerpt(result.get("raw_response", ""), 1800), 155, small, "#111111", 21)
    return page


def _write_pdf(output_root: Path, tasks: list[dict[str, Any]], results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    by_task = {row["task_id"]: row for row in results}
    pages = [_draw_cover(tasks, args)]
    for task in tasks:
        pages.append(_draw_page(task, by_task.get(task["task_id"], {})))
    pdf_name = (
        "stage7_case99_bbox_refinement_feedback_normalized.pdf"
        if args.prompt_mode == "feedback-normalized"
        else "stage7_case99_bbox_refinement_from_high_shrinkability.pdf"
    )
    pdf_path = output_root / "visuals" / pdf_name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_summary(output_root: Path, tasks: list[dict[str, Any]], results: list[dict[str, Any]], pdf_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in results:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                "case_index": row.get("case_index"),
                "case_display": row.get("case_display"),
                "final_box_index": row.get("final_box_index"),
                "prompt_mode": row.get("prompt_mode"),
                "parse_status": row.get("parse_status"),
                "error": row.get("error"),
                "current_box_xyxy": json.dumps(row.get("source_bbox_in_crop")),
                "refined_box_input_yxyx_normalized": json.dumps(row.get("refined_box_input_yxyx_normalized")),
                "refined_box_xyxy": json.dumps(row.get("refined_box_xyxy")),
                "refined_box_yxyx_normalized": json.dumps(row.get("refined_box_yxyx_normalized")),
                "side_changes_measured": json.dumps(row.get("side_changes_measured")),
                "area_ratio_refined_over_current": metrics.get("area_ratio_refined_over_current"),
                "delta_left_px": metrics.get("delta_left_px"),
                "delta_top_px": metrics.get("delta_top_px"),
                "delta_right_px": metrics.get("delta_right_px"),
                "delta_bottom_px": metrics.get("delta_bottom_px"),
                "reason": row.get("reason", ""),
                "current_overlay_no_enum_path": row.get("current_overlay_no_enum_path"),
                "refined_overlay_path": row.get("refined_overlay_path"),
            }
        )
    csv_path = output_root / "summary/stage7_bbox_refinement_summary.csv"
    _write_csv(
        csv_path,
        rows,
        [
            "case_index",
            "case_display",
            "final_box_index",
            "prompt_mode",
            "parse_status",
            "error",
            "current_box_xyxy",
            "refined_box_input_yxyx_normalized",
            "refined_box_xyxy",
            "refined_box_yxyx_normalized",
            "side_changes_measured",
            "area_ratio_refined_over_current",
            "delta_left_px",
            "delta_top_px",
            "delta_right_px",
            "delta_bottom_px",
            "reason",
            "current_overlay_no_enum_path",
            "refined_overlay_path",
        ],
    )
    summary = {
        "created_at": _timestamp(),
        "ticket": TICKET,
        "git_commit": _repo_git_commit(),
        "prompt_mode": args.prompt_mode,
        "prompt_version": _prompt_version(args.prompt_mode),
        "source_high_thinking_results": str(args.high_results.resolve()),
        "output_root": str(output_root.resolve()),
        "tasks": len(tasks),
        "results": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "parse_status_counts": {
            status: sum(1 for row in results if row.get("parse_status") == status)
            for status in sorted({str(row.get("parse_status")) for row in results})
        },
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_concurrent": args.max_concurrent,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "known_usage_cost_if_reported": round(
            sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in results),
            6,
        ),
        "results_jsonl": str((output_root / "reviews/stage7_bbox_refinement_results.jsonl").resolve()),
        "summary_csv": str(csv_path.resolve()),
        "comparison_pdf": str(pdf_path.resolve()),
    }
    _write_json(output_root / "summary/stage7_bbox_refinement_summary.json", summary)
    return summary


def _write_reproduction(output_root: Path, args: argparse.Namespace, tasks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    generation_command_parts = [
        "python",
        "scripts/stage7_bbox_refinement_from_shrinkability.py",
        "--high-results",
        str(args.high_results.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--prompt-mode",
        args.prompt_mode,
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.indices:
        generation_command_parts.extend(["--indices", *(str(v) for v in args.indices)])
    generation_command = " ".join(shlex.quote(part) for part in generation_command_parts)
    last_command_parts = list(generation_command_parts)
    if args.reuse_existing:
        last_command_parts.append("--reuse-existing")
    last_command = " ".join(shlex.quote(part) for part in last_command_parts)
    text = f"""\
Stage 7 bbox refinement from high-thinking shrinkability
=======================================================

Created: {_timestamp()}
Ticket: {TICKET}
Git commit: {_repo_git_commit()}
Prompt mode: {args.prompt_mode}
Prompt version: {_prompt_version(args.prompt_mode)}
Model: {args.model}
Reasoning effort: {args.reasoning_effort}
Backend: OpenRouter-compatible chat completions
Last invocation reused existing model outputs: {args.reuse_existing}

Objective:
Test whether a VLM can convert the prior high-thinking shrinkability assessment
and a no-enumeration crop detection overlay into explicit refined bounding-box
coordinates.

Input policy:
- Source high-thinking shrinkability JSONL: {args.high_results.resolve()}
- Each row supplies a 30%-padded high-resolution WSI crop, the current Stage 7
  final box in local crop-pixel xyxy coordinates, and the high-thinking textual
  answer about which sides can shrink.
- The model image input is a regenerated crop overlay with only the red current
  box. Numeric enumeration was intentionally removed.
- The model text input includes the high-thinking answer. For
  `feedback-normalized`, the prompt is the simplified feedback prompt and asks
  for one normalized 0-1000 `box_2d`; otherwise it asks for one refined local
  crop-pixel xyxy box.

Output policy:
- For `feedback-normalized`, parsed image-normalized 0-1000 yxyx boxes are
  converted to local crop-pixel xyxy first. Parsed or converted local crop-pixel
  xyxy boxes are then projected to level-0 WSI xyxy by using the padded level-0
  box extent and the saved crop size.
- Level-0 boxes are normalized back to yxyx 0-1000 coordinates using the WSI
  level-0 dimensions from each crop metadata file.

Prompt example from the first task:
{tasks[0]['prompt'] if tasks else ''}

Command to regenerate API responses:
{generation_command}

Last command used to write the current summaries:
{last_command}

Outputs:
- Output root: {output_root.resolve()}
- Results JSONL: {summary['results_jsonl']}
- Summary CSV: {summary['summary_csv']}
- Review PDF: {summary['comparison_pdf']}

Boxes:
{chr(10).join(f"- {task['case_index']:03d} final box {task['final_box_index']:02d}: {task['case_display']}" for task in tasks)}
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    if args.output_root is None:
        args.output_root = _default_output_root_for_mode(args.prompt_mode)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "task_ids": [task["task_id"] for task in tasks],
                    "tasks_jsonl": str((args.output_root / "tasks/stage7_bbox_refinement_tasks.jsonl").resolve()),
                    "output_root": str(args.output_root.resolve()),
                },
                indent=2,
            )
        )
        return 0

    results_path = args.output_root / "reviews/stage7_bbox_refinement_results.jsonl"
    if args.reuse_existing:
        if not results_path.exists():
            raise SystemExit(f"Missing existing results for --reuse-existing: {results_path}")
        task_by_id = {task["task_id"]: task for task in tasks}
        results = []
        for row in _read_jsonl(results_path):
            task = task_by_id.get(row.get("task_id"))
            if task and row.get("raw_response") and not row.get("error"):
                _postprocess_record_from_raw(row, task, args)
            results.append(row)
        results.sort(key=lambda row: (int(row["case_index"]), int(row["final_box_index"])))
        _write_jsonl(results_path, results)
    else:
        base_url, api_key = _api_settings(args)
        if args.max_concurrent > 1:
            results = []
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
                futures = [pool.submit(_run_one, task, args, base_url, api_key) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            results = [_run_one(task, args, base_url, api_key) for task in tasks]
        results.sort(key=lambda row: (int(row["case_index"]), int(row["final_box_index"])))
        _write_jsonl(results_path, results)
    pdf_path = _write_pdf(args.output_root, tasks, results, args)
    summary = _write_summary(args.output_root, tasks, results, pdf_path, args)
    _write_reproduction(args.output_root, args, tasks, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-results", type=Path, default=DEFAULT_HIGH_RESULTS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--prompt-mode", default="xyxy", choices=["xyxy", "feedback-normalized"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--indices", type=int, nargs="+", default=[99])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
