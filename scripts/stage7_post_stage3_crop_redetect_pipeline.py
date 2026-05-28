#!/usr/bin/env python3
"""Run the integrated post-Stage-3 crop-redetect detector pipeline smoke."""

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
    _extract_json_payload,
    _font,
    _load_raw_orientation_bboxes,
    _normalised_detection_items,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
)
from stage4_crop_prompt_packet import (
    _normalised_yxyx_to_level0,
    _pad_level0_bbox,
    _read_padded_crop,
)
from stage6_crop_tp_fp_review import _parse_tissue_yes_no
from stage6_odd_one_out_artifact_review import _parse_response as _parse_odd_one_out_response
from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1/summary/high_recall_stage1_cases.csv"
)
STAGE2B_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage2b_nonminor_two_pass_gemini_flash_low_v1"
    / "reviews/stage2b_two_pass_results.jsonl"
)
STAGE3_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage3_refinement_minimal_feedback_gemini_flash_high_v1"
    / "reviews/stage3_refinement_results.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage7_post_stage3_crop_redetect_oddoneout_smoke_v1"
)
STAGE1_PROMPT = REPO_ROOT / "prompts/stage1_high_recall_potential_tissue_candidates.txt"
STAGE6_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage6_crop_true_false_positive.txt"
ODD_ONE_OUT_PROMPT = (
    REPO_ROOT
    / "prompts/stage1_detector_oracle/stage6_odd_one_out_artifact_review_v2_contains_consensus.txt"
)
DEFAULT_INDICES = [23, 31, 49, 70, 74, 80, 84, 85, 95, 99]
DEFAULT_MODEL = "google/gemini-3-flash-preview"
PROMPT_VERSION = "stage7_post_stage3_crop_redetect_oddoneout_smoke_2026-05-28"
TICKET = "PER-207"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, --api-key-stdin, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return args.api_base or "https://openrouter.ai/api/v1", api_key


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _load_stage2b_flags(path: Path) -> dict[int, bool]:
    return {
        int(row["case_index"]): _boolish(row.get("final_non_minor_detection_failure"))
        for row in _read_jsonl(path)
        if "case_index" in row
    }


def _load_stage3_by_case(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["case_index"]): row
        for row in _read_jsonl(path)
        if "case_index" in row and not row.get("error")
    }


def _selected_stage1_rows(path: Path, indices: list[int]) -> list[dict[str, str]]:
    by_index = {int(row["case_index"]): row for row in _read_csv(path)}
    missing = [idx for idx in indices if idx not in by_index]
    if missing:
        raise SystemExit(f"Stage 1 case rows missing: {missing}")
    return [by_index[idx] for idx in indices]


def _case_source_bboxes(
    row: dict[str, str],
    stage2b_flags: dict[int, bool],
    stage3_by_case: dict[int, dict[str, Any]],
    use_stage3_when_available: bool,
    rotation: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    case_index = int(row["case_index"])
    if use_stage3_when_available and stage2b_flags.get(case_index) and case_index in stage3_by_case:
        stage3 = stage3_by_case[case_index]
        return (
            "stage3_feedback_redetection",
            str(stage3.get("stage3_overlay_path", "")),
            list(stage3.get("detections", [])),
        )

    thumbnail_path = Path(row["thumbnail_path"])
    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size
    bboxes, note = _load_raw_orientation_bboxes(Path(row["bboxes_json_path"]), thumbnail_size, rotation)
    if note:
        raise SystemExit(f"Could not load raw rotation {rotation} bboxes for case {case_index}: {note}")
    return f"stage1_raw_rot{rotation}", row["raw_overlay_path"], bboxes


def _yxyx_overlap_metrics(a: list[float], b: list[float]) -> tuple[float, float]:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_y1, inter_x1 = max(ay1, by1), max(ax1, bx1)
    inter_y2, inter_x2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def _merge_yxyx_boxes(
    boxes: list[list[float]],
    iou_threshold: float,
    containment_threshold: float,
) -> tuple[list[list[float]], dict[str, int]]:
    merged = [list(map(float, box)) for box in boxes]
    counts = {"total": 0, "iou": 0, "containment": 0}
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        used: set[int] = set()
        for i, box in enumerate(merged):
            if i in used:
                continue
            hull = list(box)
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                iou, overlap_over_smaller = _yxyx_overlap_metrics(hull, merged[j])
                if iou > iou_threshold or (
                    containment_threshold > 0 and overlap_over_smaller >= containment_threshold
                ):
                    other = merged[j]
                    hull = [
                        min(hull[0], other[0]),
                        min(hull[1], other[1]),
                        max(hull[2], other[2]),
                        max(hull[3], other[3]),
                    ]
                    used.add(j)
                    counts["total"] += 1
                    if iou > iou_threshold:
                        counts["iou"] += 1
                    else:
                        counts["containment"] += 1
                    changed = True
            out.append(hull)
        merged = out
    return merged, counts


def _expand_yxyx(box: list[float], frac: float) -> list[float]:
    y1, x1, y2, x2 = [float(v) for v in box]
    h = max(1.0, y2 - y1)
    w = max(1.0, x2 - x1)
    return [
        max(0.0, y1 - h * frac),
        max(0.0, x1 - w * frac),
        min(1000.0, y2 + h * frac),
        min(1000.0, x2 + w * frac),
    ]


def _norm_to_image_bbox(norm_yxyx: list[float], image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    y1, x1, y2, x2 = [float(v) for v in norm_yxyx]
    return [
        int(round(max(0.0, min(1000.0, x1)) / 1000.0 * width)),
        int(round(max(0.0, min(1000.0, y1)) / 1000.0 * height)),
        int(round(max(0.0, min(1000.0, x2)) / 1000.0 * width)),
        int(round(max(0.0, min(1000.0, y2)) / 1000.0 * height)),
    ]


def _crop_pixel_bbox_to_wsi_norm(
    crop_bbox_xyxy: list[int],
    read_info: dict[str, Any],
    wsi_size: tuple[int, int],
) -> list[float]:
    wsi_w, wsi_h = wsi_size
    downsample = float(read_info["selected_downsample"])
    scale = float(read_info.get("resize_scale_after_read") or 1.0)
    px1, py1, _, _ = [float(v) for v in read_info["padded_bbox_level0"]]
    x1, y1, x2, y2 = [float(v) for v in crop_bbox_xyxy]
    l0_x1 = px1 + (x1 / scale) * downsample
    l0_y1 = py1 + (y1 / scale) * downsample
    l0_x2 = px1 + (x2 / scale) * downsample
    l0_y2 = py1 + (y2 / scale) * downsample
    x_low, x_high = sorted((max(0.0, min(float(wsi_w), l0_x1)), max(0.0, min(float(wsi_w), l0_x2))))
    y_low, y_high = sorted((max(0.0, min(float(wsi_h), l0_y1)), max(0.0, min(float(wsi_h), l0_y2))))
    return [
        y_low / max(1.0, float(wsi_h)) * 1000.0,
        x_low / max(1.0, float(wsi_w)) * 1000.0,
        y_high / max(1.0, float(wsi_h)) * 1000.0,
        x_high / max(1.0, float(wsi_w)) * 1000.0,
    ]


def _draw_boxes_overlay(
    image_path: Path,
    output_path: Path,
    boxes: list[list[float]],
    title: str,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(max(16, min(34, max(image.size) // 38)))
    title_font = _font(max(18, min(40, max(image.size) // 34)))
    line_w = max(3, max(image.size) // 280)
    default_colors = ["#e31a1c", "#33a02c", "#1f78b4", "#ff7f00", "#6a3d9a", "#b15928", "#00bcd4", "#f781bf"]
    colors = colors or default_colors
    labels = labels or [str(idx) for idx in range(1, len(boxes) + 1)]
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = _norm_to_image_bbox(box, image.size)
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_w)
        label = labels[idx]
        label_xy = (x1 + 3, y1 + 3)
        tb = draw.textbbox(label_xy, label, font=font)
        draw.rectangle(tb, fill="white", outline=color, width=max(1, line_w // 2))
        draw.text(label_xy, label, fill=color, font=font)
    if title:
        tb = draw.textbbox((10, 8), title, font=title_font)
        draw.rectangle(tb, fill="white")
        draw.text((10, 8), title, font=title_font, fill="#111111")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _draw_crop_detection_overlay(crop_path: Path, output_path: Path, detections: list[dict[str, Any]]) -> Path:
    boxes = [det["box_2d_yxyx_normalized"] for det in detections]
    return _draw_boxes_overlay(crop_path, output_path, boxes, f"crop Stage 1 detections: {len(boxes)}")


def _paste_fit(page: Image.Image, image_path: str | Path, box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(page)
    x, y, w, h = box
    if not image_path:
        draw.rectangle((x, y, x + w, y + h), fill="#f4f4f4", outline="#cccccc")
        draw.text((x + 20, y + h // 2), "Not run", font=_font(24), fill="#555555")
        return
    path = Path(image_path)
    if not path.exists():
        draw.rectangle((x, y, x + w, y + h), fill="#f4f4f4", outline="#cccccc")
        draw.text((x + 20, y + h // 2), "Missing image", font=_font(24), fill="#aa0000")
        return
    image = _thumb(path, (w, h))
    page.paste(image, (x, y))


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_h: int = 23,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_h
    return y


def _save_vlm_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=92, optimize=True)


def _build_post_stage3_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage1_rows = _selected_stage1_rows(args.stage1_cases, args.indices)
    stage2b_flags = _load_stage2b_flags(args.stage2b_results)
    stage3_by_case = _load_stage3_by_case(args.stage3_results)
    case_records: list[dict[str, Any]] = []
    all_box_rows: list[dict[str, Any]] = []
    crop_tasks: list[dict[str, Any]] = []

    for row in stage1_rows:
        case_index = int(row["case_index"])
        metadata = json.loads(Path(row["metadata_path"]).read_text())
        wsi_path = metadata["wsi_path"]
        wsi_size = (
            int(metadata["wsi_dimensions"]["width"]),
            int(metadata["wsi_dimensions"]["height"]),
        )
        source_name, source_overlay_path, source_bboxes = _case_source_bboxes(
            row,
            stage2b_flags,
            stage3_by_case,
            args.use_stage3_when_available,
            args.rotation,
        )
        raw_boxes = [
            [float(v) for v in bbox["box_2d_yxyx_normalized"]]
            for bbox in source_bboxes
            if bbox.get("box_2d_yxyx_normalized")
        ]
        merged, merge_counts = _merge_yxyx_boxes(
            raw_boxes,
            args.merge_iou_threshold,
            args.containment_overlap_threshold,
        )
        expanded = [_expand_yxyx(box, args.post_stage3_padding_frac) for box in merged]
        case_slug = _safe_slug(f"{case_index:03d}_{row['case_display']}")
        case_dir = args.output_root / "cases" / case_slug
        post_overlay = _draw_boxes_overlay(
            Path(row["thumbnail_path"]),
            case_dir / "stage3_postprocess/stage3_postprocess_overlay.png",
            expanded,
            f"post-Stage3 merge + 15%: {len(expanded)}",
        )

        wsi, reader = load_wsi(wsi_path, args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            for order, (source_box, padded_box) in enumerate(zip(merged, expanded), start=1):
                source_bbox_l0 = _normalised_yxyx_to_level0(source_box, wsi_size)
                padded_bbox_l0 = _normalised_yxyx_to_level0(padded_box, wsi_size)
                crop, read_info = _read_padded_crop(
                    wsi,
                    reader,
                    pyramid,
                    source_bbox_l0,
                    padded_bbox_l0,
                    args.max_dim,
                )
                read_info["padding_fraction"] = float(args.post_stage3_padding_frac)
                task_id = f"crop_redetect_{case_index:03d}_{order:02d}"
                region_dir = case_dir / "stage4_crop_redetect_inputs" / f"{order:02d}"
                crop_path = region_dir / "crop.png"
                source_overlay = region_dir / "source_box_overlay.png"
                metadata_path = region_dir / "metadata.json"
                region_dir.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
                overlay = crop.copy()
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = [int(v) for v in read_info["source_bbox_in_crop"]]
                line_w = max(3, max(crop.size) // 180)
                draw.rectangle((x1, y1, x2, y2), outline="#e31a1c", width=line_w)
                overlay.save(source_overlay)
                task = {
                    "task_id": task_id,
                    "case_index": case_index,
                    "case_display": row["case_display"],
                    "case_slug": case_slug,
                    "source_stage": source_name,
                    "source_order": order,
                    "source_box_yxyx_normalized": source_box,
                    "padded_box_yxyx_normalized": padded_box,
                    "crop_path": str(crop_path),
                    "source_overlay_path": str(source_overlay),
                    "metadata_path": str(metadata_path),
                    "thumbnail_path": row["thumbnail_path"],
                    "source_detector_overlay_path": source_overlay_path,
                    "wsi_path": wsi_path,
                    "wsi_reader": reader,
                    "wsi_size": list(wsi_size),
                    "read_info": read_info,
                    "prompt_version": PROMPT_VERSION,
                    "created_at": _timestamp(),
                }
                _write_json(
                    metadata_path,
                    {
                        "case_index": case_index,
                        "case_display": row["case_display"],
                        "wsi_path": wsi_path,
                        "wsi_reader": reader,
                        "pyramid": pyramid,
                        "task": task,
                    },
                )
                crop_tasks.append(task)
                all_box_rows.append(
                    {
                        "case_index": case_index,
                        "case_display": row["case_display"],
                        "source_stage": source_name,
                        "source_order": order,
                        "source_box_yxyx_normalized": json.dumps([round(v, 3) for v in source_box]),
                        "expanded_box_yxyx_normalized": json.dumps([round(v, 3) for v in padded_box]),
                        "crop_path": str(crop_path),
                        "source_overlay_path": str(source_overlay),
                        "metadata_path": str(metadata_path),
                    }
                )
        finally:
            close_wsi(wsi, reader)

        case_records.append(
            {
                "case_index": case_index,
                "case_display": row["case_display"],
                "case_slug": case_slug,
                "thumbnail_path": row["thumbnail_path"],
                "source_stage": source_name,
                "source_detector_overlay_path": source_overlay_path,
                "source_box_count": len(raw_boxes),
                "post_stage3_merged_count": len(merged),
                "post_stage3_expanded_count": len(expanded),
                "post_stage3_merge_counts": merge_counts,
                "post_stage3_boxes_yxyx_normalized": merged,
                "post_stage3_expanded_boxes_yxyx_normalized": expanded,
                "post_stage3_overlay_path": str(post_overlay),
                "case_dir": str(case_dir),
                "wsi_path": wsi_path,
                "wsi_size": list(wsi_size),
            }
        )

    _write_jsonl(args.output_root / "stage3_postprocess/stage3_postprocess_cases.jsonl", case_records)
    _write_csv(
        args.output_root / "stage3_postprocess/stage3_postprocess_boxes.csv",
        all_box_rows,
        [
            "case_index",
            "case_display",
            "source_stage",
            "source_order",
            "source_box_yxyx_normalized",
            "expanded_box_yxyx_normalized",
            "crop_path",
            "source_overlay_path",
            "metadata_path",
        ],
    )
    _write_jsonl(args.output_root / "stage4_crop_redetect/tasks/crop_redetect_tasks.jsonl", crop_tasks)
    summary = {
        "cases": len(case_records),
        "source_boxes": sum(record["source_box_count"] for record in case_records),
        "post_stage3_boxes": sum(record["post_stage3_expanded_count"] for record in case_records),
        "post_stage3_merge_events": sum(record["post_stage3_merge_counts"]["total"] for record in case_records),
    }
    return case_records, {"crop_redetect_tasks": crop_tasks, **summary}


def _run_crop_redetect(
    tasks: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    results_path = args.output_root / "stage4_crop_redetect/reviews/crop_redetect_results.jsonl"
    if args.reuse_existing and results_path.exists():
        return _read_jsonl(results_path)

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        record = {
            key: task[key]
            for key in (
                "task_id",
                "case_index",
                "case_display",
                "case_slug",
                "source_stage",
                "source_order",
                "source_box_yxyx_normalized",
                "padded_box_yxyx_normalized",
                "crop_path",
                "source_overlay_path",
                "metadata_path",
                "thumbnail_path",
                "source_detector_overlay_path",
                "wsi_path",
                "wsi_reader",
                "wsi_size",
                "read_info",
                "prompt_version",
                "created_at",
            )
        }
        record.update(
            {
                "model": args.model,
                "reasoning_effort": args.crop_redetect_reasoning_effort,
                "raw_response": "",
                "parsed_response": None,
                "detections_crop": [],
                "detections_wsi": [],
                "detection_count": 0,
                "parser_status": "not_run",
                "error": "",
                "usage": {},
                "response_model": "",
                "crop_detection_overlay_path": "",
            }
        )
        try:
            raw, usage, response_model = _chat_with_images(
                model=args.model,
                prompt_text=prompt,
                image_paths=[Path(task["crop_path"])],
                temperature=args.temperature,
                max_tokens=args.crop_redetect_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.crop_redetect_reasoning_effort,
            )
            payload = _extract_json_payload(raw)
            with Image.open(task["crop_path"]) as image:
                crop_size = image.size
            detections_crop = _normalised_detection_items(payload, crop_size)
            wsi_size = (int(task["wsi_size"][0]), int(task["wsi_size"][1]))
            detections_wsi = []
            for idx, det in enumerate(detections_crop, start=1):
                norm = _crop_pixel_bbox_to_wsi_norm(det["bbox_thumbnail"], task["read_info"], wsi_size)
                detections_wsi.append(
                    {
                        "label": det.get("label") or f"tissue_{idx}",
                        "crop_detection": det,
                        "box_2d_yxyx_normalized": [round(float(v), 3) for v in norm],
                    }
                )
            overlay_path = Path(task["crop_path"]).with_name("crop_redetect_overlay.png")
            _draw_crop_detection_overlay(Path(task["crop_path"]), overlay_path, detections_crop)
            record.update(
                {
                    "raw_response": raw,
                    "parsed_response": payload,
                    "detections_crop": detections_crop,
                    "detections_wsi": detections_wsi,
                    "detection_count": len(detections_wsi),
                    "parser_status": "ok" if detections_wsi else "no_detections",
                    "usage": usage,
                    "response_model": response_model,
                    "crop_detection_overlay_path": str(overlay_path),
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
            record["parser_status"] = "error"
        return record

    if args.max_concurrent > 1:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(run_one, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_one(task) for task in tasks]
    results.sort(key=lambda row: (int(row["case_index"]), int(row["source_order"])))
    _write_jsonl(results_path, results)
    return results


def _build_classification_inputs(
    case_records: list[dict[str, Any]],
    redetect_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[int, list[dict[str, Any]]] = {}
    for row in redetect_results:
        by_case.setdefault(int(row["case_index"]), []).extend(row.get("detections_wsi") or [])

    candidate_rows: list[dict[str, Any]] = []
    updated_cases: list[dict[str, Any]] = []
    for record in case_records:
        case_index = int(record["case_index"])
        detections = by_case.get(case_index, [])
        boxes = [[float(v) for v in det["box_2d_yxyx_normalized"]] for det in detections]
        merged, merge_counts = _merge_yxyx_boxes(
            boxes,
            args.merge_iou_threshold,
            args.containment_overlap_threshold,
        )
        case_slug = record["case_slug"]
        case_dir = Path(record["case_dir"])
        thumbnail_path = Path(record["thumbnail_path"])
        merge_overlay = _draw_boxes_overlay(
            thumbnail_path,
            case_dir / "stage5_classification_inputs/post_redetect_merge_overlay.png",
            merged,
            f"crop redetect merge: {len(merged)}",
        )
        wsi_size = (int(record["wsi_size"][0]), int(record["wsi_size"][1]))
        wsi, reader = load_wsi(record["wsi_path"], args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            for order, box in enumerate(merged, start=1):
                source_bbox_l0 = _normalised_yxyx_to_level0(box, wsi_size)
                padded_bbox_l0 = _pad_level0_bbox(source_bbox_l0, wsi_size, args.classification_padding_frac)
                crop, read_info = _read_padded_crop(
                    wsi,
                    reader,
                    pyramid,
                    source_bbox_l0,
                    padded_bbox_l0,
                    args.max_dim,
                )
                read_info["padding_fraction"] = float(args.classification_padding_frac)
                candidate_id = f"{order:02d}_crop_redetect"
                candidate_dir = case_dir / "stage5_classification_inputs/candidates" / candidate_id
                crop_path = candidate_dir / "crop.png"
                overlay_path = candidate_dir / "selected_candidate_overlay.png"
                vlm_path = candidate_dir / "selected_candidate_overlay_vlm.jpg"
                metadata_path = candidate_dir / "metadata.json"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
                overlay = crop.copy()
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = [int(v) for v in read_info["source_bbox_in_crop"]]
                line_w = max(3, max(crop.size) // 180)
                draw.rectangle((x1, y1, x2, y2), outline="#e31a1c", width=line_w)
                label_font = _font(max(20, min(42, max(crop.size) // 24)))
                label_box = draw.textbbox((x1 + 5, y1 + 5), str(order), font=label_font)
                draw.rectangle(label_box, fill="white", outline="#e31a1c", width=max(2, line_w // 2))
                draw.text((x1 + 5, y1 + 5), str(order), fill="#e31a1c", font=label_font)
                overlay.save(overlay_path)
                _save_vlm_jpeg(overlay, vlm_path)
                candidate = {
                    "case_index": case_index,
                    "case_display": record["case_display"],
                    "case_slug": case_slug,
                    "bbox_source": "post_stage3_crop_redetect",
                    "candidate_order": order,
                    "candidate_id": candidate_id,
                    "label": candidate_id,
                    "box_2d_yxyx_normalized": box,
                    "crop_path": str(crop_path),
                    "selected_overlay_path": str(overlay_path),
                    "vlm_image_path": str(vlm_path),
                    "metadata_path": str(metadata_path),
                    "thumbnail_path": str(thumbnail_path),
                    "wsi_path": record["wsi_path"],
                    "wsi_reader": reader,
                    "read_info": read_info,
                    "prompt_version": PROMPT_VERSION,
                    "created_at": _timestamp(),
                }
                _write_json(
                    metadata_path,
                    {
                        "case_index": case_index,
                        "case_display": record["case_display"],
                        "wsi_path": record["wsi_path"],
                        "wsi_reader": reader,
                        "pyramid": pyramid,
                        "candidate": candidate,
                    },
                )
                candidate_rows.append(candidate)
        finally:
            close_wsi(wsi, reader)

        updated = {
            **record,
            "crop_redetect_detection_count": len(boxes),
            "post_redetect_merged_count": len(merged),
            "post_redetect_merge_counts": merge_counts,
            "post_redetect_boxes_yxyx_normalized": merged,
            "post_redetect_merge_overlay_path": str(merge_overlay),
        }
        updated_cases.append(updated)

    csv_rows = []
    for row in candidate_rows:
        read = row["read_info"]
        csv_rows.append(
            {
                "case_index": row["case_index"],
                "case_display": row["case_display"],
                "bbox_source": row["bbox_source"],
                "candidate_order": row["candidate_order"],
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "box_2d_yxyx_normalized": json.dumps([round(float(v), 3) for v in row["box_2d_yxyx_normalized"]]),
                "crop_path": row["crop_path"],
                "selected_overlay_path": row["selected_overlay_path"],
                "vlm_image_path": row["vlm_image_path"],
                "metadata_path": row["metadata_path"],
                "selected_level": read["selected_level"],
                "selected_downsample": read["selected_downsample"],
                "projected_long_edge_at_level": read["projected_long_edge_at_level"],
                "crop_width": read["crop_size"][0],
                "crop_height": read["crop_size"][1],
                "source_bbox_level0": json.dumps(read["source_bbox_level0"]),
                "padded_bbox_level0": json.dumps(read["padded_bbox_level0"]),
                "source_bbox_in_crop": json.dumps(read["source_bbox_in_crop"]),
            }
        )
    _write_csv(
        args.output_root / "stage5_classification_inputs/classification_candidates.csv",
        csv_rows,
        [
            "case_index",
            "case_display",
            "bbox_source",
            "candidate_order",
            "candidate_id",
            "label",
            "box_2d_yxyx_normalized",
            "crop_path",
            "selected_overlay_path",
            "vlm_image_path",
            "metadata_path",
            "selected_level",
            "selected_downsample",
            "projected_long_edge_at_level",
            "crop_width",
            "crop_height",
            "source_bbox_level0",
            "padded_bbox_level0",
            "source_bbox_in_crop",
        ],
    )
    _write_jsonl(args.output_root / "stage5_classification_inputs/classification_candidates.jsonl", candidate_rows)
    return updated_cases, candidate_rows


def _run_classification(
    candidates: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    results_path = args.output_root / "stage6_classification/reviews/stage6_crop_tissue_artifact_high_thinking.jsonl"
    if args.reuse_existing and results_path.exists():
        return _read_jsonl(results_path)

    def run_one(candidate: dict[str, Any]) -> dict[str, Any]:
        record = {
            key: candidate[key]
            for key in (
                "case_index",
                "case_display",
                "case_slug",
                "bbox_source",
                "candidate_order",
                "candidate_id",
                "label",
                "box_2d_yxyx_normalized",
                "crop_path",
                "selected_overlay_path",
                "vlm_image_path",
                "metadata_path",
                "thumbnail_path",
                "wsi_path",
                "wsi_reader",
                "prompt_version",
                "created_at",
            )
        }
        record.update(
            {
                "task_id": f"stage6_tissue_artifact_{int(candidate['case_index']):03d}_{int(candidate['candidate_order']):02d}",
                "model": args.model,
                "reasoning_effort": args.classification_reasoning_effort,
                "raw_response": "",
                "tissue_focus_decision": "unknown",
                "parser_route": "",
                "error": "",
                "usage": {},
                "response_model": "",
            }
        )
        try:
            raw, usage, response_model = _chat_with_images(
                model=args.model,
                prompt_text=prompt,
                image_paths=[Path(candidate["selected_overlay_path"])],
                temperature=args.temperature,
                max_tokens=args.classification_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.classification_reasoning_effort,
            )
            decision, parser_route = _parse_tissue_yes_no(raw)
            record.update(
                {
                    "raw_response": raw,
                    "tissue_focus_decision": decision,
                    "parser_route": parser_route,
                    "usage": usage,
                    "response_model": response_model,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
        return record

    if args.max_concurrent > 1:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(run_one, candidate) for candidate in candidates]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_one(candidate) for candidate in candidates]
    results.sort(key=lambda row: (int(row["case_index"]), int(row["candidate_order"])))
    _write_jsonl(results_path, results)
    _write_csv(
        args.output_root / "stage6_classification/summary/stage6_crop_tissue_artifact_high_thinking.csv",
        [
            {
                **row,
                "box_2d_yxyx_normalized": json.dumps(
                    [round(float(v), 3) for v in row["box_2d_yxyx_normalized"]]
                ),
            }
            for row in results
        ],
        [
            "case_index",
            "case_display",
            "candidate_order",
            "candidate_id",
            "bbox_source",
            "tissue_focus_decision",
            "parser_route",
            "error",
            "raw_response",
            "box_2d_yxyx_normalized",
            "selected_overlay_path",
            "crop_path",
            "metadata_path",
        ],
    )
    return results


def _draw_classification_overlays(
    case_records: list[dict[str, Any]],
    classification_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in classification_results:
        grouped.setdefault(int(row["case_index"]), []).append(row)

    updated: list[dict[str, Any]] = []
    color_map = {"yes": "#188038", "no": "#d93025", "unknown": "#5f6368"}
    for record in case_records:
        rows = sorted(grouped.get(int(record["case_index"]), []), key=lambda r: int(r["candidate_order"]))
        boxes = [[float(v) for v in row["box_2d_yxyx_normalized"]] for row in rows]
        colors = [color_map.get(row["tissue_focus_decision"], "#5f6368") for row in rows]
        labels = [f"{int(row['candidate_order']):02d}:{row['tissue_focus_decision']}" for row in rows]
        overlay = _draw_boxes_overlay(
            Path(record["thumbnail_path"]),
            Path(record["case_dir"]) / "stage6_classification/classification_overlay.png",
            boxes,
            "classification: green yes / red no",
            colors=colors,
            labels=labels,
        )
        updated.append(
            {
                **record,
                "classification_overlay_path": str(overlay),
                "classification_yes_count": sum(1 for row in rows if row["tissue_focus_decision"] == "yes"),
                "classification_no_count": sum(1 for row in rows if row["tissue_focus_decision"] == "no"),
                "classification_unknown_count": sum(1 for row in rows if row["tissue_focus_decision"] == "unknown"),
            }
        )
    return updated


def _build_odd_one_out_inputs(
    case_records: list[dict[str, Any]],
    classification_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    yes_by_case: dict[int, list[dict[str, Any]]] = {}
    for row in classification_results:
        if row.get("tissue_focus_decision") == "yes" and not row.get("error"):
            yes_by_case.setdefault(int(row["case_index"]), []).append(row)
    for rows in yes_by_case.values():
        rows.sort(key=lambda row: int(row["candidate_order"]))

    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in case_records:
        case_index = int(record["case_index"])
        rows = yes_by_case.get(case_index, [])
        case_dir = Path(record["case_dir"]) / "stage7_odd_one_out"
        if len(rows) <= 1:
            skipped.append(
                {
                    "case_index": case_index,
                    "case_display": record["case_display"],
                    "remaining_crop_count": len(rows),
                    "skip_reason": "remaining_crop_count_below_2",
                }
            )
            continue
        thumbnail = Image.open(record["thumbnail_path"]).convert("RGB")
        patches: list[dict[str, Any]] = []
        for patch_id, row in enumerate(rows, start=1):
            bbox = _norm_to_image_bbox([float(v) for v in row["box_2d_yxyx_normalized"]], thumbnail.size)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(thumbnail.size[0], x1))
            x2 = max(0, min(thumbnail.size[0], x2))
            y1 = max(0, min(thumbnail.size[1], y1))
            y2 = max(0, min(thumbnail.size[1], y2))
            if x2 <= x1 or y2 <= y1:
                continue
            patch_dir = case_dir / "thumbnail_crops"
            patch_dir.mkdir(parents=True, exist_ok=True)
            crop_path = patch_dir / f"{patch_id:02d}_candidate_{int(row['candidate_order']):02d}.png"
            thumbnail.crop((x1, y1, x2, y2)).save(crop_path)
            patches.append(
                {
                    "id": patch_id,
                    "candidate_order": int(row["candidate_order"]),
                    "candidate_id": row["candidate_id"],
                    "label": row["label"],
                    "crop_path": str(crop_path),
                    "bbox_thumbnail": [x1, y1, x2, y2],
                    "box_2d_yxyx_normalized": row["box_2d_yxyx_normalized"],
                    "crop_size": list(Image.open(crop_path).size),
                }
            )
        if len(patches) <= 1:
            skipped.append(
                {
                    "case_index": case_index,
                    "case_display": record["case_display"],
                    "remaining_crop_count": len(patches),
                    "skip_reason": "valid_thumbnail_crop_count_below_2",
                }
            )
            continue
        task = {
            "task_id": f"odd_one_out_{case_index:03d}",
            "case_index": case_index,
            "case_display": record["case_display"],
            "case_slug": record["case_slug"],
            "patch_count": len(patches),
            "patches": patches,
            "thumbnail_path": record["thumbnail_path"],
            "prompt_version": PROMPT_VERSION,
            "created_at": _timestamp(),
        }
        _write_json(case_dir / "odd_one_out_input.json", task)
        tasks.append(task)

    _write_jsonl(args.output_root / "stage7_odd_one_out/tasks/odd_one_out_tasks.jsonl", tasks)
    _write_jsonl(args.output_root / "stage7_odd_one_out/tasks/skipped_cases.jsonl", skipped)
    return tasks, skipped


def _odd_prompt_with_ids(prompt: str, patch_count: int) -> str:
    return (
        prompt.strip()
        + "\n\nThe attached crop images are ordered by crop id. "
        + f"Use id 1 for the first attached image through id {patch_count} for the last attached image."
    )


def _run_odd_one_out(
    tasks: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    results_path = args.output_root / "stage7_odd_one_out/reviews/odd_one_out_results.jsonl"
    existing_ok: dict[int, dict[str, Any]] = {}
    if (args.reuse_existing or args.rerun_incomplete) and results_path.exists():
        existing_rows = _read_jsonl(results_path)
        if args.reuse_existing and not args.rerun_incomplete:
            return existing_rows
        for row in existing_rows:
            if row.get("parse_status") == "ok" and not row.get("error"):
                existing_ok[int(row["case_index"])] = row

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        record = {
            key: task[key]
            for key in (
                "task_id",
                "case_index",
                "case_display",
                "case_slug",
                "patch_count",
                "patches",
                "thumbnail_path",
                "prompt_version",
                "created_at",
            )
        }
        record.update(
            {
                "model": args.model,
                "reasoning_effort": args.odd_one_out_reasoning_effort,
                "raw_response": "",
                "parsed_response": None,
                "parse_route": "",
                "parse_status": "not_run",
                "flagged_artifacts": [],
                "flagged_candidate_orders": [],
                "error": "",
                "usage": {},
                "response_model": "",
            }
        )
        try:
            image_paths = [Path(patch["crop_path"]) for patch in task["patches"]]
            raw, usage, response_model = _chat_with_images(
                model=args.model,
                prompt_text=_odd_prompt_with_ids(prompt, int(task["patch_count"])),
                image_paths=image_paths,
                temperature=args.temperature,
                max_tokens=args.odd_one_out_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.odd_one_out_reasoning_effort,
            )
            parsed, route, status = _parse_odd_one_out_response(raw, int(task["patch_count"]))
            flagged_ids = []
            if isinstance(parsed, dict):
                for item in parsed.get("flagged_artifacts", []):
                    try:
                        flagged_ids.append(int(item))
                    except Exception:
                        continue
            id_to_order = {int(patch["id"]): int(patch["candidate_order"]) for patch in task["patches"]}
            record.update(
                {
                    "raw_response": raw,
                    "parsed_response": parsed,
                    "parse_route": route,
                    "parse_status": status,
                    "flagged_artifacts": sorted(flagged_ids),
                    "flagged_candidate_orders": sorted(
                        id_to_order[item] for item in flagged_ids if item in id_to_order
                    ),
                    "usage": usage,
                    "response_model": response_model,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
            record["parse_status"] = "error"
        return record

    jobs = [task for task in tasks if int(task["case_index"]) not in existing_ok]
    if args.max_concurrent > 1:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(run_one, task) for task in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_one(task) for task in jobs]
    results.extend(existing_ok.values())
    results.sort(key=lambda row: int(row["case_index"]))
    _write_jsonl(results_path, results)
    _write_csv(
        args.output_root / "stage7_odd_one_out/summary/odd_one_out_summary.csv",
        [
            {
                "case_index": row["case_index"],
                "case_display": row["case_display"],
                "patch_count": row["patch_count"],
                "parse_status": row["parse_status"],
                "flagged_artifacts": json.dumps(row["flagged_artifacts"]),
                "flagged_candidate_orders": json.dumps(row["flagged_candidate_orders"]),
                "error": row["error"],
                "raw_response": row["raw_response"],
            }
            for row in results
        ],
        [
            "case_index",
            "case_display",
            "patch_count",
            "parse_status",
            "flagged_artifacts",
            "flagged_candidate_orders",
            "error",
            "raw_response",
        ],
    )
    return results


def _draw_odd_sheet(task: dict[str, Any] | None, flagged_ids: set[int], output_path: Path) -> str:
    if task is None:
        return ""
    patches = task["patches"]
    cols = min(4, max(1, len(patches)))
    panel_w, panel_h = 330, 240
    gap = 34
    rows = (len(patches) + cols - 1) // cols
    width = cols * panel_w + (cols + 1) * gap
    height = rows * (panel_h + 58) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for idx, patch in enumerate(patches):
        row = idx // cols
        col = idx % cols
        x = gap + col * (panel_w + gap)
        y = gap + row * (panel_h + 58)
        image = _thumb(Path(patch["crop_path"]), (panel_w, panel_h))
        sheet.paste(image, (x, y))
        patch_id = int(patch["id"])
        if patch_id in flagged_ids:
            draw.rectangle((x - 5, y - 5, x + image.width + 5, y + image.height + 5), outline="#d7191c", width=9)
        suffix = " FLAGGED" if patch_id in flagged_ids else ""
        draw.text(
            (x, y + image.height + 8),
            f"id {patch_id} / cand {patch['candidate_order']:02d}{suffix}",
            font=font,
            fill="#b00020" if patch_id in flagged_ids else "#111111",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _build_final_outputs(
    case_records: list[dict[str, Any]],
    classification_results: list[dict[str, Any]],
    odd_tasks: list[dict[str, Any]],
    odd_results: list[dict[str, Any]],
    odd_skipped: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    class_by_case: dict[int, list[dict[str, Any]]] = {}
    for row in classification_results:
        class_by_case.setdefault(int(row["case_index"]), []).append(row)
    odd_by_case = {int(row["case_index"]): row for row in odd_results}
    odd_task_by_case = {int(row["case_index"]): row for row in odd_tasks}
    skipped_by_case = {int(row["case_index"]): row for row in odd_skipped}
    final_records: list[dict[str, Any]] = []
    final_box_rows: list[dict[str, Any]] = []

    for record in case_records:
        case_index = int(record["case_index"])
        rows = sorted(class_by_case.get(case_index, []), key=lambda row: int(row["candidate_order"]))
        yes_rows = [row for row in rows if row.get("tissue_focus_decision") == "yes" and not row.get("error")]
        odd_result = odd_by_case.get(case_index)
        flagged_orders = set(int(v) for v in (odd_result or {}).get("flagged_candidate_orders", []))
        final_rows = [row for row in yes_rows if int(row["candidate_order"]) not in flagged_orders]
        final_boxes = [[float(v) for v in row["box_2d_yxyx_normalized"]] for row in final_rows]
        case_dir = Path(record["case_dir"])
        final_overlay = _draw_boxes_overlay(
            Path(record["thumbnail_path"]),
            case_dir / "final_detections/final_overlay.png",
            final_boxes,
            f"final boxes: {len(final_boxes)}",
            colors=["#188038"],
        )
        odd_sheet_path = _draw_odd_sheet(
            odd_task_by_case.get(case_index),
            set(int(v) for v in (odd_result or {}).get("flagged_artifacts", [])),
            case_dir / "stage7_odd_one_out/odd_one_out_thumbnail_crops.png",
        )
        final_record = {
            **record,
            "odd_one_out_ran": odd_result is not None,
            "odd_one_out_parse_status": (odd_result or {}).get("parse_status", ""),
            "odd_one_out_flagged_candidate_orders": sorted(flagged_orders),
            "odd_one_out_skipped": skipped_by_case.get(case_index, {}),
            "odd_one_out_sheet_path": odd_sheet_path,
            "final_box_count": len(final_boxes),
            "final_boxes_yxyx_normalized": final_boxes,
            "final_overlay_path": str(final_overlay),
        }
        final_records.append(final_record)
        for idx, row in enumerate(final_rows, start=1):
            final_box_rows.append(
                {
                    "case_index": case_index,
                    "case_display": record["case_display"],
                    "final_box_index": idx,
                    "source_candidate_order": int(row["candidate_order"]),
                    "box_2d_yxyx_normalized": json.dumps(
                        [round(float(v), 3) for v in row["box_2d_yxyx_normalized"]]
                    ),
                    "classification_decision": row["tissue_focus_decision"],
                    "odd_one_out_flagged_candidate_orders": json.dumps(sorted(flagged_orders)),
                    "final_overlay_path": str(final_overlay),
                }
            )

    _write_jsonl(args.output_root / "final_detections/final_cases.jsonl", final_records)
    _write_csv(
        args.output_root / "final_detections/final_boxes.csv",
        final_box_rows,
        [
            "case_index",
            "case_display",
            "final_box_index",
            "source_candidate_order",
            "box_2d_yxyx_normalized",
            "classification_decision",
            "odd_one_out_flagged_candidate_orders",
            "final_overlay_path",
        ],
    )
    summary = {
        "created_at": _timestamp(),
        "ticket": TICKET,
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "cases": len(final_records),
        "final_boxes": sum(record["final_box_count"] for record in final_records),
        "odd_one_out_cases": len(odd_results),
        "odd_one_out_skipped_cases": len(odd_skipped),
        "odd_one_out_flagged_crops": sum(len(row.get("flagged_candidate_orders", [])) for row in odd_results),
        "classification_yes": sum(record.get("classification_yes_count", 0) for record in final_records),
        "classification_no": sum(record.get("classification_no_count", 0) for record in final_records),
        "classification_unknown": sum(record.get("classification_unknown_count", 0) for record in final_records),
    }
    _write_json(args.output_root / "summary/pipeline_summary.json", summary)
    return final_records, summary


def _write_pdf(final_records: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> Path:
    pages: list[Image.Image] = []
    cover = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(cover)
    title = _font(42)
    header = _font(28)
    body = _font(21)
    small = _font(17)
    y = 55
    draw.text((70, y), "PER-207 Post-Stage-3 Crop-Redetect Pipeline Smoke", font=title, fill="black")
    y += 60
    lines = [
        f"Cases={summary['cases']} | final_boxes={summary['final_boxes']} | classification yes/no/unknown={summary['classification_yes']}/{summary['classification_no']}/{summary['classification_unknown']}",
        f"Odd-one-out cases={summary['odd_one_out_cases']} | skipped={summary['odd_one_out_skipped_cases']} | flagged_crops={summary['odd_one_out_flagged_crops']}",
        f"Post-Stage3: merge IoU>{args.merge_iou_threshold:.2f} or overlap/smaller>={args.containment_overlap_threshold:.2f}, then expand {args.post_stage3_padding_frac:.2f}.",
        f"After crop redetect: merge again, crop classification inputs with {args.classification_padding_frac:.2f} padding.",
    ]
    for line in lines:
        draw.text((70, y), line, font=body, fill="#111111")
        y += 34
    y += 24
    draw.text((70, y), "Pipeline", font=header, fill="black")
    y += 42
    pipeline = [
        "1. Start from current Stage 3 feedback-redetection boxes when available; otherwise current raw Stage 1 rot0 boxes.",
        "2. Deterministically merge with containment-aware logic and expand by 15%.",
        "3. Reread those WSI crops and rerun the same Stage 1 high-recall detection prompt on each crop.",
        "4. Map crop-relative boxes back to WSI coordinates, merge again, then reread classification crops with 10% padding.",
        "5. Run the current Stage 6 tissue-containment classifier.",
        "6. For cases with more than one remaining tissue-positive crop, run the PER-237 odd-one-out comparative filter on thumbnail crops.",
        "7. Display the final retained bbox set.",
    ]
    for line in pipeline:
        y = _draw_wrapped(draw, (90, y), line, 145, small)
        y += 4
    pages.append(cover)

    for record in final_records:
        page = Image.new("RGB", (2400, 3200), "white")
        draw = ImageDraw.Draw(page)
        y = 40
        draw.text((50, y), record["case_display"], font=_font(34), fill="black")
        y += 48
        summary_line = (
            f"source={record['source_stage']} | source_boxes={record['source_box_count']} | "
            f"post_stage3={record['post_stage3_expanded_count']} | crop_redetect={record.get('crop_redetect_detection_count', 0)} | "
            f"post_redetect={record.get('post_redetect_merged_count', 0)} | class yes/no/unk="
            f"{record.get('classification_yes_count', 0)}/{record.get('classification_no_count', 0)}/{record.get('classification_unknown_count', 0)} | "
            f"odd_flagged={record.get('odd_one_out_flagged_candidate_orders', [])} | final={record['final_box_count']}"
        )
        y = _draw_wrapped(draw, (50, y), summary_line, 185, _font(17))
        y += 18
        panels = [
            ("Source thumbnail", record["thumbnail_path"]),
            ("Detector source", record["source_detector_overlay_path"]),
            ("Post-Stage3 15%", record["post_stage3_overlay_path"]),
            ("Crop redetect merge", record.get("post_redetect_merge_overlay_path", "")),
            ("Classification", record.get("classification_overlay_path", "")),
            ("Final boxes", record["final_overlay_path"]),
        ]
        panel_w, panel_h = 720, 470
        for idx, (label, path) in enumerate(panels):
            col = idx % 3
            row = idx // 3
            x = 50 + col * 780
            yy = y + row * 560
            draw.text((x, yy), label, font=_font(23), fill="black")
            _paste_fit(page, path, (x, yy + 32, panel_w, panel_h))
        y += 1140
        draw.text((50, y), "Odd-one-out thumbnail crop inputs", font=_font(24), fill="black")
        y += 34
        if record.get("odd_one_out_sheet_path"):
            _paste_fit(page, record["odd_one_out_sheet_path"], (50, y, 1320, 700))
        else:
            skipped = record.get("odd_one_out_skipped", {})
            _draw_wrapped(
                draw,
                (70, y + 30),
                f"Odd-one-out skipped: {skipped.get('skip_reason', 'not_applicable')}",
                120,
                _font(19),
                "#555555",
            )
        x_text = 1430
        draw.text((x_text, y), "Case record", font=_font(24), fill="black")
        text = {
            "stage3_merge": record.get("post_stage3_merge_counts", {}),
            "post_redetect_merge": record.get("post_redetect_merge_counts", {}),
            "odd_parse": record.get("odd_one_out_parse_status", ""),
            "odd_flagged_candidate_orders": record.get("odd_one_out_flagged_candidate_orders", []),
            "final_boxes": [[round(float(v), 2) for v in box] for box in record.get("final_boxes_yxyx_normalized", [])],
        }
        _draw_wrapped(draw, (x_text, y + 36), json.dumps(text, sort_keys=True), 70, _font(16), "#111111", 20)
        pages.append(page)

    pdf_path = args.output_root / "visuals/stage7_post_stage3_crop_redetect_oddoneout_smoke.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_reproduction(args: argparse.Namespace, pdf_path: Path, summary: dict[str, Any]) -> None:
    command_parts = [
        "python",
        "scripts/stage7_post_stage3_crop_redetect_pipeline.py",
        "--stage1-cases",
        str(args.stage1_cases.resolve()),
        "--stage2b-results",
        str(args.stage2b_results.resolve()),
        "--stage3-results",
        str(args.stage3_results.resolve()),
        "--stage1-prompt",
        str(args.stage1_prompt.resolve()),
        "--classification-prompt",
        str(args.classification_prompt.resolve()),
        "--odd-one-out-prompt",
        str(args.odd_one_out_prompt.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--indices",
        *(str(v) for v in args.indices),
        "--post-stage3-padding-frac",
        str(args.post_stage3_padding_frac),
        "--classification-padding-frac",
        str(args.classification_padding_frac),
        "--merge-iou-threshold",
        str(args.merge_iou_threshold),
        "--containment-overlap-threshold",
        str(args.containment_overlap_threshold),
        "--max-dim",
        str(args.max_dim),
        "--model",
        args.model,
        "--max-concurrent",
        str(args.max_concurrent),
        "--temperature",
        str(args.temperature),
        "--wsi-reader",
        args.wsi_reader,
    ]
    if args.reuse_existing:
        command_parts.append("--reuse-existing")
    if args.rerun_incomplete:
        command_parts.append("--rerun-incomplete")
    if args.dry_run:
        command_parts.append("--dry-run")
    command = " ".join(shlex.quote(part) for part in command_parts)
    text = f"""\
Post-Stage-3 crop-redetect plus odd-one-out detector pipeline smoke
===================================================================

Created: {_timestamp()}
Ticket: {TICKET}
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions
Reuse existing model outputs: {args.reuse_existing}

Objective:
Run the updated detector-oracle pipeline on 10 cases so each stage can be
inspected. The pipeline adds high-resolution crop-level Stage 1 redetection
after the current Stage 3 source, then appends the PER-237 odd-one-out
comparative artifact filter after current tissue-containment classification.

Inputs:
- Stage 1 cases: {args.stage1_cases.resolve()}
- Stage 2b router results: {args.stage2b_results.resolve()}
- Stage 3 redetection results: {args.stage3_results.resolve()}
- Stage 1 crop-redetect prompt: {args.stage1_prompt.resolve()}
- Stage 6 classification prompt: {args.classification_prompt.resolve()}
- Odd-one-out prompt: {args.odd_one_out_prompt.resolve()}
- Selected case indices: {args.indices}

Pipeline:
1. Start from Stage 3 feedback-redetection boxes when Stage 2b triggered and
   Stage 3 exists; otherwise use raw Stage 1 rotation-{args.rotation} boxes.
2. Merge boxes with IoU > {args.merge_iou_threshold:.2f} or
   overlap-over-smaller-box >= {args.containment_overlap_threshold:.2f}, then
   expand by {args.post_stage3_padding_frac:.2f}.
3. Reread each expanded WSI crop near {args.max_dim}px max dimension and rerun
   the same Stage 1 high-recall detector prompt.
4. Map crop-relative boxes back to WSI coordinates, merge again with the same
   logic, then reread classification crops with {args.classification_padding_frac:.2f}
   padding.
5. Run the current Stage 6 tissue-containment classifier.
6. For cases with more than one remaining tissue-positive crop, crop the source
   thumbnail and run the PER-237 v2 odd-one-out comparative filter.

Command:
{command}

Outputs:
- Summary JSON: {(args.output_root / 'summary/pipeline_summary.json').resolve()}
- Review PDF: {pdf_path.resolve()}
- Stage 3 postprocess boxes: {(args.output_root / 'stage3_postprocess/stage3_postprocess_boxes.csv').resolve()}
- Crop-redetect results: {(args.output_root / 'stage4_crop_redetect/reviews/crop_redetect_results.jsonl').resolve()}
- Classification candidates: {(args.output_root / 'stage5_classification_inputs/classification_candidates.csv').resolve()}
- Classification results: {(args.output_root / 'stage6_classification/reviews/stage6_crop_tissue_artifact_high_thinking.jsonl').resolve()}
- Odd-one-out results: {(args.output_root / 'stage7_odd_one_out/reviews/odd_one_out_results.jsonl').resolve()}
- Final boxes: {(args.output_root / 'final_detections/final_boxes.csv').resolve()}

Summary:
{json.dumps(summary, indent=2, sort_keys=True)}
"""
    (args.output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage1_prompt = args.stage1_prompt.read_text().strip()
    classification_prompt = args.classification_prompt.read_text().strip()
    odd_prompt = args.odd_one_out_prompt.read_text().strip()
    (args.output_root / "prompts").mkdir(parents=True, exist_ok=True)
    (args.output_root / "prompts" / args.stage1_prompt.name).write_text(stage1_prompt + "\n")
    (args.output_root / "prompts" / args.classification_prompt.name).write_text(classification_prompt + "\n")
    (args.output_root / "prompts" / args.odd_one_out_prompt.name).write_text(odd_prompt + "\n")

    case_records, deterministic_summary = _build_post_stage3_inputs(args)
    crop_tasks = deterministic_summary["crop_redetect_tasks"]
    if args.dry_run:
        summary = {
            "dry_run": True,
            "cases": len(case_records),
            "crop_redetect_tasks": len(crop_tasks),
            "output_root": str(args.output_root.resolve()),
        }
        _write_json(args.output_root / "summary/pipeline_summary.json", summary)
        _write_reproduction(args, args.output_root / "visuals/not_created_dry_run.pdf", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    base_url, api_key = _api_settings(args)
    crop_results = _run_crop_redetect(crop_tasks, stage1_prompt, args, base_url, api_key)
    case_records, candidates = _build_classification_inputs(case_records, crop_results, args)
    classification_results = _run_classification(candidates, classification_prompt, args, base_url, api_key)
    case_records = _draw_classification_overlays(case_records, classification_results)
    odd_tasks, odd_skipped = _build_odd_one_out_inputs(case_records, classification_results, args)
    odd_results = _run_odd_one_out(odd_tasks, odd_prompt, args, base_url, api_key) if odd_tasks else []
    final_records, summary = _build_final_outputs(
        case_records,
        classification_results,
        odd_tasks,
        odd_results,
        odd_skipped,
        args,
    )
    summary.update(
        {
            "output_root": str(args.output_root.resolve()),
            "stage3_source_boxes": deterministic_summary["source_boxes"],
            "post_stage3_boxes": deterministic_summary["post_stage3_boxes"],
            "crop_redetect_tasks": len(crop_tasks),
            "crop_redetect_errors": sum(1 for row in crop_results if row.get("error")),
            "crop_redetect_detections": sum(int(row.get("detection_count") or 0) for row in crop_results),
            "classification_tasks": len(classification_results),
            "classification_errors": sum(1 for row in classification_results if row.get("error")),
            "odd_one_out_errors": sum(1 for row in odd_results if row.get("error")),
            "known_usage_cost_if_reported": sum(
                float((row.get("usage") or {}).get("cost") or 0.0)
                for row in [*crop_results, *classification_results, *odd_results]
            ),
        }
    )
    pdf_path = _write_pdf(final_records, summary, args)
    summary["pdf"] = str(pdf_path.resolve())
    _write_json(args.output_root / "summary/pipeline_summary.json", summary)
    _write_reproduction(args, pdf_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--stage2b-results", type=Path, default=STAGE2B_RESULTS)
    parser.add_argument("--stage3-results", type=Path, default=STAGE3_RESULTS)
    parser.add_argument("--stage1-prompt", type=Path, default=STAGE1_PROMPT)
    parser.add_argument("--classification-prompt", type=Path, default=STAGE6_PROMPT)
    parser.add_argument("--odd-one-out-prompt", type=Path, default=ODD_ONE_OUT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--post-stage3-padding-frac", type=float, default=0.15)
    parser.add_argument("--classification-padding-frac", type=float, default=0.10)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.40)
    parser.add_argument("--containment-overlap-threshold", type=float, default=0.80)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--wsi-reader", default="auto", choices=["auto", "openslide", "cucim", "isyntax"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--crop-redetect-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--classification-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--odd-one-out-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--crop-redetect-max-tokens", type=int, default=4000)
    parser.add_argument("--classification-max-tokens", type=int, default=800)
    parser.add_argument("--odd-one-out-max-tokens", type=int, default=16000)
    parser.add_argument("--use-stage3-when-available", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
