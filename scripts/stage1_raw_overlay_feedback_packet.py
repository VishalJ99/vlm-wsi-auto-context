#!/usr/bin/env python3
"""Run pilot-wide raw overlay review plus feedback-conditioned second pass."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    RAW_OVERLAY_REVIEW_PROMPT,
    RAW_OVERLAY_REVIEW_PROMPT_VERSION,
    _api_settings,
    _bbox_geometry,
    _case_display,
    _chat_with_images,
    _draw_redetect_overlay,
    _draw_wrapped,
    _extract_json_object,
    _extract_json_payload,
    _font,
    _load_raw_orientation_bboxes,
    _normalised_detection_items,
    _raw_overlay_bbox_text,
    _repo_git_commit,
    _safe_slug,
    _selected_rows,
    _thumb,
    _timestamp,
    _write_csv,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "raw_overlay_feedback_packet_rot0_v1"
)
COVERAGE_PROMPT_VERSION = "stage1_raw_zero_box_coverage_review_v1_2026-05-23"
SECOND_PASS_PROMPT_VERSION = "stage1_raw_overlay_feedback_second_pass_v1_2026-05-23"

COVERAGE_PROMPT = """\
You are reviewing a tissue-detection overlay on a whole-slide thumbnail.

Inputs:
- Image 1 is the raw detector orientation overlay. It contains zero parseable bounding boxes.
- The text gives the raw detector orientation and bbox count.

Task:
Decide whether the detector missed visible tissue-like foreground signal in the thumbnail.

Use only visual object-detection judgment:
- tissue-like signal means visible tissue-colored foreground material or tissue-like core fragments at thumbnail scale.
- non-tissue/noise means glass edges, dust, pen marks, bubbles, smudges, debris, or blank background.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.
- Do not output refined coordinates.

Return only one JSON object with this exact shape:
{
  "coverage_review": {
    "detected_box_count": 0,
    "visible_tissue_like_signal": true,
    "missed_detection": true,
    "confidence": "high",
    "reasoning": "short visual explanation"
  }
}

Allowed confidence values: low, medium, high.
Set missed_detection=true if visible tissue-like signal is present but not covered by any bbox.
"""

SECOND_PASS_PROMPT = """\
You are making a second-pass object-detection decision for tissue-like foreground on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the raw first-pass detection overlay.
- The text contains the first-pass bbox geometry and the reviewer feedback.

Task:
Use the reviewer feedback to decide the corrected tissue-like foreground boxes.
You may keep a box, refine a box, split a loose/merged box, discard a noise box, or add a missed tissue-like foreground box.

Use only visual object-detection judgment:
- tissue-like signal means visible tissue-colored foreground material or tissue-like core fragments at thumbnail scale.
- non-tissue/noise means glass edges, dust, pen marks, bubbles, smudges, debris, or blank background.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.

Return only one JSON object with this exact shape:
{
  "second_pass": {
    "needs_change": true,
    "summary": "short visual explanation"
  },
  "final_detections": [
    {
      "label": "tissue_1",
      "box_2d": [y_min, x_min, y_max, x_max],
      "source_bbox_ids": ["r0_01"],
      "action": "keep_or_refine",
      "reasoning": "short reason"
    }
  ],
  "discarded_bboxes": [
    {
      "bbox_id": "r0_03",
      "reasoning": "short reason"
    }
  ]
}

Allowed action values: keep_or_refine, split, add_missed, discard_noise.
Coordinates must be normalized 0-1000 integers in [y_min, x_min, y_max, x_max] order.
If no valid tissue-like foreground is visible, return an empty final_detections list.
"""


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _draw_empty_overlay(thumbnail_path: Path, output_path: Path) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        overlay_path = args.output_root / "raw_overlays" / f"{case_slug}_rot{args.rotation}_raw_overlay.png"
        if raw_bboxes:
            _draw_redetect_overlay(thumbnail_path, raw_bboxes, overlay_path)
        else:
            _draw_empty_overlay(thumbnail_path, overlay_path)
        task = {
            "task_id": f"raw_feedback_packet_{int(row['index']):03d}_rot{args.rotation}",
            "case_index": int(row["index"]),
            "case_display": _case_display(row),
            "manifest_row": row,
            "rotation": args.rotation,
            "thumbnail_path": str(thumbnail_path),
            "raw_overlay_path": str(overlay_path),
            "bboxes_json_path": str(bboxes_json_path),
            "thumbnail_size": list(thumbnail_size),
            "bbox_count": len(raw_bboxes),
            "raw_bboxes": [
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
            "bbox_text": _raw_overlay_bbox_text(raw_bboxes, thumbnail_size) if raw_bboxes else "",
            "zero_box_reason": skip_reason if not raw_bboxes else "",
            "created_at": _timestamp(),
        }
        tasks.append(task)
    return tasks


def _first_prompt(task: dict[str, Any]) -> tuple[str, str]:
    if int(task["bbox_count"]) == 0:
        text = (
            COVERAGE_PROMPT
            + "\n\nCase:\n"
            + task["case_display"]
            + f"\n\nReviewed detector orientation: rot{task['rotation']} only."
            + f"\n\nDetected bbox count: {task['bbox_count']}"
            + f"\nZero-box parser note: {task.get('zero_box_reason', '')}"
        )
        return COVERAGE_PROMPT_VERSION, text
    text = (
        RAW_OVERLAY_REVIEW_PROMPT
        + "\n\nCase:\n"
        + task["case_display"]
        + f"\n\nReviewed detector orientation: rot{task['rotation']} only."
        + "\n\nDetected bboxes:\n"
        + task["bbox_text"]
    )
    return RAW_OVERLAY_REVIEW_PROMPT_VERSION, text


def _call_first_review(
    task: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    prompt_version, prompt_text = _first_prompt(task)
    record = {
        "task_id": task["task_id"],
        "case_index": task["case_index"],
        "case_display": task["case_display"],
        "prompt_version": prompt_version,
        "model": args.model,
        "rotation": task["rotation"],
        "thumbnail_path": task["thumbnail_path"],
        "raw_overlay_path": task["raw_overlay_path"],
        "bboxes_json_path": task["bboxes_json_path"],
        "thumbnail_size": task["thumbnail_size"],
        "bbox_count": task["bbox_count"],
        "raw_bboxes": task["raw_bboxes"],
        "zero_box_reason": task.get("zero_box_reason", ""),
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt_text,
            image_paths=[Path(task["raw_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        record["raw_response"] = raw
        record["parsed_response"] = _extract_json_object(raw)
        record["usage"] = usage
        record["response_model"] = response_model
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def _bbox_reviews(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    rows = parsed.get("bbox_reviews")
    return rows if isinstance(rows, list) else []


def _coverage_review(parsed: dict[str, Any]) -> dict[str, Any]:
    review = parsed.get("coverage_review")
    return review if isinstance(review, dict) else {}


def _overlay_review(parsed: dict[str, Any]) -> dict[str, Any]:
    review = parsed.get("overlay_review")
    return review if isinstance(review, dict) else {}


def _needs_second_pass(first: dict[str, Any]) -> tuple[bool, str]:
    if first.get("error"):
        return False, "first_review_error"
    parsed = first.get("parsed_response") if isinstance(first.get("parsed_response"), dict) else {}
    if int(first.get("bbox_count") or 0) == 0:
        coverage = _coverage_review(parsed)
        if bool(coverage.get("missed_detection")):
            return True, "zero_box_missed_detection"
        return False, "zero_box_no_miss_feedback"
    overlay = _overlay_review(parsed)
    reasons: list[str] = []
    if overlay.get("overall_quality") and overlay.get("overall_quality") != "ok":
        reasons.append(f"overall_quality={overlay.get('overall_quality')}")
    for bbox in _bbox_reviews(parsed):
        tightness = str(bbox.get("tightness", ""))
        signal = str(bbox.get("detection_signal", ""))
        bbox_id = bbox.get("bbox_id", "")
        if tightness and tightness != "ok":
            reasons.append(f"{bbox_id}.tightness={tightness}")
        if signal and signal != "signal":
            reasons.append(f"{bbox_id}.signal={signal}")
    if reasons:
        return True, "; ".join(reasons)
    return False, "all_reviewed_bboxes_ok_signal"


def _reviewer_feedback_text(first: dict[str, Any]) -> str:
    parsed = first.get("parsed_response") if isinstance(first.get("parsed_response"), dict) else {}
    payload = {
        "case": first.get("case_display", ""),
        "rotation": first.get("rotation", ""),
        "bbox_count": first.get("bbox_count", ""),
        "zero_box_reason": first.get("zero_box_reason", ""),
        "reviewer_response": parsed,
        "raw_bbox_geometry": first.get("raw_bboxes", []),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _second_payload_to_detections(parsed: Any, thumbnail_size: tuple[int, int]) -> list[dict[str, Any]]:
    payload = parsed
    if isinstance(parsed, dict):
        for key in ("final_detections", "detections", "detected_regions", "bboxes", "boxes"):
            if isinstance(parsed.get(key), list):
                payload = parsed[key]
                break
    detections = _normalised_detection_items(payload, thumbnail_size)
    for idx, detection in enumerate(detections, start=1):
        if not detection.get("label"):
            detection["label"] = f"tissue_{idx}"
    return detections


def _call_second_pass(
    first: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    needs_second, trigger = _needs_second_pass(first)
    record = {
        "task_id": first["task_id"],
        "case_index": first["case_index"],
        "case_display": first["case_display"],
        "prompt_version": SECOND_PASS_PROMPT_VERSION,
        "model": args.model,
        "rotation": first["rotation"],
        "thumbnail_path": first["thumbnail_path"],
        "raw_overlay_path": first["raw_overlay_path"],
        "thumbnail_size": first["thumbnail_size"],
        "review_trigger": trigger,
        "ran_second_pass": needs_second,
        "created_at": _timestamp(),
        "error": "",
    }
    if not needs_second:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["detections"] = []
        record["second_overlay_path"] = ""
        record["usage"] = {}
        record["response_model"] = ""
        return record

    prompt_text = (
        SECOND_PASS_PROMPT
        + "\n\nCase:\n"
        + first["case_display"]
        + f"\n\nReviewed detector orientation: rot{first['rotation']} only."
        + "\n\nFirst-pass raw bbox geometry:\n"
        + json.dumps(first.get("raw_bboxes", []), indent=2, sort_keys=True)
        + "\n\nReviewer feedback:\n"
        + _reviewer_feedback_text(first)
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt_text,
            image_paths=[Path(first["thumbnail_path"]), Path(first["raw_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.second_max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        payload = _extract_json_payload(raw)
        parsed = payload if isinstance(payload, dict) else {"final_detections": payload}
        thumbnail_size = tuple(int(v) for v in first["thumbnail_size"])
        detections = _second_payload_to_detections(parsed, thumbnail_size)
        case_slug = _safe_slug(f"{int(first['case_index']):03d}_{Path(first['thumbnail_path']).stem}")
        overlay_path = (
            args.output_root
            / "second_pass_overlays"
            / f"{case_slug}_rot{first['rotation']}_second_pass_overlay.png"
        )
        _draw_redetect_overlay(Path(first["thumbnail_path"]), detections, overlay_path)
        record["raw_response"] = raw
        record["parsed_response"] = parsed
        record["detections"] = detections
        record["second_overlay_path"] = str(overlay_path)
        record["usage"] = usage
        record["response_model"] = response_model
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["detections"] = []
        record["second_overlay_path"] = ""
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def _run_parallel(
    items: list[dict[str, Any]],
    fn: Any,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    if args.max_concurrent <= 1:
        return [fn(item, args, base_url, api_key) for item in items]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = [pool.submit(fn, item, args, base_url, api_key) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_index"])
    return results


def _flat_rows(
    first_results: list[dict[str, Any]],
    second_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    second_by_task = {row["task_id"]: row for row in second_results}
    case_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    for first in first_results:
        parsed = first.get("parsed_response") if isinstance(first.get("parsed_response"), dict) else {}
        overlay = _overlay_review(parsed)
        coverage = _coverage_review(parsed)
        needs_second, trigger = _needs_second_pass(first)
        second = second_by_task.get(first["task_id"], {})
        case_rows.append(
            {
                "case_index": first.get("case_index", ""),
                "case_display": first.get("case_display", ""),
                "rotation": first.get("rotation", ""),
                "bbox_count": first.get("bbox_count", ""),
                "first_review_error": first.get("error", ""),
                "overall_quality": overlay.get("overall_quality", ""),
                "coverage_missed_detection": coverage.get("missed_detection", ""),
                "review_trigger": trigger,
                "needs_second_pass": needs_second,
                "ran_second_pass": second.get("ran_second_pass", ""),
                "second_pass_error": second.get("error", ""),
                "second_pass_detections": len(second.get("detections", []) or []),
                "thumbnail_path": first.get("thumbnail_path", ""),
                "raw_overlay_path": first.get("raw_overlay_path", ""),
                "second_overlay_path": second.get("second_overlay_path", ""),
            }
        )
        for bbox in _bbox_reviews(parsed):
            bbox_rows.append(
                {
                    "case_index": first.get("case_index", ""),
                    "case_display": first.get("case_display", ""),
                    "bbox_id": bbox.get("bbox_id", ""),
                    "tightness": bbox.get("tightness", ""),
                    "detection_signal": bbox.get("detection_signal", ""),
                    "reasoning": bbox.get("reasoning", ""),
                }
            )
    return case_rows, bbox_rows


def _write_packet_pdf(
    output_root: Path,
    first_results: list[dict[str, Any]],
    second_results: list[dict[str, Any]],
) -> None:
    second_by_task = {row["task_id"]: row for row in second_results}
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    for first in first_results:
        second = second_by_task.get(first["task_id"], {})
        parsed = first.get("parsed_response") if isinstance(first.get("parsed_response"), dict) else {}
        overlay = _overlay_review(parsed)
        coverage = _coverage_review(parsed)
        needs_second, trigger = _needs_second_pass(first)

        page = Image.new("RGB", (2200, 3000), "white")
        draw = ImageDraw.Draw(page)
        y = 35
        draw.text((45, y), first.get("case_display", first["task_id"]), font=title_font, fill="black")
        y += 46
        header = (
            f"raw rot{first.get('rotation')} | boxes={first.get('bbox_count')} | "
            f"quality={overlay.get('overall_quality', '')} | "
            f"missed={coverage.get('missed_detection', '')} | "
            f"second_pass={second.get('ran_second_pass', False)}"
        )
        draw.text((45, y), header, font=body_font, fill="black")
        y += 34
        draw.text((45, y), f"trigger: {trigger}", font=small_font, fill="#222222")
        y += 28

        source = _thumb(Path(first["thumbnail_path"]), (660, 420))
        raw_overlay = _thumb(Path(first["raw_overlay_path"]), (660, 420))
        if second.get("second_overlay_path"):
            second_overlay = _thumb(Path(second["second_overlay_path"]), (660, 420))
        else:
            second_overlay = Image.new("RGB", (660, 420), "#f7f7f7")
            ImageDraw.Draw(second_overlay).text((30, 185), "Second pass not run", font=body_font, fill="#555555")
        for x, title, image in (
            (45, "Source thumbnail", source),
            (770, "Raw first-pass overlay", raw_overlay),
            (1495, "Second-pass overlay", second_overlay),
        ):
            draw.text((x, y), title, font=body_font, fill="black")
            page.paste(image, (x, y + 30))
        y += 485

        if int(first.get("bbox_count") or 0) == 0:
            reason = coverage.get("reasoning", "")
            draw.text((45, y), "Coverage review", font=body_font, fill="black")
            y += 28
            y = _draw_wrapped(draw, (60, y), reason, small_font, 180, "#111111")
        else:
            draw.text((45, y), "BBox reviewer feedback", font=body_font, fill="black")
            y += 30
            reason = overlay.get("reasoning", "")
            if reason:
                y = _draw_wrapped(draw, (60, y), f"overall: {reason}", small_font, 180, "#111111")
            for bbox in _bbox_reviews(parsed):
                line = (
                    f"{bbox.get('bbox_id')}: tightness={bbox.get('tightness')} / "
                    f"signal={bbox.get('detection_signal')} | {bbox.get('reasoning')}"
                )
                y = _draw_wrapped(draw, (60, y), line, small_font, 180, "#111111")
                if y > 1590:
                    y = _draw_wrapped(draw, (60, y), "... [truncated; see CSV/JSON]", small_font, 180, "#555555")
                    break
        y += 18

        draw.text((45, y), "Second-pass decision", font=body_font, fill="black")
        y += 30
        if not needs_second:
            y = _draw_wrapped(draw, (60, y), "Skipped because the reviewer did not flag actionable feedback.", small_font, 180, "#555555")
        elif second.get("error"):
            y = _draw_wrapped(draw, (60, y), f"ERROR: {second.get('error')}", small_font, 180, "#aa0000")
        else:
            second_parsed = second.get("parsed_response") if isinstance(second.get("parsed_response"), dict) else {}
            second_meta = second_parsed.get("second_pass") if isinstance(second_parsed.get("second_pass"), dict) else {}
            y = _draw_wrapped(draw, (60, y), second_meta.get("summary", ""), small_font, 180, "#111111")
            detections = second.get("detections", []) or []
            y = _draw_wrapped(draw, (60, y), f"final detections: {len(detections)}", small_font, 180, "#111111")
            for detection in detections[:12]:
                y = _draw_wrapped(
                    draw,
                    (80, y),
                    json.dumps(
                        {
                            "label": detection.get("label", ""),
                            "box_2d": detection.get("box_2d_yxyx_normalized", []),
                            "bbox_thumbnail": detection.get("bbox_thumbnail", []),
                        },
                        sort_keys=True,
                    ),
                    small_font,
                    170,
                    "#111111",
                )
        pages.append(page)

    pdf_path = output_root / "visuals" / "raw_overlay_feedback_packet.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    tasks_path: Path,
    first_path: Path,
    second_path: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    text = f"""\
Stage 1 raw overlay reviewer plus feedback second-pass packet
=============================================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Model: {args.model}
First-pass bbox reviewer prompt version: {RAW_OVERLAY_REVIEW_PROMPT_VERSION}
Zero-box coverage reviewer prompt version: {COVERAGE_PROMPT_VERSION}
Second-pass prompt version: {SECOND_PASS_PROMPT_VERSION}
Manifest: {args.manifest.resolve()}
Detector orientation reviewed: rot{args.rotation}
Task indices: {','.join(str(i) for i in args.indices)}

Command:
python scripts/stage1_raw_overlay_feedback_packet.py \\
  --manifest {args.manifest.resolve()} \\
  --output-root {output_root} \\
  --indices {','.join(str(i) for i in args.indices)} \\
  --rotation {args.rotation} \\
  --model {args.model} \\
  --max-concurrent {args.max_concurrent} \\
  --temperature {args.temperature}

Outputs:
- Tasks: {tasks_path}
- First-pass reviewer results: {first_path}
- Second-pass results: {second_path}
- Case summary: {output_root / 'summary' / 'raw_overlay_feedback_cases.csv'}
- Bbox review summary: {output_root / 'summary' / 'raw_overlay_feedback_bboxes.csv'}
- Summary JSON: {output_root / 'summary' / 'raw_overlay_feedback_summary.json'}
- PDF: {output_root / 'visuals' / 'raw_overlay_feedback_packet.pdf'}
- Raw overlays: {output_root / 'raw_overlays'}
- Second-pass overlays: {output_root / 'second_pass_overlays'}

Notes:
- This pass uses raw single-orientation detector outputs from `per_orientation_raw`.
- For cases with parseable raw bboxes, the first pass reviews per-bbox tightness and signal/noise only.
- For cases with zero parseable raw bboxes, the first pass uses a zero-box coverage reviewer so missed detections can still trigger a second pass.
- The second Gemini call is run only when the first reviewer flags actionable feedback.
- Prompts do not use pathology/control-tissue semantics.
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    tasks = _build_tasks(args)
    tasks_path = args.output_root / "tasks" / "raw_overlay_feedback_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "with_raw_bboxes": sum(1 for task in tasks if int(task["bbox_count"]) > 0),
                    "zero_box_tasks": sum(1 for task in tasks if int(task["bbox_count"]) == 0),
                    "tasks_jsonl": str(tasks_path),
                    "output_root": str(args.output_root),
                },
                indent=2,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    first_results = _run_parallel(tasks, _call_first_review, args, base_url, api_key)
    first_path = args.output_root / "reviews" / "first_pass_review_results.jsonl"
    _write_jsonl(first_path, first_results)

    second_results = _run_parallel(first_results, _call_second_pass, args, base_url, api_key)
    second_path = args.output_root / "reviews" / "second_pass_results.jsonl"
    _write_jsonl(second_path, second_results)

    case_rows, bbox_rows = _flat_rows(first_results, second_results)
    _write_csv(
        args.output_root / "summary" / "raw_overlay_feedback_cases.csv",
        case_rows,
        [
            "case_index",
            "case_display",
            "rotation",
            "bbox_count",
            "first_review_error",
            "overall_quality",
            "coverage_missed_detection",
            "review_trigger",
            "needs_second_pass",
            "ran_second_pass",
            "second_pass_error",
            "second_pass_detections",
            "thumbnail_path",
            "raw_overlay_path",
            "second_overlay_path",
        ],
    )
    _write_csv(
        args.output_root / "summary" / "raw_overlay_feedback_bboxes.csv",
        bbox_rows,
        ["case_index", "case_display", "bbox_id", "tightness", "detection_signal", "reasoning"],
    )
    summary = {
        "tasks": len(tasks),
        "first_review_errors": sum(1 for row in first_results if row.get("error")),
        "second_pass_ran": sum(1 for row in second_results if row.get("ran_second_pass")),
        "second_pass_errors": sum(1 for row in second_results if row.get("error")),
        "second_pass_detections": sum(len(row.get("detections", []) or []) for row in second_results),
        "zero_box_tasks": sum(1 for task in tasks if int(task["bbox_count"]) == 0),
        "bbox_review_rows": len(bbox_rows),
    }
    _write_json(args.output_root / "summary" / "raw_overlay_feedback_summary.json", summary)
    _write_packet_pdf(args.output_root, first_results, second_results)
    _write_reproduction(args.output_root, args, tasks_path, first_path, second_path)
    print(json.dumps({**summary, "output_root": str(args.output_root)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, default=list(range(1, 101)))
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--second-max-tokens", type=int, default=1800)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
