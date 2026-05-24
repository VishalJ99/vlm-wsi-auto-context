#!/usr/bin/env python3
"""Probe high-recall Stage 1 edge cases with focused reviewer/refiner calls."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _api_settings,
    _bbox_geometry,
    _chat_with_images,
    _draw_redetect_overlay,
    _draw_wrapped,
    _extract_json_object,
    _extract_json_payload,
    _font,
    _normalised_detection_items,
    _raw_overlay_bbox_text,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
    _write_csv,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HIGH_RECALL_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1"
)
DEFAULT_CASES_CSV = HIGH_RECALL_ROOT / "summary" / "high_recall_stage1_cases.csv"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_edge_review_v1"
)
HIGH_RECALL_PROMPT = REPO_ROOT / "prompts" / "stage1_high_recall_potential_tissue_candidates.txt"

ZERO_COVERAGE_PROMPT_VERSION = "stage1_high_recall_zero_coverage_review_v1_2026-05-24"
BBOX_GEOMETRY_PROMPT_VERSION = "stage1_high_recall_bbox_geometry_review_v1_2026-05-24"
SECOND_PASS_PROMPT_VERSION = "stage1_high_recall_edge_second_pass_v1_2026-05-24"
QUALITATIVE_REVIEW_PROMPT_VERSION = "stage1_high_recall_short_coverage_overcoverage_review_v1_2026-05-24"

QUALITATIVE_REVIEW_PROMPT = """\
You are looking at a whole-slide thumbnail and a tissue-detection overlay.

Check if any potential tissue-like objects were missed or if the detection failed to localise and instead just encompasses everything.
"""

ZERO_COVERAGE_PROMPT = """\
You are reviewing a tissue-detection result on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the detector overlay. It contains zero accepted bounding boxes.
- The text gives the detector status and raw detector response.

Task:
Decide whether the detector missed visible potential tissue-like foreground candidates.

Use only visual object-detection judgment:
- potential tissue-like foreground means visible tissue-colored foreground material, strips, fragments, clumps, faint tissue-colored material, or partial tissue regions at thumbnail scale.
- non-tissue/noise means glass edges, dust, pen marks, bubbles, smudges, debris, or blank background.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.
- Do not output refined coordinates in this review call.

Return only one JSON object with this exact shape:
{
  "coverage_review": {
    "detected_box_count": 0,
    "visible_potential_tissue": true,
    "missed_detection": true,
    "confidence": "high",
    "reasoning": "short visual explanation"
  }
}

Allowed confidence values: low, medium, high.
Set missed_detection=true if potential tissue-like foreground is visible but no bbox covers it.
"""

BBOX_GEOMETRY_PROMPT = """\
You are reviewing object-detection bounding boxes for potential tissue-like foreground on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the detector overlay with numbered bounding boxes.
- A text list gives every bbox id and its geometry.

Task:
For each bounding box, decide whether it is detecting visual signal or noise, and whether the bbox corners have a gross localization problem.

Use only visual object-detection judgment:
- signal: visible tissue-like foreground is present inside the bbox at thumbnail scale.
- noise: the bbox mainly covers background, glass marks, dust, pen, bubble, edge artifact, debris, or other non-tissue-like visual noise at thumbnail scale.
- ok: the bbox covers the visible signal with a reasonable margin.
- too_loose: the bbox includes excessive irrelevant background compared with the visible signal.
- too_tight: the bbox appears to cut off visible signal.
- giant_fallback: limit case of too_loose where the bbox covers most of the thumbnail or a very large slide area instead of localizing the signal.

Do not use pathology domain knowledge.
Do not infer control tissue, diagnosis, specimen type, or downstream handling.
Do not decide which tissue is more interesting.
Do not output refined coordinates in this review call.

Return only one JSON object with this exact shape:
{
  "bbox_geometry_review": {
    "bbox_count": 1,
    "overall_quality": "mixed",
    "needs_second_pass": true,
    "reasoning": "short summary"
  },
  "bbox_reviews": [
    {
      "bbox_id": "r0_01",
      "detection_signal": "signal",
      "localization": "too_loose",
      "suggested_action": "refine",
      "reasoning": "short visual reason"
    }
  ]
}

Allowed detection_signal values: signal, noise, uncertain.
Allowed localization values: ok, too_loose, too_tight, giant_fallback, uncertain.
Allowed suggested_action values: accept, refine, rerun, discard.
Allowed overall_quality values: ok, mixed, poor, uncertain.

Every bbox id from the text list must appear exactly once in bbox_reviews.
Set needs_second_pass=true if any bbox is not signal, any localization is not ok, or any suggested_action is not accept.
"""

SECOND_PASS_PROMPT = """\
You are making a second-pass object-detection decision for potential tissue-like foreground on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the first-pass detection overlay that was reviewed.
- The text contains the first-pass bbox geometry and reviewer feedback.

Task:
Use the reviewer feedback to produce corrected potential tissue-like foreground boxes.

Use only visual object-detection judgment:
- potential tissue-like foreground means visible tissue-colored foreground material, strips, fragments, clumps, faint tissue-colored material, or partial tissue regions at thumbnail scale.
- non-tissue/noise means glass edges, dust, pen marks, bubbles, smudges, debris, or blank background.
- If the first pass had zero boxes but the reviewer says visible potential tissue was missed, draw boxes around the visible potential tissue-like candidates.
- If a first-pass box is too loose or a giant fallback, rerun detection from the source thumbnail; do not preserve the loose extent.
- If a first-pass box is too tight, expand it so visible signal is not cut off.
- If a first-pass box is noise, discard it.
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
      "action": "refine",
      "reasoning": "short reason"
    }
  ]
}

Allowed action values: keep, refine, add_missed, discard_noise, rerun_from_source.
Coordinates must be normalized 0-1000 integers in [y_min, x_min, y_max, x_max] order.
If no valid potential tissue-like foreground is visible, return an empty final_detections list.
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _draw_empty_overlay(thumbnail_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(thumbnail_path).convert("RGB").save(output_path)


def _case_rows(cases_csv: Path, indices: list[int]) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(cases_csv) if int(row["case_index"]) in set(indices)]
    rows.sort(key=lambda row: int(row["case_index"]))
    missing = sorted(set(indices) - {int(row["case_index"]) for row in rows})
    if missing:
        raise SystemExit(f"Missing requested case indices in {cases_csv}: {missing}")
    return rows


def _parse_raw_response(row: dict[str, str], thumbnail_size: tuple[int, int]) -> list[dict[str, Any]]:
    path_text = row.get("raw_response_path", "")
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        return []
    payload = _extract_json_payload(path.read_text())
    detections = _normalised_detection_items(payload, thumbnail_size)
    for idx, detection in enumerate(detections, start=1):
        detection["label"] = f"r0_{idx:02d}"
    return detections


def _parse_final_bboxes(row: dict[str, str]) -> list[dict[str, Any]]:
    path_text = row.get("bboxes_json_path", "")
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    regions = payload.get("detected_regions")
    if not isinstance(regions, list):
        return []
    detections: list[dict[str, Any]] = []
    for idx, region in enumerate(regions, start=1):
        if not isinstance(region, dict) or not isinstance(region.get("bbox_thumbnail"), list):
            continue
        normalized = region.get("bbox_normalized")
        detections.append(
            {
                "label": f"f0_{idx:02d}",
                "source_label": str(region.get("label", "")),
                "raw_box_2d": normalized if isinstance(normalized, list) else [],
                "box_2d_yxyx_normalized": normalized if isinstance(normalized, list) else [],
                "coordinate_interpretation": "stage1_final_bbox_normalized",
                "parser_warnings": [],
                "bbox_thumbnail": [int(v) for v in region["bbox_thumbnail"]],
            }
        )
    return detections


def _task_kind(row: dict[str, str]) -> str:
    index = int(row["case_index"])
    if index == 22:
        return "zero_coverage"
    if index == 34:
        return "final_bbox_geometry"
    if index == 50:
        return "raw_giant_bbox_geometry"
    if row.get("raw_response_status") == "no_parseable_bbox_payload":
        return "zero_coverage"
    if row.get("raw_response_status") == "giant_bbox_rejected":
        return "raw_giant_bbox_geometry"
    return "final_bbox_geometry"


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in _case_rows(args.cases_csv.resolve(), args.indices):
        thumbnail_path = Path(row["thumbnail_path"])
        if not thumbnail_path.exists():
            raise SystemExit(f"Missing thumbnail for case {row['case_index']}: {thumbnail_path}")
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size
        raw_detections = _parse_raw_response(row, thumbnail_size)
        final_detections = _parse_final_bboxes(row)
        kind = _task_kind(row)
        if kind == "zero_coverage":
            reviewed_bboxes: list[dict[str, Any]] = []
            overlay_kind = "empty_overlay"
        elif kind == "raw_giant_bbox_geometry":
            reviewed_bboxes = raw_detections
            overlay_kind = "raw_response_overlay"
        else:
            reviewed_bboxes = final_detections
            overlay_kind = "final_stage1_overlay"
        case_slug = _safe_slug(row["case_display"])
        overlay_path = args.output_root / "review_overlays" / f"{case_slug}_{overlay_kind}.png"
        if reviewed_bboxes:
            _draw_redetect_overlay(thumbnail_path, reviewed_bboxes, overlay_path)
        else:
            _draw_empty_overlay(thumbnail_path, overlay_path)
        task = {
            "task_id": f"edge_review_{int(row['case_index']):03d}",
            "case_index": int(row["case_index"]),
            "case_display": row["case_display"],
            "kind": kind,
            "overlay_kind": overlay_kind,
            "thumbnail_path": str(thumbnail_path),
            "review_overlay_path": str(overlay_path),
            "thumbnail_size": list(thumbnail_size),
            "raw_response_status": row.get("raw_response_status", ""),
            "raw_response_path": row.get("raw_response_path", ""),
            "raw_response_excerpt": Path(row["raw_response_path"]).read_text().strip()[:1200]
            if row.get("raw_response_path") and Path(row["raw_response_path"]).exists()
            else "",
            "raw_response_box_count": int(row.get("raw_response_box_count") or 0),
            "final_count": int(row.get("final_count") or 0),
            "reviewed_bbox_count": len(reviewed_bboxes),
            "reviewed_bboxes": [
                {
                    "label": bbox.get("label", ""),
                    "source_label": bbox.get("source_label", ""),
                    "raw_box_2d": bbox.get("raw_box_2d", []),
                    "box_2d_yxyx_normalized": bbox.get("box_2d_yxyx_normalized", []),
                    "coordinate_interpretation": bbox.get("coordinate_interpretation", ""),
                    "parser_warnings": bbox.get("parser_warnings", []),
                    **_bbox_geometry(bbox, thumbnail_size),
                }
                for bbox in reviewed_bboxes
            ],
            "bbox_text": _raw_overlay_bbox_text(reviewed_bboxes, thumbnail_size) if reviewed_bboxes else "",
            "manifest_row": row,
            "created_at": _timestamp(),
        }
        tasks.append(task)
    return tasks


def _review_prompt(task: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if args.qualitative_only:
        return QUALITATIVE_REVIEW_PROMPT_VERSION, QUALITATIVE_REVIEW_PROMPT
    base = (
        "\n\nCase:\n"
        + task["case_display"]
        + f"\n\nReviewed overlay kind: {task['overlay_kind']}"
        + f"\nDetector raw status: {task['raw_response_status']}"
        + f"\nReviewed bbox count: {task['reviewed_bbox_count']}"
    )
    if task["kind"] == "zero_coverage":
        text = (
            ZERO_COVERAGE_PROMPT
            + base
            + "\n\nRaw detector response excerpt:\n"
            + task.get("raw_response_excerpt", "")
        )
        return ZERO_COVERAGE_PROMPT_VERSION, text
    text = (
        BBOX_GEOMETRY_PROMPT
        + base
        + "\n\nDetected bboxes:\n"
        + task.get("bbox_text", "")
    )
    return BBOX_GEOMETRY_PROMPT_VERSION, text


def _call_review(
    task: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    prompt_version, prompt_text = _review_prompt(task, args)
    record = {
        "task_id": task["task_id"],
        "case_index": task["case_index"],
        "case_display": task["case_display"],
        "kind": task["kind"],
        "overlay_kind": task["overlay_kind"],
        "prompt_version": prompt_version,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "qualitative_only": args.qualitative_only,
        "thumbnail_path": task["thumbnail_path"],
        "review_overlay_path": task["review_overlay_path"],
        "thumbnail_size": task["thumbnail_size"],
        "reviewed_bbox_count": task["reviewed_bbox_count"],
        "reviewed_bboxes": task["reviewed_bboxes"],
        "raw_response_status": task["raw_response_status"],
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt_text,
            image_paths=[Path(task["thumbnail_path"]), Path(task["review_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        record["raw_response"] = raw
        record["parsed_response"] = {"raw_text": raw} if args.qualitative_only else _extract_json_object(raw)
        record["usage"] = usage
        record["response_model"] = response_model
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def _coverage_review(parsed: dict[str, Any]) -> dict[str, Any]:
    value = parsed.get("coverage_review")
    return value if isinstance(value, dict) else {}


def _geometry_review(parsed: dict[str, Any]) -> dict[str, Any]:
    value = parsed.get("bbox_geometry_review")
    return value if isinstance(value, dict) else {}


def _bbox_reviews(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    value = parsed.get("bbox_reviews")
    return value if isinstance(value, list) else []


def _needs_second_pass(review: dict[str, Any]) -> tuple[bool, str]:
    if review.get("qualitative_only"):
        return False, "qualitative_only_no_second_pass"
    if review.get("error"):
        return False, "review_error"
    parsed = review.get("parsed_response") if isinstance(review.get("parsed_response"), dict) else {}
    coverage = _coverage_review(parsed)
    if coverage:
        if bool(coverage.get("missed_detection")):
            return True, "zero_box_missed_detection"
        return False, "zero_box_no_visible_miss"
    geometry = _geometry_review(parsed)
    reasons: list[str] = []
    if bool(geometry.get("needs_second_pass")):
        reasons.append("geometry_needs_second_pass")
    if geometry.get("overall_quality") and geometry.get("overall_quality") != "ok":
        reasons.append(f"overall_quality={geometry.get('overall_quality')}")
    for bbox in _bbox_reviews(parsed):
        bbox_id = str(bbox.get("bbox_id", ""))
        signal = str(bbox.get("detection_signal", ""))
        localization = str(bbox.get("localization", ""))
        action = str(bbox.get("suggested_action", ""))
        if signal and signal != "signal":
            reasons.append(f"{bbox_id}.signal={signal}")
        if localization and localization != "ok":
            reasons.append(f"{bbox_id}.localization={localization}")
        if action and action != "accept":
            reasons.append(f"{bbox_id}.action={action}")
    if reasons:
        return True, "; ".join(reasons)
    return False, "review_ok"


def _second_payload_to_detections(parsed: Any, thumbnail_size: tuple[int, int]) -> list[dict[str, Any]]:
    payload = parsed
    if isinstance(parsed, dict):
        for key in ("final_detections", "detections", "detected_regions", "bboxes", "boxes"):
            if isinstance(parsed.get(key), list):
                payload = parsed[key]
                break
    detections = _normalised_detection_items(payload, thumbnail_size)
    for idx, detection in enumerate(detections, start=1):
        detection["label"] = str(detection.get("label") or f"tissue_{idx}")
    return detections


def _call_second_pass(
    review: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    needs_second, trigger = _needs_second_pass(review)
    record = {
        "task_id": review["task_id"],
        "case_index": review["case_index"],
        "case_display": review["case_display"],
        "kind": review["kind"],
        "prompt_version": SECOND_PASS_PROMPT_VERSION,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "thumbnail_path": review["thumbnail_path"],
        "review_overlay_path": review["review_overlay_path"],
        "thumbnail_size": review["thumbnail_size"],
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
        + review["case_display"]
        + f"\n\nReviewed overlay kind: {review['overlay_kind']}"
        + "\n\nFirst-pass bbox geometry:\n"
        + json.dumps(review.get("reviewed_bboxes", []), indent=2, sort_keys=True)
        + "\n\nReviewer feedback:\n"
        + json.dumps(review.get("parsed_response", {}), indent=2, sort_keys=True)
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt_text,
            image_paths=[Path(review["thumbnail_path"]), Path(review["review_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.second_max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        payload = _extract_json_payload(raw)
        parsed = payload if isinstance(payload, dict) else {"final_detections": payload}
        thumbnail_size = tuple(int(v) for v in review["thumbnail_size"])
        detections = _second_payload_to_detections(parsed, thumbnail_size)
        case_slug = _safe_slug(review["case_display"])
        overlay_path = args.output_root / "second_pass_overlays" / f"{case_slug}_second_pass_overlay.png"
        _draw_redetect_overlay(Path(review["thumbnail_path"]), detections, overlay_path)
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
    results.sort(key=lambda row: row["case_index"])
    return results


def _summary_rows(
    reviews: list[dict[str, Any]],
    seconds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seconds_by_task = {row["task_id"]: row for row in seconds}
    case_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    for review in reviews:
        parsed = review.get("parsed_response") if isinstance(review.get("parsed_response"), dict) else {}
        coverage = _coverage_review(parsed)
        geometry = _geometry_review(parsed)
        needs_second, trigger = _needs_second_pass(review)
        second = seconds_by_task.get(review["task_id"], {})
        case_rows.append(
            {
                "case_index": review.get("case_index", ""),
                "case_display": review.get("case_display", ""),
                "kind": review.get("kind", ""),
                "qualitative_review": parsed.get("raw_text", ""),
                "reviewed_bbox_count": review.get("reviewed_bbox_count", ""),
                "review_error": review.get("error", ""),
                "visible_potential_tissue": coverage.get("visible_potential_tissue", ""),
                "missed_detection": coverage.get("missed_detection", ""),
                "geometry_quality": geometry.get("overall_quality", ""),
                "geometry_needs_second_pass": geometry.get("needs_second_pass", ""),
                "review_trigger": trigger,
                "needs_second_pass": needs_second,
                "ran_second_pass": second.get("ran_second_pass", ""),
                "second_pass_error": second.get("error", ""),
                "second_pass_detections": len(second.get("detections", []) or []),
                "thumbnail_path": review.get("thumbnail_path", ""),
                "review_overlay_path": review.get("review_overlay_path", ""),
                "second_overlay_path": second.get("second_overlay_path", ""),
            }
        )
        for bbox in _bbox_reviews(parsed):
            bbox_rows.append(
                {
                    "case_index": review.get("case_index", ""),
                    "case_display": review.get("case_display", ""),
                    "bbox_id": bbox.get("bbox_id", ""),
                    "detection_signal": bbox.get("detection_signal", ""),
                    "localization": bbox.get("localization", ""),
                    "suggested_action": bbox.get("suggested_action", ""),
                    "reasoning": bbox.get("reasoning", ""),
                }
            )
    return case_rows, bbox_rows


def _draw_parsed_review(
    draw: ImageDraw.ImageDraw,
    y: int,
    parsed: dict[str, Any],
    font: Any,
) -> int:
    raw_text = parsed.get("raw_text")
    if raw_text:
        return _draw_wrapped(draw, (60, y), str(raw_text), font, 180, "#111111")
    coverage = _coverage_review(parsed)
    if coverage:
        return _draw_wrapped(
            draw,
            (60, y),
            (
                f"coverage: visible={coverage.get('visible_potential_tissue')} "
                f"missed={coverage.get('missed_detection')} confidence={coverage.get('confidence')} | "
                f"{coverage.get('reasoning', '')}"
            ),
            font,
            180,
            "#111111",
        )
    geometry = _geometry_review(parsed)
    if geometry:
        y = _draw_wrapped(
            draw,
            (60, y),
            (
                f"geometry: quality={geometry.get('overall_quality')} "
                f"needs_second={geometry.get('needs_second_pass')} | {geometry.get('reasoning', '')}"
            ),
            font,
            180,
            "#111111",
        )
    for bbox in _bbox_reviews(parsed):
        y = _draw_wrapped(
            draw,
            (80, y),
            (
                f"{bbox.get('bbox_id')}: signal={bbox.get('detection_signal')} "
                f"localization={bbox.get('localization')} action={bbox.get('suggested_action')} | "
                f"{bbox.get('reasoning', '')}"
            ),
            font,
            170,
            "#111111",
        )
    return y


def _write_pdf(
    output_root: Path,
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    seconds: list[dict[str, Any]],
) -> None:
    reviews_by_task = {row["task_id"]: row for row in reviews}
    seconds_by_task = {row["task_id"]: row for row in seconds}
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)

    title = Image.new("RGB", (2200, 2600), "white")
    draw = ImageDraw.Draw(title)
    y = 45
    draw.text((45, y), "High-Recall Stage 1 Edge Reviewer Probe", font=title_font, fill="black")
    y += 50
    y = _draw_wrapped(
        draw,
        (45, y),
        (
            f"cases={','.join(str(i) for i in args.indices)} | model={args.model} | "
            f"reasoning={args.reasoning_effort or 'unspecified'} | "
            f"qualitative_only={args.qualitative_only} | ticket=PER-207"
        ),
        body_font,
        150,
        "#222222",
    )
    y += 20
    draw.text((45, y), "Detector prompt", font=body_font, fill="black")
    y += 32
    if HIGH_RECALL_PROMPT.exists():
        y = _draw_wrapped(draw, (65, y), HIGH_RECALL_PROMPT.read_text().strip(), small_font, 170, "#111111")
        y += 20
    draw.text((45, y), "Reviewer prompt versions", font=body_font, fill="black")
    y += 32
    _draw_wrapped(
        draw,
        (65, y),
        (
            f"{ZERO_COVERAGE_PROMPT_VERSION}; {BBOX_GEOMETRY_PROMPT_VERSION}; "
            f"{SECOND_PASS_PROMPT_VERSION}; {QUALITATIVE_REVIEW_PROMPT_VERSION}"
        ),
        small_font,
        170,
        "#111111",
    )
    if args.qualitative_only:
        y += 40
        draw.text((45, y), "Qualitative reviewer prompt", font=body_font, fill="black")
        y += 32
        _draw_wrapped(draw, (65, y), QUALITATIVE_REVIEW_PROMPT.strip(), small_font, 170, "#111111")
    pages.append(title)

    for task in tasks:
        review = reviews_by_task.get(task["task_id"], {})
        second = seconds_by_task.get(task["task_id"], {})
        parsed = review.get("parsed_response") if isinstance(review.get("parsed_response"), dict) else {}
        needs_second, trigger = _needs_second_pass(review) if review else (False, "not_run")
        page = Image.new("RGB", (2200, 2700), "white")
        draw = ImageDraw.Draw(page)
        y = 35
        draw.text((45, y), task["case_display"], font=title_font, fill="black")
        y += 44
        header = (
            f"kind={task['kind']} | reviewed_boxes={task['reviewed_bbox_count']} | "
            f"trigger={trigger} | qualitative_only={args.qualitative_only} | "
            f"second_pass={second.get('ran_second_pass', False)}"
        )
        y = _draw_wrapped(draw, (45, y), header, body_font, 170, "#111111")
        y += 18
        source = _thumb(Path(task["thumbnail_path"]), (660, 420))
        reviewed = _thumb(Path(task["review_overlay_path"]), (660, 420))
        if second.get("second_overlay_path"):
            second_overlay = _thumb(Path(second["second_overlay_path"]), (660, 420))
        else:
            second_overlay = Image.new("RGB", (660, 420), "#f7f7f7")
            ImageDraw.Draw(second_overlay).text((30, 185), "Second pass not run", font=body_font, fill="#555555")
        for x, label, image in (
            (45, "Source thumbnail", source),
            (770, "Reviewed overlay", reviewed),
            (1495, "Second-pass overlay", second_overlay),
        ):
            draw.text((x, y), label, font=body_font, fill="black")
            page.paste(image, (x, y + 30))
        y += 490
        draw.text((45, y), "Reviewer output", font=body_font, fill="black")
        y += 30
        if review.get("error"):
            y = _draw_wrapped(draw, (60, y), f"ERROR: {review['error']}", small_font, 180, "#aa0000")
        else:
            y = _draw_parsed_review(draw, y, parsed, small_font)
        y += 18
        draw.text((45, y), "Second pass output", font=body_font, fill="black")
        y += 30
        if not needs_second:
            y = _draw_wrapped(draw, (60, y), "Skipped because reviewer did not flag an actionable issue.", small_font, 180, "#555555")
        elif second.get("error"):
            y = _draw_wrapped(draw, (60, y), f"ERROR: {second.get('error')}", small_font, 180, "#aa0000")
        else:
            second_parsed = second.get("parsed_response") if isinstance(second.get("parsed_response"), dict) else {}
            second_meta = second_parsed.get("second_pass") if isinstance(second_parsed.get("second_pass"), dict) else {}
            y = _draw_wrapped(draw, (60, y), second_meta.get("summary", ""), small_font, 180, "#111111")
            y = _draw_wrapped(draw, (60, y), f"final detections: {len(second.get('detections', []) or [])}", small_font, 180, "#111111")
            for detection in (second.get("detections", []) or [])[:16]:
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

    pdf_path = output_root / "visuals" / "high_recall_edge_review_probe.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    tasks_path: Path,
    reviews_path: Path,
    seconds_path: Path,
) -> None:
    command = [
        "python",
        "scripts/stage1_high_recall_edge_review.py",
        "--cases-csv",
        str(args.cases_csv.resolve()),
        "--output-root",
        str(output_root),
        "--indices",
        ",".join(str(i) for i in args.indices),
        "--model",
        args.model,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--second-max-tokens",
        str(args.second_max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.reasoning_effort:
        command[command.index("--max-concurrent"):command.index("--max-concurrent")] = [
            "--reasoning-effort",
            args.reasoning_effort,
        ]
    if args.qualitative_only:
        command[command.index("--max-concurrent"):command.index("--max-concurrent")] = ["--qualitative-only"]
    text = f"""\
High-recall Stage 1 edge reviewer probe
=======================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Model: {args.model}
Reasoning effort: {args.reasoning_effort or 'unspecified'}
Qualitative only: {args.qualitative_only}
Max tokens: {args.max_tokens}
Second-pass max tokens: {args.second_max_tokens}
Cases CSV: {args.cases_csv.resolve()}
Case indices: {','.join(str(i) for i in args.indices)}

Command:
{" ".join(shlex.quote(part) for part in command)}

Detector prompt file:
{HIGH_RECALL_PROMPT}

Detector prompt text:
{HIGH_RECALL_PROMPT.read_text().strip() if HIGH_RECALL_PROMPT.exists() else ""}

Reviewer prompt versions:
- Zero-box coverage: {ZERO_COVERAGE_PROMPT_VERSION}
- Bbox geometry: {BBOX_GEOMETRY_PROMPT_VERSION}
- Second pass: {SECOND_PASS_PROMPT_VERSION}
- Qualitative review: {QUALITATIVE_REVIEW_PROMPT_VERSION}

Qualitative reviewer prompt text:
{QUALITATIVE_REVIEW_PROMPT.strip() if args.qualitative_only else 'not used'}

Outputs:
- Tasks: {tasks_path}
- First reviewer results: {reviews_path}
- Second-pass results: {seconds_path}
- Case summary: {output_root / 'summary' / 'edge_review_cases.csv'}
- Bbox summary: {output_root / 'summary' / 'edge_review_bboxes.csv'}
- Summary JSON: {output_root / 'summary' / 'edge_review_summary.json'}
- PDF: {output_root / 'visuals' / 'high_recall_edge_review_probe.pdf'}
- Reviewed overlays: {output_root / 'review_overlays'}
- Second-pass overlays: {output_root / 'second_pass_overlays'}

Notes:
- Case 22 is reviewed as a zero-box coverage failure candidate.
- Case 34 is reviewed on the final Stage 1 padded/merged bboxes, not only the raw detector boxes.
- Case 50 is reviewed on the raw giant bbox that Stage 1 rejected, to test whether the reviewer can catch a giant fallback box and route a second-pass redetection.
- Structured prompts avoid pathology/control-tissue semantics and use visual object-detection language only.
- Qualitative-only mode asks for a natural-language detection description and does not run the second-pass refiner.
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    if not args.cases_csv.exists():
        raise SystemExit(f"Cases CSV does not exist: {args.cases_csv}")
    tasks = _build_tasks(args)
    tasks_path = args.output_root / "tasks" / "edge_review_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "task_kinds": {task["case_index"]: task["kind"] for task in tasks},
                    "output_root": str(args.output_root),
                    "tasks_jsonl": str(tasks_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    reviews = _run_parallel(tasks, _call_review, args, base_url, api_key)
    reviews_path = args.output_root / "reviews" / "edge_review_results.jsonl"
    _write_jsonl(reviews_path, reviews)

    seconds = _run_parallel(reviews, _call_second_pass, args, base_url, api_key)
    seconds_path = args.output_root / "reviews" / "edge_second_pass_results.jsonl"
    _write_jsonl(seconds_path, seconds)

    case_rows, bbox_rows = _summary_rows(reviews, seconds)
    _write_csv(
        args.output_root / "summary" / "edge_review_cases.csv",
        case_rows,
        [
            "case_index",
            "case_display",
            "kind",
            "qualitative_review",
            "reviewed_bbox_count",
            "review_error",
            "visible_potential_tissue",
            "missed_detection",
            "geometry_quality",
            "geometry_needs_second_pass",
            "review_trigger",
            "needs_second_pass",
            "ran_second_pass",
            "second_pass_error",
            "second_pass_detections",
            "thumbnail_path",
            "review_overlay_path",
            "second_overlay_path",
        ],
    )
    _write_csv(
        args.output_root / "summary" / "edge_review_bboxes.csv",
        bbox_rows,
        ["case_index", "case_display", "bbox_id", "detection_signal", "localization", "suggested_action", "reasoning"],
    )
    summary = {
        "created_at": _timestamp(),
        "git_commit": _repo_git_commit(),
        "ticket": "PER-207",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "qualitative_only": args.qualitative_only,
        "max_tokens": args.max_tokens,
        "second_max_tokens": args.second_max_tokens,
        "cases": len(case_rows),
        "review_errors": sum(1 for row in reviews if row.get("error")),
        "second_pass_ran": sum(1 for row in seconds if row.get("ran_second_pass")),
        "second_pass_errors": sum(1 for row in seconds if row.get("error")),
        "second_pass_detections": sum(len(row.get("detections", []) or []) for row in seconds),
        "output_root": str(args.output_root),
    }
    _write_json(args.output_root / "summary" / "edge_review_summary.json", summary)
    _write_pdf(args.output_root, args, tasks, reviews, seconds)
    _write_reproduction(args.output_root, args, tasks_path, reviews_path, seconds_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, default=[22, 34, 50])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high", "xhigh", "none"], default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--second-max-tokens", type=int, default=1800)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--qualitative-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
