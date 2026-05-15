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
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for idx in starts:
        try:
            payload, _ = decoder.raw_decode(text[idx:])
            return payload
        except json.JSONDecodeError:
            continue
    return {"raw_text": text}


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
        if isinstance(payload.get("detected_regions"), list):
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
        coords = item.get("box_2d") or item.get("bbox_2d") or item.get("bbox")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            y1, x1, y2, x2 = [float(value) for value in coords]
        except (TypeError, ValueError):
            continue
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        bbox_thumbnail = [
            round(x1 / 1000.0 * width),
            round(y1 / 1000.0 * height),
            round(x2 / 1000.0 * width),
            round(y2 / 1000.0 * height),
        ]
        detections.append(
            {
                "label": str(item.get("label") or f"tissue_{idx}"),
                "box_2d_yxyx_normalized": [round(y1), round(x1), round(y2), round(x2)],
                "bbox_thumbnail": bbox_thumbnail,
            }
        )
    return detections


def _draw_redetect_overlay(thumbnail_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_thumbnail"]
        label = detection["label"]
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


def _slide_review(parsed: dict[str, Any]) -> dict[str, Any]:
    slide = parsed.get("slide_review")
    return slide if isinstance(slide, dict) else {}


def _bbox_reviews(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    reviews = parsed.get("bbox_reviews")
    return reviews if isinstance(reviews, list) else []


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

    summarize = sub.add_parser("summarize-detection-review", help="Regenerate summaries and PDF from existing results.")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    summarize.set_defaults(func=cmd_summarize_detection_review)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
