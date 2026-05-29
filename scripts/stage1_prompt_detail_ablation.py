#!/usr/bin/env python3
"""Compare detailed versus simplified Stage 1 detector prompts on thumbnails."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _chat_with_images,
    _draw_redetect_overlay,
    _extract_json_payload,
    _font,
    _normalised_detection_items,
    _safe_slug,
    _thumb,
    _write_csv,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1"
    / "summary"
    / "high_recall_stage1_cases.csv"
)
DEFAULT_DETAILED_PROMPT = (
    REPO_ROOT
    / "prompts"
    / "stage1_detector_oracle"
    / "stage1_high_recall_potential_tissue_candidates.txt"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "stage1_prompt_ablation_20cases_detail_vs_simple_flash_v1"
)
DEFAULT_SAMPLE_SEED = 2070529
DEFAULT_SIMPLE_PROMPT = """\
You are looking at a whole-slide thumbnail at low magnification.
Detect potential tissue-like foreground candidates.
Output JSON only as an array of bounding boxes in normalized 0-1000 coordinates:
[{"box_2d": [y_min, x_min, y_max, x_max]}]
"""
PROMPT_KEYS = ("detailed", "simple")


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_status(raw_text: str, payload: Any, detections: list[dict[str, Any]]) -> str:
    text = raw_text.strip()
    if not text:
        return "empty_response"
    try:
        direct = json.loads(text)
        if isinstance(direct, list) and not direct:
            return "valid_empty_json_array"
        if isinstance(direct, (list, dict)):
            return "valid_json"
    except json.JSONDecodeError:
        pass
    if isinstance(payload, dict) and "raw_text" in payload and not detections:
        return "no_parseable_bbox_payload"
    if isinstance(payload, list) and not payload:
        return "recovered_empty_json_array"
    if detections:
        return "recovered_json"
    return "parse_failed"


def _case_slug(row: dict[str, str]) -> str:
    stem = Path(row.get("metadata_path") or row.get("thumbnail_path") or row.get("case_display", "")).stem
    return _safe_slug(f"{int(row['case_index']):03d}_{stem}")


def _load_wsi_path(row: dict[str, str]) -> str:
    metadata_path = row.get("metadata_path", "")
    if metadata_path and Path(metadata_path).exists():
        try:
            metadata = _read_json(Path(metadata_path))
            return str(metadata.get("wsi_path") or "")
        except Exception:
            return ""
    return ""


def _sample_cases(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("thumbnail_path")
        and Path(row["thumbnail_path"]).exists()
        and row.get("metadata_path")
        and Path(row["metadata_path"]).exists()
    ]
    if len(ok_rows) < sample_size:
        raise SystemExit(f"Need {sample_size} usable cases, found {len(ok_rows)} in {DEFAULT_CASES}")
    selected = random.Random(seed).sample(ok_rows, sample_size)
    return sorted(selected, key=lambda row: int(row["case_index"]))


def _draw_labeled_overlay(
    thumbnail_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
    color: str,
) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    for idx, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox_thumbnail"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label = str(idx)
        label_box = draw.textbbox((x1 + 4, y1 + 4), label, font=font)
        draw.rectangle(label_box, fill="white", outline=color, width=2)
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_counts(
    detailed: list[dict[str, Any]],
    simple: list[dict[str, Any]],
    threshold: float,
) -> dict[str, int]:
    pairs: list[tuple[float, int, int]] = []
    for detail_idx, detail in enumerate(detailed):
        for simple_idx, simple_item in enumerate(simple):
            iou = _bbox_iou(detail["bbox_thumbnail"], simple_item["bbox_thumbnail"])
            if iou >= threshold:
                pairs.append((iou, detail_idx, simple_idx))
    matched_detail: set[int] = set()
    matched_simple: set[int] = set()
    for _, detail_idx, simple_idx in sorted(pairs, reverse=True):
        if detail_idx in matched_detail or simple_idx in matched_simple:
            continue
        matched_detail.add(detail_idx)
        matched_simple.add(simple_idx)
    return {
        "matched_boxes_iou_ge_threshold": len(matched_detail),
        "detail_only_boxes": len(detailed) - len(matched_detail),
        "simple_only_boxes": len(simple) - len(matched_simple),
    }


def _task_paths(output_root: Path, slug: str, prompt_key: str) -> dict[str, Path]:
    return {
        "raw_response": output_root / "raw_responses" / prompt_key / f"{slug}.txt",
        "parsed_json": output_root / "parsed" / prompt_key / f"{slug}.json",
        "overlay": output_root / "overlays" / prompt_key / f"{slug}.png",
    }


def _run_prompt(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt_key = task["prompt_key"]
    paths = _task_paths(args.output_root, task["case_slug"], prompt_key)
    if paths["parsed_json"].exists() and paths["raw_response"].exists() and not args.force:
        raw_text = paths["raw_response"].read_text()
        parsed_record = _read_json(paths["parsed_json"])
        payload = parsed_record.get("payload")
        detections = parsed_record.get("detections", [])
        parse_status = _json_status(raw_text, payload, detections)
        if parse_status != parsed_record.get("parse_status"):
            parsed_record["parse_status"] = parse_status
            _write_json(paths["parsed_json"], parsed_record)
        return {
            **task,
            "status": parsed_record.get("status", "skipped_existing"),
            "raw_response_path": str(paths["raw_response"]),
            "parsed_json_path": str(paths["parsed_json"]),
            "overlay_path": str(paths["overlay"]) if paths["overlay"].exists() else "",
            "raw_response": raw_text,
            "payload": payload,
            "detections": detections,
            "parse_status": parse_status,
            "usage": parsed_record.get("usage", {}),
            "response_model": parsed_record.get("response_model", ""),
            "error": parsed_record.get("error", ""),
        }

    try:
        raw_text, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=task["prompt_text"],
            image_paths=[Path(task["thumbnail_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.api_base,
            api_key=args.api_key,
            reasoning_effort=args.reasoning_effort,
        )
        paths["raw_response"].parent.mkdir(parents=True, exist_ok=True)
        paths["raw_response"].write_text(raw_text)
        with Image.open(task["thumbnail_path"]) as image:
            thumbnail_size = image.size
        payload = _extract_json_payload(raw_text)
        detections = _normalised_detection_items(payload, thumbnail_size)
        parse_status = _json_status(raw_text, payload, detections)
        _draw_labeled_overlay(
            Path(task["thumbnail_path"]),
            detections,
            paths["overlay"],
            "red" if prompt_key == "detailed" else "blue",
        )
        record = {
            "status": "ok",
            "case_index": task["case_index"],
            "case_display": task["case_display"],
            "case_slug": task["case_slug"],
            "prompt_key": prompt_key,
            "model": args.model,
            "response_model": response_model,
            "usage": usage,
            "parse_status": parse_status,
            "bbox_count": len(detections),
            "payload": payload,
            "detections": detections,
            "raw_response_path": str(paths["raw_response"]),
            "overlay_path": str(paths["overlay"]),
            "error": "",
        }
        _write_json(paths["parsed_json"], record)
        return {
            **task,
            **record,
            "parsed_json_path": str(paths["parsed_json"]),
            "raw_response": raw_text,
        }
    except Exception as exc:
        record = {
            "status": "error",
            "case_index": task["case_index"],
            "case_display": task["case_display"],
            "case_slug": task["case_slug"],
            "prompt_key": prompt_key,
            "model": args.model,
            "response_model": "",
            "usage": {},
            "parse_status": "api_or_runtime_error",
            "bbox_count": 0,
            "payload": {},
            "detections": [],
            "raw_response_path": str(paths["raw_response"]),
            "overlay_path": "",
            "error": repr(exc),
        }
        _write_json(paths["parsed_json"], record)
        return {**task, **record, "parsed_json_path": str(paths["parsed_json"]), "raw_response": ""}


def _blank_panel(size: tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, "#f5f5f5")
    draw = ImageDraw.Draw(image)
    font = _font(18)
    draw.text((20, 20), text, fill="#555555", font=font)
    return image


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, width: int, fill: str) -> int:
    x, y = xy
    line_height = int(getattr(font, "size", 16) * 1.35)
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height
        y += 4
    return y


def _write_side_by_side_png(output_root: Path, row: dict[str, Any]) -> str:
    page = Image.new("RGB", (2200, 1450), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    y = 35
    y = _draw_wrapped(draw, (45, y), row["case_display"], title_font, 120, "black") + 4
    header = (
        f"detail={row['detailed_count']} | simple={row['simple_count']} | "
        f"delta={row['count_delta_simple_minus_detailed']} | "
        f"detail_only={row['detail_only_boxes']} | simple_only={row['simple_only_boxes']} | "
        f"status={row['detailed_parse_status']} / {row['simple_parse_status']}"
    )
    y = _draw_wrapped(draw, (45, y), header, body_font, 145, "#222222") + 16
    if row.get("wsi_path"):
        y = _draw_wrapped(draw, (45, y), row["wsi_path"], small_font, 160, "#333333") + 10

    panels = [
        ("Source thumbnail", row["thumbnail_path"]),
        ("Detailed prompt overlay", row["detailed_overlay_path"]),
        ("Simplified prompt overlay", row["simple_overlay_path"]),
    ]
    x_positions = [45, 770, 1495]
    for x, (label, path_text) in zip(x_positions, panels):
        draw.text((x, y), label, font=body_font, fill="black")
        if path_text and Path(path_text).exists():
            panel = _thumb(Path(path_text), (660, 420))
        else:
            panel = _blank_panel((660, 420), "missing")
        page.paste(panel, (x, y + 32))

    y += 500
    left = {
        "detailed_json": row.get("detailed_parsed_json_path", ""),
        "simple_json": row.get("simple_parsed_json_path", ""),
        "detailed_raw": row.get("detailed_raw_response_path", ""),
        "simple_raw": row.get("simple_raw_response_path", ""),
    }
    _draw_wrapped(draw, (45, y), json.dumps(left, indent=2), small_font, 150, "#111111")
    output_path = output_root / "visuals" / "case_pages" / f"{row['case_slug']}_detail_vs_simple.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path)
    return str(output_path)


def _write_pdf(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    pages: list[Image.Image] = []
    title_font = _font(30)
    body_font = _font(18)
    small_font = _font(14)

    title = Image.new("RGB", (2200, 1800), "white")
    draw = ImageDraw.Draw(title)
    y = 45
    draw.text((45, y), "Stage 1 Prompt Detail Ablation", font=title_font, fill="black")
    y += 50
    meta = (
        f"cases={len(rows)} | seed={args.sample_seed} | model={args.model} | "
        f"temperature={args.temperature} | source={args.cases.resolve()}"
    )
    y = _draw_wrapped(draw, (45, y), meta, body_font, 150, "#222222") + 20
    summary = _summary_from_rows(rows, args)
    y = _draw_wrapped(draw, (45, y), json.dumps(summary, indent=2, sort_keys=True), small_font, 160, "#111111") + 20
    draw.text((45, y), "Simplified prompt", font=body_font, fill="black")
    y += 30
    _draw_wrapped(draw, (65, y), args.simple_prompt_text.strip(), small_font, 160, "#111111")
    pages.append(title)

    for row in rows:
        page_path = Path(row["side_by_side_path"])
        if page_path.exists():
            pages.append(Image.open(page_path).convert("RGB"))
    pdf_path = output_root / "visuals" / "stage1_prompt_ablation_detail_vs_simple.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _summary_from_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    deltas = [int(row["count_delta_simple_minus_detailed"]) for row in rows]
    return {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "output_root": str(args.output_root),
        "source_cases_csv": str(args.cases.resolve()),
        "sample_seed": args.sample_seed,
        "cases": len(rows),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_concurrent": args.max_concurrent,
        "detailed_prompt_path": str(args.detailed_prompt.resolve()),
        "simple_prompt_source": "embedded_default" if args.simple_prompt is None else str(args.simple_prompt.resolve()),
        "api_error_cases": sum(
            1 for row in rows if row["detailed_status"] != "ok" or row["simple_status"] != "ok"
        ),
        "detailed_parse_failures": sum(
            1 for row in rows if row["detailed_parse_status"] in {"empty_response", "no_parseable_bbox_payload", "parse_failed", "api_or_runtime_error"}
        ),
        "simple_parse_failures": sum(
            1 for row in rows if row["simple_parse_status"] in {"empty_response", "no_parseable_bbox_payload", "parse_failed", "api_or_runtime_error"}
        ),
        "detailed_total_boxes": sum(int(row["detailed_count"]) for row in rows),
        "simple_total_boxes": sum(int(row["simple_count"]) for row in rows),
        "simple_minus_detailed_total_boxes": sum(deltas),
        "simple_more_boxes_cases": sum(1 for value in deltas if value > 0),
        "simple_fewer_boxes_cases": sum(1 for value in deltas if value < 0),
        "same_count_cases": sum(1 for value in deltas if value == 0),
        "iou_match_threshold": args.iou_match_threshold,
        "detail_only_boxes_total": sum(int(row["detail_only_boxes"]) for row in rows),
        "simple_only_boxes_total": sum(int(row["simple_only_boxes"]) for row in rows),
    }


def _write_contact_sheet(output_root: Path, rows: list[dict[str, Any]]) -> None:
    thumb_size = (520, 340)
    cols = 2
    rows_per_sheet = (len(rows) + cols - 1) // cols
    page = Image.new("RGB", (cols * 1100, rows_per_sheet * 500 + 80), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(20)
    body_font = _font(14)
    draw.text((30, 25), "Stage 1 prompt ablation: detailed vs simplified overlays", font=title_font, fill="black")
    for idx, row in enumerate(rows):
        col = idx % cols
        sheet_row = idx // cols
        x = 30 + col * 1100
        y = 70 + sheet_row * 500
        label = (
            f"{row['case_index']:>3}: detail {row['detailed_count']} / simple {row['simple_count']} "
            f"(delta {row['count_delta_simple_minus_detailed']:+})"
        )
        draw.text((x, y), label, font=body_font, fill="black")
        if row["detailed_overlay_path"] and Path(row["detailed_overlay_path"]).exists():
            detail = _thumb(Path(row["detailed_overlay_path"]), thumb_size)
        else:
            detail = _blank_panel(thumb_size, "missing detail")
        if row["simple_overlay_path"] and Path(row["simple_overlay_path"]).exists():
            simple = _thumb(Path(row["simple_overlay_path"]), thumb_size)
        else:
            simple = _blank_panel(thumb_size, "missing simple")
        page.paste(detail, (x, y + 28))
        page.paste(simple, (x + 540, y + 28))
    path = output_root / "visuals" / "stage1_prompt_ablation_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _sample_cases(_read_csv(args.cases), args.sample_size, args.sample_seed)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        case_slug = _case_slug(row)
        base = {
            "case_index": int(row["case_index"]),
            "case_display": row["case_display"],
            "case_slug": case_slug,
            "thumbnail_path": row["thumbnail_path"],
            "metadata_path": row["metadata_path"],
            "wsi_path": _load_wsi_path(row),
            "source_stage1_dir": row.get("stage1_dir", ""),
            "source_bboxes_json_path": row.get("bboxes_json_path", ""),
        }
        tasks.append({**base, "prompt_key": "detailed", "prompt_text": args.detailed_prompt_text})
        tasks.append({**base, "prompt_key": "simple", "prompt_text": args.simple_prompt_text})
    tasks.sort(key=lambda task: (task["case_index"], task["prompt_key"]))
    return tasks


def _rows_from_results(results: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_case: dict[int, dict[str, Any]] = {}
    for result in results:
        by_case.setdefault(result["case_index"], {})[result["prompt_key"]] = result
    rows: list[dict[str, Any]] = []
    for case_index in sorted(by_case):
        pair = by_case[case_index]
        detailed = pair.get("detailed", {})
        simple = pair.get("simple", {})
        if not detailed or not simple:
            continue
        match_counts = _match_counts(
            detailed.get("detections", []),
            simple.get("detections", []),
            args.iou_match_threshold,
        )
        row: dict[str, Any] = {
            "case_index": case_index,
            "case_display": detailed.get("case_display", simple.get("case_display", "")),
            "case_slug": detailed.get("case_slug", simple.get("case_slug", "")),
            "wsi_path": detailed.get("wsi_path", simple.get("wsi_path", "")),
            "thumbnail_path": detailed.get("thumbnail_path", simple.get("thumbnail_path", "")),
            "metadata_path": detailed.get("metadata_path", simple.get("metadata_path", "")),
            "source_stage1_dir": detailed.get("source_stage1_dir", simple.get("source_stage1_dir", "")),
            "detailed_status": detailed.get("status", ""),
            "simple_status": simple.get("status", ""),
            "detailed_parse_status": detailed.get("parse_status", ""),
            "simple_parse_status": simple.get("parse_status", ""),
            "detailed_count": len(detailed.get("detections", [])),
            "simple_count": len(simple.get("detections", [])),
            "count_delta_simple_minus_detailed": len(simple.get("detections", [])) - len(detailed.get("detections", [])),
            "detailed_raw_response_path": detailed.get("raw_response_path", ""),
            "simple_raw_response_path": simple.get("raw_response_path", ""),
            "detailed_parsed_json_path": detailed.get("parsed_json_path", ""),
            "simple_parsed_json_path": simple.get("parsed_json_path", ""),
            "detailed_overlay_path": detailed.get("overlay_path", ""),
            "simple_overlay_path": simple.get("overlay_path", ""),
            "detailed_error": detailed.get("error", ""),
            "simple_error": simple.get("error", ""),
            **match_counts,
        }
        row["side_by_side_path"] = _write_side_by_side_png(args.output_root, row)
        rows.append(row)
    return rows


def _write_reproduction(args: argparse.Namespace, rows: list[dict[str, Any]], command: list[str]) -> None:
    selected_lines = []
    for row in rows:
        selected_lines.append(
            "- "
            + json.dumps(
                {
                    "case_index": row["case_index"],
                    "case_display": row["case_display"],
                    "wsi_path": row["wsi_path"],
                    "thumbnail_path": row["thumbnail_path"],
                    "metadata_path": row["metadata_path"],
                },
                sort_keys=True,
            )
        )
    text = f"""\
Stage 1 prompt-detail ablation
==============================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Output root: {args.output_root}

Objective:
Compare Stage 1 whole-slide thumbnail detector outputs when using the current
detailed high-recall prompt versus a simplified prompt with less task detail.

Command:
{" ".join(shlex.quote(part) for part in command)}

Source case list:
{args.cases.resolve()}

Sampling:
- Eligible rows: status == ok with existing thumbnail_path and metadata_path.
- Sample size: {args.sample_size}
- Seed: {args.sample_seed}
- Selected cases:
{chr(10).join(selected_lines)}

Model/API settings:
- Backend: OpenRouter-compatible chat completions
- API base: {args.api_base}
- Model: {args.model}
- Temperature: {args.temperature}
- Max tokens: {args.max_tokens}
- Max concurrent requests: {args.max_concurrent}
- Reasoning effort: {args.reasoning_effort or "not set"}
- Thumbnail input: one saved rot0 thumbnail per case from the source Stage 1 run.

Detailed prompt file:
{args.detailed_prompt.resolve()}

Detailed prompt text:
{args.detailed_prompt_text.strip()}

Simplified prompt text:
{args.simple_prompt_text.strip()}

Outputs:
- Task manifest: {args.output_root / 'tasks' / 'prompt_ablation_tasks.jsonl'}
- Completed calls: {args.output_root / 'tasks' / 'prompt_ablation_completed.jsonl'}
- Raw responses: {args.output_root / 'raw_responses'}
- Parsed JSON: {args.output_root / 'parsed'}
- Per-prompt overlays: {args.output_root / 'overlays'}
- Case pages: {args.output_root / 'visuals' / 'case_pages'}
- Review PDF: {args.output_root / 'visuals' / 'stage1_prompt_ablation_detail_vs_simple.pdf'}
- Contact sheet: {args.output_root / 'visuals' / 'stage1_prompt_ablation_contact_sheet.png'}
- Case CSV: {args.output_root / 'summary' / 'prompt_ablation_cases.csv'}
- Summary JSON: {args.output_root / 'summary' / 'prompt_ablation_summary.json'}

Notes:
- Boxes are parsed as normalized 0-1000 yxyx coordinates and rendered on the
  saved thumbnails using the same parser used by Stage 1 review helpers.
- `detail_only_boxes` and `simple_only_boxes` are greedy IoU unmatched counts
  at threshold {args.iou_match_threshold}; they are review cues, not proof of
  true missed tissue or false positives.
- This is a 20-case prompt-sensitivity probe and should not be treated as a
  calibrated detector benchmark.
"""
    (args.output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    args.cases = args.cases.resolve()
    args.detailed_prompt = args.detailed_prompt.resolve()
    if args.simple_prompt is not None:
        args.simple_prompt = args.simple_prompt.resolve()
    if not args.cases.exists():
        raise SystemExit(f"Cases CSV does not exist: {args.cases}")
    if not args.detailed_prompt.exists():
        raise SystemExit(f"Detailed prompt does not exist: {args.detailed_prompt}")
    if args.simple_prompt is not None and not args.simple_prompt.exists():
        raise SystemExit(f"Simple prompt does not exist: {args.simple_prompt}")
    args.detailed_prompt_text = args.detailed_prompt.read_text().strip()
    args.simple_prompt_text = (
        args.simple_prompt.read_text().strip() if args.simple_prompt is not None else DEFAULT_SIMPLE_PROMPT.strip()
    )
    if not args.api_key:
        args.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not args.api_key and not args.summarize_only:
        raise SystemExit("No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or --api-key.")

    command = ["python", "scripts/stage1_prompt_detail_ablation.py"] + sys.argv[1:]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "prompts").mkdir(parents=True, exist_ok=True)
    (args.output_root / "prompts" / "detailed_prompt.txt").write_text(args.detailed_prompt_text + "\n")
    (args.output_root / "prompts" / "simple_prompt.txt").write_text(args.simple_prompt_text + "\n")

    tasks = _build_tasks(args)
    tasks_path = args.output_root / "tasks" / "prompt_ablation_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    completed_path = args.output_root / "tasks" / "prompt_ablation_completed.jsonl"
    if args.summarize_only:
        completed = [
            json.loads(line)
            for line in completed_path.read_text().splitlines()
            if line.strip()
        ]
    elif args.max_concurrent <= 1:
        completed = [_run_prompt(task, args) for task in tasks]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(_run_prompt, task, args) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
        completed.sort(key=lambda item: (item["case_index"], item["prompt_key"]))
    _write_jsonl(completed_path, completed)

    rows = _rows_from_results(completed, args)
    cases_path = args.output_root / "summary" / "prompt_ablation_cases.csv"
    _write_csv(cases_path, rows, list(rows[0].keys()) if rows else [])
    summary = _summary_from_rows(rows, args)
    _write_json(args.output_root / "summary" / "prompt_ablation_summary.json", summary)
    _write_contact_sheet(args.output_root, rows)
    _write_pdf(args.output_root, args, rows)
    _write_reproduction(args, rows, command)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--detailed-prompt", type=Path, default=DEFAULT_DETAILED_PROMPT)
    parser.add_argument("--simple-prompt", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--iou-match-threshold", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
