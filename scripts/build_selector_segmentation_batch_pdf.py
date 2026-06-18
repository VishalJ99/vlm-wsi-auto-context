#!/usr/bin/env python3
"""Render a selector-seeded foreground pipeline PDF packet."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PAGE_W = 2200
PAGE_H = 1550
MARGIN = 48
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
DEFAULT_REVIEWER_BATCH_NAME = "per250_scale500_5perstain_openrouter_gemini3flash_high_calibration_review_v1"
FALLBACK_REVIEWER_BATCH_TEMPLATE = "per250_scale500_5perstain_qwen16_icl0_array_calibration_review_v1_task{task:02d}"
DEFAULT_5PERSTAIN_RUN_ID_TEMPLATE = "per250_scale500_5perstain_verifier_qwen16_icl0_array_v1_task{task:02d}"
DEFAULT_5PERSTAIN_REPORT_TITLE = "PER-250 5-Per-Stain Selector-Seeded Foreground Pipeline"
DEFAULT_ALL500_REVIEWER_BATCH_NAME = "per250_scale500_all500_openrouter_gemini3flash_high_calibration_review_v1"
DEFAULT_ALL500_RUN_ID_TEMPLATE = "per250_scale500_all500_verifier_qwen16_icl0_array_v1_task{task:03d}"
DEFAULT_ALL500_REPORT_TITLE = "PER-250 All-Scale500 Selector-Seeded Foreground Pipeline"
DEFAULT_ALL500_VERIFIER_JSONL = (
    Path("runs/detector_pipeline_scale500_v1")
    / "analysis"
    / "artifact_redundancy_probe_all500_prohigh_flashlow_v1"
    / "summary"
    / "results.jsonl"
)
DISPLAY_QC_THRESHOLD = 0.85


RESULT_CACHE: dict[Path, list[dict[str, Any]]] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(38, True)
FONT_H1 = font(30, True)
FONT_H2 = font(24, True)
FONT_BODY = font(19)
FONT_SMALL = font(16)
FONT_TINY = font(13)
FONT_MONO = font(15)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return parsed if isinstance(parsed, list) else [parsed]


def case_to_dash(case_id: str) -> str:
    if case_id.startswith("anon_"):
        return "anon_" + case_id[len("anon_") :].replace("_", "-")
    return case_id.replace("_", "-")


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, RESAMPLE_LANCZOS)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str | tuple[int, int, int] = "black",
    font_obj: ImageFont.ImageFont = FONT_BODY,
    max_chars: int = 90,
    line_gap: int = 4,
    bottom: int | None = None,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines():
        lines = wrap(paragraph, width=max_chars, replace_whitespace=False) or [""]
        for line in lines:
            if bottom is not None and y > bottom:
                return y
            draw.text((x, y), line, fill=fill, font=font_obj)
            bbox = draw.textbbox((x, y), line or " ", font=font_obj)
            y += bbox[3] - bbox[1] + line_gap
    return y


def draw_panel_title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str) -> None:
    draw.text((x, y), title, fill="black", font=FONT_H2)


def paste_image_panel(
    page: Image.Image,
    title: str,
    img: Image.Image | None,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    caption: str = "",
) -> None:
    draw = ImageDraw.Draw(page)
    draw_panel_title(draw, x, y, title)
    image_y = y + 34
    caption_h = 56 if caption else 0
    image_h = h - 34 - caption_h
    draw.rectangle([x, image_y, x + w, image_y + image_h], outline=(210, 210, 210), width=2)
    if img is None:
        draw.text((x + 18, image_y + 18), "missing", fill=(170, 0, 0), font=FONT_H2)
    else:
        fitted = fit_image(img, w - 8, image_h - 8)
        ox = x + (w - fitted.width) // 2
        oy = image_y + (image_h - fitted.height) // 2
        page.paste(fitted, (ox, oy))
    if caption:
        draw_wrapped(
            draw,
            (x, image_y + image_h + 10),
            caption,
            fill=(35, 35, 35),
            font_obj=FONT_SMALL,
            max_chars=max(28, w // 10),
        )


def image_from_path(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def render_stage1_overlay(run_dir: Path) -> Image.Image | None:
    thumb_path = run_dir / "stage1" / "thumbnail.png"
    bbox_path = run_dir / "stage1" / "bboxes.json"
    if not thumb_path.exists() or not bbox_path.exists():
        return image_from_path(thumb_path)

    img = Image.open(thumb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    data = read_json(bbox_path)
    regions = data.get("detected_regions", [])
    for i, region in enumerate(regions, start=1):
        bbox = region.get("bbox_thumbnail")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = str(region.get("source_detection_id") or i)
        color = (10, 150, 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, img.width // 450))
        tx, ty = max(0, x1 + 4), max(0, y1 + 4)
        tb = draw.textbbox((tx, ty), label, font=FONT_H2)
        draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 5, tb[3] + 3], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=color, font=FONT_H2)
    return img


def make_contact_sheet(items: list[tuple[Path, str]], box_w: int, box_h: int) -> Image.Image | None:
    existing = [(p, label) for p, label in items if p.exists()]
    if not existing:
        return None
    n = len(existing)
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    gap = 10
    label_h = 22
    cell_w = (box_w - gap * (cols - 1)) // cols
    cell_h = (box_h - gap * (rows - 1)) // rows
    sheet = Image.new("RGB", (box_w, box_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, label) in enumerate(existing):
        row = idx // cols
        col = idx % cols
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(220, 220, 220), width=1)
        img = fit_image(Image.open(path), cell_w - 8, cell_h - label_h - 8)
        sheet.paste(img, (x + (cell_w - img.width) // 2, y + 4))
        draw_wrapped(
            draw,
            (x + 5, y + cell_h - label_h),
            label,
            fill=(30, 30, 30),
            font_obj=FONT_TINY,
            max_chars=max(12, cell_w // 9),
            line_gap=1,
        )
    return sheet


def metric_passes(value: Any, threshold: float = DISPLAY_QC_THRESHOLD) -> bool:
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def display_overall_pass(qc: dict[str, Any], threshold: float = DISPLAY_QC_THRESHOLD) -> bool:
    return metric_passes(qc.get("precision"), threshold) and metric_passes(qc.get("recall"), threshold)


def make_review_contact_sheet(
    items: list[dict[str, Any]],
    box_w: int,
    box_h: int,
    *,
    threshold: float = DISPLAY_QC_THRESHOLD,
) -> Image.Image | None:
    existing = [item for item in items if item["path"].exists()]
    if not existing:
        return None
    n = len(existing)
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    gap = 10
    label_h = 22
    cell_w = (box_w - gap * (cols - 1)) // cols
    cell_h = (box_h - gap * (rows - 1)) // rows
    sheet = Image.new("RGB", (box_w, box_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(existing):
        row = idx // cols
        col = idx % cols
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        qc = item.get("qc") or {}
        is_pass = display_overall_pass(qc, threshold)
        color = (35, 170, 90) if is_pass else (220, 50, 50)
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=color, width=5)

        header_h = 44
        img = fit_image(Image.open(item["path"]), cell_w - 16, cell_h - label_h - header_h - 12)
        ox = x + (cell_w - img.width) // 2
        oy = y + header_h + (cell_h - label_h - header_h - img.height) // 2
        sheet.paste(img, (ox, oy))

        label = str(item["index"])
        badge_bbox = draw.textbbox((0, 0), label, font=FONT_H2)
        badge_w = badge_bbox[2] - badge_bbox[0] + 18
        badge_h = badge_bbox[3] - badge_bbox[1] + 12
        bx = x + 12
        by = y + 10
        draw.rectangle([bx, by, bx + badge_w, by + badge_h], fill="white", outline=color, width=4)
        draw.text((bx + 9, by + 5), label, fill=color, font=FONT_H2)

        draw_wrapped(
            draw,
            (x + 7, y + cell_h - label_h),
            f"{item['index']}: {'pass' if is_pass else 'fail'}",
            fill=color,
            font_obj=FONT_TINY,
            max_chars=max(12, cell_w // 9),
            line_gap=1,
        )
    return sheet


def load_jsonl_by_case(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if case_id:
                rows[str(case_id)] = row
    return rows


def selector_manifest_from_summary(run_root: Path) -> Path | None:
    summary_path = run_root / "manifests" / "manifest_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    selector_manifest = summary.get("selector_manifest") if isinstance(summary, dict) else None
    if not selector_manifest:
        return None
    path = Path(selector_manifest)
    if path.exists():
        return path
    repo_relative = Path.cwd() / path
    return repo_relative if repo_relative.exists() else None


def load_selector_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return {}
    return {str(row["case_id"]): row for row in records if isinstance(row, dict) and row.get("case_id")}


def populate_row_paths(run_root: Path, row: dict[str, Any], run_id_template: str) -> dict[str, Any]:
    task = int(row["task"])
    run_id = run_id_template.format(task=task)
    task_case_path = run_root / "_array_task_cases" / f"{run_id}.txt"
    task_case = task_case_path.read_text(encoding="utf-8").strip() if task_case_path.exists() else row["case_id"]
    if task_case != row["case_id"]:
        raise SystemExit(f"Task {task:03d} case mismatch: manifest={row['case_id']} task_file={task_case}")
    row["run_id"] = run_id
    row["case_dash"] = case_to_dash(row["case_id"])
    row["run_dir"] = run_root / row["case_dash"] / run_id
    row["selected_box_ids"] = parse_list(row.get("verifier_selected_box_ids"))
    row["baseline_selected_box_ids"] = parse_list(row.get("baseline_selected_box_ids"))
    row["direct_selected_box_ids"] = parse_list(row.get("direct_selected_box_ids"))
    return row


def load_csv_manifest(run_root: Path, manifest: Path, run_id_template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for task, row in enumerate(csv.DictReader(f)):
            row["task"] = int(row.get("task") or task)
            rows.append(populate_row_paths(run_root, row, run_id_template))
    return rows


def load_case_list_manifest(
    run_root: Path,
    case_list: Path,
    run_id_template: str,
    selector_manifest: Path | None,
    verifier_jsonl: Path | None,
) -> list[dict[str, Any]]:
    selector_rows = load_selector_records(selector_manifest)
    verifier_rows = load_jsonl_by_case(verifier_jsonl)
    case_ids = [line.strip() for line in case_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for task, case_id in enumerate(case_ids):
        selector = selector_rows.get(case_id, {})
        verifier = verifier_rows.get(case_id, {})
        row: dict[str, Any] = {
            "task": task,
            "case_id": case_id,
            "stain": selector.get("stain") or selector.get("stain_label") or "unknown",
            "selection_index_within_stain": selector.get("selection_index_within_stain") or "",
            "wsi_path": selector.get("worklist_wsi_path") or selector.get("wsi_path") or "",
            "source_wsi_path": selector.get("source_wsi_path") or selector.get("wsi_path") or "",
            "verifier_selected_box_ids": verifier.get("verifier_selected_box_ids", []),
            "verifier_confidence": verifier.get("verifier_confidence", ""),
            "verifier_needs_revision": verifier.get("verifier_needs_revision", ""),
            "baseline_selected_box_ids": verifier.get("baseline_selected_box_ids", []),
            "direct_selected_box_ids": verifier.get("direct_selected_box_ids", []),
        }
        rows.append(populate_row_paths(run_root, row, run_id_template))
    return rows


def load_manifest(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    if args.manifest_csv is not None:
        return load_csv_manifest(run_root, args.manifest_csv.resolve(), args.run_id_template)
    default_csv = run_root / "manifests" / "selected_cases_5_per_stain.csv"
    if default_csv.exists():
        return load_csv_manifest(run_root, default_csv, args.run_id_template)

    case_list = args.case_list or (run_root / "manifests" / "selected_cases_all500.txt")
    if not case_list.exists():
        raise SystemExit(f"Missing case manifest: {case_list}")
    selector_manifest = args.selector_manifest or selector_manifest_from_summary(run_root)
    verifier_jsonl = args.verifier_jsonl or DEFAULT_ALL500_VERIFIER_JSONL
    return load_case_list_manifest(
        run_root,
        case_list.resolve(),
        args.run_id_template,
        selector_manifest.resolve() if selector_manifest else None,
        verifier_jsonl.resolve() if verifier_jsonl else None,
    )


def stage6_items(run_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for path in sorted(run_dir.glob("bboxes/*/stage6/class_overlay.png")):
        items.append((path, path.parts[-3]))
    return items


def reviewer_input_items(run_root: Path, row: dict[str, Any]) -> list[tuple[Path, str]]:
    review_run = f"{row['run_id']}_stage7_l0_review_1024"
    review_root = run_root / "reviewer_inputs" / row["case_id"] / review_run
    items = []
    for path in sorted(review_root.glob("bboxes/*/stage3/overlay.png")):
        items.append((path, path.parts[-3]))
    return items


def load_result_rows(batch: Path) -> list[dict[str, Any]]:
    if batch in RESULT_CACHE:
        return RESULT_CACHE[batch]
    result_path = batch / "results.jsonl"
    rows: list[dict[str, Any]] = []
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(normalize_result_row(json.loads(line)))
    else:
        csv_path = batch / "results.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(normalize_result_row(row) for row in csv.DictReader(f))
    latest_by_item: dict[tuple[str, str, str], dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("case_id") or ""), str(row.get("run_id") or ""), str(row.get("bbox_id") or ""))
        if all(key):
            latest_by_item[key] = row
        else:
            deduped.append(row)
    deduped.extend(latest_by_item.values())
    RESULT_CACHE[batch] = deduped
    return deduped


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "pass"}:
        return True
    if lowered in {"false", "0", "no", "fail"}:
        return False
    return None


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    qc = normalized.get("qc") if isinstance(normalized.get("qc"), dict) else {}
    if not qc and any(key in normalized for key in ("qc_precision", "qc_recall", "qc_overall_pass")):
        qc = {
            "precision": normalized.get("qc_precision"),
            "recall": normalized.get("qc_recall"),
            "precision_pass": parse_bool(normalized.get("qc_precision_pass")),
            "recall_pass": parse_bool(normalized.get("qc_recall_pass")),
            "overall_pass": parse_bool(normalized.get("qc_overall_pass")),
        }
    normalized["qc"] = qc
    if normalized.get("json_parsed") in ("True", "true", True) and not normalized.get("status"):
        normalized["status"] = "success"
    return normalized


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "total_scheduled": len(rows),
        "total_completed": len(rows),
        "success_count": 0,
        "failed_count": 0,
        "qc_overall_pass_count": 0,
        "qc_precision_pass_count": 0,
        "qc_recall_pass_count": 0,
        "qc_unavailable_count": 0,
        "openrouter_retry_count": 0,
    }
    for row in rows:
        status = row.get("status")
        if status == "success":
            summary["success_count"] += 1
        else:
            summary["failed_count"] += 1
            summary["openrouter_retry_count"] += 1
            reason = row.get("failure_reason") or row.get("error") or status or "unknown"
            reasons[str(reason)] += 1
        qc = row.get("qc") or {}
        if qc.get("overall_pass") is True:
            summary["qc_overall_pass_count"] += 1
        if qc.get("precision_pass") is True:
            summary["qc_precision_pass_count"] += 1
        if qc.get("recall_pass") is True:
            summary["qc_recall_pass_count"] += 1
        if qc.get("overall_pass") is None:
            summary["qc_unavailable_count"] += 1
    summary["failure_reason_counts"] = dict(reasons)
    return summary


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def pass_flag(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "n/a"


def load_batch_config(batch: Path) -> dict[str, Any]:
    manifest_path = batch / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    config = manifest.get("config") if isinstance(manifest, dict) else {}
    return config if isinstance(config, dict) else {}


def load_reviewer_summary(run_root: Path, row: dict[str, Any], reviewer_batch_name: str) -> dict[str, Any]:
    preferred_batch = run_root / "reviewer" / reviewer_batch_name
    if preferred_batch.exists():
        expected_run = f"{row['run_id']}_stage7_l0_review_1024"
        rows = [
            result
            for result in load_result_rows(preferred_batch)
            if result.get("case_id") == row["case_id"] and result.get("run_id") == expected_run
        ]
        summary = summarize_result_rows(rows)
        config = load_batch_config(preferred_batch)
        summary.update(
            {
                "batch_name": reviewer_batch_name,
                "batch_path": str(preferred_batch),
                "review_rows": rows,
                "reviewer_backend": config.get("backend"),
                "reviewer_model": config.get("model"),
                "reviewer_reasoning_effort": config.get("reasoning_effort"),
                "reviewer_source": "preferred_openrouter_batch",
            }
        )
        return summary

    task = int(row["task"])
    batch_name = FALLBACK_REVIEWER_BATCH_TEMPLATE.format(task=task)
    batch = run_root / "reviewer" / batch_name
    summary_path = batch / "summary.json"
    if not summary_path.exists():
        return {"batch_name": reviewer_batch_name, "reviewer_source": "missing"}
    summary = read_json(summary_path)
    review_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    result_path = batch / "results.jsonl"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                review_rows.append(result)
                reason = result.get("failure_reason") or result.get("error") or result.get("status") or "unknown"
                reason_counts[str(reason)] += 1
    summary["failure_reason_counts"] = dict(reason_counts)
    config = load_batch_config(batch)
    summary.update(
            {
                "batch_name": batch_name,
                "batch_path": str(batch),
                "review_rows": review_rows,
                "reviewer_backend": config.get("backend"),
                "reviewer_model": config.get("model"),
                "reviewer_reasoning_effort": config.get("reasoning_effort"),
            "reviewer_source": "fallback_qwen_batch",
        }
    )
    return summary


def enumerated_reviewer_items(run_root: Path, row: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_bbox = {
        str(result.get("bbox_id")): result
        for result in summary.get("review_rows") or []
        if result.get("bbox_id") is not None
    }
    items: list[dict[str, Any]] = []
    for index, (path, bbox_id) in enumerate(reviewer_input_items(run_root, row), start=1):
        result = rows_by_bbox.get(str(bbox_id), {})
        items.append(
            {
                "index": index,
                "path": path,
                "bbox_id": bbox_id,
                "result": result,
                "qc": result.get("qc") if isinstance(result, dict) else {},
            }
        )
    seen = {str(item["bbox_id"]) for item in items}
    for result in sorted(summary.get("review_rows") or [], key=lambda r: str(r.get("bbox_id", ""))):
        bbox_id = str(result.get("bbox_id", "bbox"))
        if bbox_id in seen:
            continue
        items.append(
            {
                "index": len(items) + 1,
                "path": Path("missing"),
                "bbox_id": bbox_id,
                "result": result,
                "qc": result.get("qc") or {},
            }
        )
    return items


def aggregate_counts(
    run_root: Path,
    rows: list[dict[str, Any]],
    reviewer_batch_name: str,
    completion_summary: Path | None,
) -> dict[str, Any]:
    artifact_summary = completion_summary or run_root / "manifests" / "alex_array_completion_summary.json"
    completion = read_json(artifact_summary) if artifact_summary.exists() else {}
    reviewer = Counter()
    reasons: Counter[str] = Counter()
    reviewer_sources: Counter[str] = Counter()
    reviewer_batches: Counter[str] = Counter()
    for row in rows:
        summary = load_reviewer_summary(run_root, row, reviewer_batch_name)
        for key in (
            "total_scheduled",
            "total_completed",
            "success_count",
            "failed_count",
            "qc_overall_pass_count",
            "qc_precision_pass_count",
            "qc_recall_pass_count",
            "qc_unavailable_count",
            "openrouter_retry_count",
        ):
            reviewer[key] += int(summary.get(key) or 0)
        reasons.update(summary.get("failure_reason_counts", {}))
        reviewer_sources[str(summary.get("reviewer_source") or "unknown")] += 1
        reviewer_batches[str(summary.get("batch_name") or "missing")] += 1
    stains = Counter(row["stain"] for row in rows)
    return {
        "case_count": len(rows),
        "stains": dict(stains),
        "reviewer": dict(reviewer),
        "reviewer_failure_reasons": dict(reasons),
        "reviewer_sources": dict(reviewer_sources),
        "reviewer_batches": dict(reviewer_batches),
        "completion_totals": completion.get("artifact_totals", {}),
        "complete_marker_count": len(list((run_root / "_array_task_cases").glob("*.complete.txt"))),
        "stage7_mask_count": len(list(run_root.glob("*/**/stage7/mask_overlay.png"))),
    }


def draw_summary_page(
    run_root: Path,
    rows: list[dict[str, Any]],
    counts: dict[str, Any],
    generated_at: str,
    args: argparse.Namespace,
) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN
    draw.text((MARGIN, y), args.report_title, fill="black", font=FONT_TITLE)
    y += 58
    y = draw_wrapped(draw, (MARGIN, y), f"Run root: {run_root}", font_obj=FONT_BODY, max_chars=150)
    y = draw_wrapped(draw, (MARGIN, y + 4), f"Generated: {generated_at}", font_obj=FONT_BODY, max_chars=150)
    y += 28

    reviewer = counts["reviewer"]
    stain_text = ", ".join(f"{k}={v}" for k, v in sorted(counts["stains"].items()))
    totals = counts.get("completion_totals") or {}
    bullets = [
        f"Cases: {counts['case_count']} ({stain_text}).",
        f"Alex array: job {args.array_job_id}, {counts.get('complete_marker_count', 0)}/{args.array_total} complete markers, one WSI per task, array concurrency {args.array_concurrency}.",
        "Stage 6: Qwen/Qwen3-VL-8B-Instruct-FP8 via vLLM, ICL k=0, max workers=16, query batch size=1.",
        "Stage 7: fill holes enabled; binary close skipped.",
        "Reviewer inputs: Stage 7 high-resolution exports at max_dim=1024.",
        f"Artifacts: case statuses={totals.get('case_status', 'n/a')}, selected bbox runs={totals.get('bbox_dirs', 'n/a')}, Stage 6={totals.get('stage6', 'n/a')}, Stage 7 bbox={totals.get('stage7', 'n/a')}, WSI Stage 7={totals.get('wsi_stage7', counts.get('stage7_mask_count', 'n/a'))}.",
        f"Reviewer: scheduled={reviewer.get('total_scheduled', 0)}, success={reviewer.get('success_count', 0)}, failed={reviewer.get('failed_count', 0)}, QC unavailable={reviewer.get('qc_unavailable_count', 0)}.",
        f"Reviewer QC: precision pass={reviewer.get('qc_precision_pass_count', 0)}, recall pass={reviewer.get('qc_recall_pass_count', 0)}, overall pass={reviewer.get('qc_overall_pass_count', 0)} at strict >0.9 thresholds.",
    ]
    if counts["reviewer_failure_reasons"]:
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(counts["reviewer_failure_reasons"].items()))
        bullets.append(f"Reviewer failure reasons: {reason_text}.")
    source_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.get("reviewer_sources", {}).items()))
    if source_text:
        bullets.append(f"Reviewer source batches: {source_text}.")
    if reviewer.get("success_count", 0) > 0:
        bullets.append("Interpretation: segmentation outputs are available and calibration self-review QC is available from the selected reviewer batch.")
    else:
        bullets.append("Interpretation: segmentation outputs are available for all selected crops; calibration self-review still needs a Gemini/OpenRouter retry.")

    draw.text((MARGIN, y), "Run Summary", fill="black", font=FONT_H1)
    y += 45
    for bullet in bullets:
        y = draw_wrapped(draw, (MARGIN + 20, y), f"- {bullet}", font_obj=FONT_BODY, max_chars=138)
        y += 6

    y += 22
    draw.text((MARGIN, y), "Case Order", fill="black", font=FONT_H1)
    y += 42
    cols = 2
    col_w = (PAGE_W - 2 * MARGIN - 30) // cols
    start_y = y
    visible_rows = rows[:26]
    for idx, row in enumerate(visible_rows):
        col = idx // 13
        row_idx = idx % 13
        x = MARGIN + col * (col_w + 30)
        yy = start_y + row_idx * 31
        text = f"task{row['task']:02d} | {row['stain']} | {row['case_id']} | selected={row['selected_box_ids']}"
        draw_wrapped(draw, (x, yy), text, font_obj=FONT_SMALL, max_chars=82, line_gap=1)
    if len(rows) > len(visible_rows):
        yy = start_y + 13 * 31 + 12
        draw_wrapped(
            draw,
            (MARGIN, yy),
            f"... {len(rows) - len(visible_rows)} additional cases omitted from this summary list; case pages follow in manifest order.",
            font_obj=FONT_SMALL,
            max_chars=150,
        )
    return page


def draw_case_page(run_root: Path, row: dict[str, Any], reviewer_batch_name: str) -> Image.Image:
    run_dir = Path(row["run_dir"])
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)

    header = f"{row['case_id']} | {row['stain']} | task{row['task']:02d}"
    draw.text((MARGIN, MARGIN), header, fill="black", font=FONT_TITLE)
    sub = (
        f"selected={row['selected_box_ids']} | baseline={row['baseline_selected_box_ids']} | "
        f"direct={row['direct_selected_box_ids']} | confidence={row.get('verifier_confidence')} | "
        f"revise={row.get('verifier_needs_revision')}"
    )
    draw_wrapped(draw, (MARGIN, MARGIN + 50), sub, font_obj=FONT_BODY, max_chars=150)

    top_y = 130
    top_h = 525
    panel_w = (PAGE_W - 2 * MARGIN - 32) // 2
    paste_image_panel(
        page,
        "1. Verifier-Selected Detector Bboxes",
        render_stage1_overlay(run_dir),
        MARGIN,
        top_y,
        panel_w,
        top_h,
        caption="Only verifier-selected scale-500 boxes are seeded as Stage 1 detections; labels show original detector box IDs.",
    )
    paste_image_panel(
        page,
        "2. WSI-Level Stage 7 Foreground",
        image_from_path(run_dir / "stage7" / "mask_overlay.png"),
        MARGIN + panel_w + 32,
        top_y,
        panel_w,
        top_h,
        caption="Orange mask overlay is the WSI-level Stage 7 result assembled from selected bbox runs.",
    )

    bottom_y = 700
    contact_w = 610
    text_w = PAGE_W - 2 * MARGIN - (contact_w * 2) - 56
    contact_h = 720
    stage6_sheet = make_contact_sheet(stage6_items(run_dir), contact_w - 8, contact_h - 98)
    paste_image_panel(
        page,
        "3. Stage 6 Crop Classifications",
        stage6_sheet,
        MARGIN,
        bottom_y,
        contact_w,
        contact_h,
        caption="Per-selected-bbox Qwen patch-classification overlays before Stage 7 postprocessing.",
    )
    summary = load_reviewer_summary(run_root, row, reviewer_batch_name)
    review_items = enumerated_reviewer_items(run_root, row, summary)
    review_sheet = make_review_contact_sheet(review_items, contact_w - 8, contact_h - 98)
    paste_image_panel(
        page,
        "4. Reviewer Input Overlays",
        review_sheet,
        MARGIN + contact_w + 28,
        bottom_y,
        contact_w,
        contact_h,
        caption="High-resolution Stage 7 review inputs exported at max_dim=1024 for the calibration reviewer.",
    )

    text_x = MARGIN + contact_w * 2 + 56
    draw.text((text_x, bottom_y), "5. Reviewer Output", fill="black", font=FONT_H2)
    reason_counts = summary.get("failure_reason_counts", {})
    y = bottom_y + 42
    if reason_counts:
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        y = draw_wrapped(draw, (text_x, y), f"Failure reasons: {reason_text}", font_obj=FONT_BODY, max_chars=max(40, text_w // 11))
        y += 12
    if review_items:
        y = draw_wrapped(
            draw,
            (text_x, y),
            f"Per-bbox precision / recall (>={DISPLAY_QC_THRESHOLD:.2f} pass):",
            font_obj=FONT_BODY,
            max_chars=max(40, text_w // 11),
        )
        y += 4
        for item in review_items:
            qc = item.get("qc") or {}
            precision_pass = metric_passes(qc.get("precision"))
            recall_pass = metric_passes(qc.get("recall"))
            line = (
                f"{item['index']}: "
                f"precision={format_metric(qc.get('precision'))} ({pass_flag(precision_pass)}), "
                f"recall={format_metric(qc.get('recall'))} ({pass_flag(recall_pass)}), "
                f"overall={pass_flag(precision_pass and recall_pass)}"
            )
            y = draw_wrapped(
                draw,
                (text_x, y),
                line,
                font_obj=FONT_SMALL,
                max_chars=max(38, text_w // 10),
                bottom=PAGE_H - 58,
            )
            y += 4
    else:
        draw_wrapped(
            draw,
            (text_x, y),
            "Per-bbox precision / recall: missing",
            font_obj=FONT_BODY,
            max_chars=max(40, text_w // 11),
        )
    return page


def build_prompt_pages(run_root: Path, rows: list[dict[str, Any]], prompt_file: Path) -> list[Image.Image]:
    first_run = Path(rows[0]["run_dir"])
    stage1_meta = read_json(first_run / "stage1" / "metadata.json")
    stage6_meta_path = next(first_run.glob("bboxes/*/stage6/metadata.json"))
    stage6_meta = read_json(stage6_meta_path)
    prompts = [
        ("Stage 1 adapter detector prompt", str(first_run / "stage1" / "metadata.json"), stage1_meta.get("prompt", "")),
        ("Stage 6 patch-classifier rendered prompt", str(stage6_meta_path), stage6_meta.get("prompt_rendered_text", "")),
        ("Calibration reviewer prompt", str(prompt_file), read_text(prompt_file)),
    ]

    pages: list[Image.Image] = []
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    pages.append(page)
    y = MARGIN
    draw.text((MARGIN, y), "Prompt Provenance", fill="black", font=FONT_TITLE)
    y += 60

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        next_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        next_draw = ImageDraw.Draw(next_page)
        next_draw.text((MARGIN, MARGIN), "Prompt Provenance", fill="black", font=FONT_TITLE)
        pages.append(next_page)
        return next_page, next_draw, MARGIN + 60

    for label, source, text in prompts:
        if y > PAGE_H - 150:
            page, draw, y = new_page()
        draw.text((MARGIN, y), label, fill="black", font=FONT_H1)
        y += 38
        y = draw_wrapped(draw, (MARGIN, y), source, fill=(70, 70, 70), font_obj=FONT_SMALL, max_chars=170)
        y += 12
        for raw_line in text.splitlines():
            for line in wrap(raw_line, width=160, replace_whitespace=False) or [""]:
                if y > PAGE_H - 50:
                    page, draw, y = new_page()
                draw.text((MARGIN, y), line, fill=(20, 20, 20), font=FONT_MONO)
                bbox = draw.textbbox((MARGIN, y), line or " ", font=FONT_MONO)
                y += bbox[3] - bbox[1] + 5
        y += 30
    return pages


def write_reproduction(output_pdf: Path, run_root: Path, args: argparse.Namespace, counts: dict[str, Any], generated_at: str) -> None:
    repro_path = output_pdf.parent / "reproduction.txt"
    rel_script = Path("scripts/build_selector_segmentation_batch_pdf.py")
    command_parts = [
        "python",
        str(rel_script),
        "--run-root",
        str(run_root),
        "--output-pdf",
        str(output_pdf),
        "--prompt-file",
        str(args.prompt_file),
        "--reviewer-batch-name",
        str(args.reviewer_batch_name),
        "--run-id-template",
        str(args.run_id_template),
        "--report-title",
        str(args.report_title),
        "--array-job-id",
        str(args.array_job_id),
        "--array-total",
        str(args.array_total),
        "--array-concurrency",
        str(args.array_concurrency),
    ]
    for option, value in (
        ("--manifest-csv", args.manifest_csv),
        ("--case-list", args.case_list),
        ("--selector-manifest", args.selector_manifest),
        ("--verifier-jsonl", args.verifier_jsonl),
        ("--completion-summary", args.completion_summary),
    ):
        if value is not None:
            command_parts.extend([option, str(value)])
    command = shlex.join(command_parts)
    text = f"""PER-250 selector-seeded Stage 7 review PDF

Created: {generated_at}
Ticket: PER-250

Output PDF:
{output_pdf}

Run root:
{run_root}

Command:
{command}

Inputs summarized:
- selected case manifest: {args.manifest_csv or args.case_list or 'auto-detected under run-root/manifests'}
- selector manifest: {args.selector_manifest or 'auto-detected from manifest_summary.json when available'}
- verifier JSONL: {args.verifier_jsonl or DEFAULT_ALL500_VERIFIER_JSONL}
- per-task auto-context run directories
- per-task Stage 6 class overlays
- WSI-level Stage 7 mask overlays
- Stage 7 high-resolution reviewer inputs
- calibration reviewer result summaries
- prompts/calibration_reviewer.txt
- selected reviewer batch: {args.reviewer_batch_name}

Reviewer aggregate at generation time:
{json.dumps(counts.get("reviewer", {}), sort_keys=True)}

Reviewer failure reasons at generation time:
{json.dumps(counts.get("reviewer_failure_reasons", {}), sort_keys=True)}

Reviewer source batches at generation time:
{json.dumps(counts.get("reviewer_sources", {}), sort_keys=True)}
"""
    repro_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/auto_context_scale500_selector_5perstain_v1"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("runs/auto_context_scale500_selector_5perstain_v1/visuals/per250_5perstain_selector_seeded_stage7_self_review_packet.pdf"),
    )
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/calibration_reviewer.txt"))
    parser.add_argument("--manifest-csv", type=Path, default=None, help="CSV case manifest. Defaults to selected_cases_5_per_stain.csv when present.")
    parser.add_argument("--case-list", type=Path, default=None, help="Plain-text case list for all500-style runs.")
    parser.add_argument("--selector-manifest", type=Path, default=None, help="Selector input manifest with stain/source WSI metadata.")
    parser.add_argument("--verifier-jsonl", type=Path, default=None, help="Verifier results JSONL with selected bbox IDs.")
    parser.add_argument(
        "--run-id-template",
        default=DEFAULT_5PERSTAIN_RUN_ID_TEMPLATE,
        help="Python format string used to derive per-task run IDs from task index.",
    )
    parser.add_argument(
        "--report-title",
        default=DEFAULT_5PERSTAIN_REPORT_TITLE,
        help="Title displayed on the summary page.",
    )
    parser.add_argument("--array-job-id", default="3696263", help="SLURM array job ID shown in the summary page.")
    parser.add_argument("--array-total", type=int, default=25, help="Total task count shown in the summary page.")
    parser.add_argument("--array-concurrency", type=int, default=5, help="Array throttle shown in the summary page.")
    parser.add_argument("--completion-summary", type=Path, default=None, help="Optional artifact completion summary JSON.")
    parser.add_argument(
        "--reviewer-batch-name",
        default=DEFAULT_REVIEWER_BATCH_NAME,
        help=(
            "Preferred reviewer batch folder under <run-root>/reviewer. "
            "If absent, the script falls back to the original per-task Qwen/vLLM reviewer batches."
        ),
    )
    return parser.parse_args()


def apply_auto_defaults(args: argparse.Namespace) -> argparse.Namespace:
    run_root = args.run_root
    all500_case_list = run_root / "manifests" / "selected_cases_all500.txt"
    has_5perstain_csv = (run_root / "manifests" / "selected_cases_5_per_stain.csv").exists()
    looks_all500 = all500_case_list.exists() and not has_5perstain_csv
    if looks_all500:
        if args.case_list is None:
            args.case_list = all500_case_list
        if args.run_id_template == DEFAULT_5PERSTAIN_RUN_ID_TEMPLATE:
            args.run_id_template = DEFAULT_ALL500_RUN_ID_TEMPLATE
        if args.reviewer_batch_name == DEFAULT_REVIEWER_BATCH_NAME:
            args.reviewer_batch_name = DEFAULT_ALL500_REVIEWER_BATCH_NAME
        if args.report_title == DEFAULT_5PERSTAIN_REPORT_TITLE:
            args.report_title = DEFAULT_ALL500_REPORT_TITLE
        if args.array_job_id == "3696263":
            args.array_job_id = "3697617"
        if args.array_total == 25:
            args.array_total = 500
    return args


def main() -> None:
    args = apply_auto_defaults(parse_args())
    run_root = args.run_root.resolve()
    output_pdf = args.output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args, run_root)
    counts = aggregate_counts(run_root, rows, args.reviewer_batch_name, args.completion_summary)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pages = [draw_summary_page(run_root, rows, counts, generated_at, args)]
    pages.extend(draw_case_page(run_root, row, args.reviewer_batch_name) for row in rows)
    pages.extend(build_prompt_pages(run_root, rows, args.prompt_file.resolve()))

    pages[0].save(output_pdf, "PDF", resolution=144.0, save_all=True, append_images=pages[1:])
    manifest_path = output_pdf.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": generated_at,
                "run_root": str(run_root),
                "output_pdf": str(output_pdf),
                "page_count": len(pages),
                "case_count": len(rows),
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_reproduction(output_pdf, run_root, args, counts, generated_at)
    print(output_pdf)
    print(manifest_path)
    print(output_pdf.parent / "reproduction.txt")


if __name__ == "__main__":
    main()
