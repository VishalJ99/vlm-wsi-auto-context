#!/usr/bin/env python3
"""Run PER-237 odd-one-out artifact review on Stage 1 thumbnail crops."""

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
    _font,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runs/stage1_detector_pilot_v1/review_packet/all_detections_manifest.csv"
DEFAULT_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage6_odd_one_out_artifact_review.txt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_odd_one_out_artifact_review_v1"
)
DEFAULT_MIN_CROPS = 2
DEFAULT_SLIDES = [
    "he_patient_001_slide_001.svs",
    "evg_patient_011_slide_001.svs",
    "jones_patient_011_slide_001.svs",
    "pas_patient_011_slide_001.svs",
    "he_patient_011_slide_001.svs",
]
DEFAULT_MODELS = ["google/gemini-3-flash-preview", "google/gemini-3.1-pro-preview"]
PROMPT_VERSION = "stage6_odd_one_out_artifact_review_user_prompt_2026-05-28"
TICKET = "PER-237"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


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


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
    return slug or "model"


def _default_prompt_version(path: Path) -> str:
    if "v2" in path.stem:
        return "stage6_odd_one_out_artifact_review_v2_contains_consensus_2026-05-28"
    return PROMPT_VERSION


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _json_object_candidates(text: str) -> list[str]:
    candidates = []
    starts = [idx for idx, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = _strip_json_fences(text)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    candidates.extend(reversed(_json_object_candidates(cleaned)))
    first_payload: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            if first_payload is None:
                first_payload = payload
            if any(key in payload for key in ("consensus", "flagged_artifacts", "crops", "patches")):
                return payload, "json"
    if first_payload is not None:
        return first_payload, "json"
    return None, "unparsed"


def _parse_status_ok(status: object) -> bool:
    return str(status or "").startswith("ok")


def _recover_single_flagged_item(payload: dict[str, Any], expected_count: int) -> dict[str, Any] | None:
    """Recover when the model returns only the inconsistent crop object."""
    if not {"id", "label"}.issubset(payload):
        return None
    try:
        item_id = int(payload["id"])
    except Exception:
        return None
    if item_id < 1 or item_id > expected_count:
        return None
    label = str(payload.get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
    artifact_labels = {
        "artifact",
        "inconsistent",
        "outlier",
        "debris",
        "not_consensus",
        "non_consensus",
        "non_tissue",
    }
    if label not in artifact_labels:
        return None
    return {
        "consensus": "",
        "flagged_artifacts": [item_id],
        "crops": [dict(payload)],
        "recovered_from": "single_inconsistent_crop_object",
    }


def _parse_response(text: str, expected_count: int) -> tuple[dict[str, Any] | None, str, str]:
    payload, route = _extract_json_object(text)
    if payload is None:
        return None, route, "no_json_object"
    missing = [key for key in ("consensus", "flagged_artifacts") if key not in payload]
    items_key = "crops" if "crops" in payload else "patches" if "patches" in payload else ""
    if not items_key:
        missing.append("crops_or_patches")
    if missing:
        recovered = _recover_single_flagged_item(payload, expected_count)
        if recovered is not None:
            return recovered, route, "ok_single_flag_recovered"
        return payload, route, f"missing_keys:{','.join(missing)}"
    items = payload.get(items_key)
    flagged = payload.get("flagged_artifacts")
    if not isinstance(items, list) or not isinstance(flagged, list):
        return payload, route, "wrong_types"
    ids = []
    for item in items:
        if isinstance(item, dict) and "id" in item:
            try:
                ids.append(int(item["id"]))
            except Exception:
                pass
    if len(items) != expected_count or sorted(ids) != list(range(1, expected_count + 1)):
        return payload, route, "patch_id_mismatch"
    try:
        flagged_ids = sorted(int(item) for item in flagged)
    except Exception:
        return payload, route, "flagged_id_parse_error"
    if any(item < 1 or item > expected_count for item in flagged_ids):
        return payload, route, "flagged_id_out_of_range"
    return payload, route, "ok"


def _resolve_rows(manifest: Path, slides: list[str], all_manifest: bool) -> list[dict[str, str]]:
    rows = _read_csv(manifest)
    if all_manifest:
        return rows
    by_basename: dict[str, dict[str, str]] = {}
    for row in rows:
        basename = Path(row["wsi_path"]).name
        by_basename[basename] = row
    missing = [slide for slide in slides if slide not in by_basename]
    if missing:
        raise SystemExit(f"Missing slides in manifest: {missing}")
    return [by_basename[slide] for slide in slides]


def _filter_rows_by_crop_count(rows: list[dict[str, str]], min_crops: int) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    kept = []
    skipped = []
    for row in rows:
        crop_count = len(_load_detected_regions(Path(row["bboxes_json_path"])))
        if crop_count >= min_crops:
            kept.append(row)
            continue
        skipped.append(
            {
                "index": int(row["index"]),
                "pilot_row_id": row["pilot_row_id"],
                "case_id": row["case_id"],
                "stain": row["stain"],
                "wsi_name": Path(row["wsi_path"]).name,
                "crop_count": crop_count,
                "min_crops": min_crops,
                "skip_reason": "crop_count_below_min_crops",
            }
        )
    return kept, skipped


def _load_detected_regions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    regions = payload.get("detected_regions", [])
    if not isinstance(regions, list):
        raise SystemExit(f"Invalid detected_regions in {path}")
    return regions


def _crop_bbox(image: Image.Image, bbox: list[Any]) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {[x1, y1, x2, y2]}")
    return image.crop((x1, y1, x2, y2)).convert("RGB"), [x1, y1, x2, y2]


def _draw_contact_sheet(case_record: dict[str, Any], path: Path) -> None:
    patches = case_record["patches"]
    cols = min(4, max(1, len(patches)))
    panel_w, panel_h = 390, 300
    gap = 36
    rows = (len(patches) + cols - 1) // cols
    width = cols * panel_w + (cols + 1) * gap
    height = rows * (panel_h + 52) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    small = _font(20)
    for idx, patch in enumerate(patches):
        row = idx // cols
        col = idx % cols
        x = gap + col * (panel_w + gap)
        y = gap + row * (panel_h + 52)
        image = _thumb(Path(patch["crop_path"]), (panel_w, panel_h))
        sheet.paste(image, (x, y))
        draw.text((x, y + image.height + 8), f"patch {patch['id']} | {patch['label']}", font=small, fill="#111111")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _build_case_inputs(rows: list[dict[str, str]], args: argparse.Namespace, prompt: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        wsi_name = Path(row["wsi_path"]).name
        case_slug = _safe_slug(f"{int(row['index']):03d}_{row['stain']}_{row['case_id']}_{wsi_name}")
        case_dir = args.output_root / "inputs/cases" / case_slug
        thumb_path = Path(row["thumbnail_path"])
        bbox_path = Path(row["bboxes_json_path"])
        thumbnail = Image.open(thumb_path).convert("RGB")
        regions = _load_detected_regions(bbox_path)
        patches: list[dict[str, Any]] = []
        for patch_id, region in enumerate(regions, start=1):
            crop, clipped = _crop_bbox(thumbnail, region["bbox_thumbnail"])
            patch_dir = case_dir / "patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
            crop_path = patch_dir / f"{patch_id:02d}_{region.get('label', f'tissue_{patch_id}')}.png"
            crop.save(crop_path)
            patches.append(
                {
                    "id": patch_id,
                    "label": str(region.get("label", f"tissue_{patch_id}")),
                    "crop_path": str(crop_path),
                    "bbox_thumbnail": [int(round(float(v))) for v in region["bbox_thumbnail"]],
                    "bbox_thumbnail_clipped": clipped,
                    "bbox_level0": region.get("bbox_level0", []),
                    "bbox_normalized": region.get("bbox_normalized", []),
                    "crop_size": list(crop.size),
                }
            )
        contact_sheet_path = case_dir / "contact_sheet.png"
        case_record = {
            "task_id": f"stage6_odd_one_out_artifact_{int(row['index']):03d}",
            "ticket": TICKET,
            "case_index": int(row["index"]),
            "pilot_row_id": row["pilot_row_id"],
            "case_id": row["case_id"],
            "anon_path_id": row["Anon_Path_ID"],
            "stain": row["stain"],
            "wsi_path": row["wsi_path"],
            "wsi_name": wsi_name,
            "thumbnail_path": row["thumbnail_path"],
            "bboxes_json_path": row["bboxes_json_path"],
            "bbox_count": len(patches),
            "case_slug": case_slug,
            "patches": patches,
            "contact_sheet_path": str(contact_sheet_path),
            "prompt_version": args.prompt_version,
            "created_at": _timestamp(),
        }
        _draw_contact_sheet(case_record, contact_sheet_path)
        _write_json(case_dir / "case_input.json", case_record)
        records.append(case_record)
    _write_jsonl(args.output_root / "inputs/input_manifest.jsonl", records)
    return records


def _task_prompt(prompt: str, patch_count: int) -> str:
    return (
        prompt.strip()
        + "\n\nThe attached crop images are ordered by crop id. "
        + f"Use id 1 for the first attached image through id {patch_count} for the last attached image."
    )


def _run_one(
    case_record: dict[str, Any],
    model: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    prompt: str,
) -> dict[str, Any]:
    image_paths = [Path(patch["crop_path"]) for patch in case_record["patches"]]
    record: dict[str, Any] = {
        "task_id": case_record["task_id"],
        "ticket": TICKET,
        "case_index": case_record["case_index"],
        "pilot_row_id": case_record["pilot_row_id"],
        "case_id": case_record["case_id"],
        "anon_path_id": case_record["anon_path_id"],
        "stain": case_record["stain"],
        "wsi_path": case_record["wsi_path"],
        "wsi_name": case_record["wsi_name"],
        "bbox_count": case_record["bbox_count"],
        "patch_ids": [patch["id"] for patch in case_record["patches"]],
        "crop_paths": [str(path) for path in image_paths],
        "contact_sheet_path": case_record["contact_sheet_path"],
        "prompt_version": args.prompt_version,
        "model": model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "raw_response": "",
        "parsed_response": None,
        "parse_route": "",
        "parse_status": "not_run",
        "flagged_artifacts": [],
        "flagged_count": 0,
        "error": "",
        "usage": {},
        "response_model": "",
        "created_at": _timestamp(),
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=model,
            prompt_text=_task_prompt(prompt, case_record["bbox_count"]),
            image_paths=image_paths,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        parsed, route, status = _parse_response(raw, case_record["bbox_count"])
        flagged = parsed.get("flagged_artifacts", []) if isinstance(parsed, dict) else []
        flagged_ids = []
        for item in flagged:
            try:
                flagged_ids.append(int(item))
            except Exception:
                continue
        record.update(
            {
                "raw_response": raw,
                "parsed_response": parsed,
                "parse_route": route,
                "parse_status": status,
                "flagged_artifacts": sorted(flagged_ids),
                "flagged_count": len(flagged_ids),
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
        record["parse_status"] = "error"
    return record


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_height: int = 24,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
    return y


def _flagged_ids(result_rows: list[dict[str, Any]]) -> set[int]:
    flagged: set[int] = set()
    for row in result_rows:
        for item in row.get("flagged_artifacts", []):
            try:
                flagged.add(int(item))
            except Exception:
                continue
    return flagged


def _draw_patch_grid(
    page: Image.Image,
    patches: list[dict[str, Any]],
    x0: int,
    y0: int,
    flagged_ids: set[int] | None = None,
) -> int:
    draw = ImageDraw.Draw(page)
    small = _font(17)
    panel_w, panel_h = 300, 220
    gap_x, gap_y = 24, 48
    cols = min(4, max(1, len(patches)))
    bottom = y0
    flagged_ids = flagged_ids or set()
    for idx, patch in enumerate(patches):
        col = idx % cols
        row = idx // cols
        x = x0 + col * (panel_w + gap_x)
        y = y0 + row * (panel_h + gap_y)
        image = _thumb(Path(patch["crop_path"]), (panel_w, panel_h))
        page.paste(image, (x, y))
        is_flagged = int(patch["id"]) in flagged_ids
        if is_flagged:
            draw.rectangle(
                (x - 6, y - 6, x + image.width + 6, y + image.height + 6),
                outline="#d7191c",
                width=10,
            )
        label_fill = "#b00020" if is_flagged else "#111111"
        suffix = " | FLAGGED" if is_flagged else ""
        draw.text(
            (x, y + image.height + 5),
            f"id {patch['id']} | {patch['crop_size'][0]}x{patch['crop_size'][1]}{suffix}",
            font=small,
            fill=label_fill,
        )
        bottom = max(bottom, y + panel_h + gap_y)
    return bottom


def _patch_label_summary(parsed: dict[str, Any] | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    rows = []
    items = parsed.get("crops") if isinstance(parsed.get("crops"), list) else parsed.get("patches", [])
    for item in items or []:
        if not isinstance(item, dict):
            continue
        rows.append(f"{item.get('id')}: {item.get('label')} - {item.get('reason')}")
    return "\n".join(rows)


def _draw_case_page(case_record: dict[str, Any], result_rows: list[dict[str, Any]]) -> Image.Image:
    page = Image.new("RGB", (2550, 3300), "white")
    draw = ImageDraw.Draw(page)
    title = _font(38)
    header = _font(27)
    body = _font(19)
    small = _font(16)
    y = 45
    draw.text((55, y), f"{case_record['case_index']:03d} | {case_record['stain']} | {case_record['wsi_name']}", font=title, fill="black")
    y += 48
    y = _draw_wrapped(
        draw,
        (55, y),
        f"{case_record['case_id']} | {case_record['anon_path_id']} | crops={case_record['bbox_count']}",
        170,
        body,
        "#111111",
        24,
    )
    y += 18
    draw.text((55, y), "Raw thumbnail crops sent to VLM (no boxes drawn)", font=header, fill="black")
    y += 36
    flagged_ids = _flagged_ids(result_rows)
    if flagged_ids:
        y = _draw_wrapped(
            draw,
            (55, y),
            f"Red outline = crop ids in flagged_artifacts: {sorted(flagged_ids)}",
            140,
            small,
            "#b00020",
            20,
        )
        y += 8
    y = _draw_patch_grid(page, case_record["patches"], 55, y, flagged_ids) + 22

    col_w = 1190
    x_positions = [55, 1320]
    for idx, row in enumerate(result_rows):
        x = x_positions[idx % 2]
        yy = y
        model_label = row["model"].replace("google/", "")
        draw.text((x, yy), model_label, font=header, fill="black")
        yy += 34
        meta = (
            f"thinking={row.get('reasoning_effort')} | parse={row.get('parse_status')} | "
            f"flagged={row.get('flagged_artifacts')} | error={row.get('error', '')}"
        )
        yy = _draw_wrapped(draw, (x, yy), meta, 112, small, "#333333", 20)
        yy += 10
        parsed = row.get("parsed_response") if isinstance(row.get("parsed_response"), dict) else None
        if parsed:
            yy = _draw_wrapped(draw, (x, yy), "Consensus: " + str(parsed.get("consensus", "")), 110, body, "#111111", 24)
            yy += 12
            yy = _draw_wrapped(draw, (x, yy), _patch_label_summary(parsed), 110, small, "#111111", 19)
        else:
            yy = _draw_wrapped(draw, (x, yy), row.get("raw_response", ""), 110, small, "#111111", 19)
    return page


def _draw_cover(
    case_records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> Image.Image:
    page = Image.new("RGB", (2550, 3300), "white")
    draw = ImageDraw.Draw(page)
    title = _font(44)
    header = _font(30)
    body = _font(22)
    small = _font(18)
    y = 60
    draw.text((70, y), "PER-237 Stage 6 Odd-One-Out Artifact Review", font=title, fill="black")
    y += 62
    draw.text((70, y), f"models={', '.join(args.models)} | thinking={args.reasoning_effort}", font=body, fill="#111111")
    y += 38
    draw.text((70, y), "Inputs are exact Stage 1 thumbnail bbox crops, without drawn boxes.", font=body, fill="#111111")
    y += 50
    draw.text((70, y), "Run Summary", font=header, fill="black")
    y += 38
    model_counts: dict[str, dict[str, int]] = {}
    for row in results:
        bucket = model_counts.setdefault(
            row["model"],
            {"calls": 0, "flagged_cases": 0, "flagged_crops": 0, "parse_not_ok": 0},
        )
        bucket["calls"] += 1
        bucket["flagged_cases"] += int(bool(row.get("flagged_artifacts")))
        bucket["flagged_crops"] += len(row.get("flagged_artifacts", []))
        bucket["parse_not_ok"] += int(not _parse_status_ok(row.get("parse_status")))
    for model, counts in model_counts.items():
        line = (
            f"{model.replace('google/', '')}: calls={counts['calls']} | "
            f"flagged_cases={counts['flagged_cases']} | flagged_crops={counts['flagged_crops']} | "
            f"parse_not_ok={counts['parse_not_ok']}"
        )
        y = _draw_wrapped(draw, (95, y), line, 160, small, "#111111", 22)
    y += 24
    draw.text((70, y), "Flagged Or Non-OK Rows", font=header, fill="black")
    y += 38
    rows_to_show = [
        row
        for row in results
        if row.get("flagged_artifacts") or not _parse_status_ok(row.get("parse_status")) or row.get("error")
    ]
    if not rows_to_show:
        y = _draw_wrapped(draw, (95, y), "No rows were flagged and all parses were ok.", 160, small, "#111111", 22)
    cover_row_limit = 45
    for row in rows_to_show[:cover_row_limit]:
        line = f"{row['case_index']:03d} {row['stain']} {row['wsi_name']} | {row['model'].replace('google/', '')}: flagged={row.get('flagged_artifacts')} parse={row.get('parse_status')}"
        y = _draw_wrapped(draw, (95, y), line, 160, small, "#111111", 22)
    if len(rows_to_show) > cover_row_limit:
        y = _draw_wrapped(draw, (95, y), f"... {len(rows_to_show) - cover_row_limit} more rows in the summary CSV.", 160, small, "#111111", 22)
    y += 30
    draw.text((70, y), "Cases", font=header, fill="black")
    y += 38
    if len(case_records) <= 20:
        for case in case_records:
            line = f"{case['case_index']:03d} | {case['stain']} | {case['case_id']} | {case['anon_path_id']} | {case['wsi_name']} | crops={case['bbox_count']}"
            y = _draw_wrapped(draw, (95, y), line, 160, small, "#111111", 22)
    else:
        stain_counts: dict[str, int] = {}
        crop_counts = []
        for case in case_records:
            stain_counts[case["stain"]] = stain_counts.get(case["stain"], 0) + 1
            crop_counts.append(int(case["bbox_count"]))
        line = (
            f"{len(case_records)} cases | stains={dict(sorted(stain_counts.items()))} | "
            f"crop_count_min={min(crop_counts)} max={max(crop_counts)}"
        )
        y = _draw_wrapped(draw, (95, y), line, 160, small, "#111111", 22)
    y += 30
    draw.text((70, y), "Prompt", font=header, fill="black")
    y += 38
    _draw_wrapped(draw, (95, y), prompt, 150, small, "#111111", 22)
    return page


def _write_pdf(
    output_root: Path,
    case_records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> Path:
    by_case: dict[int, list[dict[str, Any]]] = {}
    model_order = {model: idx for idx, model in enumerate(args.models)}
    for row in results:
        by_case.setdefault(int(row["case_index"]), []).append(row)
    for rows in by_case.values():
        rows.sort(key=lambda item: model_order.get(item["model"], 999))
    pages = [_draw_cover(case_records, results, prompt, args)]
    for case in case_records:
        pages.append(_draw_case_page(case, by_case.get(int(case["case_index"]), [])))
    if len(args.models) == 1:
        pdf_name = f"stage6_odd_one_out_artifact_review_{_model_slug(args.models[0])}.pdf"
    else:
        pdf_name = "stage6_odd_one_out_artifact_review_flash_vs_pro.pdf"
    pdf_path = output_root / "visuals" / pdf_name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_summaries(output_root: Path, results: list[dict[str, Any]], args: argparse.Namespace, pdf_path: Path) -> Path:
    csv_rows = []
    for row in results:
        parsed = row.get("parsed_response") if isinstance(row.get("parsed_response"), dict) else {}
        csv_rows.append(
            {
                "case_index": row["case_index"],
                "stain": row["stain"],
                "wsi_name": row["wsi_name"],
                "model": row["model"],
                "reasoning_effort": row["reasoning_effort"],
                "bbox_count": row["bbox_count"],
                "parse_status": row["parse_status"],
                "flagged_artifacts": json.dumps(row.get("flagged_artifacts", [])),
                "flagged_count": row.get("flagged_count", 0),
                "consensus": parsed.get("consensus", ""),
                "error": row.get("error", ""),
                "raw_response": row.get("raw_response", ""),
            }
        )
    csv_path = output_root / "summary/stage6_odd_one_out_artifact_review_summary.csv"
    _write_csv(
        csv_path,
        csv_rows,
        [
            "case_index",
            "stain",
            "wsi_name",
            "model",
            "reasoning_effort",
            "bbox_count",
            "parse_status",
            "flagged_artifacts",
            "flagged_count",
            "consensus",
            "error",
            "raw_response",
        ],
    )
    model_counts: dict[str, dict[str, Any]] = {}
    for row in results:
        bucket = model_counts.setdefault(row["model"], {"cases": 0, "flagged_cases": 0, "flagged_crops": 0, "parse_status_counts": {}})
        bucket["cases"] += 1
        bucket["flagged_cases"] += int(bool(row.get("flagged_artifacts")))
        bucket["flagged_crops"] += len(row.get("flagged_artifacts", []))
        status_counts = bucket["parse_status_counts"]
        status = str(row.get("parse_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "created_at": _timestamp(),
        "ticket": TICKET,
        "git_commit": _repo_git_commit(),
        "prompt_version": args.prompt_version,
        "prompt_file": str(args.prompt.resolve()),
        "manifest": str(args.manifest.resolve()),
        "all_manifest": args.all_manifest,
        "min_crops": args.min_crops,
        "slides": args.slides,
        "skipped_cases": args.skipped_cases,
        "skipped_cases_jsonl": str((output_root / "tasks/skipped_cases_min_crops.jsonl").resolve()),
        "models": args.models,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_concurrent": args.max_concurrent,
        "cases": len({row["case_index"] for row in results}),
        "calls": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "known_usage_cost_if_reported": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in results),
        "model_counts": model_counts,
        "results_jsonl": str((output_root / "reviews/stage6_odd_one_out_artifact_review_results.jsonl").resolve()),
        "summary_csv": str(csv_path.resolve()),
        "pdf": str(pdf_path.resolve()),
    }
    summary_path = output_root / "summary/stage6_odd_one_out_artifact_review_summary.json"
    _write_json(summary_path, summary)
    return summary_path


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    prompt: str,
    pdf_path: Path,
    summary_path: Path,
) -> None:
    command_parts = [
        "python",
        "scripts/stage6_odd_one_out_artifact_review.py",
        "--manifest",
        str(args.manifest.resolve()),
        "--prompt",
        str(args.prompt.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--min-crops",
        str(args.min_crops),
        "--models",
        *args.models,
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.all_manifest:
        command_parts.append("--all-manifest")
    else:
        command_parts.extend(["--slides", *args.slides])
    if args.reuse_existing:
        command_parts.append("--reuse-existing")
    if args.rerun_incomplete:
        command_parts.append("--rerun-incomplete")
    command = " ".join(shlex.quote(part) for part in command_parts)
    text = f"""\
PER-237 Stage 6 odd-one-out artifact review
============================================

Created: {_timestamp()}
Ticket: {TICKET}
Git commit: {_repo_git_commit()}
Prompt version: {args.prompt_version}
Backend: OpenRouter-compatible chat completions
Models: {', '.join(args.models)}
Reasoning effort: {args.reasoning_effort}
Reuse existing model outputs: {args.reuse_existing}
Rerun incomplete calls only: {args.rerun_incomplete}
Paid-call regeneration: remove `--reuse-existing` from the command below to
make fresh API calls with the same inputs and settings.

Objective:
Explore whether the supplied consensus-signature / odd-one-out prompt can
identify clear low-level artifact outliers among Stage 1 thumbnail detections
without forcing a most-different crop when all crops are consistent.

Input construction:
- Read cases from {args.manifest.resolve()}.
- Select {'all manifest rows' if args.all_manifest else 'slides: ' + ', '.join(args.slides)}.
- Skip cases with fewer than {args.min_crops} crop(s). Skipped cases are listed
  in {(output_root / 'tasks/skipped_cases_min_crops.jsonl').resolve()}.
- Read each existing Stage 1 thumbnail and bboxes.json.
- Crop each detected bbox using bbox_thumbnail coordinates.
- Send only the raw thumbnail crop images to the VLM, with no boxes drawn.
- Crop ids correspond to image attachment order, starting at 1.
- PDF case pages outline any crop ids in flagged_artifacts in red.

Prompt:
{prompt}

Command:
{command}

Outputs:
- Input manifest: {(output_root / 'inputs/input_manifest.jsonl').resolve()}
- Skipped cases: {(output_root / 'tasks/skipped_cases_min_crops.jsonl').resolve()}
- Results JSONL: {(output_root / 'reviews/stage6_odd_one_out_artifact_review_results.jsonl').resolve()}
- Summary JSON: {summary_path.resolve()}
- Summary CSV: {(output_root / 'summary/stage6_odd_one_out_artifact_review_summary.csv').resolve()}
- PDF: {pdf_path.resolve()}
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    if args.all_manifest and args.slides:
        raise SystemExit("--all-manifest cannot be combined with --slides")
    if args.min_crops < 1:
        raise SystemExit("--min-crops must be >= 1")
    if args.slides is None:
        args.slides = [] if args.all_manifest else DEFAULT_SLIDES
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prompt_version = args.prompt_version or _default_prompt_version(args.prompt)
    prompt = args.prompt.read_text().strip()
    (args.output_root / "prompt").mkdir(parents=True, exist_ok=True)
    (args.output_root / "prompt" / args.prompt.name).write_text(prompt + "\n")
    rows = _resolve_rows(args.manifest, args.slides, args.all_manifest)
    rows, skipped_cases = _filter_rows_by_crop_count(rows, args.min_crops)
    args.skipped_cases = skipped_cases
    _write_jsonl(args.output_root / "tasks/skipped_cases_min_crops.jsonl", skipped_cases)
    case_records = _build_case_inputs(rows, args, prompt)
    _write_jsonl(
        args.output_root / "tasks/stage6_odd_one_out_artifact_review_tasks.jsonl",
        [
            {
                "task_id": case["task_id"],
                "case_index": case["case_index"],
                "wsi_name": case["wsi_name"],
                "stain": case["stain"],
                "crop_count": case["bbox_count"],
                "models": args.models,
                "reasoning_effort": args.reasoning_effort,
            }
            for case in case_records
        ],
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "cases": len(case_records),
                    "skipped_cases": len(skipped_cases),
                    "min_crops": args.min_crops,
                    "output_root": str(args.output_root),
                },
                indent=2,
            )
        )
        return 0

    results_path = args.output_root / "reviews/stage6_odd_one_out_artifact_review_results.jsonl"
    selected_case_indices = {int(case["case_index"]) for case in case_records}
    if args.reuse_existing and results_path.exists():
        results = [
            row
            for row in (json.loads(line) for line in results_path.read_text().splitlines() if line.strip())
            if int(row["case_index"]) in selected_case_indices and row.get("model") in args.models
        ]
        for row in results:
            row["prompt_version"] = args.prompt_version
        _write_jsonl(results_path, results)
    else:
        base_url, api_key = _api_settings(args)
        existing_results: dict[tuple[int, str], dict[str, Any]] = {}
        if args.rerun_incomplete and results_path.exists():
            for line in results_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row["case_index"]) not in selected_case_indices or row.get("model") not in args.models:
                    continue
                key = (int(row["case_index"]), str(row["model"]))
                if _parse_status_ok(row.get("parse_status")) and not row.get("error"):
                    existing_results[key] = row
        jobs = [
            (case, model)
            for model in args.models
            for case in case_records
            if (int(case["case_index"]), model) not in existing_results
        ]
        if args.max_concurrent > 1:
            results = []
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
                futures = [pool.submit(_run_one, case, model, args, base_url, api_key, prompt) for case, model in jobs]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            results = [_run_one(case, model, args, base_url, api_key, prompt) for case, model in jobs]
        results.extend(existing_results.values())
        model_order = {model: idx for idx, model in enumerate(args.models)}
        results.sort(key=lambda row: (int(row["case_index"]), model_order.get(row["model"], 999)))
        _write_jsonl(results_path, results)

    pdf_path = _write_pdf(args.output_root, case_records, results, prompt, args)
    summary_path = _write_summaries(args.output_root, results, args, pdf_path)
    _write_reproduction(args.output_root, args, prompt, pdf_path, summary_path)
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--slides", nargs="+", default=None)
    parser.add_argument("--all-manifest", action="store_true")
    parser.add_argument("--min-crops", type=int, default=DEFAULT_MIN_CROPS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
