#!/usr/bin/env python3
"""Run a VLM representative-bbox selection probe on scale-500 detections."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent, wrap
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from stage1_detection_review_pilot import _chat_with_images, _repo_git_commit


DEFAULT_SCALE500_RUN = Path("runs/detector_pipeline_scale500_v1/non_sv40")
DEFAULT_SAMPLE_MANIFEST = (
    DEFAULT_SCALE500_RUN / "visuals/per_stain20_final_detection_sample/sample_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_SCALE500_RUN / "analysis/bbox_diversity_selection_probe_v2"
)
DEFAULT_MODELS = ["google/gemini-3-flash-preview", "google/gemini-3.1-pro-preview"]
DEFAULT_REASONING_EFFORTS = ["low", "high"]
PROMPT_VERSION = "scale500_bbox_diversity_selection_v3_explicit_allowed_ids"


PROMPT = """You are given one low-resolution renal whole-slide overview image.
Green numbered rectangles are final detector boxes that could be sent to a downstream segmentation pipeline.

The source slides are serial renal sections. Multiple boxes can therefore be redundant: they may show the same tissue geometry, stain appearance, fragmentation pattern, artifact/background pattern, and foreground-boundary difficulty repeated at another location.

Task:
Select the smallest set of box IDs that preserves coverage of all distinct segmentation inputs.

Decision rule:
- Treat two boxes as equivalent if a segmentation model would face the same foreground extraction problem in both boxes.
- Equivalence is based on tissue geometry/topology, number and arrangement of tissue fragments, stain intensity/contrast, tissue texture, artifact/background contamination, crop looseness, edge truncation, and boundary difficulty.
- Translation, rotation, mirror orientation, small scale differences, and slide location alone do not make a box distinct.
- Do not select a box because it is diagnostically interesting. Do not infer disease. This is only a segmentation-input coverage task.
- For each equivalent group, select one representative box. Prefer the representative with clearer visible boundaries and less unnecessary background.
- If the overview is insufficient to decide whether a box is redundant, select that box and include it in uncertain_box_ids.
- Include a concise selection_justification explaining why the selected set preserves the distinct segmentation inputs and why the omitted boxes are redundant.

Coverage contract:
- Every input box ID must appear in exactly one redundancy_groups[].member_box_ids list.
- Each redundancy group must have exactly one representative_box_id, and that representative must be in selected_box_ids.
- selected_box_ids should be the union of all representative_box_id values plus any uncertain boxes that need to be kept.

Return only valid JSON with this schema:
{
  "selected_box_ids": [1],
  "selection_justification": "brief visual justification for the selected set based only on segmentation-input coverage",
  "redundancy_groups": [
    {
      "representative_box_id": 1,
      "member_box_ids": [1],
      "equivalence_reason": "short visual reason based only on segmentation-input equivalence"
    }
  ],
  "uncertain_box_ids": [],
  "notes": "brief note; empty string if none"
}
"""
PROMPT_APPENDIX_TEMPLATE = """Per-case appendix appended to every model call:
Allowed box IDs exactly for this image: {allowed_box_ids}.
Do not use any box ID outside this list.
"""


def prompt_record_text() -> str:
    return f"{PROMPT.rstrip()}\n\n{PROMPT_APPENDIX_TEMPLATE.rstrip()}\n"


def prompt_text_for_case(case: dict[str, Any]) -> str:
    allowed_ids = list(range(1, int(case["bbox_count"]) + 1))
    appendix = PROMPT_APPENDIX_TEMPLATE.format(allowed_box_ids=allowed_ids)
    return f"{PROMPT.rstrip()}\n\n{appendix.rstrip()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def norm_yxyx_to_xyxy(box: list[float], image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    y1, x1, y2, x2 = [float(v) for v in box]
    return [
        int(round(x1 / 1000.0 * width)),
        int(round(y1 / 1000.0 * height)),
        int(round(x2 / 1000.0 * width)),
        int(round(y2 / 1000.0 * height)),
    ]


EXPECTED_SELECTION_KEYS = {"selected_box_ids", "redundancy_groups"}


def is_selection_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and EXPECTED_SELECTION_KEYS.issubset(payload.keys())


def extract_json_payload(text: str) -> tuple[Any, str]:
    stripped = text.strip()
    candidates: list[tuple[str, str]] = []
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part:
                candidates.append((part, "fenced"))
    candidates.append((stripped, "direct"))
    decoder = json.JSONDecoder()
    for candidate, source in candidates:
        try:
            payload = json.loads(candidate)
            if is_selection_payload(payload):
                return payload, source
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        if start >= 0:
            try:
                payload, _ = decoder.raw_decode(candidate[start:])
                if is_selection_payload(payload):
                    return payload, f"{source}_raw_decode"
            except json.JSONDecodeError:
                pass
    return {"raw_text": text}, "parse_error"


def normalize_id_list(value: Any, max_id: int) -> tuple[list[int], list[Any]]:
    out: list[int] = []
    invalid: list[Any] = []
    if not isinstance(value, list):
        if value is not None:
            invalid.append(value)
        return out, invalid
    for item in value:
        try:
            idx = int(item)
        except Exception:
            invalid.append(item)
            continue
        if 1 <= idx <= max_id:
            if idx not in out:
                out.append(idx)
        else:
            invalid.append(item)
    return out, invalid


def validate_selection(payload: Any, bbox_count: int) -> tuple[dict[str, Any] | None, str]:
    if not is_selection_payload(payload):
        selected = list(range(1, bbox_count + 1))
        return (
            {
                "selected_box_ids": selected,
                "raw_selected_box_ids": [],
                "redundancy_groups": [],
                "uncertain_box_ids": [],
                "missing_group_box_ids": selected,
                "duplicate_group_box_ids": [],
                "invalid_box_id_values": [],
                "representative_not_in_members": [],
                "representative_not_in_raw_selected": [],
                "raw_selected_not_equal_contract": False,
                "selection_justification": "",
                "usable_for_comparison": False,
                "fallback_reason": "fallback_all_due_to_parse_failure",
                "notes": "",
            },
            "parse_error",
        )
    raw_selected, invalid_selected = normalize_id_list(payload.get("selected_box_ids"), bbox_count)
    uncertain, invalid_uncertain = normalize_id_list(payload.get("uncertain_box_ids"), bbox_count)
    groups_out: list[dict[str, Any]] = []
    member_counts: Counter[int] = Counter()
    representative_ids: list[int] = []
    invalid_values: list[Any] = invalid_selected + invalid_uncertain
    representative_not_in_members: list[int] = []
    representative_not_in_raw_selected: list[int] = []
    groups = payload.get("redundancy_groups")
    if not isinstance(groups, list):
        groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        members, invalid_members = normalize_id_list(group.get("member_box_ids"), bbox_count)
        invalid_values.extend(invalid_members)
        try:
            rep = int(group.get("representative_box_id"))
        except Exception:
            invalid_values.append(group.get("representative_box_id"))
            rep = members[0] if members else 0
        if not (1 <= rep <= bbox_count):
            invalid_values.append(group.get("representative_box_id"))
            rep = members[0] if members else 0
        if rep and rep not in members:
            representative_not_in_members.append(rep)
            members.insert(0, rep)
        if rep and rep not in representative_ids:
            representative_ids.append(rep)
        if rep and rep not in raw_selected:
            representative_not_in_raw_selected.append(rep)
        for member in members:
            member_counts[member] += 1
        groups_out.append(
            {
                "representative_box_id": rep if rep else None,
                "member_box_ids": members,
                "equivalence_reason": str(group.get("equivalence_reason", ""))[:500],
            }
        )
    expected_selected = sorted(set(representative_ids + uncertain))
    raw_selected_not_equal_contract = bool(expected_selected and sorted(raw_selected) != expected_selected)
    selected = sorted(set(raw_selected + expected_selected))
    missing = [idx for idx in range(1, bbox_count + 1) if member_counts[idx] == 0]
    duplicates = sorted(idx for idx, count in member_counts.items() if count > 1)
    fallback_reason = ""
    if not selected:
        selected = [idx for idx in range(1, bbox_count + 1)]
        fallback_reason = "fallback_all_due_to_empty_selection"
    normalized = {
        "selected_box_ids": selected,
        "raw_selected_box_ids": raw_selected,
        "redundancy_groups": groups_out,
        "uncertain_box_ids": uncertain,
        "missing_group_box_ids": missing,
        "duplicate_group_box_ids": duplicates,
        "invalid_box_id_values": invalid_values,
        "representative_not_in_members": representative_not_in_members,
        "representative_not_in_raw_selected": representative_not_in_raw_selected,
        "raw_selected_not_equal_contract": raw_selected_not_equal_contract,
        "selection_justification": str(payload.get("selection_justification", ""))[:2000],
        "usable_for_comparison": False,
        "fallback_reason": fallback_reason,
        "notes": str(payload.get("notes", ""))[:1000],
    }
    status = "ok"
    if (
        missing
        or duplicates
        or invalid_values
        or representative_not_in_members
        or representative_not_in_raw_selected
        or raw_selected_not_equal_contract
        or fallback_reason
    ):
        status = "schema_warning"
    normalized["usable_for_comparison"] = status == "ok"
    return normalized, status


def draw_overview(
    thumbnail_path: Path,
    boxes: list[list[float]],
    output_path: Path,
    *,
    selected_ids: set[int] | None = None,
    title: str = "",
) -> Path:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    max_dim = max(image.size)
    line_width = max(5, max_dim // 300)
    label_font = font(max(38, min(72, max_dim // 34)))
    title_font = font(max(30, min(58, max_dim // 42)))
    green = "#138a3d"
    red = "#d7191c"
    selected_ids = selected_ids or set()
    label_items: list[tuple[str, tuple[int, int], str]] = []
    for order, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = norm_yxyx_to_xyxy(box, image.size)
        color = red if order in selected_ids else green
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = str(order)
        tb = draw.textbbox((0, 0), label, font=label_font, stroke_width=3)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        x = max(2, min(image.size[0] - text_w - 2, x1 + 8))
        y = max(2, min(image.size[1] - text_h - 2, y1 + 8))
        label_items.append((label, (x, y), color))
    for label, xy, color in label_items:
        draw.text(
            xy,
            label,
            font=label_font,
            fill=color,
            stroke_width=3,
            stroke_fill="white",
        )
    if title:
        draw.text(
            (12, 10),
            title,
            font=title_font,
            fill="#111111",
            stroke_width=4,
            stroke_fill="white",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def manifest_stain(row: dict[str, Any]) -> str:
    value = row.get("stain") or row.get("stain_label") or row.get("stain_type")
    return str(value or "").strip()


def case_records_from_manifest(
    manifest_path: Path,
    case_ids: list[str],
    count: int,
    cases_per_stain: int | None,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise SystemExit(f"Manifest does not contain records list: {manifest_path}")
    if case_ids:
        wanted = {case_id.replace("-", "_") for case_id in case_ids}
        selected = [row for row in records if str(row.get("case_id", "")).replace("-", "_") in wanted]
    elif cases_per_stain is not None:
        if cases_per_stain <= 0:
            raise SystemExit("--cases-per-stain must be positive")
        by_stain: dict[str, list[dict[str, Any]]] = {}
        stain_order: list[str] = []
        for row in records:
            stain = manifest_stain(row)
            if not stain:
                raise SystemExit(
                    f"Manifest row is missing stain/stain_label/stain_type: {row.get('case_id')}"
                )
            if stain not in by_stain:
                by_stain[stain] = []
                stain_order.append(stain)
            by_stain[stain].append(row)
        selected = []
        for stain in stain_order:
            candidates = by_stain[stain]
            if len(candidates) < cases_per_stain:
                raise SystemExit(
                    f"Requested {cases_per_stain} cases for stain {stain}, "
                    f"but manifest contains only {len(candidates)}"
                )
            selected.extend(candidates[:cases_per_stain])
    else:
        selected = records[:count]
    if len(selected) != (len(case_ids) if case_ids else count):
        if cases_per_stain is not None and not case_ids:
            return selected
        raise SystemExit(f"Could not resolve requested cases from {manifest_path}")
    return selected


def case_records_from_run_dir(run_dir: Path, stain: str, count: int) -> list[dict[str, Any]]:
    all_detections_path = run_dir / "all_detections.json"
    records = read_json(all_detections_path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected list in {all_detections_path}")
    if len(records) < count:
        raise SystemExit(f"Requested {count} cases from {run_dir}, but found only {len(records)}")
    selected: list[dict[str, Any]] = []
    for record in records[:count]:
        case_id = str(record["case_id"]).replace("-", "_")
        case_dir = Path(record.get("case_dir") or (run_dir / case_id))
        selected.append(
            {
                "case_id": case_id,
                "stain": stain,
                "stain_label": stain,
                "wsi_path": record.get("wsi_path", ""),
                "pipeline_status": record.get("pipeline_status", ""),
                "final_boxes": len(record.get("detections", [])),
                "case_dir": str(case_dir),
                "detections_path": str(case_dir / "detections.json"),
            }
        )
    return selected


def build_case_payload(case_row: dict[str, Any], scale500_run_dir: Path, output_dir: Path) -> dict[str, Any]:
    case_id = str(case_row["case_id"]).replace("-", "_")
    if case_row.get("detections_path"):
        detections_path = Path(str(case_row["detections_path"]))
    elif case_row.get("case_dir"):
        detections_path = Path(str(case_row["case_dir"])) / "detections.json"
    else:
        detections_path = scale500_run_dir / case_id / "detections.json"
    record = read_json(detections_path)
    boxes = [det["box_2d"] for det in record.get("detections", [])]
    if not boxes:
        raise SystemExit(f"No detections found in {detections_path}")
    thumbnail_path = Path(record["paths"]["thumbnail_path"])
    case_dir = output_dir / "cases" / case_id
    input_path = draw_overview(
        thumbnail_path,
        boxes,
        case_dir / "model_input.png",
        title=f"{case_id} | boxes={len(boxes)}",
    )
    write_json(
        case_dir / "case_input.json",
        {
            "case_id": case_id,
            "case_row": case_row,
            "detections_path": str(detections_path),
            "thumbnail_path": str(thumbnail_path),
            "model_input_path": str(input_path),
            "model_input_size_px": list(Image.open(input_path).size),
            "boxes_yxyx_normalized_0_1000": boxes,
        },
    )
    return {
        "case_id": case_id,
        "case_row": case_row,
        "detections_path": str(detections_path),
        "thumbnail_path": str(thumbnail_path),
        "model_input_path": str(input_path),
        "bbox_count": len(boxes),
        "boxes": boxes,
    }


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def result_dir_for(args: argparse.Namespace, case: dict[str, Any], model: str, effort: str) -> Path:
    return Path(args.output_dir) / "results" / case["case_id"] / f"{model_slug(model)}_{effort}"


def result_record_template(
    case: dict[str, Any],
    model: str,
    effort: str,
    args: argparse.Namespace,
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_dir = result_dir_for(args, case, model, effort)
    existing = existing or {}
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "bbox_count": case["bbox_count"],
        "model": model,
        "reasoning_effort": effort,
        "temperature": existing.get("temperature", args.temperature),
        "max_tokens": existing.get("max_tokens", args.max_tokens),
        "prompt_version": existing.get("prompt_version", PROMPT_VERSION),
        "model_input_path": case["model_input_path"],
        "raw_response_path": str(result_dir / "raw_response.txt"),
        "parsed_response_path": str(result_dir / "parsed_response.json"),
        "selected_overlay_path": str(result_dir / "selected_overlay.png"),
        "parse_route": "",
        "parse_status": "not_run",
        "error": "",
        "selected_box_ids": [],
        "selection_justification": "",
        "usable_for_comparison": False,
        "fallback_reason": "",
        "usage": existing.get("usage", {}),
        "response_model": existing.get("response_model", ""),
        "created_at": existing.get("created_at", utc_now()),
    }
    return record


def parse_response_to_result(
    raw: str,
    case: dict[str, Any],
    model: str,
    effort: str,
    args: argparse.Namespace,
    *,
    existing: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    response_model: str = "",
) -> dict[str, Any]:
    result_dir = result_dir_for(args, case, model, effort)
    record = result_record_template(case, model, effort, args, existing=existing)
    payload, route = extract_json_payload(raw)
    normalized, status = validate_selection(payload, case["bbox_count"])
    write_json(result_dir / "parsed_response.json", normalized or payload)
    selected = set(normalized["selected_box_ids"] if normalized else [])
    fallback = str((normalized or {}).get("fallback_reason", ""))
    status_suffix = f" | {status}" if status != "ok" else ""
    overlay_path = draw_overview(
        Path(case["thumbnail_path"]),
        case["boxes"],
        result_dir / "selected_overlay.png",
        selected_ids=selected,
        title=(
            f"{model.split('/')[-1]} | {effort} | selected "
            f"{len(selected)}/{case['bbox_count']}{status_suffix}"
        ),
    )
    record.update(
        {
            "parse_route": route,
            "parse_status": status,
            "selected_box_ids": sorted(selected),
            "selection_justification": str((normalized or {}).get("selection_justification", "")),
            "usable_for_comparison": bool((normalized or {}).get("usable_for_comparison", False)),
            "fallback_reason": fallback,
            "usage": usage if usage is not None else record.get("usage", {}),
            "response_model": response_model or record.get("response_model", ""),
            "selected_overlay_path": str(overlay_path),
            "error": "",
        }
    )
    write_json(result_dir / "result.json", record)
    return record


def run_one(job: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    case = job["case"]
    model = job["model"]
    effort = job["reasoning_effort"]
    result_dir = result_dir_for(args, case, model, effort)
    raw_response_path = result_dir / "raw_response.txt"
    record = result_record_template(case, model, effort, args)
    try:
        raw, usage, response_model = _chat_with_images(
            model=model,
            prompt_text=prompt_text_for_case(case),
            image_paths=[Path(case["model_input_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=effort,
        )
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        raw_response_path.write_text(raw, encoding="utf-8")
        return parse_response_to_result(
            raw,
            case,
            model,
            effort,
            args,
            existing=record,
            usage=usage,
            response_model=response_model,
        )
    except Exception as exc:  # pragma: no cover - preserves batch progress.
        record["error"] = repr(exc)
        record["parse_status"] = "error"
    write_json(result_dir / "result.json", record)
    return record


def reparse_existing_one(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = job["case"]
    model = job["model"]
    effort = job["reasoning_effort"]
    result_dir = result_dir_for(args, case, model, effort)
    result_path = result_dir / "result.json"
    existing = read_json(result_path) if result_path.exists() else {}
    raw_response_path = result_dir / "raw_response.txt"
    if not raw_response_path.exists():
        record = result_record_template(case, model, effort, args, existing=existing)
        record["error"] = f"Missing raw response: {raw_response_path}"
        record["parse_status"] = "error"
        write_json(result_path, record)
        return record
    raw = raw_response_path.read_text(encoding="utf-8")
    return parse_response_to_result(
        raw,
        case,
        model,
        effort,
        args,
        existing=existing,
        usage=existing.get("usage", {}),
        response_model=str(existing.get("response_model", "")),
    )


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), candidate, font=text_font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    max_width: int,
    *,
    fill: str = "black",
    line_spacing: int = 7,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap_text_to_width(draw, text, text_font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[: max(0, max_lines - 1)] + ["..."]
    line_height = draw.textbbox((0, 0), "Ag", font=text_font)[3] + line_spacing
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height
    return y


def parsed_response_for_result(row: dict[str, Any]) -> dict[str, Any]:
    parsed_path = Path(str(row.get("parsed_response_path", "")))
    if not parsed_path.exists():
        return {}
    try:
        payload = read_json(parsed_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def format_group_summary(parsed: dict[str, Any], *, limit: int = 5) -> str:
    groups = parsed.get("redundancy_groups")
    if not isinstance(groups, list):
        return ""
    parts: list[str] = []
    for group in groups[:limit]:
        if not isinstance(group, dict):
            continue
        rep = group.get("representative_box_id")
        members = group.get("member_box_ids")
        reason = str(group.get("equivalence_reason", "")).strip()
        parts.append(f"rep {rep}: members {members}; {reason}")
    if len(groups) > limit:
        parts.append(f"... {len(groups) - limit} more groups")
    return " | ".join(parts)


def prompt_pages(prompt: str, page_w: int, page_h: int) -> list[Image.Image]:
    title_font = font(46)
    body_font = font(25)
    margin_x = 70
    margin_y = 60
    line_h = 36
    usable_lines = (page_h - 180) // line_h
    paragraphs = prompt.splitlines()
    dummy = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(dummy)
    prompt_lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            prompt_lines.append("")
            continue
        prompt_lines.extend(wrap(paragraph, width=120))
    pages: list[Image.Image] = []
    for offset in range(0, len(prompt_lines), usable_lines):
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin_x, margin_y), "Prompt", font=title_font, fill="black")
        y = margin_y + 70
        for line in prompt_lines[offset : offset + usable_lines]:
            draw.text((margin_x, y), line, font=body_font, fill="black")
            y += line_h
        pages.append(page)
    return pages


def make_review_pdf(case_payloads: list[dict[str, Any]], results: list[dict[str, Any]], output_path: Path) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_case.setdefault(row["case_id"], []).append(row)
    page_w, page_h = 2200, 3000
    pages: list[Image.Image] = prompt_pages(prompt_record_text(), page_w, page_h)
    title_font = font(42)
    body_font = font(26)
    small_font = font(20)
    for case in case_payloads:
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = 45
        draw.text((55, y), f"{case['case_id']} | boxes={case['bbox_count']}", font=title_font, fill="black")
        y += 60
        allowed_ids = list(range(1, int(case["bbox_count"]) + 1))
        draw.text((55, y), f"Allowed IDs in prompt: {allowed_ids}", font=small_font, fill="black")
        y += 34
        input_img = Image.open(case["model_input_path"]).convert("RGB")
        input_img.thumbnail((1000, 620))
        draw.text((55, y), "Exact model input", font=body_font, fill="black")
        page.paste(input_img, (55, y + 40))
        x0 = 1120
        draw.text((x0, y), "Model selections: red=selected, green=not selected", font=body_font, fill="black")
        yy = y + 40
        for row in sorted(by_case.get(case["case_id"], []), key=lambda r: (r["model"], r["reasoning_effort"])):
            img_path = Path(row["selected_overlay_path"])
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((470, 300))
            else:
                img = Image.new("RGB", (470, 300), "#f4f4f4")
            label = (
                f"{row['model'].replace('google/', '')} | {row['reasoning_effort']} | "
                f"selected={row.get('selected_box_ids')} | {row.get('parse_status')}"
            )
            draw.text((x0, yy), label[:84], font=small_font, fill="black")
            page.paste(img, (x0, yy + 28))
            yy += 350
        y_text = max(y + 730, yy + 10)
        draw.text((55, y_text), "Model justifications", font=body_font, fill="black")
        y_text += 38
        for row in sorted(by_case.get(case["case_id"], []), key=lambda r: (r["model"], r["reasoning_effort"])):
            parsed = parsed_response_for_result(row)
            justification = str(
                row.get("selection_justification")
                or parsed.get("selection_justification")
                or parsed.get("notes")
                or ""
            ).strip()
            group_summary = format_group_summary(parsed)
            block = (
                f"{row['model'].replace('google/', '')} | {row['reasoning_effort']} | "
                f"selected={row.get('selected_box_ids')} | {row.get('parse_status')}\n"
                f"Justification: {justification or '<none returned>'}\n"
                f"Groups: {group_summary or '<none returned>'}"
            )
            y_text = draw_wrapped_text(
                draw,
                (70, y_text),
                block,
                small_font,
                page_w - 140,
                max_lines=8,
            )
            y_text += 18
        pages.append(page)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = pages[0], pages[1:]
    first.save(output_path, "PDF", save_all=True, append_images=rest, resolution=150)


def write_reproduction(args: argparse.Namespace, case_payloads: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    cases = "\n".join(f"- {case['case_id']}" for case in case_payloads)
    stain_counts: dict[str, int] = {}
    for case in case_payloads:
        stain = manifest_stain(case.get("case_row", {})) or "unknown"
        stain_counts[stain] = stain_counts.get(stain, 0) + 1
    stain_count_lines = "\n".join(f"- {stain}: {count}" for stain, count in stain_counts.items())
    models = ", ".join(args.models)
    efforts = ", ".join(args.reasoning_efforts)
    status_counts = Counter(str(row.get("parse_status", "unknown")) for row in results)
    status_count_lines = "\n".join(f"- {status}: {count}" for status, count in sorted(status_counts.items())) or "- none: 0"
    max_token_counts = Counter(str(row.get("max_tokens", "unknown")) for row in results)
    max_token_count_lines = "\n".join(f"- {value}: {count}" for value, count in sorted(max_token_counts.items()))
    usable = sum(1 for row in results if row.get("usable_for_comparison"))
    if args.case:
        selection_command = " \\\n  ".join(f"--case {case_id}" for case_id in args.case)
    elif args.cases_per_stain is not None:
        selection_command = f"--cases-per-stain {args.cases_per_stain}"
    else:
        selection_command = f"--case-count {args.case_count}"
    if args.sv40_count:
        selection_command = (
            f"{selection_command} \\\n"
            f"  --sv40-run-dir {Path(args.sv40_run_dir)} \\\n"
            f"  --sv40-count {args.sv40_count}"
        )
    git_status = command_report(["git", "status", "--short", "--branch"])
    dvc_status = command_report(["dvc", "status"])
    text = f"""Scale500 bbox diversity selection probe

Created: {utc_now()}
Ticket/context: PER-250 follow-on selector probe; branch had no active Linear issue from linear-cli context.
Git commit: {_repo_git_commit()}
Git status at creation:
{indent(git_status, "  ")}
DVC status at creation:
{indent(dvc_status, "  ")}

Inputs:
- Scale500 run dir: {Path(args.scale500_run_dir).resolve()}
- Sample manifest: {Path(args.sample_manifest).resolve()}
- Case selection: {selection_command}
- Stain counts:
{stain_count_lines}
- Cases:
{cases}

Prompt:
- Version: {PROMPT_VERSION}
- Prompt text is stored at: {output_dir.resolve() / 'prompt.txt'}
- Prompt text is embedded as the first page(s) of the review PDF.
- Each model call appended the case-specific allowed box-id list.

Model calls:
- Backend: OpenRouter-compatible chat completions
- API base: {args.api_base}
- Models: {models}
- Reasoning efforts: {efforts}
- Temperature: {args.temperature}
- Max tokens: {args.max_tokens}
- Max concurrent calls: {args.max_concurrent}
- Reparse existing raw responses: {args.reparse_existing}
- Rerun statuses: {", ".join(args.rerun_statuses) if args.rerun_statuses else ""}
- Review PDF skipped: {args.skip_pdf}
- Result max-token counts:
{max_token_count_lines}
- Usable for comparison: {usable}/{len(results)}
- Parse status counts:
{status_count_lines}

Primary outputs:
- Exact model inputs: {output_dir.resolve() / 'cases/<case_id>/model_input.png'}
- Raw responses: {output_dir.resolve() / 'results/<case_id>/<model>_<effort>/raw_response.txt'}
- Parsed responses: {output_dir.resolve() / 'results/<case_id>/<model>_<effort>/parsed_response.json'}
- Red/green selection overlays: {output_dir.resolve() / 'results/<case_id>/<model>_<effort>/selected_overlay.png'}
- Summary JSONL: {output_dir.resolve() / 'summary/results.jsonl'}
- Summary CSV: {output_dir.resolve() / 'summary/results.csv'}
- Review PDF: {output_dir.resolve() / 'visuals/bbox_diversity_selection_probe.pdf'}

Regeneration command:
python scripts/scale500_bbox_diversity_selection_probe.py \\
  --scale500-run-dir {Path(args.scale500_run_dir)} \\
  --sample-manifest {Path(args.sample_manifest)} \\
  --output-dir {Path(args.output_dir)} \\
  {selection_command} \\
  --models {' '.join(args.models)} \\
  --reasoning-efforts {' '.join(args.reasoning_efforts)} \\
  --temperature {args.temperature} \\
  --max-tokens {args.max_tokens} \\
  --max-concurrent {args.max_concurrent}

Existing-response reparse command:
python scripts/scale500_bbox_diversity_selection_probe.py \\
  --scale500-run-dir {Path(args.scale500_run_dir)} \\
  --sample-manifest {Path(args.sample_manifest)} \\
  --output-dir {Path(args.output_dir)} \\
  {selection_command} \\
  --models {' '.join(args.models)} \\
  --reasoning-efforts {' '.join(args.reasoning_efforts)} \\
  --temperature {args.temperature} \\
  --max-tokens {args.max_tokens} \\
  --max-concurrent {args.max_concurrent} \\
  --reparse-existing
"""
    (output_dir / "reproduction.txt").write_text(text, encoding="utf-8")
    (output_dir / "prompt.txt").write_text(prompt_record_text(), encoding="utf-8")


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "bbox_count",
        "model",
        "reasoning_effort",
        "parse_status",
        "selected_box_ids",
        "selection_justification",
        "usable_for_comparison",
        "fallback_reason",
        "parse_route",
        "selected_overlay_path",
        "model_input_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field)) if field == "selected_box_ids" else row.get(field, "") for field in fields})


def command_report(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return f"unavailable: {exc}"
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if not output:
        output = "<no output>"
    return f"exit_code={result.returncode}\n{output}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale500-run-dir", type=Path, default=DEFAULT_SCALE500_RUN)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", default=[], help="Case id to include. Defaults to first --case-count records.")
    parser.add_argument("--case-count", type=int, default=5)
    parser.add_argument("--cases-per-stain", type=int, default=None, help="Select the first N manifest records for each stain.")
    parser.add_argument("--sv40-run-dir", type=Path, default=None, help="Optional scale500 SV40 detector run dir to append.")
    parser.add_argument("--sv40-count", type=int, default=0, help="Number of cases to append from --sv40-run-dir, labelled SV40.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reasoning-efforts", nargs="+", default=DEFAULT_REASONING_EFFORTS, choices=["low", "medium", "high"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip the review PDF; useful for large production batches.")
    parser.add_argument("--reparse-existing", action="store_true", help="Reparse saved raw responses without making model calls.")
    parser.add_argument(
        "--rerun-statuses",
        nargs="+",
        default=[],
        help="Only rerun existing rows whose current result.json parse_status is in this list.",
    )
    return parser.parse_args()


def load_existing_result(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    result_path = result_dir_for(
        args,
        job["case"],
        job["model"],
        job["reasoning_effort"],
    ) / "result.json"
    if not result_path.exists():
        return None
    result = read_json(result_path)
    if isinstance(result, dict):
        return result
    return None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = case_records_from_manifest(
        Path(args.sample_manifest),
        args.case,
        args.case_count,
        args.cases_per_stain,
    )
    if args.sv40_count:
        if args.sv40_run_dir is None:
            raise SystemExit("--sv40-count requires --sv40-run-dir")
        case_rows.extend(case_records_from_run_dir(Path(args.sv40_run_dir), "SV40", args.sv40_count))
    case_payloads = [build_case_payload(row, Path(args.scale500_run_dir), output_dir) for row in case_rows]
    write_json(output_dir / "summary/cases.json", case_payloads)
    (output_dir / "prompt.txt").write_text(prompt_record_text(), encoding="utf-8")
    if args.prepare_only:
        write_reproduction(args, case_payloads, [])
        print(json.dumps({"prepared_cases": len(case_payloads), "output_dir": str(output_dir)}, indent=2))
        return 0
    jobs = [
        {"case": case, "model": model, "reasoning_effort": effort}
        for case in case_payloads
        for model in args.models
        for effort in args.reasoning_efforts
    ]
    if args.reparse_existing:
        results = [reparse_existing_one(job, args) for job in jobs]
        results.sort(key=lambda row: (row["case_id"], row["model"], row["reasoning_effort"]))
        write_jsonl(output_dir / "summary/results.jsonl", results)
        write_csv_summary(output_dir / "summary/results.csv", results)
        if not args.skip_pdf:
            make_review_pdf(case_payloads, results, output_dir / "visuals/bbox_diversity_selection_probe.pdf")
        write_reproduction(args, case_payloads, results)
        print(json.dumps({"output_dir": str(output_dir), "cases": len(case_payloads), "results": len(results), "mode": "reparse_existing"}, indent=2))
        return 0
    jobs_to_run = jobs
    results: list[dict[str, Any]] = []
    rerun_statuses = set(args.rerun_statuses)
    if rerun_statuses:
        jobs_to_run = []
        for job in jobs:
            existing = load_existing_result(job, args)
            if existing is not None and str(existing.get("parse_status")) not in rerun_statuses:
                results.append(existing)
            else:
                jobs_to_run.append(job)
        print(
            json.dumps(
                {
                    "mode": "rerun_statuses",
                    "statuses": sorted(rerun_statuses),
                    "jobs_to_run": len(jobs_to_run),
                    "existing_kept": len(results),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if jobs_to_run:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
        with ThreadPoolExecutor(max_workers=max(1, args.max_concurrent)) as pool:
            futures = [pool.submit(run_one, job, args, args.api_base, api_key) for job in jobs_to_run]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"case": result["case_id"], "model": result["model"], "effort": result["reasoning_effort"], "status": result["parse_status"], "selected": result["selected_box_ids"]}, sort_keys=True), flush=True)
    results.sort(key=lambda row: (row["case_id"], row["model"], row["reasoning_effort"]))
    write_jsonl(output_dir / "summary/results.jsonl", results)
    write_csv_summary(output_dir / "summary/results.csv", results)
    if not args.skip_pdf:
        make_review_pdf(case_payloads, results, output_dir / "visuals/bbox_diversity_selection_probe.pdf")
    write_reproduction(args, case_payloads, results)
    print(json.dumps({"output_dir": str(output_dir), "cases": len(case_payloads), "results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
