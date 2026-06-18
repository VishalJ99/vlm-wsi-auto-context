#!/usr/bin/env python3
"""Compare prompt-side and verifier-side artifact redundancy handling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent, wrap
from typing import Any, Iterable

from PIL import Image, ImageDraw

from scale500_bbox_diversity_selection_probe import (
    PROMPT,
    draw_overview,
    extract_json_payload,
    font,
    read_json,
    validate_selection,
    write_json,
    write_jsonl,
)
from stage1_detection_review_pilot import _chat_with_images, _repo_git_commit


DEFAULT_INPUT_ROOT = Path(
    "runs/detector_pipeline_scale500_v1/analysis/bbox_diversity_selection_probe_50case_per_stain10_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/detector_pipeline_scale500_v1/analysis/artifact_redundancy_probe_v1"
)
DEFAULT_ALL_CASE_OUTPUT_DIR = Path(
    "runs/detector_pipeline_scale500_v1/analysis/artifact_redundancy_probe_50case_prohigh_v1"
)
DEFAULT_TARGET_CASES = [
    "anon_17ae73f3_f345_4529_acac_fc117d1eda3b",
    "anon_18576685_8921_446a_a027_e1e330187f18",
]
PROMPT_UPDATE_VERSION = "artifact_dominance_selector_v1"
VERIFIER_PROMPT_VERSION = "artifact_dominance_verifier_flash_low_v1"


ARTIFACT_DOMINANCE_RULE = """Additional artifact-dominance rule:
- If two boxes show the same main tissue fragment or same serial-section counterpart, but one selected box is clean and the other selected box contains an added marker/ink/dark artifact/background contaminant, do not keep both just because one is clean and one is artifacted.
- Default to the artifacted box as the single representative for that main fragment, because it includes the same tissue foreground problem plus the added artifact/background challenge.
- Keep the clean counterpart separately only if it has a distinct tissue geometry, boundary/truncation condition, or foreground fragment not present in the artifacted box.
- In artifact_dominance_decisions, explicitly list every clean-vs-artifact selected-pair you considered and whether the clean box was dropped or retained.
"""


DIRECT_SCHEMA = """Return only valid JSON with this schema:
{
  "selected_box_ids": [1],
  "selection_justification": "brief visual justification for the selected set",
  "artifact_dominance_decisions": [
    {
      "artifacted_box_id": 1,
      "clean_counterpart_box_ids": [2],
      "action": "dropped_clean_counterpart",
      "reason": "short visual reason"
    }
  ],
  "redundancy_groups": [
    {
      "representative_box_id": 1,
      "member_box_ids": [1],
      "equivalence_reason": "short visual reason"
    }
  ],
  "uncertain_box_ids": [],
  "notes": "brief note; empty string if none"
}
"""


VERIFIER_PROMPT = """You are auditing a bbox selection for a renal whole-slide overview.
The image has green numbered boxes. A previous selector already chose some box IDs.

Audit target:
Catch selected clean/artifact pairs where both boxes show the same main tissue fragment or serial-section counterpart, but one selected box has an added marker/ink/dark artifact/background contaminant.

Default rule:
- If an artifacted selected box contains the same main tissue fragment as a clean selected box, keep only the artifacted box for that fragment.
- Drop the clean selected counterpart unless it has a distinct tissue geometry, boundary/truncation condition, or foreground fragment not present in the artifacted box.
- Do not drop boxes that cover genuinely different tissue fragments.

Return only valid JSON with this schema:
{
  "needs_revision": true,
  "revised_selected_box_ids": [1],
  "artifacted_dominates_clean_pairs": [
    {
      "artifacted_box_id": 1,
      "clean_box_id": 2,
      "drop_clean_box": true,
      "reason": "short visual reason"
    }
  ],
  "keep_original_reason": "brief reason if no revision is needed",
  "confidence": "low|medium|high"
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def command_report(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return f"unavailable: {exc}"
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if not output:
        output = "<no output>"
    return f"exit_code={result.returncode}\n{output}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def allowed_ids(case: dict[str, Any]) -> list[int]:
    return list(range(1, int(case["bbox_count"]) + 1))


def direct_prompt_for_case(case: dict[str, Any]) -> str:
    base_without_schema = PROMPT.split("Return only valid JSON with this schema:", 1)[0].rstrip()
    return (
        f"{base_without_schema}\n\n"
        f"{ARTIFACT_DOMINANCE_RULE}\n"
        f"Allowed box IDs exactly for this image: {allowed_ids(case)}.\n"
        "Do not use any box ID outside this list.\n\n"
        f"{DIRECT_SCHEMA.rstrip()}"
    )


def verifier_prompt_for_task(case: dict[str, Any], baseline: dict[str, Any], parsed: dict[str, Any]) -> str:
    payload = {
        "allowed_box_ids": allowed_ids(case),
        "previous_selected_box_ids": baseline.get("selected_box_ids", []),
        "previous_selection_justification": baseline.get("selection_justification", ""),
        "previous_redundancy_groups": parsed.get("redundancy_groups", []),
    }
    return f"{VERIFIER_PROMPT.rstrip()}\n\nPrevious selector payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"


def load_cases(input_root: Path) -> dict[str, dict[str, Any]]:
    cases = read_json(input_root / "summary/cases.json")
    if not isinstance(cases, list):
        raise SystemExit(f"Expected list in {input_root / 'summary/cases.json'}")
    return {str(case["case_id"]): case for case in cases}


def baseline_paths(input_root: Path, case_id: str, model: str, effort: str) -> tuple[Path, Path]:
    result_dir = input_root / "results" / case_id / f"{model_slug(model)}_{effort}"
    return result_dir / "result.json", result_dir / "parsed_response.json"


def raw_response_path(output_dir: Path, approach: str, case_id: str) -> Path:
    subdir = "direct_prompt_update" if approach == "direct_prompt_update" else "second_pass_verifier"
    return output_dir / subdir / case_id / "raw_response.txt"


def normalize_id_list(value: Any, max_id: int) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            idx = int(item)
        except Exception:
            continue
        if 1 <= idx <= max_id and idx not in out:
            out.append(idx)
    return sorted(out)


def extract_any_json_payload(text: str) -> tuple[Any, str]:
    stripped = text.strip()
    candidates: list[tuple[str, str]] = []
    if "```" in stripped:
        for part in stripped.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part:
                candidates.append((part, "fenced"))
    candidates.append((stripped, "direct"))
    decoder = json.JSONDecoder()
    for candidate, source in candidates:
        try:
            return json.loads(candidate), source
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        if start >= 0:
            try:
                payload, _ = decoder.raw_decode(candidate[start:])
                return payload, f"{source}_raw_decode"
            except json.JSONDecodeError:
                pass
    return {"raw_text": text}, "parse_error"


def parse_verifier_payload(payload: Any, case: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "needs_revision": False,
            "revised_selected_box_ids": baseline.get("selected_box_ids", []),
            "artifacted_dominates_clean_pairs": [],
            "keep_original_reason": "parse_error",
            "confidence": "low",
            "usable": False,
        }
    revised = normalize_id_list(payload.get("revised_selected_box_ids"), int(case["bbox_count"]))
    if not revised:
        revised = normalize_id_list(baseline.get("selected_box_ids"), int(case["bbox_count"]))
    pairs = payload.get("artifacted_dominates_clean_pairs")
    if not isinstance(pairs, list):
        pairs = []
    return {
        "needs_revision": bool(payload.get("needs_revision", False)),
        "revised_selected_box_ids": revised,
        "artifacted_dominates_clean_pairs": pairs,
        "keep_original_reason": str(payload.get("keep_original_reason", ""))[:2000],
        "confidence": str(payload.get("confidence", ""))[:100],
        "usable": True,
    }


def run_direct(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    case = task["case"]
    out_dir = Path(args.output_dir) / "direct_prompt_update" / case["case_id"]
    prompt = direct_prompt_for_case(case)
    raw, usage, response_model = _chat_with_images(
        model=args.direct_model,
        prompt_text=prompt,
        image_paths=[Path(case["model_input_path"])],
        temperature=args.temperature,
        max_tokens=args.direct_max_tokens,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=args.direct_reasoning_effort,
    )
    write_text(out_dir / "raw_response.txt", raw)
    payload, route = extract_any_json_payload(raw)
    normalized, status = validate_selection(payload, int(case["bbox_count"]))
    write_json(out_dir / "parsed_response.json", normalized or payload)
    selected = sorted((normalized or {}).get("selected_box_ids", []))
    overlay = draw_overview(
        Path(case["thumbnail_path"]),
        case["boxes"],
        out_dir / "selected_overlay.png",
        selected_ids=set(selected),
        title=f"direct prompt | selected {len(selected)}/{case['bbox_count']} | {status}",
    )
    result = {
        "case_id": case["case_id"],
        "approach": "direct_prompt_update",
        "prompt_version": PROMPT_UPDATE_VERSION,
        "model": args.direct_model,
        "reasoning_effort": args.direct_reasoning_effort,
        "parse_route": route,
        "parse_status": status,
        "selected_box_ids": selected,
        "selection_justification": str((normalized or {}).get("selection_justification", "")),
        "artifact_dominance_decisions": payload.get("artifact_dominance_decisions", []) if isinstance(payload, dict) else [],
        "raw_response_path": str(out_dir / "raw_response.txt"),
        "parsed_response_path": str(out_dir / "parsed_response.json"),
        "selected_overlay_path": str(overlay),
        "usage": usage,
        "response_model": response_model,
    }
    write_json(out_dir / "result.json", result)
    return result


def run_verifier(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    case = task["case"]
    baseline = task["baseline"]
    baseline_parsed = task["baseline_parsed"]
    out_dir = Path(args.output_dir) / "second_pass_verifier" / case["case_id"]
    prompt = verifier_prompt_for_task(case, baseline, baseline_parsed)
    raw, usage, response_model = _chat_with_images(
        model=args.verifier_model,
        prompt_text=prompt,
        image_paths=[Path(case["model_input_path"])],
        temperature=args.temperature,
        max_tokens=args.verifier_max_tokens,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=args.verifier_reasoning_effort,
    )
    write_text(out_dir / "raw_response.txt", raw)
    payload, route = extract_any_json_payload(raw)
    parsed = parse_verifier_payload(payload, case, baseline)
    write_json(out_dir / "parsed_response.json", parsed)
    selected = parsed["revised_selected_box_ids"]
    overlay = draw_overview(
        Path(case["thumbnail_path"]),
        case["boxes"],
        out_dir / "revised_overlay.png",
        selected_ids=set(selected),
        title=f"verifier | selected {len(selected)}/{case['bbox_count']} | revise={parsed['needs_revision']}",
    )
    result = {
        "case_id": case["case_id"],
        "approach": "second_pass_verifier",
        "prompt_version": VERIFIER_PROMPT_VERSION,
        "model": args.verifier_model,
        "reasoning_effort": args.verifier_reasoning_effort,
        "parse_route": route,
        "parse_status": "ok" if parsed["usable"] else "parse_error",
        "baseline_selected_box_ids": baseline.get("selected_box_ids", []),
        "selected_box_ids": selected,
        **parsed,
        "raw_response_path": str(out_dir / "raw_response.txt"),
        "parsed_response_path": str(out_dir / "parsed_response.json"),
        "selected_overlay_path": str(overlay),
        "usage": usage,
        "response_model": response_model,
    }
    write_json(out_dir / "result.json", result)
    return result


def reparse_direct(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = task["case"]
    out_dir = Path(args.output_dir) / "direct_prompt_update" / case["case_id"]
    raw_path = out_dir / "raw_response.txt"
    if not raw_path.exists():
        raise SystemExit(f"Missing direct raw response: {raw_path}")
    raw = raw_path.read_text(encoding="utf-8")
    existing_path = out_dir / "result.json"
    existing = read_json(existing_path) if existing_path.exists() else {}
    payload, route = extract_any_json_payload(raw)
    normalized, status = validate_selection(payload, int(case["bbox_count"]))
    write_json(out_dir / "parsed_response.json", normalized or payload)
    selected = sorted((normalized or {}).get("selected_box_ids", []))
    overlay = draw_overview(
        Path(case["thumbnail_path"]),
        case["boxes"],
        out_dir / "selected_overlay.png",
        selected_ids=set(selected),
        title=f"direct prompt | selected {len(selected)}/{case['bbox_count']} | {status}",
    )
    result = {
        "case_id": case["case_id"],
        "approach": "direct_prompt_update",
        "prompt_version": PROMPT_UPDATE_VERSION,
        "model": existing.get("model", args.direct_model),
        "reasoning_effort": existing.get("reasoning_effort", args.direct_reasoning_effort),
        "parse_route": route,
        "parse_status": status,
        "selected_box_ids": selected,
        "selection_justification": str((normalized or {}).get("selection_justification", "")),
        "artifact_dominance_decisions": payload.get("artifact_dominance_decisions", []) if isinstance(payload, dict) else [],
        "raw_response_path": str(raw_path),
        "parsed_response_path": str(out_dir / "parsed_response.json"),
        "selected_overlay_path": str(overlay),
        "usage": existing.get("usage", {}),
        "response_model": existing.get("response_model", ""),
    }
    write_json(out_dir / "result.json", result)
    return result


def reparse_verifier(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = task["case"]
    baseline = task["baseline"]
    out_dir = Path(args.output_dir) / "second_pass_verifier" / case["case_id"]
    raw_path = out_dir / "raw_response.txt"
    if not raw_path.exists():
        raise SystemExit(f"Missing verifier raw response: {raw_path}")
    raw = raw_path.read_text(encoding="utf-8")
    existing_path = out_dir / "result.json"
    existing = read_json(existing_path) if existing_path.exists() else {}
    payload, route = extract_any_json_payload(raw)
    parsed = parse_verifier_payload(payload, case, baseline)
    write_json(out_dir / "parsed_response.json", parsed)
    selected = parsed["revised_selected_box_ids"]
    overlay = draw_overview(
        Path(case["thumbnail_path"]),
        case["boxes"],
        out_dir / "revised_overlay.png",
        selected_ids=set(selected),
        title=f"verifier | selected {len(selected)}/{case['bbox_count']} | revise={parsed['needs_revision']}",
    )
    result = {
        "case_id": case["case_id"],
        "approach": "second_pass_verifier",
        "prompt_version": VERIFIER_PROMPT_VERSION,
        "model": existing.get("model", args.verifier_model),
        "reasoning_effort": existing.get("reasoning_effort", args.verifier_reasoning_effort),
        "parse_route": route,
        "parse_status": "ok" if parsed["usable"] else "parse_error",
        "baseline_selected_box_ids": baseline.get("selected_box_ids", []),
        "selected_box_ids": selected,
        **parsed,
        "raw_response_path": str(raw_path),
        "parsed_response_path": str(out_dir / "parsed_response.json"),
        "selected_overlay_path": str(overlay),
        "usage": existing.get("usage", {}),
        "response_model": existing.get("response_model", ""),
    }
    write_json(out_dir / "result.json", result)
    return result


def wrap_text_lines(text: str, width: int = 115) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(wrap(paragraph, width=width))
    return lines


def draw_text_block(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, *, text_font: Any, max_lines: int = 16) -> int:
    line_h = 29
    lines = wrap_text_lines(text)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    for line in lines:
        draw.text((x, y), line, font=text_font, fill="black")
        y += line_h
    return y


def make_prompt_pages(page_w: int, page_h: int) -> list[Image.Image]:
    title_font = font(40)
    body_font = font(22)
    line_h = 30
    margin_x = 65
    margin_y = 60
    max_y = page_h - margin_y
    prompts = [
        (
            "Direct Prompt Update Template",
            "This page shows the direct selector prompt template. The final allowed-box list is case-specific; this template shows bbox_count=3.",
            direct_prompt_for_case({"bbox_count": 3}),
        ),
        (
            "Second-Pass Verifier Prompt",
            "This page shows the lightweight verifier prompt used after the baseline selector output.",
            VERIFIER_PROMPT,
        ),
    ]
    pages: list[Image.Image] = []
    for title, subtitle, text in prompts:
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = margin_y
        draw.text((margin_x, y), title, font=title_font, fill="black")
        y += 55
        for line in wrap(subtitle, width=132):
            draw.text((margin_x, y), line, font=body_font, fill="black")
            y += line_h
        y += 20
        for raw_line in text.splitlines():
            wrapped = wrap(raw_line, width=136) if raw_line else [""]
            for line in wrapped:
                if y > max_y:
                    pages.append(page)
                    page = Image.new("RGB", (page_w, page_h), "white")
                    draw = ImageDraw.Draw(page)
                    y = margin_y
                    draw.text((margin_x, y), f"{title} continued", font=title_font, fill="black")
                    y += 70
                draw.text((margin_x, y), line, font=body_font, fill="black")
                y += line_h
        pages.append(page)
    return pages


def make_pdf(tasks: list[dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]], output_path: Path) -> None:
    page_w, page_h = 2200, 3000
    title_font = font(42)
    body_font = font(24)
    small_font = font(20)
    pages: list[Image.Image] = make_prompt_pages(page_w, page_h)
    for task in tasks:
        case = task["case"]
        baseline = task["baseline"]
        direct = results[(case["case_id"], "direct_prompt_update")]
        verifier = results[(case["case_id"], "second_pass_verifier")]
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = 45
        draw.text((55, y), f"{case['case_id']} | boxes={case['bbox_count']}", font=title_font, fill="black")
        y += 60
        draw.text((55, y), f"Baseline: {baseline['model'].replace('google/', '')} | {baseline['reasoning_effort']}", font=body_font, fill="black")
        y += 40
        input_img = Image.open(case["model_input_path"]).convert("RGB")
        input_img.thumbnail((1000, 610))
        page.paste(input_img, (55, y))
        x = 1120
        for label, path in (
            (f"baseline selected={baseline.get('selected_box_ids')}", baseline.get("selected_overlay_path")),
            (f"direct prompt selected={direct.get('selected_box_ids')}", direct.get("selected_overlay_path")),
            (f"verifier revised={verifier.get('selected_box_ids')}", verifier.get("selected_overlay_path")),
        ):
            draw.text((x, y), label[:85], font=small_font, fill="black")
            img = Image.open(path).convert("RGB") if path and Path(path).exists() else Image.new("RGB", (470, 300), "#f4f4f4")
            img.thumbnail((470, 300))
            page.paste(img, (x, y + 30))
            y += 360
        text_y = 1240
        draw.text((55, text_y), "Baseline justification", font=body_font, fill="black")
        text_y = draw_text_block(draw, 70, text_y + 36, baseline.get("selection_justification", ""), text_font=small_font, max_lines=12)
        text_y += 25
        draw.text((55, text_y), "Direct prompt update", font=body_font, fill="black")
        direct_text = (
            f"selected={direct.get('selected_box_ids')} status={direct.get('parse_status')}\n"
            f"justification={direct.get('selection_justification')}\n"
            f"artifact decisions={direct.get('artifact_dominance_decisions')}"
        )
        text_y = draw_text_block(draw, 70, text_y + 36, direct_text, text_font=small_font, max_lines=18)
        text_y += 25
        draw.text((55, text_y), "Second-pass verifier", font=body_font, fill="black")
        verifier_text = (
            f"revised_selected={verifier.get('selected_box_ids')} needs_revision={verifier.get('needs_revision')} confidence={verifier.get('confidence')}\n"
            f"pairs={verifier.get('artifacted_dominates_clean_pairs')}\n"
            f"keep_original_reason={verifier.get('keep_original_reason')}"
        )
        draw_text_block(draw, 70, text_y + 36, verifier_text, text_font=small_font, max_lines=18)
        pages.append(page)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_path, "PDF", save_all=True, append_images=pages[1:], resolution=150)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "baseline_selected_box_ids",
        "direct_selected_box_ids",
        "verifier_selected_box_ids",
        "baseline_count",
        "direct_count",
        "verifier_count",
        "direct_changed",
        "verifier_changed",
        "direct_added_vs_baseline_box_ids",
        "direct_dropped_vs_baseline_box_ids",
        "verifier_added_vs_baseline_box_ids",
        "verifier_dropped_vs_baseline_box_ids",
        "direct_equals_verifier",
        "direct_only_box_ids",
        "verifier_only_box_ids",
        "direct_count_minus_verifier_count",
        "direct_parse_status",
        "verifier_parse_status",
        "verifier_needs_revision",
        "verifier_confidence",
        "direct_selection_justification",
        "verifier_pairs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), (list, dict)) else row.get(field, "") for field in fields})


def make_summary_row(task: dict[str, Any], direct: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    case_id = task["case"]["case_id"]
    baseline = task["baseline"]
    baseline_selected = sorted(int(x) for x in baseline.get("selected_box_ids", []))
    direct_selected = sorted(int(x) for x in direct.get("selected_box_ids", []))
    verifier_selected = sorted(int(x) for x in verifier.get("selected_box_ids", []))
    baseline_set = set(baseline_selected)
    direct_set = set(direct_selected)
    verifier_set = set(verifier_selected)
    return {
        "case_id": case_id,
        "baseline_selected_box_ids": baseline_selected,
        "direct_selected_box_ids": direct_selected,
        "verifier_selected_box_ids": verifier_selected,
        "baseline_count": len(baseline_selected),
        "direct_count": len(direct_selected),
        "verifier_count": len(verifier_selected),
        "direct_changed": direct_selected != baseline_selected,
        "verifier_changed": verifier_selected != baseline_selected,
        "direct_added_vs_baseline_box_ids": sorted(direct_set - baseline_set),
        "direct_dropped_vs_baseline_box_ids": sorted(baseline_set - direct_set),
        "verifier_added_vs_baseline_box_ids": sorted(verifier_set - baseline_set),
        "verifier_dropped_vs_baseline_box_ids": sorted(baseline_set - verifier_set),
        "direct_equals_verifier": direct_selected == verifier_selected,
        "direct_only_box_ids": sorted(direct_set - verifier_set),
        "verifier_only_box_ids": sorted(verifier_set - direct_set),
        "direct_count_minus_verifier_count": len(direct_selected) - len(verifier_selected),
        "direct_parse_status": direct.get("parse_status", ""),
        "verifier_parse_status": verifier.get("parse_status", ""),
        "verifier_needs_revision": verifier.get("needs_revision", False),
        "verifier_confidence": verifier.get("confidence", ""),
        "direct_selection_justification": direct.get("selection_justification", ""),
        "verifier_pairs": verifier.get("artifacted_dominates_clean_pairs", []),
    }


def baseline_direct_stub(task: dict[str, Any]) -> dict[str, Any]:
    baseline = task["baseline"]
    return {
        "selected_box_ids": baseline.get("selected_box_ids", []),
        "parse_status": "skipped_baseline_stub",
        "selection_justification": baseline.get("selection_justification", ""),
    }


def count_by_value(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_aggregate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_changed = [row for row in summary_rows if row["direct_changed"]]
    verifier_changed = [row for row in summary_rows if row["verifier_changed"]]
    disagreements = [row for row in summary_rows if not row["direct_equals_verifier"]]
    direct_smaller = [row for row in disagreements if row["direct_count"] < row["verifier_count"]]
    verifier_smaller = [row for row in disagreements if row["verifier_count"] < row["direct_count"]]
    same_count_diff_ids = [
        row for row in disagreements if row["direct_count"] == row["verifier_count"]
    ]
    direct_drop_only = [
        row
        for row in direct_changed
        if row["direct_dropped_vs_baseline_box_ids"] and not row["direct_added_vs_baseline_box_ids"]
    ]
    verifier_drop_only = [
        row
        for row in verifier_changed
        if row["verifier_dropped_vs_baseline_box_ids"] and not row["verifier_added_vs_baseline_box_ids"]
    ]
    direct_replacements = [
        row
        for row in direct_changed
        if row["direct_dropped_vs_baseline_box_ids"] and row["direct_added_vs_baseline_box_ids"]
    ]
    verifier_replacements = [
        row
        for row in verifier_changed
        if row["verifier_dropped_vs_baseline_box_ids"] and row["verifier_added_vs_baseline_box_ids"]
    ]
    direct_added_only = [
        row
        for row in direct_changed
        if row["direct_added_vs_baseline_box_ids"] and not row["direct_dropped_vs_baseline_box_ids"]
    ]
    verifier_added_only = [
        row
        for row in verifier_changed
        if row["verifier_added_vs_baseline_box_ids"] and not row["verifier_dropped_vs_baseline_box_ids"]
    ]
    verifier_pair_cases = [
        row for row in summary_rows if isinstance(row.get("verifier_pairs"), list) and row["verifier_pairs"]
    ]
    direct_schema_warnings = [
        row for row in summary_rows if row.get("direct_parse_status") != "ok"
    ]
    conservative_verifier_counts = [
        row["verifier_count"]
        if row["verifier_dropped_vs_baseline_box_ids"] and not row["verifier_added_vs_baseline_box_ids"]
        else row["baseline_count"]
        for row in summary_rows
    ]
    return {
        "rows": len(summary_rows),
        "direct_parse_status_counts": count_by_value(summary_rows, "direct_parse_status"),
        "verifier_parse_status_counts": count_by_value(summary_rows, "verifier_parse_status"),
        "direct_changed_from_baseline_cases": len(direct_changed),
        "verifier_changed_from_baseline_cases": len(verifier_changed),
        "direct_drop_only_change_cases": len(direct_drop_only),
        "verifier_drop_only_change_cases": len(verifier_drop_only),
        "direct_replacement_change_cases": len(direct_replacements),
        "verifier_replacement_change_cases": len(verifier_replacements),
        "direct_added_only_change_cases": len(direct_added_only),
        "verifier_added_only_change_cases": len(verifier_added_only),
        "direct_equals_verifier_cases": sum(1 for row in summary_rows if row["direct_equals_verifier"]),
        "direct_vs_verifier_disagreement_cases": len(disagreements),
        "direct_selected_total": sum(row["direct_count"] for row in summary_rows),
        "direct_ok_row_selected_total": sum(
            row["direct_count"] for row in summary_rows if row.get("direct_parse_status") == "ok"
        ),
        "verifier_selected_total": sum(row["verifier_count"] for row in summary_rows),
        "baseline_selected_total": sum(row["baseline_count"] for row in summary_rows),
        "conservative_drop_only_verifier_selected_total": sum(conservative_verifier_counts),
        "conservative_drop_only_verifier_changed_cases": len(verifier_drop_only),
        "direct_smaller_than_verifier_cases": len(direct_smaller),
        "verifier_smaller_than_direct_cases": len(verifier_smaller),
        "same_count_different_ids_cases": len(same_count_diff_ids),
        "verifier_revision_cases": sum(1 for row in summary_rows if row["verifier_needs_revision"]),
        "verifier_nonempty_pair_cases": len(verifier_pair_cases),
        "verifier_confidence_counts": count_by_value(summary_rows, "verifier_confidence"),
        "disagreement_cases": [
            {
                "case_id": row["case_id"],
                "baseline": row["baseline_selected_box_ids"],
                "direct": row["direct_selected_box_ids"],
                "verifier": row["verifier_selected_box_ids"],
                "direct_added_vs_baseline": row["direct_added_vs_baseline_box_ids"],
                "direct_dropped_vs_baseline": row["direct_dropped_vs_baseline_box_ids"],
                "verifier_added_vs_baseline": row["verifier_added_vs_baseline_box_ids"],
                "verifier_dropped_vs_baseline": row["verifier_dropped_vs_baseline_box_ids"],
                "direct_only": row["direct_only_box_ids"],
                "verifier_only": row["verifier_only_box_ids"],
                "direct_count_minus_verifier_count": row["direct_count_minus_verifier_count"],
            }
            for row in disagreements
        ],
        "verifier_revision_case_ids": [
            row["case_id"] for row in summary_rows if row["verifier_needs_revision"]
        ],
        "direct_schema_warning_case_ids": [
            row["case_id"] for row in direct_schema_warnings
        ],
        "verifier_replacement_case_ids": [
            row["case_id"] for row in verifier_replacements
        ],
        "verifier_drop_only_case_ids": [
            row["case_id"] for row in verifier_drop_only
        ],
    }


def write_reproduction(args: argparse.Namespace, tasks: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    cases = "\n".join(f"- {task['case']['case_id']}" for task in tasks)
    aggregate = build_aggregate(summary_rows)
    text = f"""Scale500 artifact redundancy handling probe

Created: {utc_now()}
Ticket/context: PER-250 targeted follow-on to bbox diversity selection.
Git commit: {_repo_git_commit()}
Git status at creation:
{indent(command_report(["git", "status", "--short", "--branch"]), "  ")}
DVC status at creation:
{indent(command_report(["dvc", "status"]), "  ")}

Inputs:
- Existing selector root: {Path(args.input_root).resolve()}
- Baseline model/effort: {args.baseline_model} / {args.baseline_reasoning_effort}
- Cases:
{cases}

Approach A:
- Direct prompt update version: {PROMPT_UPDATE_VERSION}
- Model/effort: {args.direct_model} / {args.direct_reasoning_effort}
- Rule: if clean and artifacted selected boxes contain the same main tissue fragment, default to the artifacted representative unless the clean box adds distinct geometry/boundary content.

Approach B:
- Second-pass verifier prompt version: {VERIFIER_PROMPT_VERSION}
- Model/effort: {args.verifier_model} / {args.verifier_reasoning_effort}
- Verifies the existing baseline selection and can revise selected_box_ids.

Runtime:
- API base: {args.api_base}
- Temperature: {args.temperature}
- Direct max tokens: {args.direct_max_tokens}
- Verifier max tokens: {args.verifier_max_tokens}
- Max concurrent calls: {args.max_concurrent}

Outputs:
- Summary CSV: {output_dir.resolve() / 'summary/results.csv'}
- Summary JSONL: {output_dir.resolve() / 'summary/results.jsonl'}
- Aggregate JSON: {output_dir.resolve() / 'summary/aggregate.json'}
- Review PDF with prompt-template pages: {'skipped' if args.skip_pdf else output_dir.resolve() / 'visuals/artifact_redundancy_probe.pdf'}
- Prompt files: {output_dir.resolve() / 'prompts/'}

Aggregate:
{json.dumps(aggregate, indent=2, sort_keys=True)}

Result summary:
{json.dumps(summary_rows, indent=2, sort_keys=True)}
"""
    write_text(output_dir / "reproduction.txt", text)
    write_text(output_dir / "prompts/direct_prompt_template.txt", direct_prompt_for_case({"bbox_count": 3}))
    write_text(output_dir / "prompts/verifier_prompt_template.txt", VERIFIER_PROMPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--all-cases", action="store_true", help="Run all cases in summary/cases.json.")
    parser.add_argument("--case-limit", type=int, default=None, help="Limit the selected case list after offset.")
    parser.add_argument("--case-offset", type=int, default=0, help="Offset into the selected case list.")
    parser.add_argument("--baseline-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--baseline-reasoning-effort", default="high")
    parser.add_argument("--direct-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--direct-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--verifier-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--verifier-reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--direct-max-tokens", type=int, default=6000)
    parser.add_argument("--verifier-max-tokens", type=int, default=2500)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--reparse-existing", action="store_true", help="Reparse saved raw responses without new model calls.")
    parser.add_argument("--reuse-existing", action="store_true", help="Use saved raw responses where present and call the model only for missing responses.")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip the review PDF; useful for large production batches.")
    parser.add_argument("--verifier-only", action="store_true", help="Run only the second-pass verifier; direct columns use the baseline selection as a skipped stub.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_ALL_CASE_OUTPUT_DIR if args.all_cases else DEFAULT_OUTPUT_DIR
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_map = load_cases(Path(args.input_root))
    if args.case:
        target_cases = args.case
    elif args.all_cases:
        target_cases = list(case_map)
    else:
        target_cases = DEFAULT_TARGET_CASES
    if args.case_offset:
        target_cases = target_cases[args.case_offset :]
    if args.case_limit is not None:
        target_cases = target_cases[: args.case_limit]
    tasks: list[dict[str, Any]] = []
    for case_id in target_cases:
        if case_id not in case_map:
            raise SystemExit(f"Unknown case_id in input root: {case_id}")
        result_path, parsed_path = baseline_paths(
            Path(args.input_root),
            case_id,
            args.baseline_model,
            args.baseline_reasoning_effort,
        )
        baseline = read_json(result_path)
        baseline_parsed = read_json(parsed_path)
        tasks.append({"case": case_map[case_id], "baseline": baseline, "baseline_parsed": baseline_parsed})
    write_json(output_dir / "summary/tasks.json", tasks)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    approaches = ["second_pass_verifier"] if args.verifier_only else ["direct_prompt_update", "second_pass_verifier"]
    if args.reparse_existing:
        for task in tasks:
            for approach in approaches:
                result = reparse_direct(task, args) if approach == "direct_prompt_update" else reparse_verifier(task, args)
                results[(task["case"]["case_id"], approach)] = result
                print(json.dumps({"case": task["case"]["case_id"], "approach": approach, "selected": result.get("selected_box_ids"), "status": result.get("parse_status"), "mode": "reparse_existing"}, sort_keys=True), flush=True)
    else:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
        jobs = []
        job_specs = [(approach, task) for approach in approaches for task in tasks]
        for approach, task in job_specs:
            case_id = task["case"]["case_id"]
            if args.reuse_existing and raw_response_path(output_dir, approach, case_id).exists():
                result = reparse_direct(task, args) if approach == "direct_prompt_update" else reparse_verifier(task, args)
                results[(case_id, approach)] = result
                print(json.dumps({"case": case_id, "approach": approach, "selected": result.get("selected_box_ids"), "status": result.get("parse_status"), "mode": "reuse_existing"}, sort_keys=True), flush=True)
            else:
                jobs.append((approach, task))
        with ThreadPoolExecutor(max_workers=max(1, args.max_concurrent)) as pool:
            futures = {}
            for approach, task in jobs:
                fn = run_direct if approach == "direct_prompt_update" else run_verifier
                futures[pool.submit(fn, task, args, args.api_base, api_key)] = (approach, task)
            for future in as_completed(futures):
                approach, task = futures[future]
                result = future.result()
                results[(task["case"]["case_id"], approach)] = result
                print(json.dumps({"case": task["case"]["case_id"], "approach": approach, "selected": result.get("selected_box_ids"), "status": result.get("parse_status")}, sort_keys=True), flush=True)
    summary_rows: list[dict[str, Any]] = []
    for task in tasks:
        case_id = task["case"]["case_id"]
        direct = results.get((case_id, "direct_prompt_update")) or baseline_direct_stub(task)
        verifier = results[(case_id, "second_pass_verifier")]
        summary_rows.append(make_summary_row(task, direct, verifier))
    write_jsonl(output_dir / "summary/results.jsonl", summary_rows)
    write_csv(output_dir / "summary/results.csv", summary_rows)
    write_json(output_dir / "summary/aggregate.json", build_aggregate(summary_rows))
    if not args.skip_pdf:
        make_pdf(tasks, results, output_dir / "visuals/artifact_redundancy_probe.pdf")
    write_reproduction(args, tasks, summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "cases": len(tasks), "rows": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
