#!/usr/bin/env python3
"""Review Stage 4 crop localization quality with Gemini Flash."""

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

from stage1_detection_review_pilot import _chat_with_images, _font, _repo_git_commit, _thumb, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/stage4_inputs/summary/stage4_crop_prompt_packet_candidates.csv"
)
DEFAULT_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage6_localization_quality_review.txt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_localization_quality_case99_plus10_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_INDICES = [23, 31, 49, 70, 74, 80, 84, 85, 95, 99, 100]
PROMPT_VERSION = "stage6_localization_quality_review_2026-05-27"


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> tuple[dict[str, Any], str]:
    cleaned = _strip_fences(text)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, "json"
    return {}, "unparsed"


def _normalize_review(raw: str) -> tuple[dict[str, Any], str]:
    payload, parse_status = _extract_json_object(raw)
    defaults: dict[str, Any] = {
        "localization_quality": "unclear",
        "tissue_present": "unclear",
        "background_excess": "unclear",
        "needs_split": None,
        "needs_contract": None,
        "needs_expand": None,
        "suggested_action": "unclear",
        "reasoning": raw.strip(),
    }
    if not payload:
        return defaults, parse_status
    normalized = {**defaults, **payload}
    for key in ("needs_split", "needs_contract", "needs_expand"):
        value = normalized.get(key)
        if isinstance(value, bool) or value is None:
            continue
        value_s = str(value).strip().lower()
        if value_s in {"yes", "true", "1"}:
            normalized[key] = True
        elif value_s in {"no", "false", "0"}:
            normalized[key] = False
        else:
            normalized[key] = None
    return normalized, parse_status


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _case_groups(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["case_index"]), []).append(row)
    for case_rows in grouped.values():
        case_rows.sort(key=lambda row: int(row["candidate_order"]))
    return dict(sorted(grouped.items()))


def _build_tasks(args: argparse.Namespace, prompt: str) -> list[dict[str, Any]]:
    rows = _read_csv(args.candidates)
    wanted = {int(v) for v in (args.indices or DEFAULT_INDICES)}
    rows = [row for row in rows if int(row["case_index"]) in wanted]
    tasks: list[dict[str, Any]] = []
    for row in rows:
        overlay = Path(row["selected_overlay_path"])
        crop = Path(row["crop_path"])
        if not overlay.exists():
            raise SystemExit(f"Missing selected overlay image: {overlay}")
        if not crop.exists():
            raise SystemExit(f"Missing crop image: {crop}")
        task = {
            "task_id": f"stage6_localization_{int(row['case_index']):03d}_{int(row['candidate_order']):02d}",
            "case_index": int(row["case_index"]),
            "case_display": row["case_display"],
            "bbox_source": row["bbox_source"],
            "candidate_order": int(row["candidate_order"]),
            "candidate_id": row["candidate_id"],
            "label": row["label"],
            "crop_path": row["crop_path"],
            "selected_overlay_path": row["selected_overlay_path"],
            "metadata_path": row["metadata_path"],
            "source_bbox_in_crop": row.get("source_bbox_in_crop", ""),
            "prompt": prompt,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "created_at": _timestamp(),
        }
        tasks.append(task)
    tasks.sort(key=lambda row: (int(row["case_index"]), int(row["candidate_order"])))
    return tasks


def _run_one(
    task: dict[str, Any],
    effort: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    record = {
        key: task[key]
        for key in (
            "task_id",
            "case_index",
            "case_display",
            "bbox_source",
            "candidate_order",
            "candidate_id",
            "label",
            "crop_path",
            "selected_overlay_path",
            "metadata_path",
            "source_bbox_in_crop",
            "prompt_version",
            "model",
            "created_at",
        )
    }
    record.update(
        {
            "reasoning_effort": effort,
            "raw_response": "",
            "parse_status": "",
            "error": "",
            "usage": {},
            "response_model": "",
            "localization_quality": "unclear",
            "tissue_present": "unclear",
            "background_excess": "unclear",
            "needs_split": None,
            "needs_contract": None,
            "needs_expand": None,
            "suggested_action": "unclear",
            "reasoning": "",
        }
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=task["prompt"],
            image_paths=[Path(task["selected_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=effort,
        )
        parsed, parse_status = _normalize_review(raw)
        record.update(parsed)
        record.update(
            {
                "raw_response": raw,
                "parse_status": parse_status,
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_height: int = 23,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
    return y


def _draw_effort_cover(results: list[dict[str, Any]], prompt: str, effort: str, args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(42)
    header = _font(28)
    body = _font(20)
    small = _font(17)
    y = 55
    draw.text((65, y), "Stage 6 Localization Quality Review", font=title, fill="black")
    y += 58
    draw.text((65, y), f"model={args.model} | thinking={effort} | crops={len(results)}", font=body, fill="#111111")
    y += 36
    draw.text((65, y), f"localization_quality={_counts(results, 'localization_quality')}", font=small, fill="#111111")
    y += 30
    draw.text((65, y), f"suggested_action={_counts(results, 'suggested_action')}", font=small, fill="#111111")
    y += 42
    draw.text((65, y), "Prompt", font=header, fill="black")
    y += 36
    y = _draw_wrapped(draw, (85, y), prompt, 150, small)
    y += 30
    draw.text((65, y), "Cases", font=header, fill="black")
    y += 36
    for case_index, rows in _case_groups(results).items():
        first = rows[0]
        y = _draw_wrapped(
            draw,
            (85, y),
            f"{case_index}/100 | {first['case_display']} | n={len(rows)} | action={_counts(rows, 'suggested_action')}",
            150,
            small,
        )
    return page


def _draw_effort_page(row: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(34)
    header = _font(25)
    body = _font(18)
    small = _font(16)
    y = 45
    draw.text(
        (55, y),
        f"{int(row['case_index']):03d} candidate {int(row['candidate_order']):02d} | {row['reasoning_effort']} | {row['suggested_action']}",
        font=title,
        fill="black",
    )
    y += 48
    y = _draw_wrapped(draw, (55, y), row["case_display"], 150, body)
    y += 14
    details = (
        f"quality={row['localization_quality']} | tissue={row['tissue_present']} | "
        f"background={row['background_excess']} | split={row['needs_split']} | "
        f"contract={row['needs_contract']} | expand={row['needs_expand']} | "
        f"parse={row['parse_status']} | error={row['error']}"
    )
    y = _draw_wrapped(draw, (55, y), details, 170, body)
    y += 18
    draw.text((55, y), "Raw padded crop", font=header, fill="black")
    draw.text((1240, y), "Overlay sent to VLM", font=header, fill="black")
    y += 38
    raw = _thumb(Path(row["crop_path"]), (1080, 1080))
    overlay = _thumb(Path(row["selected_overlay_path"]), (1080, 1080))
    page.paste(raw, (55, y))
    page.paste(overlay, (1240, y))
    y += max(raw.size[1], overlay.size[1]) + 40
    draw.text((55, y), "Parsed reasoning", font=header, fill="black")
    y += 30
    y = _draw_wrapped(draw, (75, y), row.get("reasoning", ""), 170, small)
    y += 16
    draw.text((55, y), "Raw model output", font=header, fill="black")
    y += 30
    _draw_wrapped(draw, (75, y), row.get("raw_response", ""), 170, small)
    return page


def _write_effort_pdf(output_root: Path, results: list[dict[str, Any]], prompt: str, effort: str, args: argparse.Namespace) -> Path:
    pages = [_draw_effort_cover(results, prompt, effort, args)]
    pages.extend(_draw_effort_page(row) for row in results)
    pdf_path = output_root / "visuals" / f"stage6_localization_quality_{effort}_thinking.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _draw_comparison_cover(all_results: dict[str, list[dict[str, Any]]], prompt: str, args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(42)
    header = _font(28)
    body = _font(20)
    small = _font(17)
    y = 55
    draw.text((65, y), "Stage 6 Localization Quality Low vs High Thinking", font=title, fill="black")
    y += 58
    efforts = ", ".join(all_results)
    total = len(next(iter(all_results.values()))) if all_results else 0
    draw.text((65, y), f"model={args.model} | efforts={efforts} | crops={total}", font=body, fill="#111111")
    y += 42
    for effort, rows in all_results.items():
        draw.text((85, y), f"{effort}: quality={_counts(rows, 'localization_quality')} action={_counts(rows, 'suggested_action')}", font=small, fill="#111111")
        y += 28
    y += 26
    draw.text((65, y), "Prompt", font=header, fill="black")
    y += 36
    y = _draw_wrapped(draw, (85, y), prompt, 150, small)
    y += 30
    draw.text((65, y), "Selected cases", font=header, fill="black")
    y += 36
    first_rows = next(iter(all_results.values())) if all_results else []
    for case_index, rows in _case_groups(first_rows).items():
        first = rows[0]
        y = _draw_wrapped(draw, (85, y), f"{case_index}/100 | {first['case_display']} | n={len(rows)}", 150, small)
    return page


def _draw_comparison_page(task: dict[str, Any], low: dict[str, Any] | None, high: dict[str, Any] | None) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(32)
    header = _font(24)
    body = _font(17)
    small = _font(15)
    y = 42
    draw.text(
        (50, y),
        f"{int(task['case_index']):03d} candidate {int(task['candidate_order']):02d} | low vs high localization review",
        font=title,
        fill="black",
    )
    y += 45
    y = _draw_wrapped(draw, (50, y), task["case_display"], 160, body, line_height=22)
    y += 14
    overlay = _thumb(Path(task["selected_overlay_path"]), (1120, 950))
    crop = _thumb(Path(task["crop_path"]), (1120, 950))
    draw.text((50, y), "Raw padded crop", font=header, fill="black")
    draw.text((1240, y), "Overlay sent to VLM", font=header, fill="black")
    y += 34
    page.paste(crop, (50, y))
    page.paste(overlay, (1240, y))
    y += max(crop.size[1], overlay.size[1]) + 34
    x_positions = {"low": 50, "high": 1240}
    for effort, row in (("low", low), ("high", high)):
        x = x_positions[effort]
        draw.text((x, y), f"{effort} thinking", font=header, fill="black")
        if row is None:
            _draw_wrapped(draw, (x, y + 34), "missing result", 80, body)
            continue
        parsed = (
            f"quality={row['localization_quality']} | action={row['suggested_action']} | "
            f"background={row['background_excess']} | split={row['needs_split']} | "
            f"contract={row['needs_contract']} | expand={row['needs_expand']}"
        )
        yy = _draw_wrapped(draw, (x, y + 34), parsed, 86, body, line_height=22)
        yy += 12
        yy = _draw_wrapped(draw, (x, yy), row.get("reasoning", ""), 86, small, line_height=20)
        yy += 10
        _draw_wrapped(draw, (x, yy), row.get("raw_response", ""), 86, small, line_height=20)
    return page


def _write_comparison_pdf(
    output_root: Path,
    tasks: list[dict[str, Any]],
    all_results: dict[str, list[dict[str, Any]]],
    prompt: str,
    args: argparse.Namespace,
) -> Path | None:
    if not {"low", "high"}.issubset(all_results):
        return None
    by_effort = {
        effort: {(int(row["case_index"]), int(row["candidate_order"])): row for row in rows}
        for effort, rows in all_results.items()
    }
    pages = [_draw_comparison_cover(all_results, prompt, args)]
    for task in tasks:
        key = (int(task["case_index"]), int(task["candidate_order"]))
        pages.append(_draw_comparison_page(task, by_effort["low"].get(key), by_effort["high"].get(key)))
    pdf_path = output_root / "visuals" / "stage6_localization_quality_low_vs_high.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_effort_outputs(
    output_root: Path,
    effort: str,
    results: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    effort_root = output_root / f"{effort}_thinking"
    results_path = effort_root / "reviews" / f"stage6_localization_quality_{effort}_thinking.jsonl"
    _write_jsonl(results_path, results)
    csv_rows = []
    for row in results:
        csv_rows.append(
            {
                "case_index": row["case_index"],
                "case_display": row["case_display"],
                "candidate_order": row["candidate_order"],
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "bbox_source": row["bbox_source"],
                "reasoning_effort": row["reasoning_effort"],
                "localization_quality": row["localization_quality"],
                "tissue_present": row["tissue_present"],
                "background_excess": row["background_excess"],
                "needs_split": row["needs_split"],
                "needs_contract": row["needs_contract"],
                "needs_expand": row["needs_expand"],
                "suggested_action": row["suggested_action"],
                "parse_status": row["parse_status"],
                "error": row["error"],
                "reasoning": row["reasoning"],
                "raw_response": row["raw_response"],
                "selected_overlay_path": row["selected_overlay_path"],
                "crop_path": row["crop_path"],
                "metadata_path": row["metadata_path"],
            }
        )
    csv_path = effort_root / "summary" / f"stage6_localization_quality_{effort}_thinking.csv"
    _write_csv(
        csv_path,
        csv_rows,
        [
            "case_index",
            "case_display",
            "candidate_order",
            "candidate_id",
            "label",
            "bbox_source",
            "reasoning_effort",
            "localization_quality",
            "tissue_present",
            "background_excess",
            "needs_split",
            "needs_contract",
            "needs_expand",
            "suggested_action",
            "parse_status",
            "error",
            "reasoning",
            "raw_response",
            "selected_overlay_path",
            "crop_path",
            "metadata_path",
        ],
    )
    pdf_path = _write_effort_pdf(effort_root, results, prompt, effort, args)
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(args.prompt.resolve()),
        "candidates_csv": str(args.candidates.resolve()),
        "output_root": str(effort_root.resolve()),
        "model": args.model,
        "reasoning_effort": effort,
        "max_concurrent": args.max_concurrent,
        "max_tokens": args.max_tokens,
        "candidates": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "localization_quality_counts": _counts(results, "localization_quality"),
        "suggested_action_counts": _counts(results, "suggested_action"),
        "background_excess_counts": _counts(results, "background_excess"),
        "parse_status_counts": _counts(results, "parse_status"),
        "known_usage_cost_if_reported": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in results),
        "pdf": str(pdf_path.resolve()),
        "results_jsonl": str(results_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
    }
    summary_path = effort_root / "summary" / f"stage6_localization_quality_{effort}_thinking_summary.json"
    _write_json(summary_path, summary)
    return summary


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    prompt: str,
    tasks: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    comparison_pdf: Path | None,
) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage6_localization_quality_review.py",
            "--candidates",
            str(args.candidates.resolve()),
            "--prompt",
            str(args.prompt.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--model",
            args.model,
            "--reasoning-efforts",
            *args.reasoning_efforts,
            "--indices",
            *(str(v) for v in (args.indices or DEFAULT_INDICES)),
            "--max-concurrent",
            str(args.max_concurrent),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
    )
    if args.reuse_existing:
        command += " --reuse-existing"
    selected_cases = sorted({int(task["case_index"]) for task in tasks})
    output_lines = []
    for effort, summary in summaries.items():
        output_lines.extend(
            [
                f"- {effort} PDF: {summary['pdf']}",
                f"- {effort} summary JSON: {Path(summary['output_root']) / 'summary' / f'stage6_localization_quality_{effort}_thinking_summary.json'}",
                f"- {effort} results JSONL: {summary['results_jsonl']}",
            ]
        )
    if comparison_pdf:
        output_lines.append(f"- Low-vs-high comparison PDF: {comparison_pdf.resolve()}")
    text = f"""\
Stage 6 localization-quality review
===================================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Reasoning efforts: {', '.join(args.reasoning_efforts)}
Backend: OpenRouter-compatible chat completions
Reuse existing model outputs: {args.reuse_existing}

Objective:
Review whether Stage 4 highlighted candidate boxes are well localized around
tissue, too loose, too tight, off-target, or should be split/reduced.

Selected cases:
{', '.join(str(v) for v in selected_cases)}

Prompt:
{prompt}

Command:
{command}

Inputs:
- Candidate manifest: {args.candidates.resolve()}
- Prompt file: {args.prompt.resolve()}

Outputs:
{chr(10).join(output_lines)}

To regenerate the paid model outputs from the source crop overlays, run the
same command without `--reuse-existing`.
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text().strip()
    tasks = _build_tasks(args, prompt)
    _write_jsonl(output_root / "tasks/stage6_localization_quality_tasks.jsonl", tasks)
    task_csv_rows = [{k: v for k, v in task.items() if k != "prompt"} for task in tasks]
    _write_csv(
        output_root / "summary/stage6_localization_quality_candidates.csv",
        task_csv_rows,
        [
            "task_id",
            "case_index",
            "case_display",
            "bbox_source",
            "candidate_order",
            "candidate_id",
            "label",
            "crop_path",
            "selected_overlay_path",
            "metadata_path",
            "source_bbox_in_crop",
            "prompt_version",
            "model",
            "created_at",
        ],
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks), "cases": sorted({t["case_index"] for t in tasks})}, indent=2))
        return 0

    base_url = api_key = ""
    if not args.reuse_existing:
        base_url, api_key = _api_settings(args)
    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for effort in args.reasoning_efforts:
        results_path = output_root / f"{effort}_thinking/reviews/stage6_localization_quality_{effort}_thinking.jsonl"
        if args.reuse_existing and results_path.exists():
            results = _read_jsonl(results_path)
        elif args.reuse_existing:
            raise SystemExit(f"Missing existing results for --reuse-existing: {results_path}")
        elif args.max_concurrent > 1:
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
                futures = [pool.submit(_run_one, task, effort, args, base_url, api_key) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            results = [_run_one(task, effort, args, base_url, api_key) for task in tasks]
        results.sort(key=lambda row: (int(row["case_index"]), int(row["candidate_order"])))
        all_results[effort] = results
        summaries[effort] = _write_effort_outputs(output_root, effort, results, prompt, args)
    comparison_pdf = _write_comparison_pdf(output_root, tasks, all_results, prompt, args)
    root_summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(args.prompt.resolve()),
        "candidates_csv": str(args.candidates.resolve()),
        "output_root": str(output_root.resolve()),
        "selected_cases": sorted({int(task["case_index"]) for task in tasks}),
        "candidate_count": len(tasks),
        "model": args.model,
        "reasoning_efforts": args.reasoning_efforts,
        "comparison_pdf": str(comparison_pdf.resolve()) if comparison_pdf else "",
        "effort_summaries": summaries,
    }
    _write_json(output_root / "summary/stage6_localization_quality_summary.json", root_summary)
    _write_reproduction(output_root, args, prompt, tasks, summaries, comparison_pdf)
    print(json.dumps(root_summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-efforts", nargs="+", default=["low", "high"], choices=["low", "medium", "high"])
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
