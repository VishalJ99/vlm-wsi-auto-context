#!/usr/bin/env python3
"""Run Stage 3 minimal feedback redetection on Stage 2b-positive cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from stage1_detection_review_pilot import (
    _chat_with_images,
    _draw_redetect_overlay,
    _extract_json_payload,
    _font,
    _normalised_detection_items,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1/summary/high_recall_stage1_cases.csv"
)
STAGE2A_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "reviews/edge_review_results.jsonl"
)
STAGE2B_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage2b_nonminor_two_pass_gemini_flash_low_v1"
    / "reviews/stage2b_two_pass_results.jsonl"
)
STAGE1_PROMPT = REPO_ROOT / "prompts/stage1_high_recall_potential_tissue_candidates.txt"
WRAPPER_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage3_refinement_minimal_wrapper.txt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage3_refinement_minimal_feedback_gemini_flash_high_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
PROMPT_VERSION = "stage3_refinement_minimal_feedback_2026-05-24"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, --api-key-stdin, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return args.api_base or "https://openrouter.ai/api/v1", api_key


def _build_prompt(stage1_prompt: str, wrapper_prompt: str, reviewer_feedback: str) -> str:
    return wrapper_prompt.format(
        reviewer_feedback=reviewer_feedback.strip(),
        stage1_task_prompt=stage1_prompt.strip(),
    )


def _select_case_indices(stage2b_rows: list[dict[str, Any]], explicit_indices: list[int] | None) -> list[int]:
    if explicit_indices:
        return explicit_indices
    selected = [
        int(row["case_index"])
        for row in stage2b_rows
        if _boolish(row.get("final_non_minor_detection_failure"))
    ]
    return sorted(selected)


def _load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    stage1_rows = {int(row["case_index"]): row for row in _read_csv(args.stage1_cases)}
    stage2a_rows = {int(row["case_index"]): row for row in _read_jsonl(args.stage2a_results)}
    stage2b_rows = _read_jsonl(args.stage2b_results)
    stage2b_by_case = {int(row["case_index"]): row for row in stage2b_rows}
    selected = _select_case_indices(stage2b_rows, args.indices)
    missing = [
        idx
        for idx in selected
        if idx not in stage1_rows or idx not in stage2a_rows or idx not in stage2b_by_case
    ]
    if missing:
        raise SystemExit(f"Selected cases missing from one or more inputs: {missing}")

    stage1_prompt = args.stage1_prompt.read_text()
    wrapper_prompt = args.wrapper_prompt.read_text()
    tasks: list[dict[str, Any]] = []
    for idx in selected:
        stage1 = stage1_rows[idx]
        stage2a = stage2a_rows[idx]
        stage2b = stage2b_by_case[idx]
        thumbnail_path = Path(stage2a.get("thumbnail_path") or stage1["thumbnail_path"])
        overlay_path = Path(stage2a.get("review_overlay_path") or stage1["raw_overlay_path"])
        if not thumbnail_path.exists():
            raise SystemExit(f"Thumbnail missing for case {idx}: {thumbnail_path}")
        if not overlay_path.exists():
            raise SystemExit(f"Raw overlay missing for case {idx}: {overlay_path}")
        prompt = _build_prompt(stage1_prompt, wrapper_prompt, stage2a.get("raw_response", ""))
        case_slug = _safe_slug(f"{idx:03d}_{stage1['case_display']}")
        tasks.append(
            {
                "task_id": f"stage3_refinement_{idx:03d}",
                "case_index": idx,
                "case_display": stage1["case_display"],
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "thumbnail_path": str(thumbnail_path),
                "stage1_raw_overlay_path": str(overlay_path),
                "stage1_raw_count": int(stage1.get("raw_rot0_count") or 0),
                "stage1_final_count": int(stage1.get("final_count") or 0),
                "stage1_raw_response_status": stage1.get("raw_response_status", ""),
                "stage2a_review_text": stage2a.get("raw_response", ""),
                "stage2a_prompt_version": stage2a.get("prompt_version", ""),
                "stage2a_reasoning_effort": stage2a.get("reasoning_effort", ""),
                "stage2b_final_non_minor_detection_failure": stage2b.get("final_non_minor_detection_failure"),
                "stage2b_final_justification": stage2b.get("final_justification", ""),
                "prompt": prompt,
                "case_slug": case_slug,
                "created_at": _timestamp(),
            }
        )
    return tasks, [stage2b_by_case[idx] for idx in selected], stage1_prompt, wrapper_prompt


def _run_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        key: task[key]
        for key in (
            "task_id",
            "case_index",
            "case_display",
            "prompt_version",
            "model",
            "reasoning_effort",
            "thumbnail_path",
            "stage1_raw_overlay_path",
            "stage1_raw_count",
            "stage1_final_count",
            "stage1_raw_response_status",
            "stage2a_review_text",
            "stage2a_prompt_version",
            "stage2a_reasoning_effort",
            "stage2b_final_non_minor_detection_failure",
            "stage2b_final_justification",
            "created_at",
        )
    }
    record.update(
        {
            "error": "",
            "raw_response": "",
            "parsed_response": {},
            "detections": [],
            "stage3_detection_count": 0,
            "stage3_overlay_path": "",
            "usage": {},
            "response_model": "",
        }
    )
    try:
        thumbnail_path = Path(task["thumbnail_path"])
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=task["prompt"],
            image_paths=[thumbnail_path, Path(task["stage1_raw_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        parsed = _extract_json_payload(raw)
        detections = _normalised_detection_items(parsed, thumbnail_size)
        overlay_path = args.output_root / "overlays" / f"{task['case_slug']}_stage3_refinement_overlay.png"
        _draw_redetect_overlay(thumbnail_path, detections, overlay_path)
        record.update(
            {
                "raw_response": raw,
                "parsed_response": parsed,
                "detections": detections,
                "stage3_detection_count": len(detections),
                "stage3_overlay_path": str(overlay_path),
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines() or [""]:
        if not raw.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: str = "#111111",
    max_y: int = 2680,
    gap: int = 6,
) -> int:
    x, y = xy
    line_height = int(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + gap
    for line in _wrap(text, width):
        if y + line_height > max_y:
            draw.text((x, y), "... [truncated; see JSONL]", font=font, fill="#777777")
            return y + line_height
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _excerpt(value: Any, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated; see JSONL]"


def _make_cover(args: argparse.Namespace, results: list[dict[str, Any]]) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(34)
    header_font = _font(24)
    body_font = _font(20)
    y = 55
    draw.text((55, y), "Stage 3 Minimal Feedback Redetection", font=title_font, fill="black")
    y += 62
    errors = sum(bool(row.get("error")) for row in results)
    total_cost = sum(float(((row.get("usage") or {}).get("cost") or 0.0)) for row in results)
    summary = (
        f"Created: {_timestamp()}\n"
        f"Git commit: {_repo_git_commit()}\n"
        f"Model: {args.model}\n"
        f"Reasoning effort: {args.reasoning_effort}\n"
        f"Cases: {len(results)} Stage 2b-positive raw-overlay pilot cases\n"
        f"Errors: {errors}\n"
        f"Approximate reported OpenRouter cost: ${total_cost:.6f}\n\n"
        "Configuration: original thumbnail + previous raw tissue-candidate overlay + raw Stage 2a reviewer feedback. "
        "The detector task inside the wrapper is the unchanged Stage 1 high-recall prompt."
    )
    y = _draw_wrapped(draw, (75, y), summary, body_font, 150)
    y += 30
    draw.text((55, y), "Counts", font=header_font, fill="black")
    y += 34
    for row in results:
        line = (
            f"{row['case_index']:03d}: stage1_raw={row.get('stage1_raw_count')} -> "
            f"stage3={row.get('stage3_detection_count')} | {row['case_display']}"
        )
        y = _draw_wrapped(draw, (75, y), line, body_font, 150, max_y=2580)
    y += 30
    draw.text((55, y), "Prompt Files", font=header_font, fill="black")
    y += 34
    prompt_files = f"Wrapper: {args.wrapper_prompt.resolve()}\nStage 1 task: {args.stage1_prompt.resolve()}"
    _draw_wrapped(draw, (75, y), prompt_files, body_font, 150)
    return page


def _make_case_page(row: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(32)
    header_font = _font(23)
    body_font = _font(19)
    small_font = _font(16)
    y = 36
    draw.text((45, y), row["case_display"], font=title_font, fill="black")
    y += 52
    meta = (
        f"Stage 1 raw boxes={row.get('stage1_raw_count')} | final boxes={row.get('stage1_final_count')} | "
        f"Stage 2b final={row.get('stage2b_final_non_minor_detection_failure')} | "
        f"Stage 3 boxes={row.get('stage3_detection_count')} | error={row.get('error') or ''}"
    )
    y = _draw_wrapped(draw, (45, y), meta, body_font, 155)
    y += 18
    image_specs = [
        ("Source thumbnail", row.get("thumbnail_path")),
        ("Stage 1 raw overlay", row.get("stage1_raw_overlay_path")),
        ("Stage 3 feedback redetection", row.get("stage3_overlay_path")),
    ]
    for x, label, path in ((45, *image_specs[0]), (760, *image_specs[1]), (1475, *image_specs[2])):
        draw.text((x, y), label, font=header_font, fill="black")
        page.paste(_thumb(Path(path) if path else Path("__missing__"), (660, 430)), (x, y + 34))
    y += 500
    draw.text((45, y), "Stage 2a Reviewer Feedback Supplied To Detector", font=header_font, fill="black")
    y += 34
    y = _draw_wrapped(draw, (65, y), _excerpt(row.get("stage2a_review_text"), 1400), small_font, 185)
    y += 20
    draw.text((45, y), "Stage 2b Final Justification", font=header_font, fill="black")
    y += 34
    y = _draw_wrapped(draw, (65, y), _excerpt(row.get("stage2b_final_justification"), 800), small_font, 185)
    y += 20
    draw.text((45, y), "Stage 3 Raw Response", font=header_font, fill="black")
    y += 34
    y = _draw_wrapped(draw, (65, y), _excerpt(row.get("raw_response"), 1300), small_font, 185)
    y += 20
    draw.text((45, y), "Stage 3 Parsed Detections", font=header_font, fill="black")
    y += 34
    parsed_text = json.dumps(row.get("detections", []), sort_keys=True)
    _draw_wrapped(draw, (65, y), _excerpt(parsed_text, 1800), small_font, 185)
    return page


def write_pdf(args: argparse.Namespace, results: list[dict[str, Any]]) -> Path:
    pdf_path = args.output_root / "visuals" / "stage3_refinement_minimal_feedback_cases.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages = [_make_cover(args, results)] + [_make_case_page(row) for row in results]
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=150)
    return pdf_path


def summarize(args: argparse.Namespace, results: list[dict[str, Any]], pdf_path: Path) -> None:
    summary_rows = [
        {
            "case_index": row["case_index"],
            "case_display": row["case_display"],
            "stage1_raw_count": row.get("stage1_raw_count"),
            "stage1_final_count": row.get("stage1_final_count"),
            "stage2b_final_non_minor_detection_failure": row.get("stage2b_final_non_minor_detection_failure"),
            "stage3_detection_count": row.get("stage3_detection_count"),
            "error": row.get("error", ""),
            "stage3_overlay_path": row.get("stage3_overlay_path", ""),
            "response_model": row.get("response_model", ""),
            "raw_response_excerpt": _excerpt(row.get("raw_response"), 300).replace("\n", " "),
        }
        for row in results
    ]
    _write_csv(
        args.output_root / "summary" / "stage3_refinement_summary.csv",
        summary_rows,
        [
            "case_index",
            "case_display",
            "stage1_raw_count",
            "stage1_final_count",
            "stage2b_final_non_minor_detection_failure",
            "stage3_detection_count",
            "error",
            "stage3_overlay_path",
            "response_model",
            "raw_response_excerpt",
        ],
    )
    total_cost = sum(float(((row.get("usage") or {}).get("cost") or 0.0)) for row in results)
    _write_json(
        args.output_root / "summary" / "stage3_refinement_summary.json",
        {
            "created_at": _timestamp(),
            "git_commit": _repo_git_commit(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "case_count": len(results),
            "case_indices": [row["case_index"] for row in results],
            "error_count": sum(bool(row.get("error")) for row in results),
            "stage3_detection_counts": {
                str(row["case_index"]): row.get("stage3_detection_count") for row in results
            },
            "total_reported_cost": total_cost,
            "pdf": str(pdf_path),
            "results_jsonl": str(args.output_root / "reviews/stage3_refinement_results.jsonl"),
            "tasks_jsonl": str(args.output_root / "tasks/stage3_refinement_tasks.jsonl"),
        },
    )


def write_reproduction(args: argparse.Namespace, pdf_path: Path) -> None:
    quoted = [
        "python",
        "scripts/stage1_refinement_minimal_feedback.py",
        "--output-root",
        str(args.output_root),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--api-key-stdin",
    ]
    text = f"""\
Stage 3 minimal feedback redetection
====================================

Created: {_timestamp()}
Git commit at run time: {_repo_git_commit()}
Ticket: PER-207

Purpose
-------
Run the minimal Stage 3 refinement/redetection configuration on raw-overlay
Stage 2b-positive pilot cases. The model sees the original thumbnail, the
previous raw tissue-candidate detection overlay, and the raw Stage 2a reviewer
feedback, then reruns the unchanged Stage 1 high-recall detector task.

Inputs
------
Stage 1 cases CSV:
{args.stage1_cases.resolve()}

Stage 2a raw-overlay reviewer JSONL:
{args.stage2a_results.resolve()}

Stage 2b two-pass router JSONL:
{args.stage2b_results.resolve()}

Wrapper prompt:
{args.wrapper_prompt.resolve()}

Stage 1 task prompt:
{args.stage1_prompt.resolve()}

Model settings
--------------
Model: {args.model}
Reasoning effort: {args.reasoning_effort}
Temperature: {args.temperature}
Max tokens: {args.max_tokens}
Max concurrent calls: {args.max_concurrent}

Command
-------
Run from repository root and provide an OpenRouter/OpenAI-compatible API key on
stdin when prompted by the shell:

{" ".join(shlex.quote(part) for part in quoted)}

Outputs
-------
Tasks JSONL:
{(args.output_root / "tasks/stage3_refinement_tasks.jsonl").resolve()}

Results JSONL:
{(args.output_root / "reviews/stage3_refinement_results.jsonl").resolve()}

Summary CSV:
{(args.output_root / "summary/stage3_refinement_summary.csv").resolve()}

Summary JSON:
{(args.output_root / "summary/stage3_refinement_summary.json").resolve()}

Visual PDF:
{pdf_path.resolve()}
"""
    (args.output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks, _stage2b_rows, _stage1_prompt, _wrapper_prompt = _load_inputs(args)
    _write_jsonl(args.output_root / "tasks/stage3_refinement_tasks.jsonl", tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "case_indices": [task["case_index"] for task in tasks],
                    "tasks_jsonl": str(args.output_root / "tasks/stage3_refinement_tasks.jsonl"),
                },
                indent=2,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    results: list[dict[str, Any]] = []
    if args.max_concurrent > 1:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(_run_one, task, args, base_url, api_key) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [_run_one(task, args, base_url, api_key) for task in tasks]
    results.sort(key=lambda row: int(row["case_index"]))
    results_path = args.output_root / "reviews/stage3_refinement_results.jsonl"
    _write_jsonl(results_path, results)
    pdf_path = write_pdf(args, results)
    summarize(args, results, pdf_path)
    write_reproduction(args, pdf_path)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "case_indices": [row["case_index"] for row in results],
                "results_jsonl": str(results_path),
                "pdf": str(pdf_path),
                "summary_json": str(args.output_root / "summary/stage3_refinement_summary.json"),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--stage2a-results", type=Path, default=STAGE2A_RESULTS)
    parser.add_argument("--stage2b-results", type=Path, default=STAGE2B_RESULTS)
    parser.add_argument("--stage1-prompt", type=Path, default=STAGE1_PROMPT)
    parser.add_argument("--wrapper-prompt", type=Path, default=WRAPPER_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-concurrent", type=int, default=7)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
