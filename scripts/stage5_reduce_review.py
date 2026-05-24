#!/usr/bin/env python3
"""Run Stage 5 reduce/split review on Stage 4 crop candidates."""

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
    / "stage4_crop_prompt_packet_v1/summary/stage4_crop_prompt_packet_candidates.csv"
)
DEFAULT_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage5a_crop_split_review.txt"
DEFAULT_OUTPUT_PARENT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage5_reduce_review_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
PROMPT_VERSION = "stage5_reduce_review_2026-05-24"


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


def _parse_yes_no(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^```(?:text|json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    if re.match(r"^(yes|y)\b", cleaned):
        return "yes"
    if re.match(r"^(no|n)\b", cleaned):
        return "no"
    match = re.search(r"\b(yes|no)\b", cleaned)
    return match.group(1) if match else "unknown"


def _build_tasks(args: argparse.Namespace, prompt: str) -> list[dict[str, Any]]:
    rows = _read_csv(args.candidates)
    if args.indices:
        wanted = {int(v) for v in args.indices}
        rows = [row for row in rows if int(row["case_index"]) in wanted]
    tasks: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(row["selected_overlay_path"])
        if not image_path.exists():
            raise SystemExit(f"Missing selected overlay image: {image_path}")
        task_id = f"stage5_reduce_{int(row['case_index']):03d}_{int(row['candidate_order']):02d}"
        tasks.append(
            {
                "task_id": task_id,
                "case_index": int(row["case_index"]),
                "case_display": row["case_display"],
                "bbox_source": row["bbox_source"],
                "candidate_order": int(row["candidate_order"]),
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "crop_path": row["crop_path"],
                "selected_overlay_path": row["selected_overlay_path"],
                "metadata_path": row["metadata_path"],
                "prompt": prompt,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "created_at": _timestamp(),
            }
        )
    return tasks


def _run_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
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
            "prompt_version",
            "model",
            "reasoning_effort",
            "created_at",
        )
    }
    record.update(
        {
            "raw_response": "",
            "reduce_decision": "unknown",
            "error": "",
            "usage": {},
            "response_model": "",
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
            reasoning_effort=args.reasoning_effort,
        )
        record.update(
            {
                "raw_response": raw,
                "reduce_decision": _parse_yes_no(raw),
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font: Any, fill: str = "#111111") -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += 23
    return y


def _case_groups(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(int(row["case_index"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["candidate_order"]))
    return dict(sorted(groups.items()))


def _draw_cover(results: list[dict[str, Any]], prompt: str, args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(42)
    header = _font(28)
    body = _font(20)
    small = _font(17)
    y = 55
    draw.text((65, y), "Stage 5 Reduce Review", font=title, fill="black")
    y += 58
    draw.text(
        (65, y),
        f"model={args.model} | thinking={args.reasoning_effort} | crops={len(results)}",
        font=body,
        fill="#111111",
    )
    y += 42
    counts: dict[str, int] = {}
    for row in results:
        counts[row["reduce_decision"]] = counts.get(row["reduce_decision"], 0) + 1
    draw.text((65, y), f"Decision counts: {counts}", font=body, fill="#111111")
    y += 48
    draw.text((65, y), "Prompt", font=header, fill="black")
    y += 36
    y = _draw_wrapped(draw, (85, y), prompt, 150, small)
    y += 30
    draw.text((65, y), "Cases", font=header, fill="black")
    y += 36
    for case_index, rows in _case_groups(results).items():
        first = rows[0]
        summary = {
            "yes": sum(1 for r in rows if r["reduce_decision"] == "yes"),
            "no": sum(1 for r in rows if r["reduce_decision"] == "no"),
            "unknown": sum(1 for r in rows if r["reduce_decision"] == "unknown"),
        }
        y = _draw_wrapped(draw, (85, y), f"{case_index}/100 | {first['case_display']} | n={len(rows)} | {summary}", 150, small)
    return page


def _draw_result_page(row: dict[str, Any], args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(34)
    header = _font(26)
    body = _font(19)
    small = _font(16)
    y = 45
    draw.text(
        (55, y),
        f"{int(row['case_index']):03d} candidate {int(row['candidate_order']):02d} | decision={row['reduce_decision']}",
        font=title,
        fill="black",
    )
    y += 48
    y = _draw_wrapped(draw, (55, y), row["case_display"], 150, body)
    y += 18
    draw.text(
        (55, y),
        f"thinking={args.reasoning_effort} | source={row['bbox_source']} | label={row['label']} | error={row['error']}",
        font=body,
        fill="#111111",
    )
    y += 44
    draw.text((55, y), "Selected-candidate overlay sent to VLM", font=header, fill="black")
    y += 38
    image = _thumb(Path(row["selected_overlay_path"]), (1500, 1500))
    page.paste(image, (55, y))
    y += min(1500, image.size[1]) + 48
    draw.text((55, y), "Raw VLM output", font=header, fill="black")
    y += 34
    y = _draw_wrapped(draw, (75, y), row.get("raw_response", ""), 170, small)
    y += 18
    draw.text((55, y), "Input paths", font=header, fill="black")
    y += 34
    paths = f"overlay={row['selected_overlay_path']} | crop={row['crop_path']} | metadata={row['metadata_path']}"
    _draw_wrapped(draw, (75, y), paths, 170, small)
    return page


def _write_pdf(output_root: Path, results: list[dict[str, Any]], prompt: str, args: argparse.Namespace) -> Path:
    pages = [_draw_cover(results, prompt, args)]
    pages.extend(_draw_result_page(row, args) for row in results)
    pdf_path = output_root / "visuals" / f"stage5_reduce_review_{args.reasoning_effort}_thinking.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _summarize(output_root: Path, results: list[dict[str, Any]], args: argparse.Namespace, pdf_path: Path) -> Path:
    rows = []
    for row in results:
        rows.append(
            {
                "case_index": row["case_index"],
                "case_display": row["case_display"],
                "candidate_order": row["candidate_order"],
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "bbox_source": row["bbox_source"],
                "reasoning_effort": row["reasoning_effort"],
                "reduce_decision": row["reduce_decision"],
                "error": row["error"],
                "raw_response": row["raw_response"],
                "selected_overlay_path": row["selected_overlay_path"],
                "crop_path": row["crop_path"],
                "metadata_path": row["metadata_path"],
            }
        )
    csv_path = output_root / "summary" / f"stage5_reduce_review_{args.reasoning_effort}_thinking.csv"
    _write_csv(
        csv_path,
        rows,
        [
            "case_index",
            "case_display",
            "candidate_order",
            "candidate_id",
            "label",
            "bbox_source",
            "reasoning_effort",
            "reduce_decision",
            "error",
            "raw_response",
            "selected_overlay_path",
            "crop_path",
            "metadata_path",
        ],
    )
    usage_cost = sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in results)
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(args.prompt.resolve()),
        "candidates_csv": str(args.candidates.resolve()),
        "output_root": str(output_root.resolve()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_concurrent": args.max_concurrent,
        "candidates": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "decision_counts": {
            value: sum(1 for row in results if row["reduce_decision"] == value)
            for value in ("yes", "no", "unknown")
        },
        "known_usage_cost_if_reported": usage_cost,
        "pdf": str(pdf_path.resolve()),
        "results_jsonl": str((output_root / "reviews" / f"stage5_reduce_review_{args.reasoning_effort}_thinking.jsonl").resolve()),
        "summary_csv": str(csv_path.resolve()),
    }
    summary_path = output_root / "summary" / f"stage5_reduce_review_{args.reasoning_effort}_thinking_summary.json"
    _write_json(summary_path, summary)
    return summary_path


def _write_reproduction(output_root: Path, args: argparse.Namespace, prompt: str, pdf_path: Path, summary_path: Path) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage5_reduce_review.py",
            "--candidates",
            str(args.candidates.resolve()),
            "--prompt",
            str(args.prompt.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
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
        ]
    )
    text = f"""\
Stage 5 reduce/split review
===========================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Reasoning effort: {args.reasoning_effort}
Backend: OpenRouter-compatible chat completions

Objective:
For each Stage 4 selected-candidate overlay, ask whether the highlighted
tissue-candidate detection contains multiple instances and can be split and
reduced further, or whether the detection is already atomic.

Prompt:
{prompt}

Command:
{command}

Inputs:
- Candidate manifest: {args.candidates.resolve()}
- Prompt file: {args.prompt.resolve()}

Outputs:
- PDF: {pdf_path.resolve()}
- Summary JSON: {summary_path.resolve()}
- Results JSONL: {(output_root / 'reviews' / f'stage5_reduce_review_{args.reasoning_effort}_thinking.jsonl').resolve()}
"""
    (output_root / f"reproduction_{args.reasoning_effort}_thinking.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root / f"{args.reasoning_effort}_thinking"
    output_root.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text().strip()
    tasks = _build_tasks(args, prompt)
    _write_jsonl(output_root / "tasks" / f"stage5_reduce_review_{args.reasoning_effort}_thinking_tasks.jsonl", tasks)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks), "output_root": str(output_root)}, indent=2))
        return 0

    base_url, api_key = _api_settings(args)
    if args.max_concurrent > 1:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [pool.submit(_run_one, task, args, base_url, api_key) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [_run_one(task, args, base_url, api_key) for task in tasks]
    results.sort(key=lambda row: (int(row["case_index"]), int(row["candidate_order"])))
    results_path = output_root / "reviews" / f"stage5_reduce_review_{args.reasoning_effort}_thinking.jsonl"
    _write_jsonl(results_path, results)
    pdf_path = _write_pdf(output_root, results, prompt, args)
    summary_path = _summarize(output_root, results, args, pdf_path)
    _write_reproduction(output_root, args, prompt, pdf_path, summary_path)
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
