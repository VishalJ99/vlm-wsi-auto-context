#!/usr/bin/env python3
"""Synthetic missed-bbox stress test for the short Stage 1 reviewer."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    _font,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from stage1_high_recall_edge_review import (
    DEFAULT_CASES_CSV,
    QUALITATIVE_REVIEW_PROMPT,
    QUALITATIVE_REVIEW_PROMPT_VERSION,
    _parse_final_bboxes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_REVIEW_CSV = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_pilot100_short_reviewer_high_thinking_v1"
    / "summary"
    / "edge_review_cases.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "synthetic_missed_bbox_short_reviewer_high_thinking_v1"
)

NO_MISS_PATTERNS = [
    re.compile(r"no (?:potential |significant |visible )?tissue(?:-like)? objects? (?:appear to have been |were |was )?missed", re.I),
    re.compile(r"no (?:significant )?tissue(?:-like)? fragments? (?:appear to have been |were |was )?missed", re.I),
    re.compile(r"no (?:significant )?tissue(?:-like)? sections? (?:appear to have been |were |was )?missed", re.I),
    re.compile(r"all (?:visible |major |significant )?(?:tissue|fragments|tissue fragments|tissue-like objects).*?(?:contained|included|detected|localized|identified)", re.I),
    re.compile(r"successfully localized all", re.I),
    re.compile(r"all .*? have been correctly (?:identified|localized|detected)", re.I),
]
POSITIVE_MISS_PATTERNS = [
    re.compile(r"however[^.\n]{0,240}\b(?:missed|outside|not included|not boxed|completely outside)", re.I),
    re.compile(r"\bhas missed\b", re.I),
    re.compile(r"\bhave been missed:", re.I),
    re.compile(r"\bwere missed:", re.I),
    re.compile(r"\bwas missed:", re.I),
    re.compile(r"completely outside (?:the|any)", re.I),
    re.compile(r"outside any detection", re.I),
    re.compile(r"outside the .*?box", re.I),
    re.compile(r"not included in any", re.I),
    re.compile(r"not been included", re.I),
    re.compile(r"has not been included", re.I),
    re.compile(r"potential tissue-like objects? that were missed", re.I),
    re.compile(r"there (?:is|are).*?missed", re.I),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _review_says_no_missed_objects(text: str) -> bool:
    collapsed = " ".join(text.split())
    return any(pattern.search(collapsed) for pattern in NO_MISS_PATTERNS) and not any(
        pattern.search(collapsed) for pattern in POSITIVE_MISS_PATTERNS
    )


def _bbox_area(bbox: dict[str, Any]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox.get("bbox_thumbnail", [0, 0, 0, 0])]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _select_removed_bbox(bboxes: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    indexed = list(enumerate(bboxes))
    indexed.sort(key=lambda item: (_bbox_area(item[1]), -item[0]), reverse=True)
    return indexed[0]


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    case_rows = {int(row["case_index"]): row for row in _read_csv(args.cases_csv)}
    review_rows = _read_csv(args.baseline_review_csv)
    tasks: list[dict[str, Any]] = []
    for review_row in review_rows:
        if review_row.get("review_error"):
            continue
        if not _review_says_no_missed_objects(review_row.get("qualitative_review", "")):
            continue
        case_index = int(review_row["case_index"])
        case_row = case_rows.get(case_index)
        if not case_row:
            continue
        bboxes = _parse_final_bboxes(case_row)
        if len(bboxes) < 2:
            continue
        thumbnail_path = Path(case_row["thumbnail_path"])
        if not thumbnail_path.exists():
            continue
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size
        removed_idx, removed_bbox = _select_removed_bbox(bboxes)
        remaining_bboxes = [bbox for idx, bbox in enumerate(bboxes) if idx != removed_idx]
        case_slug = _safe_slug(case_row["case_display"])
        degraded_overlay = args.output_root / "synthetic_overlays" / f"{case_slug}_removed_box_{removed_idx + 1:02d}.png"
        removed_overlay = args.output_root / "removed_box_overlays" / f"{case_slug}_removed_box_{removed_idx + 1:02d}.png"
        _draw_redetect_overlay(thumbnail_path, remaining_bboxes, degraded_overlay)
        _draw_redetect_overlay(thumbnail_path, [removed_bbox], removed_overlay)
        geometry = _bbox_geometry(removed_bbox, thumbnail_size)
        tasks.append(
            {
                "task_id": f"synthetic_miss_{case_index:03d}",
                "case_index": case_index,
                "case_display": case_row["case_display"],
                "thumbnail_path": str(thumbnail_path),
                "original_overlay_path": review_row.get("review_overlay_path") or case_row.get("final_overlay_path", ""),
                "synthetic_overlay_path": str(degraded_overlay),
                "removed_overlay_path": str(removed_overlay),
                "baseline_review": review_row.get("qualitative_review", ""),
                "original_bbox_count": len(bboxes),
                "synthetic_bbox_count": len(remaining_bboxes),
                "removed_bbox_index_1based": removed_idx + 1,
                "removed_bbox": {
                    "label": removed_bbox.get("label", f"box_{removed_idx + 1}"),
                    "source_label": removed_bbox.get("source_label", ""),
                    "bbox_thumbnail": removed_bbox.get("bbox_thumbnail", []),
                    "bbox_normalized": removed_bbox.get("box_2d_yxyx_normalized", []),
                    **geometry,
                },
                "created_at": _timestamp(),
            }
        )
        if len(tasks) >= args.n:
            break
    if len(tasks) < args.n:
        raise SystemExit(f"Only selected {len(tasks)} eligible no-miss cases; requested {args.n}.")
    return tasks


def _review_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "case_index": task["case_index"],
        "case_display": task["case_display"],
        "prompt_version": QUALITATIVE_REVIEW_PROMPT_VERSION,
        "prompt_text": QUALITATIVE_REVIEW_PROMPT,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "thumbnail_path": task["thumbnail_path"],
        "synthetic_overlay_path": task["synthetic_overlay_path"],
        "original_overlay_path": task["original_overlay_path"],
        "removed_overlay_path": task["removed_overlay_path"],
        "removed_bbox_index_1based": task["removed_bbox_index_1based"],
        "removed_bbox": task["removed_bbox"],
        "original_bbox_count": task["original_bbox_count"],
        "synthetic_bbox_count": task["synthetic_bbox_count"],
        "baseline_review": task["baseline_review"],
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=QUALITATIVE_REVIEW_PROMPT,
            image_paths=[Path(task["thumbnail_path"]), Path(task["synthetic_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        record["raw_response"] = raw
        record["parsed_response"] = {"raw_text": raw}
        record["usage"] = usage
        record["response_model"] = response_model
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["error"] = repr(exc)
    return record


def _run_reviews(tasks: list[dict[str, Any]], args: argparse.Namespace, base_url: str, api_key: str) -> list[dict[str, Any]]:
    if args.max_concurrent <= 1:
        return [_review_one(task, args, base_url, api_key) for task in tasks]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = [pool.submit(_review_one, task, args, base_url, api_key) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["case_index"])
    return results


def _write_summary(output_root: Path, reviews: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for review in reviews:
        usage = review.get("usage") if isinstance(review.get("usage"), dict) else {}
        rows.append(
            {
                "case_index": review["case_index"],
                "case_display": review["case_display"],
                "removed_bbox_index_1based": review["removed_bbox_index_1based"],
                "original_bbox_count": review["original_bbox_count"],
                "synthetic_bbox_count": review["synthetic_bbox_count"],
                "removed_bbox_thumbnail": json.dumps(review.get("removed_bbox", {}).get("bbox_thumbnail", [])),
                "removed_bbox_area_ratio": review.get("removed_bbox", {}).get("area_ratio", ""),
                "baseline_review": review.get("baseline_review", ""),
                "synthetic_review": review.get("raw_response", ""),
                "review_error": review.get("error", ""),
                "cost": usage.get("cost", ""),
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "thumbnail_path": review.get("thumbnail_path", ""),
                "original_overlay_path": review.get("original_overlay_path", ""),
                "synthetic_overlay_path": review.get("synthetic_overlay_path", ""),
                "removed_overlay_path": review.get("removed_overlay_path", ""),
            }
        )
    _write_csv(
        output_root / "summary" / "synthetic_missed_bbox_reviews.csv",
        rows,
        [
            "case_index",
            "case_display",
            "removed_bbox_index_1based",
            "original_bbox_count",
            "synthetic_bbox_count",
            "removed_bbox_thumbnail",
            "removed_bbox_area_ratio",
            "baseline_review",
            "synthetic_review",
            "review_error",
            "cost",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "thumbnail_path",
            "original_overlay_path",
            "synthetic_overlay_path",
            "removed_overlay_path",
        ],
    )


def _write_pdf(output_root: Path, args: argparse.Namespace, tasks: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> None:
    review_by_task = {review["task_id"]: review for review in reviews}
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)

    title = Image.new("RGB", (2200, 2600), "white")
    draw = ImageDraw.Draw(title)
    y = 45
    draw.text((45, y), "Synthetic Missed-BBox Reviewer Probe", font=title_font, fill="black")
    y += 50
    y = _draw_wrapped(
        draw,
        (45, y),
        (
            f"cases={len(tasks)} | model={args.model} | reasoning={args.reasoning_effort or 'unspecified'} | "
            f"max_concurrent={args.max_concurrent} | ticket=PER-207"
        ),
        body_font,
        160,
        "#222222",
    )
    y += 20
    draw.text((45, y), "Reviewer prompt", font=body_font, fill="black")
    y += 32
    y = _draw_wrapped(draw, (65, y), QUALITATIVE_REVIEW_PROMPT.strip(), small_font, 170, "#111111")
    y += 24
    y = _draw_wrapped(
        draw,
        (65, y),
        (
            "Selection: baseline short-review text indicated no missed tissue-like objects; "
            "cases required at least two final accepted boxes. The largest final bbox was removed."
        ),
        small_font,
        170,
        "#111111",
    )
    pages.append(title)

    for task in tasks:
        review = review_by_task[task["task_id"]]
        page = Image.new("RGB", (2200, 2700), "white")
        draw = ImageDraw.Draw(page)
        y = 35
        draw.text((45, y), task["case_display"], font=title_font, fill="black")
        y += 44
        header = (
            f"removed original box {task['removed_bbox_index_1based']} of {task['original_bbox_count']} | "
            f"synthetic boxes={task['synthetic_bbox_count']} | "
            f"removed area={task['removed_bbox'].get('area_ratio')}"
        )
        y = _draw_wrapped(draw, (45, y), header, body_font, 170, "#111111")
        y += 18
        images = [
            ("Source thumbnail", Path(task["thumbnail_path"])),
            ("Original overlay", Path(task["original_overlay_path"])),
            ("Synthetic overlay", Path(task["synthetic_overlay_path"])),
            ("Removed box only", Path(task["removed_overlay_path"])),
        ]
        x_positions = [45, 590, 1135, 1680]
        for x, (label, path) in zip(x_positions, images):
            draw.text((x, y), label, font=body_font, fill="black")
            page.paste(_thumb(path, (500, 320)), (x, y + 30))
        y += 390
        draw.text((45, y), "Baseline reviewer output", font=body_font, fill="black")
        y += 30
        y = _draw_wrapped(draw, (60, y), task["baseline_review"], small_font, 180, "#333333")
        y += 18
        draw.text((45, y), "Synthetic reviewer output", font=body_font, fill="black")
        y += 30
        if review.get("error"):
            y = _draw_wrapped(draw, (60, y), f"ERROR: {review['error']}", small_font, 180, "#aa0000")
        else:
            y = _draw_wrapped(draw, (60, y), review.get("raw_response", ""), small_font, 180, "#111111")
        pages.append(page)

    pdf_path = output_root / "visuals" / "synthetic_missed_bbox_review.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_reproduction(output_root: Path, args: argparse.Namespace) -> None:
    command = [
        "python",
        "scripts/stage1_synthetic_missed_bbox_review.py",
        "--cases-csv",
        str(args.cases_csv.resolve()),
        "--baseline-review-csv",
        str(args.baseline_review_csv.resolve()),
        "--output-root",
        str(output_root),
        "--n",
        str(args.n),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort or "",
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    text = f"""\
Synthetic missed-bbox reviewer probe
====================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Model: {args.model}
Reasoning effort: {args.reasoning_effort or 'unspecified'}
Max concurrent: {args.max_concurrent}
Max tokens: {args.max_tokens}
Temperature: {args.temperature}
Cases CSV: {args.cases_csv.resolve()}
Baseline review CSV: {args.baseline_review_csv.resolve()}
Requested cases: {args.n}

Command:
{" ".join(shlex.quote(part) for part in command if part != "")}

Reviewer prompt version:
{QUALITATIVE_REVIEW_PROMPT_VERSION}

Reviewer prompt text:
{QUALITATIVE_REVIEW_PROMPT.strip()}

Selection:
- Start from the pilot-100 short reviewer CSV.
- Keep rows whose reviewer text indicates no missed tissue-like objects by heuristic phrase matching.
- Exclude rows with obvious positive missed-object language.
- Require at least two final accepted Stage 1 bboxes.
- Select the first {args.n} eligible rows by pilot case index.
- Remove the largest final accepted bbox by thumbnail area from each selected case.

Outputs:
- Tasks: {output_root / 'tasks' / 'synthetic_missed_bbox_tasks.jsonl'}
- Raw reviews: {output_root / 'reviews' / 'synthetic_missed_bbox_reviews.jsonl'}
- Summary CSV: {output_root / 'summary' / 'synthetic_missed_bbox_reviews.csv'}
- Summary JSON: {output_root / 'summary' / 'synthetic_missed_bbox_summary.json'}
- PDF: {output_root / 'visuals' / 'synthetic_missed_bbox_review.pdf'}
- Synthetic overlays: {output_root / 'synthetic_overlays'}
- Removed-box overlays: {output_root / 'removed_box_overlays'}
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    if not args.cases_csv.exists():
        raise SystemExit(f"Missing cases CSV: {args.cases_csv}")
    if not args.baseline_review_csv.exists():
        raise SystemExit(f"Missing baseline review CSV: {args.baseline_review_csv}")
    tasks = _load_tasks(args)
    tasks_path = args.output_root / "tasks" / "synthetic_missed_bbox_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tasks": len(tasks),
                    "case_indices": [task["case_index"] for task in tasks],
                    "output_root": str(args.output_root),
                    "tasks_jsonl": str(tasks_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    reviews = _run_reviews(tasks, args, base_url, api_key)
    reviews_path = args.output_root / "reviews" / "synthetic_missed_bbox_reviews.jsonl"
    _write_jsonl(reviews_path, reviews)
    _write_summary(args.output_root, reviews)
    _write_pdf(args.output_root, args, tasks, reviews)
    summary = {
        "created_at": _timestamp(),
        "git_commit": _repo_git_commit(),
        "ticket": "PER-207",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "cases": len(tasks),
        "review_errors": sum(1 for review in reviews if review.get("error")),
        "total_cost": sum(((review.get("usage") or {}).get("cost") or 0.0) for review in reviews),
        "total_tokens": sum(((review.get("usage") or {}).get("total_tokens") or 0) for review in reviews),
        "reasoning_tokens": sum(
            (((review.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            for review in reviews
        ),
        "case_indices": [task["case_index"] for task in tasks],
        "output_root": str(args.output_root),
    }
    _write_json(args.output_root / "summary" / "synthetic_missed_bbox_summary.json", summary)
    _write_reproduction(args.output_root, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    parser.add_argument("--baseline-review-csv", type=Path, default=DEFAULT_BASELINE_REVIEW_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high", "xhigh", "none"], default="high")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
