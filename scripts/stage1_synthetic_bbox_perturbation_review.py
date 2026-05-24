#!/usr/bin/env python3
"""Run Stage 1 reviewer on synthetic bbox perturbations for calibration."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _api_settings,
    _bbox_geometry,
    _chat_with_images,
    _draw_redetect_overlay,
    _repo_git_commit,
    _safe_slug,
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
DEFAULT_SEED_TASKS_JSONL = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "synthetic_missed_bbox_short_reviewer_high_thinking_v1"
    / "tasks"
    / "synthetic_missed_bbox_tasks.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "synthetic_bbox_perturbation_review_low_thinking_v1"
)
DEFAULT_VARIANTS = ("remove_largest", "shrink_largest_50", "near_full_largest")

PASS_NEGATION_PATTERNS = [
    re.compile(r"\bno\b[^.\n]{0,160}\bmiss(?:ed|ing)?\b", re.I),
    re.compile(r"\bnot\b[^.\n]{0,100}\bmissed\b", re.I),
    re.compile(r"\ball\b[^.\n]{0,180}\b(?:contained|included|localized|localised|detected|identified)\b", re.I),
    re.compile(r"\bcorrectly localized\b", re.I),
]
MISSED_PATTERNS = [
    re.compile(r"\bfailed to (?:correctly |accurately )?(?:identify|detect|locali[sz]e|include|capture|create|provide)\b", re.I),
    re.compile(r"\b(?:has|have|was|were|is|are) (?:completely |entirely |partially )?missed\b", re.I),
    re.compile(r"\bmissed (?:tissue|object|objects|fragment|fragments|core|cores|section|sections|signal|candidate|candidates|portion|part)\b", re.I),
    re.compile(r"\bmisses\b", re.I),
    re.compile(r"\bmissing\b", re.I),
    re.compile(r"\bincomplete\b", re.I),
    re.compile(r"\bnot (?:enclosed|included|localized|localised|detected|identified|captured|covered)\b", re.I),
    re.compile(r"\boutside (?:any|all|the) (?:detection|bounding|bbox|box)", re.I),
    re.compile(r"\bnot fully (?:captured|included|enclosed|covered)\b", re.I),
    re.compile(r"\black of (?:a |an )?(?:specific |individual )?(?:bbox|box|bounding box|detection)\b", re.I),
]
OVER_DETECTION_PATTERNS = [
    re.compile(r"\b(?:failed|fails|failure) to locali[sz]e\b", re.I),
    re.compile(r"\b(?:encompasses|encompassing|covers|covering|contains) (?:almost |nearly |most |all |the whole |the entire )[^.\n]{0,120}\b(?:thumbnail|slide|image|field|area)\b", re.I),
    re.compile(r"\b(?:entire|whole|full|near-full|near full|almost entire|nearly entire)[^.\n]{0,80}\b(?:thumbnail|slide|image)\b", re.I),
    re.compile(r"\b(?:giant|degenerate|fallback|too large|very large|over[- ]?detection|over[- ]?coverage)\b", re.I),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _replace_bbox_geometry(bbox: dict[str, Any], new_bbox: list[int], thumbnail_size: tuple[int, int]) -> dict[str, Any]:
    x1, y1, x2, y2 = new_bbox
    width, height = thumbnail_size
    updated = dict(bbox)
    updated["bbox_thumbnail"] = [int(x1), int(y1), int(x2), int(y2)]
    updated["box_2d_yxyx_normalized"] = [
        round(y1 / height * 1000, 2),
        round(x1 / width * 1000, 2),
        round(y2 / height * 1000, 2),
        round(x2 / width * 1000, 2),
    ]
    updated["raw_box_2d"] = updated["box_2d_yxyx_normalized"]
    return updated


def _shrink_bbox(bbox_thumbnail: list[int], thumbnail_size: tuple[int, int], factor: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_thumbnail]
    width, height = thumbnail_size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_w = max(4.0, (x2 - x1) * factor)
    new_h = max(4.0, (y2 - y1) * factor)
    nx1 = max(0, min(width - 1, round(cx - new_w / 2)))
    ny1 = max(0, min(height - 1, round(cy - new_h / 2)))
    nx2 = max(nx1 + 1, min(width, round(cx + new_w / 2)))
    ny2 = max(ny1 + 1, min(height, round(cy + new_h / 2)))
    return [int(nx1), int(ny1), int(nx2), int(ny2)]


def _near_full_bbox(thumbnail_size: tuple[int, int]) -> list[int]:
    width, height = thumbnail_size
    return [round(width * 0.03), round(height * 0.03), round(width * 0.97), round(height * 0.97)]


def _make_variant_bboxes(
    *,
    bboxes: list[dict[str, Any]],
    removed_idx: int,
    thumbnail_size: tuple[int, int],
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = bboxes[removed_idx]
    selected_bbox = [int(v) for v in selected["bbox_thumbnail"]]
    if variant == "remove_largest":
        output = [bbox for idx, bbox in enumerate(bboxes) if idx != removed_idx]
        expected = {
            "expected_trigger": True,
            "expected_reason": "missed_tissue",
            "expected_location_basis": "removed_bbox",
            "modified_bbox_thumbnail": selected_bbox,
        }
        return output, expected
    if variant == "shrink_largest_50":
        shrunk = _shrink_bbox(selected_bbox, thumbnail_size, 0.5)
        output = list(bboxes)
        output[removed_idx] = _replace_bbox_geometry(selected, shrunk, thumbnail_size)
        expected = {
            "expected_trigger": True,
            "expected_reason": "missed_tissue_or_cutoff",
            "expected_location_basis": "selected_bbox_shrunk_to_center_50pct",
            "modified_bbox_thumbnail": selected_bbox,
            "perturbed_bbox_thumbnail": shrunk,
        }
        return output, expected
    if variant == "near_full_largest":
        huge = _near_full_bbox(thumbnail_size)
        output = list(bboxes)
        output[removed_idx] = _replace_bbox_geometry(selected, huge, thumbnail_size)
        expected = {
            "expected_trigger": True,
            "expected_reason": "over_detection",
            "expected_location_basis": "selected_bbox_replaced_with_near_full_thumbnail_bbox",
            "modified_bbox_thumbnail": selected_bbox,
            "perturbed_bbox_thumbnail": huge,
        }
        return output, expected
    raise ValueError(f"Unknown variant: {variant}")


def _detect_flags(review_text: str) -> dict[str, Any]:
    collapsed = " ".join(review_text.split())
    missed = any(pattern.search(collapsed) for pattern in MISSED_PATTERNS)
    over = any(pattern.search(collapsed) for pattern in OVER_DETECTION_PATTERNS)
    if missed and not over:
        # Common clean-pass boilerplate contains "no missed". Do not suppress
        # genuinely positive text that also says "while X is localized, Y is missed".
        first_positive = min(
            [m.start() for pattern in MISSED_PATTERNS for m in pattern.finditer(collapsed)] or [10**9]
        )
        negating = any(
            m.start() <= first_positive <= m.end() + 30
            for pattern in PASS_NEGATION_PATTERNS
            for m in pattern.finditer(collapsed)
        )
        if negating:
            missed = False
    reasons: list[str] = []
    if missed:
        reasons.append("missed_tissue")
    if over:
        reasons.append("over_detection")
    return {
        "flagged": bool(reasons),
        "flag_reasons": reasons,
    }


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    case_rows = {int(row["case_index"]): row for row in _read_csv(args.cases_csv)}
    seed_tasks = _read_jsonl(args.seed_tasks_jsonl)
    tasks: list[dict[str, Any]] = []
    for seed in seed_tasks:
        case_index = int(seed["case_index"])
        case_row = case_rows[case_index]
        thumbnail_path = Path(case_row["thumbnail_path"])
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size
        bboxes = _parse_final_bboxes(case_row)
        if not bboxes:
            raise SystemExit(f"No final bboxes for case {case_index}")
        removed_idx = int(seed["removed_bbox_index_1based"]) - 1
        case_slug = _safe_slug(case_row["case_display"])
        for variant in args.variants:
            variant_bboxes, expected = _make_variant_bboxes(
                bboxes=bboxes,
                removed_idx=removed_idx,
                thumbnail_size=thumbnail_size,
                variant=variant,
            )
            overlay_path = (
                args.output_root
                / "perturbation_overlays"
                / variant
                / f"{case_slug}_{variant}_box_{removed_idx + 1:02d}.png"
            )
            selected_overlay_path = (
                args.output_root
                / "selected_bbox_overlays"
                / f"{case_slug}_selected_box_{removed_idx + 1:02d}.png"
            )
            _draw_redetect_overlay(thumbnail_path, variant_bboxes, overlay_path)
            _draw_redetect_overlay(thumbnail_path, [bboxes[removed_idx]], selected_overlay_path)
            selected_geometry = _bbox_geometry(bboxes[removed_idx], thumbnail_size)
            tasks.append(
                {
                    "task_id": f"perturb_{case_index:03d}_{variant}",
                    "case_index": case_index,
                    "case_display": case_row["case_display"],
                    "variant": variant,
                    "thumbnail_path": str(thumbnail_path),
                    "perturbation_overlay_path": str(overlay_path),
                    "selected_bbox_overlay_path": str(selected_overlay_path),
                    "original_bbox_count": len(bboxes),
                    "perturbed_bbox_count": len(variant_bboxes),
                    "selected_bbox_index_1based": removed_idx + 1,
                    "selected_bbox": {
                        "label": bboxes[removed_idx].get("label", f"box_{removed_idx + 1}"),
                        "source_label": bboxes[removed_idx].get("source_label", ""),
                        "bbox_thumbnail": bboxes[removed_idx].get("bbox_thumbnail", []),
                        "bbox_normalized": bboxes[removed_idx].get("box_2d_yxyx_normalized", []),
                        **selected_geometry,
                    },
                    "expected": expected,
                    "created_at": _timestamp(),
                }
            )
    return tasks


def _review_one(task: dict[str, Any], args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "case_index": task["case_index"],
        "case_display": task["case_display"],
        "variant": task["variant"],
        "prompt_version": QUALITATIVE_REVIEW_PROMPT_VERSION,
        "prompt_text": QUALITATIVE_REVIEW_PROMPT,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort or "",
        "thumbnail_path": task["thumbnail_path"],
        "perturbation_overlay_path": task["perturbation_overlay_path"],
        "selected_bbox_overlay_path": task["selected_bbox_overlay_path"],
        "selected_bbox_index_1based": task["selected_bbox_index_1based"],
        "selected_bbox": task["selected_bbox"],
        "expected": task["expected"],
        "created_at": _timestamp(),
        "error": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=QUALITATIVE_REVIEW_PROMPT,
            image_paths=[Path(task["thumbnail_path"]), Path(task["perturbation_overlay_path"])],
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
        record.update(_detect_flags(raw))
    except Exception as exc:
        record["raw_response"] = ""
        record["parsed_response"] = {}
        record["usage"] = {}
        record["response_model"] = ""
        record["flagged"] = False
        record["flag_reasons"] = []
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
    return sorted(results, key=lambda row: (int(row["case_index"]), row["variant"]))


def _write_summary(output_root: Path, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for review in reviews:
        usage = review.get("usage") if isinstance(review.get("usage"), dict) else {}
        selected = review.get("selected_bbox") if isinstance(review.get("selected_bbox"), dict) else {}
        expected = review.get("expected") if isinstance(review.get("expected"), dict) else {}
        rows.append(
            {
                "task_id": review["task_id"],
                "case_index": review["case_index"],
                "case_display": review["case_display"],
                "variant": review["variant"],
                "selected_bbox_index_1based": review["selected_bbox_index_1based"],
                "selected_bbox_thumbnail": json.dumps(selected.get("bbox_thumbnail", [])),
                "selected_bbox_area_ratio": selected.get("area_ratio", ""),
                "expected_reason": expected.get("expected_reason", ""),
                "expected_location_basis": expected.get("expected_location_basis", ""),
                "perturbed_bbox_thumbnail": json.dumps(expected.get("perturbed_bbox_thumbnail", [])),
                "flagged": review.get("flagged", False),
                "flag_reasons": json.dumps(review.get("flag_reasons", [])),
                "review_text": review.get("raw_response", ""),
                "review_error": review.get("error", ""),
                "cost": usage.get("cost", ""),
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "thumbnail_path": review.get("thumbnail_path", ""),
                "perturbation_overlay_path": review.get("perturbation_overlay_path", ""),
                "selected_bbox_overlay_path": review.get("selected_bbox_overlay_path", ""),
            }
        )
    _write_csv(
        output_root / "summary" / "synthetic_bbox_perturbation_reviews.csv",
        rows,
        [
            "task_id",
            "case_index",
            "case_display",
            "variant",
            "selected_bbox_index_1based",
            "selected_bbox_thumbnail",
            "selected_bbox_area_ratio",
            "expected_reason",
            "expected_location_basis",
            "perturbed_bbox_thumbnail",
            "flagged",
            "flag_reasons",
            "review_text",
            "review_error",
            "cost",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "thumbnail_path",
            "perturbation_overlay_path",
            "selected_bbox_overlay_path",
        ],
    )
    summary: dict[str, Any] = {
        "tasks": len(reviews),
        "errors": sum(1 for review in reviews if review.get("error")),
        "flagged": sum(1 for review in reviews if review.get("flagged")),
        "unflagged": sum(1 for review in reviews if not review.get("flagged")),
        "variants": {},
        "cost": sum(
            float(review.get("usage", {}).get("cost", 0) or 0)
            for review in reviews
            if isinstance(review.get("usage"), dict)
        ),
        "total_tokens": sum(
            int(review.get("usage", {}).get("total_tokens", 0) or 0)
            for review in reviews
            if isinstance(review.get("usage"), dict)
        ),
        "reasoning_tokens": sum(
            int(review.get("usage", {}).get("reasoning_tokens", 0) or 0)
            for review in reviews
            if isinstance(review.get("usage"), dict)
        ),
    }
    for review in reviews:
        variant = review["variant"]
        variant_summary = summary["variants"].setdefault(
            variant,
            {"tasks": 0, "flagged": 0, "unflagged": 0, "errors": 0},
        )
        variant_summary["tasks"] += 1
        variant_summary["flagged"] += int(bool(review.get("flagged")))
        variant_summary["unflagged"] += int(not bool(review.get("flagged")))
        variant_summary["errors"] += int(bool(review.get("error")))
    _write_json(output_root / "summary" / "synthetic_bbox_perturbation_summary.json", summary)
    return summary


def _write_reproduction(output_root: Path, args: argparse.Namespace) -> None:
    command = [
        "python",
        "scripts/stage1_synthetic_bbox_perturbation_review.py",
        "--cases-csv",
        str(args.cases_csv.resolve()),
        "--seed-tasks-jsonl",
        str(args.seed_tasks_jsonl.resolve()),
        "--output-root",
        str(output_root),
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
        "--variants",
        *args.variants,
    ]
    if args.task_ids:
        command.extend(["--task-ids", *args.task_ids])
    if args.reuse_reviews_jsonl:
        command.extend(["--reuse-reviews-jsonl", str(args.reuse_reviews_jsonl.resolve())])
    text = f"""\
Synthetic bbox perturbation reviewer probe
=========================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Model: {args.model}
Reasoning effort: {args.reasoning_effort or 'unspecified'}
Max concurrent: {args.max_concurrent}
Max tokens: {args.max_tokens}
Temperature: {args.temperature}
Cases CSV: {args.cases_csv.resolve()}
Seed tasks JSONL: {args.seed_tasks_jsonl.resolve()}
Variants: {', '.join(args.variants)}
Task IDs: {', '.join(args.task_ids)}
Reused reviews JSONL: {args.reuse_reviews_jsonl.resolve() if args.reuse_reviews_jsonl else ''}

Command:
{" ".join(shlex.quote(part) for part in command if part != "")}

Reviewer prompt version:
{QUALITATIVE_REVIEW_PROMPT_VERSION}

Reviewer prompt text:
{QUALITATIVE_REVIEW_PROMPT.strip()}

Selection:
- Reuse the 30 cases selected by the previous synthetic missed-bbox probe.
- Reuse the same selected bbox index for each case.
- For each selected bbox, create the requested perturbation variants.

Outputs:
- Tasks: {output_root / 'tasks' / 'synthetic_bbox_perturbation_tasks.jsonl'}
- Raw reviews: {output_root / 'reviews' / 'synthetic_bbox_perturbation_reviews.jsonl'}
- Summary CSV: {output_root / 'summary' / 'synthetic_bbox_perturbation_reviews.csv'}
- Summary JSON: {output_root / 'summary' / 'synthetic_bbox_perturbation_summary.json'}
- Perturbation overlays: {output_root / 'perturbation_overlays'}
- Selected bbox overlays: {output_root / 'selected_bbox_overlays'}
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    if not args.cases_csv.exists():
        raise SystemExit(f"Missing cases CSV: {args.cases_csv}")
    if not args.seed_tasks_jsonl.exists():
        raise SystemExit(f"Missing seed tasks JSONL: {args.seed_tasks_jsonl}")
    tasks = _load_tasks(args)
    if args.task_ids:
        requested = set(args.task_ids)
        tasks = [task for task in tasks if task["task_id"] in requested]
        missing = sorted(requested - {task["task_id"] for task in tasks})
        if missing:
            raise SystemExit(f"Requested task IDs were not generated: {missing}")
    _write_jsonl(args.output_root / "tasks" / "synthetic_bbox_perturbation_tasks.jsonl", tasks)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks)}, indent=2))
        _write_reproduction(args.output_root, args)
        return 0
    if args.reuse_reviews_jsonl:
        reviews = _read_jsonl(args.reuse_reviews_jsonl)
        for review in reviews:
            review.update(_detect_flags(review.get("raw_response", "")))
    else:
        base_url, api_key = _api_settings(args)
        reviews = _run_reviews(tasks, args, base_url, api_key)
    _write_jsonl(args.output_root / "reviews" / "synthetic_bbox_perturbation_reviews.jsonl", reviews)
    summary = _write_summary(args.output_root, reviews)
    _write_reproduction(args.output_root, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    parser.add_argument("--seed-tasks-jsonl", type=Path, default=DEFAULT_SEED_TASKS_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[],
        help="Optional generated task IDs to review, e.g. perturb_028_near_full_largest.",
    )
    parser.add_argument(
        "--reuse-reviews-jsonl",
        type=Path,
        default=None,
        help="Recompute summaries from an existing raw reviews JSONL without new VLM calls.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
