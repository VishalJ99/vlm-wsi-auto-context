#!/usr/bin/env python3
"""Run a two-pass Stage 2b non-minor detection-failure router."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _api_settings,
    _repo_git_commit,
    _timestamp,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from stage1_review_trigger_router import (
    DEFAULT_INPUT_REVIEWS,
    DEFAULT_SOURCE_REVIEW_ROOT,
    _chat_text,
    _parse_router_response,
    _read_jsonl,
    _review_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_REVIEW_ROOT / "stage2b_nonminor_two_pass_gemini_flash_low_smoke_case001_v1"
DEFAULT_FIRST_PROMPT_FILE = (
    REPO_ROOT / "prompts" / "stage1_detector_oracle" / "stage2b_nonminor_detection_failure_json.txt"
)
DEFAULT_SECOND_PROMPT_FILE = (
    REPO_ROOT
    / "prompts"
    / "stage1_detector_oracle"
    / "stage2b_nonminor_detection_failure_adjudicate_json.txt"
)


def _prompt_version(path: Path) -> str:
    return f"{path.stem}_2026-05-24"


def _parse_indices(value: str) -> set[int] | None:
    value = value.strip()
    if not value:
        return None
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def _routing_question_text() -> str:
    return "did this review describe a non-minor detection failure?"


def _first_prompt(prompt_template: str, record: dict[str, Any]) -> str:
    review_text = _review_text(record)
    return (
        prompt_template.strip()
        + "\n\nCase:\n"
        + str(record.get("case_display") or "")
        + "\n\nReviewer metadata:\n"
        + json.dumps(
            {
                "task_id": record.get("task_id", ""),
                "reviewer_model": record.get("model", ""),
                "reviewer_reasoning_effort": record.get("reasoning_effort", ""),
                "reviewed_bbox_count": record.get("reviewed_bbox_count", ""),
                "overlay_kind": record.get("overlay_kind", ""),
                "raw_response_status": record.get("raw_response_status", ""),
            },
            sort_keys=True,
        )
        + "\n\nReview text:\n"
        + review_text
    )


def _second_prompt(
    prompt_template: str,
    record: dict[str, Any],
    first_raw: str,
    first_parsed: dict[str, Any],
) -> str:
    review_text = _review_text(record)
    return (
        prompt_template.strip()
        + "\n\nCase:\n"
        + str(record.get("case_display") or "")
        + "\n\nOriginal reviewer output:\n"
        + review_text
        + "\n\nInitial answer:\n"
        + json.dumps(
            {
                "raw_response": first_raw,
                "parsed_response": first_parsed,
                "answer": first_parsed.get("answer", ""),
                "justification": first_parsed.get("justification", first_parsed.get("rationale", "")),
            },
            sort_keys=True,
        )
    )


def _call_two_pass(
    record: dict[str, Any],
    args: argparse.Namespace,
    first_prompt_template: str,
    second_prompt_template: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    output = {
        "task_id": record.get("task_id", ""),
        "case_index": record.get("case_index", ""),
        "case_display": record.get("case_display", ""),
        "source_review_model": record.get("model", ""),
        "source_review_reasoning_effort": record.get("reasoning_effort", ""),
        "source_review_prompt_version": record.get("prompt_version", ""),
        "source_review_text": _review_text(record),
        "routing_question": _routing_question_text(),
        "first_prompt_version": _prompt_version(args.first_prompt_file),
        "first_prompt_file": str(args.first_prompt_file),
        "second_prompt_version": _prompt_version(args.second_prompt_file),
        "second_prompt_file": str(args.second_prompt_file),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "created_at": _timestamp(),
        "first_raw_response": "",
        "first_parsed_response": {},
        "second_raw_response": "",
        "second_parsed_response": {},
        "first_non_minor_detection_failure": "",
        "first_justification": "",
        "final_non_minor_detection_failure": "",
        "final_answer": "",
        "final_justification": "",
        "first_usage": {},
        "second_usage": {},
        "first_response_model": "",
        "second_response_model": "",
        "error": "",
    }
    try:
        first_raw, first_usage, first_response_model = _chat_text(
            model=args.model,
            prompt_text=_first_prompt(first_prompt_template, record),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        first_parsed = _parse_router_response(first_raw)
        second_raw, second_usage, second_response_model = _chat_text(
            model=args.model,
            prompt_text=_second_prompt(second_prompt_template, record, first_raw, first_parsed),
            temperature=args.temperature,
            max_tokens=args.second_max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        second_parsed = _parse_router_response(second_raw)
        output.update(
            {
                "first_raw_response": first_raw,
                "first_parsed_response": first_parsed,
                "second_raw_response": second_raw,
                "second_parsed_response": second_parsed,
                "first_non_minor_detection_failure": first_parsed.get("non_minor_detection_failure"),
                "first_justification": first_parsed.get("justification") or first_parsed.get("rationale", ""),
                "final_non_minor_detection_failure": second_parsed.get("non_minor_detection_failure"),
                "final_answer": second_parsed.get("answer", ""),
                "final_justification": second_parsed.get("justification") or second_parsed.get("rationale", ""),
                "first_usage": first_usage,
                "second_usage": second_usage,
                "first_response_model": first_response_model,
                "second_response_model": second_response_model,
            }
        )
    except Exception as exc:
        output["error"] = repr(exc)
    return output


def _run(records: list[dict[str, Any]], args: argparse.Namespace, first_prompt: str, second_prompt: str) -> list[dict[str, Any]]:
    base_url, api_key = _api_settings(args)
    if args.max_concurrent <= 1:
        results = [_call_two_pass(row, args, first_prompt, second_prompt, base_url, api_key) for row in records]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [
                pool.submit(_call_two_pass, row, args, first_prompt, second_prompt, base_url, api_key)
                for row in records
            ]
            for future in as_completed(futures):
                results.append(future.result())
    return sorted(results, key=lambda row: int(row.get("case_index") or 0))


def _csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(
            {
                "case_index": row.get("case_index", ""),
                "case_display": row.get("case_display", ""),
                "first_non_minor_detection_failure": row.get("first_non_minor_detection_failure", ""),
                "first_justification": row.get("first_justification", ""),
                "final_non_minor_detection_failure": row.get("final_non_minor_detection_failure", ""),
                "final_answer": row.get("final_answer", ""),
                "final_justification": row.get("final_justification", ""),
                "error": row.get("error", ""),
                "first_raw_response": row.get("first_raw_response", "")[:800],
                "second_raw_response": row.get("second_raw_response", "")[:800],
            }
        )
    return rows


def _summary(results: list[dict[str, Any]], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    def _usage_sum(key: str) -> tuple[int, float]:
        tokens = 0
        cost = 0.0
        for row in results:
            usage = row.get(key)
            if not isinstance(usage, dict):
                continue
            tokens += int(usage.get("total_tokens") or 0)
            cost += float(usage.get("cost") or 0.0)
        return tokens, cost

    first_tokens, first_cost = _usage_sum("first_usage")
    second_tokens, second_cost = _usage_sum("second_usage")
    return {
        "created_at": _timestamp(),
        "git_commit": _repo_git_commit(),
        "ticket": "PER-207",
        "input_reviews": str(args.input_reviews.resolve()),
        "output_root": str(output_root.resolve()),
        "first_prompt_file": str(args.first_prompt_file.resolve()),
        "second_prompt_file": str(args.second_prompt_file.resolve()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "cases": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "first_non_minor_detection_failure_counts": dict(
            Counter(str(row.get("first_non_minor_detection_failure")) for row in results)
        ),
        "final_non_minor_detection_failure_counts": dict(
            Counter(str(row.get("final_non_minor_detection_failure")) for row in results)
        ),
        "first_total_tokens": first_tokens,
        "second_total_tokens": second_tokens,
        "total_tokens": first_tokens + second_tokens,
        "known_usage_cost_if_reported": first_cost + second_cost,
    }


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    results_path: Path,
    csv_path: Path,
    summary_path: Path,
    first_prompt: str,
    second_prompt: str,
) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage1_review_trigger_two_pass.py",
            "--input-reviews",
            str(args.input_reviews.resolve()),
            "--output-root",
            str(output_root.resolve()),
            "--indices",
            args.indices,
            "--first-prompt-file",
            str(args.first_prompt_file.resolve()),
            "--second-prompt-file",
            str(args.second_prompt_file.resolve()),
            "--model",
            args.model,
            "--reasoning-effort",
            args.reasoning_effort,
            "--max-concurrent",
            str(args.max_concurrent),
            "--max-tokens",
            str(args.max_tokens),
            "--second-max-tokens",
            str(args.second_max_tokens),
            "--temperature",
            str(args.temperature),
        ]
    )
    text = f"""\
Stage 2b two-pass non-minor-failure router
==========================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Source 2a reviews: {args.input_reviews.resolve()}
Model: {args.model}
Reasoning effort: {args.reasoning_effort}
Backend: OpenRouter-compatible chat completions

Command:
{command}

Outputs:
- Results JSONL: {results_path}
- Case summary CSV: {csv_path}
- Summary JSON: {summary_path}

First-pass prompt:
{first_prompt.strip()}

Second-pass prompt:
{second_prompt.strip()}
"""
    (output_root / "reproduction.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-reviews", type=Path, default=DEFAULT_INPUT_REVIEWS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default="1")
    parser.add_argument("--first-prompt-file", type=Path, default=DEFAULT_FIRST_PROMPT_FILE)
    parser.add_argument("--second-prompt-file", type=Path, default=DEFAULT_SECOND_PROMPT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--second-max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    args = parser.parse_args()

    for path in (args.input_reviews, args.first_prompt_file, args.second_prompt_file):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

    selected = _parse_indices(args.indices)
    records = _read_jsonl(args.input_reviews)
    if selected is not None:
        records = [row for row in records if int(row.get("case_index") or 0) in selected]
    if not records:
        raise SystemExit("No matching records selected.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reviews").mkdir(parents=True, exist_ok=True)
    (output_root / "summary").mkdir(parents=True, exist_ok=True)

    first_prompt = args.first_prompt_file.read_text()
    second_prompt = args.second_prompt_file.read_text()
    results = _run(records, args, first_prompt, second_prompt)

    results_path = output_root / "reviews" / "stage2b_two_pass_results.jsonl"
    csv_path = output_root / "summary" / "stage2b_two_pass_cases.csv"
    summary_path = output_root / "summary" / "stage2b_two_pass_summary.json"
    _write_jsonl(results_path, results)
    _write_csv(
        csv_path,
        _csv_rows(results),
        [
            "case_index",
            "case_display",
            "first_non_minor_detection_failure",
            "first_justification",
            "final_non_minor_detection_failure",
            "final_answer",
            "final_justification",
            "error",
            "first_raw_response",
            "second_raw_response",
        ],
    )
    _write_json(summary_path, _summary(results, args, output_root))
    _write_reproduction(output_root, args, results_path, csv_path, summary_path, first_prompt, second_prompt)
    print(json.dumps(json.loads(summary_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
