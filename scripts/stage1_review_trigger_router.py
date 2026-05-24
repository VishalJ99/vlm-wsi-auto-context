#!/usr/bin/env python3
"""Route Stage 1 thumbnail reviewer text into second-pass trigger decisions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _api_settings,
    _extract_json_payload,
    _repo_git_commit,
    _timestamp,
    _write_csv,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REVIEW_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_pilot100_short_reviewer_high_thinking_v1"
)
DEFAULT_INPUT_REVIEWS = DEFAULT_SOURCE_REVIEW_ROOT / "reviews" / "edge_review_results.jsonl"
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_REVIEW_ROOT / "stage2b_trigger_router_low_thinking_v1"
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts" / "stage1_detector_oracle" / "stage2b_review_trigger_router.txt"


def _prompt_version(prompt_file: Path) -> str:
    return f"{prompt_file.stem}_2026-05-24"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _review_text(record: dict[str, Any]) -> str:
    parsed = record.get("parsed_response")
    if isinstance(parsed, dict) and isinstance(parsed.get("raw_text"), str):
        return parsed["raw_text"].strip()
    return str(record.get("raw_response") or "").strip()


def _chat_text(
    *,
    model: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key: str,
    reasoning_effort: str | None,
) -> tuple[str, dict[str, Any], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    request_kwargs: dict[str, Any] = {}
    if reasoning_effort and reasoning_effort != "none":
        request_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        **request_kwargs,
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    response_model = getattr(response, "model", "")
    return raw, usage, response_model


def _parse_router_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    lowered = raw.lower()
    normalized = lowered.lstrip(" \t\r\n*#:-.0123456789)%(")
    binary_answer: bool | None = None
    if re.match(r"^\s*yes\b", normalized):
        binary_answer = True
    elif re.match(r"^\s*no\b", normalized):
        binary_answer = False
    else:
        answer_match = re.search(r'"answer"\s*:\s*"(yes|no)"', raw, flags=re.IGNORECASE)
        if answer_match:
            binary_answer = answer_match.group(1).lower() == "yes"

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    json_text = fenced.group(1).strip() if fenced else raw
    try:
        payload = json.loads(json_text)
    except Exception:
        try:
            payload = _extract_json_payload(raw)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    trigger = payload.get("non_minor_detection_failure", payload.get("trigger_refinement"))
    if not isinstance(trigger, bool):
        if isinstance(payload.get("answer"), str):
            answer = payload["answer"].strip().lower()
            if answer.startswith("yes"):
                trigger = True
            elif answer.startswith("no"):
                trigger = False
        elif "trigger_refinement" in lowered:
            trigger = "true" in lowered
        elif "non-minor failure" in lowered and "no non-minor failure" not in lowered:
            trigger = True
        else:
            trigger = binary_answer
    error_types = payload.get("error_types")
    if not isinstance(error_types, list):
        error_types = []
    answer = payload.get("answer")
    if isinstance(answer, str):
        answer_value = answer.strip().lower()
    elif isinstance(trigger, bool):
        answer_value = "yes" if trigger else "no"
    else:
        answer_value = ""
    severity = str(payload.get("severity") or "").strip()
    if not severity:
        severity = "non_minor" if trigger is True else "none" if trigger is False else "uncertain"
    rationale = str(payload.get("rationale") or payload.get("reasoning") or "").strip()
    justification = str(payload.get("justification") or "").strip()
    if not rationale:
        rationale = justification
    if not rationale and binary_answer is not None:
        rationale = raw
    return {
        "answer": answer_value,
        "non_minor_detection_failure": trigger,
        "trigger_refinement": trigger,
        "severity": severity,
        "error_types": [str(item) for item in error_types],
        "justification": justification or rationale,
        "rationale": rationale,
    }


def _call_router(
    record: dict[str, Any],
    args: argparse.Namespace,
    prompt_template: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    review_text = _review_text(record)
    prompt = (
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
    output = {
        "task_id": record.get("task_id", ""),
        "case_index": record.get("case_index", ""),
        "case_display": record.get("case_display", ""),
        "source_review_model": record.get("model", ""),
        "source_review_reasoning_effort": record.get("reasoning_effort", ""),
        "source_review_prompt_version": record.get("prompt_version", ""),
        "source_review_text": review_text,
        "prompt_version": _prompt_version(args.prompt_file),
        "prompt_file": str(args.prompt_file),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "created_at": _timestamp(),
        "raw_response": "",
        "parsed_response": {},
        "trigger_refinement": "",
        "severity": "",
        "error_types": [],
        "rationale": "",
        "usage": {},
        "response_model": "",
        "error": "",
    }
    try:
        raw, usage, response_model = _chat_text(
            model=args.model,
            prompt_text=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        parsed = _parse_router_response(raw)
        output["raw_response"] = raw
        output["parsed_response"] = parsed
        output["non_minor_detection_failure"] = parsed["non_minor_detection_failure"]
        output["trigger_refinement"] = parsed["trigger_refinement"]
        output["severity"] = parsed["severity"]
        output["error_types"] = parsed["error_types"]
        output["rationale"] = parsed["rationale"]
        output["usage"] = usage
        output["response_model"] = response_model
    except Exception as exc:
        output["error"] = repr(exc)
    return output


def _run_parallel(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    prompt_template: str,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    if args.max_concurrent <= 1:
        results = [_call_router(record, args, prompt_template, base_url, api_key) for record in records]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = [
                pool.submit(_call_router, record, args, prompt_template, base_url, api_key)
                for record in records
            ]
            for future in as_completed(futures):
                results.append(future.result())
    return sorted(results, key=lambda row: int(row.get("case_index") or 0))


def _csv_rows(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(
            {
                "case_index": row.get("case_index", ""),
                "case_display": row.get("case_display", ""),
                "non_minor_detection_failure": row.get("non_minor_detection_failure", row.get("trigger_refinement", "")),
                "trigger_refinement": row.get("trigger_refinement", ""),
                "severity": row.get("severity", ""),
                "error_types": ";".join(row.get("error_types", []) or []),
                "rationale": row.get("rationale", ""),
                "error": row.get("error", ""),
                "source_review_excerpt": row.get("source_review_text", "")[:800],
                "raw_router_response": row.get("raw_response", "")[:800],
            }
        )
    return rows


def _summary(results: list[dict[str, Any]], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    trigger_counts = Counter(str(row.get("trigger_refinement")) for row in results)
    non_minor_counts = Counter(
        str(row.get("non_minor_detection_failure", row.get("trigger_refinement"))) for row in results
    )
    severity_counts = Counter(str(row.get("severity")) for row in results)
    error_type_counts: Counter[str] = Counter()
    for row in results:
        for error_type in row.get("error_types", []) or []:
            error_type_counts[str(error_type)] += 1
    usage_cost = 0.0
    total_tokens = 0
    for row in results:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        total_tokens += int(usage.get("total_tokens") or 0)
        if "cost" in usage:
            usage_cost += float(usage.get("cost") or 0.0)
    return {
        "created_at": _timestamp(),
        "git_commit": _repo_git_commit(),
        "ticket": "PER-207",
        "input_reviews": str(args.input_reviews.resolve()),
        "output_root": str(output_root.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_version": _prompt_version(args.prompt_file),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "cases": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "non_minor_detection_failure_counts": dict(non_minor_counts),
        "trigger_counts": dict(trigger_counts),
        "severity_counts": dict(severity_counts),
        "error_type_counts": dict(error_type_counts),
        "total_tokens": total_tokens,
        "known_usage_cost_if_reported": usage_cost,
    }


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    results_path: Path,
    summary_path: Path,
    csv_path: Path,
    prompt_text: str,
) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage1_review_trigger_router.py",
            "--input-reviews",
            str(args.input_reviews.resolve()),
            "--output-root",
            str(output_root.resolve()),
            "--prompt-file",
            str(args.prompt_file.resolve()),
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
Stage 2b review-trigger router
==============================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Source 2a reviews: {args.input_reviews.resolve()}
Prompt version: {_prompt_version(args.prompt_file)}
Prompt file: {args.prompt_file.resolve()}
Model: {args.model}
Reasoning effort: {args.reasoning_effort}
Backend: OpenRouter-compatible chat completions

Command:
{command}

Outputs:
- Results JSONL: {results_path}
- Case summary CSV: {csv_path}
- Summary JSON: {summary_path}

Prompt text:
{prompt_text.strip()}
"""
    (output_root / "reproduction.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-reviews", type=Path, default=DEFAULT_INPUT_REVIEWS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    args = parser.parse_args()

    if not args.input_reviews.exists():
        raise SystemExit(f"Input reviews not found: {args.input_reviews}")
    if not args.prompt_file.exists():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reviews").mkdir(parents=True, exist_ok=True)
    (output_root / "summary").mkdir(parents=True, exist_ok=True)

    prompt_text = args.prompt_file.read_text()
    records = _read_jsonl(args.input_reviews)
    base_url, api_key = _api_settings(args)
    results = _run_parallel(records, args, prompt_text, base_url, api_key)

    results_path = output_root / "reviews" / "stage2b_trigger_router_results.jsonl"
    csv_path = output_root / "summary" / "stage2b_trigger_router_cases.csv"
    summary_path = output_root / "summary" / "stage2b_trigger_router_summary.json"

    _write_jsonl(results_path, results)
    _write_csv(
        csv_path,
        _csv_rows(results),
        [
            "case_index",
            "case_display",
            "non_minor_detection_failure",
            "trigger_refinement",
            "severity",
            "error_types",
            "rationale",
            "error",
            "source_review_excerpt",
            "raw_router_response",
        ],
    )
    _write_json(summary_path, _summary(results, args, output_root))
    _write_reproduction(output_root, args, results_path, summary_path, csv_path, prompt_text)

    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
