#!/usr/bin/env python3
# ABOUTME: Batch-runs vlm reviewer over Stage 3 bbox crops/masks from baseline outputs.
# ABOUTME: Supports configurable concurrency and emits throughput/rate-limit summaries.

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from utils.model_pricing import estimate_review_cost_usd
from vlm_reviewer import (
    build_green_overlay,
    build_parts,
    get_git_commit_hash,
    load_prompt,
)


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")
RATE_LIMIT_HINTS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "quota",
    "too many requests",
)
DEFAULT_PROMPT_FILE = "prompts/reviewer.txt"
DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_OUTPUT_ROOT = "auto_reviews_batch"
DEFAULT_BASELINE_DIR = "baseline"

THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ReviewTask:
    case_id: str
    run_id: str
    bbox_id: str
    crop_path: Path
    mask_path: Path
    stage3_metadata_path: Optional[Path]


class GeminiReviewerRunner:
    """Gemini runner with explicit error surfacing for rate-limit analysis."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        use_vertex: bool,
        credentials_path: Optional[str],
        location: str,
        thinking_level: Optional[str],
        include_thoughts: bool,
    ):
        from google import genai
        from google.genai import types

        if use_vertex:
            creds_path = Path(credentials_path) if credentials_path else None
            if creds_path and creds_path.exists():
                with open(creds_path, "r") as f:
                    creds = json.load(f)
                project_id = creds.get("project_id")
                if project_id:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.absolute())
            elif creds_path and not creds_path.exists():
                if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                print(f"Warning: Credentials file not found at {credentials_path}, using env var", file=sys.stderr)
            elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                raise ValueError(
                    "Gemini Vertex mode requires credentials. Provide --gemini-credentials "
                    "or set GOOGLE_APPLICATION_CREDENTIALS."
                )
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        self.client = genai.Client()
        self.model = model
        self.max_retries = max_retries
        self.include_thoughts = include_thoughts

        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else None,
        )
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )
        self.config = types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _parts_to_contents(parts: List[dict]) -> List[Any]:
        contents = []
        for part in parts:
            if part["type"] == "text":
                contents.append(part["text"])
            else:
                img = part["image"]
                if isinstance(img, dict):
                    img = img.get("pil", img)
                contents.append(img)
        return contents

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
        usage = {"prompt_tokens": None, "output_tokens": None, "thoughts_tokens": None, "total_tokens": None}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", None),
                "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                "thoughts_tokens": getattr(usage_meta, "thoughts_token_count", None),
                "total_tokens": getattr(usage_meta, "total_token_count", None),
            }
        return usage

    @staticmethod
    def _extract_finish_reason(response: Any) -> Optional[str]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return None
        return getattr(finish_reason, "name", str(finish_reason))

    def run_with_status(self, parts: List[dict]) -> Dict[str, Any]:
        contents = self._parts_to_contents(parts)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.config,
                )
                result = {
                    "text": (response.text or "").strip(),
                    "thoughts": [],
                    "usage": self._extract_usage(response),
                    "finish_reason": self._extract_finish_reason(response),
                    "error": None,
                    "attempts": attempt + 1,
                }
                if self.include_thoughts and hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if not hasattr(candidate, "content") or not candidate.content:
                            continue
                        for p in getattr(candidate.content, "parts", []) or []:
                            if getattr(p, "thought", False):
                                result["thoughts"].append(getattr(p, "text", ""))
                return result
            except Exception as exc:
                last_error = str(exc)
                if attempt == self.max_retries - 1:
                    return {
                        "text": "",
                        "thoughts": [],
                        "usage": {},
                        "finish_reason": None,
                        "error": last_error,
                        "attempts": attempt + 1,
                    }
                time.sleep(2 ** attempt)

        return {
            "text": "",
            "thoughts": [],
            "usage": {},
            "finish_reason": None,
            "error": last_error or "unknown_error",
            "attempts": self.max_retries,
        }


def parse_json_response_relaxed(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None

    cleaned = raw.replace("```json", "```").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def is_rate_limited_error(error_text: Optional[str]) -> bool:
    if not error_text:
        return False
    msg = error_text.lower()
    return any(hint in msg for hint in RATE_LIMIT_HINTS)


def load_case_allowlist(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    keep: set[str] = set()
    for line in Path(path).read_text().splitlines():
        case_id = line.strip()
        if not case_id or case_id.startswith("#"):
            continue
        keep.add(case_id)
    return keep


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and bool(RUN_ID_RE.match(path.name))


def list_case_run_dirs(case_dir: Path, run_selection: str) -> List[Path]:
    runs = sorted([p for p in case_dir.iterdir() if is_run_dir(p)])
    if run_selection == "latest":
        return runs[-1:] if runs else []
    return runs


def collect_tasks(
    baseline_dir: Path,
    run_selection: str,
    case_allowlist: Optional[set[str]],
    case_patterns: Sequence[str],
    max_cases: Optional[int],
) -> List[ReviewTask]:
    tasks: List[ReviewTask] = []
    selected_cases = 0

    case_dirs = sorted([p for p in baseline_dir.iterdir() if p.is_dir()])
    for case_dir in case_dirs:
        case_id = case_dir.name
        if case_allowlist is not None and case_id not in case_allowlist:
            continue
        if case_patterns and not any(fnmatch.fnmatch(case_id, pat) for pat in case_patterns):
            continue

        run_dirs = list_case_run_dirs(case_dir, run_selection)
        if not run_dirs:
            continue

        selected_cases += 1
        if max_cases is not None and selected_cases > max_cases:
            break

        for run_dir in run_dirs:
            bbox_root = run_dir / "bboxes"
            if not bbox_root.is_dir():
                continue
            for bbox_dir in sorted([p for p in bbox_root.iterdir() if p.is_dir()]):
                stage3_dir = bbox_dir / "stage3"
                crop_path = stage3_dir / "crop.png"
                mask_path = stage3_dir / "mask.png"
                meta_path = stage3_dir / "metadata.json"
                if not (crop_path.is_file() and mask_path.is_file()):
                    continue
                tasks.append(
                    ReviewTask(
                        case_id=case_id,
                        run_id=run_dir.name,
                        bbox_id=bbox_dir.name,
                        crop_path=crop_path,
                        mask_path=mask_path,
                        stage3_metadata_path=meta_path if meta_path.is_file() else None,
                    )
                )
    return tasks


def get_thread_runner(args: argparse.Namespace) -> GeminiReviewerRunner:
    runner = getattr(THREAD_LOCAL, "runner", None)
    if runner is None:
        runner = GeminiReviewerRunner(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            use_vertex=args.gemini_use_vertex,
            credentials_path=args.gemini_credentials,
            location=args.gemini_location,
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        )
        THREAD_LOCAL.runner = runner
    return runner


def output_dir_for_task(reviews_root: Path, task: ReviewTask) -> Path:
    return reviews_root / task.case_id / task.run_id / task.bbox_id


def should_skip_task(task: ReviewTask, reviews_root: Path, overwrite: bool, resume: bool) -> bool:
    if overwrite:
        return False
    if not resume:
        return False
    meta_path = output_dir_for_task(reviews_root, task) / "metadata.json"
    if not meta_path.is_file():
        return False
    try:
        existing = json.loads(meta_path.read_text())
    except Exception:
        return False
    return existing.get("status") == "success"


def build_item_metadata(
    args: argparse.Namespace,
    task: ReviewTask,
    prompt_text: str,
    output_dir: Path,
    elapsed: float,
    response: Dict[str, Any],
    parsed_json: Optional[dict],
    status: str,
    crop_out: Path,
    overlay_out: Path,
    raw_out: Path,
) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    cost_estimate = estimate_review_cost_usd(args.model, usage)
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "case_name": task.case_id,
        "run_id": task.run_id,
        "bbox_name": task.bbox_id,
        "inputs": {
            "crop": str(task.crop_path),
            "mask": str(task.mask_path),
            "stage3_metadata": str(task.stage3_metadata_path) if task.stage3_metadata_path else None,
        },
        "outputs": {
            "run_dir": str(output_dir),
            "crop": str(crop_out),
            "overlay_green50": str(overlay_out),
            "raw_response": str(raw_out),
        },
        "prompt": prompt_text,
        "prompt_file": args.prompt_file,
        "model": args.model,
        "thinking_level": args.thinking_level,
        "include_thoughts": args.include_thoughts,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "gemini_use_vertex": args.gemini_use_vertex,
        "gemini_credentials": args.gemini_credentials,
        "gemini_location": args.gemini_location,
        "overlay_alpha": args.overlay_alpha,
        "mask_threshold": args.mask_threshold,
        "elapsed_seconds": elapsed,
        "attempts": response.get("attempts"),
        "error": response.get("error"),
        "rate_limited": is_rate_limited_error(response.get("error")),
        "usage": usage,
        "finish_reason": response.get("finish_reason"),
        "cost_estimate_usd": cost_estimate,
        "parsed_json": parsed_json,
        "thoughts": response.get("thoughts") or [],
        "git_commit_hash": get_git_commit_hash(),
        "cwd": os.getcwd(),
    }


def run_task(
    args: argparse.Namespace,
    task: ReviewTask,
    prompt_text: str,
    reviews_root: Path,
) -> Dict[str, Any]:
    t0 = time.time()
    output_dir = output_dir_for_task(reviews_root, task)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_img = Image.open(task.crop_path).convert("RGB")
    overlay_img = build_green_overlay(
        crop_img=crop_img,
        mask_path=str(task.mask_path),
        alpha=float(args.overlay_alpha),
        threshold=int(args.mask_threshold),
    )
    if overlay_img.size != crop_img.size:
        overlay_img = overlay_img.resize(crop_img.size, resample=Image.BICUBIC)

    parts = build_parts(prompt_text, crop_img, overlay_img)
    runner = get_thread_runner(args)
    response = runner.run_with_status(parts)

    text = (response.get("text") or "").strip()
    parsed_json = parse_json_response_relaxed(text)
    status = "success" if text else "failed"
    failure_reason = None
    if status != "success":
        failure_reason = "empty_response"
    elif parsed_json is None:
        failure_reason = "parse_failed"
    elapsed = time.time() - t0

    crop_out = output_dir / "crop.png"
    overlay_out = output_dir / "overlay_green50.png"
    raw_out = output_dir / "raw_response.txt"
    meta_out = output_dir / "metadata.json"

    crop_img.save(crop_out)
    overlay_img.save(overlay_out)
    raw_out.write_text(text + ("\n" if text else ""))

    metadata = build_item_metadata(
        args=args,
        task=task,
        prompt_text=prompt_text,
        output_dir=output_dir,
        elapsed=elapsed,
        response=response,
        parsed_json=parsed_json,
        status=status,
        crop_out=crop_out,
        overlay_out=overlay_out,
        raw_out=raw_out,
    )
    meta_out.write_text(json.dumps(metadata, indent=2))

    return {
        "status": status,
        "failure_reason": failure_reason,
        "case_id": task.case_id,
        "run_id": task.run_id,
        "bbox_id": task.bbox_id,
        "crop_path": str(task.crop_path),
        "mask_path": str(task.mask_path),
        "elapsed_seconds": elapsed,
        "rate_limited": is_rate_limited_error(response.get("error")),
        "finish_reason": response.get("finish_reason"),
        "error": response.get("error"),
        "json_parsed": parsed_json is not None,
        "output_dir": str(output_dir),
        "metadata_path": str(meta_out),
        "usage": response.get("usage") or {},
        "cost_estimate_usd": metadata.get("cost_estimate_usd"),
    }


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "run_id",
        "bbox_id",
        "crop_path",
        "mask_path",
        "status",
        "failure_reason",
        "rate_limited",
        "json_parsed",
        "elapsed_seconds",
        "prompt_tokens",
        "output_tokens",
        "thoughts_tokens",
        "total_tokens",
        "estimated_input_cost_usd",
        "estimated_output_cost_usd",
        "estimated_total_cost_usd",
        "finish_reason",
        "error",
        "output_dir",
        "metadata_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage") or {}
            cost = row.get("cost_estimate_usd") or {}
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "run_id": row.get("run_id"),
                    "bbox_id": row.get("bbox_id"),
                    "crop_path": row.get("crop_path"),
                    "mask_path": row.get("mask_path"),
                    "status": row.get("status"),
                    "failure_reason": row.get("failure_reason"),
                    "rate_limited": row.get("rate_limited"),
                    "json_parsed": row.get("json_parsed"),
                    "elapsed_seconds": f"{row.get('elapsed_seconds', 0.0):.3f}",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "thoughts_tokens": usage.get("thoughts_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "estimated_input_cost_usd": cost.get("estimated_input_cost_usd"),
                    "estimated_output_cost_usd": cost.get("estimated_output_cost_usd"),
                    "estimated_total_cost_usd": cost.get("estimated_total_cost_usd"),
                    "finish_reason": row.get("finish_reason"),
                    "error": row.get("error"),
                    "output_dir": row.get("output_dir"),
                    "metadata_path": row.get("metadata_path"),
                }
            )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    total_collected: int,
    total_scheduled: int,
    skipped_existing: int,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "cwd": os.getcwd(),
        "config": vars(args),
        "counts": {
            "tasks_collected": total_collected,
            "tasks_scheduled": total_scheduled,
            "tasks_skipped_existing": skipped_existing,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process baseline Stage 3 bboxes with Gemini reviewer and save per-bbox reviews "
            "for manual validation/agreement analysis."
        )
    )

    io = parser.add_argument_group("Input Selection")
    io.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR, help=f"Baseline root (default: {DEFAULT_BASELINE_DIR})")
    io.add_argument("--run-selection", choices=["latest", "all"], default="latest", help="Use latest run per case or all runs")
    io.add_argument("--cases-file", default=None, help="Optional text file with case IDs to include")
    io.add_argument(
        "--case-pattern",
        action="append",
        default=[],
        help="Optional glob pattern for case IDs (repeatable, e.g. 'anon_8*')",
    )
    io.add_argument("--max-cases", type=int, default=None, help="Cap number of cases after filtering")
    io.add_argument("--max-items", type=int, default=None, help="Cap total bbox review items")
    io.add_argument("--shuffle", action="store_true", help="Shuffle tasks before scheduling")
    io.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle (default: 0)")

    out = parser.add_argument_group("Output")
    out.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})")
    out.add_argument("--batch-name", default=None, help="Batch folder name (default: timestamp)")
    out.add_argument("--resume", dest="resume", action="store_true", help="Skip already-successful items in this batch")
    out.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip existing items")
    out.add_argument("--overwrite", action="store_true", help="Force rerun even if output already exists")
    out.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed items (default: 10)")
    out.add_argument("--dry-run", action="store_true", help="Only enumerate tasks; do not call Gemini")
    parser.set_defaults(resume=True)

    review = parser.add_argument_group("Reviewer")
    review.add_argument("--prompt", default=None, help="Inline prompt text")
    review.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help=f"Prompt file (default: {DEFAULT_PROMPT_FILE})")
    review.add_argument("--overlay-alpha", type=float, default=0.5, help="Generated overlay alpha (default: 0.5)")
    review.add_argument("--mask-threshold", type=int, default=0, help="Mask threshold (mask > threshold, default: 0)")
    review.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=4,
        help="Max concurrent Gemini requests (default: 4)",
    )

    vlm = parser.add_argument_group("Gemini")
    vlm.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    vlm.add_argument("--thinking-level", default="High", help="Thinking level (default: High)")
    vlm.add_argument("--include-thoughts", action="store_true", default=False, help="Request thought summaries")
    vlm.add_argument("--no-include-thoughts", dest="include_thoughts", action="store_false", help="Disable thought summaries")
    vlm.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    vlm.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens (default: 8192)")
    vlm.add_argument("--max-retries", type=int, default=3, help="Max retries per request (default: 3)")
    vlm.add_argument("--gemini-use-vertex", dest="gemini_use_vertex", action="store_true")
    vlm.add_argument("--gemini-no-vertex", dest="gemini_use_vertex", action="store_false")
    vlm.add_argument(
        "--gemini-credentials",
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    vlm.add_argument("--gemini-location", default="global", help="Vertex location (default: global)")
    parser.set_defaults(gemini_use_vertex=True)

    args = parser.parse_args()
    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if args.max_concurrent_requests < 1:
        raise ValueError("--max-concurrent-requests must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("--max-items must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    prompt_text = load_prompt(args.prompt, args.prompt_file)

    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    batch_name = args.batch_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_root).expanduser().resolve() / batch_name
    reviews_root = batch_dir / "reviews"
    batch_dir.mkdir(parents=True, exist_ok=True)

    case_allowlist = load_case_allowlist(args.cases_file)
    tasks = collect_tasks(
        baseline_dir=baseline_dir,
        run_selection=args.run_selection,
        case_allowlist=case_allowlist,
        case_patterns=args.case_pattern,
        max_cases=args.max_cases,
    )

    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(tasks)

    if args.max_items is not None:
        tasks = tasks[: args.max_items]

    scheduled: List[ReviewTask] = []
    skipped_existing = 0
    for task in tasks:
        if should_skip_task(task, reviews_root, overwrite=args.overwrite, resume=args.resume):
            skipped_existing += 1
            continue
        scheduled.append(task)

    write_manifest(
        path=batch_dir / "manifest.json",
        args=args,
        total_collected=len(tasks),
        total_scheduled=len(scheduled),
        skipped_existing=skipped_existing,
    )

    print("=" * 72)
    print("VLM REVIEWER BATCH")
    print("=" * 72)
    print(f"Baseline dir:         {baseline_dir}")
    print(f"Batch dir:            {batch_dir}")
    print(f"Run selection:        {args.run_selection}")
    print(f"Collected items:      {len(tasks)}")
    print(f"Skipped existing:     {skipped_existing}")
    print(f"Scheduled items:      {len(scheduled)}")
    print(f"Max concurrent reqs:  {args.max_concurrent_requests}")
    print(f"Model:                {args.model}")
    print(f"Thinking level:       {args.thinking_level}")
    print(f"Include thoughts:     {args.include_thoughts}")
    print()

    if not scheduled:
        print("No tasks scheduled. Nothing to do.")
        return 0

    if args.dry_run:
        print("Dry-run sample (first 20 tasks):")
        for i, task in enumerate(scheduled[:20], start=1):
            print(
                f"{i:4d}. {task.case_id}/{task.run_id}/{task.bbox_id} "
                f"(crop={task.crop_path}, mask={task.mask_path})"
            )
        if len(scheduled) > 20:
            print(f"... {len(scheduled) - 20} more")
        return 0

    results_path = batch_dir / "results.jsonl"
    results_csv_path = batch_dir / "results.csv"
    summary_path = batch_dir / "summary.json"

    completed_rows: List[Dict[str, Any]] = []
    start = time.time()

    with open(results_path, "a") as results_f:
        with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
            future_to_task = {
                executor.submit(run_task, args, task, prompt_text, reviews_root): task for task in scheduled
            }

            success = 0
            failed = 0
            rate_limited = 0
            parsed_ok = 0

            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "failure_reason": "runner_exception",
                        "case_id": task.case_id,
                        "run_id": task.run_id,
                        "bbox_id": task.bbox_id,
                        "crop_path": str(task.crop_path),
                        "mask_path": str(task.mask_path),
                        "elapsed_seconds": 0.0,
                        "rate_limited": is_rate_limited_error(str(exc)),
                        "finish_reason": None,
                        "error": str(exc),
                        "json_parsed": False,
                        "output_dir": str(output_dir_for_task(reviews_root, task)),
                        "metadata_path": None,
                        "usage": {},
                    }

                if row["status"] == "success":
                    success += 1
                else:
                    failed += 1
                if row["rate_limited"]:
                    rate_limited += 1
                if row["json_parsed"]:
                    parsed_ok += 1

                completed_rows.append(row)
                results_f.write(json.dumps(row) + "\n")
                results_f.flush()

                if idx % args.progress_every == 0 or idx == len(scheduled):
                    elapsed = max(time.time() - start, 1e-6)
                    throughput = idx / elapsed
                    print(
                        f"[{idx}/{len(scheduled)}] success={success} failed={failed} "
                        f"rate_limited={rate_limited} parsed_json={parsed_ok} "
                        f"throughput={throughput:.2f} req/s"
                    )

    write_results_csv(results_csv_path, completed_rows)

    retry_rows = [r for r in completed_rows if r.get("status") != "success"]
    retry_jsonl_path = batch_dir / "openrouter_retry_tasks.jsonl"
    retry_csv_path = batch_dir / "openrouter_retry_tasks.csv"

    with open(retry_jsonl_path, "w") as f:
        for row in retry_rows:
            cost = row.get("cost_estimate_usd") or {}
            payload = {
                "case_id": row.get("case_id"),
                "run_id": row.get("run_id"),
                "bbox_id": row.get("bbox_id"),
                "crop_path": row.get("crop_path"),
                "mask_path": row.get("mask_path"),
                "failure_reason": row.get("failure_reason"),
                "error": row.get("error"),
                "rate_limited": row.get("rate_limited"),
                "finish_reason": row.get("finish_reason"),
                "estimated_total_cost_usd": cost.get("estimated_total_cost_usd"),
                "metadata_path": row.get("metadata_path"),
            }
            f.write(json.dumps(payload) + "\n")

    with open(retry_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "run_id",
                "bbox_id",
                "crop_path",
                "mask_path",
                "failure_reason",
                "error",
                "rate_limited",
                "finish_reason",
                "estimated_total_cost_usd",
                "metadata_path",
            ],
        )
        writer.writeheader()
        for row in retry_rows:
            cost = row.get("cost_estimate_usd") or {}
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "run_id": row.get("run_id"),
                    "bbox_id": row.get("bbox_id"),
                    "crop_path": row.get("crop_path"),
                    "mask_path": row.get("mask_path"),
                    "failure_reason": row.get("failure_reason"),
                    "error": row.get("error"),
                    "rate_limited": row.get("rate_limited"),
                    "finish_reason": row.get("finish_reason"),
                    "estimated_total_cost_usd": cost.get("estimated_total_cost_usd"),
                    "metadata_path": row.get("metadata_path"),
                }
            )

    elapsed_total = max(time.time() - start, 1e-6)
    latencies = sorted([float(r.get("elapsed_seconds", 0.0)) for r in completed_rows])
    estimated_total_cost_usd = 0.0
    estimated_cost_count = 0
    for row in completed_rows:
        cost = row.get("cost_estimate_usd") or {}
        total_cost = cost.get("estimated_total_cost_usd")
        if total_cost is None:
            continue
        try:
            estimated_total_cost_usd += float(total_cost)
            estimated_cost_count += 1
        except Exception:
            continue

    summary = {
        "finished_at": datetime.now().isoformat(),
        "batch_dir": str(batch_dir),
        "results_jsonl": str(results_path),
        "results_csv": str(results_csv_path),
        "total_scheduled": len(scheduled),
        "total_completed": len(completed_rows),
        "success_count": sum(1 for r in completed_rows if r.get("status") == "success"),
        "failed_count": sum(1 for r in completed_rows if r.get("status") != "success"),
        "openrouter_retry_count": len(retry_rows),
        "rate_limited_count": sum(1 for r in completed_rows if r.get("rate_limited")),
        "parsed_json_count": sum(1 for r in completed_rows if r.get("json_parsed")),
        "elapsed_seconds_total": elapsed_total,
        "throughput_req_per_sec": len(completed_rows) / elapsed_total,
        "latency_seconds_p50": percentile(latencies, 0.50),
        "latency_seconds_p90": percentile(latencies, 0.90),
        "latency_seconds_p95": percentile(latencies, 0.95),
        "latency_seconds_max": latencies[-1] if latencies else 0.0,
        "estimated_total_cost_usd": estimated_total_cost_usd,
        "estimated_cost_count": estimated_cost_count,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    print(f"Completed:            {summary['total_completed']}/{summary['total_scheduled']}")
    print(f"Success:              {summary['success_count']}")
    print(f"Failed:               {summary['failed_count']}")
    print(f"OpenRouter retries:   {summary['openrouter_retry_count']}")
    print(f"Rate-limited:         {summary['rate_limited_count']}")
    print(f"Parsed JSON:          {summary['parsed_json_count']}")
    print(f"Elapsed (s):          {summary['elapsed_seconds_total']:.2f}")
    print(f"Throughput (req/s):   {summary['throughput_req_per_sec']:.2f}")
    print(f"Latency p50/p90/p95:  {summary['latency_seconds_p50']:.2f}/"
          f"{summary['latency_seconds_p90']:.2f}/{summary['latency_seconds_p95']:.2f}")
    print(f"Latency max (s):      {summary['latency_seconds_max']:.2f}")
    print(f"Est. total cost (USD): ${summary['estimated_total_cost_usd']:.6f} ({summary['estimated_cost_count']} items)")
    print()
    print(f"Manifest:             {batch_dir / 'manifest.json'}")
    print(f"Results JSONL:        {results_path}")
    print(f"Results CSV:          {results_csv_path}")
    print(f"Retry JSONL:          {retry_jsonl_path}")
    print(f"Retry CSV:            {retry_csv_path}")
    print(f"Summary JSON:         {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
