#!/usr/bin/env python3
"""Run and summarize the Stage 1 high-recall detector prompt over the pilot set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    DEFAULT_MODEL,
    _draw_redetect_overlay,
    _draw_wrapped,
    _font,
    _load_raw_orientation_bboxes,
    _safe_slug,
    _thumb,
    _write_csv,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runs" / "stage1_detector_pilot_v1" / "worklists" / "pilot_100.csv"
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "stage1_high_recall_potential_tissue_candidates.txt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1"
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _case_display(row: dict[str, str]) -> str:
    return (
        f"{int(row['selection_rank']) * 5 - 5 + _stain_order(row.get('stain', ''))}/100 | "
        f"{row.get('stain', '')} | {row.get('case_id', '')} | {row.get('Anon_Path_ID', '')} | "
        f"{Path(row.get('wsi_path', '')).name}"
    )


def _stain_order(stain: str) -> int:
    order = {"H&E": 1, "PAS": 2, "JONES": 3, "EVG": 4, "SV40": 5}
    return order.get(stain, 0)


def _selected_rows(manifest: Path, indices: list[int]) -> list[dict[str, str]]:
    rows = _read_csv(manifest)
    if not indices:
        return rows
    wanted = set(indices)
    selected: list[dict[str, str]] = []
    for position, row in enumerate(rows, start=1):
        if position in wanted:
            selected.append(row)
    return selected


def _case_slug(position: int, row: dict[str, str]) -> str:
    return _safe_slug(f"{position:03d}_{Path(row['wsi_path']).stem}")


def _detector_command(args: argparse.Namespace, row: dict[str, str], output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "detect_foreground_regions_from_wsi_thumbnail.py"),
        "--wsi",
        row["wsi_path"],
        "--wsi-reader",
        "auto",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--max-dim",
        str(args.max_dim),
        "--openrouter-url",
        args.api_base,
        "--coord-order",
        "auto",
        "--padding",
        str(args.padding),
        "--merge-overlap-threshold",
        str(args.merge_overlap_threshold),
        "--rotations",
        str(args.rotation),
        "--prompt",
        str(args.prompt.resolve()),
        "--output-dir",
        str(output_dir),
        "--skip-dvc-check",
    ]
    if not args.no_save_intermediate:
        cmd.append("--save-intermediate")
    return cmd


def _run_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(task["stage1_dir"])
    log_path = Path(task["log_path"])
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.force:
        return {**task, "status": "skipped_existing", "returncode": 0, "error": ""}

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _detector_command(args, task["manifest_row"], output_dir)
    with log_path.open("w") as log_fh:
        log_fh.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    status = "ok" if proc.returncode == 0 else "error"
    return {
        **task,
        "status": status,
        "returncode": proc.returncode,
        "error": "" if proc.returncode == 0 else f"detector exited {proc.returncode}",
    }


def _build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _selected_rows(args.manifest.resolve(), args.indices)
    all_rows = _read_csv(args.manifest.resolve())
    task_rows: list[dict[str, Any]] = []
    for row in rows:
        position = all_rows.index(row) + 1
        slug = _case_slug(position, row)
        stage1_dir = args.output_root / "stage1_runs" / slug / "stage1"
        task_rows.append(
            {
                "case_index": position,
                "case_display": _case_display(row),
                "manifest_row": row,
                "stage1_dir": str(stage1_dir),
                "log_path": str(args.output_root / "logs" / f"{slug}.log"),
            }
        )
    return task_rows


def _summarize_case(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    stage1_dir = Path(task["stage1_dir"])
    thumbnail_path = stage1_dir / "thumbnail.png"
    final_overlay_path = stage1_dir / "bbox_overlay.png"
    bboxes_path = stage1_dir / "bboxes.json"
    metadata_path = stage1_dir / "metadata.json"
    raw_response_path = stage1_dir / "vlm_responses" / f"rot{args.rotation}_response.txt"
    raw_overlay_path = args.output_root / "raw_overlays" / f"{_safe_slug(task['case_display'])}_raw_overlay.png"

    row: dict[str, Any] = {
        "case_index": task["case_index"],
        "case_display": task["case_display"],
        "status": task.get("status", ""),
        "returncode": task.get("returncode", ""),
        "error": task.get("error", ""),
        "stage1_dir": str(stage1_dir),
        "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else "",
        "raw_overlay_path": "",
        "final_overlay_path": str(final_overlay_path) if final_overlay_path.exists() else "",
        "bboxes_json_path": str(bboxes_path) if bboxes_path.exists() else "",
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "raw_response_path": str(raw_response_path) if raw_response_path.exists() else "",
        "raw_rot0_count": 0,
        "final_count": 0,
        "raw_parse_note": "",
        "prompt_matches": "",
    }
    if not thumbnail_path.exists():
        row["error"] = row["error"] or "missing_thumbnail"
        return row

    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size

    if bboxes_path.exists():
        try:
            raw_bboxes, raw_note = _load_raw_orientation_bboxes(bboxes_path, thumbnail_size, args.rotation)
            row["raw_rot0_count"] = len(raw_bboxes)
            row["raw_parse_note"] = raw_note
            if raw_bboxes:
                _draw_redetect_overlay(thumbnail_path, raw_bboxes, raw_overlay_path)
                row["raw_overlay_path"] = str(raw_overlay_path)
            payload = _read_json(bboxes_path)
            detected = payload.get("detected_regions") if isinstance(payload.get("detected_regions"), list) else []
            row["final_count"] = len(detected)
        except Exception as exc:
            row["error"] = row["error"] or repr(exc)
    if metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
            row["prompt_matches"] = str(metadata.get("prompt", "").strip() == args.prompt.read_text().strip()).lower()
        except Exception:
            row["prompt_matches"] = "false"
    return row


def _write_pdf(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    prompt_text = args.prompt.read_text().strip()
    pages: list[Image.Image] = []
    title_font = _font(30)
    body_font = _font(18)
    small_font = _font(14)

    title = Image.new("RGB", (2200, 2600), "white")
    draw = ImageDraw.Draw(title)
    y = 45
    draw.text((45, y), "Stage 1 High-Recall Pilot Rot0", font=title_font, fill="black")
    y += 50
    meta = (
        f"cases={len(rows)} | model={args.model} | rotation=rot{args.rotation} | "
        f"prompt={args.prompt.resolve()}"
    )
    y = _draw_wrapped(draw, (45, y), meta, body_font, 150, "#222222") + 20
    draw.text((45, y), "Prompt", font=body_font, fill="black")
    y += 32
    _draw_wrapped(draw, (65, y), prompt_text, small_font, 170, "#111111")
    pages.append(title)

    for row in rows:
        page = Image.new("RGB", (2200, 1600), "white")
        draw = ImageDraw.Draw(page)
        y = 35
        draw.text((45, y), row["case_display"], font=title_font, fill="black")
        y += 46
        header = (
            f"status={row.get('status')} | raw rot{args.rotation} boxes={row.get('raw_rot0_count')} | "
            f"final boxes={row.get('final_count')} | prompt_matches={row.get('prompt_matches')}"
        )
        draw.text((45, y), header, font=body_font, fill="black")
        y += 35
        if row.get("error"):
            y = _draw_wrapped(draw, (45, y), f"error: {row['error']}", small_font, 170, "#aa0000")
        else:
            y += 12

        source = _thumb(Path(row["thumbnail_path"]), (660, 420)) if row.get("thumbnail_path") else Image.new("RGB", (660, 420), "#f7f7f7")
        raw = _thumb(Path(row["raw_overlay_path"]), (660, 420)) if row.get("raw_overlay_path") else Image.new("RGB", (660, 420), "#f7f7f7")
        final = _thumb(Path(row["final_overlay_path"]), (660, 420)) if row.get("final_overlay_path") else Image.new("RGB", (660, 420), "#f7f7f7")
        for x, label, image in (
            (45, "Source thumbnail", source),
            (770, "Raw rot0 output overlay", raw),
            (1495, "Final padded/merged overlay", final),
        ):
            draw.text((x, y), label, font=body_font, fill="black")
            page.paste(image, (x, y + 30))
        pages.append(page)

    pdf_path = output_root / "visuals" / "high_recall_stage1_rot0_pilot.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_reproduction(output_root: Path, args: argparse.Namespace, tasks_path: Path, cases_path: Path) -> None:
    command = [
        "python",
        "scripts/stage1_high_recall_pilot.py",
        "--manifest",
        str(args.manifest.resolve()),
        "--prompt",
        str(args.prompt.resolve()),
        "--output-root",
        str(output_root),
        "--indices",
        ",".join(str(i) for i in args.indices),
        "--rotation",
        str(args.rotation),
        "--model",
        args.model,
        "--max-dim",
        str(args.max_dim),
        "--max-concurrent",
        str(args.max_concurrent),
    ]
    text = f"""\
Stage 1 high-recall detector pilot
==================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207

Command:
{" ".join(shlex.quote(part) for part in command)}

Prompt file:
{args.prompt.resolve()}

Prompt text:
{args.prompt.read_text().strip()}

Outputs:
- Tasks JSONL: {tasks_path}
- Case summary CSV: {cases_path}
- Summary JSON: {output_root / 'summary' / 'high_recall_stage1_summary.json'}
- PDF: {output_root / 'visuals' / 'high_recall_stage1_rot0_pilot.pdf'}
- Stage 1 outputs: {output_root / 'stage1_runs'}
- Raw numeric overlays: {output_root / 'raw_overlays'}
- Logs: {output_root / 'logs'}

Notes:
- Each case uses a single raw detector orientation: rot{args.rotation}.
- Rendered overlays use compact numeric labels only. Raw VLM labels, if any, remain in bboxes.json and raw responses for provenance.
- Stage 1 padding/merge is still applied after the raw high-recall detector output.
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    args.prompt = args.prompt.resolve()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest does not exist: {args.manifest}")
    if not args.prompt.exists():
        raise SystemExit(f"Prompt does not exist: {args.prompt}")

    tasks = _build_tasks(args)
    tasks_path = args.output_root / "tasks" / "high_recall_stage1_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)

    if not args.summarize_only:
        if args.max_concurrent <= 1:
            completed = [_run_one(task, args) for task in tasks]
        else:
            completed = []
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
                futures = [pool.submit(_run_one, task, args) for task in tasks]
                for future in as_completed(futures):
                    completed.append(future.result())
            completed.sort(key=lambda row: row["case_index"])
        _write_jsonl(args.output_root / "tasks" / "high_recall_stage1_completed.jsonl", completed)
    else:
        completed_path = args.output_root / "tasks" / "high_recall_stage1_completed.jsonl"
        completed = [
            json.loads(line)
            for line in completed_path.read_text().splitlines()
            if line.strip()
        ] if completed_path.exists() else tasks

    case_rows = [_summarize_case(task, args) for task in completed]
    cases_path = args.output_root / "summary" / "high_recall_stage1_cases.csv"
    _write_csv(cases_path, case_rows, list(case_rows[0].keys()) if case_rows else [])
    summary = {
        "created_at": _timestamp(),
        "git_commit": _repo_git_commit(),
        "ticket": "PER-207",
        "manifest": str(args.manifest.resolve()),
        "prompt": str(args.prompt.resolve()),
        "output_root": str(args.output_root),
        "cases": len(case_rows),
        "errors": sum(1 for row in case_rows if row.get("error")),
        "raw_zero_box_cases": sum(1 for row in case_rows if int(row.get("raw_rot0_count") or 0) == 0),
        "total_raw_rot0_boxes": sum(int(row.get("raw_rot0_count") or 0) for row in case_rows),
        "total_final_boxes": sum(int(row.get("final_count") or 0) for row in case_rows),
        "prompt_mismatch_cases": sum(1 for row in case_rows if row.get("prompt_matches") != "true"),
    }
    _write_json(args.output_root / "summary" / "high_recall_stage1_summary.json", summary)
    _write_pdf(args.output_root, args, case_rows)
    _write_reproduction(args.output_root, args, tasks_path, cases_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, default=list(range(1, 101)))
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--backend", choices=["openrouter", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--merge-overlap-threshold", type=float, default=0.2)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-save-intermediate", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
