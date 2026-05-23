#!/usr/bin/env python3
"""Review whether one raw detector orientation missed visible tissue-like signal."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import textwrap
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runs" / "stage1_detector_pilot_v1" / "review_packet" / "all_detections_manifest.csv"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "raw_rot0_coverage_review_missing_cases_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
PROMPT_VERSION = "stage1_raw_orientation_coverage_review_v1_2026-05-23"

PROMPT = """\
You are reviewing a tissue-detection overlay on a whole-slide thumbnail.

Inputs:
- Image 1 is the raw detector orientation overlay. It may contain zero bounding boxes.
- The text gives the raw detector orientation and bbox count.

Task:
Decide whether the detector missed visible tissue-like foreground signal in the thumbnail.

Use only visual object-detection judgment:
- tissue-like signal means visible tissue-colored foreground material or tissue-like core fragments at thumbnail scale.
- non-tissue/noise means glass edges, dust, pen marks, bubbles, smudges, debris, or blank background.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.
- Do not output refined coordinates.

Return only one JSON object with this exact shape:
{
  "coverage_review": {
    "detected_box_count": 0,
    "visible_tissue_like_signal": true,
    "missed_detection": true,
    "confidence": "high",
    "reasoning": "short visual explanation"
  }
}

Allowed confidence values: low, medium, high.
Set missed_detection=true if visible tissue-like signal is present but not covered by any bbox.
"""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: str,
) -> int:
    x, y = xy
    for para in str(text or "").splitlines() or [""]:
        for line in textwrap.wrap(para, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += 20
    return y


def _thumb(path: Path, max_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size)
    return image


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_text": text}


def _image_to_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No API key found. Set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return key


def _chat_with_image(args: argparse.Namespace, prompt_text: str, image_path: Path) -> tuple[str, dict[str, Any], str]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        (args.api_base or DEFAULT_API_BASE).rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {_api_key(args)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    raw = message.get("content", "") if isinstance(message, dict) else ""
    return raw, data.get("usage", {}) or {}, str(data.get("model", ""))


def _raw_orientation_items(bboxes_json_path: Path, rotation: int) -> tuple[int, str]:
    payload = json.loads(bboxes_json_path.read_text())
    per_orientation = payload.get("per_orientation_raw")
    if isinstance(per_orientation, dict):
        items = per_orientation.get(str(rotation))
        if isinstance(items, list):
            return len(items), ""
        return 0, f"missing_raw_rotation_{rotation}"
    meta_path = bboxes_json_path.parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        counts = meta.get("per_orientation_bbox_counts")
        if isinstance(counts, dict) and str(rotation) in counts:
            return int(counts.get(str(rotation)) or 0), "from_metadata_counts_only"
    return 0, "missing_per_orientation_raw"


def _make_overlay(thumbnail_path: Path, output_path: Path, rotation: int, bbox_count: int) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    text = f"raw rot{rotation}: {bbox_count} boxes"
    bbox = draw.textbbox((12, 12), text, font=font)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill="white", outline="black")
    draw.text((12, 12), text, font=font, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _case_display(row: dict[str, str]) -> str:
    return (
        f"{row['index']}/100 | {row['stain']} | {row['case_id']} | "
        f"{row['Anon_Path_ID']} | {Path(row['wsi_path']).name}"
    )


def _selected_rows(manifest: Path, indices: list[int]) -> list[dict[str, str]]:
    rows = _read_rows(manifest)
    by_index = {int(row["index"]): row for row in rows}
    missing = [idx for idx in indices if idx not in by_index]
    if missing:
        raise SystemExit(f"Missing manifest indices: {missing}")
    return [by_index[idx] for idx in indices]


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in _selected_rows(args.manifest.resolve(), args.indices):
        thumbnail_path = Path(row["thumbnail_path"])
        bboxes_json_path = Path(row["bboxes_json_path"])
        bbox_count, raw_note = _raw_orientation_items(bboxes_json_path, args.rotation)
        case_slug = _safe_slug(f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}")
        overlay_path = args.output_root.resolve() / "overlays" / f"{case_slug}_rot{args.rotation}_coverage_overlay.png"
        _make_overlay(thumbnail_path, overlay_path, args.rotation, bbox_count)
        prompt_text = (
            PROMPT
            + "\n\nCase:\n"
            + _case_display(row)
            + f"\n\nReviewed raw detector orientation: rot{args.rotation}\n"
            + f"Detected bbox count for this orientation: {bbox_count}\n"
            + f"Raw detector note: {raw_note or 'per_orientation_raw'}\n"
        )
        tasks.append(
            {
                "task_id": f"raw_orientation_coverage_{int(row['index']):03d}_rot{args.rotation}",
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "case_display": _case_display(row),
                "manifest_row": row,
                "rotation": args.rotation,
                "detected_box_count": bbox_count,
                "raw_note": raw_note,
                "thumbnail_path": str(thumbnail_path),
                "overlay_path": str(overlay_path),
                "bboxes_json_path": str(bboxes_json_path),
                "prompt": prompt_text,
                "created_at": _timestamp(),
            }
        )
    return tasks


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    tasks = build_tasks(args)
    tasks_path = args.output_root / "tasks" / f"raw_orientation_coverage_rot{args.rotation}_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks), "tasks_jsonl": str(tasks_path)}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for task in tasks:
        record = {
            "task_id": task["task_id"],
            "case_display": task["case_display"],
            "prompt_version": task["prompt_version"],
            "model": args.model,
            "rotation": task["rotation"],
            "detected_box_count": task["detected_box_count"],
            "raw_note": task["raw_note"],
            "thumbnail_path": task["thumbnail_path"],
            "overlay_path": task["overlay_path"],
            "bboxes_json_path": task["bboxes_json_path"],
            "created_at": _timestamp(),
            "error": "",
        }
        try:
            raw, usage, response_model = _chat_with_image(args, task["prompt"], Path(task["overlay_path"]))
            record["raw_response"] = raw
            record["parsed_response"] = _extract_json_object(raw)
            record["usage"] = usage
            record["response_model"] = response_model
        except Exception as exc:
            record["raw_response"] = ""
            record["parsed_response"] = {}
            record["usage"] = {}
            record["response_model"] = ""
            record["error"] = repr(exc)
        results.append(record)

    results_path = args.output_root / "reviews" / f"raw_orientation_coverage_rot{args.rotation}_results.jsonl"
    _write_jsonl(results_path, results)
    write_summary(args, results, tasks_path, results_path)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "results_jsonl": str(results_path),
                "summary_csv": str(args.output_root / "summary" / f"raw_orientation_coverage_rot{args.rotation}.csv"),
                "pdf": str(args.output_root / "visuals" / f"raw_orientation_coverage_rot{args.rotation}.pdf"),
            },
            indent=2,
        )
    )
    return 0


def write_summary(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    tasks_path: Path,
    results_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        review = parsed.get("coverage_review") if isinstance(parsed.get("coverage_review"), dict) else {}
        rows.append(
            {
                "task_id": result.get("task_id", ""),
                "case_display": result.get("case_display", ""),
                "rotation": result.get("rotation", ""),
                "detected_box_count": result.get("detected_box_count", ""),
                "raw_note": result.get("raw_note", ""),
                "parse_ok": bool(review),
                "error": result.get("error", ""),
                "visible_tissue_like_signal": review.get("visible_tissue_like_signal", ""),
                "missed_detection": review.get("missed_detection", ""),
                "confidence": review.get("confidence", ""),
                "reasoning": review.get("reasoning", ""),
                "overlay_path": result.get("overlay_path", ""),
            }
        )
    summary_csv = args.output_root / "summary" / f"raw_orientation_coverage_rot{args.rotation}.csv"
    _write_csv(
        summary_csv,
        rows,
        [
            "task_id",
            "case_display",
            "rotation",
            "detected_box_count",
            "raw_note",
            "parse_ok",
            "error",
            "visible_tissue_like_signal",
            "missed_detection",
            "confidence",
            "reasoning",
            "overlay_path",
        ],
    )
    _write_json(
        args.output_root / "summary" / f"raw_orientation_coverage_rot{args.rotation}_summary.json",
        {
            "results": len(results),
            "parse_ok": sum(1 for row in rows if row["parse_ok"]),
            "errors": sum(1 for row in rows if row["error"]),
            "missed_detection": sum(1 for row in rows if str(row["missed_detection"]).lower() == "true"),
            "visible_tissue_like_signal": sum(
                1 for row in rows if str(row["visible_tissue_like_signal"]).lower() == "true"
            ),
        },
    )
    write_pdf(args, rows)
    write_reproduction(args, tasks_path, results_path)


def write_pdf(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    for row in rows:
        page = Image.new("RGB", (1800, 1700), "white")
        draw = ImageDraw.Draw(page)
        y = 30
        draw.text((40, y), row["case_display"], font=title_font, fill="black")
        y += 45
        draw.text(
            (40, y),
            f"rot{row['rotation']} boxes={row['detected_box_count']} missed={row['missed_detection']} "
            f"visible_tissue={row['visible_tissue_like_signal']} confidence={row['confidence']}",
            font=body_font,
            fill="black",
        )
        y += 34
        y = _draw_wrapped(draw, (40, y), row.get("reasoning", ""), small_font, 150, "#111111")
        y += 20
        overlay = _thumb(Path(row["overlay_path"]), (1680, 900))
        page.paste(overlay, (40, y))
        pages.append(page)
    pdf_path = args.output_root / "visuals" / f"raw_orientation_coverage_rot{args.rotation}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def write_reproduction(args: argparse.Namespace, tasks_path: Path, results_path: Path) -> None:
    text = f"""\
Stage 1 raw-orientation coverage review
=======================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions via urllib
Manifest: {args.manifest.resolve()}
Detector orientation reviewed: rot{args.rotation}
Task indices: {','.join(str(i) for i in args.indices)}

Command:
python scripts/stage1_raw_orientation_coverage_review.py \\
  --manifest {args.manifest.resolve()} \\
  --output-root {args.output_root.resolve()} \\
  --indices {','.join(str(i) for i in args.indices)} \\
  --rotation {args.rotation} \\
  --model {args.model} \\
  --temperature {args.temperature}

Outputs:
- Tasks: {tasks_path}
- Raw/parsed results: {results_path}
- Summary CSV: {args.output_root / 'summary' / f'raw_orientation_coverage_rot{args.rotation}.csv'}
- PDF: {args.output_root / 'visuals' / f'raw_orientation_coverage_rot{args.rotation}.pdf'}
- Overlays: {args.output_root / 'overlays'}

Notes:
- This review asks only whether the selected raw detector orientation missed visible tissue-like signal.
- It does not output refined coordinates and does not use pathology/control-tissue semantics.
"""
    (args.output_root / "reproduction.txt").write_text(text)


def parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
