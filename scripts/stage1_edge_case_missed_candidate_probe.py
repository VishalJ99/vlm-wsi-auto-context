#!/usr/bin/env python3
"""Point-blank probe for missed potential tissue candidates in Stage 1 overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    _api_settings,
    _case_display,
    _chat_with_images,
    _draw_wrapped,
    _extract_json_object,
    _font,
    _repo_git_commit,
    _safe_slug,
    _selected_rows,
    _thumb,
    _timestamp,
    _write_json,
    _write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "edge_case_probes"
    / "missed_candidate_point_blank_v1"
)
PROMPT_VERSION = "stage1_missed_candidate_point_blank_v1_2026-05-24"

PROMPT = """\
You are checking one tissue-detection overlay on a whole-slide thumbnail.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the same thumbnail with detection bounding boxes drawn on it.

Question:
Did the detection overlay miss any visible potential tissue candidates?

Definition:
- A potential tissue candidate is any visible foreground region that looks plausibly tissue-like at thumbnail scale, including fragmented tissue-like clumps or strips.
- Count it as missed if it is outside all boxes or substantially uncovered by the boxes.
- Ignore blank background, glass edges, pen marks, bubbles, dust, and obvious non-tissue debris.

Keep this task simple:
- Do not review per-bbox tightness.
- Do not propose refined boxes.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.

Return only one JSON object with this exact shape:
{
  "missed_candidate_review": {
    "missed_potential_tissue_candidates": true,
    "confidence": "high",
    "reasoning": "short visual explanation",
    "rough_locations": ["short location descriptions"]
  }
}

Allowed confidence values: low, medium, high.
"""


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_index",
        "case_display",
        "missed_potential_tissue_candidates",
        "confidence",
        "reasoning",
        "rough_locations",
        "error",
        "thumbnail_path",
        "overlay_path",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _result_review(parsed: dict[str, Any]) -> dict[str, Any]:
    review = parsed.get("missed_candidate_review")
    return review if isinstance(review, dict) else {}


def _overlay_path(row: dict[str, str], raw_overlay_root: Path | None, rotation: int) -> Path:
    if raw_overlay_root is None:
        return Path(row["overlay_path"])
    case_slug = _safe_slug(f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}")
    return raw_overlay_root / f"{case_slug}_rot{rotation}_raw_overlay.png"


def _call_case(
    row: dict[str, str],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    thumbnail_path = Path(row["thumbnail_path"])
    overlay_path = _overlay_path(row, args.raw_overlay_root, args.rotation)
    record = {
        "case_index": int(row["index"]),
        "case_display": _case_display(row),
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "thumbnail_path": str(thumbnail_path),
        "overlay_path": str(overlay_path),
        "created_at": _timestamp(),
        "error": "",
    }
    if not thumbnail_path.exists():
        record["error"] = f"missing_thumbnail:{thumbnail_path}"
        record["raw_response"] = ""
        record["parsed_response"] = {}
        return record
    if not overlay_path.exists():
        record["error"] = f"missing_overlay:{overlay_path}"
        record["raw_response"] = ""
        record["parsed_response"] = {}
        return record
    prompt_text = PROMPT + "\n\nCase:\n" + record["case_display"]
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt_text,
            image_paths=[thumbnail_path, overlay_path],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
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
    return record


def _write_pdf(output_root: Path, results: list[dict[str, Any]]) -> None:
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    for result in results:
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        review = _result_review(parsed)
        page = Image.new("RGB", (1800, 2200), "white")
        draw = ImageDraw.Draw(page)
        y = 35
        draw.text((45, y), result.get("case_display", ""), font=title_font, fill="black")
        y += 48
        header = (
            "missed_potential_tissue_candidates="
            f"{review.get('missed_potential_tissue_candidates', '')} | "
            f"confidence={review.get('confidence', '')} | error={result.get('error', '')}"
        )
        draw.text((45, y), header, font=body_font, fill="black")
        y += 42
        source = _thumb(Path(result["thumbnail_path"]), (820, 520))
        overlay = _thumb(Path(result["overlay_path"]), (820, 520))
        draw.text((45, y), "Source thumbnail", font=body_font, fill="black")
        draw.text((930, y), "Detection overlay", font=body_font, fill="black")
        page.paste(source, (45, y + 30))
        page.paste(overlay, (930, y + 30))
        y += 590
        draw.text((45, y), "Point-blank review", font=body_font, fill="black")
        y += 32
        y = _draw_wrapped(draw, (60, y), review.get("reasoning", ""), small_font, 155, "#111111")
        y += 12
        locations = review.get("rough_locations", [])
        if locations:
            y = _draw_wrapped(draw, (60, y), "rough_locations: " + json.dumps(locations), small_font, 155, "#111111")
        y += 20
        draw.text((45, y), "Raw response", font=body_font, fill="black")
        y += 32
        _draw_wrapped(draw, (60, y), str(result.get("raw_response", ""))[:2500], small_font, 155, "#111111")
        pages.append(page)
    pdf_path = output_root / "visuals" / "missed_candidate_point_blank.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def _write_reproduction(output_root: Path, args: argparse.Namespace, results_path: Path) -> None:
    text = f"""\
Stage 1 missed-candidate point-blank probe
==========================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Manifest: {args.manifest.resolve()}
Case indices: {','.join(str(i) for i in args.indices)}

Command:
python scripts/stage1_edge_case_missed_candidate_probe.py \\
  --manifest {args.manifest.resolve()} \\
  --output-root {output_root} \\
  --indices {','.join(str(i) for i in args.indices)} \\
  --model {args.model} \\
  --temperature {args.temperature}

Outputs:
- Results JSONL: {results_path}
- Summary CSV: {output_root / 'summary' / 'missed_candidate_point_blank.csv'}
- PDF: {output_root / 'visuals' / 'missed_candidate_point_blank.pdf'}

Notes:
- This is a deliberately simple floor test for missed visible potential tissue candidates.
- It does not ask for per-bbox grading, box refinement, or pathology/control-tissue semantics.
- By default it uses the raw single-orientation overlay packet when --raw-overlay-root is supplied.
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    rows = _selected_rows(args.manifest.resolve(), args.indices)
    if args.raw_overlay_root is not None:
        args.raw_overlay_root = args.raw_overlay_root.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "cases": [_case_display(row) for row in rows],
                    "output_root": str(output_root),
                    "raw_overlay_root": str(args.raw_overlay_root) if args.raw_overlay_root else "",
                },
                indent=2,
            )
        )
        return 0

    base_url, api_key = _api_settings(args)
    results = [_call_case(row, args, base_url, api_key) for row in rows]
    results_path = output_root / "reviews" / "missed_candidate_point_blank_results.jsonl"
    _write_jsonl(results_path, results)

    summary_rows: list[dict[str, Any]] = []
    for result in results:
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        review = _result_review(parsed)
        summary_rows.append(
            {
                "case_index": result.get("case_index", ""),
                "case_display": result.get("case_display", ""),
                "missed_potential_tissue_candidates": review.get("missed_potential_tissue_candidates", ""),
                "confidence": review.get("confidence", ""),
                "reasoning": review.get("reasoning", ""),
                "rough_locations": json.dumps(review.get("rough_locations", [])),
                "error": result.get("error", ""),
                "thumbnail_path": result.get("thumbnail_path", ""),
                "overlay_path": result.get("overlay_path", ""),
            }
        )
    _write_csv(output_root / "summary" / "missed_candidate_point_blank.csv", summary_rows)
    _write_json(
        output_root / "summary" / "missed_candidate_point_blank_summary.json",
        {
            "cases": len(results),
            "errors": sum(1 for result in results if result.get("error")),
            "missed_yes": sum(
                1
                for row in summary_rows
                if str(row.get("missed_potential_tissue_candidates", "")).lower() == "true"
            ),
        },
    )
    _write_pdf(output_root, results)
    _write_reproduction(output_root, args, results_path)
    print(json.dumps(summary_rows, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--raw-overlay-root", type=Path, default=None)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
