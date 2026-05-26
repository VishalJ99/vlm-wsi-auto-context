#!/usr/bin/env python3
"""Create Stage 6 candidate inputs with bbox rectangles but no numeric labels."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import _font, _repo_git_commit, _safe_slug, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISAGREEMENTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_gemini31pro_preview_hires_explain_v1/comparison"
    / "stage6_gemini3flash_vs_gemini31pro_disagreements.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_gemini31pro_no_enum_disagreement_rerun_v1/no_enum_inputs"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _draw_bbox_only_overlay(crop_path: Path, box: Iterable[int], output_path: Path) -> None:
    crop = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(crop)
    x1, y1, x2, y2 = [int(v) for v in box]
    line_width = max(3, max(crop.size) // 180)
    draw.rectangle((x1, y1, x2, y2), outline="#e31a1c", width=line_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)


def _draw_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    thumb_w, thumb_h = 240, 240
    cols = 6
    label_h = 70
    rows_n = (len(rows) + cols - 1) // cols
    page = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(page)
    font = _font(16)
    small = _font(13)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x = col * thumb_w
        y = grid_row * (thumb_h + label_h)
        image = Image.open(_resolve(str(row["selected_overlay_path"]))).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        page.paste(image, (x + (thumb_w - image.size[0]) // 2, y))
        label = (
            f"{int(row['case_index']):03d}/{int(row['candidate_order']):02d} "
            f"F={row.get('flash_decision', '')} P={row.get('pro_decision', '')}"
        )
        draw.text((x + 6, y + thumb_h + 5), label, fill="#111111", font=font)
        draw.text((x + 6, y + thumb_h + 28), str(row["candidate_id"])[:28], fill="#333333", font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path)


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    disagreement_rows = _read_csv(args.disagreements)
    if args.limit is not None:
        disagreement_rows = disagreement_rows[: args.limit]
    output_cases = args.output_root / "cases"
    candidate_rows: list[dict[str, Any]] = []
    for row in disagreement_rows:
        crop_path = _resolve(row["crop_path"])
        metadata_path = _resolve(row["metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        box = metadata["candidate"]["read_info"]["source_bbox_in_crop"]
        case_index = int(row["case_index"])
        candidate_order = int(row["candidate_order"])
        case_slug = _safe_slug(f"{case_index:03d}_{row['case_display']}")
        candidate_slug = _safe_slug(f"{candidate_order:02d}_{row['candidate_id']}")
        overlay_path = output_cases / case_slug / "candidates" / candidate_slug / "selected_candidate_overlay_no_enum.png"
        _draw_bbox_only_overlay(crop_path, box, overlay_path)
        candidate_rows.append(
            {
                "case_index": case_index,
                "case_display": row["case_display"],
                "bbox_source": row["bbox_source"],
                "candidate_order": candidate_order,
                "candidate_id": row["candidate_id"],
                "label": row["label"],
                "crop_path": row["crop_path"],
                "selected_overlay_path": _repo_relative(overlay_path),
                "metadata_path": row["metadata_path"],
                "overlay_style": "red_bbox_no_numeric_label",
                "source_disagreement_csv": str(args.disagreements.resolve()),
                "source_selected_overlay_path": row["selected_overlay_path"],
                "flash_decision": row["flash_decision"],
                "pro_decision": row["pro_decision"],
                "source_bbox_in_crop": json.dumps([int(v) for v in box]),
            }
        )

    candidates_csv = args.output_root / "summary/stage6_no_enum_disagreement_candidates.csv"
    _write_csv(
        candidates_csv,
        candidate_rows,
        [
            "case_index",
            "case_display",
            "bbox_source",
            "candidate_order",
            "candidate_id",
            "label",
            "crop_path",
            "selected_overlay_path",
            "metadata_path",
            "overlay_style",
            "source_disagreement_csv",
            "source_selected_overlay_path",
            "flash_decision",
            "pro_decision",
            "source_bbox_in_crop",
        ],
    )
    contact_sheet = args.output_root / "visuals/stage6_no_enum_disagreement_contact_sheet.png"
    _draw_contact_sheet(candidate_rows, contact_sheet)
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "objective": "Create Stage 6 disagreement-only candidate inputs with bbox rectangles but no numeric labels.",
        "source_disagreement_csv": str(args.disagreements.resolve()),
        "output_root": str(args.output_root.resolve()),
        "candidates": len(candidate_rows),
        "overlay_style": "red_bbox_no_numeric_label",
        "candidates_csv": str(candidates_csv.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
    }
    _write_json(args.output_root / "summary/stage6_no_enum_disagreement_inputs_summary.json", summary)
    _write_reproduction(args, candidates_csv, contact_sheet, summary)
    return summary


def _write_reproduction(args: argparse.Namespace, candidates_csv: Path, contact_sheet: Path, summary: dict[str, Any]) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage6_no_enum_disagreement_inputs.py",
            "--disagreements",
            str(args.disagreements.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
        ]
    )
    text = f"""\
Stage 6 no-enumeration disagreement inputs
==========================================

Created: {summary['created_at']}
Ticket: PER-207
Git commit: {summary['git_commit']}

Objective:
Create a disagreement-only Stage 6 candidate manifest where the image sent to
the VLM preserves the same crop and red bbox but removes the numeric candidate
label drawn inside the bbox.

Rationale:
Manual review of the Gemini 3.1 Pro disagreement packet suggested that the
model sometimes reasoned about the bbox enumeration label rather than the
highlighted histology region. This input variant tests the visual confound
directly without changing the prompt.

Inputs:
- Source disagreement CSV: {args.disagreements.resolve()}

Command:
{command}

Outputs:
- Candidate manifest: {candidates_csv.resolve()}
- Contact sheet: {contact_sheet.resolve()}
- Summary JSON: {(args.output_root / 'summary/stage6_no_enum_disagreement_inputs_summary.json').resolve()}
- Per-candidate no-enumeration overlays: {(args.output_root / 'cases').resolve()}

Counts:
- Candidates: {summary['candidates']}
"""
    (args.output_root / "reproduction.txt").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreements", type=Path, default=DEFAULT_DISAGREEMENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_inputs(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
