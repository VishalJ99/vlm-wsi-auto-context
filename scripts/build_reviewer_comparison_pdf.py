#!/usr/bin/env python3
# ABOUTME: Build a PDF comparing high-res foreground reviewer inputs and VLM outputs.
# ABOUTME: Used for PER-188 reviewer prompt/model comparison artifacts.

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.model_pricing import estimate_review_cost_usd


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    input_run_id: str


@dataclass(frozen=True)
class RunConfig:
    label: str
    prompt: str
    model: str
    batch_dir: Path


CASES = [
    CaseConfig(
        case_id="he_patient_003_slide_003",
        input_run_id="he_p003_s003_icl0_20260430_175702_stage7_l0_review",
    ),
    CaseConfig(
        case_id="tol_blue_patient_054_slide_004",
        input_run_id="tol_blue_p054_s004_icl0_20260430_175702_stage7_l0_review",
    ),
]

RUNS = [
    RunConfig(
        label="Flash calibration",
        prompt="calibration",
        model="google/gemini-3-flash-preview",
        batch_dir=Path("runs/reviewer_pilot/highres_stage7_review_openrouter_flash_calibration_v2"),
    ),
    RunConfig(
        label="Gemini 3.1 Pro calibration",
        prompt="calibration",
        model="google/gemini-3.1-pro-preview",
        batch_dir=Path("runs/reviewer_pilot/highres_stage7_review_openrouter_gemini31pro_calibration"),
    ),
    RunConfig(
        label="Flash subjective",
        prompt="subjective",
        model="google/gemini-3-flash-preview",
        batch_dir=Path("runs/reviewer_pilot/highres_stage7_review_openrouter_flash_subjective_stair_ok"),
    ),
    RunConfig(
        label="Gemini 3.1 Pro subjective",
        prompt="subjective",
        model="google/gemini-3.1-pro-preview",
        batch_dir=Path("runs/reviewer_pilot/highres_stage7_review_openrouter_gemini31pro_subjective_stair_ok"),
    ),
]


def repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        return Path(out)
    except Exception:
        return Path.cwd()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_bbox_ids(input_root: Path, case: CaseConfig) -> Iterable[str]:
    bbox_root = input_root / case.case_id / case.input_run_id / "bboxes"
    for crop in sorted(bbox_root.glob("*/stage3/crop.png")):
        yield crop.parents[1].name


def image_flowable(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    draw_width = width * scale
    draw_height = height * scale
    return Image(str(path), width=draw_width, height=draw_height)


def flatten_review(parsed: dict[str, Any] | None) -> str:
    if not parsed:
        return "No parsed JSON output."

    precision = parsed.get("precision") or {}
    recall = parsed.get("recall") or {}
    parts: list[str] = []

    if "percentage" in precision or "percentage" in recall:
        parts.append(f"Precision: {precision.get('percentage', 'NA')}")
        parts.append(f"Recall: {recall.get('percentage', 'NA')}")
        return "<br/>".join(parts)

    if parsed.get("summary"):
        parts.append(f"<b>Summary:</b> {parsed['summary']}")
    if precision:
        rating = precision.get("rating") or precision.get("score") or "NA"
        notes = precision.get("artifact_notes") or ""
        parts.append(f"<b>Precision:</b> {rating}. {notes}")
    if recall:
        rating = recall.get("rating") or recall.get("score") or "NA"
        notes = recall.get("omission_notes") or ""
        parts.append(f"<b>Recall:</b> {rating}. {notes}")
    return "<br/>".join(parts)


def review_metadata(run: RunConfig, case: CaseConfig, bbox_id: str) -> dict[str, Any]:
    path = run.batch_dir / "reviews" / case.case_id / case.input_run_id / bbox_id / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_json(path)


def batch_cost_usd(run: RunConfig) -> str:
    summary_path = run.batch_dir / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        value = summary.get("estimated_total_cost_usd")
        if isinstance(value, (int, float)) and value > 0:
            return f"${value:.6f}"

    total = 0.0
    count = 0
    for meta_path in run.batch_dir.glob("reviews/*/*/*/metadata.json"):
        meta = load_json(meta_path)
        estimate = meta.get("cost_estimate_usd")
        if not estimate:
            estimate = estimate_review_cost_usd(meta.get("model", ""), meta.get("usage"))
        if not estimate:
            continue
        total += float(estimate["estimated_total_cost_usd"])
        count += 1
    if count:
        return f"${total:.6f}"
    return "NA"


def build_pdf(args: argparse.Namespace) -> None:
    root = repo_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / args.output_name

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Tiny",
            parent=styles["BodyText"],
            fontSize=6.2,
            leading=7.4,
            alignment=TA_LEFT,
        )
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title="Foreground Reviewer Comparison",
    )

    story: list[Any] = []
    story.append(Paragraph("Foreground Reviewer Comparison", styles["Title"]))
    story.append(
        Paragraph(
            "High-resolution Stage 7 bbox reviewer inputs with per-model/per-prompt outputs "
            "for H&amp;E patient 003 slide 003 and Tol Blue patient 054 slide 004.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    metadata_rows = [
        ["Generated", datetime.now().isoformat(timespec="seconds")],
        ["Git commit", git_commit()],
        ["Input root", args.input_root],
        ["Output PDF", rel(pdf_path, root)],
        ["Prompts", "calibration_reviewer.txt; subjective_reviewer.txt"],
        ["Models", "google/gemini-3-flash-preview; google/gemini-3.1-pro-preview"],
    ]
    meta_table = Table(metadata_rows, colWidths=[1.4 * inch, 8.9 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.15 * inch))

    summary_rows = [["Case", "Bbox", "Flash cal", "3.1 Pro cal", "Flash subj", "3.1 Pro subj"]]
    input_root = Path(args.input_root)
    for case in CASES:
        for bbox_id in iter_bbox_ids(input_root, case):
            row = [case.case_id, bbox_id]
            for run in RUNS:
                parsed = review_metadata(run, case, bbox_id).get("parsed_json")
                precision = (parsed or {}).get("precision") or {}
                recall = (parsed or {}).get("recall") or {}
                if run.prompt == "calibration":
                    row.append(f"{precision.get('percentage', 'NA')} / {recall.get('percentage', 'NA')}")
                else:
                    row.append(f"{precision.get('rating', 'NA')} / {recall.get('rating', 'NA')}")
            summary_rows.append(row)

    summary = Table(
        [[Paragraph(str(cell), styles["Tiny"]) for cell in row] for row in summary_rows],
        colWidths=[1.55 * inch, 1.55 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch],
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8fb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(summary)
    story.append(PageBreak())

    for case in CASES:
        for bbox_id in iter_bbox_ids(input_root, case):
            stage3 = input_root / case.case_id / case.input_run_id / "bboxes" / bbox_id / "stage3"
            crop_path = stage3 / "crop.png"
            overlay_path = stage3 / "overlay.png"
            mask_path = stage3 / "mask.png"
            stage_meta = load_json(stage3 / "metadata.json")
            crop_size = stage_meta.get("crop_size")
            read_level = stage_meta.get("read_level")
            downsample = stage_meta.get("read_downsample")

            story.append(Paragraph(f"{case.case_id} | {bbox_id}", styles["Heading1"]))
            story.append(
                Paragraph(
                    f"High-res input crop: {rel(crop_path, root)}<br/>"
                    f"Mask: {rel(mask_path, root)}<br/>"
                    f"Overlay: {rel(overlay_path, root)}<br/>"
                    f"Crop size: {crop_size}; read level: {read_level}; downsample: {downsample}",
                    styles["Small"],
                )
            )
            story.append(Spacer(1, 0.08 * inch))
            image_table = Table(
                [
                    [
                        Paragraph("<b>Source crop</b>", styles["Small"]),
                        Paragraph("<b>Mask overlay</b>", styles["Small"]),
                    ],
                    [
                        image_flowable(crop_path, 4.7 * inch, 2.7 * inch),
                        image_flowable(overlay_path, 4.7 * inch, 2.7 * inch),
                    ],
                ],
                colWidths=[5.0 * inch, 5.0 * inch],
            )
            image_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                    ]
                )
            )
            story.append(image_table)
            story.append(Spacer(1, 0.08 * inch))

            rows = [[Paragraph("<b>Run</b>", styles["Tiny"]), Paragraph("<b>Output</b>", styles["Tiny"])]]
            for run in RUNS:
                meta = review_metadata(run, case, bbox_id)
                parsed = meta.get("parsed_json")
                run_bits = [
                    f"<b>{run.label}</b>",
                    f"Prompt: {run.prompt}",
                    f"Model: {run.model}",
                    f"Thinking: {meta.get('thinking_level')}; thoughts stored: {meta.get('include_thoughts')}",
                    f"Raw: {rel(Path(meta['outputs']['raw_response']), root)}",
                    f"Batch cost: {batch_cost_usd(run)}",
                ]
                rows.append(
                    [
                        Paragraph("<br/>".join(run_bits), styles["Tiny"]),
                        Paragraph(flatten_review(parsed), styles["Tiny"]),
                    ]
                )

            review_table = Table(rows, colWidths=[2.3 * inch, 7.7 * inch], repeatRows=1)
            review_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(KeepTogether(review_table))
            story.append(PageBreak())

    doc.build(story)

    reproduction = output_dir / "reproduction.txt"
    reproduction.write_text(
        "\n".join(
            [
                "Artifact: PER-188 foreground reviewer comparison PDF",
                f"Created: {datetime.now().isoformat(timespec='seconds')}",
                f"Repository: {root}",
                f"Git commit: {git_commit()}",
                "Ticket: PER-188",
                "",
                "Purpose:",
                "Create a PDF showing high-resolution bbox reviewer inputs and per-model/per-prompt reviewer outputs.",
                "",
                "Inputs:",
                f"- {root / args.input_root}",
                "- runs/reviewer_pilot/highres_stage7_review_openrouter_flash_calibration_v2",
                "- runs/reviewer_pilot/highres_stage7_review_openrouter_gemini31pro_calibration",
                "- runs/reviewer_pilot/highres_stage7_review_openrouter_flash_subjective_stair_ok",
                "- runs/reviewer_pilot/highres_stage7_review_openrouter_gemini31pro_subjective_stair_ok",
                "",
                "Command:",
                "export PATH=/vol/biomedic3/vj724/.conda/envs/path-agent/bin:$PATH",
                f"python scripts/build_reviewer_comparison_pdf.py --output-dir {args.output_dir} --output-name {args.output_name}",
                "",
                f"Output PDF: {pdf_path.resolve()}",
                "",
            ]
        )
    )
    print(pdf_path.resolve())
    print(reproduction.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build foreground reviewer comparison PDF.")
    parser.add_argument("--input-root", default="runs/auto_context_reviewer_inputs")
    parser.add_argument(
        "--output-dir",
        default="runs/reviewer_pilot/foreground_reviewer_comparison_report",
    )
    parser.add_argument("--output-name", default="foreground_reviewer_comparison.pdf")
    return parser


def main() -> None:
    build_pdf(build_parser().parse_args())


if __name__ == "__main__":
    main()
