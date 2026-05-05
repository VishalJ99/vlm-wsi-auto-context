#!/usr/bin/env python3
"""Create side-by-side TRIDENT vs distilled-student foreground overlays."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = (245, 246, 247)
PANEL_BG = (255, 255, 255)
TEXT = (25, 29, 33)
MUTED = (80, 86, 92)
BORDER = (195, 198, 202)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--trident-run-dir", required=True, type=Path)
    parser.add_argument("--student-a-name", default="zero_shot_student")
    parser.add_argument("--student-b-name", default="harder_qwen8bfew_student")
    parser.add_argument("--student-a-label", default="Distilled student A")
    parser.add_argument("--student-b-label", default="Distilled student B")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-panel-width", type=int, default=390)
    parser.add_argument("--max-panel-height", type=int, default=620)
    return parser.parse_args()


def load_font(name: str, size: int):
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


FONT_TITLE = load_font("DejaVuSans-Bold.ttf", 20)
FONT_SMALL = load_font("DejaVuSans.ttf", 14)


def fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def metric_line(metrics: dict | None) -> str:
    if not metrics:
        return "metrics unavailable"
    return "F1 {f1:.3f}  P {precision:.3f}  R {recall:.3f}".format(**metrics)


def fit_image(path: Path, max_w: int, max_h: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return image


def add_panel(path: Path, title: str, subtitle: str, max_w: int, max_h: int) -> Image.Image:
    image = fit_image(path, max_w, max_h)
    label_h = 64
    panel = Image.new("RGB", (max_w, image.height + label_h + 18), PANEL_BG)
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, max_w - 1, panel.height - 1], outline=BORDER)
    draw.text((12, 10), title, font=FONT_TITLE, fill=TEXT)
    draw.text((12, 38), subtitle, font=FONT_SMALL, fill=MUTED)
    panel.paste(image, ((max_w - image.width) // 2, label_h))
    return panel


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "visuals" / "trident_vs_distilled"
    comp_dir = output_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    student_a_root = args.run_root / "outputs" / args.student_a_name
    student_b_root = args.run_root / "outputs" / args.student_b_name
    student_a = json.loads((student_a_root / "summary.json").read_text())
    student_b = json.loads((student_b_root / "summary.json").read_text())
    student_b_by_key = {(e["case_id"], e["stage6_relpath"]): e for e in student_b["entries"]}

    rows: list[dict[str, object]] = []
    comparison_paths: list[Path] = []

    for entry in student_a["entries"]:
        case_id = entry["case_id"]
        rel_stage6 = entry["stage6_relpath"]
        parts = Path(rel_stage6).parts
        if "bboxes" not in parts:
            raise RuntimeError(f"cannot infer bbox from stage6 relpath: {rel_stage6}")
        bbox = parts[parts.index("bboxes") + 1]
        student_b_entry = student_b_by_key[(case_id, rel_stage6)]

        trident_stage3 = args.trident_run_dir / "bboxes" / bbox / "stage3"
        source_crop = trident_stage3 / "crop.png"
        trident_overlay = trident_stage3 / "overlay.png"
        teacher_overlay = args.teacher_root / case_id / rel_stage6 / "class_overlay.png"
        student_a_overlay = student_a_root / case_id / rel_stage6 / "student_class_overlay.png"
        student_b_overlay = student_b_root / case_id / rel_stage6 / "student_class_overlay.png"

        panels = [
            add_panel(source_crop, "Source crop", "same Stage 1 bbox", args.max_panel_width, args.max_panel_height),
            add_panel(trident_overlay, "TRIDENT HEST", "GeoJSON foreground", args.max_panel_width, args.max_panel_height),
            add_panel(teacher_overlay, "Auto-context Stage 6", "original VLM overlay", args.max_panel_width, args.max_panel_height),
            add_panel(
                student_a_overlay,
                args.student_a_label,
                metric_line(entry.get("metrics_vs_teacher_binary")),
                args.max_panel_width,
                args.max_panel_height,
            ),
            add_panel(
                student_b_overlay,
                args.student_b_label,
                metric_line(student_b_entry.get("metrics_vs_teacher_binary")),
                args.max_panel_width,
                args.max_panel_height,
            ),
        ]

        margin = 16
        title_h = 62
        width = sum(panel.width for panel in panels) + margin * (len(panels) + 1)
        height = max(panel.height for panel in panels) + margin * 2 + title_h
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 14), f"{case_id} / {bbox}", font=FONT_TITLE, fill=TEXT)
        draw.text(
            (margin, 42),
            "TRIDENT foreground compared with auto-context VLM and distilled MobileNetV3 foreground overlays.",
            font=FONT_SMALL,
            fill=MUTED,
        )
        x = margin
        y = margin + title_h
        for panel in panels:
            canvas.paste(panel, (x, y))
            x += panel.width + margin

        out_path = comp_dir / f"{case_id}__{bbox}.png"
        canvas.save(out_path, optimize=True)
        comparison_paths.append(out_path)

        a_metrics = entry.get("metrics_vs_teacher_binary", {})
        b_metrics = student_b_entry.get("metrics_vs_teacher_binary", {})
        rows.append(
            {
                "case_id": case_id,
                "bbox": bbox,
                "n_patches": entry.get("n_patches"),
                "student_a_f1_vs_auto_context_stage6": a_metrics.get("f1"),
                "student_a_precision_vs_auto_context_stage6": a_metrics.get("precision"),
                "student_a_recall_vs_auto_context_stage6": a_metrics.get("recall"),
                "student_a_auprc_vs_auto_context_stage6": a_metrics.get("auprc"),
                "student_b_f1_vs_auto_context_stage6": b_metrics.get("f1"),
                "student_b_precision_vs_auto_context_stage6": b_metrics.get("precision"),
                "student_b_recall_vs_auto_context_stage6": b_metrics.get("recall"),
                "student_b_auprc_vs_auto_context_stage6": b_metrics.get("auprc"),
                "comparison_png": str(out_path),
                "trident_overlay": str(trident_overlay),
                "teacher_overlay": str(teacher_overlay),
                "student_a_overlay": str(student_a_overlay),
                "student_b_overlay": str(student_b_overlay),
            }
        )

    metrics_csv = output_dir / "metrics_by_bbox.csv"
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics_by_bbox.json").write_text(json.dumps(rows, indent=2) + "\n")

    thumbs: list[tuple[Path, Image.Image]] = []
    for path in comparison_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1100, 440), Image.Resampling.LANCZOS)
        thumbs.append((path, image.copy()))
    pad = 18
    label_h = 28
    sheet_w = 1180
    sheet_h = pad + sum(image.height + label_h + pad for _, image in thumbs)
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    y = pad
    for path, image in thumbs:
        draw.text((pad, y), path.stem.replace("__", " / "), font=FONT_SMALL, fill=MUTED)
        y += label_h
        sheet.paste(image, ((sheet_w - image.width) // 2, y))
        y += image.height + pad
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path, optimize=True)

    html_rows = []
    for row in rows:
        rel_img = Path(row["comparison_png"]).relative_to(output_dir)
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case_id']))}</td>"
            f"<td>{html.escape(str(row['bbox']))}</td>"
            f"<td>{row['n_patches']}</td>"
            f"<td>{fmt(row['student_a_f1_vs_auto_context_stage6'])}</td>"
            f"<td>{fmt(row['student_a_precision_vs_auto_context_stage6'])}</td>"
            f"<td>{fmt(row['student_a_recall_vs_auto_context_stage6'])}</td>"
            f"<td>{fmt(row['student_b_f1_vs_auto_context_stage6'])}</td>"
            f"<td>{fmt(row['student_b_precision_vs_auto_context_stage6'])}</td>"
            f"<td>{fmt(row['student_b_recall_vs_auto_context_stage6'])}</td>"
            f"<td><a href=\"{html.escape(str(rel_img))}\">comparison</a></td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TRIDENT vs distilled foreground comparison</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #1f2328; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
th {{ background: #f6f8fa; }}
img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de; }}
.note {{ color: #57606a; max-width: 980px; }}
</style></head><body>
<h1>TRIDENT vs Distilled Foreground Comparison</h1>
<p class="note">Metrics compare each student to the existing auto-context Stage 6 labels, not to TRIDENT. Use the images for TRIDENT-vs-student visual inspection.</p>
<table><tr><th>case</th><th>bbox</th><th>patches</th><th>A F1</th><th>A P</th><th>A R</th><th>B F1</th><th>B P</th><th>B R</th><th>image</th></tr>
{''.join(html_rows)}
</table>
<h2>Contact Sheet</h2>
<p><img src="contact_sheet.png" alt="contact sheet"></p>
</body></html>
"""
    (output_dir / "index.html").write_text(html_doc)

    print(f"wrote {len(comparison_paths)} comparison PNGs")
    print(sheet_path)
    print(output_dir / "index.html")
    print(metrics_csv)


if __name__ == "__main__":
    main()
