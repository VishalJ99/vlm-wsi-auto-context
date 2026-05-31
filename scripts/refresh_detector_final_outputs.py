#!/usr/bin/env python3
"""Refresh detector final outputs from existing Stage 6 artifacts.

This script is intentionally postprocessing-only: it reads existing
classification results, rewrites final detections/overlays for selected cases,
and never calls a VLM backend.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import run_detector_pipeline as pipeline


TICKET = "PER-239"
DEFAULT_REVIEW_PDF = "visuals/final_detections_after_sv40_skip_odd_one_out.pdf"
REFRESH_METHOD = "no_vlm_stage6_positive_boxes_skip_odd_one_out"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _command() -> str:
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def _case_matches(record: dict[str, Any], pattern: str) -> bool:
    values = [
        str(record.get("case_id", "")),
        str(record.get("case_display", "")),
        Path(str(record.get("wsi_path", ""))).stem,
    ]
    return any(fnmatch.fnmatch(value, pattern) for value in values)


def _load_stage6_by_case(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "intermediate_stage_artifacts" / "stage6_classification_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 6 classification results: {path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[str(row.get("case_id", ""))].append(row)
    return grouped


def _selected_records(root: Path, pattern: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_path = root / "all_detections.json"
    if not all_path.is_file():
        raise FileNotFoundError(f"Missing aggregate detections JSON: {all_path}")
    records = _read_json(all_path)
    if not isinstance(records, list):
        raise TypeError(f"Expected list in {all_path}")
    selected = [record for record in records if _case_matches(record, pattern)]
    if not selected:
        raise ValueError(f"No cases matched pattern {pattern!r} in {all_path}")
    return records, selected


def _stage_contract_skip_odd(contract: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched = deepcopy(contract)
    for stage in patched:
        if stage.get("stage") == "stage7_comparative_thumbnail_filter":
            stage["prompt"] = None
            stage["input_image"] = "Skipped by --skip-odd-one-out-filter."
            stage["output"] = (
                "Skipped by --skip-odd-one-out-filter; all Stage 6 "
                "tissue-positive boxes are retained."
            )
    return patched


def _append_or_replace_refresh(existing: list[Any], refresh: dict[str, Any]) -> list[dict[str, Any]]:
    kept = [item for item in existing if not (isinstance(item, dict) and item.get("ticket") == TICKET)]
    kept.append(refresh)
    return kept


def _final_detection_rows(stage6_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(stage6_rows, key=lambda row: int(row["candidate_order"]))
    return [row for row in rows if row.get("tissue_focus_decision") == "yes" and not row.get("error")]


def _boxes_from_rows(rows: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(value) for value in row["box_2d_yxyx_normalized"]] for row in rows]


def _detections_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "box_2d": [round(float(value), 3) for value in row["box_2d_yxyx_normalized"]],
            "source_candidate_order": int(row["candidate_order"]),
            "classification_decision": row.get("tissue_focus_decision", ""),
            "odd_one_out_flagged": False,
        }
        for row in rows
    ]


def _backup(path: Path, root: Path, backup_root: Path | None) -> Path | None:
    if backup_root is None or not path.exists():
        return None
    target = backup_root / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def _odd_skip_record(case_id: str, yes_count: int, total_count: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "classification_candidate_count": total_count,
        "classification_yes_count": yes_count,
        "skip_reason": "skip_odd_one_out_filter",
    }


def _refresh_case(
    root: Path,
    aggregate_record: dict[str, Any],
    stage6_rows: list[dict[str, Any]],
    refresh_base: dict[str, Any],
    backup_root: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    case_id = str(aggregate_record["case_id"])
    case_dir = Path(str(aggregate_record.get("case_dir") or (root / case_id)))
    case_json_path = case_dir / "detections.json"
    if not case_json_path.is_file():
        raise FileNotFoundError(f"Missing per-case detections JSON: {case_json_path}")

    case_record = _read_json(case_json_path)
    yes_rows = _final_detection_rows(stage6_rows)
    final_boxes = _boxes_from_rows(yes_rows)
    detections = _detections_from_rows(yes_rows)

    refresh = {
        **refresh_base,
        "case_id": case_id,
        "previous_final_boxes": int(case_record.get("stage_counts", {}).get("final_boxes", 0)),
        "new_final_boxes": len(final_boxes),
        "previous_odd_one_out": case_record.get("odd_one_out", {}),
    }

    updated = deepcopy(case_record)
    updated["refreshed_at"] = refresh_base["refreshed_at"]
    updated["refresh_ticket"] = TICKET
    updated["refresh_method"] = REFRESH_METHOD
    updated["git_commit"] = refresh_base["git_commit"]
    updated["stage_contract"] = _stage_contract_skip_odd(updated.get("stage_contract", []))
    updated["detections"] = detections
    updated.setdefault("stage_counts", {})
    updated["stage_counts"]["final_boxes"] = len(final_boxes)
    updated["stage_counts"]["odd_one_out_filter_skipped"] = True
    updated["stage_counts"]["odd_one_out_flagged"] = 0
    updated["odd_one_out"] = {
        "ran": False,
        "filter_skipped": True,
        "filter_applied": False,
        "parse_status": "",
        "filter_skip_reason": "skip_odd_one_out_filter",
        "flagged_candidate_orders": [],
        "raw_flagged_candidate_orders": [],
        "skipped": _odd_skip_record(case_id, len(yes_rows), len(stage6_rows)),
        "error": "",
    }
    updated.setdefault("paths", {})
    updated["paths"]["odd_one_out_sheet_path"] = ""
    updated["posthoc_refreshes"] = _append_or_replace_refresh(
        list(updated.get("posthoc_refreshes") or []),
        refresh,
    )

    thumbnail = Path(str(updated.get("paths", {}).get("thumbnail_path", "")))
    overlay = case_dir / "final_detected_bboxes.png"
    if not thumbnail.is_file():
        raise FileNotFoundError(f"Missing thumbnail for overlay redraw: {thumbnail}")

    if not dry_run:
        _backup(case_json_path, root, backup_root)
        _backup(overlay, root, backup_root)
        _backup(case_dir / "reproduction.txt", root, backup_root)
        pipeline._draw_boxes_overlay(
            thumbnail,
            overlay,
            final_boxes,
            f"final boxes: {len(final_boxes)} | status: {updated.get('pipeline_status', 'ok')}",
            colors=["#188038"],
        )
        _write_json(case_json_path, updated)
        _append_case_reproduction(case_dir / "reproduction.txt", refresh_base)

    aggregate = deepcopy(updated)
    aggregate["case_dir"] = str(case_dir.resolve())
    return aggregate


def _append_case_reproduction(path: Path, refresh_base: dict[str, Any]) -> None:
    block = _refresh_block(refresh_base, per_case=True)
    text = path.read_text() if path.exists() else ""
    text = _without_old_block(text)
    path.write_text(text.rstrip() + "\n\n" + block + "\n")


def _without_old_block(text: str) -> str:
    start = "<!-- BEGIN PER-239 NO-VLM REFRESH -->"
    end = "<!-- END PER-239 NO-VLM REFRESH -->"
    if start not in text or end not in text:
        return text
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before.rstrip() + "\n" + after.lstrip()


def _refresh_block(refresh_base: dict[str, Any], per_case: bool = False) -> str:
    title = "PER-239 no-VLM per-case refresh" if per_case else "PER-239 no-VLM SV40 refresh"
    return f"""\
<!-- BEGIN PER-239 NO-VLM REFRESH -->
{title}
{'=' * len(title)}

Refreshed: {refresh_base['refreshed_at']}
Ticket: {TICKET}
Git commit: {refresh_base['git_commit']}
Method: {REFRESH_METHOD}
No VLM calls: true

Command:
{refresh_base['command']}

Source artifacts:
- Stage 6 classifications: {refresh_base['stage6_results_path']}
- Aggregate detections: {refresh_base['all_detections_path']}

Case selection:
- Pattern: {refresh_base['case_glob']}
- Cases: {', '.join(refresh_base['case_ids'])}

Effect:
- For selected cases, final detections were rebuilt from Stage 6 rows with
  `tissue_focus_decision == "yes"` and no candidate error.
- Stage 7 odd-one-out removals were not applied to selected cases.
- Existing Stage 7 artifacts were left on disk for audit but are not used by
  the refreshed SV40 final JSON/overlays.
<!-- END PER-239 NO-VLM REFRESH -->"""


def _update_root_reproduction(root: Path, refresh_base: dict[str, Any], dry_run: bool) -> None:
    path = root / "reproduction.txt"
    if dry_run:
        return
    text = path.read_text() if path.exists() else ""
    text = _without_old_block(text)
    path.write_text(text.rstrip() + "\n\n" + _refresh_block(refresh_base) + "\n")


def _update_summary(
    root: Path,
    records: list[dict[str, Any]],
    selected_ids: set[str],
    refresh_base: dict[str, Any],
    backup_root: Path | None,
    dry_run: bool,
) -> None:
    path = root / "summary.json"
    if not path.is_file() or dry_run:
        return
    summary = _read_json(path)
    case_counts = {
        str(record.get("case_id")): int(record.get("stage_counts", {}).get("final_boxes", 0))
        for record in records
    }
    summary["final_boxes"] = sum(case_counts.values())
    summary["refreshed_at"] = refresh_base["refreshed_at"]
    summary["refresh_ticket"] = TICKET
    summary["refresh_method"] = REFRESH_METHOD
    for case_output in summary.get("case_outputs", []):
        case_id = str(case_output.get("case_id"))
        if case_id in selected_ids:
            case_output["final_box_count"] = case_counts[case_id]
    summary["posthoc_refreshes"] = _append_or_replace_refresh(
        list(summary.get("posthoc_refreshes") or []),
        {
            **refresh_base,
            "cases_refreshed": len(selected_ids),
            "final_boxes": summary["final_boxes"],
        },
    )
    _backup(path, root, backup_root)
    _write_json(path, summary)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_review_pdf(records: list[dict[str, Any]], pdf_path: Path, summary_path: Path) -> None:
    title_font = _font(30)
    meta_font = _font(18)
    pages: list[Image.Image] = []
    page_w, page_h = 1800, 1250
    margin = 45
    image_top = 135
    max_img_w = page_w - 2 * margin
    max_img_h = page_h - image_top - margin

    for idx, record in enumerate(records, start=1):
        overlay_path = Path(str(record.get("paths", {}).get("final_overlay_png", "")))
        if not overlay_path.is_file():
            continue
        image = Image.open(overlay_path).convert("RGB")
        scale = min(max_img_w / image.width, max_img_h / image.height)
        resized = image.resize((int(image.width * scale), int(image.height * scale)))
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        case_id = record.get("case_id", "")
        box_count = record.get("stage_counts", {}).get("final_boxes", len(record.get("detections", [])))
        odd = record.get("odd_one_out", {})
        draw.text((margin, 34), f"{idx}/{len(records)} | {case_id}", fill=(0, 0, 0), font=title_font)
        draw.text(
            (margin, 82),
            (
                f"final_boxes={box_count} | "
                f"odd_skipped={odd.get('filter_skipped', False)} | "
                f"odd_ran={odd.get('ran', False)}"
            ),
            fill=(60, 60, 60),
            font=meta_font,
        )
        x = margin + (max_img_w - resized.width) // 2
        page.paste(resized, (x, image_top))
        pages.append(page)

    if not pages:
        raise RuntimeError("No final overlay images were available for review PDF generation.")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    _write_json(
        summary_path,
        {
            "created_at": pipeline._timestamp(),
            "ticket": TICKET,
            "pdf": str(pdf_path.resolve()),
            "cases": len(pages),
            "source": "final_detected_bboxes.png from refreshed detector output root",
        },
    )


def _run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if not args.skip_odd_one_out_filter:
        raise SystemExit("This refresh currently supports only --skip-odd-one-out-filter.")

    records, selected = _selected_records(root, args.case_glob)
    stage6_by_case = _load_stage6_by_case(root)
    selected_ids = {str(record["case_id"]) for record in selected}
    missing_stage6 = sorted(case_id for case_id in selected_ids if case_id not in stage6_by_case)
    if missing_stage6:
        raise FileNotFoundError(f"Missing Stage 6 rows for selected cases: {missing_stage6}")

    refreshed_at = pipeline._timestamp()
    backup_root = None
    if args.backup and not args.dry_run:
        backup_root = root / "refresh_backups" / f"per239_{refreshed_at.replace(':', '').replace('+', '_')}"
        backup_root.mkdir(parents=True, exist_ok=True)

    refresh_base = {
        "ticket": TICKET,
        "refreshed_at": refreshed_at,
        "git_commit": pipeline._repo_git_commit(),
        "method": REFRESH_METHOD,
        "no_vlm_calls": True,
        "command": _command(),
        "case_glob": args.case_glob,
        "case_ids": sorted(selected_ids),
        "stage6_results_path": str((root / "intermediate_stage_artifacts" / "stage6_classification_results.jsonl").resolve()),
        "all_detections_path": str((root / "all_detections.json").resolve()),
        "backup_root": str(backup_root.resolve()) if backup_root else "",
    }

    by_case = {str(record.get("case_id")): record for record in records}
    refreshed_records: list[dict[str, Any]] = []
    for record in selected:
        case_id = str(record["case_id"])
        refreshed = _refresh_case(
            root,
            record,
            stage6_by_case[case_id],
            refresh_base,
            backup_root,
            args.dry_run,
        )
        by_case[case_id] = refreshed
        refreshed_records.append(refreshed)

    updated_records = [by_case[str(record.get("case_id"))] for record in records]
    if not args.dry_run:
        _backup(root / "all_detections.json", root, backup_root)
        _write_json(root / "all_detections.json", updated_records)
        _update_summary(root, updated_records, selected_ids, refresh_base, backup_root, args.dry_run)
        _backup(root / "reproduction.txt", root, backup_root)
        _update_root_reproduction(root, refresh_base, args.dry_run)
        if args.write_review_pdf:
            pdf_path = root / args.review_pdf
            summary_path = pdf_path.with_name(pdf_path.stem + "_summary.json")
            _backup(pdf_path, root, backup_root)
            _backup(summary_path, root, backup_root)
            _make_review_pdf(updated_records, pdf_path, summary_path)

    print(
        json.dumps(
            {
                "output_root": str(root),
                "dry_run": bool(args.dry_run),
                "case_glob": args.case_glob,
                "cases_refreshed": sorted(selected_ids),
                "total_final_boxes": sum(
                    int(record.get("stage_counts", {}).get("final_boxes", 0))
                    for record in updated_records
                ),
                "backup_root": str(backup_root) if backup_root else "",
                "review_pdf": str((root / args.review_pdf).resolve()) if args.write_review_pdf else "",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--case-glob", default="sv40_*")
    parser.add_argument("--skip-odd-one-out-filter", action="store_true")
    parser.add_argument("--review-pdf", type=Path, default=Path(DEFAULT_REVIEW_PDF))
    parser.add_argument("--write-review-pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return _run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
