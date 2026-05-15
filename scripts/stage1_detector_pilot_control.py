#!/usr/bin/env python3
"""Build and operate the Stage 1 detector pilot control plane.

The pilot starts with a deterministic 100-WSI worklist balanced across five
stains and grouped by biopsy/path. Later subcommands export a manual review
packet, build synthetic guard examples by dropping one detected box, and run or
summarize the VLM guard.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import random
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_CSV = Path("/data2/vj724/multistain/pilot_data/manifest.csv")
DEFAULT_SOURCE_WORKBOOK = Path(
    "/vol/biomedic3/histopatho/win_share/"
    "anon_master_combined_v5_with_reports_report_fixed (2).xlsx"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "stage1_detector_pilot_v1"
DEFAULT_RUN_ID = "stage1_detector_pilot_v1"
DEFAULT_STAINS = ("H&E", "PAS", "JONES", "EVG", "SV40")
TRUTHY = {"1", "true", "t", "yes", "y", "pass", "passed"}
FALSEY = {"0", "false", "f", "no", "n", "fail", "failed"}


@dataclass(frozen=True)
class SelectedSlide:
    group_rank: int
    pilot_row_id: str
    manual_review: bool
    stain: str
    row: dict[str, str]
    wsi_path: Path
    expected_stage1_dir: Path
    run_id: str


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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_parent(path)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _numeric(value: str | None, default: int = 10**9) -> int:
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _normalize_stain(value: str) -> str:
    stain = value.strip().upper()
    if stain in {"HNE", "H AND E", "H&E"}:
        return "H&E"
    return stain


def _safe_stain(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _wsi_id_from_path(path: Path) -> str:
    stem = path.name
    for suffix in (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def _existing_wsi_path(row: dict[str, str]) -> Path | None:
    for key in ("dest_path", "source_path"):
        value = row.get(key, "").strip()
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate
    return None


def _sort_slide_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _numeric(row.get("slide_index_within_stain")),
            _numeric(row.get("slide_index_within_case")),
            row.get("dest_path") or row.get("source_path") or "",
        ),
    )


def _stage1_paths(stage1_dir: Path) -> dict[str, Path]:
    return {
        "stage1_dir": stage1_dir,
        "thumbnail": stage1_dir / "thumbnail.png",
        "overlay": stage1_dir / "bbox_overlay.png",
        "bboxes_json": stage1_dir / "bboxes.json",
    }


def _selected_to_csv_row(
    slide: SelectedSlide,
    *,
    manifest_csv: Path,
    source_workbook: Path,
    selection_rule: str,
    pilot_selection_seed: int,
    stage1_model: str,
) -> dict[str, Any]:
    row = slide.row
    paths = _stage1_paths(slide.expected_stage1_dir)
    return {
        "pilot_row_id": slide.pilot_row_id,
        "selection_rank": slide.group_rank,
        "manual_review": str(slide.manual_review).lower(),
        "case_id": row.get("case_id", ""),
        "Anon_Patient_ID": row.get("Anon_Patient_ID", ""),
        "Anon_Path_ID": row.get("Anon_Path_ID", ""),
        "selected_biopsy_date": row.get("selected_biopsy_date", ""),
        "selected_biopsy_number": row.get("selected_biopsy_number", ""),
        "selected_time_series": row.get("selected_time_series", ""),
        "stain": slide.stain,
        "wsi_path": str(slide.wsi_path),
        "dest_path": row.get("dest_path", ""),
        "source_path": row.get("source_path", ""),
        "dest_filename": row.get("dest_filename", ""),
        "dest_relpath": row.get("dest_relpath", ""),
        "Anon_Slide_ID": row.get("Anon_Slide_ID", ""),
        "source_filename": row.get("source_filename", ""),
        "source_size_bytes": row.get("source_size_bytes", ""),
        "slide_index_within_stain": row.get("slide_index_within_stain", ""),
        "slide_index_within_case": row.get("slide_index_within_case", ""),
        "source_manifest": str(manifest_csv),
        "source_workbook": str(source_workbook),
        "selection_seed": row.get("selection_seed", ""),
        "pilot_selection_seed": pilot_selection_seed,
        "selection_rule": selection_rule,
        "run_id": slide.run_id,
        "stage1_model": stage1_model,
        "expected_stage1_dir": str(paths["stage1_dir"]),
        "expected_stage1_thumbnail": str(paths["thumbnail"]),
        "expected_stage1_overlay": str(paths["overlay"]),
        "expected_stage1_bboxes": str(paths["bboxes_json"]),
    }


def _select_balanced_slides(
    rows: list[dict[str, str]],
    *,
    stains: tuple[str, ...],
    n_groups: int,
    manual_review_groups: int,
    seed: int,
    stage1_output_root: Path,
    run_id: str,
) -> tuple[list[SelectedSlide], dict[str, Any]]:
    required_stains = tuple(_normalize_stain(stain) for stain in stains)
    grouped: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = {}
    skipped_missing_path = 0
    for row in rows:
        path = _existing_wsi_path(row)
        if path is None:
            skipped_missing_path += 1
            continue
        case_id = row.get("case_id", "").strip()
        path_id = row.get("Anon_Path_ID", "").strip()
        stain = _normalize_stain(row.get("Stain_from_wsi", ""))
        if not case_id or not path_id or not stain:
            continue
        grouped.setdefault((case_id, path_id), {}).setdefault(stain, []).append(row)

    eligible: list[tuple[tuple[str, str], dict[str, dict[str, str]]]] = []
    for group_key, stain_rows in sorted(grouped.items()):
        if all(stain in stain_rows for stain in required_stains):
            chosen = {
                stain: _sort_slide_rows(stain_rows[stain])[0]
                for stain in required_stains
            }
            eligible.append((group_key, chosen))

    if len(eligible) < n_groups:
        raise SystemExit(
            f"Need {n_groups} eligible biopsy/path groups with stains "
            f"{', '.join(required_stains)}; found {len(eligible)}."
        )

    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected_groups = eligible[:n_groups]
    selected: list[SelectedSlide] = []
    for group_rank, ((case_id, path_id), chosen) in enumerate(selected_groups, start=1):
        manual_review = group_rank <= manual_review_groups
        for stain in required_stains:
            row = chosen[stain]
            path = _existing_wsi_path(row)
            if path is None:
                raise RuntimeError("Selected row lost its WSI path during selection.")
            wsi_id = _wsi_id_from_path(path)
            stage1_dir = stage1_output_root / wsi_id / run_id / "stage1"
            pilot_row_id = f"{group_rank:03d}_{case_id}_{path_id}_{_safe_stain(stain)}"
            selected.append(
                SelectedSlide(
                    group_rank=group_rank,
                    pilot_row_id=pilot_row_id,
                    manual_review=manual_review,
                    stain=stain,
                    row=row,
                    wsi_path=path,
                    expected_stage1_dir=stage1_dir,
                    run_id=run_id,
                )
            )

    diagnostics = {
        "manifest_rows": len(rows),
        "skipped_missing_path_rows": skipped_missing_path,
        "groups_seen": len(grouped),
        "eligible_groups": len(eligible),
        "selected_groups": len(selected_groups),
        "selected_wsi_count": len(selected),
        "manual_review_groups": manual_review_groups,
        "manual_review_wsi_count": sum(1 for item in selected if item.manual_review),
    }
    return selected, diagnostics


def _write_wsi_list(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    path.write_text("".join(f"{row['wsi_path']}\n" for row in rows))


def _write_stage1_command(
    path: Path,
    *,
    wsi_list: Path,
    output_root: Path,
    run_id: str,
    model: str,
    max_dim: int,
    api_base: str,
) -> None:
    cmd = [
        "python",
        "run_auto_context.py",
        "--wsi-list",
        str(wsi_list),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--max-stage",
        "1",
        "--stage1-model",
        model,
        "--stage1-max-dim",
        str(max_dim),
        "--stage1-api-base",
        api_base,
    ]
    _ensure_parent(path)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        "# Requires OPENROUTER_API_KEY or OPENAI_API_KEY in the environment.\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + "\n"
    )
    path.chmod(0o755)


def cmd_build_worklist(args: argparse.Namespace) -> int:
    manifest_csv = args.manifest_csv.resolve()
    source_workbook = args.source_workbook
    output_root = args.output_root.resolve()
    stage1_output_root = args.stage1_output_root.resolve()
    worklist_dir = output_root / "worklists"
    command_dir = output_root / "commands"

    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_csv}")
    if not source_workbook.exists():
        raise SystemExit(f"Source workbook does not exist: {source_workbook}")

    rows = _read_csv(manifest_csv)
    selected, diagnostics = _select_balanced_slides(
        rows,
        stains=tuple(args.stains),
        n_groups=args.n_groups,
        manual_review_groups=args.manual_review_groups,
        seed=args.seed,
        stage1_output_root=stage1_output_root,
        run_id=args.run_id,
    )
    selection_rule = (
        "Deterministic random biopsy/path group sample after filtering for one "
        "available WSI per required stain; within each stain choose the lowest "
        "slide_index_within_stain, then lowest slide_index_within_case."
    )
    csv_rows = [
        _selected_to_csv_row(
            slide,
            manifest_csv=manifest_csv,
            source_workbook=source_workbook,
            selection_rule=selection_rule,
            pilot_selection_seed=args.seed,
            stage1_model=args.stage1_model,
        )
        for slide in selected
    ]
    fieldnames = list(csv_rows[0].keys())
    pilot_csv = worklist_dir / "pilot_100.csv"
    manual_csv = worklist_dir / "manual_review_20.csv"
    pilot_wsi_list = worklist_dir / "pilot_100_wsi_list.txt"
    manual_wsi_list = worklist_dir / "manual_review_20_wsi_list.txt"
    _write_csv(pilot_csv, csv_rows, fieldnames)
    _write_csv(manual_csv, [row for row in csv_rows if row["manual_review"] == "true"], fieldnames)
    _write_wsi_list(pilot_wsi_list, csv_rows)
    _write_wsi_list(manual_wsi_list, [row for row in csv_rows if row["manual_review"] == "true"])

    command_path = command_dir / "run_stage1_pilot.sh"
    _write_stage1_command(
        command_path,
        wsi_list=pilot_wsi_list,
        output_root=stage1_output_root,
        run_id=args.run_id,
        model=args.stage1_model,
        max_dim=args.stage1_max_dim,
        api_base=args.stage1_api_base,
    )

    summary = {
        "created_at": _timestamp(),
        "repo_root": str(REPO_ROOT),
        "repo_commit": _repo_git_commit(),
        "manifest_csv": str(manifest_csv),
        "source_workbook": str(source_workbook),
        "output_root": str(output_root),
        "stage1_output_root": str(stage1_output_root),
        "run_id": args.run_id,
        "stage1_model": args.stage1_model,
        "stage1_max_dim": args.stage1_max_dim,
        "stage1_api_base": args.stage1_api_base,
        "required_stains": list(args.stains),
        "seed": args.seed,
        "n_groups": args.n_groups,
        "manual_review_groups": args.manual_review_groups,
        "selection_rule": selection_rule,
        "diagnostics": diagnostics,
        "outputs": {
            "pilot_csv": str(pilot_csv),
            "manual_review_csv": str(manual_csv),
            "pilot_wsi_list": str(pilot_wsi_list),
            "manual_review_wsi_list": str(manual_wsi_list),
            "stage1_command": str(command_path),
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_reproduction(output_root / "reproduction.txt", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _write_reproduction(path: Path, summary: dict[str, Any]) -> None:
    cmd = (
        "python scripts/stage1_detector_pilot_control.py build-worklist "
        f"--manifest-csv {shlex.quote(summary['manifest_csv'])} "
        f"--source-workbook {shlex.quote(summary['source_workbook'])} "
        f"--output-root {shlex.quote(summary['output_root'])} "
        f"--stage1-output-root {shlex.quote(summary['stage1_output_root'])} "
        f"--run-id {shlex.quote(summary['run_id'])} "
        f"--n-groups {summary['n_groups']} "
        f"--manual-review-groups {summary['manual_review_groups']} "
        f"--seed {summary['seed']} "
        f"--stage1-model {shlex.quote(summary['stage1_model'])} "
        f"--stage1-max-dim {summary['stage1_max_dim']} "
        f"--stage1-api-base {shlex.quote(summary['stage1_api_base'])}"
    )
    stains = " ".join(shlex.quote(stain) for stain in summary["required_stains"])
    text = f"""Stage 1 Detector Pilot Control Plane V1

Created: {summary['created_at']}
Repository: {summary['repo_root']}
Git commit: {summary['repo_commit']}
Ticket: PER-188

Inputs
- Manifest CSV: {summary['manifest_csv']}
- Source workbook: {summary['source_workbook']}
- Required stains: {', '.join(summary['required_stains'])}
- Stage 1 model: {summary['stage1_model']}
- Stage 1 max thumbnail dimension: {summary['stage1_max_dim']}
- Stage 1 API base: {summary['stage1_api_base']}
- Seed: {summary['seed']}
- Pilot groups: {summary['n_groups']}
- Manual review groups: {summary['manual_review_groups']}
- Selection rule: {summary['selection_rule']}

Outputs
- Pilot CSV: {summary['outputs']['pilot_csv']}
- Manual review CSV: {summary['outputs']['manual_review_csv']}
- Pilot WSI list: {summary['outputs']['pilot_wsi_list']}
- Manual review WSI list: {summary['outputs']['manual_review_wsi_list']}
- Stage 1 command: {summary['outputs']['stage1_command']}

Regenerate worklists
{cmd} --stains {stains}

Run Stage 1 detector calls after approval/credential check
{summary['outputs']['stage1_command']}

Next quality gate
1. Run Stage 1 using the command above.
2. Export a manual review packet:
   python scripts/stage1_detector_pilot_control.py export-review-packet --worklist-csv {summary['outputs']['manual_review_csv']} --output-root {summary['output_root']}
3. Manually review the packet CSV.
4. Build synthetic guard cases from passing detections:
   python scripts/stage1_detector_pilot_control.py build-synthetic-guard --review-manifest {summary['output_root']}/review_packet/review_manifest.csv --output-root {summary['output_root']}
"""
    _ensure_parent(path)
    path.write_text(text)


def _copy_if_present(source: Path, dest: Path) -> str:
    if not source.exists():
        return ""
    _ensure_parent(dest)
    shutil.copy2(source, dest)
    return str(dest)


def cmd_export_review_packet(args: argparse.Namespace) -> int:
    worklist_csv = args.worklist_csv.resolve()
    output_root = args.output_root.resolve()
    packet_root = output_root / "review_packet"
    items_root = packet_root / "items"
    if not worklist_csv.exists():
        raise SystemExit(f"Worklist CSV does not exist: {worklist_csv}")

    rows = _read_csv(worklist_csv)
    review_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        pilot_row_id = row["pilot_row_id"]
        item_root = items_root / pilot_row_id
        stage1_dir = Path(row["expected_stage1_dir"])
        paths = _stage1_paths(stage1_dir)
        stage1_exists = all(paths[key].exists() for key in ("thumbnail", "overlay", "bboxes_json"))
        if not stage1_exists:
            missing.append(pilot_row_id)
            if not args.allow_missing:
                continue

        review_row = dict(row)
        review_row.update(
            {
                "stage1_status": "complete" if stage1_exists else "missing",
                "thumbnail_path": "",
                "overlay_path": "",
                "bboxes_json_path": "",
                "all_cores_found": "",
                "major_false_positive": "",
                "review_pass": "",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
        if stage1_exists:
            review_row["thumbnail_path"] = _copy_if_present(
                paths["thumbnail"], item_root / "thumbnail.png"
            )
            review_row["overlay_path"] = _copy_if_present(
                paths["overlay"], item_root / "bbox_overlay.png"
            )
            review_row["bboxes_json_path"] = _copy_if_present(
                paths["bboxes_json"], item_root / "bboxes.json"
            )
        review_rows.append(review_row)

    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)} rows are missing Stage 1 outputs. "
            "Use --allow-missing to export a status packet anyway."
        )
    if not review_rows:
        raise SystemExit("No review rows to export.")

    review_manifest = packet_root / "review_manifest.csv"
    fieldnames = list(review_rows[0].keys())
    _write_csv(review_manifest, review_rows, fieldnames)
    _write_review_html(packet_root / "index.html", review_rows)
    summary = {
        "created_at": _timestamp(),
        "worklist_csv": str(worklist_csv),
        "review_manifest": str(review_manifest),
        "index_html": str(packet_root / "index.html"),
        "rows": len(review_rows),
        "complete_stage1_rows": sum(row["stage1_status"] == "complete" for row in review_rows),
        "missing_stage1_rows": sum(row["stage1_status"] == "missing" for row in review_rows),
    }
    _write_json(packet_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _relative_html_src(index_path: Path, image_path: str) -> str:
    if not image_path:
        return ""
    try:
        return html.escape(os.path.relpath(image_path, start=index_path.parent))
    except ValueError:
        return html.escape(image_path)


def _write_review_html(index_path: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in rows:
        thumb_src = _relative_html_src(index_path, row.get("thumbnail_path", ""))
        overlay_src = _relative_html_src(index_path, row.get("overlay_path", ""))
        if row["stage1_status"] == "complete":
            images = (
                f'<figure><img src="{thumb_src}" alt="thumbnail"><figcaption>Thumbnail</figcaption></figure>'
                f'<figure><img src="{overlay_src}" alt="overlay"><figcaption>Detection overlay</figcaption></figure>'
            )
        else:
            images = '<p class="missing">Stage 1 outputs missing.</p>'
        title = html.escape(
            f"{row['pilot_row_id']} | {row.get('stain', '')} | {row.get('Anon_Path_ID', '')}"
        )
        cards.append(
            f"""
<section class="card">
  <h2>{title}</h2>
  <div class="meta">{html.escape(row.get('wsi_path', ''))}</div>
  <div class="images">{images}</div>
  <div class="review-fields">CSV fields: all_cores_found, major_false_positive, review_pass, notes</div>
</section>
"""
        )
    body = "\n".join(cards)
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Detector Pilot Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f5; color: #1c1c1c; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    .card {{ background: white; border: 1px solid #d8d8d2; border-radius: 6px; padding: 16px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .meta {{ font-size: 12px; color: #555; overflow-wrap: anywhere; margin-bottom: 12px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; align-items: start; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #ccc; background: #111; }}
    figcaption, .review-fields {{ font-size: 12px; color: #555; margin-top: 6px; }}
    .missing {{ color: #8a3b00; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Stage 1 Detector Pilot Manual Review</h1>
  {body}
</body>
</html>
"""
    _ensure_parent(index_path)
    index_path.write_text(text)


def _bool_cell(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in TRUTHY:
        return True
    if normalized in FALSEY:
        return False
    return None


def _load_detected_regions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    regions = payload.get("detected_regions")
    if not isinstance(regions, list):
        raise ValueError(f"No detected_regions list in {path}")
    return regions


def _bbox_from_region(region: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = region.get("bbox_thumbnail") or region.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Region does not contain a four-value bbox: {region}")
    x, y, w, h = [int(round(float(value))) for value in bbox]
    return x, y, x + w, y + h


def _draw_overlay(
    *,
    thumbnail_path: Path,
    regions: list[dict[str, Any]],
    output_path: Path,
    removed_index: int | None = None,
) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, region in enumerate(regions):
        if removed_index is not None and index == removed_index:
            continue
        x0, y0, x1, y1 = _bbox_from_region(region)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 40, 40), width=5)
        draw.text((x0 + 4, y0 + 4), str(index + 1), fill=(255, 255, 255))
    _ensure_parent(output_path)
    image.save(output_path)


def cmd_build_synthetic_guard(args: argparse.Namespace) -> int:
    review_manifest = args.review_manifest.resolve()
    output_root = args.output_root.resolve()
    guard_root = output_root / "synthetic_guard"
    if not review_manifest.exists():
        raise SystemExit(f"Review manifest does not exist: {review_manifest}")

    rows = _read_csv(review_manifest)
    rng = random.Random(args.seed)
    guard_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        if row.get("stage1_status") != "complete":
            skipped.append({"pilot_row_id": row.get("pilot_row_id", ""), "reason": "stage1_missing"})
            continue
        if not args.allow_unreviewed:
            review_pass = _bool_cell(row.get("review_pass", ""))
            if review_pass is not True:
                skipped.append({"pilot_row_id": row.get("pilot_row_id", ""), "reason": "review_not_passed"})
                continue

        thumbnail_path = Path(row["thumbnail_path"])
        overlay_path = Path(row["overlay_path"])
        bboxes_json_path = Path(row["bboxes_json_path"])
        regions = _load_detected_regions(bboxes_json_path)
        if not regions:
            skipped.append({"pilot_row_id": row.get("pilot_row_id", ""), "reason": "no_bboxes"})
            continue
        removed_index = rng.randrange(len(regions))
        item_root = guard_root / "items" / row["pilot_row_id"]
        control_overlay = item_root / "control_overlay.png"
        degraded_overlay = item_root / "drop_one_overlay.png"
        if overlay_path.exists():
            _copy_if_present(overlay_path, control_overlay)
        else:
            _draw_overlay(
                thumbnail_path=thumbnail_path,
                regions=regions,
                output_path=control_overlay,
            )
        _draw_overlay(
            thumbnail_path=thumbnail_path,
            regions=regions,
            output_path=degraded_overlay,
            removed_index=removed_index,
        )
        copied_thumb = item_root / "thumbnail.png"
        _copy_if_present(thumbnail_path, copied_thumb)

        common = {
            "pilot_row_id": row["pilot_row_id"],
            "case_id": row.get("case_id", ""),
            "Anon_Path_ID": row.get("Anon_Path_ID", ""),
            "stain": row.get("stain", ""),
            "thumbnail_path": str(copied_thumb),
            "bboxes_json_path": str(bboxes_json_path),
            "removed_bbox_index": removed_index,
            "removed_bbox_thumbnail": json.dumps(regions[removed_index].get("bbox_thumbnail", [])),
            "bbox_count_original": len(regions),
        }
        guard_rows.append(
            {
                **common,
                "guard_case_id": f"{row['pilot_row_id']}__control",
                "case_type": "original_control",
                "expected_missing_core": "false",
                "test_overlay_path": str(control_overlay),
            }
        )
        guard_rows.append(
            {
                **common,
                "guard_case_id": f"{row['pilot_row_id']}__drop_one",
                "case_type": "drop_one_bbox",
                "expected_missing_core": "true",
                "test_overlay_path": str(degraded_overlay),
            }
        )

    if not guard_rows:
        raise SystemExit("No synthetic guard rows created.")
    fieldnames = list(guard_rows[0].keys())
    guard_cases_csv = guard_root / "guard_cases.csv"
    guard_cases_jsonl = guard_root / "guard_cases.jsonl"
    _write_csv(guard_cases_csv, guard_rows, fieldnames)
    _write_jsonl(guard_cases_jsonl, guard_rows)
    summary = {
        "created_at": _timestamp(),
        "review_manifest": str(review_manifest),
        "guard_cases_csv": str(guard_cases_csv),
        "guard_cases_jsonl": str(guard_cases_jsonl),
        "guard_case_rows": len(guard_rows),
        "source_review_rows": len(rows),
        "skipped": skipped,
        "seed": args.seed,
    }
    _write_json(guard_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"raw_text": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw_text": text}


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    if args.backend == "openrouter":
        base_url = args.api_base or "https://openrouter.ai/api/v1"
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    elif args.backend == "vllm":
        base_url = args.api_base or "http://localhost:8000/v1"
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    else:
        raise SystemExit(f"Unsupported backend: {args.backend}")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return base_url, api_key


def cmd_run_guard(args: argparse.Namespace) -> int:
    guard_cases_csv = args.guard_cases_csv.resolve()
    output_root = args.output_root.resolve()
    guard_root = output_root / "synthetic_guard"
    if not guard_cases_csv.exists():
        raise SystemExit(f"Guard cases CSV does not exist: {guard_cases_csv}")
    rows = _read_csv(guard_cases_csv)
    if args.max_cases:
        rows = rows[: args.max_cases]

    prompt = (
        "You are reviewing tissue-core detection on a whole-slide thumbnail. "
        "Image 1 is the source thumbnail. Image 2 is the detection overlay. "
        "Determine whether any tissue core visible in Image 1 is missing a bounding box in Image 2. "
        "Return only JSON with keys: missing_core (boolean), confidence (0 to 1), explanation (short string)."
    )
    tasks_path = guard_root / "guard_tasks.jsonl"
    _write_jsonl(
        tasks_path,
        (
            {
                "guard_case_id": row["guard_case_id"],
                "model": args.model,
                "thumbnail_path": row["thumbnail_path"],
                "test_overlay_path": row["test_overlay_path"],
                "prompt": prompt,
            }
            for row in rows
        ),
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks_jsonl": str(tasks_path), "tasks": len(rows)}, indent=2))
        return 0

    from openai import OpenAI

    base_url, api_key = _api_settings(args)
    client = OpenAI(base_url=base_url, api_key=api_key)
    results: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "guard_case_id": row["guard_case_id"],
            "case_type": row["case_type"],
            "expected_missing_core": row["expected_missing_core"],
            "model": args.model,
            "created_at": _timestamp(),
        }
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_to_data_url(Path(row["thumbnail_path"]))},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_to_data_url(Path(row["test_overlay_path"]))},
                            },
                        ],
                    }
                ],
                temperature=args.temperature,
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json_object(raw)
            record.update(
                {
                    "raw_response": raw,
                    "predicted_missing_core": parsed.get("missing_core"),
                    "confidence": parsed.get("confidence"),
                    "explanation": parsed.get("explanation", ""),
                    "error": "",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "raw_response": "",
                    "predicted_missing_core": "",
                    "confidence": "",
                    "explanation": "",
                    "error": repr(exc),
                }
            )
        results.append(record)

    results_jsonl = guard_root / "guard_results.jsonl"
    results_csv = guard_root / "guard_results.csv"
    _write_jsonl(results_jsonl, results)
    _write_csv(results_csv, results, list(results[0].keys()))
    print(json.dumps({"results_jsonl": str(results_jsonl), "results_csv": str(results_csv)}, indent=2))
    return 0


def cmd_summarize_guard(args: argparse.Namespace) -> int:
    results_jsonl = args.results_jsonl.resolve()
    if not results_jsonl.exists():
        raise SystemExit(f"Results JSONL does not exist: {results_jsonl}")
    rows = [json.loads(line) for line in results_jsonl.read_text().splitlines() if line.strip()]
    counts = {
        "rows": len(rows),
        "true_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "false_positive": 0,
        "errors": 0,
        "unparsed": 0,
    }
    for row in rows:
        if row.get("error"):
            counts["errors"] += 1
            continue
        expected = _bool_cell(str(row.get("expected_missing_core", "")))
        predicted = _bool_cell(str(row.get("predicted_missing_core", "")))
        if expected is None or predicted is None:
            counts["unparsed"] += 1
        elif expected and predicted:
            counts["true_positive"] += 1
        elif expected and not predicted:
            counts["false_negative"] += 1
        elif not expected and not predicted:
            counts["true_negative"] += 1
        elif not expected and predicted:
            counts["false_positive"] += 1
    denominators = {
        "positive": counts["true_positive"] + counts["false_negative"],
        "negative": counts["true_negative"] + counts["false_positive"],
    }
    metrics = {
        "synthetic_missing_core_recall": (
            counts["true_positive"] / denominators["positive"]
            if denominators["positive"]
            else None
        ),
        "control_false_alarm_rate": (
            counts["false_positive"] / denominators["negative"]
            if denominators["negative"]
            else None
        ),
    }
    summary = {
        "created_at": _timestamp(),
        "results_jsonl": str(results_jsonl),
        "counts": counts,
        "metrics": metrics,
    }
    output = args.output_json.resolve()
    _write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-worklist", help="Create the balanced 100-WSI pilot worklist.")
    build.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    build.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument("--stage1-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_runs")
    build.add_argument("--run-id", default=DEFAULT_RUN_ID)
    build.add_argument("--stains", nargs="+", default=list(DEFAULT_STAINS))
    build.add_argument("--n-groups", type=int, default=20)
    build.add_argument("--manual-review-groups", type=int, default=4)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--stage1-model", default="google/gemini-3-flash-preview")
    build.add_argument("--stage1-max-dim", type=int, default=2048)
    build.add_argument("--stage1-api-base", default="https://openrouter.ai/api/v1")
    build.set_defaults(func=cmd_build_worklist)

    review = subparsers.add_parser("export-review-packet", help="Export a small manual review packet.")
    review.add_argument("--worklist-csv", type=Path, required=True)
    review.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    review.add_argument("--allow-missing", action="store_true")
    review.set_defaults(func=cmd_export_review_packet)

    synthetic = subparsers.add_parser("build-synthetic-guard", help="Drop one bbox to create guard cases.")
    synthetic.add_argument("--review-manifest", type=Path, required=True)
    synthetic.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Build cases even when review_pass is empty; useful only for local plumbing tests.",
    )
    synthetic.set_defaults(func=cmd_build_synthetic_guard)

    run_guard = subparsers.add_parser("run-guard", help="Run or dry-run the synthetic guard VLM calls.")
    run_guard.add_argument("--guard-cases-csv", type=Path, required=True)
    run_guard.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_guard.add_argument("--backend", choices=("openrouter", "vllm"), default="openrouter")
    run_guard.add_argument("--api-base", default="")
    run_guard.add_argument("--api-key", default="")
    run_guard.add_argument("--model", default="google/gemini-3-flash-preview")
    run_guard.add_argument("--temperature", type=float, default=0.0)
    run_guard.add_argument("--max-cases", type=int, default=0)
    run_guard.add_argument("--dry-run", action="store_true")
    run_guard.set_defaults(func=cmd_run_guard)

    summarize = subparsers.add_parser("summarize-guard", help="Summarize guard results.")
    summarize.add_argument("--results-jsonl", type=Path, required=True)
    summarize.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "synthetic_guard" / "guard_summary.json",
    )
    summarize.set_defaults(func=cmd_summarize_guard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
