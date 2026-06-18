#!/usr/bin/env python3
"""Extract frozen DINOv3 features for scale-500 verifier-selected FG/BG patches."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_per_wsi_dinov3_fg_bg_probe import (  # noqa: E402
    DEFAULT_DINOV3_SMALL,
    DEFAULT_DINOV2_SMALL,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOT,
    FeatureExtractor,
    build_case_inventory,
    extract_case_features,
    load_case_patches,
    package_versions,
    patch_dict,
    read_completed_manifest,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/scale500_selected_dinov3small_features_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs; default extracts every resolved manifest case.")
    parser.add_argument("--min-bboxes", type=int, default=1)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["openslide", "cucim"], default="openslide")
    parser.add_argument(
        "--read-workers",
        type=int,
        default=16,
        help="Number of cuCIM read workers per patch read; ignored when --wsi-reader openslide.",
    )
    parser.add_argument(
        "--sample-per-bbox-per-class",
        type=int,
        default=None,
        help="Randomly sample up to this many FG and BG patches per selected bbox crop before feature extraction.",
    )
    parser.add_argument(
        "--sample-max-per-wsi",
        type=int,
        default=None,
        help=(
            "Randomly sample at most this many total patches per WSI. "
            "The target is FG/BG balanced when possible and split evenly across selected bbox crops."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-patches-per-case", type=int, default=None)
    return parser.parse_args()


def sample_records_per_bbox_per_class(records: list[Any], sample_n: int | None, seed: int) -> tuple[list[Any], list[dict[str, Any]]]:
    if sample_n is None:
        return records, []
    if sample_n <= 0:
        raise ValueError("--sample-per-bbox-per-class must be positive")
    rng = np.random.default_rng(seed)
    original_index = {id(record): idx for idx, record in enumerate(records)}
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for record in records:
        grouped[(int(record.bbox_index), int(record.label_fg))].append(record)
    selected: list[Any] = []
    audit: list[dict[str, Any]] = []
    for bbox_index in sorted({key[0] for key in grouped}):
        for label_fg in (1, 0):
            items = grouped.get((bbox_index, label_fg), [])
            take = min(sample_n, len(items))
            if take and take < len(items):
                chosen_idx = rng.choice(len(items), size=take, replace=False)
                chosen = [items[int(idx)] for idx in chosen_idx]
            else:
                chosen = list(items)
            selected.extend(chosen)
            audit.append(
                {
                    "bbox_index": bbox_index,
                    "label_fg": label_fg,
                    "available": len(items),
                    "sampled": len(chosen),
                    "target": sample_n,
                    "shortfall": max(0, sample_n - len(items)),
                }
            )
    selected.sort(key=lambda record: original_index[id(record)])
    return selected, audit


def split_quota(total: int, parts: list[int]) -> dict[int, int]:
    if not parts:
        return {}
    base, rem = divmod(total, len(parts))
    return {part: base + (1 if idx < rem else 0) for idx, part in enumerate(sorted(parts))}


def sample_records_per_wsi_even_crops(
    records: list[Any],
    max_patches: int | None,
    seed: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if max_patches is None:
        return records, []
    if max_patches <= 0:
        raise ValueError("--sample-max-per-wsi must be positive")
    rng = np.random.default_rng(seed)
    original_index = {id(record): idx for idx, record in enumerate(records)}
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    bbox_indices = sorted({int(record.bbox_index) for record in records})
    for record in records:
        grouped[(int(record.bbox_index), int(record.label_fg))].append(record)

    target_total = min(max_patches, len(records))
    available = {
        1: sum(len(grouped.get((bbox_index, 1), [])) for bbox_index in bbox_indices),
        0: sum(len(grouped.get((bbox_index, 0), [])) for bbox_index in bbox_indices),
    }
    target = {1: min(target_total // 2, available[1]), 0: min(target_total - min(target_total // 2, available[1]), available[0])}
    remaining = target_total - target[1] - target[0]
    for label_fg in sorted((1, 0), key=lambda label: available[label] - target[label], reverse=True):
        if remaining <= 0:
            break
        add = min(remaining, available[label_fg] - target[label_fg])
        target[label_fg] += add
        remaining -= add

    selected: list[Any] = []
    selected_ids: set[int] = set()
    audit: list[dict[str, Any]] = []
    for label_fg in (1, 0):
        quotas = split_quota(target[label_fg], bbox_indices)
        label_shortfall = 0
        for bbox_index in bbox_indices:
            items = grouped.get((bbox_index, label_fg), [])
            quota = quotas.get(bbox_index, 0)
            take = min(quota, len(items))
            if take and take < len(items):
                chosen_idx = rng.choice(len(items), size=take, replace=False)
                chosen = [items[int(idx)] for idx in chosen_idx]
            else:
                chosen = list(items)
            selected.extend(chosen)
            selected_ids.update(id(record) for record in chosen)
            shortfall = max(0, quota - len(items))
            label_shortfall += shortfall
            audit.append(
                {
                    "sampling_mode": "per_wsi_even_crops",
                    "bbox_index": bbox_index,
                    "label_fg": label_fg,
                    "available": len(items),
                    "target": quota,
                    "sampled": len(chosen),
                    "shortfall": shortfall,
                    "max_patches_per_wsi": max_patches,
                    "target_label_total": target[label_fg],
                }
            )
        if label_shortfall:
            leftovers = [
                record
                for bbox_index in bbox_indices
                for record in grouped.get((bbox_index, label_fg), [])
                if id(record) not in selected_ids
            ]
            fill = min(label_shortfall, len(leftovers))
            chosen: list[Any] = []
            if fill:
                chosen_idx = rng.choice(len(leftovers), size=fill, replace=False)
                chosen = [leftovers[int(idx)] for idx in chosen_idx]
                selected.extend(chosen)
                selected_ids.update(id(record) for record in chosen)
            audit.append(
                {
                    "sampling_mode": "per_wsi_even_crops_backfill",
                    "bbox_index": -1,
                    "label_fg": label_fg,
                    "available": len(leftovers),
                    "target": label_shortfall,
                    "sampled": len(chosen),
                    "shortfall": max(0, label_shortfall - len(leftovers)),
                    "max_patches_per_wsi": max_patches,
                    "target_label_total": target[label_fg],
                }
            )
    selected.sort(key=lambda record: original_index[id(record)])
    return selected, audit


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    command = [
        "python",
        "scripts/extract_scale500_selected_dinov3_features.py",
        "--run-root",
        str(args.run_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--model-backend",
        str(args.model_backend),
        "--model-name",
        str(args.model_name),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
    ]
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    if args.case_ids:
        command.extend(["--case-ids", args.case_ids])
    if args.max_patches_per_case is not None:
        command.extend(["--max-patches-per-case", str(args.max_patches_per_case)])
    if args.sample_per_bbox_per_class is not None:
        command.extend(["--sample-per-bbox-per-class", str(args.sample_per_bbox_per_class)])
        command.extend(["--sample-seed", str(args.sample_seed)])
    if args.sample_max_per_wsi is not None:
        command.extend(["--sample-max-per-wsi", str(args.sample_max_per_wsi)])
        command.extend(["--sample-seed", str(args.sample_seed)])
    lines = [
        "PER-250 Scale-500 Selected-Crop DINOv3 Feature Cache",
        "====================================================",
        "",
        f"Created: {summary['created_at']}",
        f"Ticket: {args.ticket}",
        f"Git commit: {summary['git_commit']}",
        f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
        "",
        "Command:",
        " ".join(shlex.quote(part) for part in command),
        "",
        "Inputs:",
        f"- Auto-context run root: {args.run_root.resolve()}",
        f"- Completed-case manifest: {args.manifest.resolve()}",
        "- Labels are Stage 7 postprocessed FG/BG masks aligned to Stage 6 patch grids.",
        (
            f"- Sampling: sample_per_bbox_per_class={args.sample_per_bbox_per_class}; "
            f"sample_max_per_wsi={args.sample_max_per_wsi}; sample_seed={args.sample_seed}."
        ),
        (
            "- For --sample-max-per-wsi, the per-WSI target is split as evenly as possible "
            "between FG/BG, each class quota is split evenly across selected bbox crops, "
            "and crop shortfalls are backfilled from the same WSI/class."
        ),
        "- WSI pixels are read on demand from manifest WSI links using Stage 6 level-0 patch coordinates.",
        f"- WSI reader: {args.wsi_reader}; read_workers={args.read_workers}.",
        (
            f"- Resume enabled: {args.resume}; feature cache files present at run start: "
            f"{summary['resume']['feature_cache_count_at_start']}."
        ),
        "- Resume compatibility is keyed on feature model/backend; existing cache files may have been created by an earlier reader backend.",
        "",
        "Outputs:",
        f"- Feature cache: {(args.output_dir / 'features').resolve()}",
        f"- Selected WSI manifest: {(args.output_dir / 'manifests/selected_wsis.csv').resolve()}",
        f"- Selected patch manifest: {(args.output_dir / 'manifests/selected_patch_manifest.csv').resolve()}",
        f"- Summary: {(args.output_dir / 'summary.json').resolve()}",
        "",
    ]
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.sample_per_bbox_per_class is not None and args.sample_max_per_wsi is not None:
        raise ValueError("Use only one sampling mode: --sample-per-bbox-per-class or --sample-max-per-wsi")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())
    existing_feature_cache_count_at_start = len(list((args.output_dir / "features").glob("*_features.npz")))
    requested = {part.strip() for part in args.case_ids.split(",") if part.strip()}
    created_at = datetime.now(timezone.utc).isoformat()

    manifest_rows = read_completed_manifest(args.manifest, args.run_root)
    cases = build_case_inventory(manifest_rows, min_bboxes=args.min_bboxes)
    if requested:
        cases = [case for case in cases if case.case_id in requested]
    cases.sort(key=lambda case: (case.stain, case.task, case.case_id))
    if args.case_limit is not None:
        cases = cases[: int(args.case_limit)]
    if not cases:
        raise ValueError("No cases selected for feature extraction")

    extractor = FeatureExtractor(args)
    all_patch_rows: list[dict[str, Any]] = []
    sample_audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        started = time.monotonic()
        records = load_case_patches(case)
        original_record_count = len(records)
        if args.max_patches_per_case is not None and len(records) > args.max_patches_per_case:
            records = records[: int(args.max_patches_per_case)]
        if args.sample_max_per_wsi is not None:
            records, sample_audit = sample_records_per_wsi_even_crops(
                records,
                args.sample_max_per_wsi,
                seed=int(args.sample_seed) + int(case.task) * 100003,
            )
        else:
            records, sample_audit = sample_records_per_bbox_per_class(
                records,
                args.sample_per_bbox_per_class,
                seed=int(args.sample_seed) + int(case.task) * 100003,
            )
        for item in sample_audit:
            sample_audit_rows.append({"case_id": case.case_id, "task": case.task, "stain": case.stain, **item})
        all_patch_rows.extend(patch_dict(record) for record in records)
        print(
            f"[{index}/{len(cases)}] {case.case_id} {case.stain}: "
            f"{len(records)} sampled patches from {original_record_count} available across {case.bbox_count} selected bboxes",
            flush=True,
        )
        features, labels, bbox_indices, _record_indices, case_failures = extract_case_features(
            case,
            records,
            extractor,
            args.output_dir,
            resume=bool(args.resume),
            wsi_reader=args.wsi_reader,
            read_workers=args.read_workers,
        )
        failures.extend({"case_id": case.case_id, **failure} for failure in case_failures)
        case_summaries.append(
            {
                "case_id": case.case_id,
                "task": case.task,
                "stain": case.stain,
                "patches_available_before_sampling": original_record_count,
                "patches_loaded": len(records),
                "features_extracted": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "fg_features": int(labels.sum()),
                "bg_features": int(len(labels) - labels.sum()),
                "bbox_count": len(set(bbox_indices.tolist())),
                "feature_failures": len(case_failures),
                "seconds": time.monotonic() - started,
            }
        )

    selected_rows = [
        {
            "task": case.task,
            "case_id": case.case_id,
            "stain": case.stain,
            "wsi_path": case.wsi_path,
            "source_wsi_path": case.source_wsi_path,
            "run_dir": str(case.run_dir),
            "bbox_count": case.bbox_count,
            "fg_count": case.fg_count,
            "bg_count": case.bg_count,
            "verifier_selected_box_ids": case.verifier_selected_box_ids,
        }
        for case in cases
    ]
    write_csv(args.output_dir / "manifests/selected_wsis.csv", selected_rows)
    write_csv(args.output_dir / "manifests/selected_patch_manifest.csv", all_patch_rows)
    write_csv(args.output_dir / "manifests/sample_audit.csv", sample_audit_rows)
    write_json(args.output_dir / "failures.json", failures)

    summary = {
        "created_at": created_at,
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "command": sys.argv[:],
        "package_versions": package_versions(),
        "input": {
            "run_root": str(args.run_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_rows_resolved": len(manifest_rows),
        },
        "selection": {
            "case_count": len(cases),
            "case_limit": args.case_limit,
            "case_ids": sorted(requested),
            "min_bboxes": args.min_bboxes,
            "sample_per_bbox_per_class": args.sample_per_bbox_per_class,
            "sample_max_per_wsi": args.sample_max_per_wsi,
            "sample_seed": args.sample_seed,
        },
        "label_policy": {
            "positive_class": "foreground",
            "negative_class": "background",
            "source": "stage7/tissue_mask_post.npy aligned to stage6/patches.csv row/col",
        },
        "patch_reader": {
            "wsi_reader": args.wsi_reader,
            "read_workers": args.read_workers,
        },
        "resume": {
            "enabled": bool(args.resume),
            "feature_cache_count_at_start": existing_feature_cache_count_at_start,
        },
        "feature_extractor": extractor.meta,
        "case_summaries": case_summaries,
        "totals": {
            "patches": int(sum(item["features_extracted"] for item in case_summaries)),
            "fg": int(sum(item["fg_features"] for item in case_summaries)),
            "bg": int(sum(item["bg_features"] for item in case_summaries)),
            "bboxes": int(sum(item["bbox_count"] for item in case_summaries)),
            "failures": len(failures),
        },
        "outputs": {
            "selected_wsis_csv": str((args.output_dir / "manifests/selected_wsis.csv").resolve()),
            "patch_manifest_csv": str((args.output_dir / "manifests/selected_patch_manifest.csv").resolve()),
            "sample_audit_csv": str((args.output_dir / "manifests/sample_audit.csv").resolve()),
            "feature_dir": str((args.output_dir / "features").resolve()),
            "reproduction_txt": str((args.output_dir / "reproduction.txt").resolve()),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
