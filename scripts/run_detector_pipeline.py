#!/usr/bin/env python3
"""Run the detector-oracle bbox pipeline on arbitrary WSI inputs.

This entrypoint accepts a single WSI path, a directory of WSIs, or a text file
with one WSI path per line. It runs the detector-oracle flow: thumbnail Stage 1
detection, the two-step Stage 2 review/router, optional Stage 3 feedback
redetection, optional high-resolution crop redetection, crop classification,
comparative thumbnail-crop filtering, and final bbox export.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import (
    _chat_with_images,
    _extract_json_payload,
    _font,
    _load_raw_orientation_bboxes,
    _normalised_detection_items,
    _repo_git_commit,
    _safe_slug,
    _timestamp,
)
from stage1_review_trigger_router import _chat_text, _parse_router_response
from stage4_crop_prompt_packet import (
    _normalised_yxyx_to_level0,
    _pad_level0_bbox,
    _read_padded_crop,
)
from stage6_crop_tp_fp_review import _parse_tissue_yes_no
from stage6_odd_one_out_artifact_review import _parse_response as _parse_odd_one_out_response
from stage7_post_stage3_crop_redetect_pipeline import (
    _crop_pixel_bbox_to_wsi_norm,
    _draw_boxes_overlay,
    _draw_crop_detection_overlay,
    _draw_odd_sheet,
    _expand_yxyx,
    _merge_yxyx_boxes,
    _norm_to_image_bbox,
    _save_vlm_jpeg,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DETECTOR = REPO_ROOT / "detect_foreground_regions_from_wsi_thumbnail.py"
DEFAULT_STAGE1_PROMPT = REPO_ROOT / "prompts/stage1_high_recall_potential_tissue_candidates.txt"
DEFAULT_STAGE2A_PROMPT = (
    REPO_ROOT / "prompts/stage1_detector_oracle/stage2a_missed_or_overcoverage_review.txt"
)
DEFAULT_STAGE2B_FIRST_PROMPT = (
    REPO_ROOT / "prompts/stage1_detector_oracle/stage2b_nonminor_detection_failure_json.txt"
)
DEFAULT_STAGE2B_SECOND_PROMPT = (
    REPO_ROOT / "prompts/stage1_detector_oracle/stage2b_nonminor_detection_failure_adjudicate_json.txt"
)
DEFAULT_STAGE3_WRAPPER_PROMPT = (
    REPO_ROOT / "prompts/stage1_detector_oracle/stage3_refinement_minimal_wrapper.txt"
)
DEFAULT_CLASSIFICATION_PROMPT = (
    REPO_ROOT / "prompts/stage1_detector_oracle/stage6_crop_true_false_positive.txt"
)
DEFAULT_ODD_ONE_OUT_PROMPT = (
    REPO_ROOT
    / "prompts/stage1_detector_oracle/stage6_odd_one_out_artifact_review_v2_contains_consensus.txt"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
PROMPT_VERSION = "detector_pipeline_arbitrary_wsi_v1_2026-05-29"
TICKET = "PER-207"
SUPPORTED_WSI_EXTENSIONS = (
    ".svs",
    ".ndpi",
    ".tif",
    ".tiff",
    ".isyntax",
    ".mrxs",
    ".scn",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _existing_file_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _stage_contract(args: argparse.Namespace) -> list[dict[str, Any]]:
    stage2_output = (
        "Skipped by --skip-stage2-review; Stage 3 is not triggered and Stage 4 "
        "uses Stage 1 raw boxes."
        if args.skip_stage2_review
        else (
            "Stage 2a free-text review plus Stage 2b two-pass text router. "
            "The final Stage 2b non-minor-failure boolean controls whether "
            "Stage 3 feedback redetection runs unless --force-stage3-redetect "
            "is set."
        )
    )
    stage3_output = (
        "Skipped because --skip-stage2-review disables the review trigger."
        if args.skip_stage2_review
        else (
            "Runs only for Stage 2b-positive cases. Positive cases get new "
            "thumbnail detections from the Stage 3 wrapper prompt; negative "
            "cases keep the Stage 1 raw boxes unless --force-stage3-redetect "
            "is set."
        )
    )
    stage4_output = (
        "Active boxes are merged with IoU/overlap-over-smaller logic. The "
        "unpadded merged boxes are forwarded directly to Stage 5; 15% expanded "
        "boxes are still recorded for comparison, but no Stage 4 crops are read."
        if args.skip_crop_redetect
        else (
            "Active boxes are merged with IoU/overlap-over-smaller logic, then padded "
            f"by {args.post_stage3_padding_frac:.3f}; these define the high-res WSI crops."
        )
    )
    crop_redetect_output = (
        "Skipped by --skip-crop-redetect; Stage 5 reads the Stage 4 merged "
        "boxes directly."
        if args.skip_crop_redetect
        else (
            "Crop-relative Stage 1 detections mapped back into full-WSI "
            "normalized 0-1000 coordinates."
        )
    )
    stage5_input = (
        "No VLM input. Uses Stage 4 merged boxes directly because crop redetection is skipped."
        if args.skip_crop_redetect
        else "No VLM input. Uses mapped high-res crop-redetection boxes."
    )
    return [
        {
            "stage": "stage1_thumbnail_detection",
            "prompt": str(args.stage1_prompt.resolve()),
            "input_image": (
                "Whole-slide thumbnail read from the source WSI, resized to "
                f"max_dim={args.stage1_thumbnail_max_dim}."
            ),
            "output": (
                "Raw per-orientation tissue-candidate bboxes in normalized "
                "0-1000 y_min,x_min,y_max,x_max coordinates, plus the Stage 1 "
                "thumbnail and overlay."
            ),
        },
        {
            "stage": "stage2_detection_review_router",
            "prompt": (
                f"2a: {args.stage2a_prompt.resolve()}; "
                f"2b first: {args.stage2b_first_prompt.resolve()}; "
                f"2b second: {args.stage2b_second_prompt.resolve()}"
            ),
            "input_image": (
                "Stage 2a sees the source thumbnail and Stage 1 raw-overlay image. "
                "Stage 2b is text-only over the Stage 2a review."
            ),
            "output": stage2_output,
        },
        {
            "stage": "stage3_feedback_redetection",
            "prompt": (
                f"wrapper: {args.stage3_wrapper_prompt.resolve()}; "
                f"task: {args.stage1_prompt.resolve()}"
            ),
            "input_image": (
                "Original whole-slide thumbnail plus previous Stage 1 raw-overlay image, "
                "only for Stage 2b-positive cases or forced Stage 3 runs."
            ),
            "output": stage3_output,
        },
        {
            "stage": "stage4_high_res_crop_redetect",
            "prompt": str(args.stage1_prompt.resolve()),
            "input_image": (
                "High-resolution WSI crops read from Stage 4 postprocessed active boxes, "
                f"with target max_dim={args.crop_max_dim}."
                if not args.skip_crop_redetect
                else "Skipped by --skip-crop-redetect."
            ),
            "output": stage4_output + " " + crop_redetect_output,
        },
        {
            "stage": "stage5_post_redetect_merge_and_crop",
            "prompt": None,
            "input_image": stage5_input,
            "output": (
                "Input boxes merged again, then reread as classification crops "
                f"with {args.classification_padding_frac:.3f} padding and "
                f"target max_dim={args.classification_max_dim}."
            ),
        },
        {
            "stage": "stage6_tissue_artifact_classification",
            "prompt": str(args.classification_prompt.resolve()),
            "input_image": (
                "High-resolution Stage 5 classification crop with the selected bbox overlaid."
            ),
            "output": "Per-crop yes/no/unknown decision for whether the box is focused on tissue.",
        },
        {
            "stage": "stage7_comparative_thumbnail_filter",
            "prompt": str(args.odd_one_out_prompt.resolve()),
            "input_image": (
                "Thumbnail crops from the source WSI thumbnail for every remaining tissue-positive box; "
                "this stage runs only when more than one crop remains."
            ),
            "output": "Candidate orders flagged as tissue-artifact outliers by comparative review.",
        },
        {
            "stage": "final_detections",
            "prompt": None,
            "input_image": "Source WSI thumbnail for visualization only.",
            "output": (
                "Final retained normalized 0-1000 y_min,x_min,y_max,x_max boxes, "
                "a final overlay PNG, and detections.json."
            ),
        },
    ]


def _redacted_argv(argv: list[str]) -> str:
    redacted: list[str] = []
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--api-key":
            redacted.extend([token, "<redacted>"])
            skip_next = True
        elif token.startswith("--api-key="):
            redacted.append("--api-key=<redacted>")
        else:
            redacted.append(token)
    return " ".join(shlex.quote(part) for part in redacted)


def _resolve_chat_settings(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    if args.backend == "vllm":
        return args.api_base or args.vllm_url, api_key or "EMPTY"
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set --api-key, --api-key-stdin, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        )
    return args.api_base or args.openrouter_url, api_key


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    explicit = [value for value in (args.wsi, args.wsi_dir, args.wsi_list) if value is not None]
    if args.input is not None:
        if explicit:
            raise SystemExit("Use either positional input or explicit --wsi/--wsi-dir/--wsi-list, not both.")
        source = args.input
        if source.is_dir():
            args.wsi_dir = source
        elif source.suffix.lower() == ".txt":
            args.wsi_list = source
        else:
            args.wsi = source
    elif len(explicit) != 1:
        raise SystemExit("Provide exactly one input: positional input, --wsi, --wsi-dir, or --wsi-list.")

    if args.wsi is not None:
        paths = [args.wsi]
    elif args.wsi_list is not None:
        base = args.wsi_list.resolve().parent
        paths = []
        for line in args.wsi_list.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            paths.append(path if path.is_absolute() else base / path)
    else:
        wsi_dir = args.wsi_dir
        assert wsi_dir is not None
        extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.wsi_exts}
        iterator = wsi_dir.rglob("*") if args.recursive else wsi_dir.glob("*")
        paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)

    resolved = [path.expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise SystemExit("Missing WSI path(s):\n" + "\n".join(missing[:20]))
    if not resolved:
        raise SystemExit("No WSI paths resolved from input.")
    return resolved


def _case_records_for_paths(wsi_paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    single_case = len(wsi_paths) == 1
    records: list[dict[str, Any]] = []
    for index, wsi_path in enumerate(wsi_paths, start=1):
        stem_slug = _safe_slug(wsi_path.stem)
        counts[stem_slug] += 1
        case_slug = stem_slug if counts[stem_slug] == 1 else f"{stem_slug}_{counts[stem_slug]:02d}"
        case_dir = output_dir if single_case else output_dir / case_slug
        records.append(
            {
                "case_index": index,
                "case_id": case_slug,
                "case_slug": case_slug,
                "case_display": wsi_path.name,
                "wsi_path": str(wsi_path),
                "case_dir": str(case_dir),
                "artifacts_dir": str(case_dir / "intermediate_stage_artifacts"),
                "errors": [],
            }
        )
    return records


def _parallel_map(
    items: list[Any],
    fn: Callable[[Any], Any],
    max_workers: int,
    stage_name: str,
) -> list[Any]:
    if not items:
        return []
    print(f"{stage_name}: {len(items)} item(s), max_concurrent={max_workers}", flush=True)
    if max_workers <= 1 or len(items) == 1:
        return [fn(item) for item in items]
    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_stage1_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    case = {**case, "errors": list(case.get("errors") or [])}
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage1_thumbnail_detection"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "stage1_command.log"
    bboxes_path = stage_dir / "bboxes.json"
    metadata_path = stage_dir / "metadata.json"
    thumbnail_path = stage_dir / "thumbnail.png"

    if not (args.reuse_existing and bboxes_path.exists() and metadata_path.exists()):
        command = [
            sys.executable,
            str(STAGE1_DETECTOR),
            "--wsi",
            case["wsi_path"],
            "--wsi-reader",
            args.wsi_reader,
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--max-dim",
            str(args.stage1_thumbnail_max_dim),
            "--coord-order",
            args.stage1_coord_order,
            "--padding",
            str(args.stage1_padding_frac),
            "--merge-overlap-threshold",
            str(args.stage1_merge_overlap_threshold),
            "--rotations",
            *(str(rotation) for rotation in args.stage1_rotations),
            "--max-retries",
            str(args.stage1_max_retries),
            "--prompt",
            str(args.stage1_prompt),
            "--output-dir",
            str(stage_dir),
            "--skip-dvc-check",
        ]
        if args.stage1_repair_model:
            command.extend(["--repair-model", args.stage1_repair_model])
        if args.save_all_stage_artifacts:
            command.append("--save-intermediate")
        if args.backend == "openrouter":
            command.extend(["--openrouter-url", base_url])
        elif args.backend == "vllm":
            command.extend(["--vllm-url", base_url])
        env = os.environ.copy()
        if api_key:
            env["OPENROUTER_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        if args.skip_repro:
            env["WSI_SKIP_STAGE_REPRO_CHECK"] = "1"
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(
            "Command:\n"
            + _redacted_argv(command)
            + "\n\nExit code: "
            + str(completed.returncode)
            + "\n\nOutput:\n"
            + completed.stdout
        )
        if completed.returncode != 0:
            case["errors"].append(
                {
                    "stage": "stage1_thumbnail_detection",
                    "message": f"Stage 1 command failed with exit code {completed.returncode}",
                    "log_path": str(log_path),
                }
            )

    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    wsi_dims = metadata.get("wsi_dimensions") or {}
    wsi_size = [
        int(wsi_dims.get("width") or 0),
        int(wsi_dims.get("height") or 0),
    ]
    thumbnail_size = None
    if thumbnail_path.is_file():
        with Image.open(thumbnail_path) as image:
            thumbnail_size = image.size

    raw_boxes: list[dict[str, Any]] = []
    raw_note = ""
    if bboxes_path.exists() and thumbnail_size is not None:
        raw_boxes, raw_note = _load_raw_orientation_bboxes(
            bboxes_path,
            thumbnail_size,
            int(args.stage1_source_rotation),
        )
    if raw_note and bboxes_path.exists():
        case["errors"].append(
            {
                "stage": "stage1_thumbnail_detection",
                "message": raw_note,
                "log_path": str(log_path),
            }
        )

    source_boxes = [
        [float(value) for value in bbox["box_2d_yxyx_normalized"]]
        for bbox in raw_boxes
        if bbox.get("box_2d_yxyx_normalized")
    ]
    raw_overlay_path = stage_dir / f"stage1_raw_rot{int(args.stage1_source_rotation)}_overlay.png"
    if thumbnail_path.is_file():
        _draw_boxes_overlay(
            thumbnail_path,
            raw_overlay_path,
            source_boxes,
            f"Stage 1 raw rot{int(args.stage1_source_rotation)}: {len(source_boxes)}",
        )

    case.update(
        {
            "stage1_dir": str(stage_dir),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else "",
            "stage1_bboxes_json_path": str(bboxes_path) if bboxes_path.exists() else "",
            "stage1_metadata_path": str(metadata_path) if metadata_path.exists() else "",
            "stage1_command_log_path": str(log_path),
            "stage1_source_stage": f"stage1_raw_rot{int(args.stage1_source_rotation)}",
            "source_detector_overlay_path": str(raw_overlay_path) if raw_overlay_path.exists() else "",
            "stage1_source_boxes_yxyx_normalized": source_boxes,
            "stage1_source_box_count": len(source_boxes),
            "stage1_raw_parse_status": raw_note or ("ok" if source_boxes else "empty"),
            "source_boxes_yxyx_normalized": source_boxes,
            "source_box_count": len(source_boxes),
            "active_source_stage": f"stage1_raw_rot{int(args.stage1_source_rotation)}",
            "active_detector_overlay_path": str(raw_overlay_path) if raw_overlay_path.exists() else "",
            "active_boxes_yxyx_normalized": source_boxes,
            "active_box_count": len(source_boxes),
            "wsi_size": wsi_size,
            "wsi_reader": metadata.get("wsi_reader", args.wsi_reader),
        }
    )
    return case


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _stage2_first_prompt(prompt_template: str, review_record: dict[str, Any]) -> str:
    return (
        prompt_template.strip()
        + "\n\nCase:\n"
        + str(review_record.get("case_display") or "")
        + "\n\nReviewer metadata:\n"
        + json.dumps(
            {
                "task_id": review_record.get("task_id", ""),
                "reviewer_model": review_record.get("model", ""),
                "reviewer_reasoning_effort": review_record.get("reasoning_effort", ""),
                "reviewed_bbox_count": review_record.get("reviewed_bbox_count", ""),
                "overlay_kind": review_record.get("overlay_kind", ""),
                "raw_response_status": review_record.get("raw_response_status", ""),
            },
            sort_keys=True,
        )
        + "\n\nReview text:\n"
        + str(review_record.get("raw_response") or "").strip()
    )


def _stage2_second_prompt(
    prompt_template: str,
    review_record: dict[str, Any],
    first_raw: str,
    first_parsed: dict[str, Any],
) -> str:
    return (
        prompt_template.strip()
        + "\n\nCase:\n"
        + str(review_record.get("case_display") or "")
        + "\n\nOriginal reviewer output:\n"
        + str(review_record.get("raw_response") or "").strip()
        + "\n\nInitial answer:\n"
        + json.dumps(
            {
                "raw_response": first_raw,
                "parsed_response": first_parsed,
                "answer": first_parsed.get("answer", ""),
                "justification": first_parsed.get("justification", first_parsed.get("rationale", "")),
            },
            sort_keys=True,
        )
    )


def _run_stage2a_case(
    case: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    case = {**case, "errors": list(case.get("errors") or [])}
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage2_detection_review_router"
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "stage2a_detection_review.json"
    review_overlay_path = stage_dir / "stage2a_stage1_raw_overlay.png"
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))

    boxes = [
        [float(value) for value in box]
        for box in case.get("stage1_source_boxes_yxyx_normalized", case.get("source_boxes_yxyx_normalized", []))
    ]
    if thumbnail_path is not None:
        _draw_boxes_overlay(
            thumbnail_path,
            review_overlay_path,
            boxes,
            f"Stage 2a review input: {len(boxes)} Stage 1 raw box(es)",
        )

    if args.skip_stage2_review:
        record = {
            "task_id": f"{case['case_slug']}_stage2a",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage2a_detection_review",
            "skipped": True,
            "skip_reason": "skip_stage2_review",
            "raw_response": "",
            "parsed_response": {"raw_text": ""},
            "error": "",
            "review_overlay_path": str(review_overlay_path) if review_overlay_path.exists() else "",
            "reviewed_bbox_count": len(boxes),
            "created_at": _timestamp(),
        }
        _write_json(result_path, record)
    elif args.reuse_existing and result_path.exists():
        record = _read_json(result_path)
    else:
        record = {
            "task_id": f"{case['case_slug']}_stage2a",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage2a_detection_review",
            "prompt_file": str(args.stage2a_prompt),
            "prompt_version": f"{args.stage2a_prompt.stem}_integrated",
            "model": args.model,
            "reasoning_effort": args.stage2a_reasoning_effort,
            "thumbnail_path": case.get("thumbnail_path", ""),
            "review_overlay_path": str(review_overlay_path) if review_overlay_path.exists() else "",
            "overlay_kind": "stage1_raw_overlay",
            "reviewed_bbox_count": len(boxes),
            "raw_response_status": case.get("stage1_raw_parse_status", ""),
            "raw_response": "",
            "parsed_response": {},
            "usage": {},
            "response_model": "",
            "error": "",
            "created_at": _timestamp(),
        }
        try:
            if thumbnail_path is None or not review_overlay_path.exists():
                raise FileNotFoundError("Missing Stage 2a thumbnail or review overlay input.")
            raw, usage, response_model = _chat_with_images(
                model=args.model,
                prompt_text=prompt,
                image_paths=[thumbnail_path, review_overlay_path],
                temperature=args.temperature,
                max_tokens=args.stage2a_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.stage2a_reasoning_effort,
            )
            record.update(
                {
                    "raw_response": raw,
                    "parsed_response": {"raw_text": raw},
                    "usage": usage,
                    "response_model": response_model,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
        _write_json(result_path, record)

    if record.get("error"):
        case["errors"].append(
            {
                "stage": "stage2a_detection_review",
                "message": record["error"],
                "result_path": str(result_path),
            }
        )
    case.update(
        {
            "stage2_dir": str(stage_dir),
            "stage2a_review_result_path": str(result_path),
            "stage2a_review_overlay_path": record.get("review_overlay_path", ""),
            "stage2a_review_text": record.get("raw_response", ""),
            "stage2a_review_error": record.get("error", ""),
            "stage2a_review_skipped": bool(record.get("skipped", False)),
        }
    )
    return case


def _run_stage2b_case(
    case: dict[str, Any],
    first_prompt: str,
    second_prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    case = {**case, "errors": list(case.get("errors") or [])}
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage2_detection_review_router"
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "stage2b_nonminor_router.json"
    stage2a_path = _existing_file_path(case.get("stage2a_review_result_path"))
    stage2a_record = _read_json(stage2a_path) if stage2a_path is not None else {}

    if args.skip_stage2_review:
        record = {
            "task_id": f"{case['case_slug']}_stage2b",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage2b_nonminor_router",
            "skipped": True,
            "skip_reason": "skip_stage2_review",
            "final_non_minor_detection_failure": False,
            "final_answer": "no",
            "final_justification": "Skipped because Stage 2 review is disabled.",
            "error": "",
            "created_at": _timestamp(),
        }
        _write_json(result_path, record)
    elif args.reuse_existing and result_path.exists():
        record = _read_json(result_path)
    else:
        record = {
            "task_id": f"{case['case_slug']}_stage2b",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage2b_nonminor_router",
            "source_stage2a_result_path": str(stage2a_path) if stage2a_path else "",
            "source_review_text": stage2a_record.get("raw_response", ""),
            "first_prompt_file": str(args.stage2b_first_prompt),
            "second_prompt_file": str(args.stage2b_second_prompt),
            "model": args.model,
            "reasoning_effort": args.stage2b_reasoning_effort,
            "first_raw_response": "",
            "first_parsed_response": {},
            "second_raw_response": "",
            "second_parsed_response": {},
            "ran_second_pass": False,
            "second_pass_skip_reason": "",
            "first_non_minor_detection_failure": "",
            "first_justification": "",
            "final_non_minor_detection_failure": "",
            "final_answer": "",
            "final_justification": "",
            "first_usage": {},
            "second_usage": {},
            "first_response_model": "",
            "second_response_model": "",
            "error": "",
            "created_at": _timestamp(),
        }
        try:
            if stage2a_record.get("error"):
                raise RuntimeError(f"Stage 2a review errored: {stage2a_record['error']}")
            first_raw, first_usage, first_response_model = _chat_text(
                model=args.model,
                prompt_text=_stage2_first_prompt(first_prompt, stage2a_record),
                temperature=args.temperature,
                max_tokens=args.stage2b_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.stage2b_reasoning_effort,
            )
            first_parsed = _parse_router_response(first_raw)
            if first_parsed.get("non_minor_detection_failure") is False:
                second_raw = ""
                second_usage: dict[str, Any] = {}
                second_response_model = ""
                second_parsed = {
                    "answer": "no",
                    "non_minor_detection_failure": False,
                    "trigger_refinement": False,
                    "severity": "none",
                    "error_types": [],
                    "justification": "Skipped because the first pass answered no.",
                    "rationale": "Skipped because the first pass answered no.",
                }
                ran_second_pass = False
                second_pass_skip_reason = "first_pass_no"
            else:
                second_raw, second_usage, second_response_model = _chat_text(
                    model=args.model,
                    prompt_text=_stage2_second_prompt(second_prompt, stage2a_record, first_raw, first_parsed),
                    temperature=args.temperature,
                    max_tokens=args.stage2b_second_max_tokens,
                    base_url=base_url,
                    api_key=api_key,
                    reasoning_effort=args.stage2b_reasoning_effort,
                )
                second_parsed = _parse_router_response(second_raw)
                ran_second_pass = True
                second_pass_skip_reason = ""
            record.update(
                {
                    "first_raw_response": first_raw,
                    "first_parsed_response": first_parsed,
                    "ran_second_pass": ran_second_pass,
                    "second_pass_skip_reason": second_pass_skip_reason,
                    "second_raw_response": second_raw,
                    "second_parsed_response": second_parsed,
                    "first_non_minor_detection_failure": first_parsed.get("non_minor_detection_failure"),
                    "first_justification": first_parsed.get("justification") or first_parsed.get("rationale", ""),
                    "final_non_minor_detection_failure": second_parsed.get("non_minor_detection_failure"),
                    "final_answer": second_parsed.get("answer", ""),
                    "final_justification": second_parsed.get("justification") or second_parsed.get("rationale", ""),
                    "first_usage": first_usage,
                    "second_usage": second_usage,
                    "first_response_model": first_response_model,
                    "second_response_model": second_response_model,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
        _write_json(result_path, record)

    if record.get("error"):
        case["errors"].append(
            {
                "stage": "stage2b_nonminor_router",
                "message": record["error"],
                "result_path": str(result_path),
            }
        )
    trigger = _boolish(record.get("final_non_minor_detection_failure")) or bool(args.force_stage3_redetect)
    case.update(
        {
            "stage2b_router_result_path": str(result_path),
            "stage2b_final_non_minor_detection_failure": record.get("final_non_minor_detection_failure", False),
            "stage2b_final_answer": record.get("final_answer", ""),
            "stage2b_final_justification": record.get("final_justification", ""),
            "stage2b_ran_second_pass": bool(record.get("ran_second_pass", False)),
            "stage2b_router_error": record.get("error", ""),
            "stage3_redetect_triggered": trigger,
        }
    )
    return case


def _run_stage3_case(
    case: dict[str, Any],
    stage1_prompt: str,
    wrapper_prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    case = {**case, "errors": list(case.get("errors") or [])}
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage3_feedback_redetection"
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "stage3_feedback_redetection.json"
    triggered = bool(case.get("stage3_redetect_triggered"))
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))
    overlay_path = _existing_file_path(case.get("stage2a_review_overlay_path")) or _existing_file_path(
        case.get("source_detector_overlay_path")
    )

    if not triggered:
        record = {
            "task_id": f"{case['case_slug']}_stage3",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage3_feedback_redetection",
            "ran": False,
            "skip_reason": "stage2b_no_non_minor_detection_failure",
            "detections": [],
            "stage3_detection_count": 0,
            "stage3_overlay_path": "",
            "error": "",
            "created_at": _timestamp(),
        }
        _write_json(result_path, record)
    elif args.reuse_existing and result_path.exists():
        record = _read_json(result_path)
    else:
        record = {
            "task_id": f"{case['case_slug']}_stage3",
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "case_slug": case["case_slug"],
            "case_display": case["case_display"],
            "stage": "stage3_feedback_redetection",
            "ran": True,
            "skip_reason": "",
            "prompt_file": str(args.stage3_wrapper_prompt),
            "stage1_prompt_file": str(args.stage1_prompt),
            "model": args.model,
            "reasoning_effort": args.stage3_reasoning_effort,
            "thumbnail_path": case.get("thumbnail_path", ""),
            "stage1_raw_overlay_path": str(overlay_path) if overlay_path else "",
            "stage2a_review_text": case.get("stage2a_review_text", ""),
            "stage2b_final_justification": case.get("stage2b_final_justification", ""),
            "raw_response": "",
            "parsed_response": {},
            "detections": [],
            "stage3_detection_count": 0,
            "stage3_overlay_path": "",
            "usage": {},
            "response_model": "",
            "error": "",
            "created_at": _timestamp(),
        }
        try:
            if thumbnail_path is None or overlay_path is None:
                raise FileNotFoundError("Missing Stage 3 thumbnail or previous-overlay input.")
            prompt = wrapper_prompt.format(
                reviewer_feedback=str(case.get("stage2a_review_text", "")).strip(),
                stage1_task_prompt=stage1_prompt.strip(),
            )
            raw, usage, response_model = _chat_with_images(
                model=args.model,
                prompt_text=prompt,
                image_paths=[thumbnail_path, overlay_path],
                temperature=args.temperature,
                max_tokens=args.stage3_max_tokens,
                base_url=base_url,
                api_key=api_key,
                reasoning_effort=args.stage3_reasoning_effort,
            )
            payload = _extract_json_payload(raw)
            with Image.open(thumbnail_path) as image:
                thumbnail_size = image.size
            detections = _normalised_detection_items(payload, thumbnail_size)
            boxes = [[float(value) for value in det["box_2d_yxyx_normalized"]] for det in detections]
            redetect_overlay = stage_dir / "stage3_feedback_redetection_overlay.png"
            _draw_boxes_overlay(
                thumbnail_path,
                redetect_overlay,
                boxes,
                f"Stage 3 feedback redetection: {len(boxes)}",
            )
            record.update(
                {
                    "raw_response": raw,
                    "parsed_response": payload,
                    "detections": detections,
                    "stage3_detection_count": len(detections),
                    "stage3_overlay_path": str(redetect_overlay),
                    "usage": usage,
                    "response_model": response_model,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
        _write_json(result_path, record)

    if record.get("error"):
        case["errors"].append(
            {
                "stage": "stage3_feedback_redetection",
                "message": record["error"],
                "result_path": str(result_path),
            }
        )

    if record.get("ran") and not record.get("error"):
        boxes = [
            [float(value) for value in det.get("box_2d_yxyx_normalized", [])]
            for det in record.get("detections", [])
            if det.get("box_2d_yxyx_normalized")
        ]
        active_stage = "stage3_feedback_redetection"
        active_overlay = record.get("stage3_overlay_path", "")
    else:
        boxes = [
            [float(value) for value in box]
            for box in case.get("stage1_source_boxes_yxyx_normalized", case.get("source_boxes_yxyx_normalized", []))
        ]
        active_stage = case.get("stage1_source_stage", "stage1_raw")
        active_overlay = case.get("source_detector_overlay_path", "")

    case.update(
        {
            "stage3_result_path": str(result_path),
            "stage3_redetect_ran": bool(record.get("ran", False)),
            "stage3_redetect_error": record.get("error", ""),
            "stage3_detection_count": int(record.get("stage3_detection_count") or 0),
            "stage3_boxes_yxyx_normalized": boxes if record.get("ran") and not record.get("error") else [],
            "stage3_overlay_path": record.get("stage3_overlay_path", ""),
            "active_source_stage": active_stage,
            "active_detector_overlay_path": active_overlay,
            "active_boxes_yxyx_normalized": boxes,
            "active_box_count": len(boxes),
        }
    )
    return case


def _build_postprocess_case(
    case: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = {**case, "errors": list(case.get("errors") or [])}
    tasks: list[dict[str, Any]] = []
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage4_high_res_crop_redetect"
    stage_dir.mkdir(parents=True, exist_ok=True)

    raw_boxes = [
        [float(value) for value in box]
        for box in case.get("active_boxes_yxyx_normalized", case.get("source_boxes_yxyx_normalized", []))
    ]
    merged, merge_counts = _merge_yxyx_boxes(
        raw_boxes,
        args.merge_iou_threshold,
        args.containment_overlap_threshold,
    )
    expanded = [_expand_yxyx(box, args.post_stage3_padding_frac) for box in merged]

    post_overlay = ""
    if thumbnail_path is not None:
        post_overlay = str(
            _draw_boxes_overlay(
                thumbnail_path,
                stage_dir / "stage4_postprocess_overlay.png",
                expanded,
                f"Stage 4 merge + {args.post_stage3_padding_frac:.0%}: {len(expanded)}",
            )
        )

    case.update(
        {
            "stage4_input_source_stage": case.get("active_source_stage", ""),
            "stage4_input_boxes_yxyx_normalized": raw_boxes,
            "stage4_input_box_count": len(raw_boxes),
            "stage4_merge_counts": merge_counts,
            "stage4_boxes_yxyx_normalized": merged,
            "stage4_expanded_boxes_yxyx_normalized": expanded,
            "stage4_expanded_count": len(expanded),
            "stage4_overlay_path": post_overlay,
            # Compatibility aliases for older analysis helpers.
            "post_stage3_merge_counts": merge_counts,
            "post_stage3_boxes_yxyx_normalized": merged,
            "post_stage3_expanded_boxes_yxyx_normalized": expanded,
            "post_stage3_expanded_count": len(expanded),
            "post_stage3_overlay_path": post_overlay,
        }
    )

    if args.skip_crop_redetect:
        _write_json(stage_dir / "stage4_high_res_crop_redetect.json", case)
        return case, tasks

    if not expanded:
        _write_json(stage_dir / "stage4_high_res_crop_redetect.json", case)
        return case, tasks

    wsi_size = tuple(int(value) for value in case.get("wsi_size") or [0, 0])
    if wsi_size[0] <= 0 or wsi_size[1] <= 0:
        case["errors"].append(
            {
                "stage": "stage4_high_res_crop_redetect",
                "message": "Missing WSI dimensions; cannot read postprocessed crops.",
            }
        )
        _write_json(stage_dir / "stage4_high_res_crop_redetect.json", case)
        return case, tasks

    try:
        wsi, reader = load_wsi(case["wsi_path"], args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            for order, (source_box, padded_box) in enumerate(zip(merged, expanded), start=1):
                source_bbox_l0 = _normalised_yxyx_to_level0(source_box, wsi_size)
                padded_bbox_l0 = _normalised_yxyx_to_level0(padded_box, wsi_size)
                crop, read_info = _read_padded_crop(
                    wsi,
                    reader,
                    pyramid,
                    source_bbox_l0,
                    padded_bbox_l0,
                    args.crop_max_dim,
                )
                read_info["padding_fraction"] = float(args.post_stage3_padding_frac)
                task_dir = stage_dir / "inputs" / f"{order:02d}"
                task_dir.mkdir(parents=True, exist_ok=True)
                crop_path = task_dir / "crop.png"
                source_overlay_path = task_dir / "source_box_overlay.png"
                crop.save(crop_path)
                overlay = crop.copy()
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = [int(value) for value in read_info["source_bbox_in_crop"]]
                line_width = max(3, max(crop.size) // 180)
                draw.rectangle((x1, y1, x2, y2), outline="#e31a1c", width=line_width)
                overlay.save(source_overlay_path)
                task = {
                    "task_id": f"{case['case_slug']}_crop_redetect_{order:02d}",
                    "case_index": int(case["case_index"]),
                    "case_id": case["case_id"],
                    "case_slug": case["case_slug"],
                    "case_display": case["case_display"],
                    "task_dir": str(task_dir),
                    "source_stage": case.get("active_source_stage", ""),
                    "source_order": order,
                    "source_box_yxyx_normalized": source_box,
                    "padded_box_yxyx_normalized": padded_box,
                    "crop_path": str(crop_path),
                    "source_overlay_path": str(source_overlay_path),
                    "thumbnail_path": case.get("thumbnail_path", ""),
                    "source_detector_overlay_path": case.get("active_detector_overlay_path", ""),
                    "wsi_path": case["wsi_path"],
                    "wsi_reader": reader,
                    "wsi_size": list(wsi_size),
                    "read_info": read_info,
                    "prompt_version": PROMPT_VERSION,
                    "created_at": _timestamp(),
                }
                _write_json(task_dir / "metadata.json", {"task": task, "pyramid": pyramid})
                tasks.append(task)
        finally:
            close_wsi(wsi, reader)
    except Exception as exc:
        case["errors"].append(
            {
                "stage": "stage4_high_res_crop_redetect",
                "message": repr(exc),
            }
        )

    _write_json(stage_dir / "stage4_high_res_crop_redetect.json", case)
    return case, tasks


def _run_crop_redetect_task(
    task: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    task_dir = Path(task["task_dir"])
    result_path = task_dir / "crop_redetect_result.json"
    if args.reuse_existing and result_path.exists():
        return _read_json(result_path)

    record = {
        **task,
        "model": args.model,
        "reasoning_effort": args.crop_redetect_reasoning_effort,
        "raw_response": "",
        "parsed_response": None,
        "detections_crop": [],
        "detections_wsi": [],
        "detection_count": 0,
        "parser_status": "not_run",
        "error": "",
        "usage": {},
        "response_model": "",
        "crop_detection_overlay_path": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt,
            image_paths=[Path(task["crop_path"])],
            temperature=args.temperature,
            max_tokens=args.crop_redetect_max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.crop_redetect_reasoning_effort,
        )
        payload = _extract_json_payload(raw)
        with Image.open(task["crop_path"]) as image:
            crop_size = image.size
        detections_crop = _normalised_detection_items(payload, crop_size)
        wsi_size = (int(task["wsi_size"][0]), int(task["wsi_size"][1]))
        detections_wsi = []
        for idx, detection in enumerate(detections_crop, start=1):
            norm = _crop_pixel_bbox_to_wsi_norm(detection["bbox_thumbnail"], task["read_info"], wsi_size)
            detections_wsi.append(
                {
                    "label": detection.get("label") or f"tissue_{idx}",
                    "crop_detection": detection,
                    "box_2d_yxyx_normalized": [round(float(value), 3) for value in norm],
                }
            )
        overlay_path = task_dir / "crop_redetect_overlay.png"
        _draw_crop_detection_overlay(Path(task["crop_path"]), overlay_path, detections_crop)
        record.update(
            {
                "raw_response": raw,
                "parsed_response": payload,
                "detections_crop": detections_crop,
                "detections_wsi": detections_wsi,
                "detection_count": len(detections_wsi),
                "parser_status": "ok" if detections_wsi else "no_detections",
                "usage": usage,
                "response_model": response_model,
                "crop_detection_overlay_path": str(overlay_path),
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
        record["parser_status"] = "error"
    _write_json(result_path, record)
    return record


def _build_classification_inputs_case(
    case: dict[str, Any],
    redetect_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = {**case, "errors": list(case.get("errors") or [])}
    candidates: list[dict[str, Any]] = []
    artifacts_dir = Path(case["artifacts_dir"])
    stage_dir = artifacts_dir / "stage5_post_redetect_merge_and_crop"
    stage_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))

    if args.skip_crop_redetect:
        boxes = [
            [float(value) for value in box]
            for box in case.get("stage4_boxes_yxyx_normalized", case.get("post_stage3_boxes_yxyx_normalized", []))
        ]
        crop_redetect_detection_count = 0
        crop_redetect_error_count = 0
        stage5_input_source = "stage4_merged_boxes"
        bbox_source = "stage4_merged_no_crop_redetect"
        overlay_title = f"stage4 direct merge: {len(boxes)}"
        candidate_suffix = "stage4"
    else:
        detections = []
        for row in redetect_results:
            detections.extend(row.get("detections_wsi") or [])
        boxes = [[float(value) for value in detection["box_2d_yxyx_normalized"]] for detection in detections]
        crop_redetect_detection_count = len(boxes)
        crop_redetect_error_count = sum(1 for row in redetect_results if row.get("error"))
        stage5_input_source = "stage4_high_res_crop_redetect"
        bbox_source = "stage4_high_res_crop_redetect"
        overlay_title = f"crop redetect merge: {len(boxes)}"
        candidate_suffix = "post_redetect"

    merged, merge_counts = _merge_yxyx_boxes(
        boxes,
        args.merge_iou_threshold,
        args.containment_overlap_threshold,
    )
    merge_overlay = ""
    if thumbnail_path is not None:
        merge_overlay = str(
            _draw_boxes_overlay(
                thumbnail_path,
                stage_dir / "post_redetect_merge_overlay.png",
                merged,
                f"{overlay_title} -> {len(merged)}",
            )
        )

    case.update(
        {
            "crop_redetect_skipped": bool(args.skip_crop_redetect),
            "crop_redetect_detection_count": crop_redetect_detection_count,
            "crop_redetect_error_count": crop_redetect_error_count,
            "post_redetect_input_source": stage5_input_source,
            "post_redetect_input_box_count": len(boxes),
            "post_redetect_merge_counts": merge_counts,
            "post_redetect_boxes_yxyx_normalized": merged,
            "post_redetect_merged_count": len(merged),
            "post_redetect_merge_overlay_path": merge_overlay,
        }
    )
    if not merged:
        _write_json(stage_dir / "stage5_post_redetect_merge_and_crop.json", case)
        return case, candidates

    wsi_size = tuple(int(value) for value in case.get("wsi_size") or [0, 0])
    if wsi_size[0] <= 0 or wsi_size[1] <= 0:
        case["errors"].append(
            {
                "stage": "stage5_post_redetect_merge_and_crop",
                "message": "Missing WSI dimensions; cannot read classification crops.",
            }
        )
        _write_json(stage_dir / "stage5_post_redetect_merge_and_crop.json", case)
        return case, candidates

    try:
        wsi, reader = load_wsi(case["wsi_path"], args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            for order, box in enumerate(merged, start=1):
                source_bbox_l0 = _normalised_yxyx_to_level0(box, wsi_size)
                padded_bbox_l0 = _pad_level0_bbox(source_bbox_l0, wsi_size, args.classification_padding_frac)
                crop, read_info = _read_padded_crop(
                    wsi,
                    reader,
                    pyramid,
                    source_bbox_l0,
                    padded_bbox_l0,
                    args.classification_max_dim,
                )
                read_info["padding_fraction"] = float(args.classification_padding_frac)
                candidate_id = f"{order:02d}_{candidate_suffix}"
                candidate_dir = stage_dir / "candidates" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                crop_path = candidate_dir / "crop.png"
                selected_overlay_path = candidate_dir / "selected_candidate_overlay.png"
                vlm_image_path = candidate_dir / "selected_candidate_overlay_vlm.jpg"
                crop.save(crop_path)
                overlay = crop.copy()
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = [int(value) for value in read_info["source_bbox_in_crop"]]
                line_width = max(3, max(crop.size) // 180)
                draw.rectangle((x1, y1, x2, y2), outline="#e31a1c", width=line_width)
                label_font = _font(max(20, min(42, max(crop.size) // 24)))
                label_box = draw.textbbox((x1 + 5, y1 + 5), str(order), font=label_font)
                draw.rectangle(label_box, fill="white", outline="#e31a1c", width=max(2, line_width // 2))
                draw.text((x1 + 5, y1 + 5), str(order), fill="#e31a1c", font=label_font)
                overlay.save(selected_overlay_path)
                _save_vlm_jpeg(overlay, vlm_image_path)
                candidate = {
                    "task_id": f"{case['case_slug']}_classification_{order:02d}",
                    "case_index": int(case["case_index"]),
                    "case_id": case["case_id"],
                    "case_slug": case["case_slug"],
                    "case_display": case["case_display"],
                    "candidate_order": order,
                    "candidate_id": candidate_id,
                    "candidate_dir": str(candidate_dir),
                    "bbox_source": bbox_source,
                    "box_2d_yxyx_normalized": box,
                    "crop_path": str(crop_path),
                    "selected_overlay_path": str(selected_overlay_path),
                    "vlm_image_path": str(vlm_image_path),
                    "thumbnail_path": case.get("thumbnail_path", ""),
                    "wsi_path": case["wsi_path"],
                    "wsi_reader": reader,
                    "read_info": read_info,
                    "prompt_version": PROMPT_VERSION,
                    "created_at": _timestamp(),
                }
                _write_json(candidate_dir / "metadata.json", {"candidate": candidate, "pyramid": pyramid})
                candidates.append(candidate)
        finally:
            close_wsi(wsi, reader)
    except Exception as exc:
        case["errors"].append(
            {
                "stage": "stage5_post_redetect_merge_and_crop",
                "message": repr(exc),
            }
        )

    _write_json(stage_dir / "stage5_post_redetect_merge_and_crop.json", case)
    return case, candidates


def _run_classification_candidate(
    candidate: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    candidate_dir = Path(candidate["candidate_dir"])
    result_path = candidate_dir / "classification_result.json"
    if args.reuse_existing and result_path.exists():
        return _read_json(result_path)

    record = {
        **candidate,
        "model": args.model,
        "reasoning_effort": args.classification_reasoning_effort,
        "raw_response": "",
        "tissue_focus_decision": "unknown",
        "parser_route": "",
        "error": "",
        "usage": {},
        "response_model": "",
    }
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt,
            image_paths=[Path(candidate["selected_overlay_path"])],
            temperature=args.temperature,
            max_tokens=args.classification_max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.classification_reasoning_effort,
        )
        decision, parser_route = _parse_tissue_yes_no(raw)
        record.update(
            {
                "raw_response": raw,
                "tissue_focus_decision": decision,
                "parser_route": parser_route,
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
    _write_json(result_path, record)
    return record


def _draw_classification_overlay_case(
    case: dict[str, Any],
    classification_results: list[dict[str, Any]],
) -> dict[str, Any]:
    case = {**case}
    rows = sorted(classification_results, key=lambda row: int(row["candidate_order"]))
    boxes = [[float(value) for value in row["box_2d_yxyx_normalized"]] for row in rows]
    colors = {
        "yes": "#188038",
        "no": "#d93025",
        "unknown": "#5f6368",
    }
    overlay_path = ""
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))
    if thumbnail_path is not None:
        overlay_path = str(
            _draw_boxes_overlay(
                thumbnail_path,
                Path(case["artifacts_dir"]) / "stage6_tissue_artifact_classification" / "classification_overlay.png",
                boxes,
                "classification: green yes / red no",
                colors=[colors.get(row.get("tissue_focus_decision"), "#5f6368") for row in rows],
                labels=[
                    f"{int(row['candidate_order']):02d}:{row.get('tissue_focus_decision', 'unknown')}"
                    for row in rows
                ],
            )
        )
    case.update(
        {
            "classification_overlay_path": overlay_path,
            "classification_candidate_count": len(rows),
            "classification_yes_count": sum(1 for row in rows if row.get("tissue_focus_decision") == "yes"),
            "classification_no_count": sum(1 for row in rows if row.get("tissue_focus_decision") == "no"),
            "classification_unknown_count": sum(1 for row in rows if row.get("tissue_focus_decision") == "unknown"),
            "classification_error_count": sum(1 for row in rows if row.get("error")),
        }
    )
    return case


def _build_odd_one_out_task_case(
    case: dict[str, Any],
    classification_results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    yes_rows = [
        row
        for row in sorted(classification_results, key=lambda item: int(item["candidate_order"]))
        if row.get("tissue_focus_decision") == "yes" and not row.get("error")
    ]
    stage_dir = Path(case["artifacts_dir"]) / "stage7_comparative_thumbnail_filter"
    if len(yes_rows) <= 1:
        return None, {
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "remaining_crop_count": len(yes_rows),
            "skip_reason": "remaining_crop_count_below_2",
        }
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))
    if thumbnail_path is None:
        return None, {
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "remaining_crop_count": len(yes_rows),
            "skip_reason": "missing_thumbnail",
        }

    thumbnail = Image.open(thumbnail_path).convert("RGB")
    patches: list[dict[str, Any]] = []
    patch_dir = stage_dir / "thumbnail_crops"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for patch_id, row in enumerate(yes_rows, start=1):
        x1, y1, x2, y2 = _norm_to_image_bbox([float(v) for v in row["box_2d_yxyx_normalized"]], thumbnail.size)
        x1 = max(0, min(thumbnail.size[0], x1))
        x2 = max(0, min(thumbnail.size[0], x2))
        y1 = max(0, min(thumbnail.size[1], y1))
        y2 = max(0, min(thumbnail.size[1], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_path = patch_dir / f"{patch_id:02d}_candidate_{int(row['candidate_order']):02d}.png"
        crop = thumbnail.crop((x1, y1, x2, y2))
        crop.save(crop_path)
        patches.append(
            {
                "id": patch_id,
                "candidate_order": int(row["candidate_order"]),
                "candidate_id": row["candidate_id"],
                "crop_path": str(crop_path),
                "bbox_thumbnail": [x1, y1, x2, y2],
                "box_2d_yxyx_normalized": row["box_2d_yxyx_normalized"],
                "crop_size": list(crop.size),
            }
        )
    thumbnail.close()
    if len(patches) <= 1:
        return None, {
            "case_index": int(case["case_index"]),
            "case_id": case["case_id"],
            "remaining_crop_count": len(patches),
            "skip_reason": "valid_thumbnail_crop_count_below_2",
        }
    task = {
        "task_id": f"{case['case_slug']}_odd_one_out",
        "case_index": int(case["case_index"]),
        "case_id": case["case_id"],
        "case_slug": case["case_slug"],
        "case_display": case["case_display"],
        "task_dir": str(stage_dir),
        "patch_count": len(patches),
        "patches": patches,
        "thumbnail_path": str(thumbnail_path),
        "prompt_version": PROMPT_VERSION,
        "created_at": _timestamp(),
    }
    _write_json(stage_dir / "odd_one_out_input.json", task)
    return task, None


def _odd_prompt_with_ids(prompt: str, patch_count: int) -> str:
    return (
        prompt.strip()
        + "\n\nThe attached crop images are ordered by crop id. "
        + f"Use id 1 for the first attached image through id {patch_count} for the last attached image."
    )


def _run_odd_one_out_task(
    task: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    stage_dir = Path(task["task_dir"])
    result_path = stage_dir / "odd_one_out_result.json"
    if args.reuse_existing and result_path.exists():
        record = _read_json(result_path)
        raw = str(record.get("raw_response") or "")
        if raw and not str(record.get("parse_status") or "").startswith("ok"):
            parsed, route, status = _parse_odd_one_out_response(raw, int(task["patch_count"]))
            flagged_ids = []
            if isinstance(parsed, dict):
                for item in parsed.get("flagged_artifacts", []):
                    try:
                        flagged_ids.append(int(item))
                    except Exception:
                        continue
            id_to_order = {int(patch["id"]): int(patch["candidate_order"]) for patch in task["patches"]}
            record.update(
                {
                    "parsed_response": parsed,
                    "parse_route": route,
                    "parse_status": status,
                    "flagged_artifacts": sorted(flagged_ids),
                    "flagged_candidate_orders": sorted(
                        id_to_order[item] for item in flagged_ids if item in id_to_order
                    ),
                }
            )
            _write_json(result_path, record)
        return record
    record = {
        **task,
        "model": args.model,
        "reasoning_effort": args.odd_one_out_reasoning_effort,
        "raw_response": "",
        "parsed_response": None,
        "parse_route": "",
        "parse_status": "not_run",
        "flagged_artifacts": [],
        "flagged_candidate_orders": [],
        "error": "",
        "usage": {},
        "response_model": "",
    }
    try:
        image_paths = [Path(patch["crop_path"]) for patch in task["patches"]]
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=_odd_prompt_with_ids(prompt, int(task["patch_count"])),
            image_paths=image_paths,
            temperature=args.temperature,
            max_tokens=args.odd_one_out_max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=args.odd_one_out_reasoning_effort,
        )
        parsed, route, status = _parse_odd_one_out_response(raw, int(task["patch_count"]))
        flagged_ids = []
        if isinstance(parsed, dict):
            for item in parsed.get("flagged_artifacts", []):
                try:
                    flagged_ids.append(int(item))
                except Exception:
                    continue
        id_to_order = {int(patch["id"]): int(patch["candidate_order"]) for patch in task["patches"]}
        record.update(
            {
                "raw_response": raw,
                "parsed_response": parsed,
                "parse_route": route,
                "parse_status": status,
                "flagged_artifacts": sorted(flagged_ids),
                "flagged_candidate_orders": sorted(id_to_order[item] for item in flagged_ids if item in id_to_order),
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)
        record["parse_status"] = "error"
    _write_json(result_path, record)
    return record


def _write_case_reproduction(case: dict[str, Any], args: argparse.Namespace) -> None:
    case_dir = Path(case["case_dir"])
    text = f"""\
Detector pipeline case output
=============================

Created: {_timestamp()}
Ticket: {TICKET}
Git commit: {_repo_git_commit()}
Pipeline version: {PROMPT_VERSION}
Case: {case['case_display']}
WSI: {case['wsi_path']}

This case was produced by the parent output run:
{args.output_dir.resolve()}

Root reproduction file:
{(args.output_dir / 'reproduction.txt').resolve()}
"""
    (case_dir / "reproduction.txt").write_text(text)


def _finalize_case(
    case: dict[str, Any],
    classification_results: list[dict[str, Any]],
    odd_task: dict[str, Any] | None,
    odd_result: dict[str, Any] | None,
    odd_skipped: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = sorted(classification_results, key=lambda row: int(row["candidate_order"]))
    yes_rows = [row for row in rows if row.get("tissue_focus_decision") == "yes" and not row.get("error")]
    flagged_orders = set(int(value) for value in (odd_result or {}).get("flagged_candidate_orders", []))
    final_rows = [row for row in yes_rows if int(row["candidate_order"]) not in flagged_orders]
    final_boxes = [[float(value) for value in row["box_2d_yxyx_normalized"]] for row in final_rows]

    case_dir = Path(case["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    final_overlay_path = case_dir / "final_detected_bboxes.png"
    thumbnail_path = _existing_file_path(case.get("thumbnail_path"))
    if thumbnail_path is not None:
        _draw_boxes_overlay(
            thumbnail_path,
            final_overlay_path,
            final_boxes,
            f"final boxes: {len(final_boxes)}",
            colors=["#188038"],
        )

    odd_sheet_path = ""
    if odd_task is not None:
        odd_sheet_path = _draw_odd_sheet(
            odd_task,
            set(int(value) for value in (odd_result or {}).get("flagged_artifacts", [])),
            Path(case["artifacts_dir"]) / "stage7_comparative_thumbnail_filter" / "odd_one_out_thumbnail_crops.png",
        )

    detections = [
        {
            "box_2d": [round(float(value), 3) for value in row["box_2d_yxyx_normalized"]],
            "source_candidate_order": int(row["candidate_order"]),
            "classification_decision": row.get("tissue_focus_decision", ""),
            "odd_one_out_flagged": int(row["candidate_order"]) in flagged_orders,
        }
        for row in final_rows
    ]
    case_output = {
        "created_at": _timestamp(),
        "ticket": TICKET,
        "git_commit": _repo_git_commit(),
        "pipeline_version": PROMPT_VERSION,
        "coordinate_system": "normalized_0_1000_y_min_x_min_y_max_x_max",
        "case_id": case["case_id"],
        "case_display": case["case_display"],
        "wsi_path": case["wsi_path"],
        "stage_contract": _stage_contract(args),
        "stage_counts": {
            "stage1_source_boxes": int(case.get("stage1_source_box_count", case.get("source_box_count")) or 0),
            "stage2b_triggered_stage3": bool(case.get("stage3_redetect_triggered", False)),
            "stage2b_ran_second_pass": bool(case.get("stage2b_ran_second_pass", False)),
            "stage3_redetect_ran": bool(case.get("stage3_redetect_ran", False)),
            "stage3_redetect_boxes": int(case.get("stage3_detection_count") or 0),
            "active_source_boxes": int(case.get("active_box_count") or 0),
            "stage4_input_boxes": int(case.get("stage4_input_box_count") or 0),
            "stage4_expanded_boxes": int(case.get("stage4_expanded_count", case.get("post_stage3_expanded_count")) or 0),
            "crop_redetect_skipped": bool(args.skip_crop_redetect),
            "crop_redetect_detections": int(case.get("crop_redetect_detection_count") or 0),
            "post_redetect_input_boxes": int(case.get("post_redetect_input_box_count") or 0),
            "post_redetect_merged_boxes": int(case.get("post_redetect_merged_count") or 0),
            "classification_candidates": int(case.get("classification_candidate_count") or 0),
            "classification_yes": int(case.get("classification_yes_count") or 0),
            "classification_no": int(case.get("classification_no_count") or 0),
            "classification_unknown": int(case.get("classification_unknown_count") or 0),
            "odd_one_out_flagged": len(flagged_orders),
            "final_boxes": len(final_boxes),
        },
        "stage_artifacts_dir": str(Path(case["artifacts_dir"]).resolve())
        if args.save_all_stage_artifacts
        else "",
        "stage_artifacts_saved": bool(args.save_all_stage_artifacts),
        "paths": {
            "final_overlay_png": str(final_overlay_path.resolve()) if final_overlay_path.exists() else "",
            "thumbnail_path": case.get("thumbnail_path", "") if args.save_all_stage_artifacts else "",
            "odd_one_out_sheet_path": odd_sheet_path if args.save_all_stage_artifacts else "",
        },
        "odd_one_out": {
            "ran": odd_result is not None,
            "parse_status": (odd_result or {}).get("parse_status", ""),
            "flagged_candidate_orders": sorted(flagged_orders),
            "skipped": odd_skipped or {},
            "error": (odd_result or {}).get("error", ""),
        },
        "errors": case.get("errors") or [],
        "detections": detections,
    }
    _write_json(case_dir / "detections.json", case_output)
    _write_case_reproduction(case, args)

    if not args.save_all_stage_artifacts and not case_output["errors"]:
        artifacts_dir = Path(case["artifacts_dir"])
        if artifacts_dir.exists() and artifacts_dir.name == "intermediate_stage_artifacts":
            shutil.rmtree(artifacts_dir)
    return case_output


def _copy_prompts(args: argparse.Namespace) -> None:
    prompt_dir = args.output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        args.stage1_prompt,
        args.stage2a_prompt,
        args.stage2b_first_prompt,
        args.stage2b_second_prompt,
        args.stage3_wrapper_prompt,
        args.classification_prompt,
        args.odd_one_out_prompt,
    ):
        (prompt_dir / path.name).write_text(path.read_text().strip() + "\n")


def _write_root_reproduction(
    args: argparse.Namespace,
    input_paths: list[Path],
    summary: dict[str, Any],
) -> None:
    stage_lines = []
    for stage in _stage_contract(args):
        stage_lines.append(
            textwrap.dedent(
                f"""\
                - {stage['stage']}
                  prompt: {stage['prompt']}
                  input_image: {stage['input_image']}
                  output: {stage['output']}
                """
            ).rstrip()
        )
    text = f"""\
Detector pipeline run
=====================

Created: {_timestamp()}
Ticket: {TICKET}
Git commit: {_repo_git_commit()}
Pipeline version: {PROMPT_VERSION}
Batch mode: {args.batch_mode}
Max concurrency: {args.max_concurrent}
Model: {args.model}
Backend: {args.backend}
Child stage reproducibility gate skipped: {bool(args.skip_repro)}
Stage 2 review skipped: {bool(args.skip_stage2_review)}
Crop-redetect stage skipped: {bool(args.skip_crop_redetect)}

Command:
{_redacted_argv(sys.argv)}

Inputs:
{chr(10).join('- ' + str(path.resolve()) for path in input_paths)}

Stage contract:
{chr(10).join(stage_lines)}

Key parameters:
- Stage 1 thumbnail max dim: {args.stage1_thumbnail_max_dim}
- Stage 1 rotations: {args.stage1_rotations}
- Stage 1 source rotation: {args.stage1_source_rotation}
- Stage 1 max retries: {args.stage1_max_retries}
- Stage 1 repair model: {args.stage1_repair_model or args.model}
- Stage 1 prompt: {args.stage1_prompt.resolve()}
- Stage 2 review skipped: {bool(args.skip_stage2_review)}
- Stage 2a prompt: {args.stage2a_prompt.resolve()}
- Stage 2a reasoning effort: {args.stage2a_reasoning_effort}
- Stage 2a max tokens: {args.stage2a_max_tokens}
- Stage 2b first prompt: {args.stage2b_first_prompt.resolve()}
- Stage 2b second prompt: {args.stage2b_second_prompt.resolve()}
- Stage 2b reasoning effort: {args.stage2b_reasoning_effort}
- Stage 2b max tokens: {args.stage2b_max_tokens}
- Stage 2b second max tokens: {args.stage2b_second_max_tokens}
- Stage 3 wrapper prompt: {args.stage3_wrapper_prompt.resolve()}
- Stage 3 reasoning effort: {args.stage3_reasoning_effort}
- Stage 3 max tokens: {args.stage3_max_tokens}
- Force Stage 3 redetect: {bool(args.force_stage3_redetect)}
- Stage 4 merge IoU threshold: {args.merge_iou_threshold}
- Stage 4 overlap-over-smaller threshold: {args.containment_overlap_threshold}
- Stage 4 padding fraction for crop-redetect context: {args.post_stage3_padding_frac}
- Crop redetect skipped: {bool(args.skip_crop_redetect)}
- High-res crop-redetect max dim: {args.crop_max_dim}
- Classification crop padding fraction: {args.classification_padding_frac}
- Classification crop max dim: {args.classification_max_dim}
- Classification prompt: {args.classification_prompt.resolve()}
- Odd-one-out prompt: {args.odd_one_out_prompt.resolve()}
- Child Stage 1 reproducibility gate skipped: {bool(args.skip_repro)}

Outputs:
- Summary JSON: {(args.output_dir / 'summary.json').resolve()}
- All detections JSON: {(args.output_dir / 'all_detections.json').resolve()}
- Prompt copies: {(args.output_dir / 'prompts').resolve()}
- For a single input WSI: final_detected_bboxes.png and detections.json are written directly in the output dir.
- For multiple input WSIs: each case has its own subdir named from the WSI filename stem.
- Intermediate stage artifacts saved: {bool(args.save_all_stage_artifacts)}

Summary:
{json.dumps(summary, indent=2, sort_keys=True)}
"""
    (args.output_dir / "reproduction.txt").write_text(text)


def _write_root_outputs(
    args: argparse.Namespace,
    input_paths: list[Path],
    final_records: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = {
        "created_at": _timestamp(),
        "ticket": TICKET,
        "git_commit": _repo_git_commit(),
        "pipeline_version": PROMPT_VERSION,
        "dry_run": dry_run,
        "batch_mode": args.batch_mode,
        "max_concurrent": args.max_concurrent,
        "cases": len(final_records),
        "final_boxes": sum(int(record.get("stage_counts", {}).get("final_boxes", 0)) for record in final_records),
        "cases_with_errors": sum(1 for record in final_records if record.get("errors")),
        "stage_contract": _stage_contract(args),
        "case_outputs": [
            {
                "case_id": record.get("case_id"),
                "case_display": record.get("case_display"),
                "wsi_path": record.get("wsi_path"),
                "final_box_count": record.get("stage_counts", {}).get("final_boxes", 0),
                "detections_json": str((Path(record.get("case_dir", "")) / "detections.json").resolve())
                if record.get("case_dir")
                else "",
                "final_overlay_png": record.get("paths", {}).get("final_overlay_png", ""),
                "errors": record.get("errors", []),
            }
            for record in final_records
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "all_detections.json", final_records)
    _write_root_reproduction(args, input_paths, summary)
    return summary


def _group_by_case(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["case_index"])].append(row)
    return grouped


def _read_case_stage_records(cases: list[dict[str, Any]], path_key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda row: int(row["case_index"])):
        path = _existing_file_path(case.get(path_key))
        if path is not None:
            records.append(_read_json(path))
    return records


def _write_intermediate_tables(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    stage2a_results: list[dict[str, Any]],
    stage2b_results: list[dict[str, Any]],
    stage3_results: list[dict[str, Any]],
    crop_tasks: list[dict[str, Any]],
    crop_results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    classification_results: list[dict[str, Any]],
    odd_tasks: list[dict[str, Any]],
    odd_skipped: list[dict[str, Any]],
    odd_results: list[dict[str, Any]],
) -> None:
    if not args.save_all_stage_artifacts:
        return
    root = args.output_dir / "intermediate_stage_artifacts"
    _write_jsonl(root / "stage1_cases.jsonl", cases)
    _write_jsonl(root / "stage2a_detection_review_results.jsonl", stage2a_results)
    _write_jsonl(root / "stage2b_nonminor_router_results.jsonl", stage2b_results)
    _write_jsonl(root / "stage3_feedback_redetection_results.jsonl", stage3_results)
    _write_jsonl(root / "stage4_crop_redetect_tasks.jsonl", crop_tasks)
    _write_jsonl(root / "stage4_crop_redetect_results.jsonl", crop_results)
    _write_jsonl(root / "stage5_classification_candidates.jsonl", candidates)
    _write_jsonl(root / "stage6_classification_results.jsonl", classification_results)
    _write_jsonl(root / "stage7_odd_one_out_tasks.jsonl", odd_tasks)
    _write_jsonl(root / "stage7_odd_one_out_skipped.jsonl", odd_skipped)
    _write_jsonl(root / "stage7_odd_one_out_results.jsonl", odd_results)
    _write_csv(
        root / "final_stage_counts.csv",
        [
            {
                "case_index": case["case_index"],
                "case_id": case["case_id"],
                "case_display": case["case_display"],
                "stage1_source_box_count": case.get("stage1_source_box_count", case.get("source_box_count", 0)),
                "stage2b_final_non_minor_detection_failure": case.get(
                    "stage2b_final_non_minor_detection_failure", ""
                ),
                "stage2b_ran_second_pass": case.get("stage2b_ran_second_pass", ""),
                "stage3_redetect_triggered": case.get("stage3_redetect_triggered", ""),
                "stage3_redetect_ran": case.get("stage3_redetect_ran", ""),
                "stage3_detection_count": case.get("stage3_detection_count", 0),
                "active_source_stage": case.get("active_source_stage", ""),
                "active_box_count": case.get("active_box_count", 0),
                "stage4_input_box_count": case.get("stage4_input_box_count", 0),
                "stage4_expanded_count": case.get("stage4_expanded_count", case.get("post_stage3_expanded_count", 0)),
                "crop_redetect_skipped": bool(case.get("crop_redetect_skipped", False)),
                "crop_redetect_detection_count": case.get("crop_redetect_detection_count", 0),
                "post_redetect_input_source": case.get("post_redetect_input_source", ""),
                "post_redetect_input_box_count": case.get("post_redetect_input_box_count", 0),
                "post_redetect_merged_count": case.get("post_redetect_merged_count", 0),
                "classification_yes_count": case.get("classification_yes_count", 0),
                "classification_no_count": case.get("classification_no_count", 0),
                "classification_unknown_count": case.get("classification_unknown_count", 0),
                "error_count": len(case.get("errors") or []),
            }
            for case in cases
        ],
        [
            "case_index",
            "case_id",
            "case_display",
            "stage1_source_box_count",
            "stage2b_final_non_minor_detection_failure",
            "stage2b_ran_second_pass",
            "stage3_redetect_triggered",
            "stage3_redetect_ran",
            "stage3_detection_count",
            "active_source_stage",
            "active_box_count",
            "stage4_input_box_count",
            "stage4_expanded_count",
            "crop_redetect_skipped",
            "crop_redetect_detection_count",
            "post_redetect_input_source",
            "post_redetect_input_box_count",
            "post_redetect_merged_count",
            "classification_yes_count",
            "classification_no_count",
            "classification_unknown_count",
            "error_count",
        ],
    )


def _run_breadth_first(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    stage1_prompt = args.stage1_prompt.read_text().strip()
    stage2a_prompt = args.stage2a_prompt.read_text().strip()
    stage2b_first_prompt = args.stage2b_first_prompt.read_text().strip()
    stage2b_second_prompt = args.stage2b_second_prompt.read_text().strip()
    stage3_wrapper_prompt = args.stage3_wrapper_prompt.read_text().strip()
    classification_prompt = args.classification_prompt.read_text().strip()
    odd_prompt = args.odd_one_out_prompt.read_text().strip()

    cases = _parallel_map(
        cases,
        lambda case: _run_stage1_case(case, args, base_url, api_key),
        args.max_concurrent,
        "stage1_thumbnail_detection",
    )
    cases.sort(key=lambda case: int(case["case_index"]))

    cases = _parallel_map(
        cases,
        lambda case: _run_stage2a_case(case, stage2a_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage2a_detection_review",
    )
    cases.sort(key=lambda case: int(case["case_index"]))
    stage2a_results = _read_case_stage_records(cases, "stage2a_review_result_path")

    cases = _parallel_map(
        cases,
        lambda case: _run_stage2b_case(case, stage2b_first_prompt, stage2b_second_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage2b_nonminor_router",
    )
    cases.sort(key=lambda case: int(case["case_index"]))
    stage2b_results = _read_case_stage_records(cases, "stage2b_router_result_path")

    cases = _parallel_map(
        cases,
        lambda case: _run_stage3_case(case, stage1_prompt, stage3_wrapper_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage3_feedback_redetection",
    )
    cases.sort(key=lambda case: int(case["case_index"]))
    stage3_results = _read_case_stage_records(cases, "stage3_result_path")

    post_results = _parallel_map(
        cases,
        lambda case: _build_postprocess_case(case, args),
        args.max_concurrent,
        "stage4_high_res_crop_redetect:prepare_inputs",
    )
    cases = [item[0] for item in post_results]
    cases.sort(key=lambda case: int(case["case_index"]))
    crop_tasks = [task for _, tasks in post_results for task in tasks]

    crop_results = _parallel_map(
        crop_tasks,
        lambda task: _run_crop_redetect_task(task, stage1_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage4_high_res_crop_redetect",
    )
    crop_results.sort(key=lambda row: (int(row["case_index"]), int(row["source_order"])))
    crop_by_case = _group_by_case(crop_results)

    class_input_results = _parallel_map(
        cases,
        lambda case: _build_classification_inputs_case(case, crop_by_case.get(int(case["case_index"]), []), args),
        args.max_concurrent,
        "stage5_post_redetect_merge_and_crop",
    )
    cases = [item[0] for item in class_input_results]
    cases.sort(key=lambda case: int(case["case_index"]))
    candidates = [candidate for _, rows in class_input_results for candidate in rows]

    classification_results = _parallel_map(
        candidates,
        lambda candidate: _run_classification_candidate(candidate, classification_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage6_tissue_artifact_classification",
    )
    classification_results.sort(key=lambda row: (int(row["case_index"]), int(row["candidate_order"])))
    class_by_case = _group_by_case(classification_results)
    cases = [
        _draw_classification_overlay_case(case, class_by_case.get(int(case["case_index"]), []))
        for case in cases
    ]

    odd_task_pairs = [
        _build_odd_one_out_task_case(case, class_by_case.get(int(case["case_index"]), []))
        for case in cases
    ]
    odd_tasks = [task for task, _ in odd_task_pairs if task is not None]
    odd_skipped = [skipped for _, skipped in odd_task_pairs if skipped is not None]
    odd_results = _parallel_map(
        odd_tasks,
        lambda task: _run_odd_one_out_task(task, odd_prompt, args, base_url, api_key),
        args.max_concurrent,
        "stage7_comparative_thumbnail_filter",
    )
    odd_results.sort(key=lambda row: int(row["case_index"]))
    odd_by_case = {int(row["case_index"]): row for row in odd_results}
    odd_task_by_case = {int(row["case_index"]): row for row in odd_tasks}
    skipped_by_case = {int(row["case_index"]): row for row in odd_skipped}

    final_records = []
    for case in cases:
        case_index = int(case["case_index"])
        final_record = _finalize_case(
            case,
            class_by_case.get(case_index, []),
            odd_task_by_case.get(case_index),
            odd_by_case.get(case_index),
            skipped_by_case.get(case_index),
            args,
        )
        final_record["case_dir"] = case["case_dir"]
        final_records.append(final_record)

    _write_intermediate_tables(
        args,
        cases,
        stage2a_results,
        stage2b_results,
        stage3_results,
        crop_tasks,
        crop_results,
        candidates,
        classification_results,
        odd_tasks,
        odd_skipped,
        odd_results,
    )
    return final_records


def _run_depth_first_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    stage1_prompt: str,
    stage2a_prompt: str,
    stage2b_first_prompt: str,
    stage2b_second_prompt: str,
    stage3_wrapper_prompt: str,
    classification_prompt: str,
    odd_prompt: str,
) -> dict[str, Any]:
    case = _run_stage1_case(case, args, base_url, api_key)
    case = _run_stage2a_case(case, stage2a_prompt, args, base_url, api_key)
    case = _run_stage2b_case(case, stage2b_first_prompt, stage2b_second_prompt, args, base_url, api_key)
    case = _run_stage3_case(case, stage1_prompt, stage3_wrapper_prompt, args, base_url, api_key)
    case, crop_tasks = _build_postprocess_case(case, args)
    crop_results = _parallel_map(
        crop_tasks,
        lambda task: _run_crop_redetect_task(task, stage1_prompt, args, base_url, api_key),
        args.max_concurrent,
        f"stage4_high_res_crop_redetect:{case['case_slug']}",
    )
    case, candidates = _build_classification_inputs_case(case, crop_results, args)
    classification_results = _parallel_map(
        candidates,
        lambda candidate: _run_classification_candidate(candidate, classification_prompt, args, base_url, api_key),
        args.max_concurrent,
        f"stage6_tissue_artifact_classification:{case['case_slug']}",
    )
    classification_results.sort(key=lambda row: int(row["candidate_order"]))
    case = _draw_classification_overlay_case(case, classification_results)
    odd_task, odd_skipped = _build_odd_one_out_task_case(case, classification_results)
    odd_result = None
    if odd_task is not None:
        odd_result = _run_odd_one_out_task(odd_task, odd_prompt, args, base_url, api_key)
    final_record = _finalize_case(case, classification_results, odd_task, odd_result, odd_skipped, args)
    final_record["case_dir"] = case["case_dir"]
    if args.save_all_stage_artifacts:
        stage2a_results = _read_case_stage_records([case], "stage2a_review_result_path")
        stage2b_results = _read_case_stage_records([case], "stage2b_router_result_path")
        stage3_results = _read_case_stage_records([case], "stage3_result_path")
        _write_intermediate_tables(
            args,
            [case],
            stage2a_results,
            stage2b_results,
            stage3_results,
            crop_tasks,
            crop_results,
            candidates,
            classification_results,
            [odd_task] if odd_task is not None else [],
            [odd_skipped] if odd_skipped is not None else [],
            [odd_result] if odd_result is not None else [],
        )
    return final_record


def _run_depth_first(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    stage1_prompt = args.stage1_prompt.read_text().strip()
    stage2a_prompt = args.stage2a_prompt.read_text().strip()
    stage2b_first_prompt = args.stage2b_first_prompt.read_text().strip()
    stage2b_second_prompt = args.stage2b_second_prompt.read_text().strip()
    stage3_wrapper_prompt = args.stage3_wrapper_prompt.read_text().strip()
    classification_prompt = args.classification_prompt.read_text().strip()
    odd_prompt = args.odd_one_out_prompt.read_text().strip()
    final_records = []
    for case in cases:
        print(f"depth-first case {case['case_index']}/{len(cases)}: {case['case_display']}", flush=True)
        final_records.append(
            _run_depth_first_case(
                case,
                args,
                base_url,
                api_key,
                stage1_prompt,
                stage2a_prompt,
                stage2b_first_prompt,
                stage2b_second_prompt,
                stage3_wrapper_prompt,
                classification_prompt,
                odd_prompt,
            )
        )
    return final_records


def run(args: argparse.Namespace) -> int:
    if args.skip_stage2_review and args.force_stage3_redetect:
        raise SystemExit("--force-stage3-redetect requires Stage 2 review; remove --skip-stage2-review.")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = _resolve_input_paths(args)
    cases = _case_records_for_paths(input_paths, args.output_dir)
    _copy_prompts(args)
    _write_json(
        args.output_dir / "pipeline_manifest.json",
        {
            "created_at": _timestamp(),
            "ticket": TICKET,
            "git_commit": _repo_git_commit(),
            "pipeline_version": PROMPT_VERSION,
            "input_paths": [str(path) for path in input_paths],
            "batch_mode": args.batch_mode,
            "max_concurrent": args.max_concurrent,
            "stage_contract": _stage_contract(args),
            "cases": cases,
        },
    )

    if args.dry_run:
        dry_records = []
        for case in cases:
            case_dir = Path(case["case_dir"])
            case_dir.mkdir(parents=True, exist_ok=True)
            dry_record = {
                "case_dir": str(case_dir),
                "case_id": case["case_id"],
                "case_display": case["case_display"],
                "wsi_path": case["wsi_path"],
                "stage_counts": {"final_boxes": 0},
                "paths": {"final_overlay_png": ""},
                "errors": [],
                "dry_run": True,
            }
            _write_json(case_dir / "detections.json", dry_record)
            _write_case_reproduction(case, args)
            dry_records.append(dry_record)
        summary = _write_root_outputs(args, input_paths, dry_records, dry_run=True)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    base_url, api_key = _resolve_chat_settings(args)
    if args.batch_mode == "breadth-first":
        final_records = _run_breadth_first(cases, args, base_url, api_key)
    else:
        final_records = _run_depth_first(cases, args, base_url, api_key)
    final_records.sort(key=lambda row: str(row.get("case_id", "")))
    summary = _write_root_outputs(args, input_paths, final_records, dry_run=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["cases_with_errors"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="WSI path, WSI directory, or .txt file of WSI paths.")
    parser.add_argument("--wsi", type=Path, default=None, help="Single WSI path.")
    parser.add_argument("--wsi-dir", type=Path, default=None, help="Directory containing WSIs.")
    parser.add_argument("--wsi-list", type=Path, default=None, help="Text file with one WSI path per line.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--save-all-stage-artifacts", action="store_true")
    parser.add_argument(
        "--skip-repro",
        action="store_true",
        help=(
            "Set WSI_SKIP_STAGE_REPRO_CHECK=1 for child Stage 1 calls. "
            "Use this for parent pipeline runs that create their own output directory "
            "and write root reproduction.txt."
        ),
    )
    parser.add_argument(
        "--skip-crop-redetect",
        action="store_true",
        help=(
            "Skip Stage 4 high-resolution crop redetection. Stage 5 then merges "
            "the Stage 4 merged boxes directly, rereads them with classification "
            "padding, and continues through classification and comparative filtering."
        ),
    )
    parser.add_argument(
        "--skip-stage2-review",
        action="store_true",
        help=(
            "Skip the Stage 2a/2b review router and the optional Stage 3 feedback "
            "redetection. Stage 4 then starts from Stage 1 raw boxes."
        ),
    )
    parser.add_argument(
        "--force-stage3-redetect",
        action="store_true",
        help="Run Stage 3 feedback redetection even if Stage 2b does not trigger it.",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["breadth-first", "depth-first"],
        default="breadth-first",
        help=(
            "breadth-first completes a stage across all WSIs before moving on; "
            "depth-first completes one WSI at a time and parallelizes crops within that WSI."
        ),
    )
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wsi-exts", nargs="+", default=list(SUPPORTED_WSI_EXTENSIONS))
    parser.add_argument("--wsi-reader", default="auto", choices=["auto", "openslide", "cucim", "isyntax"])
    parser.add_argument("--backend", default="openrouter", choices=["openrouter", "vllm"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL)
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stage1-prompt", type=Path, default=DEFAULT_STAGE1_PROMPT)
    parser.add_argument("--stage2a-prompt", type=Path, default=DEFAULT_STAGE2A_PROMPT)
    parser.add_argument("--stage2b-first-prompt", type=Path, default=DEFAULT_STAGE2B_FIRST_PROMPT)
    parser.add_argument("--stage2b-second-prompt", type=Path, default=DEFAULT_STAGE2B_SECOND_PROMPT)
    parser.add_argument("--stage3-wrapper-prompt", type=Path, default=DEFAULT_STAGE3_WRAPPER_PROMPT)
    parser.add_argument("--classification-prompt", type=Path, default=DEFAULT_CLASSIFICATION_PROMPT)
    parser.add_argument("--odd-one-out-prompt", type=Path, default=DEFAULT_ODD_ONE_OUT_PROMPT)
    parser.add_argument("--stage1-thumbnail-max-dim", type=int, default=2048)
    parser.add_argument("--stage1-rotations", type=int, nargs="+", default=[0])
    parser.add_argument("--stage1-source-rotation", type=int, default=0)
    parser.add_argument("--stage1-coord-order", default="auto")
    parser.add_argument("--stage1-padding-frac", type=float, default=0.25)
    parser.add_argument("--stage1-merge-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--stage1-max-retries", type=int, default=3)
    parser.add_argument(
        "--stage1-repair-model",
        default=None,
        help="Optional JSON-repair model for Stage 1. Defaults to --model.",
    )
    parser.add_argument(
        "--post-stage3-padding-frac",
        "--stage3-bbox-padding-frac",
        "--stage4-bbox-padding-frac",
        dest="post_stage3_padding_frac",
        type=float,
        default=0.15,
        help="Padding fraction applied to Stage 4 merged source boxes before high-res crop redetection.",
    )
    parser.add_argument("--classification-padding-frac", type=float, default=0.10)
    parser.add_argument(
        "--merge-iou-threshold",
        type=float,
        default=0.40,
        help="Merge boxes when IoU is greater than this threshold.",
    )
    parser.add_argument(
        "--containment-overlap-threshold",
        "--area-over-smaller-threshold",
        "--overlap-over-smaller-threshold",
        dest="containment_overlap_threshold",
        type=float,
        default=0.80,
        help="Merge boxes when intersection area divided by the smaller box area is at least this threshold.",
    )
    parser.add_argument("--crop-max-dim", type=int, default=1024)
    parser.add_argument("--classification-max-dim", type=int, default=1024)
    parser.add_argument("--stage2a-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--stage2b-reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--stage3-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--crop-redetect-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--classification-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--odd-one-out-reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--stage2a-max-tokens", type=int, default=1200)
    parser.add_argument("--stage2b-max-tokens", type=int, default=600)
    parser.add_argument("--stage2b-second-max-tokens", type=int, default=600)
    parser.add_argument("--stage3-max-tokens", type=int, default=10000)
    parser.add_argument("--crop-redetect-max-tokens", type=int, default=4000)
    parser.add_argument("--classification-max-tokens", type=int, default=800)
    parser.add_argument("--odd-one-out-max-tokens", type=int, default=16000)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
