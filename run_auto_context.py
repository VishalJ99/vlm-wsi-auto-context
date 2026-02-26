#!/usr/bin/env python3
# ABOUTME: Aggregate WSI foreground method runner (Stages 1-7) with per-bbox outputs.
# ABOUTME: Supports single WSI or list input and materializes a clean stage-organized output tree.
"""
Aggregate WSI Foreground Method Runner.

Runs Stages 1-7 in order:
  1) detect_foreground_regions_from_wsi_thumbnail.py
  2) run_artifact_qc_pipeline.py
  3) run_color_segmentation.py
  4) find_icl_regions.py
  5) reranker.py
  6) run_vlm_bbox_inference.py
  7) postprocess_mask.py

Input:
  - --wsi <path-or-filename>
  - --wsi-list <one or more txt files, one WSI per line each>

Output layout per WSI:
  {output_root}/{wsi_id}/{run_id}/
    stage1/
    stage7/                        <- WSI-level mask + metadata
      mask.npy
      metadata.json
    bboxes/{bbox_str}/stage2
    bboxes/{bbox_str}/stage3
    bboxes/{bbox_str}/stage4
    bboxes/{bbox_str}/stage5
    bboxes/{bbox_str}/stage6
    bboxes/{bbox_str}/stage7       <- per-bbox postprocessed outputs
    logs/
    pipeline_metadata.json
    pipeline_status.json

Implementation note:
  The runner stores script-native stage outputs under {run_dir}/_native during execution,
  then syncs the canonical outputs into the clean tree above.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import math
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

from utils.reproducibility import (
    SKIP_STAGE_REPRO_CHECK_ENV,
    check_dvc_clean,
    check_git_clean,
    log_unclean_state,
)
from utils.wsi_backend import (
    close_wsi,
    is_ndpi_path,
    load_wsi,
    normalize_wsi_reader,
    read_region_rgb,
)
from utils.wsi_paths import resolve_wsi_path


REPO_ROOT = Path(__file__).resolve().parent


class PipelineError(RuntimeError):
    """Fatal pipeline orchestration error."""


@dataclass
class CommandSpec:
    cmd: List[str]
    cwd: Path
    log_path: Path
    extra_env: Optional[Dict[str, str]] = None


@dataclass
class ResumePlan:
    action: str  # "new", "resume", "skip"
    run_id: Optional[str] = None
    detail: str = ""


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_model_path(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace("-", "_")


def sanitize_stage2_model(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def bbox_to_str(bbox: Sequence[int]) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return f"{x1}_{y1}_{x2}_{y2}"


def read_wsi_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"WSI list file not found: {path}")
    items: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    if not items:
        raise PipelineError(f"No WSI paths found in list: {path}")
    return items


def _looks_like_wsi_xml_header(col0: str, col1: str) -> bool:
    """Best-effort detection for header rows in wsi,xml CSV inputs."""
    left = col0.strip().lower()
    right = col1.strip().lower()
    wsi_tokens = {"wsi", "wsi_path", "slide", "slide_path", "svs", "svs_path"}
    xml_tokens = {"xml", "xml_path", "roi_xml", "roi_xml_path", "annotation_xml"}
    return left in wsi_tokens and (right in xml_tokens or "xml" in right)


def read_wsi_xml_list(path: Path) -> List[Tuple[str, str]]:
    """
    Parse a list file where each non-comment row is CSV: wsi_path,xml_path.

    Supports optional header rows (e.g. "wsi_path,xml_path").
    """
    if not path.exists():
        raise FileNotFoundError(f"WSI list file not found: {path}")

    items: List[Tuple[str, str]] = []
    first_data_row = True
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = next(csv.reader([raw]))
        except Exception as exc:
            raise PipelineError(f"Invalid CSV row at line {line_no} in {path}: {exc}") from exc
        if len(row) < 2:
            raise PipelineError(
                f"Expected CSV row with at least 2 columns (wsi_path,xml_path) at "
                f"{path}:{line_no}"
            )
        wsi_item = row[0].strip()
        xml_item = row[1].strip()
        if not wsi_item or not xml_item:
            raise PipelineError(
                f"Empty wsi/xml column at {path}:{line_no}. "
                "Expected: wsi_path,xml_path"
            )
        if first_data_row and _looks_like_wsi_xml_header(wsi_item, xml_item):
            first_data_row = False
            continue
        first_data_row = False
        items.append((wsi_item, xml_item))

    if not items:
        raise PipelineError(f"No WSI/XML rows found in list: {path}")
    return items


def cli_flag_present(argv: Sequence[str], flag: str) -> bool:
    """Return True if CLI flag appears in argv (including --flag=value form)."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def cli_any_flag_present(argv: Sequence[str], flags: Sequence[str]) -> bool:
    return any(cli_flag_present(argv, flag) for flag in flags)


def parse_port_from_url(url: str) -> Optional[int]:
    """Parse explicit port from URL-like input; return None when absent/unparseable."""
    candidate = (url or "").strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    try:
        return parsed.port
    except ValueError:
        return None


ALLSTAGE_BACKEND_STAGE_VALUES = {
    "openrouter": {
        "stage1_backend": "openrouter",
        "stage2_backend": "openrouter",
        "stage4_backend": "openrouter",
        "stage5_vlm_backend": "openrouter",
        "stage6_backend": "openrouter",
    },
    "vllm": {
        "stage1_backend": "vllm",
        "stage2_backend": "vllm",
        "stage4_backend": "vllm",
        "stage5_vlm_backend": "vllm",
        "stage6_backend": "vllm",
    },
    "vertex": {
        "stage1_backend": "vertex",
        "stage2_backend": "vertex",
        "stage4_backend": "vertex",
        "stage5_vlm_backend": "vertex",
        "stage6_backend": "vertex",
    },
}


def apply_allstage_overrides(
    args: argparse.Namespace,
    argv: Sequence[str],
) -> None:
    """Apply all-stage convenience flags unless a stage-specific flag was explicitly set."""
    if args.allstage_backend:
        backend_stage_values = ALLSTAGE_BACKEND_STAGE_VALUES[args.allstage_backend]
        backend_targets = (
            ("--stage1-backend", "stage1_backend"),
            ("--stage2-backend", "stage2_backend"),
            ("--stage4-backend", "stage4_backend"),
            ("--stage5-vlm-backend", "stage5_vlm_backend"),
            ("--stage6-backend", "stage6_backend"),
        )
        for flag, attr in backend_targets:
            if not cli_flag_present(argv, flag):
                setattr(args, attr, backend_stage_values[attr])

        if args.allstage_backend == "vertex":
            if not cli_any_flag_present(argv, ("--stage5-gemini-use-vertex", "--stage5-gemini-no-vertex")):
                args.stage5_gemini_use_vertex = True
            if not cli_any_flag_present(argv, ("--stage6-gemini-use-vertex", "--stage6-gemini-no-vertex")):
                args.stage6_gemini_use_vertex = True

    if args.allstage_model:
        model_targets = (
            ("--stage1-model", "stage1_model"),
            ("--stage2-model", "stage2_model"),
            ("--stage4-model", "stage4_model"),
            ("--stage5-vlm-model", "stage5_vlm_model"),
            ("--stage6-model", "stage6_model"),
        )
        for flag, attr in model_targets:
            if not cli_flag_present(argv, flag):
                setattr(args, attr, args.allstage_model)

    if args.allstage_wsi_reader:
        reader_targets = (
            (("--stage1-wsi-reader", "--stage1-reader"), "stage1_wsi_reader"),
            (("--stage2-wsi-reader", "--stage2-reader"), "stage2_wsi_reader"),
            (("--stage5-wsi-reader", "--stage5-reader"), "stage5_wsi_reader"),
            (("--stage6-wsi-reader", "--stage6-reader"), "stage6_wsi_reader"),
        )
        for flags, attr in reader_targets:
            if not cli_any_flag_present(argv, flags):
                setattr(args, attr, args.allstage_wsi_reader)

    if args.allstage_vllm_url:
        vllm_url_targets = (
            ("--stage1-vllm-url", "stage1_vllm_url"),
            ("--stage2-vllm-url", "stage2_vllm_url"),
            ("--stage4-vllm-url", "stage4_vllm_url"),
            ("--stage6-vllm-url", "stage6_vllm_url"),
        )
        for flag, attr in vllm_url_targets:
            if not cli_flag_present(argv, flag):
                setattr(args, attr, args.allstage_vllm_url)

        if not cli_flag_present(argv, "--stage5-vlm-port"):
            inferred_port = parse_port_from_url(args.allstage_vllm_url)
            if inferred_port is not None:
                args.stage5_vlm_port = inferred_port


def resolve_case_wsi_readers(args: argparse.Namespace, wsi_path: str) -> Dict[str, object]:
    ndpi = is_ndpi_path(wsi_path)
    requested = {
        "stage1": normalize_wsi_reader(args.stage1_wsi_reader),
        "stage2": normalize_wsi_reader(args.stage2_wsi_reader),
        "stage5": normalize_wsi_reader(args.stage5_wsi_reader),
        "stage6": normalize_wsi_reader(args.stage6_wsi_reader),
    }
    effective: Dict[str, str] = {}
    forced_to_openslide: List[str] = []
    for stage, reader in requested.items():
        if ndpi:
            effective[stage] = "openslide"
            if reader != "openslide":
                forced_to_openslide.append(stage)
        else:
            effective[stage] = reader
    return {
        "is_ndpi": ndpi,
        "requested": requested,
        "effective": effective,
        "forced_to_openslide": forced_to_openslide,
    }


def _stream_pipe_to_log_and_stdout(
    pipe,
    log_file,
    echo: bool = False,
    stdout_lock: Optional[threading.Lock] = None,
) -> None:
    while True:
        # read1() reduces latency for carriage-return progress updates.
        chunk = pipe.read1(256)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        log_file.write(text)
        log_file.flush()
        if echo:
            if stdout_lock is not None:
                with stdout_lock:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            else:
                sys.stdout.write(text)
                sys.stdout.flush()


def run_command(spec: CommandSpec, dry_run: bool = False, live_logs: bool = False) -> None:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(subprocess.list2cmdline([x]) if " " in x else x for x in spec.cmd)
    if dry_run:
        dry_log_path = spec.log_path.with_name(f"{spec.log_path.stem}.dryrun{spec.log_path.suffix}")
        dry_log_path.write_text(f"[DRY RUN] cwd={spec.cwd}\n{line}\n", encoding="utf-8")
        print(f"[DRY RUN] {line}")
        return

    with spec.log_path.open("w", encoding="utf-8") as f:
        f.write(f"[cwd] {spec.cwd}\n")
        f.write(f"[cmd] {' '.join(spec.cmd)}\n\n")
        f.flush()
        env = os.environ.copy()
        if spec.extra_env:
            env.update(spec.extra_env)
        if live_logs:
            proc = subprocess.Popen(
                spec.cmd,
                cwd=str(spec.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                env=env,
            )
            if proc.stdout is None:  # pragma: no cover
                raise PipelineError("Failed to capture subprocess stdout for live logs")
            _stream_pipe_to_log_and_stdout(proc.stdout, f, echo=True)
            proc.stdout.close()
            returncode = proc.wait()
        else:
            proc_done = subprocess.run(
                spec.cmd,
                cwd=str(spec.cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                env=env,
            )
            returncode = proc_done.returncode
    if returncode != 0:
        raise PipelineError(
            f"Command failed (exit={returncode}). See log: {spec.log_path}"
        )


def run_parallel(
    spec_a: CommandSpec,
    spec_b: CommandSpec,
    dry_run: bool = False,
    live_logs: bool = False,
) -> None:
    if dry_run:
        run_command(spec_a, dry_run=True, live_logs=live_logs)
        run_command(spec_b, dry_run=True, live_logs=live_logs)
        return

    spec_a.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec_b.log_path.parent.mkdir(parents=True, exist_ok=True)

    with spec_a.log_path.open("w", encoding="utf-8") as fa, spec_b.log_path.open(
        "w", encoding="utf-8"
    ) as fb:
        fa.write(f"[cwd] {spec_a.cwd}\n[cmd] {' '.join(spec_a.cmd)}\n\n")
        fb.write(f"[cwd] {spec_b.cwd}\n[cmd] {' '.join(spec_b.cmd)}\n\n")
        fa.flush()
        fb.flush()
        env_a = os.environ.copy()
        if spec_a.extra_env:
            env_a.update(spec_a.extra_env)
        env_b = os.environ.copy()
        if spec_b.extra_env:
            env_b.update(spec_b.extra_env)

        if live_logs:
            pa = subprocess.Popen(
                spec_a.cmd,
                cwd=str(spec_a.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                env=env_a,
            )
            pb = subprocess.Popen(
                spec_b.cmd,
                cwd=str(spec_b.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                env=env_b,
            )
            if pa.stdout is None or pb.stdout is None:  # pragma: no cover
                raise PipelineError("Failed to capture subprocess stdout for live logs")

            stdout_lock = threading.Lock()
            ta = threading.Thread(
                target=_stream_pipe_to_log_and_stdout,
                args=(pa.stdout, fa, True, stdout_lock),
                daemon=True,
            )
            tb = threading.Thread(
                target=_stream_pipe_to_log_and_stdout,
                args=(pb.stdout, fb, True, stdout_lock),
                daemon=True,
            )
            ta.start()
            tb.start()
            ra = pa.wait()
            rb = pb.wait()
            ta.join()
            tb.join()
            pa.stdout.close()
            pb.stdout.close()
        else:
            pa = subprocess.Popen(
                spec_a.cmd,
                cwd=str(spec_a.cwd),
                stdout=fa,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_a,
            )
            pb = subprocess.Popen(
                spec_b.cmd,
                cwd=str(spec_b.cwd),
                stdout=fb,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_b,
            )
            ra = pa.wait()
            rb = pb.wait()

    failures: List[str] = []
    if ra != 0:
        failures.append(f"{spec_a.log_path} (exit={ra})")
    if rb != 0:
        failures.append(f"{spec_b.log_path} (exit={rb})")
    if failures:
        raise PipelineError(
            "Parallel stage command failed. Logs:\n- " + "\n- ".join(failures)
        )


def latest_subdir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    dirs = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
    return dirs[-1] if dirs else None


def sync_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise PipelineError(f"Missing source directory to sync: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ensure_files(base: Path, required: Sequence[str], context: str) -> None:
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise PipelineError(f"Missing {context} files in {base}: {missing}")


STAGE2_BBOX_REQUIRED_FILES: Tuple[str, ...] = (
    "bbox_region.png",
    "stage1_artifact_perception.json",
    "stage2_claim_evidence.json",
    "stage3_strength.json",
    "stage4_verdicts.json",
)
ARTIFACT_KEYS: Tuple[str, ...] = (
    "ink_or_pen_marks",
    "debris",
    "labels",
    "air_bubbles",
    "cracks",
    "tissue_folds",
    "paraffin_mounting_medium",
)
ROTATION_KEYS: Tuple[str, ...] = ("0", "90", "180", "270")
STAGE1_MIN_REQUIRED_FILES: Tuple[str, ...] = ("metadata.json", "thumbnail.png")
STAGE1_REQUIRED_FILES: Tuple[str, ...] = ("bboxes.json", "metadata.json", "thumbnail.png")
STAGE3_BBOX_REQUIRED_FILES: Tuple[str, ...] = ("mask.png", "metadata.json", "crop.png")
STAGE4_BBOX_REQUIRED_FILES: Tuple[str, ...] = ("metadata.json",)
STAGE5_BBOX_REQUIRED_FILES: Tuple[str, ...] = ("metadata.json",)
STAGE6_BBOX_REQUIRED_FILES: Tuple[str, ...] = ("metadata.json", "patches.csv", "class_map.npy")
STAGE7_BBOX_REQUIRED_FILES: Tuple[str, ...] = (
    "postprocess_metadata.json",
    "tissue_mask_post.npy",
    "class_map_postprocessed.npy",
)
STAGE5_FALLBACK_CLASSES: Tuple[str, ...] = ("background", "tissue")
STAGE45_SKIP_REASON_NO_ICL_BASELINE = "stage6_icl_k0_skip_stage45"
STAGE6_GENERIC_CLASS_DESCRIPTIONS: Dict[str, str] = {
    "background": "Glass/background or empty region without diagnostic tissue structure.",
    "tissue": "Histopathology tissue region with visible cellular or stromal morphology.",
    "paraffin_mounting_medium": "Paraffin or mounting-medium artifact region.",
    "pen_ink_marks": "Pen or ink marking artifact on the slide.",
}


def summarize_stage4_points(stage4_dir: Path) -> Dict[str, object]:
    """
    Summarize point counts for Stage 4 output.

    Prefers counting entries from points.json files. If none exist, falls back to
    metadata.json total_count when available.
    """
    class_counts: Dict[str, int] = {}
    points_sources: List[str] = []
    computed_total = 0

    point_files: List[Path] = []
    root_points = stage4_dir / "points.json"
    if root_points.exists():
        point_files.append(root_points)
    point_files.extend(sorted(stage4_dir.glob("rot_*/points.json")))

    for points_path in point_files:
        points_sources.append(str(points_path))
        try:
            with points_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        classes = payload.get("classes")
        if not isinstance(classes, dict):
            continue
        for label, entries in classes.items():
            if not isinstance(entries, list):
                continue
            count = len(entries)
            if count <= 0:
                continue
            key = str(label)
            class_counts[key] = class_counts.get(key, 0) + count
            computed_total += count

    metadata_total: Optional[int] = None
    meta_path = stage4_dir / "metadata.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                stage4_meta = json.load(f)
            if isinstance(stage4_meta, dict):
                raw_total = stage4_meta.get("total_count")
                if isinstance(raw_total, (int, float)):
                    metadata_total = max(0, int(raw_total))
        except Exception:
            metadata_total = None

    if points_sources:
        total_points = computed_total
        source = "points_json"
    elif metadata_total is not None:
        total_points = metadata_total
        source = "metadata_total_count"
    else:
        total_points = 0
        source = "none"

    return {
        "total_points": int(total_points),
        "class_counts": class_counts,
        "points_sources": points_sources,
        "source": source,
        "metadata_total_count": metadata_total,
    }


def should_skip_stage45_for_no_icl_baseline(args: argparse.Namespace) -> bool:
    """
    Skip Stage 4/5 when Stage 6 is explicitly configured for no-ICL baseline.

    We only auto-skip when Stage 5 description generation is disabled to avoid
    silently changing behavior for users who requested Stage 5 descriptions.
    """
    return int(args.stage6_icl_k) == 0 and not bool(args.stage5_generate_descriptions)


def _effective_stage5_transport_backend(args: argparse.Namespace) -> str:
    """Return effective Stage 5 transport backend after convenience aliases."""
    backend = (args.stage5_vlm_backend or "openrouter").strip().lower()
    if backend == "vertex":
        return "vertex"
    if backend == "gemini":
        return "vertex" if bool(args.stage5_gemini_use_vertex) else "gemini"
    return backend


def _effective_stage6_transport_backend(args: argparse.Namespace) -> str:
    """Return effective Stage 6 transport backend after convenience aliases."""
    backend = (args.stage6_backend or "vllm").strip().lower()
    if backend == "vertex":
        return "vertex"
    if backend == "gemini":
        return "vertex" if bool(args.stage6_gemini_use_vertex) else "gemini"
    return backend


def resolve_bbox_parallel_workers(
    args: argparse.Namespace,
    *,
    max_stage: int,
    bbox_count: int,
    skip_stage45_no_icl_baseline: bool,
) -> Tuple[int, Optional[str]]:
    """
    Decide per-WSI bbox concurrency for Stage 4+ processing.

    Parallel bbox execution is enabled only when all active per-bbox VLM stages
    use remote backends (`openrouter` or `vertex` transport). This avoids
    overwhelming local vLLM/GPU configurations by default.
    """
    requested = int(args.parallelise_bboxes)
    if requested <= 1 or bbox_count <= 1:
        return 1, None

    allowed_backends = {"openrouter", "vertex"}
    backend_checks: List[Tuple[str, str]] = []

    if max_stage >= 4 and not skip_stage45_no_icl_baseline:
        backend_checks.append(("stage4", (args.stage4_backend or "").strip().lower()))
    if max_stage >= 5 and not skip_stage45_no_icl_baseline:
        backend_checks.append(("stage5", _effective_stage5_transport_backend(args)))
    if max_stage >= 6:
        backend_checks.append(("stage6", _effective_stage6_transport_backend(args)))

    disallowed = [f"{stage}:{backend}" for stage, backend in backend_checks if backend not in allowed_backends]
    if disallowed:
        reason = (
            "disabled --parallelise-bboxes because active per-bbox stages are not "
            f"all on vertex/openrouter transport ({', '.join(disallowed)})"
        )
        return 1, reason

    return min(requested, bbox_count), None


def materialize_stage4_no_icl_baseline_fallback(
    *,
    stage4_dir: Path,
    wsi_path: str,
    bbox: Sequence[int],
) -> None:
    """Create synthetic Stage 4 metadata when no-ICL baseline mode skips Stage 4."""
    if stage4_dir.exists():
        shutil.rmtree(stage4_dir)
    stage4_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "wsi_path": os.path.abspath(wsi_path),
        "bbox": [int(v) for v in bbox],
        "total_count": 0,
        "classes": {c: [] for c in STAGE5_FALLBACK_CLASSES},
        "fallback": {
            "used": True,
            "reason": STAGE45_SKIP_REASON_NO_ICL_BASELINE,
            "created_by": "run_auto_context.py",
            "created_at": datetime.now().isoformat(),
        },
        "reproducibility": {
            "created_at": datetime.now().isoformat(),
            "run_config": {"mode": "stage4_skipped_no_icl_baseline"},
        },
    }
    write_json(stage4_dir / "metadata.json", metadata)


def materialize_stage5_disable_icl_fallback(
    *,
    stage5_dir: Path,
    stage4_dir: Path,
    wsi_path: str,
    bbox: Sequence[int],
    stage5_patch_size: int,
    stage5_k: int,
) -> None:
    """
    Create synthetic Stage 5 metadata for no-ICL baseline mode.

    This preserves Stage 6 contract (Stage 5 metadata + patch size) while
    intentionally providing an empty ICL pool.
    """
    if stage5_dir.exists():
        shutil.rmtree(stage5_dir)
    stage5_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = stage5_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    class_list = list(STAGE5_FALLBACK_CLASSES)
    selected_indices = {c: [] for c in class_list}
    reasoning = {
        c: "No-ICL baseline mode: Stage 4/5 were skipped by pipeline."
        for c in class_list
    }
    output = {c: [] for c in class_list}
    candidate_counts = {c: 0 for c in class_list}

    write_json(
        intermediate_dir / "stage4_input.json",
        {
            "stage4_dir": str(stage4_dir),
            "mode": "stage4_stage5_skipped_no_icl_baseline",
        },
    )

    metadata = {
        "stage4_input": str(stage4_dir),
        "wsi_path": os.path.abspath(wsi_path),
        "bbox": [int(v) for v in bbox],
        "classes_present": class_list,
        "classes_ranked": class_list,
        "classes_skipped": [],
        "k_per_class": {c: 0 for c in class_list},
        "patch_extraction": {
            "k": int(stage5_k),
            "patch_size": int(stage5_patch_size),
            "level": 0,
            "candidates_per_class": candidate_counts,
            "candidate_manifest": None,
            "sampling": None,
        },
        "ranking": {
            "ranker": "none",
            "config": {"mode": "stage5_disable_icl_fallback"},
            "selected_indices": selected_indices,
            "index_space": "global_flat_list",
            "reasoning": reasoning,
        },
        "output": output,
        "fallback": {
            "used": True,
            "reason": STAGE45_SKIP_REASON_NO_ICL_BASELINE,
            "disable_icl": True,
            "created_by": "run_auto_context.py",
            "created_at": datetime.now().isoformat(),
        },
        "reproducibility": {
            "created_at": datetime.now().isoformat(),
            "run_config": {"mode": "stage4_stage5_skipped_no_icl_baseline"},
        },
    }
    write_json(stage5_dir / "metadata.json", metadata)
    (stage5_dir / "reproduce.txt").write_text(
        "Synthetic fallback created by run_auto_context.py "
        "because no-ICL baseline mode skipped Stage 4/5.\n",
        encoding="utf-8",
    )


def load_stage5_fallback_info(stage5_dir: Path) -> Dict[str, object]:
    """Read Stage 5 fallback info from metadata.json."""
    info: Dict[str, object] = {
        "used": False,
        "reason": None,
        "disable_icl": False,
        "no_points": False,
    }
    meta_path = stage5_dir / "metadata.json"
    if not meta_path.exists():
        return info
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return info
    if not isinstance(meta, dict):
        return info

    reason: Optional[str] = None
    fallback = meta.get("fallback")
    if isinstance(fallback, dict):
        reason = str(fallback.get("reason", "")).strip().lower() or None
        if reason:
            info["used"] = True
            info["reason"] = reason
        if bool(fallback.get("disable_icl")):
            info["disable_icl"] = True

    ranking = meta.get("ranking")
    if isinstance(ranking, dict):
        config = ranking.get("config")
        if isinstance(config, dict):
            mode = str(config.get("mode", "")).strip().lower()
            if mode == "stage4_no_points_fallback":
                info["used"] = True
                info["reason"] = info["reason"] or "stage4_no_points"
                info["no_points"] = True
                info["disable_icl"] = True
            elif mode == "stage5_disable_icl_fallback":
                info["used"] = True
                info["reason"] = info["reason"] or "stage5_disable_icl_fallback"
                info["disable_icl"] = True

    if reason == "stage4_no_points":
        info["no_points"] = True
        info["disable_icl"] = True
    elif reason in (
        "stage5_no_tissue_after_blur",
        "stage5_no_tissue_candidates",
        "stage5_no_tissue_selected",
        STAGE45_SKIP_REASON_NO_ICL_BASELINE,
    ):
        info["disable_icl"] = True

    return info


def is_stage5_no_points_fallback(stage5_dir: Path) -> bool:
    """Detect whether stage5 metadata indicates Stage 4 no-points fallback."""
    return bool(load_stage5_fallback_info(stage5_dir).get("no_points"))


def materialize_stage5_no_points_fallback(
    *,
    stage5_dir: Path,
    stage4_dir: Path,
    wsi_path: str,
    bbox: Sequence[int],
    stage4_points_summary: Dict[str, object],
    stage5_patch_size: int,
    stage5_k: int,
) -> None:
    """
    Create a synthetic Stage 5 run when Stage 4 produced zero points.

    This preserves Stage 6 contract (metadata + class list + patch size) while
    intentionally providing an empty ICL pool.
    """
    if stage5_dir.exists():
        shutil.rmtree(stage5_dir)
    stage5_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = stage5_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    stage2_input = None
    stage4_meta_path = stage4_dir / "metadata.json"
    if stage4_meta_path.exists():
        try:
            with stage4_meta_path.open("r", encoding="utf-8") as f:
                stage4_meta = json.load(f)
            if isinstance(stage4_meta, dict):
                stage2_input = stage4_meta.get("stage2_input")
        except Exception:
            stage2_input = None

    class_list = list(STAGE5_FALLBACK_CLASSES)
    selected_indices = {c: [] for c in class_list}
    reasoning = {
        c: "No Stage 4 points detected; reranker skipped by pipeline fallback."
        for c in class_list
    }
    output = {c: [] for c in class_list}
    candidate_counts = {c: 0 for c in class_list}

    write_json(
        intermediate_dir / "stage4_input.json",
        {
            "stage4_dir": str(stage4_dir),
            "points_summary": stage4_points_summary,
        },
    )

    metadata = {
        "stage4_input": str(stage4_dir),
        "stage2_input": stage2_input,
        "wsi_path": os.path.abspath(wsi_path),
        "bbox": [int(v) for v in bbox],
        "classes_present": class_list,
        "classes_ranked": class_list,
        "classes_skipped": [],
        "k_per_class": {c: 0 for c in class_list},
        "patch_extraction": {
            "k": int(stage5_k),
            "patch_size": int(stage5_patch_size),
            "level": 0,
            "candidates_per_class": candidate_counts,
            "candidate_manifest": None,
            "sampling": None,
        },
        "ranking": {
            "ranker": "none",
            "config": {"mode": "stage4_no_points_fallback"},
            "selected_indices": selected_indices,
            "index_space": "global_flat_list",
            "reasoning": reasoning,
        },
        "output": output,
        "fallback": {
            "used": True,
            "reason": "stage4_no_points",
            "stage4_points_total": int(stage4_points_summary.get("total_points", 0)),
            "stage4_points_source": stage4_points_summary.get("source"),
            "created_by": "run_auto_context.py",
            "created_at": datetime.now().isoformat(),
        },
        "reproducibility": {
            "created_at": datetime.now().isoformat(),
            "run_config": {"mode": "stage4_no_points_fallback"},
        },
    }
    write_json(stage5_dir / "metadata.json", metadata)
    (stage5_dir / "reproduce.txt").write_text(
        "Synthetic fallback created by run_auto_context.py "
        "because Stage 4 produced no points.\n",
        encoding="utf-8",
    )


def write_generic_stage6_class_descriptions(path: Path) -> None:
    """Write generic class descriptions for Stage 6 fallback path."""
    write_json(path, STAGE6_GENERIC_CLASS_DESCRIPTIONS)


def run_repro_preflight(skip_dvc_check: bool = False) -> Dict[str, object]:
    git_clean, git_details = check_git_clean()
    if skip_dvc_check:
        dvc_clean = True
        dvc_details = {"warning": "DVC check skipped"}
    else:
        dvc_clean, dvc_details = check_dvc_clean()

    if not git_clean or not dvc_clean:
        log_unclean_state(git_details, dvc_details)
        raise PipelineError("Repro preflight failed: repository state is not clean")

    return {
        "enabled": True,
        "git_hash": git_details.get("commit_hash", "unknown"),
        "git_clean": git_clean,
        "dvc_clean": dvc_clean,
        "skip_dvc_check": skip_dvc_check,
    }


def acquire_run_lock(run_dir: Path, wsi_id: str, run_id: str):
    lock_path = run_dir / ".pipeline.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        holder = lock_file.read().strip()
        lock_file.close()
        detail = f" lock holder: {holder}" if holder else ""
        raise PipelineError(
            f"Run directory is already in use for {wsi_id} (run_id={run_id}).{detail}"
        ) from exc

    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(
        f"pid={os.getpid()} started_at={datetime.now().isoformat()} "
        f"wsi_id={wsi_id} run_id={run_id}\n"
    )
    lock_file.flush()
    return lock_file


def release_run_lock(lock_file) -> None:
    if lock_file is None:
        return
    lock_path = Path(lock_file.name)
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()
    try:
        lock_path.unlink()
    except OSError:
        pass


def load_bboxes(stage1_dir: Path) -> List[Tuple[str, List[int]]]:
    bboxes_path = stage1_dir / "bboxes.json"
    if not bboxes_path.exists():
        raise PipelineError(f"Stage1 bboxes.json not found: {bboxes_path}")

    with bboxes_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        regions = data.get("detected_regions", [])
    elif isinstance(data, list):
        regions = data
    else:
        raise PipelineError(f"Unsupported bboxes format in {bboxes_path}")

    out: List[Tuple[str, List[int]]] = []
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox_level0")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bbox_key = bbox_to_str(bbox)
        out.append((bbox_key, [int(v) for v in bbox]))
    if not out:
        return []
    # preserve order from stage1
    seen = set()
    uniq: List[Tuple[str, List[int]]] = []
    for k, b in out:
        if k in seen:
            continue
        seen.add(k)
        uniq.append((k, b))
    return uniq


def _coerce_bbox_xyxy(value: object, max_w: int, max_h: int) -> Optional[List[int]]:
    """Parse and clamp [x1, y1, x2, y2] to image bounds."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None

    w = int(max_w)
    h = int(max_h)
    if w <= 0 or h <= 0:
        return None

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_level0_to_thumbnail(
    bbox_level0: Sequence[int],
    wsi_w: int,
    wsi_h: int,
    thumb_w: int,
    thumb_h: int,
) -> List[int]:
    sx = thumb_w / float(wsi_w)
    sy = thumb_h / float(wsi_h)
    x1 = int(math.floor(int(bbox_level0[0]) * sx))
    y1 = int(math.floor(int(bbox_level0[1]) * sy))
    x2 = int(math.ceil(int(bbox_level0[2]) * sx))
    y2 = int(math.ceil(int(bbox_level0[3]) * sy))
    x1 = max(0, min(max(thumb_w - 1, 0), x1))
    y1 = max(0, min(max(thumb_h - 1, 0), y1))
    x2 = max(x1 + 1, min(thumb_w, x2))
    y2 = max(y1 + 1, min(thumb_h, y2))
    return [x1, y1, x2, y2]


def _bbox_level0_to_normalized(bbox_level0: Sequence[int], wsi_w: int, wsi_h: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_level0]
    vals = [
        int(round((x1 / float(wsi_w)) * 1000.0)),
        int(round((y1 / float(wsi_h)) * 1000.0)),
        int(round((x2 / float(wsi_w)) * 1000.0)),
        int(round((y2 / float(wsi_h)) * 1000.0)),
    ]
    return [max(0, min(1000, v)) for v in vals]


def ensure_stage1_bboxes_with_fullslide_fallback(stage1_dir: Path) -> Dict[str, object]:
    """
    Normalize Stage 1 bbox payloads and synthesize one full-slide bbox when empty.

    Returns a summary dict with fallback metadata for pipeline-level logging.
    """
    meta_path = stage1_dir / "metadata.json"
    thumb_path = stage1_dir / "thumbnail.png"
    bboxes_path = stage1_dir / "bboxes.json"

    if not meta_path.exists() or not thumb_path.exists():
        raise PipelineError(
            f"Stage 1 fallback requires metadata.json and thumbnail.png under {stage1_dir}"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise PipelineError(f"Invalid Stage 1 metadata format in {meta_path}")

    with Image.open(thumb_path) as thumb:
        thumb_w, thumb_h = thumb.size
    if thumb_w <= 0 or thumb_h <= 0:
        raise PipelineError(f"Invalid Stage 1 thumbnail dimensions in {thumb_path}")

    wsi_dims = metadata.get("wsi_dimensions", {})
    wsi_w = int(wsi_dims.get("width", 0))
    wsi_h = int(wsi_dims.get("height", 0))
    if wsi_w <= 0 or wsi_h <= 0:
        raise PipelineError(f"Invalid stage1 wsi_dimensions in {meta_path}")

    raw_bboxes: Optional[object] = None
    raw_regions: List[object] = []
    fallback_reason: Optional[str] = None

    if bboxes_path.exists():
        try:
            with bboxes_path.open("r", encoding="utf-8") as f:
                raw_bboxes = json.load(f)
            if isinstance(raw_bboxes, dict):
                maybe_regions = raw_bboxes.get("detected_regions", [])
                raw_regions = maybe_regions if isinstance(maybe_regions, list) else []
            elif isinstance(raw_bboxes, list):
                raw_regions = raw_bboxes
            else:
                fallback_reason = "unsupported_bboxes_json_format"
        except Exception as exc:  # pragma: no cover - defensive against malformed json
            fallback_reason = f"bboxes_json_unreadable:{type(exc).__name__}"
    else:
        fallback_reason = "bboxes_json_missing"

    normalized_regions: List[Dict[str, object]] = []
    for region in raw_regions:
        if isinstance(region, dict):
            bbox_raw = region.get("bbox_level0") or region.get("bbox") or region.get("box_2d")
            region_out = dict(region)
        elif isinstance(region, (list, tuple)):
            bbox_raw = region
            region_out = {}
        else:
            continue

        bbox_level0 = _coerce_bbox_xyxy(bbox_raw, max_w=wsi_w, max_h=wsi_h)
        if bbox_level0 is None:
            continue

        bbox_thumb = _coerce_bbox_xyxy(region_out.get("bbox_thumbnail"), max_w=thumb_w, max_h=thumb_h)
        if bbox_thumb is None:
            bbox_thumb = _bbox_level0_to_thumbnail(bbox_level0, wsi_w=wsi_w, wsi_h=wsi_h, thumb_w=thumb_w, thumb_h=thumb_h)

        region_out["label"] = region_out.get("label") or f"tissue_{len(normalized_regions) + 1}"
        region_out["bbox_level0"] = bbox_level0
        region_out["bbox_thumbnail"] = bbox_thumb
        region_out["bbox_normalized"] = _bbox_level0_to_normalized(bbox_level0, wsi_w=wsi_w, wsi_h=wsi_h)
        normalized_regions.append(region_out)

    fallback_applied = False
    if not normalized_regions:
        fallback_applied = True
        if fallback_reason is None:
            fallback_reason = "no_valid_bbox_level0_entries"
        normalized_regions = [
            {
                "label": "tissue_1",
                "bbox_normalized": [0, 0, 1000, 1000],
                "bbox_thumbnail": [0, 0, int(thumb_w), int(thumb_h)],
                "bbox_level0": [0, 0, int(wsi_w), int(wsi_h)],
                "synthetic": True,
                "synthetic_source": "run_foreground_pipeline_stage1_fullslide_fallback",
            }
        ]

    bboxes_payload: Dict[str, object] = {}
    if isinstance(raw_bboxes, dict):
        bboxes_payload.update(raw_bboxes)
    bboxes_payload["detected_regions"] = normalized_regions
    bboxes_payload["regions_count"] = len(normalized_regions)
    write_json(bboxes_path, bboxes_payload)

    prior_fallback = metadata.get("stage1_fallback")
    prior_used = bool(prior_fallback.get("used")) if isinstance(prior_fallback, dict) else False
    fallback_used = fallback_applied or prior_used
    stage1_fallback_meta: Dict[str, object] = {
        "used": fallback_used,
        "mode": "full_slide_bbox",
    }
    if fallback_reason:
        stage1_fallback_meta["reason"] = fallback_reason
    if fallback_applied:
        stage1_fallback_meta["applied_at"] = datetime.now().isoformat()
        stage1_fallback_meta["synthetic_bbox_level0"] = normalized_regions[0]["bbox_level0"]
    elif isinstance(prior_fallback, dict):
        for key in ("reason", "applied_at", "synthetic_bbox_level0"):
            if key in prior_fallback:
                stage1_fallback_meta[key] = prior_fallback[key]

    metadata["detected_regions"] = normalized_regions
    metadata["regions_count"] = len(normalized_regions)
    metadata["stage1_fallback"] = stage1_fallback_meta
    write_json(meta_path, metadata)

    return {
        "used": fallback_used,
        "applied": fallback_applied,
        "mode": "full_slide_bbox",
        "reason": fallback_reason,
        "bbox_count": len(normalized_regions),
        "bbox_level0": normalized_regions[0]["bbox_level0"] if normalized_regions else None,
    }


def stage_done(stage_dir: Path, required_files: Sequence[str]) -> bool:
    return stage_dir.exists() and all((stage_dir / f).exists() for f in required_files)


def find_stage1_native_dir(native_root: Path, wsi_id: str, model: str) -> Optional[Path]:
    return latest_subdir(native_root / "stage1_output" / wsi_id / sanitize_model_path(model))


def find_stage2_native_run(native_root: Path, wsi_id: str, model: str) -> Optional[Path]:
    return latest_subdir(native_root / "stage2_output" / wsi_id / sanitize_stage2_model(model))


def _rotation_vote_map(value: str) -> Dict[str, str]:
    return {k: value for k in ROTATION_KEYS}


def build_exclude_all_verdicts() -> Dict[str, Dict[str, object]]:
    verdicts: Dict[str, Dict[str, object]] = {}
    for key in ARTIFACT_KEYS:
        verdicts[key] = {
            "votes": _rotation_vote_map("SD"),
            "counts": {"SD": len(ROTATION_KEYS), "WA": 0, "SA": 0},
            "verdict": "EXCLUDE",
        }
    return verdicts


def build_exclude_all_strength() -> Dict[str, Dict[str, str]]:
    return {rot: {key: "SD" for key in ARTIFACT_KEYS} for rot in ROTATION_KEYS}


def _read_stage1_thumbnail_crop(
    stage1_dir: Path,
    bbox: Sequence[int],
    max_dim: int,
) -> Image.Image:
    """Approximate Stage2 bbox_region.png from Stage1 thumbnail and bbox."""
    meta_path = stage1_dir / "metadata.json"
    thumb_path = stage1_dir / "thumbnail.png"
    if not meta_path.exists() or not thumb_path.exists():
        raise PipelineError(f"Stage1 metadata/thumbnail missing under {stage1_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    thumb = Image.open(thumb_path).convert("RGB")
    thumb_w, thumb_h = thumb.size

    bbox_thumb: Optional[List[int]] = None
    bbox_list = [int(v) for v in bbox]
    for region in meta.get("detected_regions", []):
        if region.get("bbox_level0") == bbox_list:
            cand = region.get("bbox_thumbnail")
            if isinstance(cand, list) and len(cand) == 4:
                bbox_thumb = [int(v) for v in cand]
                break

    if bbox_thumb is None:
        wsi_dims = meta.get("wsi_dimensions", {})
        wsi_w = int(wsi_dims.get("width", 0))
        wsi_h = int(wsi_dims.get("height", 0))
        if wsi_w <= 0 or wsi_h <= 0:
            raise PipelineError(f"Invalid stage1 wsi_dimensions in {meta_path}")
        sx = thumb_w / float(wsi_w)
        sy = thumb_h / float(wsi_h)
        x1 = int(math.floor(int(bbox[0]) * sx))
        y1 = int(math.floor(int(bbox[1]) * sy))
        x2 = int(math.ceil(int(bbox[2]) * sx))
        y2 = int(math.ceil(int(bbox[3]) * sy))
    else:
        # Stage1 stores thumbnail boxes in standard xyxy order.
        x1, y1, x2, y2 = bbox_thumb

    x1 = max(0, min(thumb_w - 1, x1))
    y1 = max(0, min(thumb_h - 1, y1))
    x2 = max(x1 + 1, min(thumb_w, x2))
    y2 = max(y1 + 1, min(thumb_h, y2))

    crop = thumb.crop((x1, y1, x2, y2))
    target_max = max(1, int(max_dim))
    long_edge = max(crop.size)
    if long_edge > 0 and long_edge != target_max:
        scale = target_max / float(long_edge)
        new_size = (
            max(1, int(round(crop.size[0] * scale))),
            max(1, int(round(crop.size[1] * scale))),
        )
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize(new_size, resample=resampling)
    return crop


def _read_wsi_level0_crop(
    wsi_path: str,
    bbox: Sequence[int],
    max_dim: int,
    wsi_reader: str,
) -> Image.Image:
    """Read bbox directly from WSI level 0 and downsample to max_dim if needed."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    wsi, resolved_wsi_reader = load_wsi(wsi_path, wsi_reader)
    try:
        arr = read_region_rgb(
            wsi,
            resolved_wsi_reader,
            x=x1,
            y=y1,
            width=width,
            height=height,
            level=0,
        )
    finally:
        close_wsi(wsi, resolved_wsi_reader)

    crop = Image.fromarray(arr).convert("RGB")
    target_max = max(1, int(max_dim))
    long_edge = max(crop.size)
    if long_edge > target_max:
        scale = target_max / float(long_edge)
        new_size = (
            max(1, int(round(crop.size[0] * scale))),
            max(1, int(round(crop.size[1] * scale))),
        )
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize(new_size, resample=resampling)
    return crop


def _resolve_stage1_bbox_region_max_dim(
    stage1_dir: Path,
    default_max_dim: int = 1024,
) -> int:
    """Resolve Stage 1 bbox-region max_dim from metadata when available."""
    meta_path = stage1_dir / "metadata.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        raw = meta.get("max_dim")
        value = int(raw)
        if value > 0:
            return value
    except Exception:
        pass
    return max(1, int(default_max_dim))


def ensure_stage1_bbox_region_exports(
    *,
    stage1_dir: Path,
    wsi_path: str,
    bboxes: List[Tuple[str, List[int]]],
    wsi_reader: str,
    max_dim: int,
    dry_run: bool,
) -> Dict[str, object]:
    """
    Ensure Stage 1 bbox_regions/{bbox_str}/bbox_region.png exists for all bboxes.

    Missing files are backfilled from WSI level-0 reads.
    """
    bbox_regions_dir = stage1_dir / "bbox_regions"
    missing: List[Tuple[str, List[int]]] = []
    existing = 0
    for bbox_str, bbox in bboxes:
        out_png = bbox_regions_dir / bbox_str / "bbox_region.png"
        if out_png.exists():
            existing += 1
        else:
            missing.append((bbox_str, bbox))

    info: Dict[str, object] = {
        "enabled": True,
        "max_dim": int(max_dim),
        "total_bboxes": len(bboxes),
        "existing_before": existing,
        "missing_before": len(missing),
        "generated": 0,
        "wsi_reader": wsi_reader,
        "dry_run": bool(dry_run),
    }

    if not missing:
        print("Stage 1 bbox regions: all bbox_region.png files already present")
        return info

    print(
        "Stage 1 bbox regions: missing "
        f"{len(missing)} file(s); backfilling from L0 reads (max_dim={max_dim})"
    )
    if dry_run:
        info["generated"] = len(missing)
        return info

    for i, (bbox_str, bbox) in enumerate(missing, start=1):
        out_dir = bbox_regions_dir / bbox_str
        out_dir.mkdir(parents=True, exist_ok=True)
        img = _read_wsi_level0_crop(
            wsi_path=wsi_path,
            bbox=bbox,
            max_dim=max_dim,
            wsi_reader=wsi_reader,
        )
        out_png = out_dir / "bbox_region.png"
        img.save(out_png)
        print(f"  Stage 1 bbox regions [{i}/{len(missing)}]: {out_png}")

    info["generated"] = len(missing)
    return info


def _next_stage2_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = now_ts()
    candidate = base_dir / stem
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_skipstage2_{suffix}"
        suffix += 1
    return candidate


def materialize_skip_stage2_native_run(
    *,
    native_root: Path,
    bboxes_root: Path,
    wsi_id: str,
    wsi_path: str,
    stage2_model: str,
    stage2_max_dim: int,
    stage2_wsi_reader: str,
    stage2_force_read_l0: bool,
    stage1_dir: Path,
    bboxes: List[Tuple[str, List[int]]],
    prefer_existing_stage2: bool,
) -> Path:
    """Create a synthetic native Stage2 run for Stage4 compatibility."""
    model_dir = sanitize_stage2_model(stage2_model)
    stage2_root = native_root / "stage2_output" / wsi_id / model_dir
    run_dir = _next_stage2_run_dir(stage2_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "wsi_path": wsi_path,
        "model": stage2_model,
        "stage2_mode": "synthetic_skip_stage2",
        "force_read_l0": bool(stage2_force_read_l0),
        "wsi_reader": stage2_wsi_reader,
        "prefer_existing_stage2": bool(prefer_existing_stage2),
        "artifact_verdict_policy": "exclude_all",
        "created_at": datetime.now().isoformat(),
    }
    write_json(run_dir / "metadata.json", run_meta)

    for bbox_str, bbox in bboxes:
        dst_bbox = run_dir / bbox_str
        src_stage2 = bboxes_root / bbox_str / "stage2"
        if prefer_existing_stage2 and stage_done(src_stage2, STAGE2_BBOX_REQUIRED_FILES):
            sync_tree(src_stage2, dst_bbox)
            continue

        dst_bbox.mkdir(parents=True, exist_ok=True)
        if stage2_force_read_l0:
            bbox_img = _read_wsi_level0_crop(
                wsi_path=wsi_path,
                bbox=bbox,
                max_dim=stage2_max_dim,
                wsi_reader=stage2_wsi_reader,
            )
            bbox_source = "wsi_level0"
        else:
            bbox_img = _read_stage1_thumbnail_crop(stage1_dir, bbox, stage2_max_dim)
            bbox_source = "stage1_thumbnail"
        bbox_img.save(dst_bbox / "bbox_region.png")

        stage1_perception = {
            rot: "SKIPPED stage2 artifact perception; synthetic EXCLUDE-all verdicts used."
            for rot in ROTATION_KEYS
        }
        write_json(dst_bbox / "stage1_artifact_perception.json", stage1_perception)
        write_json(dst_bbox / "stage2_claim_evidence.json", {})
        write_json(dst_bbox / "stage3_strength.json", build_exclude_all_strength())
        verdicts = build_exclude_all_verdicts()
        write_json(dst_bbox / "stage4_verdicts.json", verdicts)
        write_json(dst_bbox / "verdicts.json", verdicts)
        write_json(
            dst_bbox / "metadata.json",
            {
                "wsi_path": wsi_path,
                "bbox_level0": [int(v) for v in bbox],
                "stage2_mode": "synthetic_skip_stage2",
                "bbox_region_source": bbox_source,
                "force_read_l0": bool(stage2_force_read_l0),
                "wsi_reader": stage2_wsi_reader,
                "artifact_verdict_policy": "exclude_all",
                "created_at": datetime.now().isoformat(),
            },
        )

        ensure_files(dst_bbox, STAGE2_BBOX_REQUIRED_FILES, "Synthetic Stage 2 bbox")
    return run_dir


def find_stage3_native_run(native_root: Path, wsi_id: str, stage3_model_dir: str) -> Optional[Path]:
    return latest_subdir(native_root / "stage3_output" / wsi_id / stage3_model_dir)


def find_stage4_native_dir(
    native_root: Path, wsi_id: str, bbox_str: str, model: str
) -> Optional[Path]:
    return latest_subdir(
        native_root / "stage4_output" / wsi_id / bbox_str / sanitize_model_path(model)
    )


def find_stage6_native_dir(stage6_tmp_root: Path) -> Optional[Path]:
    if not stage6_tmp_root.exists():
        return None
    candidates: List[Path] = []
    seen: set = set()

    # Current layout: {stage6_tmp_root}/{wsi}/{model}/{timestamp_hash}/
    # Legacy layout:  {stage6_tmp_root}/attempt_*/{wsi}/{model}/{timestamp_hash}/
    for pattern in ("*/*/*", "attempt_*/*/*/*"):
        for p in stage6_tmp_root.glob(pattern):
            if not p.is_dir() or not (p / "metadata.json").exists():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(p)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def list_case_run_dirs(output_root: Path, wsi_id: str) -> List[Path]:
    case_root = output_root / wsi_id
    if not case_root.exists():
        return []
    run_dirs: List[Path] = []
    for child in case_root.iterdir():
        if not child.is_dir():
            continue
        # Only include dirs that look like pipeline run roots.
        if (child / "stage1").exists() or (child / "pipeline_status.json").exists():
            run_dirs.append(child)
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs


def is_run_complete(run_dir: Path) -> Tuple[bool, str]:
    stage1_dir = run_dir / "stage1"
    if not stage_done(stage1_dir, STAGE1_MIN_REQUIRED_FILES):
        return False, "stage1_missing"

    try:
        bboxes = load_bboxes(stage1_dir)
    except Exception as exc:  # pragma: no cover - defensive against malformed outputs
        return False, f"stage1_bboxes_unreadable: {exc}"

    if not bboxes:
        return False, "stage1_no_bboxes"

    bboxes_root = run_dir / "bboxes"
    for bbox_str, _ in bboxes:
        bbox_dir = bboxes_root / bbox_str
        checks = (
            ("stage2", STAGE2_BBOX_REQUIRED_FILES),
            ("stage3", STAGE3_BBOX_REQUIRED_FILES),
            ("stage4", STAGE4_BBOX_REQUIRED_FILES),
            ("stage5", STAGE5_BBOX_REQUIRED_FILES),
            ("stage6", STAGE6_BBOX_REQUIRED_FILES),
            ("stage7", STAGE7_BBOX_REQUIRED_FILES),
        )
        for stage_name, required_files in checks:
            if not stage_done(bbox_dir / stage_name, required_files):
                return False, f"{bbox_str}:{stage_name}_missing"

    return True, "complete"


def build_resume_plan(output_root: Path, wsi_id: str) -> ResumePlan:
    run_dirs = list_case_run_dirs(output_root, wsi_id)
    if not run_dirs:
        return ResumePlan(action="new", detail="no_prior_runs")

    latest = run_dirs[-1]
    complete, detail = is_run_complete(latest)
    if complete:
        return ResumePlan(action="skip", run_id=latest.name, detail=detail)
    return ResumePlan(action="resume", run_id=latest.name, detail=detail)


def build_stage1_command(args: argparse.Namespace, wsi_path: str, wsi_reader: str) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "detect_foreground_regions_from_wsi_thumbnail.py"),
        "--wsi",
        wsi_path,
        "--wsi-reader",
        wsi_reader,
        "--backend",
        args.stage1_backend,
        "--model",
        args.stage1_model,
        "--openrouter-url",
        args.stage1_openrouter_url,
        "--vllm-url",
        args.stage1_vllm_url,
        "--coord-order",
        args.stage1_coord_order,
        "--padding",
        str(args.stage1_padding),
        "--merge-overlap-threshold",
        str(args.stage1_merge_overlap_threshold),
        "--rotations",
        *[str(v) for v in args.stage1_rotations],
        "--vertex-location",
        args.stage1_vertex_location,
    ]
    if args.stage1_vertex_credentials:
        cmd.extend(["--vertex-credentials", str(args.stage1_vertex_credentials)])
    if args.stage1_api_key:
        cmd.extend(["--api-key", args.stage1_api_key])
    if args.stage1_repair_model:
        cmd.extend(["--repair-model", args.stage1_repair_model])
    if args.stage1_save_intermediate:
        cmd.append("--save-intermediate")
    if args.stage1_save_bbox_region:
        cmd.append("--save-bbox-region")
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage1_xml_command(
    args: argparse.Namespace,
    wsi_path: str,
    xml_path: str,
    stage1_dir: Path,
    wsi_reader: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "materialize_stage1_from_xml.py"),
        "--wsi",
        wsi_path,
        "--wsi-reader",
        wsi_reader,
        "--xml",
        xml_path,
        "--output-dir",
        str(stage1_dir),
        "--max-dim",
        str(args.stage1_xml_max_dim),
        "--xml-group",
        args.stage1_xml_group,
        "--model-tag",
        args.stage1_xml_model_tag,
    ]
    if args.stage1_xml_include_non_rect:
        cmd.append("--include-non-rect")
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage2_command(
    args: argparse.Namespace, stage1_dir: Path, native_root: Path, wsi_reader: str
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_artifact_qc_pipeline.py"),
        "--stage1-dir",
        str(stage1_dir),
        "--wsi-reader",
        wsi_reader,
        "--backend",
        args.stage2_backend,
        "--model",
        args.stage2_model,
        "--openrouter-url",
        args.stage2_openrouter_url,
        "--vllm-url",
        args.stage2_vllm_url,
        "--vertex-location",
        args.stage2_vertex_location,
        "--output-base",
        str(native_root / "stage2_output"),
        "--max-dim",
        str(args.stage2_max_dim),
    ]
    if args.stage2_vertex_credentials:
        cmd.extend(["--vertex-credentials", str(args.stage2_vertex_credentials)])
    if args.stage2_force_read_l0:
        cmd.append("--force-read-l0")
    if args.stage2_api_key:
        cmd.extend(["--api-key", args.stage2_api_key])
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage3_command(
    args: argparse.Namespace, stage1_dir: Path, native_root: Path
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_color_segmentation.py"),
        str(stage1_dir),
        "--method",
        args.stage3_method,
        "--output-base",
        str(native_root / "stage3_output"),
    ]
    if args.stage3_method == "kmeans":
        cmd.extend(["--k", str(args.stage3_k)])
    else:
        cmd.extend(["--min-cluster-size", str(args.stage3_min_cluster_size)])
    if args.stage3_blur > 0:
        cmd.extend(["--blur", str(args.stage3_blur)])
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage4_command(
    args: argparse.Namespace, stage2_bbox_dir_native: Path, native_root: Path
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "find_icl_regions.py"),
        "--stage2-dir",
        str(stage2_bbox_dir_native),
        "--backend",
        args.stage4_backend,
        "--models",
        args.stage4_model,
        "--openrouter-url",
        args.stage4_openrouter_url,
        "--vllm-url",
        args.stage4_vllm_url,
        "--vertex-location",
        args.stage4_vertex_location,
        "--point-order",
        args.stage4_point_order,
        "--point-key",
        args.stage4_point_key,
        "--output-base",
        str(native_root / "stage4_output"),
        "--max-items",
        str(args.stage4_max_items),
        "--max-tokens",
        str(args.stage4_max_tokens),
    ]
    if args.stage4_vertex_credentials:
        cmd.extend(["--vertex-credentials", str(args.stage4_vertex_credentials)])
    if args.stage4_api_key:
        cmd.extend(["--api-key", args.stage4_api_key])
    if args.stage4_repair_model:
        cmd.extend(["--repair-model", args.stage4_repair_model])
    if args.stage4_thinking_level:
        cmd.extend(["--thinking-level", args.stage4_thinking_level])
    if args.stage4_include_thoughts:
        cmd.append("--include-thoughts")
    if args.stage4_use_visual_descriptions:
        cmd.append("--use-visual-descriptions")
    if args.stage4_no_tta:
        cmd.append("--no-tta")
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage5_command(
    args: argparse.Namespace,
    stage4_dir: Path,
    stage5_dir: Path,
    wsi_reader: str,
) -> List[str]:
    stage5_backend = (args.stage5_vlm_backend or "openrouter").lower()
    force_stage5_vertex = stage5_backend == "vertex"
    if force_stage5_vertex:
        stage5_backend = "gemini"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "reranker.py"),
        "--stage4-dir",
        str(stage4_dir),
        "--output-dir",
        str(stage5_dir),
        "--wsi-reader",
        wsi_reader,
        "--top-k",
        str(args.stage5_top_k),
        "--k",
        str(args.stage5_k),
        "--patch-size",
        str(args.stage5_patch_size),
        "--vlm-backend",
        stage5_backend,
        "--vlm-model",
        args.stage5_vlm_model,
        "--vlm-port",
        str(args.stage5_vlm_port),
        "--vlm-max-tokens",
        str(args.stage5_vlm_max_tokens),
        "--selection-mode",
        args.stage5_selection_mode,
        "--max-total-candidates",
        str(args.stage5_max_total_candidates),
    ]
    if args.stage5_vlm_image_size is not None:
        cmd.extend(["--vlm-image-size", str(args.stage5_vlm_image_size)])
    if stage5_backend == "openrouter" and args.stage5_openrouter_reasoning_effort:
        cmd.extend(["--reasoning-effort", args.stage5_openrouter_reasoning_effort])
    if stage5_backend == "gemini":
        use_vertex = True if force_stage5_vertex else bool(args.stage5_gemini_use_vertex)
        cmd.append("--gemini-use-vertex" if use_vertex else "--gemini-no-vertex")
        if args.stage5_gemini_credentials:
            cmd.extend(["--gemini-credentials", str(args.stage5_gemini_credentials)])
        if args.stage5_gemini_location:
            cmd.extend(["--gemini-location", args.stage5_gemini_location])
        if args.stage5_gemini_thinking_level:
            cmd.extend(["--gemini-thinking-level", args.stage5_gemini_thinking_level])
        if args.stage5_gemini_include_thoughts:
            cmd.append("--gemini-include-thoughts")
    if args.stage5_max_candidates_per_class is not None:
        cmd.extend([
            "--max-candidates-per-class",
            str(args.stage5_max_candidates_per_class),
        ])
    if args.stage5_tournament_round1_k is not None:
        cmd.extend([
            "--tournament-round1-k",
            str(args.stage5_tournament_round1_k),
        ])
    if args.stage5_disable_tissue_blur_filter:
        cmd.append("--disable-tissue-blur-filter")
    cmd.extend([
        "--tissue-blur-threshold",
        str(args.stage5_tissue_blur_threshold),
        "--tissue-blur-sigma",
        str(args.stage5_tissue_blur_sigma),
        "--tissue-blur-pixel-threshold",
        str(args.stage5_tissue_blur_pixel_threshold),
    ])
    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage5_descriptions_command(args: argparse.Namespace, stage5_dir: Path) -> List[str]:
    stage5_backend = (args.stage5_vlm_backend or "openrouter").lower()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "generate_stage5_descriptions.py"),
        "--stage5-dir",
        str(stage5_dir),
        "--output-name",
        "class_descriptions.json",
        "--backend",
        stage5_backend,
        "--model",
        args.stage5_vlm_model,
        "--max-tokens",
        str(args.stage5_vlm_max_tokens),
    ]
    if stage5_backend == "vllm":
        cmd.extend(["--vllm-url", f"http://localhost:{args.stage5_vlm_port}/v1"])
    if stage5_backend in {"gemini", "vertex"}:
        if stage5_backend == "vertex" or args.stage5_gemini_use_vertex:
            cmd.append("--gemini-use-vertex")
        else:
            cmd.append("--gemini-no-vertex")
        if args.stage5_gemini_credentials:
            cmd.extend(["--gemini-credentials", str(args.stage5_gemini_credentials)])
        if args.stage5_gemini_location:
            cmd.extend(["--gemini-location", args.stage5_gemini_location])
    if args.stage5_descriptions_force:
        cmd.append("--force")
    return cmd


def build_stage6_command(
    args: argparse.Namespace,
    stage5_dir: Path,
    stage3_dir: Optional[Path],
    stage6_tmp_root: Path,
    wsi_reader: str,
    class_defs_path: Optional[Path] = None,
    icl_k_override: Optional[int] = None,
) -> List[str]:
    icl_k_effective = args.stage6_icl_k if icl_k_override is None else int(icl_k_override)
    stage6_backend = (args.stage6_backend or "vllm").lower()
    force_stage6_vertex = stage6_backend == "vertex"
    if force_stage6_vertex:
        stage6_backend = "gemini"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_vlm_bbox_inference.py"),
        "--stage5-run",
        str(stage5_dir),
        "--output-dir",
        str(stage6_tmp_root),
        "--wsi-reader",
        wsi_reader,
        "--backend",
        stage6_backend,
        "--icl-k",
        str(icl_k_effective),
        "--icl-shuffle-n",
        str(args.stage6_icl_shuffle_n),
        "--rotations",
        args.stage6_rotations,
        "--query-batch-size",
        str(args.stage6_query_batch_size),
        "--max-workers",
        str(args.stage6_max_workers),
        "--timeout",
        str(args.stage6_timeout),
        "--temperature",
        str(args.stage6_temperature),
        "--max-tokens",
        str(args.stage6_max_tokens),
        "--max-retries",
        str(args.stage6_max_retries),
        "--label-mode",
        args.stage6_label_mode,
    ]
    if args.stage6_model:
        cmd.extend(["--model", args.stage6_model])
    if args.stage6_prompt_template:
        cmd.extend(["--prompt-template", str(args.stage6_prompt_template)])
    if class_defs_path:
        cmd.extend(["--class-defs", str(class_defs_path)])
    elif args.stage6_class_defs:
        cmd.extend(["--class-defs", str(args.stage6_class_defs)])
    if args.stage6_vlm_image_size:
        cmd.extend(["--vlm-image-size", str(args.stage6_vlm_image_size)])
    if args.stage6_patch_size:
        cmd.extend(["--patch-size", str(args.stage6_patch_size)])

    if not args.no_stage3_gating and stage3_dir is not None:
        cmd.extend(["--stage3-run", str(stage3_dir)])
        cmd.extend(["--stage3-fg-threshold", str(args.stage6_stage3_fg_threshold)])

    if args.resume:
        cmd.append("--resume")

    if stage6_backend == "vllm" and args.stage6_vllm_url:
        cmd.extend(["--vllm-url", args.stage6_vllm_url])
    if stage6_backend == "openrouter":
        if args.stage6_openrouter_url:
            cmd.extend(["--openrouter-url", args.stage6_openrouter_url])
        if args.stage6_openrouter_api_key:
            cmd.extend(["--openrouter-api-key", args.stage6_openrouter_api_key])
        if args.stage6_openrouter_referer:
            cmd.extend(["--openrouter-referer", args.stage6_openrouter_referer])
    if stage6_backend == "gemini":
        use_vertex = True if force_stage6_vertex else bool(args.stage6_gemini_use_vertex)
        if use_vertex:
            cmd.append("--gemini-use-vertex")
        else:
            cmd.append("--gemini-no-vertex")
        if args.stage6_gemini_credentials:
            cmd.extend(["--gemini-credentials", str(args.stage6_gemini_credentials)])
        if args.stage6_gemini_location:
            cmd.extend(["--gemini-location", args.stage6_gemini_location])

    if args.skip_dvc_check:
        cmd.append("--skip-dvc-check")
    return cmd


def build_stage7_command(
    args: argparse.Namespace,
    stage6_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "postprocess_mask.py"),
        str(stage6_dir),
        "--selection", "all",
        "--min-component-size", str(args.stage7_min_component_size),
        "--connectivity", str(args.stage7_connectivity),
        "--close-kernel", str(args.stage7_close_kernel),
    ]
    if args.stage7_skip_remove_small:
        cmd.append("--skip-remove-small")
    if args.stage7_skip_close:
        cmd.append("--skip-close")
    if args.stage7_skip_fill_holes:
        cmd.append("--skip-fill-holes")
    if args.stage7_allow_artifact_overwrite:
        cmd.append("--allow-artifact-overwrite")
    return cmd


def _read_patch_size_from_stage6(bboxes_root: Path, bboxes: List[Tuple[str, List[int]]]) -> int:
    """Read patch_size_level0 from the first bbox's stage6 metadata."""
    for bbox_str, _ in bboxes:
        meta_path = bboxes_root / bbox_str / "stage6" / "metadata.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)
            ps = meta.get("patch_size_level0") or meta.get("patch_size")
            if ps is not None:
                return int(ps)
    raise PipelineError("Could not read patch_size_level0 from any bbox's stage6 metadata")


def assemble_wsi_mask(
    stage1_dir: Path,
    bboxes: List[Tuple[str, List[int]]],
    bboxes_root: Path,
    stage7_wsi_dir: Path,
) -> None:
    """Assemble per-bbox tissue masks into a single WSI-level compact mask.

    Covers the full WSI extent. Shape: (ceil(wsi_h/patch_size), ceil(wsi_w/patch_size)).
    Each pixel = patch_size x patch_size at L0. Values: 0=background, 1=tissue.
    patch_size is inherited from stage6 metadata (patch_size_level0 field).
    """
    patch_size = _read_patch_size_from_stage6(bboxes_root, bboxes)

    with (stage1_dir / "metadata.json").open("r") as f:
        stage1_meta = json.load(f)
    wsi_dims = stage1_meta["wsi_dimensions"]
    wsi_w, wsi_h = int(wsi_dims["width"]), int(wsi_dims["height"])

    mask_rows = math.ceil(wsi_h / patch_size)
    mask_cols = math.ceil(wsi_w / patch_size)
    wsi_mask = np.zeros((mask_rows, mask_cols), dtype=np.uint8)

    for bbox_str, bbox in bboxes:
        stage7_dir = bboxes_root / bbox_str / "stage7"
        tissue_mask_path = stage7_dir / "tissue_mask_post.npy"
        if not tissue_mask_path.exists():
            continue
        tissue_mask = np.load(tissue_mask_path).astype(bool)

        x1, y1, x2, y2 = bbox
        row_start = y1 // patch_size
        col_start = x1 // patch_size
        t_rows, t_cols = tissue_mask.shape

        r_end = min(row_start + t_rows, mask_rows)
        c_end = min(col_start + t_cols, mask_cols)
        t_r_end = r_end - row_start
        t_c_end = c_end - col_start

        wsi_mask[row_start:r_end, col_start:c_end] = np.maximum(
            wsi_mask[row_start:r_end, col_start:c_end],
            tissue_mask[:t_r_end, :t_c_end].astype(np.uint8),
        )

    stage7_wsi_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage7_wsi_dir / "mask.npy", wsi_mask)

    meta = {
        "description": "WSI-level binary tissue mask. 1px = patch_size x patch_size at L0.",
        "wsi_dimensions_level0": {"width": wsi_w, "height": wsi_h},
        "patch_size_level0": patch_size,
        "mask_shape": {"rows": mask_rows, "cols": mask_cols},
        "origin_level0": {"x": 0, "y": 0},
        "value_map": {"0": "background", "1": "tissue"},
        "dtype": "uint8",
        "coordinate_formula": {
            "mask_to_level0": "L0_x = col * patch_size, L0_y = row * patch_size",
            "level0_to_mask": "col = L0_x // patch_size, row = L0_y // patch_size",
        },
        "bbox_placements": {
            bbox_str: {
                "bbox_level0": bbox,
                "row_start": bbox[1] // patch_size,
                "col_start": bbox[0] // patch_size,
            }
            for bbox_str, bbox in bboxes
        },
    }
    write_json(stage7_wsi_dir / "metadata.json", meta)

    # Render overlay on stage1 thumbnail using bbox-origin placement.
    # Avoid global grid snapping so non-aligned bbox origins stay registered.
    thumb_path = stage1_dir / "thumbnail.png"
    if thumb_path.exists():
        thumb = Image.open(thumb_path).convert("RGB")
        gray = ImageOps.grayscale(thumb).convert("RGB")
        base = gray.convert("RGBA")
        tw, th = thumb.size
        overlay_arr = np.zeros((th, tw, 4), dtype=np.uint8)

        scale_x = tw / wsi_w
        scale_y = th / wsi_h

        for bbox_str, bbox in bboxes:
            stage7_dir = bboxes_root / bbox_str / "stage7"
            tissue_mask_path = stage7_dir / "tissue_mask_post.npy"
            if not tissue_mask_path.exists():
                continue
            tissue_mask = np.load(tissue_mask_path).astype(bool)

            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(v) for v in bbox]
            t_rows, t_cols = tissue_mask.shape

            for r in range(t_rows):
                for c in range(t_cols):
                    if not tissue_mask[r, c]:
                        continue

                    wsi_x1 = bbox_x1 + c * patch_size
                    wsi_y1 = bbox_y1 + r * patch_size
                    if wsi_x1 >= bbox_x2 or wsi_y1 >= bbox_y2:
                        continue

                    # Clip edge cells to bbox bounds so partial patches are drawn correctly.
                    wsi_x2 = min(wsi_x1 + patch_size, bbox_x2, wsi_w)
                    wsi_y2 = min(wsi_y1 + patch_size, bbox_y2, wsi_h)

                    px1 = max(0, int(wsi_x1 * scale_x))
                    py1 = max(0, int(wsi_y1 * scale_y))
                    px2 = min(tw, int(wsi_x2 * scale_x))
                    py2 = min(th, int(wsi_y2 * scale_y))
                    if px2 > px1 and py2 > py1:
                        overlay_arr[py1:py2, px1:px2] = [255, 140, 0, 140]

        out = Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA"))
        out.convert("RGB").save(stage7_wsi_dir / "mask_overlay.png")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def run_single_wsi(
    wsi_input: str,
    args: argparse.Namespace,
    repro_preflight: Optional[Dict[str, object]] = None,
    run_id_override: Optional[str] = None,
    resolved_wsi_path: Optional[str] = None,
    stage1_xml_input: Optional[str] = None,
    source_worklist: Optional[str] = None,
    source_worklist_path: Optional[str] = None,
) -> dict:
    started_at = datetime.now().isoformat()

    wsi_path = resolved_wsi_path or resolve_wsi_path(wsi_input)
    wsi_id = Path(wsi_path).stem
    run_id = run_id_override or args.run_id or now_ts()

    run_dir = Path(args.output_root) / wsi_id / run_id
    native_root = run_dir / "_native"
    stage1_dir = run_dir / "stage1"
    bboxes_root = run_dir / "bboxes"
    logs_dir = run_dir / "logs"

    run_dir.mkdir(parents=True, exist_ok=True)
    native_root.mkdir(parents=True, exist_ok=True)
    bboxes_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stage_subprocess_env = {
        SKIP_STAGE_REPRO_CHECK_ENV: "1",
        "PYTHONUNBUFFERED": "1",
    }
    reader_info = resolve_case_wsi_readers(args, wsi_path)
    case_requested_wsi_readers = reader_info["requested"]
    case_effective_wsi_readers = reader_info["effective"]
    wsi_is_ndpi = bool(reader_info["is_ndpi"])
    forced_to_openslide = reader_info["forced_to_openslide"]

    status = {
        "ok": False,
        "wsi_input": wsi_input,
        "stage1_xml_input": stage1_xml_input,
        "source_worklist": source_worklist,
        "source_worklist_path": source_worklist_path,
        "wsi_path": wsi_path,
        "wsi_id": wsi_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "wsi_is_ndpi": wsi_is_ndpi,
        "wsi_readers": {
            "requested": case_requested_wsi_readers,
            "effective": case_effective_wsi_readers,
            "forced_to_openslide_stages": forced_to_openslide,
        },
        "started_at": started_at,
        "finished_at": None,
        "error": None,
        "bbox_status": {},
    }
    metadata = {
        "wsi_input": wsi_input,
        "stage1_xml_input": stage1_xml_input,
        "source_worklist": source_worklist,
        "source_worklist_path": source_worklist_path,
        "wsi_path": wsi_path,
        "wsi_id": wsi_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "native_root": str(native_root),
        "wsi_is_ndpi": wsi_is_ndpi,
        "wsi_readers": {
            "requested": case_requested_wsi_readers,
            "effective": case_effective_wsi_readers,
            "forced_to_openslide_stages": forced_to_openslide,
        },
        "config": vars(args).copy(),
        "repro_preflight": repro_preflight or {"enabled": False},
        "stage_runs": {},
        "bboxes": [],
        "started_at": started_at,
    }

    run_lock = None
    try:
        run_lock = acquire_run_lock(run_dir, wsi_id, run_id)
        print(f"\n=== WSI: {wsi_id} ===")
        print(f"Run dir: {run_dir}")
        if source_worklist:
            if source_worklist_path:
                print(f"Source worklist: {source_worklist} ({source_worklist_path})")
            else:
                print(f"Source worklist: {source_worklist}")
        if wsi_is_ndpi:
            print("WSI type: .ndpi detected")
            if forced_to_openslide:
                print(
                    "WSI readers: forcing openslide for NDPI on stages "
                    + ",".join(forced_to_openslide)
                )
        print(
            "WSI readers: "
            f"s1={case_effective_wsi_readers['stage1']} "
            f"s2={case_effective_wsi_readers['stage2']} "
            f"s5={case_effective_wsi_readers['stage5']} "
            f"s6={case_effective_wsi_readers['stage6']}"
        )
        metadata["stage_runs"]["wsi_readers"] = {
            "requested": case_requested_wsi_readers,
            "effective": case_effective_wsi_readers,
        }
        max_stage = int(args.max_stage)
        metadata["stage_runs"]["max_stage"] = max_stage
        if max_stage < 7:
            print(f"Stage cap: running through Stage {max_stage}")

        skip_stage45_requested = should_skip_stage45_for_no_icl_baseline(args)
        skip_stage45_no_icl_baseline = bool(skip_stage45_requested and max_stage >= 6)
        metadata["stage_runs"]["skip_stage45_no_icl_baseline"] = bool(
            skip_stage45_no_icl_baseline
        )
        if skip_stage45_no_icl_baseline:
            print(
                "Pipeline mode: Stage 6 no-ICL baseline detected "
                "(--stage6-icl-k 0, no Stage 5 descriptions); skipping Stage 4/5."
            )
        elif skip_stage45_requested:
            print(
                "Pipeline mode: no-ICL baseline requested, but --max-stage < 6; "
                "Stage 4/5 fallback shortcut disabled."
            )

        # ------------------------------------------------------------------
        # Stage 1
        # ------------------------------------------------------------------
        if args.resume and stage_done(stage1_dir, STAGE1_MIN_REQUIRED_FILES):
            print("Stage 1: resume hit, skipping")
        else:
            if stage1_xml_input:
                print("Stage 1: materializing from XML ROI")
                cmd = build_stage1_xml_command(
                    args,
                    wsi_path,
                    stage1_xml_input,
                    stage1_dir,
                    case_effective_wsi_readers["stage1"],
                )
                run_command(
                    CommandSpec(
                        cmd=cmd,
                        cwd=native_root,
                        log_path=logs_dir / "stage1.log",
                        extra_env=stage_subprocess_env,
                    ),
                    dry_run=args.dry_run,
                    live_logs=args.live_logs,
                )
                metadata["stage_runs"]["stage1_xml"] = {
                    "xml_path": stage1_xml_input,
                    "xml_group": args.stage1_xml_group,
                    "include_non_rect": bool(args.stage1_xml_include_non_rect),
                    "wsi_reader": case_effective_wsi_readers["stage1"],
                }
            else:
                print("Stage 1: running")
                cmd = build_stage1_command(
                    args,
                    wsi_path,
                    case_effective_wsi_readers["stage1"],
                )
                run_command(
                    CommandSpec(
                        cmd=cmd,
                        cwd=native_root,
                        log_path=logs_dir / "stage1.log",
                        extra_env=stage_subprocess_env,
                    ),
                    dry_run=args.dry_run,
                    live_logs=args.live_logs,
                )
                stage1_native = find_stage1_native_dir(native_root, wsi_id, args.stage1_model)
                if stage1_native is None:
                    raise PipelineError("Could not locate Stage 1 native output directory")
                sync_tree(stage1_native, stage1_dir)
                metadata["stage_runs"]["stage1_native"] = str(stage1_native)

        ensure_files(stage1_dir, STAGE1_MIN_REQUIRED_FILES, "Stage 1")
        stage1_fallback_info = ensure_stage1_bboxes_with_fullslide_fallback(stage1_dir)
        metadata["stage_runs"]["stage1_fallback"] = stage1_fallback_info
        if bool(stage1_fallback_info.get("applied")):
            print(
                "Stage 1: no valid bbox output from detector, "
                "synthesized full-slide fallback bbox"
            )
        ensure_files(stage1_dir, STAGE1_REQUIRED_FILES, "Stage 1")
        bboxes = load_bboxes(stage1_dir)
        metadata["bboxes"] = [b for _, b in bboxes]
        if args.stage1_save_bbox_region:
            stage1_bbox_region_max_dim = _resolve_stage1_bbox_region_max_dim(stage1_dir)
            stage1_bbox_regions = ensure_stage1_bbox_region_exports(
                stage1_dir=stage1_dir,
                wsi_path=wsi_path,
                bboxes=bboxes,
                wsi_reader=case_effective_wsi_readers["stage1"],
                max_dim=stage1_bbox_region_max_dim,
                dry_run=args.dry_run,
            )
            metadata["stage_runs"]["stage1_bbox_regions"] = stage1_bbox_regions
        print(f"Stage 1: detected {len(bboxes)} bbox(es)")

        if not bboxes:
            raise PipelineError("Stage 1 fallback failed to produce any bbox output")

        for bbox_str, _ in bboxes:
            (bboxes_root / bbox_str).mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Stage 2 + Stage 3
        # ------------------------------------------------------------------
        stage2_all_done = all(
            stage_done(
                bboxes_root / bbox_str / "stage2",
                STAGE2_BBOX_REQUIRED_FILES,
            )
            for bbox_str, _ in bboxes
        )
        stage3_all_done = all(
            stage_done(
                bboxes_root / bbox_str / "stage3",
                STAGE3_BBOX_REQUIRED_FILES,
            )
            for bbox_str, _ in bboxes
        )

        need_stage2 = (not (args.resume and stage2_all_done)) and max_stage >= 2
        need_stage3 = (not (args.resume and stage3_all_done)) and max_stage >= 3
        stage2_native_run: Optional[Path] = None
        stage2_synthetic_created = False

        if need_stage2 and args.skip_stage2:
            skip_stage2_bbox_source = (
                "WSI level-0 reads" if args.stage2_force_read_l0 else "Stage 1 thumbnail crops"
            )
            if args.dry_run:
                print(
                    "Stage 2: skip enabled (dry-run), would synthesize EXCLUDE-all outputs "
                    f"using {skip_stage2_bbox_source}"
                )
                (logs_dir / "stage2.log").write_text(
                    "[DRY RUN] --skip-stage2 enabled; would synthesize Stage 2 outputs "
                    "with EXCLUDE-all artifact verdicts.\n",
                    encoding="utf-8",
                )
            else:
                print(
                    "Stage 2: skip enabled, synthesizing EXCLUDE-all outputs "
                    f"using {skip_stage2_bbox_source}"
                )
                stage2_native_run = materialize_skip_stage2_native_run(
                    native_root=native_root,
                    bboxes_root=bboxes_root,
                    wsi_id=wsi_id,
                    wsi_path=wsi_path,
                    stage2_model=args.stage2_model,
                    stage2_max_dim=args.stage2_max_dim,
                    stage2_wsi_reader=case_effective_wsi_readers["stage2"],
                    stage2_force_read_l0=args.stage2_force_read_l0,
                    stage1_dir=stage1_dir,
                    bboxes=bboxes,
                    prefer_existing_stage2=False,
                )
                metadata["stage_runs"]["stage2_native"] = str(stage2_native_run)
                metadata["stage_runs"]["stage2_mode"] = "synthetic_skip_stage2_exclude_all"
                for bbox_str, _ in bboxes:
                    src = stage2_native_run / bbox_str
                    dst = bboxes_root / bbox_str / "stage2"
                    if not src.exists():
                        raise PipelineError(f"Synthetic Stage 2 output missing bbox dir: {src}")
                    sync_tree(src, dst)
                    ensure_files(dst, STAGE2_BBOX_REQUIRED_FILES, "Synthetic Stage 2 bbox")
                (logs_dir / "stage2.log").write_text(
                    "--skip-stage2 enabled: synthesized Stage 2 outputs with EXCLUDE-all "
                    "artifact verdicts for all bboxes.\n",
                    encoding="utf-8",
                )
            stage2_synthetic_created = True
            need_stage2 = False

        if need_stage2 and need_stage3:
            print("Stages 2+3: running in parallel")
            s2 = CommandSpec(
                cmd=build_stage2_command(
                    args,
                    stage1_dir,
                    native_root,
                    case_effective_wsi_readers["stage2"],
                ),
                cwd=REPO_ROOT,
                log_path=logs_dir / "stage2.log",
                extra_env=stage_subprocess_env,
            )
            s3 = CommandSpec(
                cmd=build_stage3_command(args, stage1_dir, native_root),
                cwd=REPO_ROOT,
                log_path=logs_dir / "stage3.log",
                extra_env=stage_subprocess_env,
            )
            if args.parallel_stage23:
                run_parallel(s2, s3, dry_run=args.dry_run, live_logs=args.live_logs)
            else:
                run_command(s2, dry_run=args.dry_run, live_logs=args.live_logs)
                run_command(s3, dry_run=args.dry_run, live_logs=args.live_logs)
        else:
            if need_stage2:
                print("Stage 2: running")
                run_command(
                    CommandSpec(
                        cmd=build_stage2_command(
                            args,
                            stage1_dir,
                            native_root,
                            case_effective_wsi_readers["stage2"],
                        ),
                        cwd=REPO_ROOT,
                        log_path=logs_dir / "stage2.log",
                        extra_env=stage_subprocess_env,
                    ),
                    dry_run=args.dry_run,
                    live_logs=args.live_logs,
                )
            else:
                if stage2_synthetic_created:
                    print("Stage 2: synthetic outputs ready, skipping command")
                elif max_stage < 2:
                    print("Stage 2: capped by --max-stage, skipping")
                else:
                    print("Stage 2: resume hit, skipping")
            if need_stage3:
                print("Stage 3: running")
                run_command(
                    CommandSpec(
                        cmd=build_stage3_command(args, stage1_dir, native_root),
                        cwd=REPO_ROOT,
                        log_path=logs_dir / "stage3.log",
                        extra_env=stage_subprocess_env,
                    ),
                    dry_run=args.dry_run,
                    live_logs=args.live_logs,
                )
            else:
                if max_stage < 3:
                    print("Stage 3: capped by --max-stage, skipping")
                else:
                    print("Stage 3: resume hit, skipping")

        # Locate native runs and sync to canonical tree where needed.
        if stage2_native_run is None:
            stage2_native_run = find_stage2_native_run(native_root, wsi_id, args.stage2_model)
        if need_stage2:
            if stage2_native_run is None:
                raise PipelineError("Could not locate Stage 2 native run directory")
            metadata["stage_runs"]["stage2_native"] = str(stage2_native_run)
            for bbox_str, _ in bboxes:
                src = stage2_native_run / bbox_str
                dst = bboxes_root / bbox_str / "stage2"
                if not src.exists():
                    raise PipelineError(f"Stage 2 output missing bbox dir: {src}")
                sync_tree(src, dst)
                ensure_files(dst, STAGE2_BBOX_REQUIRED_FILES, "Stage 2 bbox")

        stage3_model_dir = stage1_dir.parent.name  # see run_color_segmentation fallback logic
        stage3_native_run = find_stage3_native_run(native_root, wsi_id, stage3_model_dir)
        if need_stage3:
            if stage3_native_run is None:
                raise PipelineError("Could not locate Stage 3 native run directory")
            metadata["stage_runs"]["stage3_native"] = str(stage3_native_run)
            for bbox_str, _ in bboxes:
                src = stage3_native_run / bbox_str
                dst = bboxes_root / bbox_str / "stage3"
                if not src.exists():
                    raise PipelineError(f"Stage 3 output missing bbox dir: {src}")
                sync_tree(src, dst)
                ensure_files(dst, STAGE3_BBOX_REQUIRED_FILES, "Stage 3 bbox")

        # If stage2 was skipped but we still need stage4 later, we need native stage2 dirs.
        if max_stage < 4:
            needs_any_stage4 = False
        elif skip_stage45_no_icl_baseline:
            needs_any_stage4 = False
        else:
            needs_any_stage4 = any(
                not (args.resume and stage_done(bboxes_root / bbox_str / "stage4", ["metadata.json"]))
                for bbox_str, _ in bboxes
            )
        if needs_any_stage4 and stage2_native_run is None and not args.dry_run:
            if args.skip_stage2:
                print(
                    "Stage 2 native run missing; synthesizing native Stage 2 outputs "
                    "for Stage 4 compatibility"
                )
                stage2_native_run = materialize_skip_stage2_native_run(
                    native_root=native_root,
                    bboxes_root=bboxes_root,
                    wsi_id=wsi_id,
                    wsi_path=wsi_path,
                    stage2_model=args.stage2_model,
                    stage2_max_dim=args.stage2_max_dim,
                    stage2_wsi_reader=case_effective_wsi_readers["stage2"],
                    stage2_force_read_l0=args.stage2_force_read_l0,
                    stage1_dir=stage1_dir,
                    bboxes=bboxes,
                    prefer_existing_stage2=True,
                )
                metadata["stage_runs"]["stage2_native"] = str(stage2_native_run)
                metadata["stage_runs"].setdefault(
                    "stage2_mode",
                    "synthetic_skip_stage2_exclude_all",
                )
            else:
                print("Stage 2 native run missing; regenerating stage2 outputs for stage4 compatibility")
                run_command(
                    CommandSpec(
                        cmd=build_stage2_command(
                            args,
                            stage1_dir,
                            native_root,
                            case_effective_wsi_readers["stage2"],
                        ),
                        cwd=REPO_ROOT,
                        log_path=logs_dir / "stage2_regen.log",
                        extra_env=stage_subprocess_env,
                    ),
                    dry_run=False,
                    live_logs=args.live_logs,
                )
                stage2_native_run = find_stage2_native_run(native_root, wsi_id, args.stage2_model)
                if stage2_native_run is None:
                    raise PipelineError("Failed to regenerate Stage 2 native run")

        # ------------------------------------------------------------------
        # Stage 4/5/6 per bbox
        # ------------------------------------------------------------------
        stage2_native_bbox_map: Dict[str, Path] = {}
        if stage2_native_run and stage2_native_run.exists():
            for bbox_str, _ in bboxes:
                p = stage2_native_run / bbox_str
                if p.exists():
                    stage2_native_bbox_map[bbox_str] = p

        bbox_lookup: Dict[str, List[int]] = {bbox_str: bbox for bbox_str, bbox in bboxes}
        bbox_parallel_workers, bbox_parallel_reason = resolve_bbox_parallel_workers(
            args,
            max_stage=max_stage,
            bbox_count=len(bboxes),
            skip_stage45_no_icl_baseline=skip_stage45_no_icl_baseline,
        )
        metadata["stage_runs"]["bbox_parallel_workers_requested"] = int(args.parallelise_bboxes)
        metadata["stage_runs"]["bbox_parallel_workers_effective"] = bbox_parallel_workers
        metadata["stage_runs"]["bbox_parallel_enabled"] = bool(bbox_parallel_workers > 1)
        if bbox_parallel_reason:
            metadata["stage_runs"]["bbox_parallel_disabled_reason"] = bbox_parallel_reason
            print(f"BBox parallelism: {bbox_parallel_reason}")
        elif bbox_parallel_workers > 1:
            print(
                "BBox parallelism: enabled for Stage 4+ "
                f"(workers={bbox_parallel_workers})"
            )

        def process_single_bbox(bbox_str: str, bbox: Sequence[int]) -> Dict[str, object]:
            print(f"\nBBox {bbox_str}")
            bbox_dir = bboxes_root / bbox_str
            stage3_dir = bbox_dir / "stage3"
            stage4_dir = bbox_dir / "stage4"
            stage5_dir = bbox_dir / "stage5"
            stage6_dir = bbox_dir / "stage6"
            stage7_dir = bbox_dir / "stage7"

            bbox_state: Dict[str, object] = {
                "bbox_level0": [int(v) for v in bbox],
                "stage4_done": False,
                "stage5_done": False,
                "stage6_done": False,
                "stage7_done": False,
                "error": None,
            }

            try:
                stage5_no_points_fallback = False
                stage5_disable_icl_fallback = False
                stage5_fallback_reason: Optional[str] = None

                if max_stage < 4:
                    print("  Stage 4+: capped by --max-stage, skipping")
                    bbox_state["stage4_skipped_by_cap"] = True
                    bbox_state["stage5_skipped_by_cap"] = True
                    bbox_state["stage6_skipped_by_cap"] = True
                    bbox_state["stage7_skipped_by_cap"] = True
                    return bbox_state

                # Stage 4 / Stage 5
                if skip_stage45_no_icl_baseline:
                    if args.resume and stage_done(stage4_dir, STAGE4_BBOX_REQUIRED_FILES):
                        print("  Stage 4: resume hit, skipping (no-ICL baseline mode)")
                        bbox_state["stage4_done"] = True
                    else:
                        print("  Stage 4: skipping point grounding (no-ICL baseline mode)")
                        if args.dry_run:
                            (logs_dir / f"{bbox_str}_stage4.log").write_text(
                                "[DRY RUN] Stage 4 skipped due to no-ICL baseline mode; "
                                "would materialize synthetic Stage 4 metadata.\n",
                                encoding="utf-8",
                            )
                        else:
                            materialize_stage4_no_icl_baseline_fallback(
                                stage4_dir=stage4_dir,
                                wsi_path=wsi_path,
                                bbox=bbox,
                            )
                            ensure_files(stage4_dir, STAGE4_BBOX_REQUIRED_FILES, "Stage 4 bbox")
                            (logs_dir / f"{bbox_str}_stage4.log").write_text(
                                "Stage 4 skipped due to no-ICL baseline mode. "
                                "Created synthetic Stage 4 metadata.\n",
                                encoding="utf-8",
                            )
                        bbox_state["stage4_done"] = True

                    stage4_points_summary = summarize_stage4_points(stage4_dir)
                    stage4_points_total = int(stage4_points_summary.get("total_points", 0))
                    bbox_state["stage4_points_total"] = stage4_points_total
                    bbox_state["stage4_no_points"] = True

                    if args.resume and stage_done(stage5_dir, STAGE5_BBOX_REQUIRED_FILES):
                        print("  Stage 5: resume hit, skipping (no-ICL baseline mode)")
                        bbox_state["stage5_done"] = True
                        fallback_info = load_stage5_fallback_info(stage5_dir)
                        stage5_no_points_fallback = bool(fallback_info.get("no_points"))
                        stage5_disable_icl_fallback = True
                        reason_raw = fallback_info.get("reason")
                        stage5_fallback_reason = (
                            str(reason_raw)
                            if reason_raw
                            else STAGE45_SKIP_REASON_NO_ICL_BASELINE
                        )
                    else:
                        print("  Stage 5: skipping reranker (no-ICL baseline mode)")
                        stage5_contract_patch_size = (
                            int(args.stage6_patch_size)
                            if args.stage6_patch_size is not None
                            else int(args.stage5_patch_size)
                        )
                        if args.dry_run:
                            (logs_dir / f"{bbox_str}_stage5.log").write_text(
                                "[DRY RUN] Stage 5 skipped due to no-ICL baseline mode; "
                                "would materialize synthetic Stage 5 fallback metadata.\n",
                                encoding="utf-8",
                            )
                        else:
                            materialize_stage5_disable_icl_fallback(
                                stage5_dir=stage5_dir,
                                stage4_dir=stage4_dir,
                                wsi_path=wsi_path,
                                bbox=bbox,
                                stage5_patch_size=stage5_contract_patch_size,
                                stage5_k=args.stage5_k,
                            )
                            ensure_files(stage5_dir, STAGE5_BBOX_REQUIRED_FILES, "Stage 5 bbox")
                            (logs_dir / f"{bbox_str}_stage5.log").write_text(
                                "Stage 5 skipped due to no-ICL baseline mode. "
                                "Created synthetic Stage 5 fallback metadata for Stage 6.\n",
                                encoding="utf-8",
                            )
                        bbox_state["stage5_done"] = True
                        stage5_no_points_fallback = False
                        stage5_disable_icl_fallback = True
                        stage5_fallback_reason = STAGE45_SKIP_REASON_NO_ICL_BASELINE
                else:
                    if args.resume and stage_done(stage4_dir, STAGE4_BBOX_REQUIRED_FILES):
                        print("  Stage 4: resume hit, skipping")
                        bbox_state["stage4_done"] = True
                    else:
                        stage2_native_bbox = stage2_native_bbox_map.get(bbox_str)
                        if stage2_native_bbox is None:
                            raise PipelineError(
                                f"Stage 4 requires native Stage 2 bbox dir, missing for {bbox_str}"
                            )
                        print("  Stage 4: running")
                        run_command(
                            CommandSpec(
                                cmd=build_stage4_command(args, stage2_native_bbox, native_root),
                                cwd=REPO_ROOT,
                                log_path=logs_dir / f"{bbox_str}_stage4.log",
                                extra_env=stage_subprocess_env,
                            ),
                            dry_run=args.dry_run,
                            live_logs=args.live_logs,
                        )
                        stage4_native = find_stage4_native_dir(
                            native_root=native_root,
                            wsi_id=wsi_id,
                            bbox_str=bbox_str,
                            model=args.stage4_model,
                        )
                        if stage4_native is None:
                            raise PipelineError(f"Could not locate Stage 4 native output for {bbox_str}")
                        sync_tree(stage4_native, stage4_dir)
                        ensure_files(stage4_dir, STAGE4_BBOX_REQUIRED_FILES, "Stage 4 bbox")
                        bbox_state["stage4_done"] = True

                    stage4_points_summary = summarize_stage4_points(stage4_dir)
                    stage4_points_total = int(stage4_points_summary.get("total_points", 0))
                    stage4_no_points = stage4_points_total <= 0
                    bbox_state["stage4_points_total"] = stage4_points_total
                    bbox_state["stage4_no_points"] = stage4_no_points
                    if stage4_no_points:
                        print(
                            "  Stage 4: no points detected; Stage 5 reranker will be "
                            "skipped via synthetic fallback"
                        )

                    if max_stage < 5:
                        print("  Stage 5: capped by --max-stage, skipping")
                        bbox_state["stage5_skipped_by_cap"] = True
                    elif args.resume and stage_done(stage5_dir, STAGE5_BBOX_REQUIRED_FILES):
                        print("  Stage 5: resume hit, skipping")
                        bbox_state["stage5_done"] = True
                        fallback_info = load_stage5_fallback_info(stage5_dir)
                        stage5_no_points_fallback = bool(fallback_info.get("no_points"))
                        stage5_disable_icl_fallback = bool(fallback_info.get("disable_icl"))
                        reason_raw = fallback_info.get("reason")
                        stage5_fallback_reason = str(reason_raw) if reason_raw else None
                    elif stage4_no_points:
                        print("  Stage 5: skipping reranker due to empty Stage 4 points")
                        if args.dry_run:
                            (logs_dir / f"{bbox_str}_stage5.log").write_text(
                                "[DRY RUN] Stage 5 skipped: Stage 4 produced zero points; "
                                "would materialize synthetic Stage 5 fallback metadata.\n",
                                encoding="utf-8",
                            )
                        else:
                            materialize_stage5_no_points_fallback(
                                stage5_dir=stage5_dir,
                                stage4_dir=stage4_dir,
                                wsi_path=wsi_path,
                                bbox=bbox,
                                stage4_points_summary=stage4_points_summary,
                                stage5_patch_size=args.stage5_patch_size,
                                stage5_k=args.stage5_k,
                            )
                            ensure_files(stage5_dir, STAGE5_BBOX_REQUIRED_FILES, "Stage 5 bbox")
                            (logs_dir / f"{bbox_str}_stage5.log").write_text(
                                "Stage 5 skipped: Stage 4 produced zero points. "
                                "Created synthetic Stage 5 fallback metadata for Stage 6.\n",
                                encoding="utf-8",
                            )
                        bbox_state["stage5_done"] = True
                        stage5_no_points_fallback = True
                        stage5_disable_icl_fallback = True
                        stage5_fallback_reason = "stage4_no_points"
                    else:
                        print("  Stage 5: running")
                        if stage5_dir.exists():
                            shutil.rmtree(stage5_dir)
                        run_command(
                            CommandSpec(
                                cmd=build_stage5_command(
                                    args,
                                    stage4_dir,
                                    stage5_dir,
                                    case_effective_wsi_readers["stage5"],
                                ),
                                cwd=REPO_ROOT,
                                log_path=logs_dir / f"{bbox_str}_stage5.log",
                                extra_env=stage_subprocess_env,
                            ),
                            dry_run=args.dry_run,
                            live_logs=args.live_logs,
                        )
                        ensure_files(stage5_dir, STAGE5_BBOX_REQUIRED_FILES, "Stage 5 bbox")
                        bbox_state["stage5_done"] = True
                        fallback_info = load_stage5_fallback_info(stage5_dir)
                        stage5_no_points_fallback = bool(fallback_info.get("no_points"))
                        stage5_disable_icl_fallback = bool(fallback_info.get("disable_icl"))
                        reason_raw = fallback_info.get("reason")
                        stage5_fallback_reason = str(reason_raw) if reason_raw else None
                        if stage5_disable_icl_fallback:
                            reason_txt = stage5_fallback_reason or "unknown"
                            print(
                                "  Stage 5: metadata requests Stage 6 ICL disable fallback "
                                f"(reason={reason_txt})"
                            )
                bbox_state["stage5_no_points_fallback"] = stage5_no_points_fallback
                bbox_state["stage5_disable_icl_fallback"] = stage5_disable_icl_fallback
                bbox_state["stage5_fallback_reason"] = stage5_fallback_reason

                # Optional Stage 5 class description generation for Stage 6 --class-defs.
                auto_stage6_class_defs: Optional[Path] = None
                if args.stage5_generate_descriptions and max_stage >= 5:
                    generated_class_defs = stage5_dir / "class_descriptions.json"
                    if stage5_disable_icl_fallback:
                        if (
                            args.resume
                            and generated_class_defs.exists()
                            and not args.stage5_descriptions_force
                        ):
                            print(
                                "  Stage 5 descriptions: resume hit, using existing "
                                "generic fallback JSON"
                            )
                        else:
                            fallback_reason_txt = stage5_fallback_reason or "unknown"
                            if args.dry_run:
                                print(
                                    "  Stage 5 descriptions: Stage 5 fallback "
                                    f"({fallback_reason_txt}); "
                                    "would write generic fallback JSON"
                                )
                            else:
                                print(
                                    "  Stage 5 descriptions: Stage 5 fallback "
                                    f"({fallback_reason_txt}); "
                                    "writing generic fallback JSON"
                                )
                                write_generic_stage6_class_descriptions(generated_class_defs)
                    else:
                        if (
                            args.resume
                            and generated_class_defs.exists()
                            and not args.stage5_descriptions_force
                        ):
                            print("  Stage 5 descriptions: resume hit, skipping")
                        else:
                            print("  Stage 5 descriptions: running")
                            run_command(
                                CommandSpec(
                                    cmd=build_stage5_descriptions_command(args, stage5_dir),
                                    cwd=REPO_ROOT,
                                    log_path=logs_dir / f"{bbox_str}_stage5_descriptions.log",
                                    extra_env=stage_subprocess_env,
                                ),
                                dry_run=args.dry_run,
                                live_logs=args.live_logs,
                            )
                    if not args.dry_run and not generated_class_defs.exists():
                        raise PipelineError(
                            f"Stage 5 descriptions missing: {generated_class_defs}"
                        )
                    if args.stage6_class_defs is None:
                        auto_stage6_class_defs = generated_class_defs
                    else:
                        print(
                            "  Stage 5 descriptions: generated, but keeping explicit "
                            "--stage6-class-defs override"
                        )
                elif args.stage5_generate_descriptions:
                    print("  Stage 5 descriptions: capped by --max-stage, skipping")

                # Stage 6 (per-bbox ICL classification via run_vlm_bbox_inference.py)
                if max_stage < 6:
                    print("  Stage 6: capped by --max-stage, skipping")
                    bbox_state["stage6_skipped_by_cap"] = True
                elif args.resume and stage_done(stage6_dir, STAGE6_BBOX_REQUIRED_FILES):
                    print("  Stage 6: resume hit, skipping")
                    bbox_state["stage6_done"] = True
                else:
                    stage6_icl_k_override = 0 if stage5_disable_icl_fallback else None
                    if stage6_icl_k_override is not None:
                        reason_txt = stage5_fallback_reason or "unknown"
                        print(
                            "  Stage 6: running with --icl-k 0 due to Stage 5 fallback "
                            f"(reason={reason_txt})"
                        )
                    else:
                        print("  Stage 6: running")
                    stage6_tmp_parent = native_root / "stage6_tmp" / bbox_str
                    stage6_tmp_parent.mkdir(parents=True, exist_ok=True)
                    # Use a stable tmp root per bbox so Stage 6 --resume can reuse
                    # partial patches.csv checkpoints across interrupted runs.
                    stage6_tmp_root = stage6_tmp_parent

                    stage3_for_stage6 = None if args.no_stage3_gating else stage3_dir
                    cmd_stage6 = build_stage6_command(
                        args=args,
                        stage5_dir=stage5_dir,
                        stage3_dir=stage3_for_stage6,
                        stage6_tmp_root=stage6_tmp_root,
                        wsi_reader=case_effective_wsi_readers["stage6"],
                        class_defs_path=auto_stage6_class_defs,
                        icl_k_override=stage6_icl_k_override,
                    )
                    run_command(
                        CommandSpec(
                            cmd=cmd_stage6,
                            cwd=REPO_ROOT,
                            log_path=logs_dir / f"{bbox_str}_stage6.log",
                            extra_env=stage_subprocess_env,
                        ),
                        dry_run=args.dry_run,
                        live_logs=args.live_logs,
                    )

                    if not args.dry_run:
                        stage6_native = find_stage6_native_dir(stage6_tmp_root)
                        if stage6_native is None:
                            raise PipelineError(
                                f"Could not locate Stage 6 native output for {bbox_str}"
                            )
                        sync_tree(stage6_native, stage6_dir)
                        ensure_files(
                            stage6_dir,
                            STAGE6_BBOX_REQUIRED_FILES,
                            "Stage 6 bbox",
                        )
                    bbox_state["stage6_done"] = True

                # Stage 7 (morphological postprocessing)
                if max_stage < 7:
                    print("  Stage 7: capped by --max-stage, skipping")
                    bbox_state["stage7_skipped_by_cap"] = True
                elif args.no_stage7:
                    print("  Stage 7: disabled (--no-stage7), skipping")
                    bbox_state["stage7_skipped_by_flag"] = True
                elif args.resume and stage_done(stage7_dir, STAGE7_BBOX_REQUIRED_FILES):
                    print("  Stage 7: resume hit, skipping")
                    bbox_state["stage7_done"] = True
                else:
                    print("  Stage 7: running")
                    cmd_stage7 = build_stage7_command(args=args, stage6_dir=stage6_dir)
                    run_command(
                        CommandSpec(
                            cmd=cmd_stage7,
                            cwd=REPO_ROOT,
                            log_path=logs_dir / f"{bbox_str}_stage7.log",
                            extra_env=stage_subprocess_env,
                        ),
                        dry_run=args.dry_run,
                        live_logs=args.live_logs,
                    )
                    if not args.dry_run:
                        stage7_native = stage6_dir / "stage7_postprocess"
                        if not stage7_native.exists():
                            raise PipelineError(
                                f"Stage 7 output not found at {stage7_native}"
                            )
                        sync_tree(stage7_native, stage7_dir)
                        ensure_files(stage7_dir, STAGE7_BBOX_REQUIRED_FILES, "Stage 7 bbox")
                        shutil.rmtree(stage7_native)
                    bbox_state["stage7_done"] = True
            except Exception as bbox_exc:
                bbox_state["error"] = str(bbox_exc)
                raise PipelineError(f"BBox {bbox_str} failed: {bbox_exc}") from bbox_exc

            return bbox_state

        if bbox_parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=bbox_parallel_workers) as executor:
                future_map = {
                    executor.submit(process_single_bbox, bbox_str, bbox): bbox_str
                    for bbox_str, bbox in bboxes
                }
                for future in as_completed(future_map):
                    bbox_str = future_map[future]
                    try:
                        bbox_state = future.result()
                    except Exception as bbox_exc:
                        bbox_level0 = bbox_lookup.get(bbox_str, [])
                        status["bbox_status"][bbox_str] = {
                            "bbox_level0": bbox_level0,
                            "stage4_done": False,
                            "stage5_done": False,
                            "stage6_done": False,
                            "stage7_done": False,
                            "error": str(bbox_exc),
                        }
                        raise
                    status["bbox_status"][bbox_str] = bbox_state
        else:
            for bbox_str, bbox in bboxes:
                bbox_state = process_single_bbox(bbox_str, bbox)
                status["bbox_status"][bbox_str] = bbox_state

        # WSI-level Stage 7 mask assembly
        if max_stage >= 7 and not args.no_stage7 and not args.dry_run:
            print("\nStage 7: assembling WSI-level tissue mask")
            stage7_wsi_dir = run_dir / "stage7"
            assemble_wsi_mask(
                stage1_dir=stage1_dir,
                bboxes=bboxes,
                bboxes_root=bboxes_root,
                stage7_wsi_dir=stage7_wsi_dir,
            )
            print(f"Stage 7: WSI mask saved to {stage7_wsi_dir / 'mask.npy'}")

        status["ok"] = True
        status["finished_at"] = datetime.now().isoformat()
        metadata["finished_at"] = status["finished_at"]
        write_json(run_dir / "pipeline_metadata.json", metadata)
        write_json(run_dir / "pipeline_status.json", status)
        if not args.keep_native and not args.dry_run and native_root.exists():
            shutil.rmtree(native_root)
        return status

    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        status["finished_at"] = datetime.now().isoformat()
        metadata["finished_at"] = status["finished_at"]
        metadata["failed"] = True
        metadata["failure_reason"] = str(exc)
        write_json(run_dir / "pipeline_metadata.json", metadata)
        write_json(run_dir / "pipeline_status.json", status)
        if not args.keep_native and not args.dry_run:
            # Keep _native on failure for debugging regardless of keep_native flag.
            pass
        raise
    finally:
        release_run_lock(run_lock)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Stages 1-7 for WSI foreground segmentation with per-bbox outputs.",
    )

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--wsi", type=str, help="Single WSI path (absolute or relative).")
    inp.add_argument(
        "--wsi-list",
        nargs="+",
        type=Path,
        help=(
            "One or more text files with one WSI path per line each. "
            "With --stage1-from-xml, parse each list as CSV rows "
            "\"wsi_path,xml_path\"."
        ),
    )

    parser.add_argument("--output-root", type=Path, required=True, help="Root output directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id; default is timestamp.")

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip completed stage outputs if present. In --wsi-list mode without --run-id, "
            "auto-reuses the latest incomplete run per case and skips fully complete cases."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "In --wsi-list mode, continue processing other WSIs after failures "
            "and exit 0 with a printed failure summary."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print/record commands without executing.")
    parser.add_argument(
        "--live-logs",
        "--live-log",
        dest="live_logs",
        action="store_true",
        help="Stream stage subprocess logs to terminal while also writing log files.",
    )
    parser.add_argument("--keep-native", action="store_true", help="Keep run-local _native stage outputs.")
    parser.add_argument(
        "--repro-mode",
        action="store_true",
        help="Run one git/DVC clean-state preflight at startup, then bypass per-stage checks.",
    )
    parser.add_argument(
        "--skip-dvc-check",
        action="store_true",
        help="Skip DVC portion of clean-state checks (startup preflight + forwarded stage args).",
    )
    parser.add_argument(
        "--max-stage",
        type=int,
        choices=range(1, 8),
        default=7,
        help="Run only up to this stage number (1-7, default: 7).",
    )
    parser.add_argument(
        "--parallelise-bboxes",
        "--parallelize-bboxes",
        dest="parallelise_bboxes",
        type=int,
        default=1,
        help=(
            "Max concurrent bbox workers for Stage 4+ per WSI. "
            "Parallel mode is enabled only when active per-bbox VLM stages "
            "use vertex/openrouter transport."
        ),
    )

    parser.add_argument("--parallel-stage23", action="store_true", default=True)
    parser.add_argument("--no-parallel-stage23", dest="parallel_stage23", action="store_false")

    parser.add_argument(
        "--allstage-backend",
        choices=["openrouter", "vllm", "vertex"],
        default=None,
        help=(
            "Convenience override for Stage 1/2/4/5/6 backends. "
            "For Stage 5/6, value 'vertex' uses the Gemini backend with Vertex transport. "
            "Ignored for any stage where an explicit --stage*-backend flag is provided."
        ),
    )
    parser.add_argument(
        "--allstage-model",
        type=str,
        default=None,
        help=(
            "Convenience override for Stage 1/2/4/5/6 models. "
            "Ignored for any stage where an explicit --stage*-model flag is provided."
        ),
    )
    parser.add_argument(
        "--allstage-wsi-reader",
        "--allstage-reader",
        dest="allstage_wsi_reader",
        choices=["auto", "openslide", "cucim"],
        default=None,
        help=(
            "Convenience override for Stage 1/2/5/6 WSI readers. "
            "Ignored for any stage where an explicit --stage*-wsi-reader/--stage*-reader flag is provided."
        ),
    )
    parser.add_argument(
        "--allstage-vllm-url",
        "--allstage-vlm-url",
        dest="allstage_vllm_url",
        type=str,
        default=None,
        help=(
            "Convenience override for Stage 1/2/4/6 --stage*-vllm-url values. "
            "If --stage5-vlm-port is not explicitly set, derive Stage 5 port from this URL. "
            "Ignored for any stage where an explicit stage-specific URL/port flag is provided."
        ),
    )

    # Stage 1
    parser.add_argument("--stage1-backend", choices=["openrouter", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--stage1-model", type=str, default="google/gemini-3-flash-preview")
    parser.add_argument("--stage1-openrouter-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--stage1-vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--stage1-api-key", type=str, default=None)
    parser.add_argument(
        "--stage1-vertex-credentials",
        type=Path,
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument("--stage1-vertex-location", type=str, default="global")
    parser.add_argument("--stage1-repair-model", type=str, default=None)
    parser.add_argument("--stage1-coord-order", choices=["auto", "yxxy", "xyxy"], default="auto")
    parser.add_argument(
        "--stage1-padding",
        type=float,
        default=0.25,
        help="Stage 1 bbox padding fraction (set 0 to disable padding).",
    )
    parser.add_argument(
        "--stage1-merge-overlap-threshold",
        type=float,
        default=0.2,
        help="Stage 1 merge overlap threshold (intersection/min-area) for bbox merge.",
    )
    parser.add_argument(
        "--stage1-rotations",
        nargs="+",
        type=int,
        default=[0, 90, 180, 270],
        choices=[0, 90, 180, 270],
        help="Stage 1 orientations for TTA (default: 0 90 180 270).",
    )
    parser.add_argument(
        "--stage1-save-intermediate",
        action="store_true",
        help="Pass through to Stage 1 --save-intermediate for debug artifacts.",
    )
    parser.add_argument(
        "--stage1-save-bbox-region",
        action="store_true",
        help="Pass through to Stage 1 --save-bbox-region (export optional bbox regions from L0).",
    )
    parser.add_argument(
        "--stage1-wsi-reader",
        "--stage1-reader",
        dest="stage1_wsi_reader",
        choices=["auto", "openslide", "cucim"],
        default="auto",
        help="WSI reader backend for Stage 1 thumbnail extraction (default: auto).",
    )
    parser.add_argument(
        "--stage1-from-xml",
        action="store_true",
        help=(
            "Bypass Stage 1 VLM detection and synthesize stage1 outputs from XML ROI annotations. "
            "For --wsi, provide --stage1-xml. For --wsi-list, rows must be CSV wsi_path,xml_path."
        ),
    )
    parser.add_argument(
        "--stage1-xml",
        type=Path,
        default=None,
        help="ROI XML path for single-case mode (--wsi) when --stage1-from-xml is set.",
    )
    parser.add_argument(
        "--stage1-xml-group",
        type=str,
        default="biopsy",
        help="Annotation group name in XML to interpret as tissue ROI bboxes (default: biopsy).",
    )
    parser.add_argument(
        "--stage1-xml-include-non-rect",
        action="store_true",
        help="Allow non-Rectangle annotations in ROI XML by using their coordinate extents.",
    )
    parser.add_argument(
        "--stage1-xml-model-tag",
        type=str,
        default="xml_roi",
        help="Model tag written into synthesized Stage 1 metadata for XML mode.",
    )
    parser.add_argument(
        "--stage1-xml-max-dim",
        type=int,
        default=1024,
        help="Thumbnail max dimension used by XML Stage 1 materialization.",
    )

    # Stage 2
    parser.add_argument("--stage2-backend", choices=["openrouter", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--stage2-model", type=str, default="google/gemini-3-flash-preview")
    parser.add_argument("--stage2-openrouter-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--stage2-vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument(
        "--stage2-vertex-credentials",
        type=Path,
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument("--stage2-vertex-location", type=str, default="global")
    parser.add_argument("--stage2-api-key", type=str, default=None)
    parser.add_argument("--stage2-max-dim", type=int, default=1024)
    parser.add_argument(
        "--stage2-force-read-l0",
        action="store_true",
        help=(
            "Force Stage 2 to read bbox crops at level 0 and downsample to "
            "--stage2-max-dim. Also applies to synthetic bbox_region.png creation "
            "when --skip-stage2 is enabled."
        ),
    )
    parser.add_argument(
        "--stage2-wsi-reader",
        "--stage2-reader",
        dest="stage2_wsi_reader",
        choices=["auto", "openslide", "cucim"],
        default="auto",
        help="WSI reader backend for Stage 2 bbox extraction (default: auto).",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help=(
            "Skip Stage 2 artifact QC model calls and synthesize Stage 2 outputs with "
            "EXCLUDE-all artifact verdicts for every bbox."
        ),
    )

    # Stage 3
    parser.add_argument("--stage3-method", choices=["kmeans", "hdbscan"], default="kmeans")
    parser.add_argument("--stage3-k", type=int, default=2)
    parser.add_argument("--stage3-min-cluster-size", type=int, default=50)
    parser.add_argument("--stage3-blur", type=float, default=0.0)

    # Stage 4
    parser.add_argument("--stage4-backend", choices=["openrouter", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--stage4-model", type=str, default="google/gemini-3-flash-preview")
    parser.add_argument("--stage4-openrouter-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--stage4-vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument(
        "--stage4-vertex-credentials",
        type=Path,
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument("--stage4-vertex-location", type=str, default="global")
    parser.add_argument("--stage4-api-key", type=str, default=None)
    parser.add_argument("--stage4-point-order", choices=["auto", "xy", "yx"], default="auto")
    parser.add_argument("--stage4-point-key", choices=["auto", "point", "point_2d"], default="auto")
    parser.add_argument("--stage4-repair-model", type=str, default=None)
    parser.add_argument("--stage4-max-items", type=int, default=5)
    parser.add_argument("--stage4-max-tokens", type=int, default=512)
    parser.add_argument(
        "--stage4-thinking-level",
        type=str,
        default=None,
        help="Optional Stage 4 Gemini thinking level (Low/High). Used with vertex backend.",
    )
    parser.add_argument(
        "--stage4-include-thoughts",
        action="store_true",
        help="Request Stage 4 Gemini thought summaries (vertex backend only).",
    )
    parser.add_argument("--stage4-use-visual-descriptions", action="store_true")
    parser.add_argument("--stage4-no-tta", action="store_true")

    # Stage 5
    parser.add_argument("--stage5-top-k", type=int, default=3)
    parser.add_argument("--stage5-k", type=int, choices=[1, 4], default=1)
    parser.add_argument("--stage5-patch-size", type=int, default=512)
    parser.add_argument(
        "--stage5-wsi-reader",
        "--stage5-reader",
        dest="stage5_wsi_reader",
        choices=["auto", "openslide", "cucim"],
        default="auto",
        help="WSI reader backend for Stage 5 patch extraction (default: auto).",
    )
    parser.add_argument(
        "--stage5-vlm-backend",
        choices=["vllm", "openrouter", "gemini", "vertex"],
        default="openrouter",
    )
    parser.add_argument("--stage5-vlm-model", type=str, default="google/gemini-2.0-flash-001")
    parser.add_argument("--stage5-vlm-port", type=int, default=8000)
    parser.add_argument("--stage5-vlm-max-tokens", type=int, default=512)
    parser.add_argument(
        "--stage5-selection-mode",
        choices=["auto", "tournament", "single_pass"],
        default="auto",
        help=(
            "Stage 5 ranking reduction strategy. auto uses single_pass for gemini "
            "and tournament for non-gemini backends."
        ),
    )
    parser.add_argument(
        "--stage5-vlm-image-size",
        type=int,
        default=None,
        help=(
            "Optional square image size used only for Stage 5 VLM ranking payloads "
            "(e.g., 256 for Gemini single-pass)."
        ),
    )
    parser.add_argument("--stage5-gemini-use-vertex", action="store_true", default=True)
    parser.add_argument("--stage5-gemini-no-vertex", dest="stage5_gemini_use_vertex", action="store_false")
    parser.add_argument(
        "--stage5-gemini-credentials",
        type=Path,
        default=None,
        help="Optional Gemini Vertex credentials JSON. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument("--stage5-gemini-location", type=str, default="global")
    parser.add_argument(
        "--stage5-gemini-thinking-level",
        type=str,
        default="High",
        help="Optional Stage 5 Gemini thinking level (Low/High).",
    )
    parser.add_argument(
        "--stage5-gemini-include-thoughts",
        action="store_true",
        help="Request Stage 5 Gemini thought summaries (if supported).",
    )
    parser.add_argument(
        "--stage5-openrouter-reasoning-effort",
        type=str,
        default="high",
        help="Stage 5 OpenRouter reasoning effort: low/medium/high.",
    )
    parser.add_argument(
        "--stage5-max-total-candidates",
        type=int,
        default=25,
        help=(
            "Hard cap on candidates per Stage 5 VLM call. "
            "Used by reranker tournament mode (default: 25)."
        ),
    )
    parser.add_argument(
        "--stage5-tournament-round1-k",
        type=int,
        default=None,
        help=(
            "Optional per-class top-k override for Stage 5 tournament reduction round 1. "
            "When omitted, reranker computes it automatically."
        ),
    )
    parser.add_argument(
        "--stage5-max-candidates-per-class",
        type=int,
        default=None,
        help=(
            "Optional pass-through to reranker.py --max-candidates-per-class. "
            "When omitted, reranker.py default is used."
        ),
    )
    parser.add_argument(
        "--stage5-disable-tissue-blur-filter",
        action="store_true",
        help="Disable Stage 5 post-selection blur gating for tissue ICL picks.",
    )
    parser.add_argument(
        "--stage5-tissue-blur-threshold",
        type=float,
        default=0.1,
        help="Keep tissue candidates with blur_score <= this threshold (default: 0.1).",
    )
    parser.add_argument(
        "--stage5-tissue-blur-sigma",
        type=float,
        default=0.5,
        help="Gaussian sigma for Stage 5 tissue blur scoring (default: 0.5).",
    )
    parser.add_argument(
        "--stage5-tissue-blur-pixel-threshold",
        type=float,
        default=0.005,
        help="Per-pixel sharpness threshold for Stage 5 tissue blur scoring (default: 0.005).",
    )
    parser.add_argument(
        "--stage5-generate-descriptions",
        action="store_true",
        help=(
            "After Stage 5, run generate_stage5_descriptions.py and, unless --stage6-class-defs "
            "is explicitly set, pass stage5/class_descriptions.json into Stage 6."
        ),
    )
    parser.add_argument(
        "--stage5-descriptions-force",
        action="store_true",
        help="Force regeneration of stage5/class_descriptions.json when using --stage5-generate-descriptions.",
    )

    # Stage 6 (run_vlm_bbox_inference.py)
    parser.add_argument("--stage6-backend", choices=["gemini", "vllm", "openrouter", "vertex"], default="vllm")
    parser.add_argument("--stage6-model", type=str, default=None)
    parser.add_argument(
        "--stage6-wsi-reader",
        "--stage6-reader",
        dest="stage6_wsi_reader",
        choices=["auto", "openslide", "cucim"],
        default="auto",
        help="WSI reader backend for Stage 6 patch extraction (default: auto).",
    )
    parser.add_argument("--stage6-vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--stage6-openrouter-url", type=str, default=None)
    parser.add_argument("--stage6-openrouter-api-key", type=str, default=None)
    parser.add_argument("--stage6-openrouter-referer", type=str, default=None)
    parser.add_argument("--stage6-gemini-use-vertex", action="store_true", default=True)
    parser.add_argument("--stage6-gemini-no-vertex", dest="stage6_gemini_use_vertex", action="store_false")
    parser.add_argument(
        "--stage6-gemini-credentials",
        type=Path,
        default=None,
        help="Optional Gemini Vertex credentials JSON. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    parser.add_argument("--stage6-gemini-location", type=str, default="global")
    parser.add_argument("--stage6-prompt-template", type=Path, default=None)
    parser.add_argument("--stage6-class-defs", type=Path, default=None)
    parser.add_argument("--stage6-label-mode", choices=["semantic", "neutral"], default="semantic")
    parser.add_argument(
        "--stage6-icl-k",
        type=int,
        default=3,
        help=(
            "Stage 6 ICL examples per class. When set to 0 and "
            "--stage5-generate-descriptions is not set, Stage 4/5 are skipped "
            "and synthetic fallback metadata is materialized."
        ),
    )
    parser.add_argument("--stage6-icl-shuffle-n", type=int, default=1)
    parser.add_argument("--stage6-rotations", type=str, default="0")
    parser.add_argument(
        "--stage6-query-batch-size",
        type=int,
        default=1,
        help=(
            "Stage 6 query patches per VLM request (maps to "
            "run_vlm_bbox_inference.py --query-batch-size; default: 1)."
        ),
    )
    parser.add_argument("--stage6-vlm-image-size", type=int, default=None)
    parser.add_argument("--stage6-patch-size", type=int, default=None)
    parser.add_argument("--stage6-max-workers", type=int, default=16)
    parser.add_argument("--stage6-timeout", type=int, default=120)
    parser.add_argument("--stage6-temperature", type=float, default=0.0)
    parser.add_argument("--stage6-max-tokens", type=int, default=256)
    parser.add_argument("--stage6-max-retries", type=int, default=3)
    parser.add_argument("--stage6-stage3-fg-threshold", type=float, default=0.05)
    parser.add_argument("--no-stage3-gating", action="store_true")

    # Stage 7 (postprocess_mask.py - morphological postprocessing)
    parser.add_argument("--no-stage7", action="store_true",
                        help="Skip Stage 7 morphological postprocessing.")
    parser.add_argument("--stage7-min-component-size", type=int, default=3)
    parser.add_argument("--stage7-connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--stage7-close-kernel", type=int, default=3)
    parser.add_argument("--stage7-skip-remove-small", action="store_true")
    parser.add_argument("--stage7-skip-close", action="store_true")
    parser.add_argument("--stage7-skip-fill-holes", action="store_true")
    parser.add_argument("--stage7-allow-artifact-overwrite", action="store_true")

    return parser


def main() -> int:
    parser = create_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    apply_allstage_overrides(args, argv)
    if (
        args.stage5_max_candidates_per_class is not None
        and args.stage5_max_candidates_per_class < 1
    ):
        parser.error("--stage5-max-candidates-per-class must be >= 1")
    if args.stage5_max_total_candidates < 1:
        parser.error("--stage5-max-total-candidates must be >= 1")
    if (
        args.stage5_tournament_round1_k is not None
        and args.stage5_tournament_round1_k < 1
    ):
        parser.error("--stage5-tournament-round1-k must be >= 1")
    if args.stage5_vlm_image_size is not None and args.stage5_vlm_image_size < 1:
        parser.error("--stage5-vlm-image-size must be >= 1")
    if not (0.0 <= args.stage5_tissue_blur_threshold <= 1.0):
        parser.error("--stage5-tissue-blur-threshold must be in [0, 1]")
    if args.stage5_tissue_blur_sigma <= 0.0:
        parser.error("--stage5-tissue-blur-sigma must be > 0")
    if args.stage5_tissue_blur_pixel_threshold < 0.0:
        parser.error("--stage5-tissue-blur-pixel-threshold must be >= 0")
    if args.stage6_icl_k < 0:
        parser.error("--stage6-icl-k must be >= 0")
    if args.stage6_query_batch_size < 1:
        parser.error("--stage6-query-batch-size must be >= 1")
    if args.parallelise_bboxes < 1:
        parser.error("--parallelise-bboxes must be >= 1")
    if args.stage4_max_tokens < 1:
        parser.error("--stage4-max-tokens must be >= 1")
    if args.stage4_thinking_level is not None:
        thinking_level_normalized = args.stage4_thinking_level.strip().lower()
        if thinking_level_normalized not in {"low", "high"}:
            parser.error("--stage4-thinking-level must be one of: Low, High")
        args.stage4_thinking_level = "Low" if thinking_level_normalized == "low" else "High"
    if args.stage5_gemini_thinking_level is not None:
        stage5_thinking_level_normalized = args.stage5_gemini_thinking_level.strip().lower()
        if stage5_thinking_level_normalized not in {"low", "high"}:
            parser.error("--stage5-gemini-thinking-level must be one of: Low, High")
        args.stage5_gemini_thinking_level = (
            "Low" if stage5_thinking_level_normalized == "low" else "High"
        )
    if args.stage5_openrouter_reasoning_effort is not None:
        reasoning_effort_normalized = args.stage5_openrouter_reasoning_effort.strip().lower()
        if not reasoning_effort_normalized:
            args.stage5_openrouter_reasoning_effort = None
        elif reasoning_effort_normalized not in {"low", "medium", "high"}:
            parser.error(
                "--stage5-openrouter-reasoning-effort must be one of: low, medium, high"
            )
        else:
            args.stage5_openrouter_reasoning_effort = reasoning_effort_normalized

    if args.stage1_xml is not None and not args.stage1_from_xml:
        parser.error("--stage1-xml requires --stage1-from-xml")
    if args.stage1_padding < 0:
        parser.error("--stage1-padding must be >= 0")
    if not (0.0 <= args.stage1_merge_overlap_threshold <= 1.0):
        parser.error("--stage1-merge-overlap-threshold must be in [0, 1]")
    if args.stage1_from_xml and args.wsi and args.stage1_xml is None:
        parser.error("--stage1-from-xml with --wsi requires --stage1-xml <path>")
    if args.stage1_xml is not None and args.wsi_list is not None:
        parser.error("--stage1-xml is only valid with --wsi (single-case mode)")
    if args.stage1_xml_max_dim < 32:
        parser.error("--stage1-xml-max-dim must be >= 32")

    items: List[Tuple[str, Optional[str], Optional[str], Optional[str]]]
    if args.wsi:
        single_xml = str(args.stage1_xml) if args.stage1_xml is not None else None
        items = [(args.wsi, single_xml, None, None)]
    else:
        list_paths = [Path(p).expanduser() for p in args.wsi_list]
        if args.stage1_from_xml:
            items = [
                (w, x, list_path.stem, str(list_path))
                for list_path in list_paths
                for w, x in read_wsi_xml_list(list_path)
            ]
        else:
            items = [
                (w, None, list_path.stem, str(list_path))
                for list_path in list_paths
                for w in read_wsi_list(list_path)
            ]
        worklist_counts: Dict[str, int] = {}
        for _, _, source_worklist, _ in items:
            key = source_worklist or "unknown"
            worklist_counts[key] = worklist_counts.get(key, 0) + 1
        counts_text = ", ".join(f"{k}={v}" for k, v in sorted(worklist_counts.items()))
        print(f"Input worklist rows: {len(items)} total ({counts_text})")

    args.output_root = Path(args.output_root).expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    repro_preflight: Optional[Dict[str, object]] = None
    if args.repro_mode:
        print("Repro preflight: checking repository state once at startup")
        repro_preflight = run_repro_preflight(skip_dvc_check=args.skip_dvc_check)
        print(f"Repro preflight: clean at git commit {repro_preflight['git_hash']}")

    failures: List[Tuple[str, str]] = []
    for wsi_input, stage1_xml_input, source_worklist, source_worklist_path in items:
        try:
            resolved_wsi_path = resolve_wsi_path(wsi_input)
            wsi_id = Path(resolved_wsi_path).stem

            run_id_override: Optional[str] = None
            if args.resume and args.run_id is None:
                plan = build_resume_plan(args.output_root, wsi_id)
                if plan.action == "skip":
                    print(
                        f"\nWSI already complete: {wsi_id} "
                        f"(run_id={plan.run_id}, detail={plan.detail})"
                    )
                    continue
                if plan.action == "resume":
                    run_id_override = plan.run_id
                    print(
                        f"\nWSI resume target: {wsi_id} "
                        f"(run_id={plan.run_id}, detail={plan.detail})"
                    )

            result = run_single_wsi(
                wsi_input,
                args,
                repro_preflight=repro_preflight,
                run_id_override=run_id_override,
                resolved_wsi_path=resolved_wsi_path,
                stage1_xml_input=stage1_xml_input,
                source_worklist=source_worklist,
                source_worklist_path=source_worklist_path,
            )
            print(f"\nWSI complete: {result['wsi_id']} (ok={result['ok']})")
        except Exception as exc:
            item_label = f"{source_worklist}:{wsi_input}" if source_worklist else wsi_input
            failures.append((item_label, str(exc)))
            if source_worklist:
                print(
                    f"\nWSI failed [{source_worklist}]: {wsi_input}\n  {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"\nWSI failed: {wsi_input}\n  {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailures:")
        for item, err in failures:
            print(f"  - {item}: {err}")
        if args.continue_on_error:
            print(
                "\nCompleted with skipped failures (--continue-on-error enabled). "
                "Exiting with code 0."
            )
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
