#!/usr/bin/env python3
"""Run foreground segmentation from selector-filtered scale500 bboxes.

This is the orchestration layer for PER-250: it exports verified selector
bboxes into the `run_auto_context.py` resume layout, then optionally launches
the downstream auto-context foreground stages with Stage 1/2/3 treated as
cached hits.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_PATH_AGENT_PYTHON = Path("/vol/biomedic3/vj724/.conda/envs/path-agent/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def as_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.expanduser().resolve())


def command_to_shell(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def default_python() -> str:
    if DEFAULT_PATH_AGENT_PYTHON.exists():
        return str(DEFAULT_PATH_AGENT_PYTHON)
    return sys.executable


def build_export_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "export_scale500_detector_to_auto_context.py"),
        "--scale500-run-dir",
        str(args.scale500_run_dir),
        "--output-root",
        str(args.output_root),
        "--run-id",
        args.run_id,
        "--selection-jsonl",
        str(args.selection_jsonl),
        "--selection-policy",
        args.selection_policy,
        "--stage2-crop-mode",
        args.stage2_crop_mode,
        "--seed-stage3",
        args.seed_stage3,
    ]
    if args.selection_case_input_root is not None:
        cmd.extend(["--selection-case-input-root", str(args.selection_case_input_root)])
    for case_id in args.case:
        cmd.extend(["--case", case_id])
    if args.case_list is not None:
        cmd.extend(["--case-list", str(args.case_list)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    for prefix_map in args.wsi_prefix_map:
        cmd.extend(["--wsi-prefix-map", prefix_map])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.export_dry_run:
        cmd.append("--dry-run")
    return cmd


def build_auto_context_command(args: argparse.Namespace) -> list[str]:
    manifest = (
        Path(args.output_root).expanduser().resolve()
        / "_scale500_adapter_manifests"
        / f"{args.run_id}.wsi_list.txt"
    )
    cmd = [
        args.python,
        str(REPO_ROOT / "run_auto_context.py"),
        "--wsi-list",
        str(manifest),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--run-id",
        args.run_id,
        "--resume",
        "--skip-stage2",
        "--stage4-backend",
        args.stage4_backend,
        "--stage4-model",
        args.stage4_model,
        "--stage5-vlm-backend",
        args.stage5_vlm_backend,
        "--stage5-vlm-model",
        args.stage5_vlm_model,
        "--stage5-openrouter-reasoning-effort",
        args.stage5_openrouter_reasoning_effort,
        "--stage6-backend",
        args.stage6_backend,
        "--stage6-model",
        args.stage6_model,
        "--stage6-icl-k",
        str(args.stage6_icl_k),
        "--stage6-rotations",
        args.stage6_rotations,
        "--stage6-query-batch-size",
        str(args.stage6_query_batch_size),
        "--stage6-max-workers",
        str(args.stage6_max_workers),
        "--parallelise-bboxes",
        str(args.parallelise_bboxes),
        "--skip-dvc-check",
    ]
    if args.max_stage is not None:
        cmd.extend(["--max-stage", str(args.max_stage)])
    if args.stage6_backend == "vllm":
        cmd.extend(["--stage6-vllm-url", args.stage6_vllm_url])
    cmd.extend(["--stage7-max-hole-size", str(args.stage7_max_hole_size)])
    if args.stage7_skip_fill_holes:
        cmd.append("--stage7-skip-fill-holes")
    if args.stage7_skip_close:
        cmd.append("--stage7-skip-close")
    if args.stage7_skip_remove_small:
        cmd.append("--stage7-skip-remove-small")
    return cmd


def will_run_stage6(args: argparse.Namespace) -> bool:
    return args.max_stage is None or args.max_stage >= 6


def parse_openrouter_key_from_zshrc(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or "OPENROUTER_API_KEY" not in line or "=" not in line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        if key.strip() != "OPENROUTER_API_KEY":
            continue
        value = value.strip()
        if not value:
            continue
        try:
            parts = shlex.split(value, comments=False, posix=True)
        except ValueError:
            parts = [value.strip("\"'")]
        if parts:
            return parts[0]
    return None


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.load_openrouter_from_zshrc and not env.get("OPENROUTER_API_KEY"):
        key = parse_openrouter_key_from_zshrc(args.zshrc)
        if key:
            env["OPENROUTER_API_KEY"] = key
    return env


def extract_model_ids(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    data = payload.get("data")
    if not isinstance(data, list):
        return set()
    ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            ids.add(model_id)
    return ids


def check_vllm_ready(
    vllm_url: str,
    *,
    expected_model: str,
    model_aliases: Iterable[str] = (),
    timeout_seconds: float = 5.0,
) -> None:
    url = vllm_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if response.status >= 400:
                raise RuntimeError(f"vLLM readiness check failed with HTTP {response.status}: {url}")
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"vLLM endpoint is not ready at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"vLLM endpoint returned non-JSON model list at {url}: {exc}") from exc
    model_ids = extract_model_ids(payload)
    allowed_ids = {expected_model, *model_aliases}
    if allowed_ids and not (model_ids & allowed_ids):
        observed = ", ".join(sorted(model_ids)) if model_ids else "none"
        expected = ", ".join(sorted(allowed_ids))
        raise RuntimeError(
            f"vLLM endpoint at {url} is ready but does not serve expected model. "
            f"expected one of [{expected}], observed [{observed}]"
        )


def run_command(cmd: Sequence[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(command_to_shell(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=REPO_ROOT, env=env, check=True)


def write_plan(
    *,
    args: argparse.Namespace,
    export_cmd: Sequence[str],
    auto_context_cmd: Sequence[str],
    env: dict[str, str],
) -> dict[str, str]:
    plan_dir = Path(args.output_root).expanduser().resolve() / "_selector_seeded_foreground_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_json = plan_dir / f"{args.run_id}.json"
    plan_sh = plan_dir / f"{args.run_id}.sh"
    payload = {
        "created_at": utc_now(),
        "run_id": args.run_id,
        "purpose": "selector-filtered scale500 bboxes -> run_auto_context foreground segmentation",
        "selection_policy": args.selection_policy,
        "stage7": {
            "skip_close": bool(args.stage7_skip_close),
            "skip_fill_holes": bool(args.stage7_skip_fill_holes),
            "skip_remove_small": bool(args.stage7_skip_remove_small),
            "max_hole_size": int(args.stage7_max_hole_size),
        },
        "stage6": {
            "backend": args.stage6_backend,
            "model": args.stage6_model,
            "icl_k": args.stage6_icl_k,
            "max_workers": args.stage6_max_workers,
            "query_batch_size": args.stage6_query_batch_size,
            "vllm_url": args.stage6_vllm_url if args.stage6_backend == "vllm" else None,
        },
        "openrouter_key_available": bool(env.get("OPENROUTER_API_KEY")),
        "commands": {
            "export": list(export_cmd),
            "auto_context": list(auto_context_cmd),
        },
    }
    plan_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    key_loader_lines = [
        "if [ -z \"${OPENROUTER_API_KEY:-}\" ]; then",
        f"  export SELECTOR_PIPELINE_ZSHRC={shlex.quote(str(args.zshrc.expanduser()))}",
        f"  export SELECTOR_PIPELINE_REPO={shlex.quote(str(REPO_ROOT))}",
        "  OPENROUTER_API_KEY=\"$(",
        f"    {shlex.quote(args.python)} - <<'PY'",
        "import os",
        "import sys",
        "from pathlib import Path",
        "sys.path.insert(0, str(Path(os.environ['SELECTOR_PIPELINE_REPO']) / 'scripts'))",
        "from run_selector_seeded_foreground_pipeline import parse_openrouter_key_from_zshrc",
        "print(parse_openrouter_key_from_zshrc(Path(os.environ['SELECTOR_PIPELINE_ZSHRC'])) or '')",
        "PY",
        "  )\"",
        "  export OPENROUTER_API_KEY",
        "fi",
        "if [ -z \"${OPENROUTER_API_KEY:-}\" ]; then",
        "  echo 'OPENROUTER_API_KEY is missing; export it or add it to the configured zshrc.' >&2",
        "  exit 2",
        "fi",
        "",
    ]
    plan_sh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                *key_loader_lines,
                command_to_shell(export_cmd),
                command_to_shell(auto_context_cmd),
                "",
            ]
        ),
        encoding="utf-8",
    )
    plan_sh.chmod(0o755)
    return {"json": str(plan_json), "shell": str(plan_sh)}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale500-run-dir", required=True, type=Path)
    parser.add_argument("--selection-jsonl", required=True, type=Path)
    parser.add_argument("--selection-case-input-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection-policy",
        choices=[
            "baseline",
            "direct",
            "verifier",
            "verifier-or-baseline",
            "conservative-verifier-drop-only",
        ],
        default="verifier",
    )
    parser.add_argument(
        "--stage2-crop-mode",
        choices=["source-bbox", "full-context"],
        default="source-bbox",
    )
    parser.add_argument(
        "--seed-stage3",
        choices=["all-foreground", "none"],
        default="all-foreground",
    )
    parser.add_argument("--wsi-prefix-map", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-dry-run", action="store_true")

    parser.add_argument("--stage4-backend", choices=["openrouter", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--stage4-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--stage5-vlm-backend", choices=["openrouter", "gemini", "vllm", "vertex"], default="openrouter")
    parser.add_argument("--stage5-vlm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--stage5-openrouter-reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--stage6-backend", choices=["gemini", "vllm", "openrouter", "vertex"], default="vllm")
    parser.add_argument("--stage6-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--stage6-vllm-url", default=DEFAULT_VLLM_URL)
    parser.add_argument(
        "--stage6-vllm-model-alias",
        action="append",
        default=[],
        help="Additional /v1/models id accepted as compatible with --stage6-model.",
    )
    parser.add_argument("--stage6-icl-k", type=int, default=0)
    parser.add_argument("--stage6-rotations", default="0")
    parser.add_argument("--stage6-query-batch-size", type=int, default=1)
    parser.add_argument("--stage6-max-workers", type=int, default=16)
    parser.add_argument("--parallelise-bboxes", type=int, default=1)
    parser.add_argument("--max-stage", type=int, default=None)

    parser.add_argument(
        "--stage7-fill-holes",
        dest="stage7_skip_fill_holes",
        action="store_false",
        help="Enable Stage 7 binary_fill_holes. This is the default; kept for explicit replay plans.",
    )
    parser.add_argument(
        "--stage7-skip-fill-holes",
        dest="stage7_skip_fill_holes",
        action="store_true",
        help="Disable Stage 7 binary_fill_holes.",
    )
    parser.set_defaults(stage7_skip_fill_holes=False)
    parser.add_argument(
        "--stage7-max-hole-size",
        type=int,
        default=1,
        help="Fill enclosed Stage 7 background holes up to this many patch-grid cells; 0 fills all holes.",
    )
    parser.add_argument(
        "--stage7-skip-close",
        action="store_true",
        default=True,
        help="Disable Stage 7 binary closing. Default is to avoid closing and rely on fill holes.",
    )
    parser.add_argument(
        "--stage7-close",
        dest="stage7_skip_close",
        action="store_false",
        help="Enable Stage 7 binary closing.",
    )
    parser.add_argument("--stage7-skip-remove-small", action="store_true")

    parser.add_argument("--python", default=default_python())
    parser.add_argument(
        "--execute-auto-context",
        action="store_true",
        help="After export, run run_auto_context.py. Without this, only export and write a launch plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/write the commands without executing either export or auto-context.",
    )
    parser.add_argument(
        "--require-vllm-ready",
        action="store_true",
        default=True,
        help="When executing with Stage 6 vLLM, require /v1/models to be reachable before auto-context.",
    )
    parser.add_argument("--no-require-vllm-ready", dest="require_vllm_ready", action="store_false")
    parser.add_argument("--zshrc", type=Path, default=Path.home() / ".zshrc")
    parser.add_argument("--load-openrouter-from-zshrc", action="store_true", default=True)
    parser.add_argument("--no-zshrc-openrouter-key", dest="load_openrouter_from_zshrc", action="store_false")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.export_dry_run and args.execute_auto_context:
        raise SystemExit("--export-dry-run cannot be combined with --execute-auto-context")
    export_cmd = build_export_command(args)
    auto_context_cmd = build_auto_context_command(args)
    env = build_env(args)
    plan_paths = write_plan(args=args, export_cmd=export_cmd, auto_context_cmd=auto_context_cmd, env=env)
    print("launch plan:")
    for key, value in plan_paths.items():
        print(f"  {key}: {value}")
    print(f"openrouter_key={'set' if env.get('OPENROUTER_API_KEY') else 'missing'}")

    run_command(export_cmd, env=env, dry_run=args.dry_run)
    if args.execute_auto_context:
        if (
            args.stage6_backend == "vllm"
            and args.require_vllm_ready
            and will_run_stage6(args)
            and not args.dry_run
        ):
            check_vllm_ready(
                args.stage6_vllm_url,
                expected_model=args.stage6_model,
                model_aliases=args.stage6_vllm_model_alias,
            )
        run_command(auto_context_cmd, env=env, dry_run=args.dry_run)
    else:
        print(command_to_shell(auto_context_cmd))
        print("auto-context execution skipped; rerun with --execute-auto-context to segment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
