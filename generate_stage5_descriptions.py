#!/usr/bin/env python3
# ABOUTME: Generate high-magnification class descriptions from Stage 5 ICL patches.
# ABOUTME: Uses OpenAI-compatible backends or Gemini SDK (Vertex/AI Studio) to produce per-class visual summaries.
"""
Generate class descriptions from Stage 5 ICL patches.

This script reads curated high-magnification patches from a Stage 5 run directory
and asks a VLM to summarize the visual features for each class in 1-2 sentences.

Usage:
  python generate_stage5_descriptions.py --stage5-dir <run_dir>

  # Process multiple Stage 5 run directories
  python generate_stage5_descriptions.py --stage5-dir dir1 --stage5-dir dir2

  # Limit examples per class
  python generate_stage5_descriptions.py --stage5-dir <run_dir> --max-per-class 2

  # Force regeneration
  python generate_stage5_descriptions.py --stage5-dir <run_dir> --force
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from utils.vlm_utils import encode_image_base64, normalize_class_label


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_BACKEND = "openrouter"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_CREDENTIALS = None

_GEMINI_CLIENT = None
_GEMINI_CLIENT_CACHE_KEY: Optional[Tuple[bool, Optional[str], Optional[str], str]] = None
_GEMINI_CLIENT_LOCK = threading.Lock()

PROMPT_TEMPLATE = """You will be shown labeled high-magnification patches from a whole slide image. \
For each class, write 1-2 sentences describing the visual features that distinguish it. \
Focus on color, texture, and cellular structures. Do NOT describe location or position, and do NOT infer diagnosis.

Classes present: {classes}

Format your response exactly as:
{format_lines}
"""


def _load_stage5_output(stage5_dir: Path) -> Dict[str, List[Path]]:
    """
    Load class -> image path list from a Stage 5 run directory.

    Prefers metadata.json "output" if available, otherwise scans subdirs.
    """
    class_to_paths: Dict[str, List[Path]] = {}

    meta_path = stage5_dir / "metadata.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            output = meta.get("output")
            if isinstance(output, dict):
                for raw_class, rel_paths in output.items():
                    if not isinstance(rel_paths, list):
                        continue
                    label = normalize_class_label(raw_class)
                    if not label:
                        continue
                    for rel_path in rel_paths:
                        path = stage5_dir / rel_path
                        if path.is_file():
                            class_to_paths.setdefault(label, []).append(path)
                if class_to_paths:
                    return class_to_paths
        except Exception as e:
            print(f"Warning: Failed to read {meta_path}: {e}", file=sys.stderr)

    # Fallback: scan subdirs
    for entry in sorted(stage5_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in ("intermediate", "__pycache__", ".git"):
            continue
        label = normalize_class_label(entry.name)
        if not label:
            continue
        files = sorted(
            p for p in entry.iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not files:
            continue
        class_to_paths[label] = files

    return class_to_paths


def _build_prompt(class_labels: List[str]) -> str:
    classes_str = ", ".join(c.upper() for c in class_labels)
    format_lines = "\n".join(f"{c.upper()}: [visual features]" for c in class_labels)
    return PROMPT_TEMPLATE.format(classes=classes_str, format_lines=format_lines)


def _build_message_content(class_to_paths: Dict[str, List[Path]], max_per_class: int) -> List[dict]:
    content: List[dict] = []
    class_labels = list(class_to_paths.keys())
    content.append({
        "type": "text",
        "text": _build_prompt(class_labels),
    })

    for label in class_labels:
        paths = class_to_paths[label]
        if max_per_class > 0:
            paths = paths[:max_per_class]
        content.append({
            "type": "text",
            "text": f"Class {label.upper()} examples:",
        })
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to open {path}: {e}", file=sys.stderr)
                continue
            b64 = encode_image_base64(img, resize=True)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    content.append({
        "type": "text",
        "text": "Now provide the class descriptions in the exact format requested above.",
    })

    return content


def _resolve_api_settings(
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
) -> Tuple[str, str]:
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vllm":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return vllm_url.rstrip("/"), resolved_key

    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
    return openrouter_url.rstrip("/"), resolved_key


def _normalize_model_name_for_backend(model: str, backend: str, use_vertex: bool) -> str:
    backend = (backend or DEFAULT_BACKEND).lower()
    if (backend == "vertex" or use_vertex) and model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def _extract_image_from_data_url(url: str) -> Image.Image:
    if not url.startswith("data:image/"):
        raise ValueError("Gemini path expects inline data URL images.")
    try:
        _, b64_part = url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Malformed data URL for image content.") from exc
    raw = base64.b64decode(b64_part)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _messages_to_gemini_parts(messages: List[dict]) -> List[object]:
    parts: List[object] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if isinstance(url, str):
                            parts.append(_extract_image_from_data_url(url))
        elif isinstance(content, str) and content.strip():
            parts.append(content)
    return parts


def _get_gemini_client(
    *,
    use_vertex: bool,
    api_key: Optional[str],
    credentials_path: Optional[str],
    location: str,
):
    global _GEMINI_CLIENT, _GEMINI_CLIENT_CACHE_KEY
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for Gemini/Vertex backends. Install with `pip install google-genai`."
        ) from exc

    resolved_location = location or DEFAULT_VERTEX_LOCATION
    resolved_creds = str(Path(credentials_path).expanduser().resolve()) if credentials_path else None
    cache_key = (bool(use_vertex), api_key, resolved_creds, resolved_location)
    with _GEMINI_CLIENT_LOCK:
        if _GEMINI_CLIENT is not None and _GEMINI_CLIENT_CACHE_KEY == cache_key:
            return _GEMINI_CLIENT

        if use_vertex:
            if credentials_path:
                creds_path = Path(credentials_path).expanduser()
                if creds_path.exists():
                    with creds_path.open("r", encoding="utf-8") as f:
                        creds = json.load(f)
                    project_id = creds.get("project_id")
                    if project_id:
                        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())
                else:
                    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    if env_creds:
                        print(
                            f"Warning: Vertex credentials not found at {credentials_path}; "
                            "using GOOGLE_APPLICATION_CREDENTIALS from environment",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"Warning: Vertex credentials not found at {credentials_path}; "
                            "falling back to ambient Google ADC",
                            file=sys.stderr,
                        )
            os.environ["GOOGLE_CLOUD_LOCATION"] = resolved_location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            client = genai.Client()
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Gemini backend without Vertex requires GEMINI_API_KEY/GOOGLE_API_KEY or --api-key."
                )
            # Ensure Vertex mode is not accidentally inherited from env.
            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            client = genai.Client(api_key=resolved_key)

        _GEMINI_CLIENT = client
        _GEMINI_CLIENT_CACHE_KEY = cache_key
        return _GEMINI_CLIENT


def _call_gemini(
    *,
    messages: List[dict],
    model: str,
    backend: str,
    use_vertex: bool,
    api_key: Optional[str],
    credentials_path: Optional[str],
    location: str,
    max_tokens: int,
) -> str:
    from google.genai import types

    gemini_parts = _messages_to_gemini_parts(messages)
    if not gemini_parts:
        raise ValueError("No prompt/image content available for Gemini request.")

    client = _get_gemini_client(
        use_vertex=use_vertex,
        api_key=api_key,
        credentials_path=credentials_path,
        location=location,
    )
    model_name = _normalize_model_name_for_backend(model, backend, use_vertex=use_vertex)
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_tokens if max_tokens else None,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=gemini_parts,
        config=config,
    )
    return (response.text or "").strip()


def _call_vlm(
    messages: List[dict],
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    timeout: int,
    max_tokens: int,
    gemini_use_vertex: bool,
    gemini_credentials: Optional[str],
    gemini_location: str,
) -> str:
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend in {"vertex", "gemini"}:
        effective_use_vertex = True if backend == "vertex" else bool(gemini_use_vertex)
        return _call_gemini(
            messages=messages,
            model=model,
            backend=backend,
            use_vertex=effective_use_vertex,
            api_key=api_key,
            credentials_path=gemini_credentials,
            location=gemini_location,
            max_tokens=max_tokens,
        )

    base_url, resolved_key = _resolve_api_settings(
        backend=backend,
        openrouter_url=openrouter_url,
        vllm_url=vllm_url,
        api_key=api_key,
    )

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    if (backend or DEFAULT_BACKEND).lower() == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/wsi-agents"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    import requests
    resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content", "")


def _parse_descriptions(response: str, expected_classes: List[str]) -> Dict[str, str]:
    import re

    class_pattern = "|".join(re.escape(c.upper()) for c in expected_classes)
    pattern = rf"({class_pattern})\s*:\s*(.+?)(?=(?:{class_pattern})\s*:|\Z)"
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

    descriptions: Dict[str, str] = {}
    for class_name, description in matches:
        key = normalize_class_label(class_name)
        if key:
            descriptions[key] = description.strip()
    return descriptions


def process_stage5_dir(
    stage5_dir: Path,
    output_path: Path,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    model: str,
    max_per_class: int,
    timeout: int,
    max_tokens: int,
    force: bool,
    gemini_use_vertex: bool,
    gemini_credentials: Optional[str],
    gemini_location: str,
) -> bool:
    if output_path.exists() and not force:
        print(f"  Skipping (exists): {output_path.name}")
        return False

    class_to_paths = _load_stage5_output(stage5_dir)
    if not class_to_paths:
        print(f"  ERROR: No class patches found in {stage5_dir}", file=sys.stderr)
        return False

    class_labels = sorted(class_to_paths.keys())
    print(f"  Classes: {', '.join(class_labels)}")

    content = _build_message_content(class_to_paths, max_per_class)
    messages = [{"role": "user", "content": content}]

    try:
        response = _call_vlm(
            messages=messages,
            model=model,
            backend=backend,
            openrouter_url=openrouter_url,
            vllm_url=vllm_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            gemini_use_vertex=gemini_use_vertex,
            gemini_credentials=gemini_credentials,
            gemini_location=gemini_location,
        )
    except Exception as e:
        print(f"  ERROR: {backend} call failed: {e}", file=sys.stderr)
        return False

    descriptions = _parse_descriptions(response, class_labels)
    if not descriptions:
        print("  WARNING: No descriptions parsed from response", file=sys.stderr)

    examples_used = {
        label: [
            str(path.relative_to(stage5_dir))
            for path in class_to_paths[label][:max_per_class if max_per_class > 0 else None]
        ]
        for label in class_labels
    }

    output = {
        "descriptions": descriptions,
        "examples": examples_used,
        "generated_at": datetime.now().isoformat(),
        "backend": backend,
        "model": model,
        "prompt_version": "stage5_v1",
        "stage5_run": str(stage5_dir),
        "max_per_class": max_per_class,
        "raw_response": response,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate class descriptions from Stage 5 ICL patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage5-dir",
        action="append",
        help="Stage 5 run directory (can be specified multiple times)",
    )
    parser.add_argument(
        "--stage5-list",
        type=str,
        default=None,
        help="Text file with Stage 5 run dirs (one per line)",
    )
    parser.add_argument(
        "--stage5-csv",
        type=str,
        default=None,
        help="CSV file containing a 'stage5_dir' column (e.g., stage_6_test.csv)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="class_descriptions.json",
        help="Output filename placed inside each stage5 dir (default: class_descriptions.json)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Explicit output path (only allowed with a single --stage5-dir)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=3,
        help="Max examples per class to include (default: 3; use 0 for all)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openrouter", "vllm", "gemini", "vertex"],
        default=DEFAULT_BACKEND,
        help=f"VLM backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--openrouter-url",
        type=str,
        default=OPENROUTER_BASE_URL,
        help=f"OpenRouter-compatible base URL (default: {OPENROUTER_BASE_URL})",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_BASE_URL,
        help=f"Local vLLM OpenAI-compatible base URL (default: {DEFAULT_VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model id served by selected backend (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Max tokens for response (default: 400)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Optional API key override "
            "(OpenRouter requires key; vLLM typically does not; Gemini non-Vertex uses this as API key)."
        ),
    )
    parser.add_argument("--gemini-use-vertex", dest="gemini_use_vertex", action="store_true")
    parser.add_argument("--gemini-no-vertex", dest="gemini_use_vertex", action="store_false")
    parser.add_argument(
        "--gemini-credentials",
        type=str,
        default=DEFAULT_VERTEX_CREDENTIALS,
        help=(
            "Optional Vertex service account JSON path "
            "(default: unset; falls back to GOOGLE_APPLICATION_CREDENTIALS if set)."
        ),
    )
    parser.add_argument(
        "--gemini-location",
        type=str,
        default=DEFAULT_VERTEX_LOCATION,
        help=f"Vertex location (default: {DEFAULT_VERTEX_LOCATION}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output file exists",
    )

    parser.set_defaults(gemini_use_vertex=True)
    args = parser.parse_args()

    # Gather stage5 dirs from all sources
    stage5_dirs = []
    if args.stage5_dir:
        stage5_dirs.extend(args.stage5_dir)
    if args.stage5_list:
        list_path = Path(args.stage5_list)
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                stage5_dirs.append(line)
    if args.stage5_csv:
        import csv as _csv
        csv_path = Path(args.stage5_csv)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or "stage5_dir" not in reader.fieldnames:
                print("Error: CSV must contain a 'stage5_dir' column", file=sys.stderr)
                return 1
            for row in reader:
                val = (row.get("stage5_dir") or "").strip()
                if val:
                    stage5_dirs.append(val)

    if not stage5_dirs:
        print("Error: must provide --stage5-dir, --stage5-list, or --stage5-csv", file=sys.stderr)
        return 1

    if args.output_path and len(stage5_dirs) != 1:
        print("Error: --output-path requires exactly one --stage5-dir", file=sys.stderr)
        return 1

    if args.backend == "openrouter":
        openrouter_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not openrouter_key:
            print("ERROR: OpenRouter backend requires API key (set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key)", file=sys.stderr)
            return 1
    if args.backend == "vertex" and not args.gemini_use_vertex:
        print("Warning: --backend vertex implies --gemini-use-vertex; ignoring --gemini-no-vertex", file=sys.stderr)

    success = 0
    for dir_path in stage5_dirs:
        stage5_dir = Path(dir_path).resolve()
        if not stage5_dir.is_dir():
            print(f"ERROR: Not a directory: {stage5_dir}", file=sys.stderr)
            continue

        if args.output_path:
            output_path = Path(args.output_path).resolve()
        else:
            output_path = stage5_dir / args.output_name

        print(f"Processing: {stage5_dir}")
        if process_stage5_dir(
            stage5_dir=stage5_dir,
            output_path=output_path,
            backend=args.backend,
            openrouter_url=args.openrouter_url,
            vllm_url=args.vllm_url,
            api_key=args.api_key,
            model=args.model,
            max_per_class=args.max_per_class,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            force=args.force,
            gemini_use_vertex=args.gemini_use_vertex,
            gemini_credentials=args.gemini_credentials,
            gemini_location=args.gemini_location,
        ):
            success += 1

    if success == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
