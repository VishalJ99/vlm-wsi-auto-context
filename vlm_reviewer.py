#!/usr/bin/env python3
# ABOUTME: Minimal VLM reviewer for segmentation quality using crop + mask (or overlay).
# ABOUTME: Builds a clean green alpha overlay and sends both source + overlay to the configured backend.

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from run_vlm_bbox_inference import (
    DEFAULT_OPENROUTER_REFERER,
    DEFAULT_OPENROUTER_URL,
    DEFAULT_VLLM_URL,
    GeminiRunner,
    OpenRouterRunner,
    VLLMRunner,
    encode_image_base64,
)
from utils.model_pricing import estimate_review_cost_usd


DEFAULT_PROMPT_FILE = "prompts/objective_reviewer.txt"
DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_OUTPUT_ROOT = "auto_reviews"


def load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt:
        return prompt.strip()
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    return Path(DEFAULT_PROMPT_FILE).read_text().strip()


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(mask_path: str, size: tuple[int, int], threshold: int) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, resample=Image.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8) > int(threshold)
    return mask


def build_green_overlay(
    crop_img: Image.Image,
    mask_path: str,
    alpha: float = 0.5,
    threshold: int = 0,
) -> Image.Image:
    """Apply a clean green foreground overlay where mask is positive."""
    crop = np.asarray(crop_img.convert("RGB"), dtype=np.float32)
    mask = _load_mask(mask_path, crop_img.size, threshold=threshold)

    overlay = crop.copy()
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * green
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


def parse_json_response(text: str) -> Optional[dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def infer_case_and_bbox(crop_path: str, mask_path: Optional[str], overlay_path: Optional[str]) -> tuple[str, str]:
    """Infer case + bbox names from common pipeline paths."""
    for source in [crop_path, mask_path, overlay_path]:
        if not source:
            continue
        parts = Path(source).resolve().parts
        if "bboxes" in parts:
            idx = parts.index("bboxes")
            if idx + 1 < len(parts):
                bbox_name = parts[idx + 1]
            else:
                bbox_name = "unknown_bbox"
            if idx >= 2:
                case_name = parts[idx - 2]
            else:
                case_name = "unknown_case"
            return case_name, bbox_name
    return "unknown_case", "unknown_bbox"


def build_output_dir(output_root: str, case_name: str, bbox_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / case_name / bbox_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parts(prompt_text: str, crop_img: Image.Image, overlay_img: Image.Image) -> list:
    return [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "Input 1: Source image"},
        {"type": "image", "image": {"pil": crop_img, "b64": encode_image_base64(crop_img, resize=False)}},
        {"type": "text", "text": "Input 2: Segmentation overlay"},
        {"type": "image", "image": {"pil": overlay_img, "b64": encode_image_base64(overlay_img, resize=False)}},
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal segmentation reviewer: crop + mask/overlay -> VLM response."
    )

    io = parser.add_argument_group("Input")
    io.add_argument("--crop", required=True, help="Path to source crop image (crop.png).")
    io.add_argument(
        "--mask",
        default=None,
        help="Path to binary mask image (mask.png). Used to generate clean green overlay.",
    )
    io.add_argument(
        "--overlay",
        default=None,
        help="Path to existing overlay image. If set, --mask is ignored for inference input.",
    )
    io.add_argument(
        "--save-overlay",
        default=None,
        help="Optional path to save the generated clean overlay image.",
    )
    io.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Overlay opacity for generated overlay (default: 0.5).",
    )
    io.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Foreground threshold for mask (mask > threshold). Default: 0.",
    )
    io.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory for persisted reviews (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    io.add_argument(
        "--case-name",
        default=None,
        help="Optional case name override for output path.",
    )
    io.add_argument(
        "--bbox-name",
        default=None,
        help="Optional bbox name override for output path.",
    )

    prompt = parser.add_argument_group("Prompt")
    prompt.add_argument("--prompt", default=None, help="Inline prompt text.")
    prompt.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt file path (default: {DEFAULT_PROMPT_FILE}).",
    )

    backend = parser.add_argument_group("Backend")
    backend.add_argument(
        "--backend",
        choices=["gemini", "openrouter", "vllm"],
        default="gemini",
        help="Inference backend (default: gemini).",
    )
    backend.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL}).")
    backend.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0).")
    backend.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens (default: 8192).")
    backend.add_argument("--max-retries", type=int, default=3, help="Max retries (default: 3).")
    backend.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for openrouter/vllm backends (default: 120).",
    )

    gemini = parser.add_argument_group("Gemini")
    gemini.add_argument("--thinking-level", default="High", help="Gemini thinking level: Low/High (default: High).")
    gemini.add_argument("--include-thoughts", action="store_true", default=False, help="Request thought summaries.")
    gemini.add_argument(
        "--no-include-thoughts",
        dest="include_thoughts",
        action="store_false",
        help="Disable thought summaries.",
    )
    gemini.add_argument("--gemini-use-vertex", dest="gemini_use_vertex", action="store_true")
    gemini.add_argument("--gemini-no-vertex", dest="gemini_use_vertex", action="store_false")
    gemini.add_argument(
        "--gemini-credentials",
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    gemini.add_argument("--gemini-location", default="global", help="Vertex location (default: global).")

    openrouter = parser.add_argument_group("OpenRouter")
    openrouter.add_argument("--openrouter-api-key", default=None, help="OpenRouter API key (optional if env set).")
    openrouter.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL, help=f"OpenRouter URL (default: {DEFAULT_OPENROUTER_URL}).")
    openrouter.add_argument(
        "--openrouter-referer",
        default=DEFAULT_OPENROUTER_REFERER,
        help=f"OpenRouter referer header (default: {DEFAULT_OPENROUTER_REFERER}).",
    )
    openrouter.add_argument(
        "--reasoning-effort",
        default=None,
        help="OpenRouter reasoning effort: low/medium/high (backend=openrouter).",
    )

    vllm = parser.add_argument_group("vLLM")
    vllm.add_argument("--vllm-url", default=DEFAULT_VLLM_URL, help=f"vLLM URL (default: {DEFAULT_VLLM_URL}).")

    parser.set_defaults(gemini_use_vertex=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1].")

    if not args.overlay and not args.mask:
        raise ValueError("Provide either --overlay or --mask.")

    crop_img = _load_rgb(args.crop)

    if args.overlay:
        overlay_img = _load_rgb(args.overlay)
    else:
        overlay_img = build_green_overlay(
            crop_img=crop_img,
            mask_path=args.mask,
            alpha=float(args.overlay_alpha),
            threshold=int(args.mask_threshold),
        )

    if overlay_img.size != crop_img.size:
        overlay_img = overlay_img.resize(crop_img.size, resample=Image.BICUBIC)

    if args.save_overlay:
        Path(args.save_overlay).parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(args.save_overlay)
        print(f"Saved generated overlay: {args.save_overlay}")

    prompt_text = load_prompt(args.prompt, args.prompt_file)
    parts = build_parts(prompt_text, crop_img, overlay_img)

    if args.backend == "gemini":
        runner = GeminiRunner(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            use_vertex=args.gemini_use_vertex,
            credentials_path=args.gemini_credentials,
            location=args.gemini_location,
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        )
    elif args.backend == "openrouter":
        runner = OpenRouterRunner(
            model=args.model,
            api_key=args.openrouter_api_key,
            url=args.openrouter_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            referer=args.openrouter_referer,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        runner = VLLMRunner(
            model=args.model,
            url=args.vllm_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )

    print("=" * 60)
    print("=== VLM REVIEW INPUTS ===")
    print("=" * 60)
    print(f"crop:    {args.crop}")
    print(f"overlay: {args.overlay if args.overlay else '(generated from mask)'}")
    if args.mask:
        print(f"mask:    {args.mask}")
    print(f"backend: {args.backend}")
    print(f"model:   {args.model}")
    print(f"thinking_level: {args.thinking_level}")
    print(f"include_thoughts: {args.include_thoughts}")
    print()

    print("=" * 60)
    print("=== PROMPT SENT ===")
    print("=" * 60)
    print(prompt_text)
    print()

    t0 = time.time()
    # Prefer run_full to capture usage + finish reason; fallback to run for backends without run_full.
    if hasattr(runner, "run_full"):
        result = runner.run_full(parts)
    else:
        result = {
            "text": runner.run(parts),
            "thoughts": [],
            "usage": {},
            "finish_reason": None,
            "error": None,
            "attempts": 1,
        }
    elapsed = time.time() - t0

    text = (result.get("text") or "").strip()
    thoughts = result.get("thoughts", []) or []
    usage = result.get("usage", {}) or {}
    finish_reason = result.get("finish_reason")
    error = result.get("error")
    attempts = result.get("attempts")
    parsed = parse_json_response(text)
    cost_estimate = estimate_review_cost_usd(args.model, usage)

    inferred_case, inferred_bbox = infer_case_and_bbox(args.crop, args.mask, args.overlay)
    case_name = args.case_name or inferred_case
    bbox_name = args.bbox_name or inferred_bbox
    run_dir = build_output_dir(args.output_root, case_name, bbox_name)

    crop_out = run_dir / "crop.png"
    overlay_out = run_dir / "overlay_green50.png"
    raw_out = run_dir / "raw_response.txt"
    meta_out = run_dir / "metadata.json"

    crop_img.save(crop_out)
    overlay_img.save(overlay_out)
    raw_out.write_text(text + ("\n" if text else ""))

    metadata = {
        "timestamp": run_dir.name,
        "case_name": case_name,
        "bbox_name": bbox_name,
        "inputs": {
            "crop": str(Path(args.crop)),
            "mask": str(Path(args.mask)) if args.mask else None,
            "overlay": str(Path(args.overlay)) if args.overlay else None,
        },
        "outputs": {
            "run_dir": str(run_dir),
            "crop": str(crop_out),
            "overlay_green50": str(overlay_out),
            "raw_response": str(raw_out),
        },
        "prompt": prompt_text,
        "prompt_file": args.prompt_file,
        "backend": args.backend,
        "model": args.model,
        "thinking_level": args.thinking_level,
        "include_thoughts": args.include_thoughts,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        "gemini_use_vertex": args.gemini_use_vertex,
        "gemini_credentials": args.gemini_credentials,
        "gemini_location": args.gemini_location,
        "openrouter_url": args.openrouter_url,
        "openrouter_referer": args.openrouter_referer,
        "reasoning_effort": args.reasoning_effort,
        "vllm_url": args.vllm_url,
        "overlay_alpha": args.overlay_alpha,
        "mask_threshold": args.mask_threshold,
        "elapsed_seconds": elapsed,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "error": error,
        "usage": usage,
        "cost_estimate_usd": cost_estimate,
        "parsed_json": parsed,
        "thoughts": thoughts,
        "git_commit_hash": get_git_commit_hash(),
        "cwd": os.getcwd(),
    }
    meta_out.write_text(json.dumps(metadata, indent=2))

    print("=" * 60)
    print("=== RAW RESPONSE ===")
    print("=" * 60)
    print(text if text else "(empty text response)")
    print()

    if parsed is not None:
        print("=" * 60)
        print("=== PARSED JSON ===")
        print("=" * 60)
        print(json.dumps(parsed, indent=2))
        print()
    else:
        print("Parsed JSON: unavailable")
        print()

    if thoughts:
        print("=" * 60)
        print(f"=== THINKING ({len(thoughts)} part(s)) ===")
        print("=" * 60)
        for i, thought in enumerate(thoughts, start=1):
            if len(thoughts) > 1:
                print(f"--- Thought {i} ---")
            print(thought)
            print()

    if finish_reason or error or attempts is not None:
        print("=" * 60)
        print("=== RESPONSE STATUS ===")
        print("=" * 60)
        if attempts is not None:
            print(f"  Attempts:        {attempts}")
        if finish_reason:
            print(f"  Finish reason:   {finish_reason}")
        if error:
            print(f"  Error:           {error}")
        print()

    if any(v is not None for v in usage.values()):
        print("=" * 60)
        print("=== TOKEN USAGE ===")
        print("=" * 60)
        if usage.get("prompt_tokens") is not None:
            print(f"  Prompt tokens:   {usage['prompt_tokens']}")
        if usage.get("thoughts_tokens") is not None:
            print(f"  Thinking tokens: {usage['thoughts_tokens']}")
        if usage.get("output_tokens") is not None:
            print(f"  Output tokens:   {usage['output_tokens']}")
        if usage.get("total_tokens") is not None:
            print(f"  Total tokens:    {usage['total_tokens']}")
        print()

    if cost_estimate is not None:
        print("=" * 60)
        print("=== COST ESTIMATE (USD) ===")
        print("=" * 60)
        print(f"  Pricing key:     {cost_estimate['pricing_model_key']}")
        print(f"  Input cost:      ${cost_estimate['estimated_input_cost_usd']:.6f}")
        print(f"  Output cost:     ${cost_estimate['estimated_output_cost_usd']:.6f}")
        print(f"  Total cost:      ${cost_estimate['estimated_total_cost_usd']:.6f}")
        print()

    print("=" * 60)
    print("=== OUTPUTS SAVED ===")
    print("=" * 60)
    print(f"run_dir:           {run_dir}")
    print(f"crop:              {crop_out}")
    print(f"overlay_green50:   {overlay_out}")
    print(f"raw_response:      {raw_out}")
    print(f"metadata:          {meta_out}")
    print()

    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
