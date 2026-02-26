# ABOUTME: Shared VLM utilities for image resizing and ICL message construction.
# ABOUTME: Used by VLM backends and inference scripts.

import base64
import io
import json
import math
import os
import re
from typing import List, Optional, Tuple

from PIL import Image


# =============================================================================
# Image Resizing Helpers
# =============================================================================

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 12845056
) -> Tuple[int, int]:
    """
    Smart resize image dimensions based on factor and pixel constraints.

    Ensures dimensions are divisible by factor (32) for optimal Qwen3-VL performance.

    Args:
        height: Original height
        width: Original width
        factor: Alignment factor (dimensions will be multiples of this)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        (new_width, new_height): Resized dimensions aligned to factor
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    return w_bar, h_bar


def encode_image_base64(image_pil: Image.Image, resize: bool = True) -> str:
    """
    Encode PIL image as base64 string, optionally resizing for VLM compatibility.

    Args:
        image_pil: PIL Image to encode
        resize: If True, resize to factor-32 aligned dimensions

    Returns:
        Base64-encoded PNG string
    """
    if resize:
        new_w, new_h = smart_resize(image_pil.height, image_pil.width)
        image_pil = image_pil.resize((new_w, new_h), resample=Image.BICUBIC)

    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# ICL Example Loading
# =============================================================================

CLASS_ALIASES = {
    "foreground": "tissue",
    "fg": "tissue",
    "bg": "background",
    "paraffin": "paraffin_mounting_medium",
    "paraffin_mounting": "paraffin_mounting_medium",
    "paraffin_mounting_medium": "paraffin_mounting_medium",
    "pen": "pen_ink_marks",
    "pen_ink": "pen_ink_marks",
    "pen_ink_mark": "pen_ink_marks",
    "pen_ink_marks": "pen_ink_marks",
}

QUALITY_LABELS = {
    "sharp": "Sharp",
    "somewhat_blurred": "Somewhat Blurred",
    "somewhat blurred": "Somewhat Blurred",
    "blurred": "Somewhat Blurred",
    "slightly_blurred": "Somewhat Blurred",
    "slightly blurred": "Somewhat Blurred",
    "out_of_focus": "Out of Focus",
    "out of focus": "Out of Focus",
    "completely_out_of_focus": "Out of Focus",
    "completely out of focus": "Out of Focus",
    "oof": "Out of Focus",
    "na": "NA",
    "n/a": "NA",
}


def normalize_class_label(label: str) -> str:
    """Normalize class labels to canonical snake_case names."""
    if not label:
        return ""
    lowered = label.strip().lower()
    lowered = re.sub(r"[\s\-\/]+", "_", lowered)
    lowered = re.sub(r"[^a-z0-9_]", "", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return CLASS_ALIASES.get(lowered, lowered)


def normalize_quality_label(label: str) -> str:
    """Normalize quality labels to canonical form."""
    if not label:
        return ""
    lowered = label.strip().lower()
    lowered = re.sub(r"[\s\-\/]+", "_", lowered)
    lowered = re.sub(r"[^a-z0-9_]", "", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return QUALITY_LABELS.get(lowered, "")


def infer_icl_classes(icl_dir: str) -> List[str]:
    """
    Infer available class labels from an ICL directory.

    Supports Stage 5 run directories (metadata.json with 'output')
    or subdir-per-class layouts.
    """
    classes = []
    meta_path = os.path.join(icl_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            output = meta.get("output")
            if isinstance(output, dict):
                classes = [normalize_class_label(k) for k in output.keys()]
                return sorted(set([c for c in classes if c]))
        except Exception:
            pass

    if os.path.isdir(icl_dir):
        for name in sorted(os.listdir(icl_dir)):
            dir_path = os.path.join(icl_dir, name)
            if not os.path.isdir(dir_path):
                continue
            if name in ("intermediate", "__pycache__"):
                continue
            image_files = [
                f for f in os.listdir(dir_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if not image_files:
                continue
            classes.append(normalize_class_label(name))

    return sorted(set([c for c in classes if c]))


def load_icl_examples(
    icl_dir: str,
    max_per_class: int = 4
) -> List[dict]:
    """
    Load labeled examples from directory structure for in-context learning.

    Expected structure (either):
        - Stage 5 run dir with metadata.json and output paths
        - icl_dir/
            tissue/
            background/
            paraffin_mounting_medium/
            pen_ink_marks/

    Args:
        icl_dir: Directory containing foreground/ and background/ subdirs
        max_per_class: Maximum examples to load per class

    Returns:
        List of dicts with 'image_base64' and 'label' keys
    """
    examples: List[dict] = []
    limit = max_per_class if max_per_class and max_per_class > 0 else None

    meta_path = os.path.join(icl_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            output = meta.get("output")
            if isinstance(output, dict):
                for class_name, rel_paths in output.items():
                    label = normalize_class_label(class_name)
                    if not label:
                        continue
                    paths = rel_paths[:limit] if limit else rel_paths
                    for rel_path in paths:
                        fpath = os.path.join(icl_dir, rel_path)
                        try:
                            img = Image.open(fpath).convert("RGB")
                            b64 = encode_image_base64(img, resize=True)
                            examples.append({"image_base64": b64, "label": label})
                        except Exception as e:
                            print(f"Warning: Failed to load ICL example {fpath}: {e}")
                if examples:
                    return examples
        except Exception as e:
            print(f"Warning: Failed to read ICL metadata {meta_path}: {e}")

    if os.path.isdir(icl_dir):
        for class_dir in sorted(os.listdir(icl_dir)):
            dir_path = os.path.join(icl_dir, class_dir)
            if not os.path.isdir(dir_path):
                continue
            if class_dir in ("intermediate", "__pycache__"):
                continue
            image_files = sorted([
                f for f in os.listdir(dir_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if not image_files:
                continue
            label = normalize_class_label(class_dir)
            if not label:
                continue
            for fname in (image_files[:limit] if limit else image_files):
                fpath = os.path.join(dir_path, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    b64 = encode_image_base64(img, resize=True)
                    examples.append({"image_base64": b64, "label": label})
                except Exception as e:
                    print(f"Warning: Failed to load ICL example {fpath}: {e}")

    return examples


# =============================================================================
# ICL Message Building
# =============================================================================

def build_icl_messages(
    test_image_b64: str,
    prompt: str,
    icl_examples: Optional[List[dict]] = None,
    mode: str = "single"
) -> list:
    """
    Build messages array for VLM API with optional ICL examples.

    Args:
        test_image_b64: Base64-encoded test image
        prompt: Classification prompt
        icl_examples: List of {"image_base64": str, "label": str}
        mode: "single", "single-prompt-first", "multi-turn", or "multi-turn-single-prompt"

    Returns:
        List of message dicts for the API (OpenAI chat completions format)
    """
    if mode == "single":
        return _build_single_message(test_image_b64, prompt, icl_examples)
    elif mode == "single-prompt-first":
        return _build_single_prompt_first_message(test_image_b64, prompt, icl_examples)
    elif mode == "multi-turn-single-prompt":
        return _build_multi_turn_single_prompt_messages(test_image_b64, prompt, icl_examples)
    else:  # default to multi-turn
        return _build_multi_turn_messages(test_image_b64, prompt, icl_examples)


def _build_single_message(
    test_image_b64: str,
    prompt: str,
    icl_examples: Optional[List[dict]] = None
) -> list:
    """All examples + test in one user message content array."""
    content = []

    if icl_examples:
        for ex in icl_examples:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ex['image_base64']}"}
            })
            content.append({
                "type": "text",
                "text": _format_icl_answer(ex)
            })

    # Test image + prompt
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}
    })
    if prompt:
        content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def _build_single_prompt_first_message(
    test_image_b64: str,
    prompt: str,
    icl_examples: Optional[List[dict]] = None
) -> list:
    """Single message with prompt only after first image to establish task context."""
    content = []

    if icl_examples:
        for idx, ex in enumerate(icl_examples):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ex['image_base64']}"}
            })
            # Add prompt only after first image
            if idx == 0 and prompt:
                content.append({"type": "text", "text": prompt})
            content.append({
                "type": "text",
                "text": _format_icl_answer(ex)
            })

    # Test image (no prompt - model should continue pattern)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}
    })

    return [{"role": "user", "content": content}]


def _build_multi_turn_messages(
    test_image_b64: str,
    prompt: str,
    icl_examples: Optional[List[dict]] = None
) -> list:
    """Simulated conversation with user->assistant turns for each example."""
    messages = []

    if icl_examples:
        for ex in icl_examples:
            # User shows image and asks question
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex['image_base64']}"}}
            ]
            if prompt:
                user_content.append({"type": "text", "text": prompt})

            messages.append({"role": "user", "content": user_content})
            # Assistant responds with yes/no based on label
            messages.append({
                "role": "assistant",
                "content": _format_icl_answer(ex)
            })

    # Final test image
    test_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}}
    ]
    if prompt:
        test_content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": test_content})

    return messages


def _build_multi_turn_single_prompt_messages(
    test_image_b64: str,
    prompt: str,
    icl_examples: Optional[List[dict]] = None
) -> list:
    """Multi-turn with prompt only in first message. Tests if repeated prompts are redundant."""
    messages = []

    if icl_examples:
        for idx, ex in enumerate(icl_examples):
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex['image_base64']}"}}
            ]
            # Only add prompt to first example
            if idx == 0 and prompt:
                user_content.append({"type": "text", "text": prompt})

            messages.append({"role": "user", "content": user_content})
            messages.append({
                "role": "assistant",
                "content": _format_icl_answer(ex)
            })

    # Test image (no prompt - model should remember from first turn)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}}
        ]
    })

    return messages


def parse_vlm_response(answer: str) -> bool:
    """
    Parse VLM response to boolean classification.

    Args:
        answer: Raw text response from VLM

    Returns:
        True if foreground, False if background
    """
    return "true" in answer.lower()


def _format_icl_answer(example: dict) -> str:
    """Format an ICL assistant response with optional quality."""
    label = example.get("label", "").strip()
    quality = example.get("quality")
    if quality:
        return f"Class: {label}\nQuality: {quality}"
    return label


def parse_vlm_output(
    answer: str,
    allowed_labels: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Parse VLM response into (class_label, quality_label).

    If class_label is not tissue, quality is forced to NA.
    """
    if not answer:
        return ("unknown", "NA")

    text = answer.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    lowered = text.lower()
    lowered_spaced = lowered.replace("_", " ")

    class_label = ""
    quality_label = ""

    # --- Try JSON parsing first ---
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Pass 1: Fuzzy KEY match — any key containing "class" or "quality"
            for key, val in data.items():
                key_lower = key.lower()
                val_str = str(val).strip()
                if not val_str:
                    continue
                if "class" in key_lower and not class_label:
                    class_label = normalize_class_label(val_str)
                elif "quality" in key_lower and not quality_label:
                    quality_label = normalize_quality_label(val_str)

            # Pass 2: Reverse VALUE match — scan all values for recognized labels
            if not class_label or not quality_label:
                known_classes = set(CLASS_ALIASES.values())
                if allowed_labels:
                    known_classes = set(normalize_class_label(c) for c in allowed_labels)
                for key, val in data.items():
                    val_str = str(val).strip()
                    if not val_str:
                        continue
                    if not class_label:
                        normalized = normalize_class_label(val_str)
                        if normalized and normalized in known_classes:
                            class_label = normalized
                    if not quality_label:
                        normalized = normalize_quality_label(val_str)
                        if normalized:
                            quality_label = normalized

            # JSON was valid — if we still couldn't extract class, crash
            if not class_label:
                raise ValueError(
                    f"Valid JSON but could not extract class label. "
                    f"Keys: {list(data.keys())}. Raw response:\n{text}"
                )
    except json.JSONDecodeError:
        pass  # Not JSON — fall through to text parsing

    # --- Existing text parsing (fallback for non-JSON responses) ---
    if not class_label:
        class_match = re.search(r"class\s*:\s*([^\n\r]+)", text, re.IGNORECASE)
        if class_match:
            raw = class_match.group(1).strip()
            raw = re.split(r"[|,;]", raw)[0].strip()
            class_label = normalize_class_label(raw)

    if not class_label and allowed_labels:
        for cand in sorted(allowed_labels, key=len, reverse=True):
            cand_lower = cand.lower()
            if cand_lower in lowered or cand_lower.replace("_", " ") in lowered_spaced:
                class_label = normalize_class_label(cand)
                break

    if not class_label and "true" in lowered and allowed_labels and "tissue" in allowed_labels:
        class_label = "tissue"
    if not class_label and "false" in lowered and allowed_labels and "background" in allowed_labels:
        class_label = "background"

    if not class_label:
        class_label = "unknown"

    if allowed_labels:
        allowed_norm = set([normalize_class_label(c) for c in allowed_labels])
        if class_label not in allowed_norm:
            class_label = "unknown"

    if not quality_label:
        quality_match = re.search(r"quality\s*:\s*([^\n\r]+)", text, re.IGNORECASE)
        if quality_match:
            raw_q = quality_match.group(1).strip()
            raw_q = re.split(r"[|,;]", raw_q)[0].strip()
            quality_label = normalize_quality_label(raw_q)

    if not quality_label:
        # Fallback: try to infer from full text
        quality_label = normalize_quality_label(lowered)

    if class_label != "tissue":
        quality_label = "NA"
    elif not quality_label:
        quality_label = "NA"

    return (class_label, quality_label)
