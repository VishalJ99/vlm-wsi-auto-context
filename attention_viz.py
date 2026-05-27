#!/usr/bin/env python3
"""Visualize Qwen-VL attention for Stage 6 tissue false positives."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts/stage1_detector_oracle/stage6_crop_true_false_positive.txt"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RESAMPLING = getattr(Image, "Resampling", Image)


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "patch"


def patch_id_from_path(path: Path) -> str:
    parts = path.resolve().parts
    if "regions" in parts:
        idx = parts.index("regions")
        if idx + 1 < len(parts):
            roi = sanitize_id(parts[idx + 1])
            case_slug = sanitize_id(parts[idx - 1]) if idx >= 1 else "case"
            return f"{case_slug}_roi{roi}"
    return sanitize_id(path.stem)


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    prompt_file = Path(args.prompt_file)
    if not prompt_file.is_file():
        raise FileNotFoundError(
            f"prompt file does not exist: {prompt_file}. "
            "Pass --prompt-file or --prompt with the byte-identical Stage 6 prompt."
        )
    return prompt_file.read_text()


def load_case_images(case_json: Path, image_field: str) -> list[Path]:
    data = json.loads(case_json.read_text())
    images: list[Path] = []
    for region in data.get("regions", []):
        image_path = region.get(image_field)
        if not image_path:
            raise KeyError(f"region {region.get('region_index')} lacks field {image_field!r}")
        images.append(Path(image_path))
    if not images:
        raise ValueError(f"no regions found in {case_json}")
    return images


def otsu_tissue_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(gray, dtype=bool)
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros_like(denom)
    valid = denom > 0
    sigma_b[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    threshold = int(np.argmax(sigma_b))
    return gray < threshold


def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int], resample: int = RESAMPLING.BILINEAR) -> np.ndarray:
    heatmap = normalize_array(heatmap)
    img = Image.fromarray(np.uint8(np.round(heatmap * 255.0)), mode="L")
    return np.asarray(img.resize(size, resample=resample), dtype=np.float32) / 255.0


def token_id_for(tokenizer: Any, candidates: list[str], label: str) -> int:
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise RuntimeError(f"could not find a single-token {label} ID from candidates {candidates}")


def find_vision_span(input_ids: Any, tokenizer: Any) -> tuple[np.ndarray, int, int]:
    ids = input_ids.detach().cpu().numpy().tolist()
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if start_id is None or end_id is None or start_id < 0 or end_id < 0:
        raise RuntimeError("tokenizer does not expose <|vision_start|> / <|vision_end|> token IDs")
    starts = [i for i, tok in enumerate(ids) if tok == start_id]
    ends = [i for i, tok in enumerate(ids) if tok == end_id]
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError(f"expected one vision span, found starts={starts}, ends={ends}")
    start, end = starts[0], ends[0]
    if end <= start + 1:
        raise RuntimeError(f"empty vision span: start={start}, end={end}")
    return np.arange(start + 1, end, dtype=np.int64), start, end


def extract_grid(inputs: dict[str, Any], n_vision_tokens: int) -> tuple[int, int, int, int, int]:
    if "image_grid_thw" not in inputs:
        raise RuntimeError("processor output lacks image_grid_thw")
    grid = inputs["image_grid_thw"]
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid)
    if grid.ndim == 2:
        grid = grid[0]
    if grid.size != 3:
        raise RuntimeError(f"unexpected image_grid_thw shape/value: {grid!r}")
    t, raw_h_grid, raw_w_grid = [int(x) for x in grid.tolist()]
    if t != 1:
        raise RuntimeError(f"expected one image frame/token group, got image_grid_thw={grid.tolist()}")
    raw_tokens = raw_h_grid * raw_w_grid
    if raw_tokens == n_vision_tokens:
        return t, raw_h_grid, raw_w_grid, raw_h_grid, raw_w_grid

    if raw_tokens % n_vision_tokens != 0:
        raise RuntimeError(
            "vision span length does not match image_grid_thw: "
            f"raw_H_grid*raw_W_grid={raw_tokens}, vision_tokens={n_vision_tokens}. "
            "Cannot infer an integer spatial merge size."
        )
    # Qwen3-VL reports the raw ViT grid; the language sequence contains spatially merged vision tokens.
    merge_area = raw_tokens // n_vision_tokens
    merge_size = int(round(math.sqrt(merge_area)))
    if merge_size * merge_size != merge_area:
        raise RuntimeError(
            "vision span length implies non-square spatial merge: "
            f"merge_area={merge_area}, raw_grid=({raw_h_grid}, {raw_w_grid}), vision_tokens={n_vision_tokens}"
        )
    if raw_h_grid % merge_size != 0 or raw_w_grid % merge_size != 0:
        raise RuntimeError(
            "raw image grid is not divisible by inferred spatial merge size: "
            f"raw_grid=({raw_h_grid}, {raw_w_grid}), merge_size={merge_size}"
        )
    h_grid = raw_h_grid // merge_size
    w_grid = raw_w_grid // merge_size
    if h_grid * w_grid != n_vision_tokens:
        raise RuntimeError(
            "merged image grid still does not match vision token span: "
            f"merged_grid=({h_grid}, {w_grid}), vision_tokens={n_vision_tokens}"
        )
    return t, h_grid, w_grid, raw_h_grid, raw_w_grid


def attention_range_mean(attentions: tuple[Any, ...], q_pos: int, vision_idx: np.ndarray, start: int, end: int) -> np.ndarray:
    if start >= end:
        raise RuntimeError(f"empty attention layer range [{start}, {end})")
    vectors = []
    for layer in range(start, end):
        attn = attentions[layer]
        if attn is None:
            raise RuntimeError(f"attention tensor is None at layer {layer}; use attn_implementation='eager'")
        idx = attn.new_tensor(vision_idx).long()
        slice_v = attn[0, :, q_pos, idx]
        vectors.append(slice_v.mean(dim=0).detach().float().cpu().numpy())
    vec = np.stack(vectors, axis=0).mean(axis=0)
    total = float(vec.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"attention over vision tokens has invalid sum: {total}")
    return vec / total


def head_attention(attentions: tuple[Any, ...], layer: int, q_pos: int, vision_idx: np.ndarray) -> np.ndarray:
    attn = attentions[layer]
    if attn is None:
        raise RuntimeError(f"attention tensor is None at layer {layer}; use attn_implementation='eager'")
    idx = attn.new_tensor(vision_idx).long()
    heads = attn[0, :, q_pos, idx].detach().float().cpu().numpy()
    denom = heads.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0):
        raise RuntimeError(f"invalid per-head attention sums at layer {layer}")
    return heads / denom


def entropy_normalized(vec: np.ndarray) -> float:
    vec = np.asarray(vec, dtype=np.float64)
    vec = vec[vec > 0]
    if len(vec) <= 1:
        return 0.0
    return float(-(vec * np.log(vec)).sum() / math.log(len(vec)))


def iou_top_attention_with_otsu(vec: np.ndarray, grid_shape: tuple[int, int], otsu_mask: np.ndarray) -> float:
    heat = vec.reshape(grid_shape)
    threshold = float(np.percentile(heat, 80.0))
    top = heat >= threshold
    top_img = Image.fromarray(np.uint8(top) * 255, mode="L").resize(
        (otsu_mask.shape[1], otsu_mask.shape[0]),
        resample=RESAMPLING.NEAREST,
    )
    top_full = np.asarray(top_img) > 0
    inter = np.logical_and(top_full, otsu_mask).sum()
    union = np.logical_or(top_full, otsu_mask).sum()
    return float(inter / union) if union else 0.0


def find_gradcam_module(model: Any) -> Any:
    candidates = [
        "visual.merger",
        "visual.patch_merger",
        "visual.patch_embed",
        "model.visual.merger",
        "model.visual.patch_merger",
        "model.visual.patch_embed",
    ]
    modules = dict(model.named_modules())
    for name in candidates:
        module = modules.get(name)
        if module is not None:
            return module
    for name, module in modules.items():
        lowered = name.lower()
        if "visual" in lowered and ("merger" in lowered or "patch_embed" in lowered):
            return module
    raise RuntimeError("could not find a ViT patch/merger module for GradCAM")


def activation_to_tokens(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value


def gradcam_from_activation(activation: Any, n_tokens: int) -> np.ndarray:
    grad = activation.grad
    if grad is None:
        raise RuntimeError("GradCAM hook activation has no gradient")
    act = activation.detach().float()
    grad = grad.detach().float()
    while act.ndim > 2 and act.shape[0] == 1:
        act = act.squeeze(0)
        grad = grad.squeeze(0)
    if act.ndim != 2:
        act = act.reshape(-1, act.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1])
    if act.shape[0] != n_tokens:
        if act.shape[0] < n_tokens:
            raise RuntimeError(f"GradCAM activation has {act.shape[0]} tokens, expected {n_tokens}")
        act = act[-n_tokens:]
        grad = grad[-n_tokens:]
    weights = grad.mean(dim=0)
    cam = (act * weights).sum(dim=-1)
    cam = cam.clamp(min=0).cpu().numpy()
    return normalize_array(cam)


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(image_path)},
            ],
        }
    ]


def prepare_inputs(processor: Any, image: Image.Image, image_path: Path, prompt: str) -> dict[str, Any]:
    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        return processor(text=[text], images=[image], return_tensors="pt")
    except TypeError:
        return processor(text=text, images=image, return_tensors="pt")


def move_inputs_to_device(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def overlay_on_image(image: Image.Image, heatmap: np.ndarray, cmap_name: str = "inferno", alpha: float = 0.5) -> np.ndarray:
    import matplotlib.pyplot as plt

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = resize_heatmap(heatmap, image.size)
    colored = plt.get_cmap(cmap_name)(heat)[..., :3]
    return np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)


def otsu_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 1] = 1.0
    mask_f = mask.astype(np.float32)[..., None]
    return np.clip(base * (1.0 - alpha * mask_f) + color * (alpha * mask_f), 0.0, 1.0)


def save_heads_figure(
    image: Image.Image,
    heads: np.ndarray,
    grid_shape: tuple[int, int],
    out_path: Path,
    layer: int,
) -> None:
    import matplotlib.pyplot as plt

    n_heads = heads.shape[0]
    ncols = min(8, int(math.ceil(math.sqrt(n_heads))))
    nrows = int(math.ceil(n_heads / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False)
    vmax = float(np.max(heads)) if heads.size else 1.0
    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx >= n_heads:
            continue
        heat = heads[idx].reshape(grid_shape)
        base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        heat_up = resize_heatmap(heat / vmax if vmax > 0 else heat, image.size)
        colored = plt.get_cmap("inferno")(heat_up)[..., :3]
        ax.imshow(np.clip(0.5 * base + 0.5 * colored, 0.0, 1.0))
        ax.set_title(f"h{idx}", fontsize=8)
    fig.suptitle(f"Layer {layer} per-head attention to vision tokens", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_main_figure(
    image: Image.Image,
    otsu_mask: np.ndarray,
    early_heat: np.ndarray,
    mid_heat: np.ndarray,
    late_heat: np.ndarray,
    gradcam_heat: np.ndarray,
    heads_path: Path,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    panels = [
        (np.asarray(image.convert("RGB")), "original"),
        (otsu_overlay(image, otsu_mask), "Otsu tissue mask"),
        (overlay_on_image(image, mid_heat), "mid-layer mean attention"),
        (overlay_on_image(image, early_heat), "early-layer mean attention"),
        (overlay_on_image(image, late_heat), "late-layer mean attention"),
        (overlay_on_image(image, gradcam_heat), "GradCAM Yes-No"),
    ]
    for ax, (arr, label) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(label)
        ax.axis("off")
    heads_ax = fig.add_subplot(grid[2, :])
    heads_img = Image.open(heads_path).convert("RGB")
    heads_ax.imshow(heads_img)
    heads_ax.set_title("per-head small multiples")
    heads_ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_reproduction(out_dir: Path, args: argparse.Namespace, images: list[Path], prompt_file: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_images = " ".join(str(p) for p in images)
    prompt_part = f" --prompt-file {prompt_file}" if prompt_file else " --prompt '<inline prompt omitted>'"
    text = f"""\
    VLM attention visualization run
    Created: {datetime.now(timezone.utc).isoformat()}
    Working directory: {Path.cwd()}
    Script: {Path(__file__).resolve()}
    Model: {args.model}
    Prompt file: {prompt_file if prompt_file else '<inline prompt>'}
    Case JSON: {args.case_json or '<none>'}
    Case image field: {args.case_image_field}
    Output directory: {out_dir.resolve()}

    Re-run one-image/pair mode:
      python attention_viz.py{prompt_part} --model {args.model} --output-dir {out_dir} {cmd_images}

    Re-run case mode:
      python attention_viz.py{prompt_part} --model {args.model} --output-dir {out_dir} --case-json {args.case_json or '<case_json>'}
    """
    (out_dir / "reproduction.txt").write_text(textwrap.dedent(text))


def load_model_and_processor(model_name: str):
    local_hf_cache = Path("/data2/vj724/hf_cache")
    if "HF_HOME" not in os.environ and local_hf_cache.exists():
        os.environ["HF_HOME"] = str(local_hf_cache)
        os.environ.setdefault("HF_HUB_CACHE", str(local_hf_cache / "hub"))

    import torch
    from transformers import AutoProcessor

    model_errors: list[str] = []
    model_cls = None
    for class_name in [
        "AutoModelForImageTextToText",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq",
    ]:
        try:
            module = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(module, class_name)
            break
        except Exception as exc:
            model_errors.append(f"{class_name}: {exc}")
    if model_cls is None:
        raise RuntimeError("could not import a compatible Transformers VLM class:\n" + "\n".join(model_errors))

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_attentions=True,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.output_attentions = True
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    except Exception:
        pass
    model.eval()
    return model, processor, torch


def analyze_image(
    image_path: Path,
    prompt: str,
    model: Any,
    processor: Any,
    torch: Any,
    out_dir: Path,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    patch_id = patch_id_from_path(image_path)
    inputs_cpu = prepare_inputs(processor, image, image_path, prompt)
    inputs = move_inputs_to_device(inputs_cpu, model.device)
    tokenizer = processor.tokenizer

    input_ids = inputs["input_ids"][0]
    vision_idx, vision_start, vision_end = find_vision_span(input_ids, tokenizer)
    q_pos = int(input_ids.shape[0] - 1)
    _, h_grid, w_grid, raw_h_grid, raw_w_grid = extract_grid(inputs, len(vision_idx))
    grid_shape = (h_grid, w_grid)

    yes_id = token_id_for(tokenizer, ["Yes", " yes", "yes", " YES"], "yes")
    no_id = token_id_for(tokenizer, ["No", " no", "no", " NO"], "no")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    attentions = outputs.attentions
    if attentions is None or not isinstance(attentions, tuple) or not attentions:
        raise RuntimeError("model returned no attentions; use attn_implementation='eager'")
    logits = outputs.logits
    gap = float((logits[0, -1, yes_id] - logits[0, -1, no_id]).detach().float().cpu())
    next_id = int(logits[0, -1].argmax().detach().cpu())
    answer = tokenizer.decode([next_id], skip_special_tokens=True).strip()
    if not answer:
        answer = tokenizer.convert_ids_to_tokens(next_id)

    depth = len(attentions)
    ranges = {
        "early": (0, depth // 3),
        "mid": (depth // 3, 2 * depth // 3),
        "late": (2 * depth // 3, depth),
    }
    early_vec = attention_range_mean(attentions, q_pos, vision_idx, *ranges["early"])
    mid_vec = attention_range_mean(attentions, q_pos, vision_idx, *ranges["mid"])
    late_vec = attention_range_mean(attentions, q_pos, vision_idx, *ranges["late"])
    mid_layer = depth // 2
    heads = head_attention(attentions, mid_layer, q_pos, vision_idx)
    del outputs, attentions, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    captured: dict[str, Any] = {}
    gradcam_module = find_gradcam_module(model)
    for param in model.parameters():
        param.requires_grad_(False)

    def hook(_module, _inp, output):
        activation = activation_to_tokens(output)
        if not activation.requires_grad:
            activation.requires_grad_(True)
        activation.retain_grad()
        captured["activation"] = activation

    handle = gradcam_module.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        grad_outputs = model(**inputs, output_attentions=False, use_cache=False)
        grad_logits = grad_outputs.logits
        loss = grad_logits[0, -1, yes_id] - grad_logits[0, -1, no_id]
        loss.backward()
        if "activation" not in captured:
            raise RuntimeError("GradCAM hook did not capture an activation")
        gradcam_vec = gradcam_from_activation(captured["activation"], len(vision_idx))
    finally:
        handle.remove()
    del grad_outputs, grad_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    early_heat = early_vec.reshape(grid_shape)
    mid_heat = mid_vec.reshape(grid_shape)
    late_heat = late_vec.reshape(grid_shape)
    gradcam_heat = gradcam_vec.reshape(grid_shape)
    otsu_mask = otsu_tissue_mask(image)

    out_dir.mkdir(parents=True, exist_ok=True)
    heads_path = out_dir / f"{patch_id}_layer{mid_layer}_heads.png"
    png_path = out_dir / f"{patch_id}_attention.png"
    json_path = out_dir / f"{patch_id}_summary.json"
    save_heads_figure(image, heads, grid_shape, heads_path, mid_layer)
    save_main_figure(
        image=image,
        otsu_mask=otsu_mask,
        early_heat=early_heat,
        mid_heat=mid_heat,
        late_heat=late_heat,
        gradcam_heat=gradcam_heat,
        heads_path=heads_path,
        out_path=png_path,
        title=f"{patch_id} | answer={answer} | logit_gap(Yes-No)={gap:+.2f}",
    )

    summary = {
        "patch_id": patch_id,
        "image_path": str(image_path),
        "model_answer": answer,
        "yes_no_logit_gap": gap,
        "mid_layer_attention_entropy_normalized": entropy_normalized(mid_vec),
        "iou_top20pct_attention_with_otsu": iou_top_attention_with_otsu(mid_vec, grid_shape, otsu_mask),
        "model_depth": depth,
        "mid_layer": mid_layer,
        "vision_start_index": vision_start,
        "vision_end_index": vision_end,
        "query_position": q_pos,
        "image_grid_thw": [1, raw_h_grid, raw_w_grid],
        "attention_grid_thw": [1, h_grid, w_grid],
        "outputs": {
            "attention_png": str(png_path),
            "heads_png": str(heads_path),
            "summary_json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen-VL attention maps for Stage 6 tissue yes/no patch decisions.",
    )
    parser.add_argument("images", nargs="*", type=Path, help="FP image path followed by optional TP image path(s)")
    parser.add_argument("--case-json", type=Path, help="case_sampler_input.json containing regions to run")
    parser.add_argument(
        "--case-image-field",
        default="vlm_image_path",
        choices=["vlm_image_path", "crop_path", "selected_overlay_path"],
        help="image field to use from --case-json; default is the actual VLM input image",
    )
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="byte-identical Stage 6 prompt file")
    parser.add_argument("--prompt", help="inline byte-identical Stage 6 prompt; overrides --prompt-file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--dry-run", action="store_true", help="resolve inputs and write reproduction.txt without loading the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list(args.images)
    if args.case_json:
        images.extend(load_case_images(args.case_json, args.case_image_field))
    if not images:
        raise SystemExit("provide at least one image path or --case-json")
    for image_path in images:
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")

    prompt = load_prompt(args)
    prompt_file = None if args.prompt is not None else Path(args.prompt_file)
    write_reproduction(args.output_dir, args, images, prompt_file)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "model": args.model,
            "prompt_file": str(prompt_file) if prompt_file else None,
            "images": [str(path) for path in images],
            "output_dir": str(args.output_dir),
        }, indent=2))
        return
    model, processor, torch = load_model_and_processor(args.model)

    summaries = []
    for image_path in images:
        summaries.append(analyze_image(image_path, prompt, model, processor, torch, args.output_dir))
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
