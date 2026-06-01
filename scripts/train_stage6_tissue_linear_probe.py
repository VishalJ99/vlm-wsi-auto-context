#!/usr/bin/env python3
"""Train a frozen-DINO linear probe on Stage 6 tissue/artifact crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO_ROOT / "runs/detector_pipeline_scale500_v1/non_sv40",
    REPO_ROOT / "runs/detector_pipeline_scale500_v1/sv40_skip_odd",
]
DEFAULT_WORKLIST_DIR = REPO_ROOT / "runs/detector_pipeline_scale500_v1/worklists"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stage6_tissue_classifier/scale500_dinov2small_linear_probe_v1"
STAIN_WORKLISTS = {
    "EVG": "evg_100_wsi_list.txt",
    "H&E": "he_100_wsi_list.txt",
    "JONES": "jones_100_wsi_list.txt",
    "PAS": "pas_100_wsi_list.txt",
    "SV40": "sv40_100_wsi_list.txt",
}


@dataclass(frozen=True)
class CropRecord:
    record_id: str
    run_name: str
    stain: str
    case_id: str
    case_slug: str
    case_display: str
    candidate_order: int
    crop_path: Path
    selected_overlay_path: Path
    thumbnail_path: Path
    wsi_path: str
    tissue_focus_decision: str
    label: int
    raw: dict[str, Any]


class CropDataset(Dataset):
    def __init__(
        self,
        rows: list[CropRecord],
        input_size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_source: str,
        thumbnail_padding_frac: float,
    ):
        self.rows = rows
        self.input_size = input_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.image_source = image_source
        self.thumbnail_padding_frac = thumbnail_padding_frac

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        image = load_record_image(row, self.image_source, self.thumbnail_padding_frac)
        image = pad_square(image, fill=(255, 255, 255))
        image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor, idx


def pad_square(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    out = Image.new("RGB", (side, side), fill)
    out.paste(image, ((side - width) // 2, (side - height) // 2))
    return out


def crop_thumbnail_bbox(row: CropRecord, padding_frac: float) -> Image.Image:
    image = Image.open(row.thumbnail_path).convert("RGB")
    width, height = image.size
    y1, x1, y2, x2 = [float(value) for value in row.raw["box_2d_yxyx_normalized"]]
    left = x1 / 1000.0 * width
    top = y1 / 1000.0 * height
    right = x2 / 1000.0 * width
    bottom = y2 / 1000.0 * height
    box_w = max(1.0, right - left)
    box_h = max(1.0, bottom - top)
    pad_x = box_w * padding_frac
    pad_y = box_h * padding_frac
    left = max(0, int(math.floor(left - pad_x)))
    top = max(0, int(math.floor(top - pad_y)))
    right = min(width, int(math.ceil(right + pad_x)))
    bottom = min(height, int(math.ceil(bottom + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid thumbnail crop for {row.record_id}: {(left, top, right, bottom)}")
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    inner_left = int(round((x1 / 1000.0 * width) - left))
    inner_top = int(round((y1 / 1000.0 * height) - top))
    inner_right = int(round((x2 / 1000.0 * width) - left))
    inner_bottom = int(round((y2 / 1000.0 * height) - top))
    line_width = max(2, max(crop.size) // 120)
    draw.rectangle((inner_left, inner_top, inner_right, inner_bottom), outline="#e31a1c", width=line_width)
    return crop


def load_record_image(row: CropRecord, image_source: str, thumbnail_padding_frac: float) -> Image.Image:
    if image_source == "highres_crop":
        return Image.open(row.crop_path).convert("RGB")
    if image_source == "thumbnail_bbox":
        return crop_thumbnail_bbox(row, thumbnail_padding_frac)
    raise ValueError(f"Unknown image source: {image_source}")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def stain_map_from_worklists(worklist_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for stain, filename in STAIN_WORKLISTS.items():
        path = worklist_dir / filename
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            link_path = Path(line.strip())
            mapping[link_path.name] = stain
            try:
                mapping[link_path.resolve(strict=False).name] = stain
            except Exception:
                pass
    return mapping


def load_records(pipeline_roots: list[Path], worklist_dir: Path) -> list[CropRecord]:
    stain_by_basename = stain_map_from_worklists(worklist_dir)
    records: list[CropRecord] = []
    for root in pipeline_roots:
        results_path = root / "intermediate_stage_artifacts/stage6_classification_results.jsonl"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        run_name = root.name
        with results_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                decision = str(row.get("tissue_focus_decision", "")).lower()
                if decision not in {"yes", "no"}:
                    continue
                if row.get("error"):
                    continue
                crop_path = Path(row.get("crop_path", ""))
                overlay_path = Path(row.get("selected_overlay_path", ""))
                thumbnail_path = Path(row.get("thumbnail_path", ""))
                if not crop_path.exists():
                    continue
                stain = stain_by_basename.get(Path(str(row.get("wsi_path", ""))).name, "UNKNOWN")
                label = 1 if decision == "no" else 0
                record_id = f"{run_name}:{row.get('case_id')}:{int(row.get('candidate_order', 0)):02d}"
                records.append(
                    CropRecord(
                        record_id=record_id,
                        run_name=run_name,
                        stain=stain,
                        case_id=str(row.get("case_id", "")),
                        case_slug=str(row.get("case_slug", "")),
                        case_display=str(row.get("case_display", "")),
                        candidate_order=int(row.get("candidate_order", 0)),
                        crop_path=crop_path,
                        selected_overlay_path=overlay_path,
                        thumbnail_path=thumbnail_path,
                        wsi_path=str(row.get("wsi_path", "")),
                        tissue_focus_decision=decision,
                        label=label,
                        raw=row,
                    )
                )
    return records


def row_to_dict(row: CropRecord) -> dict[str, Any]:
    return {
        "record_id": row.record_id,
        "run_name": row.run_name,
        "stain": row.stain,
        "case_id": row.case_id,
        "case_slug": row.case_slug,
        "case_display": row.case_display,
        "candidate_order": row.candidate_order,
        "crop_path": str(row.crop_path),
        "selected_overlay_path": str(row.selected_overlay_path),
        "thumbnail_path": str(row.thumbnail_path),
        "wsi_path": row.wsi_path,
        "tissue_focus_decision": row.tissue_focus_decision,
        "label_artifact": row.label,
    }


def select_balanced_training(records: list[CropRecord], seed: int) -> tuple[list[CropRecord], dict[str, Any]]:
    rng = random.Random(seed)
    rejected = sorted([r for r in records if r.label == 1], key=lambda r: (r.stain, r.case_id, r.candidate_order))
    accepted = [r for r in records if r.label == 0]
    accepted_by_case: dict[tuple[str, str], list[CropRecord]] = defaultdict(list)
    accepted_by_stain: dict[str, list[CropRecord]] = defaultdict(list)
    for row in accepted:
        accepted_by_case[(row.stain, row.case_id)].append(row)
        accepted_by_stain[row.stain].append(row)
    for rows in accepted_by_case.values():
        rows.sort(key=lambda r: r.candidate_order)
    for rows in accepted_by_stain.values():
        rows.sort(key=lambda r: (r.case_id, r.candidate_order))
        rng.shuffle(rows)

    selected_accepts: list[CropRecord] = []
    selected_ids: set[str] = set()
    match_rows: list[dict[str, Any]] = []

    for reject in rejected:
        chosen: CropRecord | None = None
        for candidate in accepted_by_case[(reject.stain, reject.case_id)]:
            if candidate.record_id not in selected_ids:
                chosen = candidate
                break
        if chosen is None:
            match_rows.append(
                {
                    "reject_record_id": reject.record_id,
                    "accept_record_id": "",
                    "match_type": "pending_same_stain_fill",
                    "stain": reject.stain,
                    "case_id": reject.case_id,
                }
            )
            continue
        selected_accepts.append(chosen)
        selected_ids.add(chosen.record_id)
        match_rows.append(
            {
                "reject_record_id": reject.record_id,
                "accept_record_id": chosen.record_id,
                "match_type": "same_case",
                "stain": reject.stain,
                "case_id": reject.case_id,
            }
        )

    stain_needed = Counter(r.stain for r in rejected) - Counter(r.stain for r in selected_accepts)
    pending_by_stain = defaultdict(list)
    for item in match_rows:
        if item["match_type"] == "pending_same_stain_fill":
            pending_by_stain[item["stain"]].append(item)

    for stain, needed in sorted(stain_needed.items()):
        pool = [row for row in accepted_by_stain[stain] if row.record_id not in selected_ids]
        if len(pool) < needed:
            raise RuntimeError(f"Need {needed} accepted crops for stain {stain}, but only {len(pool)} available")
        for item, chosen in zip(pending_by_stain[stain], pool[:needed]):
            item["accept_record_id"] = chosen.record_id
            item["match_type"] = "same_stain"
            selected_accepts.append(chosen)
            selected_ids.add(chosen.record_id)

    if len(selected_accepts) != len(rejected):
        raise RuntimeError(f"Selected {len(selected_accepts)} accepted crops for {len(rejected)} rejected crops")

    selected = rejected + selected_accepts
    selected.sort(key=lambda r: (r.label, r.stain, r.case_id, r.candidate_order))
    train_ids = {r.record_id for r in selected}
    stats = {
        "rejected_count": len(rejected),
        "accepted_count": len(selected_accepts),
        "same_case_accept_matches": sum(1 for r in match_rows if r["match_type"] == "same_case"),
        "same_stain_accept_matches": sum(1 for r in match_rows if r["match_type"] == "same_stain"),
        "train_class_counts": dict(Counter("artifact" if r.label == 1 else "accepted" for r in selected)),
        "train_stain_class_counts": {
            f"{stain}:{label}": count
            for (stain, label), count in sorted(
                Counter((r.stain, "artifact" if r.label == 1 else "accepted") for r in selected).items()
            )
        },
        "match_rows": match_rows,
        "train_record_ids": sorted(train_ids),
    }
    return selected, stats


def resolve_model_config(model: torch.nn.Module, requested_input_size: int | None) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    cfg = timm.data.resolve_model_data_config(model)
    input_size = requested_input_size or int(cfg.get("input_size", (3, 518, 518))[-1])
    mean = tuple(float(x) for x in cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in cfg.get("std", (0.229, 0.224, 0.225)))
    return input_size, mean, std


def extract_features(
    rows: list[CropRecord],
    model_name: str,
    input_size_arg: int | None,
    batch_size: int,
    device_name: str,
    num_workers: int,
    image_source: str,
    thumbnail_padding_frac: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    input_size, mean, std = resolve_model_config(model, input_size_arg)
    dataset = CropDataset(
        rows,
        input_size=input_size,
        mean=mean,
        std=std,
        image_source=image_source,
        thumbnail_padding_frac=thumbnail_padding_frac,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _indices in loader:
            images = images.to(device, non_blocking=True)
            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))
            features.append(output.detach().cpu().numpy().astype("float32"))
    metadata = {
        "model_name": model_name,
        "pretrained": True,
        "feature_dim": int(features[0].shape[1]) if features else 0,
        "input_size": input_size,
        "pad_fill_rgb": [255, 255, 255],
        "image_source": image_source,
        "thumbnail_padding_fraction": thumbnail_padding_frac if image_source == "thumbnail_bbox" else None,
        "resize": "bicubic_after_square_padding",
        "mean": list(mean),
        "std": list(std),
        "device": str(device),
    }
    return np.concatenate(features, axis=0), metadata


def evaluate_cv(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    n_splits = min(5, int(np.bincount(labels).min()))
    if n_splits < 2:
        return {"skipped": True, "reason": "Not enough samples per class"}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, features, labels, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "skipped": False,
        "n_splits": n_splits,
        "accuracy": float(accuracy_score(labels, pred)),
        "roc_auc": float(roc_auc_score(labels, prob)),
        "average_precision": float(average_precision_score(labels, prob)),
        "threshold": 0.5,
        "confusion": {
            "tn_accepted_as_accepted": int(((labels == 0) & (pred == 0)).sum()),
            "fp_accepted_as_artifact": int(((labels == 0) & (pred == 1)).sum()),
            "fn_artifact_as_accepted": int(((labels == 1) & (pred == 0)).sum()),
            "tp_artifact_as_artifact": int(((labels == 1) & (pred == 1)).sum()),
        },
    }


def fit_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
    )
    clf.fit(features, labels)
    return clf


def image_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_rejection_pdf(rows: list[dict[str, Any]], output_path: Path, title: str, max_pages: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 2400, 3200
    margin = 80
    header_h = 150
    cols, rows_per_page = 3, 4
    gap = 35
    card_w = (page_w - 2 * margin - (cols - 1) * gap) // cols
    card_h = (page_h - 2 * margin - header_h - (rows_per_page - 1) * gap) // rows_per_page
    image_h = card_h - 86
    title_font = image_font(44, bold=True)
    label_font = image_font(24, bold=False)
    small_font = image_font(21, bold=False)

    pages: list[Image.Image] = []
    total_pages = max(1, math.ceil(len(rows) / (cols * rows_per_page)))
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)
    if not rows:
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin), title, fill="black", font=title_font)
        draw.text((margin, margin + 90), "No Stage 6 accepted crops crossed the artifact threshold.", fill="black", font=label_font)
        pages.append(page)
    else:
        for page_idx in range(total_pages):
            page = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(page)
            start = page_idx * cols * rows_per_page
            end = min(len(rows), start + cols * rows_per_page)
            draw.text((margin, margin), title, fill="black", font=title_font)
            draw.text(
                (margin, margin + 62),
                f"Rows {start + 1}-{end} of {len(rows)} | sorted by artifact probability",
                fill="#333333",
                font=label_font,
            )
            for local_idx, row in enumerate(rows[start:end]):
                grid_r, grid_c = divmod(local_idx, cols)
                x = margin + grid_c * (card_w + gap)
                y = margin + header_h + grid_r * (card_h + gap)
                draw.rectangle((x, y, x + card_w, y + card_h), outline="#bbbbbb", width=2)
                image_path = Path(row.get("pdf_image_path") or row.get("selected_overlay_path") or row["crop_path"])
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (card_w, image_h), "#eeeeee")
                image.thumbnail((card_w - 16, image_h), Image.Resampling.LANCZOS)
                image_x = x + (card_w - image.width) // 2
                image_y = y + 8
                draw.rectangle((x + 8, y + 8, x + card_w - 8, y + image_h + 8), fill="#f7f7f7")
                page.paste(image, (image_x, image_y))
                text_y = y + image_h + 18
                label = (
                    f"{row['stain']} p_art={float(row['artifact_probability']):.3f} "
                    f"case={row['case_id']} cand={row['candidate_order']}"
                )
                draw.text((x + 10, text_y), label[:62], fill="black", font=small_font)
                draw.text((x + 10, text_y + 32), row["record_id"][:64], fill="#444444", font=small_font)
            pages.append(page)
    pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=180.0)


def write_reproduction(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "Stage 6 Tissue Linear Probe",
        "===========================",
        "",
        f"Created: {summary['created_at']}",
        f"Ticket: {summary['ticket']}",
        f"Git commit: {summary['git_commit']}",
        "",
        "Command:",
        " ".join(summary["command"]),
        "",
        "Environment:",
        f"- Python executable: {summary['python_executable']}",
        f"- Feature backbone: {summary['feature_extraction']['model_name']}",
        "- Requested by user: DINOv3 small; local runnable fallback used: DINOv2 small via timm.",
        "",
        "Input roots:",
        *[f"- {Path(root).resolve()}" for root in args.pipeline_roots],
        "",
        "Transform:",
        f"- Image source: {args.image_source}.",
        "- highres_crop uses Stage 5 crop.png, the higher-resolution WSI reread around the candidate bbox.",
        "- thumbnail_bbox crops the Stage 1 thumbnail using the Stage 6 normalized candidate bbox and thumbnail padding.",
        "- Convert selected image to RGB, square-pad with white background, bicubic resize to model input size, normalize with timm model config.",
        "",
        "Outputs:",
        f"- Summary: {output_dir / 'summary.json'}",
        f"- All labeled crop inventory: {output_dir / 'manifests/all_labeled_crops.csv'}",
        f"- Balanced train manifest: {output_dir / 'manifests/train_balanced_51_reject_51_accept.csv'}",
        f"- Fitted probe: {output_dir / 'models/linear_probe.pkl'}",
        f"- Remaining accepted scores: {output_dir / 'scores/remaining_accepted_scores.csv'}",
        f"- Proposed rejection CSV: {output_dir / 'scores/proposed_rejections.csv'}",
        f"- Proposed rejection PDF: {output_dir / 'visuals/proposed_rejections.pdf'}",
        "",
        "Label policy:",
        "- Stage 6 tissue_focus_decision == no -> artifact/rejected class.",
        "- Stage 6 tissue_focus_decision == yes -> accepted/tissue class.",
        "- Unknown/error/missing-crop rows are excluded from supervised training and scoring.",
    ]
    (output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--worklist-dir", type=Path, default=DEFAULT_WORKLIST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--image-source", choices=["highres_crop", "thumbnail_bbox"], default="highres_crop")
    parser.add_argument("--thumbnail-padding-frac", type=float, default=0.1)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument("--artifact-threshold", type=float, default=0.5)
    parser.add_argument("--ticket", default="PER-249")
    parser.add_argument("--max-pdf-pages", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = sys.argv[:]
    records = load_records(args.pipeline_roots, args.worklist_dir)
    training_rows, selection = select_balanced_training(records, args.seed)
    train_ids = {row.record_id for row in training_rows}
    remaining_accepted = sorted(
        [row for row in records if row.label == 0 and row.record_id not in train_ids],
        key=lambda r: (r.stain, r.case_id, r.candidate_order),
    )

    all_fields = list(row_to_dict(records[0]).keys()) if records else []
    write_csv(args.output_dir / "manifests/all_labeled_crops.csv", [row_to_dict(r) for r in records], all_fields)
    write_csv(
        args.output_dir / "manifests/train_balanced_51_reject_51_accept.csv",
        [row_to_dict(r) for r in training_rows],
        all_fields,
    )
    write_csv(
        args.output_dir / "manifests/train_accept_match_map.csv",
        selection["match_rows"],
        ["reject_record_id", "accept_record_id", "match_type", "stain", "case_id"],
    )

    train_features, feature_meta = extract_features(
        training_rows,
        args.model_name,
        args.input_size,
        args.batch_size,
        args.device,
        args.num_workers,
        args.image_source,
        args.thumbnail_padding_frac,
    )
    labels = np.asarray([row.label for row in training_rows], dtype=np.int64)
    cv_metrics = evaluate_cv(train_features, labels, args.seed)
    probe = fit_probe(train_features, labels, args.seed)
    write_pickle(args.output_dir / "models/linear_probe.pkl", probe)

    remaining_features, _ = extract_features(
        remaining_accepted,
        args.model_name,
        feature_meta["input_size"],
        args.batch_size,
        args.device,
        args.num_workers,
        args.image_source,
        args.thumbnail_padding_frac,
    )
    artifact_prob = probe.predict_proba(remaining_features)[:, 1] if len(remaining_accepted) else np.asarray([])
    score_rows: list[dict[str, Any]] = []
    for row, prob in zip(remaining_accepted, artifact_prob):
        out = row_to_dict(row)
        out["artifact_probability"] = float(prob)
        out["linear_probe_reject"] = bool(prob >= args.artifact_threshold)
        score_rows.append(out)
    score_rows.sort(key=lambda r: float(r["artifact_probability"]), reverse=True)
    proposed = [row for row in score_rows if row["linear_probe_reject"]]
    if args.image_source == "highres_crop":
        for row in proposed:
            row["pdf_image_path"] = row["selected_overlay_path"]
    elif args.image_source == "thumbnail_bbox":
        record_by_id = {row.record_id: row for row in remaining_accepted}
        preview_dir = args.output_dir / "visuals/thumbnail_bbox_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for row in proposed:
            record = record_by_id[row["record_id"]]
            image = crop_thumbnail_bbox(record, args.thumbnail_padding_frac)
            preview_path = preview_dir / f"{record.record_id.replace(':', '__')}.png"
            image.save(preview_path)
            row["pdf_image_path"] = str(preview_path)

    score_fields = all_fields + ["artifact_probability", "linear_probe_reject", "pdf_image_path"]
    write_csv(args.output_dir / "scores/remaining_accepted_scores.csv", score_rows, score_fields)
    write_csv(args.output_dir / "scores/proposed_rejections.csv", proposed, score_fields)
    make_rejection_pdf(
        proposed,
        args.output_dir / "visuals/proposed_rejections.pdf",
        title=f"Stage 6 Accepted Crops Rejected By Linear Probe (p >= {args.artifact_threshold:.2f})",
        max_pages=args.max_pdf_pages,
    )
    (args.output_dir / "features").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features/train_features.npz",
        features=train_features,
        labels=labels,
        record_ids=np.asarray([row.record_id for row in training_rows]),
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "python_executable": sys.executable,
        "command": command,
        "input": {
            "pipeline_roots": [str(path.resolve()) for path in args.pipeline_roots],
            "worklist_dir": str(args.worklist_dir.resolve()),
        },
        "available_labeled_crops": {
            "total": len(records),
            "class_counts": dict(Counter("artifact" if r.label == 1 else "accepted" for r in records)),
            "stain_class_counts": {
                f"{stain}:{label}": count
                for (stain, label), count in sorted(
                    Counter((r.stain, "artifact" if r.label == 1 else "accepted") for r in records).items()
                )
            },
        },
        "training_selection": {key: value for key, value in selection.items() if key != "match_rows"},
        "remaining_accepted_count": len(remaining_accepted),
        "feature_extraction": feature_meta,
        "linear_probe": {
            "model": "StandardScaler + LogisticRegression(liblinear)",
            "positive_class": "artifact_or_rejected_crop",
            "threshold": args.artifact_threshold,
            "cross_validation": cv_metrics,
        },
        "application_to_remaining_accepted": {
            "scored_count": len(score_rows),
            "proposed_rejection_count": len(proposed),
            "proposed_rejection_rate": float(len(proposed) / len(score_rows)) if score_rows else 0.0,
            "by_stain": {
                stain: {
                    "scored": sum(row["stain"] == stain for row in score_rows),
                    "proposed_rejections": sum(row["stain"] == stain for row in proposed),
                }
                for stain in sorted({row["stain"] for row in score_rows})
            },
        },
        "outputs": {
            "all_labeled_crops_csv": str((args.output_dir / "manifests/all_labeled_crops.csv").resolve()),
            "train_balanced_csv": str((args.output_dir / "manifests/train_balanced_51_reject_51_accept.csv").resolve()),
            "linear_probe_pickle": str((args.output_dir / "models/linear_probe.pkl").resolve()),
            "remaining_scores_csv": str((args.output_dir / "scores/remaining_accepted_scores.csv").resolve()),
            "proposed_rejections_csv": str((args.output_dir / "scores/proposed_rejections.csv").resolve()),
            "proposed_rejections_pdf": str((args.output_dir / "visuals/proposed_rejections.pdf").resolve()),
            "reproduction_txt": str((args.output_dir / "reproduction.txt").resolve()),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args.output_dir, args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
