#!/usr/bin/env python3
# ABOUTME: Stage 3 - Color-based foreground segmentation using two-pass clustering.
# ABOUTME: Takes stage1 output, clusters thumbnail to identify background, then segments bbox crops.
"""
Stage 3: Color-Based Foreground Segmentation.

This script performs two-pass clustering to segment foreground tissue from background:
1. Pass 1: Cluster full thumbnail to identify background (largest cluster)
2. Pass 2: Cluster each bbox crop, match to thumbnail background, create foreground mask

Usage:
    # Single stage1 directory
    python run_color_segmentation.py stage1_output/anon_xxx/.../timestamp/ --method kmeans --k 2

    # Batch processing
    python run_color_segmentation.py --batch dir1 dir2 dir3 --workers 8

    # Reprocess a single bbox (pass 2 only)
    python run_color_segmentation.py --reprocess stage3_output/.../bbox_dir --k 3

Output:
    stage3_output/{wsi_id}/{model}/{timestamp}/{x1_y1_x2_y2}/
        - crop.png: Original crop from thumbnail
        - mask.png: Binary foreground mask (white=foreground)
        - overlay.png: Clustering visualization
        - metadata.json: Resize info for level 0 conversion

    Reprocess output:
        stage3_output/.../{x1_y1_x2_y2}/{reprocess_timestamp}/
            - crop.png, mask.png, overlay.png, metadata.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import gaussian
from sklearn.cluster import KMeans

# Try to import HDBSCAN (optional)
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Import reproducibility utilities
from utils.reproducibility import require_clean_state, create_reproduce_command


# =============================================================================
# Clustering Functions
# =============================================================================

def detect_thumbnail_background(
    thumbnail_rgb: np.ndarray,
    method: str,
    params: dict,
    blur_sigma: float = 0.0
) -> Tuple[np.ndarray, int, np.ndarray, Optional[np.ndarray]]:
    """
    First pass: Cluster full thumbnail, identify background as largest cluster.

    Args:
        thumbnail_rgb: Full thumbnail image (H, W, 3) RGB uint8
        method: "kmeans" or "hdbscan"
        params: {"k": 2} or {"min_cluster_size": 50}
        blur_sigma: Optional Gaussian blur before clustering

    Returns:
        bg_center_lab: LAB center of background cluster [L, a, b]
        bg_cluster_idx: Index of background cluster
        labels_2d: Full label map (H, W) for visualization
        blurred_rgb: Blurred thumbnail (None if blur_sigma=0)
    """
    # Optional blur preprocessing
    blurred_rgb = None
    if blur_sigma > 0:
        thumbnail_rgb = (gaussian(thumbnail_rgb, sigma=blur_sigma, channel_axis=2) * 255).astype(np.uint8)
        blurred_rgb = thumbnail_rgb.copy()

    # Convert to LAB
    thumbnail_lab = rgb2lab(thumbnail_rgb)
    pixels = thumbnail_lab.reshape(-1, 3)

    if method == "kmeans":
        k = params.get("k", 2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
    elif method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")
        min_cluster_size = params.get("min_cluster_size", 50)
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, core_dist_n_jobs=1)
        labels = hdb.fit_predict(pixels)
        # Compute cluster centers for HDBSCAN
        centers = []
        valid_labels = sorted([l for l in set(labels) if l >= 0])
        for label in valid_labels:
            mask = labels == label
            centers.append(pixels[mask].mean(axis=0))
        centers = np.array(centers) if centers else np.empty((0, 3))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find largest cluster (by pixel count)
    unique, counts = np.unique(labels, return_counts=True)
    # Filter out noise label (-1) for HDBSCAN
    valid_mask = unique >= 0
    valid_unique = unique[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_unique) == 0:
        # No valid clusters found (all noise) - use first non-noise center if any
        raise ValueError("No valid clusters found in thumbnail")

    largest_idx = valid_unique[np.argmax(valid_counts)]

    if method == "kmeans":
        bg_center_lab = centers[largest_idx]
    else:
        # For HDBSCAN, find center index in valid_labels
        center_idx = valid_labels.index(largest_idx)
        bg_center_lab = centers[center_idx]

    labels_2d = labels.reshape(thumbnail_rgb.shape[:2])

    return bg_center_lab, int(largest_idx), labels_2d, blurred_rgb


def segment_bbox_crop(
    crop_rgb: np.ndarray,
    bg_center_lab: np.ndarray,
    method: str,
    params: dict,
    blur_sigma: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Second pass: Cluster bbox crop, match to background, create foreground mask.

    Args:
        crop_rgb: Bbox crop image (H, W, 3) RGB uint8
        bg_center_lab: Background cluster center from thumbnail [L, a, b]
        method: "kmeans" or "hdbscan"
        params: {"k": 2} or {"min_cluster_size": 50}
        blur_sigma: Optional Gaussian blur before clustering

    Returns:
        mask: Binary mask (H, W) - True=foreground, False=background
        overlay: Cluster visualization on original (H, W, 3) RGB uint8
        overlay_blurred: Cluster visualization on blurred (H, W, 3) or None
        stats: Clustering statistics dict
    """
    original_crop = crop_rgb.copy()

    # Optional blur preprocessing
    blurred_crop = None
    if blur_sigma > 0:
        crop_rgb = (gaussian(crop_rgb, sigma=blur_sigma, channel_axis=2) * 255).astype(np.uint8)
        blurred_crop = crop_rgb.copy()

    # Convert to LAB
    crop_lab = rgb2lab(crop_rgb)
    pixels = crop_lab.reshape(-1, 3)

    if method == "kmeans":
        k = params.get("k", 2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        valid_labels = list(range(k))
    elif method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")
        min_cluster_size = params.get("min_cluster_size", 50)
        # Scale min_cluster_size based on crop size
        n_pixels = len(pixels)
        actual_min_size = min(min_cluster_size, max(5, n_pixels // 20))
        hdb = HDBSCAN(min_cluster_size=actual_min_size, min_samples=5, core_dist_n_jobs=1)
        labels = hdb.fit_predict(pixels)
        # Compute cluster centers
        centers = []
        valid_labels = sorted([l for l in set(labels) if l >= 0])
        for label in valid_labels:
            mask = labels == label
            centers.append(pixels[mask].mean(axis=0))
        centers = np.array(centers) if centers else np.empty((0, 3))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find cluster closest to thumbnail background center (Euclidean in LAB)
    if len(centers) == 0:
        # No valid clusters - treat all as foreground
        bg_cluster_idx = -1
    else:
        distances = np.linalg.norm(centers - bg_center_lab, axis=1)
        closest_idx = np.argmin(distances)
        if method == "kmeans":
            bg_cluster_idx = closest_idx
        else:
            bg_cluster_idx = valid_labels[closest_idx]

    # Create binary mask: everything NOT background = foreground
    labels_2d = labels.reshape(crop_rgb.shape[:2])
    mask = labels_2d != bg_cluster_idx  # True = foreground

    # Create overlay visualization on original
    overlay = create_cluster_overlay(original_crop, labels_2d)

    # Create overlay on blurred version if blur was applied
    overlay_blurred = None
    if blurred_crop is not None:
        overlay_blurred = create_cluster_overlay(blurred_crop, labels_2d)

    # Compute stats
    foreground_ratio = float(mask.sum() / mask.size)

    stats = {
        "crop_bg_cluster_idx": int(bg_cluster_idx),
        "foreground_pixel_ratio": foreground_ratio,
        "cluster_centers_lab": centers.tolist() if len(centers) > 0 else [],
        "thumbnail_bg_center_lab": bg_center_lab.tolist(),
        "n_clusters": len(valid_labels),
    }

    return mask, overlay, overlay_blurred, stats


def create_cluster_overlay(crop: np.ndarray, labels: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create color overlay for cluster assignments.

    Args:
        crop: Original crop image (H, W, 3) RGB uint8
        labels: Cluster labels (H, W), -1 = noise
        alpha: Overlay transparency (0-1)

    Returns:
        overlay: RGB image with cluster colors overlaid
    """
    if crop.size == 0 or labels.size == 0:
        return crop

    # Colors for clusters (up to 8 distinct)
    CLUSTER_COLORS = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
    ]

    overlay = crop.astype(np.float32).copy()

    unique_labels = sorted(set(labels.flatten()))
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise -> gray
            color = np.array([128, 128, 128])
        else:
            color = np.array(CLUSTER_COLORS[label % len(CLUSTER_COLORS)])

        mask = labels == label
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# =============================================================================
# Processing Functions
# =============================================================================

def extract_crop(thumbnail: np.ndarray, bbox_thumbnail: List[int]) -> np.ndarray:
    """
    Extract crop from thumbnail given bbox coordinates.

    Args:
        thumbnail: Full thumbnail image (H, W, 3)
        bbox_thumbnail: [x1, y1, x2, y2] in thumbnail pixel coords

    Returns:
        crop: Cropped region (H, W, 3)
    """
    x1, y1, x2, y2 = bbox_thumbnail
    h, w = thumbnail.shape[:2]

    # Clamp to valid range
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    return thumbnail[y1:y2, x1:x2]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_bbox_level0_from_dirname(name: str) -> Optional[List[int]]:
    parts = name.split("_")
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def resolve_bg_center_lab(bbox_meta: dict, run_meta: Optional[dict]) -> np.ndarray:
    bg_center = bbox_meta.get("thumbnail_bg_center_lab")
    if bg_center is None and run_meta:
        bg_center = run_meta.get("thumbnail_bg_center_lab")
    if bg_center is None:
        raise ValueError(
            "Background cluster center not found in metadata "
            "(expected key: thumbnail_bg_center_lab)."
        )
    if not isinstance(bg_center, list) or len(bg_center) != 3:
        raise ValueError("Invalid background center format in metadata.")
    return np.array(bg_center, dtype=np.float32)


def process_single_stage1_dir(
    stage1_dir: str,
    method: str,
    params: dict,
    blur_sigma: float,
    output_base: str,
    no_overlay: bool = False,
    parser: Optional[argparse.ArgumentParser] = None,
    skip_repro_check: bool = False,
    skip_dvc_check: bool = False,
) -> dict:
    """
    Process a single stage1 directory (all bboxes).

    Args:
        stage1_dir: Path to stage1 output directory
        method: Clustering method ("kmeans" or "hdbscan")
        params: Method parameters
        blur_sigma: Gaussian blur sigma
        output_base: Base output directory
        no_overlay: Skip generating overlay visualization
        parser: ArgumentParser for reproduce.txt generation
        skip_repro_check: If True, skip git/DVC state check (for batch mode)

    Returns:
        result dict with output_dir, bbox_count, etc.
    """
    # === REPRODUCIBILITY CHECK ===
    if skip_repro_check:
        state_info = {"bypassed": True, "reason": "batch mode", "git_hash": "batch_deferred"}
    else:
        state_info = require_clean_state([stage1_dir], skip_dvc_check=skip_dvc_check)
        if state_info.get("bypassed"):
            print(f"  Warning: Reproducibility check bypassed: {state_info.get('reason')}")
    stage1_path = Path(stage1_dir)

    # Load required files
    thumbnail_path = stage1_path / "thumbnail.png"
    metadata_path = stage1_path / "metadata.json"

    if not thumbnail_path.exists():
        raise FileNotFoundError(f"thumbnail.png not found in {stage1_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {stage1_dir}")

    # Load thumbnail
    thumbnail = np.array(Image.open(thumbnail_path).convert("RGB"))

    # Load metadata
    with open(metadata_path) as f:
        stage1_metadata = json.load(f)

    detected_regions = stage1_metadata.get("detected_regions", [])
    if not detected_regions:
        print(f"  Warning: No detected regions in {stage1_dir}")
        return {
            "stage1_dir": str(stage1_dir),
            "output_dir": None,
            "bbox_count": 0,
            "status": "no_regions",
        }

    # Extract path components for output directory
    # stage1_dir: stage1_output/{wsi_id}/{model}/{timestamp}
    parts = stage1_path.parts
    # Find index of stage1_output in path
    try:
        stage1_idx = parts.index("stage1_output")
        wsi_id = parts[stage1_idx + 1]
        model = parts[stage1_idx + 2]
        # Use fresh timestamp for stage3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        # Fallback: use directory name components
        wsi_id = stage1_path.parent.parent.name
        model = stage1_path.parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path(output_base) / wsi_id / model / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Detect thumbnail background
    t0 = time.time()
    print(f"  Pass 1: Clustering thumbnail ({thumbnail.shape[1]}x{thumbnail.shape[0]})...")
    bg_center_lab, bg_cluster_idx, thumb_labels, thumb_blurred = detect_thumbnail_background(
        thumbnail, method, params, blur_sigma
    )
    t1 = time.time()
    print(f"    Background cluster: {bg_cluster_idx}, LAB center: [{bg_center_lab[0]:.1f}, {bg_center_lab[1]:.1f}, {bg_center_lab[2]:.1f}]")
    print(f"    [TIMING] Pass 1 clustering took {t1-t0:.2f}s")

    # Save thumbnail cluster overlay (Pass 1 visualization)
    thumb_overlay = create_cluster_overlay(thumbnail, thumb_labels)
    Image.fromarray(thumb_overlay).save(output_dir / "thumbnail_clusters.png")
    print(f"    Saved: {output_dir / 'thumbnail_clusters.png'}")

    # Save blurred thumbnail overlay if blur was applied
    if thumb_blurred is not None:
        thumb_overlay_blurred = create_cluster_overlay(thumb_blurred, thumb_labels)
        Image.fromarray(thumb_overlay_blurred).save(output_dir / "thumbnail_clusters_blurred.png")
        print(f"    Saved: {output_dir / 'thumbnail_clusters_blurred.png'}")

    # Process each bbox
    bbox_results = []
    for i, region in enumerate(detected_regions):
        bbox_thumbnail = region.get("bbox_thumbnail")
        bbox_level0 = region.get("bbox_level0")

        if not bbox_thumbnail or not bbox_level0:
            print(f"    Skipping region {i}: missing bbox coordinates")
            continue

        # Create bbox subdirectory using level0 coords
        x1, y1, x2, y2 = [int(c) for c in bbox_level0]
        bbox_dir_name = f"{x1}_{y1}_{x2}_{y2}"
        bbox_output_dir = output_dir / bbox_dir_name
        bbox_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract crop
        crop = extract_crop(thumbnail, bbox_thumbnail)
        if crop.size == 0:
            print(f"    Skipping region {i}: empty crop")
            continue

        print(f"  Pass 2: Segmenting bbox {i+1}/{len(detected_regions)} ({crop.shape[1]}x{crop.shape[0]})...")

        # Pass 2: Segment crop
        t2 = time.time()
        mask, overlay, overlay_blurred, stats = segment_bbox_crop(
            crop, bg_center_lab, method, params, blur_sigma
        )
        t3 = time.time()
        print(f"    [TIMING] Pass 2 bbox {i+1} segmentation took {t3-t2:.2f}s")

        # Save outputs
        # 1. crop.png
        Image.fromarray(crop).save(bbox_output_dir / "crop.png")

        # 2. mask.png (white=foreground, black=background)
        mask_img = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_img, mode="L").save(bbox_output_dir / "mask.png")

        # 3. overlay.png (optional)
        if not no_overlay:
            Image.fromarray(overlay).save(bbox_output_dir / "overlay.png")
            # Save blurred overlay if blur was applied
            if overlay_blurred is not None:
                Image.fromarray(overlay_blurred).save(bbox_output_dir / "overlay_blurred.png")

        # 4. Per-bbox metadata.json
        crop_h, crop_w = crop.shape[:2]
        level0_w = x2 - x1
        level0_h = y2 - y1

        bbox_metadata = {
            "bbox_level0": bbox_level0,
            "bbox_thumbnail": bbox_thumbnail,
            "crop_dims": {"width": crop_w, "height": crop_h},
            "level0_dims": {"width": level0_w, "height": level0_h},
            "scale_factor": {
                "x": level0_w / crop_w if crop_w > 0 else 1.0,
                "y": level0_h / crop_h if crop_h > 0 else 1.0,
            },
            "method": method,
            "params": params,
            "blur_sigma": blur_sigma,
            **stats,
        }

        with open(bbox_output_dir / "metadata.json", "w") as f:
            json.dump(bbox_metadata, f, indent=2)

        bbox_results.append({
            "bbox_dir": bbox_dir_name,
            "foreground_ratio": stats["foreground_pixel_ratio"],
        })

        print(f"    Foreground: {stats['foreground_pixel_ratio']*100:.1f}%")

    # Save run-level metadata with reproducibility info
    run_metadata = {
        "stage1_dir": str(stage1_dir),
        "wsi_path": stage1_metadata.get("wsi_path"),
        "wsi_id": wsi_id,
        "wsi_dimensions": stage1_metadata.get("wsi_dimensions"),
        "thumbnail_dimensions": stage1_metadata.get("thumbnail_dimensions"),
        "method": method,
        "params": params,
        "blur_sigma": blur_sigma,
        "bbox_count": len(bbox_results),
        "thumbnail_bg_center_lab": bg_center_lab.tolist(),
        "thumbnail_bg_cluster_idx": bg_cluster_idx,
        "summaries": bbox_results,
        "timestamp": timestamp,
        "git_hash": state_info.get("git_hash", "unknown"),
        "reproducibility_bypassed": state_info.get("bypassed", False),
        "created_at": datetime.now().isoformat(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    # Generate reproduce.txt using utility
    if parser is not None:
        reproduce_path = output_dir / "reproduce.txt"
        create_reproduce_command(parser, str(reproduce_path), git_hash=state_info.get("git_hash"))
        print(f"  Saved reproduce.txt: {reproduce_path}")

    return {
        "stage1_dir": str(stage1_dir),
        "output_dir": str(output_dir),
        "bbox_count": len(bbox_results),
        "status": "success",
    }


def reprocess_bbox_dir(
    bbox_dir: str,
    method: str,
    params: dict,
    blur_sigma: float,
    no_overlay: bool = False,
    parser: Optional[argparse.ArgumentParser] = None,
    skip_dvc_check: bool = False,
) -> dict:
    """
    Reprocess a single stage3 bbox directory using stored background center.

    Args:
        bbox_dir: Path to stage3 bbox directory (contains crop.png, metadata.json)
        method: Clustering method ("kmeans" or "hdbscan")
        params: Method parameters
        blur_sigma: Gaussian blur sigma
        no_overlay: Skip generating overlay visualization
        parser: ArgumentParser for reproduce.txt generation

    Returns:
        result dict with output_dir, status, etc.
    """
    # === REPRODUCIBILITY CHECK ===
    state_info = require_clean_state([bbox_dir], skip_dvc_check=skip_dvc_check)
    if state_info.get("bypassed"):
        print(f"  Warning: Reproducibility check bypassed: {state_info.get('reason')}")

    bbox_path = Path(bbox_dir)
    if not bbox_path.exists():
        raise FileNotFoundError(f"Bbox directory not found: {bbox_dir}")

    crop_path = bbox_path / "crop.png"
    bbox_meta_path = bbox_path / "metadata.json"
    if not crop_path.exists():
        raise FileNotFoundError(f"crop.png not found in {bbox_dir}")
    if not bbox_meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {bbox_dir}")

    bbox_meta = load_json(bbox_meta_path)
    run_meta_path = bbox_path.parent / "metadata.json"
    run_meta = load_json(run_meta_path) if run_meta_path.exists() else None

    bg_center_lab = resolve_bg_center_lab(bbox_meta, run_meta)

    crop = np.array(Image.open(crop_path).convert("RGB"))
    if crop.size == 0:
        raise ValueError(f"Empty crop in {crop_path}")

    print(f"Reprocessing bbox: {bbox_dir}")
    print(f"  Method: {method}, params: {params}, blur: {blur_sigma}")

    mask, overlay, overlay_blurred, stats = segment_bbox_crop(
        crop, bg_center_lab, method, params, blur_sigma
    )

    # Create output subdir inside bbox dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = bbox_path / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    Image.fromarray(crop).save(output_dir / "crop.png")
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(output_dir / "mask.png")

    if not no_overlay:
        Image.fromarray(overlay).save(output_dir / "overlay.png")
        if overlay_blurred is not None:
            Image.fromarray(overlay_blurred).save(output_dir / "overlay_blurred.png")

    bbox_level0 = bbox_meta.get("bbox_level0") or parse_bbox_level0_from_dirname(bbox_path.name)
    bbox_thumbnail = bbox_meta.get("bbox_thumbnail")

    crop_h, crop_w = crop.shape[:2]
    level0_dims = None
    scale_factor = None
    if bbox_level0:
        x1, y1, x2, y2 = [int(c) for c in bbox_level0]
        level0_w = x2 - x1
        level0_h = y2 - y1
        level0_dims = {"width": level0_w, "height": level0_h}
        scale_factor = {
            "x": level0_w / crop_w if crop_w > 0 else 1.0,
            "y": level0_h / crop_h if crop_h > 0 else 1.0,
        }

    bbox_metadata = {
        "bbox_level0": bbox_level0,
        "bbox_thumbnail": bbox_thumbnail,
        "crop_dims": {"width": crop_w, "height": crop_h},
        "level0_dims": level0_dims,
        "scale_factor": scale_factor,
        "method": method,
        "params": params,
        "blur_sigma": blur_sigma,
        **stats,
        "reprocess": {
            "source_bbox_dir": str(bbox_dir),
            "source_metadata": str(bbox_meta_path),
            "source_run_metadata": str(run_meta_path) if run_meta_path.exists() else None,
            "timestamp": timestamp,
            "git_hash": state_info.get("git_hash", "unknown"),
            "reproducibility_bypassed": state_info.get("bypassed", False),
            "created_at": datetime.now().isoformat(),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(bbox_metadata, f, indent=2)

    if parser is not None:
        reproduce_path = output_dir / "reproduce.txt"
        create_reproduce_command(parser, str(reproduce_path), git_hash=state_info.get("git_hash"))
        print(f"  Saved reproduce.txt: {reproduce_path}")

    print(f"  Foreground: {stats['foreground_pixel_ratio']*100:.1f}%")

    return {
        "bbox_dir": str(bbox_dir),
        "output_dir": str(output_dir),
        "status": "success",
    }


def run_batch_parallel(
    stage1_dirs: List[str],
    method: str,
    params: dict,
    blur_sigma: float,
    output_base: str,
    workers: int,
    no_overlay: bool = False,
    skip_dvc_check: bool = False,
) -> List[dict]:
    """
    Process multiple stage1 directories in parallel.

    Args:
        stage1_dirs: List of stage1 directory paths
        method: Clustering method
        params: Method parameters
        blur_sigma: Gaussian blur sigma
        output_base: Base output directory
        workers: Number of parallel workers
        no_overlay: Skip generating overlay visualization

    Returns:
        List of result dicts
    """
    # === PRE-BATCH REPRODUCIBILITY CHECK ===
    print("Checking git/DVC state...")
    t_dvc_start = time.time()
    state_info = require_clean_state([stage1_dirs[0]], skip_dvc_check=skip_dvc_check)
    t_dvc_end = time.time()
    if state_info.get("bypassed"):
        print(f"  Bypassed: {state_info.get('reason')}")
    else:
        print(f"  Git hash: {state_info.get('git_hash', 'unknown')}")
    print(f"  [TIMING] DVC/git check took {t_dvc_end-t_dvc_start:.2f}s")
    print()

    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_dir = {
            executor.submit(
                process_single_stage1_dir,
                stage1_dir,
                method,
                params,
                blur_sigma,
                output_base,
                no_overlay,
                None,  # parser - skip reproduce.txt in batch mode
                True,  # skip_repro_check - already checked above
                skip_dvc_check,
            ): stage1_dir
            for stage1_dir in stage1_dirs
        }

        for future in as_completed(future_to_dir):
            stage1_dir = future_to_dir[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed: {stage1_dir} -> {result.get('output_dir', 'N/A')}")
            except Exception as e:
                results.append({
                    "stage1_dir": stage1_dir,
                    "status": "error",
                    "error": str(e),
                })
                print(f"Error processing {stage1_dir}: {e}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Color-based foreground segmentation using two-pass clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single stage1 directory
    python run_color_segmentation.py stage1_output/anon_xxx/.../timestamp/ --method kmeans --k 2

    # Batch processing
    python run_color_segmentation.py --batch dir1 dir2 dir3 --workers 8

    # Use HDBSCAN with blur
    python run_color_segmentation.py stage1_dir --method hdbscan --min-cluster-size 100 --blur 3

    # Reprocess a single bbox (pass 2 only)
    python run_color_segmentation.py --reprocess stage3_output/.../bbox_dir --k 3
"""
    )

    # Input
    parser.add_argument(
        "stage1_dir",
        type=str,
        nargs="?",
        help="Path to stage1 output directory (e.g., stage1_output/anon_xxx/.../timestamp/)"
    )
    parser.add_argument(
        "--reprocess",
        type=str,
        help="Path to a stage3 bbox directory to reprocess (contains crop.png, metadata.json)"
    )

    # Clustering method
    parser.add_argument(
        "--method",
        choices=["kmeans", "hdbscan"],
        default=None,
        help="Clustering method (default: kmeans)"
    )

    # K-means params
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters for K-means (default: 2)"
    )

    # HDBSCAN params
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum cluster size for HDBSCAN (default: 50)"
    )

    # Optional blur
    parser.add_argument(
        "--blur",
        type=float,
        default=None,
        help="Gaussian blur sigma before clustering (0 = no blur, default: 0)"
    )

    # Batch processing
    parser.add_argument(
        "--batch",
        type=str,
        nargs="+",
        metavar="DIR",
        help="Process multiple stage1 directories in batch mode"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for batch mode (default: 4)"
    )

    # Output control
    parser.add_argument(
        "--output-base",
        type=str,
        default="stage3_output",
        help="Base output directory (default: stage3_output)"
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip generating overlay visualization"
    )
    parser.add_argument(
        "--skip-dvc-check",
        action="store_true",
        help="Bypass DVC clean-state check (still checks git)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.stage1_dir and not args.batch and not args.reprocess:
        parser.error("Either stage1_dir, --batch, or --reprocess must be provided")

    if args.reprocess and (args.stage1_dir or args.batch):
        parser.error("Use --reprocess alone (do not combine with stage1_dir or --batch)")

    # Reprocess mode
    if args.reprocess:
        bbox_path = Path(args.reprocess)
        bbox_meta_path = bbox_path / "metadata.json"
        if not bbox_meta_path.exists():
            parser.error(f"metadata.json not found in {args.reprocess}")
        bbox_meta = load_json(bbox_meta_path)
        run_meta_path = bbox_path.parent / "metadata.json"
        run_meta = load_json(run_meta_path) if run_meta_path.exists() else {}

        method = args.method or bbox_meta.get("method") or run_meta.get("method") or "kmeans"
        blur_sigma = args.blur if args.blur is not None else (
            bbox_meta.get("blur_sigma") if "blur_sigma" in bbox_meta else run_meta.get("blur_sigma", 0.0)
        )
        if blur_sigma is None:
            blur_sigma = 0.0

        if method == "hdbscan" and not HDBSCAN_AVAILABLE:
            print("Error: HDBSCAN not installed. Install with: pip install hdbscan")
            sys.exit(1)

        if method == "kmeans":
            k = args.k if args.k is not None else (
                bbox_meta.get("params", {}).get("k") or run_meta.get("params", {}).get("k") or 2
            )
            if k < 2:
                parser.error("--k must be at least 2")
            params = {"k": k}
        else:
            min_cluster_size = args.min_cluster_size if args.min_cluster_size is not None else (
                bbox_meta.get("params", {}).get("min_cluster_size")
                or run_meta.get("params", {}).get("min_cluster_size")
                or 50
            )
            params = {"min_cluster_size": min_cluster_size}

        result = reprocess_bbox_dir(
            args.reprocess,
            method,
            params,
            blur_sigma,
            args.no_overlay,
            parser=parser,
            skip_dvc_check=args.skip_dvc_check,
        )
        print(f"\nOutput: {result.get('output_dir', 'N/A')}")
        return

    # Resolve method/params defaults (standard mode)
    if args.method is None:
        args.method = "kmeans"
    if args.blur is None:
        args.blur = 0.0

    if args.method == "hdbscan" and not HDBSCAN_AVAILABLE:
        print("Error: HDBSCAN not installed. Install with: pip install hdbscan")
        sys.exit(1)

    # Standard mode defaults
    if args.method == "kmeans":
        if args.k is None:
            args.k = 2
        if args.k < 2:
            parser.error("--k must be at least 2")
        params = {"k": args.k}
    else:
        if args.min_cluster_size is None:
            args.min_cluster_size = 50
        params = {"min_cluster_size": args.min_cluster_size}

    # Process
    if args.batch:
        print(f"Batch processing {len(args.batch)} directories with {args.workers} workers...")
        results = run_batch_parallel(
            args.batch,
            args.method,
            params,
            args.blur,
            args.output_base,
            args.workers,
            args.no_overlay,
            args.skip_dvc_check,
        )

        # Summary
        success = sum(1 for r in results if r.get("status") == "success")
        errors = sum(1 for r in results if r.get("status") == "error")
        print(f"\nBatch complete: {success} succeeded, {errors} failed")

        if errors > 0:
            print("\nErrors:")
            for r in results:
                if r.get("status") == "error":
                    print(f"  {r['stage1_dir']}: {r.get('error', 'unknown')}")

    else:
        # === SINGLE DIR MODE ===
        print(f"Processing {args.stage1_dir}...")
        result = process_single_stage1_dir(
            args.stage1_dir,
            args.method,
            params,
            args.blur,
            args.output_base,
            args.no_overlay,
            parser=parser,  # Pass parser for reproduce.txt generation
            skip_dvc_check=args.skip_dvc_check,
        )
        print(f"\nOutput: {result.get('output_dir', 'N/A')}")
        print(f"Processed {result.get('bbox_count', 0)} bboxes")


if __name__ == "__main__":
    main()
