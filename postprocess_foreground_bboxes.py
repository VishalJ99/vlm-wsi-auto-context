#!/usr/bin/env python3
# ABOUTME: Postprocess bbox detections from orientation experiments.
# ABOUTME: Aggregates bboxes from all orientations, transforms to 0° space, merges overlapping regions.
"""
Postprocess Foreground BBoxes from Orientation Experiments.

This script:
1. Collects bboxes from all 4 orientations (0°, 90°, 180°, 270°) for Flash model
2. Transforms bboxes back to 0° coordinate space
3. Merges overlapping bboxes into unified regions
4. Runs k-means (k=2) within each merged bbox
5. Generates visualization PDF

Usage:
    python postprocess_foreground_bboxes.py --output-pdf postprocessed_bboxes.pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.color import rgb2lab, rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# Try to import HDBSCAN (may not be installed)
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed, HDBSCAN method will be skipped")


# =============================================================================
# Constants
# =============================================================================

STAGE1_DIR = Path("stage1_output")
THUMBNAIL_DIR = Path("cached_thumbnails")

MODEL_DIR_NAMES = {
    "flash": "google_gemini_3_flash_preview",
    "pro": "google_gemini_3_pro_preview",
}

# Colors for orientation visualization (RGBA)
ORIENTATION_COLORS = {
    0: "red",
    90: "blue",
    180: "green",
    270: "orange",
}


# =============================================================================
# Coordinate Transformation Functions
# =============================================================================

def transform_bbox_to_rot0(bbox: List[int], rotation: int) -> Tuple[int, int, int, int]:
    """
    Transform bbox from rotated space back to 0° space.

    BBox format: [y_min, x_min, y_max, x_max] (Gemini format) in 0-1000 normalized coords.

    For PIL rotate(-degrees, expand=True) which rotates clockwise:
    - 90° CW: original (x,y) -> rotated (y, W-x), inverse: (x', y') -> (W-y', x')
    - 180°: original (x,y) -> rotated (W-x, H-y), inverse: same
    - 270° CW: original (x,y) -> rotated (H-y, x), inverse: (x', y') -> (y', H-x')

    In normalized 0-1000 coords, W=H=1000.
    """
    y1, x1, y2, x2 = bbox

    if rotation == 0:
        return (y1, x1, y2, x2)
    elif rotation == 90:
        # Inverse of 90° CW rotation
        # Point (x,y) in rotated image -> (y, 1000-x) in original
        # For bbox [y1, x1, y2, x2]:
        # Corner (x1, y1) -> original (y1, 1000-x1) -> Gemini (1000-x1, y1)
        # Corner (x2, y2) -> original (y2, 1000-x2) -> Gemini (1000-x2, y2)
        new_y1 = 1000 - x2
        new_x1 = y1
        new_y2 = 1000 - x1
        new_x2 = y2
        return (new_y1, new_x1, new_y2, new_x2)
    elif rotation == 180:
        # Inverse of 180° rotation
        new_y1 = 1000 - y2
        new_x1 = 1000 - x2
        new_y2 = 1000 - y1
        new_x2 = 1000 - x1
        return (new_y1, new_x1, new_y2, new_x2)
    elif rotation == 270:
        # Inverse of 270° CW rotation (= 90° CCW)
        # Point (x,y) in rotated image -> (1000-y, x) in original
        # For bbox [y1, x1, y2, x2]:
        # Corner (x1, y1) -> original (1000-y1, x1) -> Gemini (x1, 1000-y1)
        # Corner (x2, y2) -> original (1000-y2, x2) -> Gemini (x2, 1000-y2)
        new_y1 = x1
        new_x1 = 1000 - y2
        new_y2 = x2
        new_x2 = 1000 - y1
        return (new_y1, new_x1, new_y2, new_x2)
    else:
        raise ValueError(f"Unsupported rotation: {rotation}")


def bbox_overlaps(a: Tuple, b: Tuple) -> bool:
    """Check if two bboxes overlap. Format: (y1, x1, y2, x2)."""
    # No overlap if one is completely left/right/above/below the other
    return not (a[3] <= b[1] or b[3] <= a[1] or a[2] <= b[0] or b[2] <= a[0])


def merge_overlapping_bboxes(bboxes: List[Tuple]) -> List[Tuple]:
    """Merge overlapping bboxes into unified hulls using iterative greedy merge."""
    if not bboxes:
        return []

    merged = [tuple(b) for b in bboxes]
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i, box_a in enumerate(merged):
            if i in used:
                continue

            hull = list(box_a)
            for j, box_b in enumerate(merged):
                if j <= i or j in used:
                    continue

                if bbox_overlaps(tuple(hull), box_b):
                    # Merge: expand hull to encompass both
                    hull = [
                        min(hull[0], box_b[0]),  # y1
                        min(hull[1], box_b[1]),  # x1
                        max(hull[2], box_b[2]),  # y2
                        max(hull[3], box_b[3]),  # x2
                    ]
                    used.add(j)
                    changed = True

            new_merged.append(tuple(hull))

        merged = new_merged

    return merged


# =============================================================================
# Data Collection Functions
# =============================================================================

def find_orientation_runs(wsi_id: str, model: str = "flash") -> Dict[int, Path]:
    """
    Find experiment runs for each orientation (0, 90, 180, 270) for a WSI.

    Returns dict mapping rotation -> run directory path.
    """
    model_dir_name = MODEL_DIR_NAMES[model]
    wsi_dir = STAGE1_DIR / wsi_id / model_dir_name

    if not wsi_dir.exists():
        return {}

    orientation_runs = {}

    for run_dir in wsi_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Check for metadata.json to get rotation
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            rotation = metadata.get("rotation")
            if rotation is not None and rotation in [0, 90, 180, 270]:
                # Keep most recent run per rotation (lexicographically latest timestamp)
                if rotation not in orientation_runs or str(run_dir) > str(orientation_runs[rotation]):
                    orientation_runs[rotation] = run_dir
        except (json.JSONDecodeError, KeyError):
            continue

    return orientation_runs


def load_bboxes_from_run(run_dir: Path) -> List[List[int]]:
    """Load bboxes from a run directory."""
    bboxes_path = run_dir / "bboxes.json"
    if not bboxes_path.exists():
        return []

    try:
        with open(bboxes_path) as f:
            data = json.load(f)

        bboxes = []
        for item in data:
            if isinstance(item, dict):
                bbox = item.get("box_2d") or item.get("bbox_2d") or item.get("bbox")
            elif isinstance(item, list):
                bbox = item
            else:
                continue

            if bbox and len(bbox) == 4:
                bboxes.append(bbox)

        return bboxes
    except (json.JSONDecodeError, KeyError):
        return []


def collect_all_wsi_cases(model: str = "flash") -> List[str]:
    """Collect all WSI case IDs that have orientation experiments."""
    if not STAGE1_DIR.exists():
        return []

    model_dir_name = MODEL_DIR_NAMES[model]
    cases = []

    for wsi_dir in sorted(STAGE1_DIR.iterdir()):
        if not wsi_dir.is_dir():
            continue
        if wsi_dir.name.startswith("."):
            continue
        # Check if model directory exists
        if (wsi_dir / model_dir_name).exists():
            cases.append(wsi_dir.name)

    return cases


# =============================================================================
# K-Means Segmentation
# =============================================================================

def run_kmeans_on_bbox(
    thumbnail: np.ndarray,
    bbox: Tuple[int, int, int, int],
    k: int = 2
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Run k-means clustering on a bbox region.

    Args:
        thumbnail: Full thumbnail image (H, W, 3) RGB
        bbox: (y1, x1, y2, x2) in 0-1000 normalized coords
        k: Number of clusters

    Returns:
        crop: The cropped region (H, W, 3)
        labels: Cluster labels (H, W)
        centers: Cluster centers
    """
    h, w = thumbnail.shape[:2]

    # Convert normalized coords to pixel coords
    y1, x1, y2, x2 = bbox
    py1 = int(y1 / 1000 * h)
    px1 = int(x1 / 1000 * w)
    py2 = int(y2 / 1000 * h)
    px2 = int(x2 / 1000 * w)

    # Clamp to valid range
    py1 = max(0, min(py1, h))
    py2 = max(0, min(py2, h))
    px1 = max(0, min(px1, w))
    px2 = max(0, min(px2, w))

    # Extract crop
    crop = thumbnail[py1:py2, px1:px2]

    if crop.size == 0:
        return crop, np.array([]), []

    # Convert to LAB for clustering
    crop_lab = rgb2lab(crop)
    pixels = crop_lab.reshape(-1, 3)

    # K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    labels = labels.reshape(crop.shape[:2])

    return crop, labels, kmeans.cluster_centers_.tolist()


def create_cluster_overlay(crop: np.ndarray, labels: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """
    Create color overlay for cluster assignments with distinct colors.

    Supports variable number of clusters. -1 labels (noise in HDBSCAN) shown in gray.
    """
    if crop.size == 0 or labels.size == 0:
        return crop

    # Colors for clusters (up to 8 distinct)
    CLUSTER_COLORS = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
    ]

    # Create RGBA overlay
    overlay = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)

    unique_labels = sorted(set(labels.flatten()))
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise (HDBSCAN) -> gray
            overlay[labels == label] = [128, 128, 128, 100]
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            overlay[labels == label] = [*color, 100]

    # Composite
    crop_rgba = np.dstack([crop, np.full(crop.shape[:2], 255, dtype=np.uint8)])
    result = crop_rgba.astype(float)

    # Alpha blend
    alpha = overlay[:, :, 3:4] / 255.0
    result[:, :, :3] = result[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha

    return result[:, :, :3].astype(np.uint8)


# Keep old function for backwards compatibility
def create_kmeans_overlay(crop: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Backwards compatible wrapper."""
    return create_cluster_overlay(crop, labels)


# =============================================================================
# Multi-Method Clustering
# =============================================================================

def extract_crop_from_bbox(thumbnail: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract crop from thumbnail given bbox in 0-1000 normalized coords."""
    h, w = thumbnail.shape[:2]
    y1, x1, y2, x2 = bbox

    py1 = max(0, min(int(y1 / 1000 * h), h))
    px1 = max(0, min(int(x1 / 1000 * w), w))
    py2 = max(0, min(int(y2 / 1000 * h), h))
    px2 = max(0, min(int(x2 / 1000 * w), w))

    return thumbnail[py1:py2, px1:px2]


def run_kmeans_auto_k(crop_rgb: np.ndarray, k_range: Tuple[int, int] = (2, 5)) -> Tuple[np.ndarray, int, float]:
    """
    Run K-means with automatic k selection via silhouette score.

    Returns: (labels, best_k, best_silhouette_score)
    """
    if crop_rgb.size == 0:
        return np.array([]), 0, 0.0

    crop_lab = rgb2lab(crop_rgb)
    pixels = crop_lab.reshape(-1, 3)

    # Need at least 2 samples per cluster
    min_samples = k_range[1] * 2
    if len(pixels) < min_samples:
        # Fall back to k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        return labels.reshape(crop_rgb.shape[:2]), 2, 0.0

    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(pixels):
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)

        # Silhouette score requires at least 2 clusters with >1 sample each
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            continue

        try:
            score = silhouette_score(pixels, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except ValueError:
            continue

    if best_labels is None:
        # Fallback
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=3)
        best_labels = kmeans.fit_predict(pixels)
        best_k = 2
        best_score = 0.0

    return best_labels.reshape(crop_rgb.shape[:2]), best_k, best_score


def run_otsu(crop_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run Otsu thresholding on grayscale/luminance.

    Returns: (binary_labels, threshold_value)
    """
    if crop_rgb.size == 0:
        return np.array([]), 0.0

    gray = rgb2gray(crop_rgb)

    try:
        thresh = threshold_otsu(gray)
        labels = (gray > thresh).astype(int)
        return labels, thresh
    except ValueError:
        # Image may be uniform
        return np.zeros(gray.shape, dtype=int), 0.5


def run_hdbscan(crop_rgb: np.ndarray, min_cluster_size: int = 50) -> Tuple[np.ndarray, int]:
    """
    Run HDBSCAN density-based clustering.

    Returns: (labels, n_clusters) where -1 = noise
    """
    if not HDBSCAN_AVAILABLE:
        return np.zeros(crop_rgb.shape[:2], dtype=int), 0

    if crop_rgb.size == 0:
        return np.array([]), 0

    crop_lab = rgb2lab(crop_rgb)
    pixels = crop_lab.reshape(-1, 3)

    # Adjust min_cluster_size based on image size
    actual_min_size = min(min_cluster_size, max(5, len(pixels) // 10))

    try:
        hdb = HDBSCAN(min_cluster_size=actual_min_size, min_samples=5)
        labels = hdb.fit_predict(pixels)

        # Count clusters (excluding noise label -1)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        return labels.reshape(crop_rgb.shape[:2]), n_clusters
    except Exception as e:
        print(f"    HDBSCAN error: {e}")
        return np.zeros(crop_rgb.shape[:2], dtype=int), 0


def run_all_clustering_methods(crop_rgb: np.ndarray) -> Dict[str, Tuple[np.ndarray, dict]]:
    """
    Run all 3 clustering methods on a crop.

    Returns: dict of method_name -> (labels, info_dict)
    """
    results = {}

    # 1. K-means with auto-k via silhouette
    labels_km, best_k, sil_score = run_kmeans_auto_k(crop_rgb)
    results['kmeans'] = (labels_km, {'k': best_k, 'silhouette': sil_score})

    # 2. Otsu thresholding
    labels_otsu, thresh = run_otsu(crop_rgb)
    results['otsu'] = (labels_otsu, {'threshold': thresh})

    # 3. HDBSCAN
    labels_hdb, n_clusters = run_hdbscan(crop_rgb)
    results['hdbscan'] = (labels_hdb, {'n_clusters': n_clusters})

    return results


def run_kmeans_fixed_k(crop_rgb: np.ndarray, k: int) -> np.ndarray:
    """
    Run K-means with fixed K, return labels.
    """
    if crop_rgb.size == 0:
        return np.array([])

    crop_lab = rgb2lab(crop_rgb)
    pixels = crop_lab.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(crop_rgb.shape[:2])


def run_hdbscan_thumbnail(thumbnail_rgb: np.ndarray, blur_sigma: float = 5.0) -> Tuple[np.ndarray, int, int]:
    """
    Run HDBSCAN on full thumbnail with blur.
    min_cluster_size = 1% of pixels.

    Returns: (labels_img, n_clusters, largest_cluster_label)
    """
    if not HDBSCAN_AVAILABLE:
        return np.zeros(thumbnail_rgb.shape[:2], dtype=int), 0, 0

    from skimage.filters import gaussian

    h, w = thumbnail_rgb.shape[:2]
    n_pixels = h * w

    # Apply blur
    blurred = (gaussian(thumbnail_rgb, sigma=blur_sigma, channel_axis=2) * 255).astype(np.uint8)

    # Convert to LAB
    thumb_lab = rgb2lab(blurred)
    pixels = thumb_lab.reshape(-1, 3)

    # min_cluster_size = 1% of pixels
    min_cluster_size = max(100, int(n_pixels * 0.01))

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
    labels = hdb.fit_predict(pixels)
    labels_img = labels.reshape(h, w)

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    non_noise = [(u, c) for u, c in zip(unique, counts) if u >= 0]
    if non_noise:
        non_noise.sort(key=lambda x: -x[1])
        largest_label = non_noise[0][0]
    else:
        largest_label = -1

    n_clusters = len([u for u in unique if u >= 0])
    return labels_img, n_clusters, largest_label


def create_hdbscan_binary_overlay(crop_rgb: np.ndarray, labels: np.ndarray, largest_label: int, alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay: largest_cluster = green, everything_else = red.
    """
    if crop_rgb.size == 0 or labels.size == 0:
        return crop_rgb

    overlay = crop_rgb.copy().astype(np.float32)

    # Largest cluster = green overlay
    mask_largest = labels == largest_label
    # Everything else (including noise -1) = red overlay
    mask_other = labels != largest_label

    green = np.array([0, 255, 0])
    red = np.array([255, 0, 0])

    overlay[mask_largest] = overlay[mask_largest] * (1 - alpha) + green * alpha
    overlay[mask_other] = overlay[mask_other] * (1 - alpha) + red * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_bboxes_on_ax(
    ax,
    img: np.ndarray,
    bboxes: List,
    colors=None,
    linewidth: int = 2,
    title: str = ""
):
    """Draw bboxes on a matplotlib axis."""
    ax.imshow(img)
    h, w = img.shape[:2]

    if colors is None:
        colors = ["red"] * len(bboxes)
    elif isinstance(colors, str):
        colors = [colors] * len(bboxes)

    for i, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = bbox
        # Convert to pixel coords
        py1 = int(y1 / 1000 * h)
        px1 = int(x1 / 1000 * w)
        py2 = int(y2 / 1000 * h)
        px2 = int(x2 / 1000 * w)

        color = colors[i % len(colors)]
        rect = Rectangle(
            (px1, py1), px2 - px1, py2 - py1,
            linewidth=linewidth, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def generate_case_page(
    pdf: PdfPages,
    wsi_id: str,
    orientation_runs: Dict[int, Path],
    thumbnail_path: Path,
    kmeans_k: int = 2
):
    """Generate 2 PDF pages for a single WSI case.

    Page 1: Overview (thumbnail, orientation detections, merged bboxes)
    Page 2: Clustering comparison (K=2, K=3, Otsu, HDBSCAN per bbox)
    """
    from matplotlib.gridspec import GridSpec

    # Load original (0°) thumbnail
    thumbnail = np.array(Image.open(thumbnail_path))

    # Collect bboxes from each orientation
    orientation_bboxes = {}  # rotation -> list of bboxes (original coords)
    orientation_thumbnails = {}  # rotation -> thumbnail image

    for rotation in [0, 90, 180, 270]:
        if rotation in orientation_runs:
            run_dir = orientation_runs[rotation]
            orientation_bboxes[rotation] = load_bboxes_from_run(run_dir)
            # Load rotated thumbnail from experiment
            thumb_path = run_dir / "thumbnail.png"
            if thumb_path.exists():
                orientation_thumbnails[rotation] = np.array(Image.open(thumb_path))
            else:
                orientation_thumbnails[rotation] = None
        else:
            orientation_bboxes[rotation] = []
            orientation_thumbnails[rotation] = None

    # Transform all bboxes to 0° space
    transformed_bboxes = []  # list of (bbox, source_rotation)
    for rotation, bboxes in orientation_bboxes.items():
        for bbox in bboxes:
            transformed = transform_bbox_to_rot0(bbox, rotation)
            transformed_bboxes.append((transformed, rotation))

    # Merge overlapping bboxes
    all_transformed = [b[0] for b in transformed_bboxes]
    merged_bboxes = merge_overlapping_bboxes(all_transformed)

    # =========================================================================
    # PAGE 1: Overview
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle(f"Overview: {wsi_id}", fontsize=14, fontweight="bold")
    gs1 = GridSpec(3, 4, figure=fig1, hspace=0.3, wspace=0.2)

    # Row 1: Original thumbnail (span 2 cols) + stats
    ax_thumb = fig1.add_subplot(gs1[0, :2])
    ax_thumb.imshow(thumbnail)
    ax_thumb.set_title("Original Thumbnail", fontsize=10)
    ax_thumb.axis("off")

    ax_stats = fig1.add_subplot(gs1[0, 2:])
    ax_stats.axis("off")
    stats_text = (
        f"Statistics:\n\n"
        f"  0° boxes: {len(orientation_bboxes.get(0, []))}\n"
        f"  90° boxes: {len(orientation_bboxes.get(90, []))}\n"
        f"  180° boxes: {len(orientation_bboxes.get(180, []))}\n"
        f"  270° boxes: {len(orientation_bboxes.get(270, []))}\n"
        f"  ─────────────\n"
        f"  Total: {len(transformed_bboxes)}\n"
        f"  Merged: {len(merged_bboxes)}\n\n"
        f"Legend:\n"
        f"  Red = 0°, Blue = 90°\n"
        f"  Green = 180°, Orange = 270°\n"
        f"  Purple = Merged"
    )
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment="top", family="monospace")

    # Row 2: Orientation detections (4 panels)
    for i, rotation in enumerate([0, 90, 180, 270]):
        ax = fig1.add_subplot(gs1[1, i])
        thumb = orientation_thumbnails.get(rotation)
        if thumb is not None:
            bboxes = orientation_bboxes.get(rotation, [])
            draw_bboxes_on_ax(
                ax, thumb, bboxes,
                colors=ORIENTATION_COLORS[rotation],
                title=f"Flash {rotation}° ({len(bboxes)} boxes)"
            )
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
            ax.set_title(f"Flash {rotation}° (missing)")
            ax.axis("off")

    # Row 3: Transformed + Merged bboxes
    ax_trans = fig1.add_subplot(gs1[2, :2])
    colors_for_transformed = [ORIENTATION_COLORS[rot] for _, rot in transformed_bboxes]
    draw_bboxes_on_ax(
        ax_trans, thumbnail,
        [b[0] for b in transformed_bboxes],
        colors=colors_for_transformed,
        title=f"Transformed to 0° ({len(transformed_bboxes)} boxes)"
    )

    ax_merged = fig1.add_subplot(gs1[2, 2:])
    draw_bboxes_on_ax(
        ax_merged, thumbnail, merged_bboxes,
        colors="purple",
        linewidth=3,
        title=f"Merged Hulls ({len(merged_bboxes)} regions)"
    )

    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    # =========================================================================
    # PAGE 2: Clustering Comparison
    # =========================================================================
    n_merged = len(merged_bboxes)
    if n_merged == 0:
        # No bboxes - skip page 2
        return {
            "wsi_id": wsi_id,
            "orientation_bbox_counts": {r: len(b) for r, b in orientation_bboxes.items()},
            "total_transformed": len(transformed_bboxes),
            "merged_count": 0,
            "merged_bboxes": [],
            "clustering_results": [],
        }

    # Run HDBSCAN on full thumbnail (once, shared for all crops)
    print(f"    Running HDBSCAN on thumbnail...", end=" ", flush=True)
    hdbscan_labels, hdbscan_n_clusters, hdbscan_largest = run_hdbscan_thumbnail(thumbnail)
    print(f"{hdbscan_n_clusters} clusters, largest={hdbscan_largest}")

    # Create figure: 5 columns (Original, K=2, K=3, Otsu, HDBSCAN), n_merged rows
    fig2 = plt.figure(figsize=(20, 4 * n_merged))
    fig2.suptitle(f"Clustering Comparison: {wsi_id}", fontsize=14, fontweight="bold")
    gs2 = GridSpec(n_merged, 5, figure=fig2, hspace=0.3, wspace=0.2)

    clustering_results = []

    for i, bbox in enumerate(merged_bboxes):
        crop = extract_crop_from_bbox(thumbnail, bbox)

        if crop.size == 0:
            for j in range(5):
                ax = fig2.add_subplot(gs2[i, j])
                ax.text(0.5, 0.5, "Empty", ha="center", va="center")
                ax.axis("off")
            continue

        # Extract HDBSCAN crop from thumbnail-level labels
        h, w = thumbnail.shape[:2]
        y1, x1, y2, x2 = bbox
        py1 = max(0, min(int(y1 / 1000 * h), h))
        px1 = max(0, min(int(x1 / 1000 * w), w))
        py2 = max(0, min(int(y2 / 1000 * h), h))
        px2 = max(0, min(int(x2 / 1000 * w), w))
        hdbscan_crop_labels = hdbscan_labels[py1:py2, px1:px2]

        # Column 0: Original crop
        ax_orig = fig2.add_subplot(gs2[i, 0])
        ax_orig.imshow(crop)
        ax_orig.set_title(f"BBox {i+1}: Original", fontsize=9)
        ax_orig.axis("off")

        # Column 1: K-means K=2
        ax_k2 = fig2.add_subplot(gs2[i, 1])
        labels_k2 = run_kmeans_fixed_k(crop, k=2)
        if labels_k2.size > 0:
            overlay_k2 = create_cluster_overlay(crop, labels_k2)
            ax_k2.imshow(overlay_k2)
            ax_k2.set_title("K-means K=2", fontsize=9)
        else:
            ax_k2.text(0.5, 0.5, "Failed", ha="center", va="center")
        ax_k2.axis("off")

        # Column 2: K-means K=3
        ax_k3 = fig2.add_subplot(gs2[i, 2])
        labels_k3 = run_kmeans_fixed_k(crop, k=3)
        if labels_k3.size > 0:
            overlay_k3 = create_cluster_overlay(crop, labels_k3)
            ax_k3.imshow(overlay_k3)
            ax_k3.set_title("K-means K=3", fontsize=9)
        else:
            ax_k3.text(0.5, 0.5, "Failed", ha="center", va="center")
        ax_k3.axis("off")

        # Column 3: Otsu
        ax_otsu = fig2.add_subplot(gs2[i, 3])
        labels_otsu, thresh = run_otsu(crop)
        if labels_otsu.size > 0:
            overlay_otsu = create_cluster_overlay(crop, labels_otsu)
            ax_otsu.imshow(overlay_otsu)
            ax_otsu.set_title(f"Otsu (t={thresh:.2f})", fontsize=9)
        else:
            ax_otsu.text(0.5, 0.5, "Failed", ha="center", va="center")
        ax_otsu.axis("off")

        # Column 4: HDBSCAN (binary: largest vs rest)
        ax_hdb = fig2.add_subplot(gs2[i, 4])
        if hdbscan_crop_labels.size > 0 and HDBSCAN_AVAILABLE:
            overlay_hdb = create_hdbscan_binary_overlay(crop, hdbscan_crop_labels, hdbscan_largest)
            ax_hdb.imshow(overlay_hdb)
            ax_hdb.set_title(f"HDBSCAN ({hdbscan_n_clusters} cls)", fontsize=9)
        else:
            ax_hdb.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax_hdb.axis("off")

        clustering_results.append({
            'bbox_idx': i,
            'bbox': bbox,
            'otsu_threshold': thresh,
            'hdbscan_n_clusters': hdbscan_n_clusters,
        })

    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    return {
        "wsi_id": wsi_id,
        "orientation_bbox_counts": {r: len(b) for r, b in orientation_bboxes.items()},
        "total_transformed": len(transformed_bboxes),
        "merged_count": len(merged_bboxes),
        "merged_bboxes": merged_bboxes,
        "clustering_results": clustering_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess bbox detections from orientation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python postprocess_foreground_bboxes.py --output-pdf postprocessed_bboxes.pdf

    # Use Pro model instead of Flash
    python postprocess_foreground_bboxes.py --model pro --output-pdf pro_bboxes.pdf
"""
    )
    parser.add_argument(
        "--model",
        choices=["flash", "pro"],
        default="flash",
        help="Model to analyze (default: flash)"
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        default="postprocessed_bboxes.pdf",
        help="Output PDF path (default: postprocessed_bboxes.pdf)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="postprocessed_bboxes",
        help="Output directory for JSON data (default: postprocessed_bboxes)"
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=2,
        help="Number of clusters for k-means (default: 2)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases to process (for testing)"
    )

    args = parser.parse_args()

    # Collect all WSI cases
    print(f"Scanning {STAGE1_DIR} for {args.model} experiments...")
    cases = collect_all_wsi_cases(args.model)
    print(f"Found {len(cases)} WSI cases")

    if args.limit:
        cases = cases[:args.limit]
        print(f"Limited to {args.limit} cases")

    if not cases:
        print("No cases found. Exiting.")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    print(f"\nGenerating {args.output_pdf}...")
    all_results = []

    with PdfPages(args.output_pdf) as pdf:
        for i, wsi_id in enumerate(cases):
            print(f"[{i+1}/{len(cases)}] Processing {wsi_id}...")

            # Find orientation runs
            orientation_runs = find_orientation_runs(wsi_id, args.model)
            if not orientation_runs:
                print(f"  Skipping: no orientation runs found")
                continue

            # Find thumbnail
            thumbnail_path = THUMBNAIL_DIR / f"{wsi_id}.png"
            if not thumbnail_path.exists():
                print(f"  Skipping: no cached thumbnail")
                continue

            # Generate page
            result = generate_case_page(
                pdf, wsi_id, orientation_runs, thumbnail_path, args.kmeans_k
            )
            all_results.append(result)

            # Save per-case JSON
            case_dir = output_dir / wsi_id
            case_dir.mkdir(parents=True, exist_ok=True)
            with open(case_dir / "aggregated_bboxes.json", "w") as f:
                json.dump(result, f, indent=2)

    # Save summary
    summary = {
        "model": args.model,
        "total_cases": len(all_results),
        "kmeans_k": args.kmeans_k,
        "results": all_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Generated {args.output_pdf} with {len(all_results)} pages")
    print(f"Per-case data saved to {output_dir}/")


if __name__ == "__main__":
    main()
