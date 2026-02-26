#!/usr/bin/env python3
# ABOUTME: HistoQC-style patch blur scoring helpers for in-repo reuse.
# ABOUTME: Computes blur_score from Laplace+Gaussian sharpness map on RGB patch arrays.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, laplace


@dataclass
class PatchBlurResult:
    """Container for patch-level blur outputs."""

    sharpness_map: np.ndarray
    blur_mask: np.ndarray
    blur_score: float
    sharp_score: float


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize image to a 3-channel RGB ndarray."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, repeats=3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")


def compute_blur_from_patch(
    patch_rgb: np.ndarray,
    sigma: float = 2.0,
    pixel_threshold: float = 0.05,
) -> PatchBlurResult:
    """
    Compute HistoQC-style blur maps and scalar scores for one patch.

    `blur_score` is mean(blur_mask), where blur_mask is defined by:
      gaussian(abs(laplace(gray)), sigma=sigma) <= pixel_threshold
    """
    patch_rgb = ensure_rgb(patch_rgb)
    gray = rgb2gray(patch_rgb)
    sharpness_map = gaussian(np.abs(laplace(gray)), sigma=sigma)
    blur_mask = sharpness_map <= pixel_threshold
    blur_score = float(np.mean(blur_mask))
    sharp_score = float(np.mean(sharpness_map))
    return PatchBlurResult(
        sharpness_map=sharpness_map,
        blur_mask=blur_mask,
        blur_score=blur_score,
        sharp_score=sharp_score,
    )

