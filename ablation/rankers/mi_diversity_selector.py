# ABOUTME: Mutual Information diversity selector for background patch selection.
# ABOUTME: Selects patches with maximum pairwise diversity using histogram-based MI.
"""
MI Diversity Selector - Maximum Diversity Background Patch Selection

Uses pairwise mutual information on RGB histograms to select the most
diverse set of background patches. No VLM call needed - purely computational.

Key insight from LOGBOOK: Background patches rarely have false positives,
so diversity (covering different background types) is more important than
perceptual quality validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import PatchRanker


class MIDiversityRanker(PatchRanker):
    """
    Mutual Information diversity-based patch selection.

    Computes pairwise MI on RGB histograms and greedily selects patches
    that maximize diversity (minimize MI to already-selected patches).

    This is the recommended strategy for background patches where false
    positives are rare and diversity is the main goal.
    """

    def __init__(
        self,
        n_bins: int = 32,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize MI diversity ranker.

        Args:
            n_bins: Number of histogram bins per channel (default: 32)
            seed: Random seed for tie-breaking (optional)
            **kwargs: Additional parameters (unused)
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Store MI matrix for metadata
        self._last_mi_matrix = None

    def rank(
        self,
        patches: List[Image.Image],
        patch_names: List[str],
        label: str,
        k: int,
        icl_dir: Path,
    ) -> Tuple[List[int], str]:
        """
        Select k most diverse patches using MI-based greedy selection.

        Args:
            patches: List of candidate patch images
            patch_names: List of filenames for each patch
            label: 'foreground' or 'background' (works for both)
            k: Number of patches to select
            icl_dir: Output ICL directory

        Returns:
            (selected_indices, reasoning)
        """
        n = len(patches)

        # Handle trivial cases
        if n == 0:
            return [], "No candidates provided"

        if n <= k:
            selected_indices = list(range(n))
            self._save_selected_patches(patches, patch_names, selected_indices, label, icl_dir)
            return selected_indices, f"All {n} candidates selected (n <= k)"

        # Compute RGB histograms
        histograms = [self._compute_rgb_histogram(p) for p in patches]

        # Compute pairwise MI matrix
        mi_matrix = self._compute_mi_matrix(histograms)
        self._last_mi_matrix = mi_matrix

        # Greedy selection: maximize diversity (minimize MI)
        selected_indices = self._greedy_diverse_selection(mi_matrix, k)

        # Build reasoning string
        avg_mi = np.mean([mi_matrix[i, j] for i in selected_indices for j in selected_indices if i != j])
        reasoning = (
            f"Selected {k} patches with maximum diversity. "
            f"Average pairwise MI among selected: {avg_mi:.4f}. "
            f"Selection order: {selected_indices}"
        )

        # Save selected patches
        self._save_selected_patches(patches, patch_names, selected_indices, label, icl_dir)

        return selected_indices, reasoning

    def rank_multiclass(
        self,
        patches_by_class: Dict[str, Tuple[List[Image.Image], List[str]]],
        k_per_class: Dict[str, int],
        class_descriptions: Optional[Dict[str, str]],
        icl_dir: Path,
    ) -> Dict[str, Tuple[List[int], str]]:
        """
        Rank patches for multiple classes by running per-class MI selection.

        This does not perform a single merged ranking; it applies the MI
        strategy independently for each class.
        """
        results: Dict[str, Tuple[List[int], str]] = {}
        for class_name, (patches, patch_names) in patches_by_class.items():
            k = k_per_class.get(class_name, 0)
            if k <= 0:
                results[class_name] = ([], "k_per_class <= 0, no selections requested")
                continue
            selected_indices, reasoning = self.rank(
                patches=patches,
                patch_names=patch_names,
                label=class_name,
                k=k,
                icl_dir=icl_dir,
            )
            results[class_name] = (selected_indices, reasoning)
        return results

    def _compute_rgb_histogram(self, image: Image.Image) -> np.ndarray:
        """
        Compute normalized RGB histogram for an image.

        Args:
            image: PIL Image

        Returns:
            Flattened histogram array (n_bins * 3,)
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Compute histogram for each channel
        histograms = []
        for channel in range(3):
            hist, _ = np.histogram(
                img_array[:, :, channel].flatten(),
                bins=self.n_bins,
                range=(0, 256),
                density=True
            )
            histograms.append(hist)

        # Concatenate and normalize
        combined = np.concatenate(histograms)
        return combined / (combined.sum() + 1e-10)

    def _compute_mi_matrix(self, histograms: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise mutual information matrix.

        MI(X, Y) = sum_i sum_j p(x_i, y_j) * log(p(x_i, y_j) / (p(x_i) * p(y_j)))

        For histograms, we approximate this using histogram intersection
        which correlates with MI for color distributions.

        Args:
            histograms: List of normalized histogram arrays

        Returns:
            n x n MI matrix (symmetric)
        """
        n = len(histograms)
        mi_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Use histogram intersection as MI proxy
                # Higher intersection = more similar = higher MI
                mi = self._histogram_intersection(histograms[i], histograms[j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _histogram_intersection(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """
        Compute histogram intersection (sum of element-wise minimum).

        Higher value = more similar distributions.

        Args:
            h1, h2: Normalized histogram arrays

        Returns:
            Intersection score in [0, 1]
        """
        return np.minimum(h1, h2).sum()

    def _greedy_diverse_selection(self, mi_matrix: np.ndarray, k: int) -> List[int]:
        """
        Greedily select k patches that maximize pairwise diversity.

        Algorithm:
        1. Start with the most unique patch (lowest avg MI to others)
        2. Iteratively add patch with minimum max-MI to already selected
           (i.e., the patch most different from all selected patches)

        Args:
            mi_matrix: n x n pairwise MI matrix
            k: Number to select

        Returns:
            List of selected indices
        """
        n = mi_matrix.shape[0]
        selected = []
        remaining = set(range(n))

        # Start with patch that has lowest average MI to others (most unique)
        avg_mi = mi_matrix.sum(axis=1) / (n - 1)
        first = int(np.argmin(avg_mi))
        selected.append(first)
        remaining.remove(first)

        # Greedily add patches
        while len(selected) < k and remaining:
            min_max_mi = float('inf')
            best = None

            for idx in remaining:
                # Max MI to any already selected patch
                max_mi_to_selected = max(mi_matrix[idx, s] for s in selected)

                if max_mi_to_selected < min_max_mi:
                    min_max_mi = max_mi_to_selected
                    best = idx
                elif max_mi_to_selected == min_max_mi and best is not None:
                    # Tie-break randomly
                    if self._rng.random() < 0.5:
                        best = idx

            if best is not None:
                selected.append(best)
                remaining.remove(best)

        return selected

    def get_config(self) -> Dict[str, Any]:
        """Return ranker configuration for metadata logging."""
        config = super().get_config()
        config.update({
            "n_bins": self.n_bins,
            "seed": self.seed,
        })
        return config

    def get_mi_matrix(self) -> Optional[np.ndarray]:
        """Get the MI matrix from the last ranking call."""
        return self._last_mi_matrix
