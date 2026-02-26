# ABOUTME: Abstract base class for ICL patch ranking strategies.
# ABOUTME: Rankers select top-k patches from candidates based on quality criteria.
"""
PatchRanker Abstract Base Class

Defines the interface for ICL patch ranking strategies. Unlike samplers (which
operate on masks to select regions), rankers perform perceptual/semantic
evaluation of actual extracted patch images.

Usage:
    from ablation.rankers import get_ranker

    # Create VLM-based ranker
    ranker = get_ranker("vlm", backend="openrouter", model="google/gemini-2.0-flash-001")

    # Rank foreground candidates
    selected_indices, reasoning = ranker.rank(
        patches=fg_patches,
        label='foreground',
        k=3,
        icl_dir=output_dir
    )
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


class PatchRanker(ABC):
    """
    Abstract base class for ICL patch ranking strategies.

    Rankers receive a batch of candidate patches and select the top-k most
    suitable for ICL examples. They perform perceptual or computational
    evaluation to identify the best representatives of each class.

    Subclasses must implement the `rank` method which:
    1. Evaluates all candidate patches
    2. Selects the top-k best candidates
    3. Saves selected patches to the ICL directory
    4. Returns the selected indices and reasoning/explanation
    """

    def __init__(self, **kwargs):
        """
        Initialize ranker with optional configuration.

        Args:
            **kwargs: Strategy-specific parameters
        """
        pass

    @abstractmethod
    def rank(
        self,
        patches: List[Image.Image],
        patch_names: List[str],
        label: str,
        k: int,
        icl_dir: Path,
    ) -> Tuple[List[int], str]:
        """
        Rank patches, select top-k, and save to ICL directory.

        Args:
            patches: List of candidate patch images (PIL.Image)
            patch_names: List of filenames for each patch (for provenance)
            label: Class label - 'foreground' or 'background'
            k: Number of patches to select
            icl_dir: Output ICL directory (creates {icl_dir}/{label}/)

        Returns:
            Tuple of:
                - selected_indices: List of k indices into the patches list
                - reasoning: Explanation string (VLM reasoning or selection criteria)

        Side Effects:
            Saves selected patches to {icl_dir}/{label}/{patch_name}.png
        """
        pass

    @abstractmethod
    def rank_multiclass(
        self,
        patches_by_class: Dict[str, Tuple[List[Image.Image], List[str]]],
        k_per_class: Dict[str, int],
        class_descriptions: Optional[Dict[str, str]],
        icl_dir: Path,
    ) -> Dict[str, Tuple[List[int], str]]:
        """
        Rank patches for multiple classes, saving selections into class directories.

        Args:
            patches_by_class: Mapping of class -> (patches, patch_names)
            k_per_class: Mapping of class -> number of patches to select
            class_descriptions: Optional mapping of class -> description for prompts
            icl_dir: Output ICL directory (creates {icl_dir}/{class}/)

        Returns:
            Dict of class -> (selected_indices, reasoning)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return ranker configuration for metadata logging.

        Returns:
            Dict containing ranker type and parameters
        """
        return {
            "type": self.__class__.__name__,
        }

    def _save_selected_patches(
        self,
        patches: List[Image.Image],
        patch_names: List[str],
        selected_indices: List[int],
        label: str,
        icl_dir: Path,
    ) -> List[str]:
        """
        Save selected patches to ICL directory.

        Args:
            patches: All candidate patches
            patch_names: Filenames for each patch
            selected_indices: Indices of selected patches
            label: 'foreground' or 'background'
            icl_dir: Output directory

        Returns:
            List of saved file paths (relative to icl_dir)
        """
        output_dir = icl_dir / label
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx in selected_indices:
            if idx < len(patches):
                patch = patches[idx]
                name = patch_names[idx]
                output_path = output_dir / name
                patch.save(output_path)
                saved_paths.append(f"{label}/{name}")

        return saved_paths
