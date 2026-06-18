from __future__ import annotations

import numpy as np

from postprocess_mask import fill_small_holes


def test_fill_small_holes_limits_by_enclosed_background_area() -> None:
    mask = np.ones((5, 8), dtype=bool)
    mask[2, 2] = False
    mask[2, 5:7] = False

    one_patch_only = fill_small_holes(mask, max_hole_size=1, connectivity=4)
    assert one_patch_only[2, 2]
    assert not one_patch_only[2, 5]
    assert not one_patch_only[2, 6]

    two_patch_allowed = fill_small_holes(mask, max_hole_size=2, connectivity=4)
    assert two_patch_allowed[2, 2]
    assert two_patch_allowed[2, 5]
    assert two_patch_allowed[2, 6]


def test_fill_small_holes_zero_preserves_unrestricted_fill_mode() -> None:
    mask = np.ones((5, 8), dtype=bool)
    mask[2, 2] = False
    mask[2, 5:7] = False

    filled = fill_small_holes(mask, max_hole_size=0, connectivity=4)

    assert filled.all()


def test_fill_small_holes_does_not_fill_border_background() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    mask[2, 2] = False

    filled = fill_small_holes(mask, max_hole_size=1, connectivity=4)

    assert not filled[0, 0]
    assert filled[2, 2]
