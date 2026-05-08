# ABOUTME: Shared image reader helpers with explicit WSI/single-slice backend selection.
# ABOUTME: Provides consistent level-0/native patch reads for WSI and non-pyramidal images.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import tifffile  # type: ignore

    TIFFFILE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tifffile = None  # type: ignore[assignment]
    TIFFFILE_AVAILABLE = False

try:
    from cucim import CuImage  # type: ignore

    CUCIM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CuImage = None  # type: ignore[assignment]
    CUCIM_AVAILABLE = False

try:
    import openslide  # type: ignore

    OPENSLIDE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    openslide = None  # type: ignore[assignment]
    OPENSLIDE_AVAILABLE = False

try:
    from isyntax import ISyntax  # type: ignore

    ISYNTAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ISyntax = None  # type: ignore[assignment]
    ISYNTAX_AVAILABLE = False

VALID_WSI_READERS = ("auto", "openslide", "cucim", "isyntax", "image")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SingleImageSlide:
    """Adapter exposing a flat image through the WSI reader interface."""

    def __init__(self, image_path: str):
        self.path = str(image_path)
        self._image = self._load_rgb_image(self.path)
        self.dimensions = self._image.size
        self.level_count = 1
        self.level_dimensions = [self.dimensions]
        self.level_downsamples = [1.0]

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] > 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.dtype == np.uint8:
            return np.asarray(Image.fromarray(arr).convert("RGB"))

        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.zeros((*arr.shape, 3), dtype=np.uint8)
            low, high = np.nanpercentile(finite, [1, 99])
            if high <= low:
                low = float(np.nanmin(finite))
                high = float(np.nanmax(finite))
            scaled = (arr - low) / max(float(high - low), 1e-6)
            plane = np.clip(scaled * 255, 0, 255).astype(np.uint8)
            return np.stack([plane, plane, plane], axis=-1)

        channels = []
        for channel_index in range(min(arr.shape[-1], 3)):
            plane = arr[..., channel_index]
            finite = plane[np.isfinite(plane)]
            if finite.size == 0:
                low, high = 0.0, 0.0
            else:
                low, high = np.nanpercentile(finite, [1, 99])
            scaled = (plane - low) / max(float(high - low), 1e-6)
            channels.append(np.clip(scaled * 255, 0, 255).astype(np.uint8))
        while len(channels) < 3:
            channels.append(channels[-1])
        return np.stack(channels[:3], axis=-1)

    @classmethod
    def _load_rgb_image(cls, image_path: str) -> Image.Image:
        path = Path(image_path)
        if path.suffix.lower() in {".tif", ".tiff"} and TIFFFILE_AVAILABLE:
            arr = tifffile.imread(str(path))
            return Image.fromarray(cls._normalize_array(arr), mode="RGB")
        with Image.open(path) as image:
            if image.mode in {"F", "I", "I;16"}:
                arr = np.asarray(image)
                return Image.fromarray(cls._normalize_array(arr), mode="RGB")
            return image.convert("RGB")

    def read_region(self, location, level_or_size, maybe_size=None):
        if maybe_size is None:
            level = 0
            size = level_or_size
        else:
            level = int(level_or_size)
            size = maybe_size
        if level != 0:
            raise ValueError("SingleImageSlide only has level 0")

        x, y = [int(v) for v in location]
        width, height = [int(v) for v in size]
        if width < 1 or height < 1:
            width, height = 1, 1

        canvas = Image.new("RGB", (width, height), "white")
        crop_box = (
            max(0, x),
            max(0, y),
            min(self.dimensions[0], x + width),
            min(self.dimensions[1], y + height),
        )
        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
            crop = self._image.crop(crop_box)
            canvas.paste(crop, (crop_box[0] - x, crop_box[1] - y))
        return canvas

    def close(self) -> None:
        self._image.close()


def is_ndpi_path(wsi_path: str) -> bool:
    return Path(str(wsi_path)).suffix.lower() == ".ndpi"


def is_isyntax_path(wsi_path: str) -> bool:
    return Path(str(wsi_path)).suffix.lower() == ".isyntax"


def is_single_image_path(wsi_path: str) -> bool:
    return Path(str(wsi_path)).suffix.lower() in IMAGE_SUFFIXES


def normalize_wsi_reader(reader: str) -> str:
    value = (reader or "auto").strip().lower()
    if value not in VALID_WSI_READERS:
        raise ValueError(
            f"Invalid WSI reader '{reader}'. Use one of: {', '.join(VALID_WSI_READERS)}."
        )
    return value


def _unwrap_cucim_region(data: Any) -> np.ndarray:
    if hasattr(data, "__iter__") and not isinstance(data, np.ndarray):
        for batch in data:
            data = batch
            break
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected region shape from cuCIM: {arr.shape}")
    return arr


def _load_openslide(wsi_path: str):
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("OpenSlide backend requested but openslide is not installed.")
    return openslide.OpenSlide(wsi_path), "openslide"  # type: ignore[union-attr]


def _load_cucim(wsi_path: str):
    if not CUCIM_AVAILABLE:
        raise RuntimeError("cuCIM backend requested but cucim is not installed.")
    return CuImage(wsi_path), "cucim"  # type: ignore[operator]


def _load_isyntax(wsi_path: str):
    if not ISYNTAX_AVAILABLE:
        raise RuntimeError("ISyntax backend requested but pyisyntax is not installed.")
    return ISyntax.open(wsi_path), "isyntax"  # type: ignore[union-attr]


def _load_image(wsi_path: str):
    return SingleImageSlide(wsi_path), "image"


def load_wsi(wsi_path: str, wsi_reader: str = "auto"):
    """
    Load WSI with explicit backend selection.

    Reader policy:
      - .isyntax files: always use pyisyntax.
      - openslide/cucim/isyntax/image: force that backend.
      - auto: use image for flat image files, prefer openslide for .ndpi,
        otherwise prefer cucim.
    """
    reader = normalize_wsi_reader(wsi_reader)
    errors = []

    if is_isyntax_path(wsi_path):
        reader = "isyntax"
    elif reader == "auto" and is_single_image_path(wsi_path):
        reader = "image"

    if reader == "openslide":
        return _load_openslide(wsi_path)
    if reader == "cucim":
        return _load_cucim(wsi_path)
    if reader == "isyntax":
        return _load_isyntax(wsi_path)
    if reader == "image":
        return _load_image(wsi_path)

    if is_ndpi_path(wsi_path):
        backends = ("openslide", "cucim")
    else:
        backends = ("cucim", "openslide")

    for backend in backends:
        try:
            if backend == "openslide":
                return _load_openslide(wsi_path)
            if backend == "isyntax":
                return _load_isyntax(wsi_path)
            return _load_cucim(wsi_path)
        except Exception as exc:
            errors.append(f"{backend}:{type(exc).__name__}:{exc}")

    raise RuntimeError(
        "Failed to open WSI with available readers. "
        f"Requested='{reader}', path='{wsi_path}', errors={' | '.join(errors)}"
    )


def close_wsi(wsi: Any, backend: str) -> None:
    if backend in {"openslide", "isyntax", "image"} and hasattr(wsi, "close"):
        try:
            wsi.close()
        except Exception:
            pass


def get_level0_dimensions(wsi: Any, backend: str) -> Tuple[int, int]:
    if backend == "cucim":
        resolutions = wsi.resolutions
        wsi_w, wsi_h = resolutions["level_dimensions"][0]
        return int(wsi_w), int(wsi_h)
    wsi_w, wsi_h = wsi.dimensions
    return int(wsi_w), int(wsi_h)


def get_pyramid_info(wsi: Any, backend: str) -> Dict[str, Any]:
    if backend == "cucim":
        res = wsi.resolutions
        level_dims = [
            (int(dim[0]), int(dim[1]))
            for dim in res.get("level_dimensions", [])
        ]
        level_downsamples = [float(v) for v in res.get("level_downsamples", [])]
        level_count = int(res.get("level_count", len(level_dims)))
    elif backend == "image":
        level_dims = [(int(wsi.dimensions[0]), int(wsi.dimensions[1]))]
        level_downsamples = [1.0]
        level_count = 1
    else:
        level_count = int(wsi.level_count)
        level_dims = [
            (int(dim[0]), int(dim[1]))
            for dim in wsi.level_dimensions
        ]
        level_downsamples = [float(v) for v in wsi.level_downsamples]

    if level_count <= 0 or not level_dims:
        raise RuntimeError("Invalid pyramid metadata: empty level structure")

    if len(level_downsamples) != len(level_dims):
        # Conservative fallback when backends return incomplete downsamples.
        level_downsamples = [float(2**i) for i in range(len(level_dims))]

    return {
        "level_count": int(level_count),
        "level_dimensions": level_dims,
        "level_downsamples": level_downsamples,
    }


def read_region_rgb(
    wsi: Any,
    backend: str,
    x: int,
    y: int,
    width: int,
    height: int,
    level: int = 0,
) -> np.ndarray:
    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    x_i = int(x)
    y_i = int(y)
    w_i = int(width)
    h_i = int(height)
    level_i = int(level)

    if backend == "cucim":
        region = wsi.read_region((x_i, y_i), (w_i, h_i), level=level_i)
        arr = _unwrap_cucim_region(region)
        return np.asarray(Image.fromarray(arr).convert("RGB"))

    if backend == "isyntax":
        arr = np.asarray(wsi.read_region(x_i, y_i, w_i, h_i, level=level_i))
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected region shape from ISyntax: {arr.shape}")
        return np.asarray(Image.fromarray(arr).convert("RGB"))

    region = wsi.read_region((x_i, y_i), level_i, (w_i, h_i))
    return np.asarray(region.convert("RGB"))
