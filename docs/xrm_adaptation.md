# XRM slice adaptation

This branch adds a flat-image reader path for x-ray microscopy (XRM) slice images.
It treats a TIFF/PNG/JPEG slice as a one-level native coordinate frame, so existing
`bbox_level0` and `point_l0` fields remain usable as native slice pixel coordinates.

For the Crick XRM TIFF stack observed under `/Volumes/proj-mrc-mm`, `1900 x 1900`
is the height/width of one reconstructed 2D z slice. It is not a WSI pyramid
level-0 size. A `1900 x 1900` slice tiles to roughly:

- `15 x 15` patches at `128 px`
- `9 x 9` patches at `224 px`
- `4 x 4` patches at `512 px`

## Reader

Use `--wsi-reader image` or `--allstage-wsi-reader image` for flat slices. The
`auto` reader also chooses this path for `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`,
and `.bmp`.

The image reader loads a single slice into memory, exposes one pyramid level, and
uses native pixel coordinates for crops. Float and 16-bit TIFFs are percentile
normalized for VLM display.

## Prompt files

- Stage 1 bbox detection: `prompts/xrm_bbox_tissue_cores.txt`
- Stage 4 point grounding: `prompts/stage4/xrm_default.txt`
- Stage 6 patch classification: `prompts/xrm_patch_classify.txt`
- Stage 6 class definitions: `prompts/xrm_foreground_class_defs.json`

These avoid light-microscopy assumptions such as stain, glass slide background,
nuclei, H&E/PAS, and histopathology-specific artifacts.

## Example commands

Stage 1, tissue/sample bbox detection on one slice:

```bash
OPENROUTER_API_KEY=... python detect_foreground_regions_from_wsi_thumbnail.py \
  --wsi /path/to/slice_0950.tif \
  --wsi-reader image \
  --backend openrouter \
  --model qwen/qwen3-vl-8b-instruct \
  --prompt prompts/xrm_bbox_tissue_cores.txt \
  --max-dim 1024 \
  --rotations 0 \
  --save-bbox-region
```

Stage 6, foreground/background patch classification with one in-context example
per class:

```bash
OPENROUTER_API_KEY=... python run_vlm_bbox_inference.py \
  --stage5-run /path/to/stage5_bbox_run \
  --wsi-reader image \
  --backend openrouter \
  --model qwen/qwen3-vl-8b-instruct \
  --prompt-template prompts/xrm_patch_classify.txt \
  --class-defs prompts/xrm_foreground_class_defs.json \
  --icl-k 1 \
  --patch-size 128 \
  --vlm-image-size 128 \
  --max-workers 1 \
  --query-batch-size 1 \
  --skip-dvc-check
```

For a `1900 x 1900` XRM slice, prefer `128` or `224` over the historical `512`
default if the goal is dense foreground sampling rather than a coarse sanity
check.
