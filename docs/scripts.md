# Scripts Reference

This document lists the scripts intentionally kept in the release-pruned repository.

`python <script> --help` is the source of truth for options.

## Canonical Run Wrappers

### `scripts/run_paper_method.sh`
Runs `run_foreground_method.py` using one-argument-per-line args from `configs/paper_method.args`.

```bash
bash scripts/run_paper_method.sh [optional_args_file]
```

### `scripts/run_paper_reviewer.sh`
Runs `run_vlm_reviewer_batch.py` using one-argument-per-line args from `configs/paper_reviewer.args`.

```bash
bash scripts/run_paper_reviewer.sh [optional_args_file]
```

## Orchestrator

### `run_foreground_method.py`
Stage 1-7 foreground method orchestrator.

### `run_auto_context_method.py`
Alias entrypoint for the same method runner.

```bash
python run_foreground_method.py --wsi /path/to/slide.svs [options]
```

Typical paper policy:

```bash
python run_foreground_method.py \
  --wsi /path/to/slide.svs \
  --skip-stage2 \
  --stage3-method kmeans
```

## Reviewer

### `run_vlm_reviewer_batch.py`
Batch reviewer runner over baseline segmentation outputs.

```bash
python run_vlm_reviewer_batch.py --baseline-dir <dir> [options]
```

### `vlm_reviewer.py`
Single-item reviewer primitive (crop + mask + optional overlay).

```bash
python vlm_reviewer.py --crop <png> --mask <png> [options]
```

## Stage Scripts (Method-Addressable)

### Stage 1
- `detect_foreground_regions_from_wsi_thumbnail.py`

### Stage 2 (legacy/optional)
- `run_artifact_qc_pipeline.py`

### Stage 3
- `run_color_segmentation.py`

### Stage 4
- `find_icl_regions.py`

### Stage 5
- `reranker.py`
- `generate_stage5_descriptions.py`

### Stage 6
- `run_vlm_bbox_inference.py`

### Stage 7
- `postprocess_mask.py`
- `postprocess_foreground_bboxes.py` (bbox-level utility)

## Supporting Utility

### `materialize_stage1_from_xml.py`
Converts XML-derived detections into Stage 1-compatible outputs for downstream method use.

## Credential Notes

For Gemini Vertex paths in kept scripts:
- Credential CLI flags are optional.
- If omitted, scripts use `GOOGLE_APPLICATION_CREDENTIALS` when Vertex mode is enabled.
- Missing credentials in Vertex mode should produce explicit runtime errors.

For OpenRouter paths:
- Set `OPENROUTER_API_KEY` (or compatible fallback env var used by the specific script).
