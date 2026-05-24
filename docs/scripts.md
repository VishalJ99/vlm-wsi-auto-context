# Scripts Reference

This document lists the scripts intentionally kept in the release-pruned repository.

`python <script> --help` is the source of truth for options.

## Canonical Run Wrappers

### `scripts/run_paper_method.sh`
Runs `run_auto_context.py` using one-argument-per-line args from `configs/paper_method.args`.

```bash
bash scripts/run_paper_method.sh [optional_args_file]
```

### `scripts/run_paper_reviewer.sh`
Runs `run_vlm_reviewer_batch.py` using one-argument-per-line args from `configs/paper_reviewer.args`.

```bash
bash scripts/run_paper_reviewer.sh [optional_args_file]
```

## Orchestrator

### `run_auto_context.py`
Stage 1-7 foreground method orchestrator.

```bash
python run_auto_context.py --wsi /path/to/slide.svs [options]
```

Typical paper policy:

```bash
python run_auto_context.py \
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

### `scripts/stage1_detector_pilot_control.py`
Builds and operates the PER-188 Stage 1 detector pilot control plane:

```bash
python scripts/stage1_detector_pilot_control.py build-worklist
bash runs/stage1_detector_pilot_v1/commands/run_stage1_pilot.sh
python scripts/stage1_detector_pilot_control.py export-review-packet \
  --worklist-csv runs/stage1_detector_pilot_v1/worklists/manual_review_20.csv \
  --output-root runs/stage1_detector_pilot_v1
```

After manual review passes, the same script can build synthetic guard cases by
dropping one detector bbox, dry-run the guard request list, run the guard VLM,
and summarize the expected-missing-core results.

### `scripts/stage1_detection_review_pilot.py`
Runs a focused VLM smoke test over existing Stage 1 thumbnail detections. The
reviewer sees the source thumbnail plus Stage 1 overlay and returns slide-level
flags plus per-bbox localization grades:

```bash
PYTHONPATH=/data2/vj724/python_deps/openai_py310:$PYTHONPATH \
python scripts/stage1_detection_review_pilot.py run-detection-review
```

Outputs are written under
`runs/stage1_detector_pilot_v1/stage1_detection_review_v1/`, including raw
JSONL results, slide/bbox CSV summaries, a PDF visual packet, and
`reproduction.txt`.

The same script can run a targeted second-pass detector call using the previous
overlay and reviewer feedback:

```bash
PYTHONPATH=/data2/vj724/python_deps/openai_py310:$PYTHONPATH \
python scripts/stage1_detection_review_pilot.py run-feedback-redetect --index 70
```

### `scripts/stage1_high_recall_pilot.py`
Runs the PER-207 high-recall Stage 1 detector prompt across the balanced pilot
worklist using one raw orientation, then exports numeric-only raw overlays,
summary CSV/JSON, a PDF packet, logs, and `reproduction.txt`:

```bash
python scripts/stage1_high_recall_pilot.py --max-concurrent 4
```

### `materialize_stage1_from_xml.py`
Converts XML-derived detections into Stage 1-compatible outputs for downstream method use.

### `scripts/export_trident_reviewer_inputs.py`
Converts TRIDENT `contours_geojson/<slide>.geojson` foreground contours into this repo's
Stage 3-style reviewer inputs:

```bash
python scripts/export_trident_reviewer_inputs.py \
  --wsi /path/to/slide.svs \
  --trident-job-dir /path/to/trident_job \
  --output-root runs/trident_reviewer_inputs
```

The output can be passed directly to `run_vlm_reviewer_batch.py --baseline-dir`.

## Credential Notes

For Gemini Vertex paths in kept scripts:
- Credential CLI flags are optional.
- If omitted, scripts use `GOOGLE_APPLICATION_CREDENTIALS` when Vertex mode is enabled.
- Missing credentials in Vertex mode should produce explicit runtime errors.

For OpenRouter paths:
- Set `OPENROUTER_API_KEY` (or compatible fallback env var used by the specific script).
