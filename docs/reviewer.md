# Reviewer

Canonical reviewer entrypoints:

- Batch: `run_vlm_reviewer_batch.py`
- Single item: `vlm_reviewer.py`

Run these commands in the `path-agent` conda environment for this repo.

## Batch Workflow

1. Run foreground method first.
2. Point reviewer batch to baseline outputs.
3. Review `results.csv` / `results.jsonl` and apply your rerun policy.

Example:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/paper_foreground \
  --output-root runs/paper_reviewer \
  --batch-name paper_reviewer_v1 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

Default batch/single-item review uses OpenRouter
`google/gemini-3-flash-preview`, `prompts/calibration_reviewer.txt`,
OpenRouter reasoning effort `high`, temperature `0.0`, and strict thresholding:
`precision_pass = precision > --qc-precision-threshold`,
`recall_pass = recall > --qc-recall-threshold`, and
`overall_pass = precision_pass and recall_pass`.

For distilled-classifier routing, use reviewer QC as the gate before downstream
linear-probe experiments. Tol Blue is the hard OOD acceptance stain. If any
required tissue-core bbox has `qc.overall_pass=false`, or if the reviewer inputs
are too low-resolution to judge, route the case to the heavier repo VLM
foreground path with `--stage6-icl-k 1` and `--stage2-force-read-l0`.

## TRIDENT / External Mask Workflow

TRIDENT emits contour JPGs and GeoJSON contours, not this repo's Stage 3
crop/mask layout. If the user gives a TRIDENT contour path, infer the
corresponding `contours_geojson/<slide>.geojson`, resolve the source WSI from
the anonymous slide ID, then use this repo's Stage 1 VLM bbox detector to
identify tissue cores. The Stage 1 VLM route is the same OpenRouter route used
by the foreground pipeline:

```bash
python scripts/export_trident_reviewer_inputs.py \
  --contour /data2/vj724/path-agent/outputs/trident_output_hest_task1/contours/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64.jpg \
  --resolve-only
```

```bash
python detect_foreground_regions_from_wsi_thumbnail.py \
  --wsi /resolved/source_slide.svs \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --rotations 0 90 180 270 \
  --wsi-reader auto
```

This writes `stage1_output/<slide>/<model>/<timestamp>/bboxes.json`.
Rasterize the WSI-level TRIDENT contours into each Stage 1 tissue-core bbox:

```bash
python scripts/export_trident_reviewer_inputs.py \
  --contour /data2/vj724/path-agent/outputs/trident_output_hest_task1/contours/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64.jpg \
  --stage1-run-dir stage1_output/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64/google_gemini_3_flash_preview/<timestamp> \
  --output-root runs/trident_reviewer_inputs \
  --wsi-manifest /vol/biomedic3/histopatho/win_share/all_svs_fpaths.csv \
  --min-review-long-edge 1024
```

The exporter also accepts `--stage1-bboxes-json`, `--geojson`,
`--trident-job-dir`, and explicit `--wsi`. If `--wsi` is omitted, it resolves
`anon_<uuid>.svs` via known local WSI manifests, preferring the WSI-root
manifest `/vol/biomedic3/histopatho/win_share/all_svs_fpaths.csv`. If Stage 1
bboxes are omitted,
the exporter falls back to legacy per-contour-feature crops, which is usually
not the right unit for tissue-core review.

Review input quality rule: inspect `bboxes/*/stage3/metadata.json`. If the saved
crop max dimension (`quality.crop_long_edge`) is below 1024, or
`quality.needs_force_read_l0_review=true`, rerun that export with
`--force-read-l0` when the level-0 bbox size is below the
`--max-l0-read-mpix` guardrail. If you only need to test the high-resolution
read path and do not need reviewer-ready PNGs, add `--skip-image-save` to avoid
the PNG encoding cost; those metadata-only outputs are not reviewer inputs until
PNG files are written.

Then review the exported inputs:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/trident_reviewer_inputs \
  --run-selection latest \
  --output-root runs/paper_reviewer \
  --batch-name trident_reviewer_v1 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

If a case has multiple exported runs, pass `--run-name <run_id>` or
`--run-pattern <glob>` so paid reviewer calls target the intended inputs rather
than the lexicographically latest run directory.

## Single-Item Workflow

```bash
python vlm_reviewer.py \
  --crop /path/to/crop.png \
  --mask /path/to/mask.png \
  --overlay /path/to/overlay.png \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

## Auto-Context Stage 7 High-Resolution Bbox Workflow

If the foreground run did not use `--stage2-force-read-l0`, the original
`bboxes/*/stage3/crop.png` files may be thumbnail-derived and too blurry for
review. Export reviewer inputs from the final Stage 7 patch-grid mask instead:

```bash
python scripts/export_auto_context_reviewer_inputs.py \
  --run-dir runs/auto_context_pilot/<case>/<run_id> \
  --output-root runs/auto_context_reviewer_inputs \
  --max-dim 1024 \
  --padding-frac 0.02
```

This writes:

```text
runs/auto_context_reviewer_inputs/<case>/<run_id>_stage7_l0_review/bboxes/<bbox>/stage3/crop.png
runs/auto_context_reviewer_inputs/<case>/<run_id>_stage7_l0_review/bboxes/<bbox>/stage3/mask.png
runs/auto_context_reviewer_inputs/<case>/<run_id>_stage7_l0_review/bboxes/<bbox>/stage3/overlay.png
```

Then run the batch reviewer on the exported root. OpenRouter is useful when
Vertex model access is unavailable:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/auto_context_reviewer_inputs \
  --run-selection latest \
  --output-root runs/reviewer \
  --batch-name auto_context_stage7_review_v1 \
  --prompt-file prompts/calibration_reviewer.txt \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --reasoning-effort high \
  --max-concurrent-requests 2 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

When the user gives a natural-language request like "review the auto-context
run at `<run-dir>`", treat `<run-dir>` as the run directory containing
`stage7/tissue_mask_post.npy` and `bboxes/*`. If existing
`bboxes/*/stage3/crop.png` files are known to be high-resolution because the
run used `--stage2-force-read-l0` and they are visually sufficient, review the
run root directly. Otherwise export Stage 7 high-resolution reviewer inputs
with `scripts/export_auto_context_reviewer_inputs.py` and review the exported
`runs/auto_context_reviewer_inputs` root.

## Output Schema (Batch)

`results.jsonl` stores one JSON object per reviewed item. Common fields include:

- `case_id`
- `bbox_id`
- `reviewer_decision`
- `confidence`
- `reasoning`
- `qc.precision`
- `qc.recall`
- `qc.precision_pass`
- `qc.recall_pass`
- `qc.overall_pass`
- file references used for review input

`results.csv` contains the same decision payload in tabular form for
sorting/filtering, including `qc_precision_pass`, `qc_recall_pass`, and
`qc_overall_pass`.

## Prompt Files

Reviewer prompt files kept in repo:

- `prompts/reviewer.txt`
- `prompts/calibration_reviewer.txt`: default for numeric precision/recall calibration.
- `prompts/subjective_reviewer.txt`: narrative expert assessment when notes matter more than sortable percentages.
- `prompts/objective_reviewer.txt`: legacy prompt; do not use for foreground review unless explicitly comparing old runs.
