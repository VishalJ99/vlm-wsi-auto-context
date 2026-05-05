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

## TRIDENT / External Mask Workflow

TRIDENT emits contour JPGs and GeoJSON contours, not this repo's Stage 3
crop/mask layout. If the user gives a TRIDENT contour path, infer the
corresponding `contours_geojson/<slide>.geojson`, resolve the source WSI from
the anonymous slide ID, then export TRIDENT contours first:

```bash
python scripts/export_trident_reviewer_inputs.py \
  --contour /data2/vj724/path-agent/outputs/trident_output_hest_task1/contours/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64.jpg \
  --output-root runs/trident_reviewer_inputs
```

The exporter also accepts `--geojson`, `--trident-job-dir`, and explicit
`--wsi`. If `--wsi` is omitted, it resolves `anon_<uuid>.svs` via known local
WSI manifests, including `/data2/vj724/wsi-agents/all_svs_fpaths.csv`.

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
  --max-dim 2048 \
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
