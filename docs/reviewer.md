# Reviewer

Canonical reviewer entrypoints:

- Batch: `run_vlm_reviewer_batch.py`
- Single item: `vlm_reviewer.py`

## Batch Workflow

1. Run foreground method first.
2. Point reviewer batch to baseline outputs.
3. Review `results.csv` / `results.jsonl` and apply your rerun policy.

Example:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/paper_foreground \
  --output-root runs/paper_reviewer \
  --batch-name paper_reviewer_v1
```

## TRIDENT / External Mask Workflow

TRIDENT emits GeoJSON contours, not this repo's Stage 3 crop/mask layout. Export
TRIDENT contours first:

```bash
python scripts/export_trident_reviewer_inputs.py \
  --wsi /path/to/slide.svs \
  --trident-job-dir /path/to/trident_output \
  --output-root runs/trident_reviewer_inputs
```

Then review the exported inputs:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/trident_reviewer_inputs \
  --run-selection latest \
  --output-root runs/paper_reviewer \
  --batch-name trident_reviewer_v1
```

## Single-Item Workflow

```bash
python vlm_reviewer.py \
  --crop /path/to/crop.png \
  --mask /path/to/mask.png \
  --overlay /path/to/overlay.png
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
  --max-concurrent-requests 2
```

## Output Schema (Batch)

`results.jsonl` stores one JSON object per reviewed item. Common fields include:

- `case_id`
- `bbox_id`
- `reviewer_decision`
- `confidence`
- `reasoning`
- file references used for review input

`results.csv` contains the same decision payload in tabular form for sorting/filtering.

## Prompt Files

Reviewer prompt files kept in repo:

- `prompts/reviewer.txt`
- `prompts/calibration_reviewer.txt`: default for numeric precision/recall calibration.
- `prompts/subjective_reviewer.txt`: narrative expert assessment when notes matter more than sortable percentages.
- `prompts/objective_reviewer.txt`: legacy prompt; do not use for foreground review unless explicitly comparing old runs.
