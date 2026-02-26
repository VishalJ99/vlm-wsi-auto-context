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

## Single-Item Workflow

```bash
python vlm_reviewer.py \
  --crop /path/to/crop.png \
  --mask /path/to/mask.png \
  --overlay /path/to/overlay.png
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
- `prompts/objective_reviewer.txt`
- `prompts/subjective_reviewer.txt`
- `prompts/calibration_reviewer.txt`
