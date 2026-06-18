# PER-250: Seed Scale500 Bboxes With All-Foreground Stage 3 Masks

## Status

Pending human review.

## Context

PER-250 uses finalized scale500 detector boxes as the bbox source for
`run_auto_context.py` foreground segmentation. The scale500 detector run saved
high-resolution candidate crops under each case's
`intermediate_stage_artifacts/stage5_post_redetect_merge_and_crop/candidates/`
directory.

`run_auto_context.py --resume` can skip Stage 1/2 when the canonical `stage1/`
and `bboxes/<bbox>/stage2/` files already exist. If Stage 3 is missing, it runs
thumbnail KMeans from the Stage 1 thumbnail crop. A local smoke case showed this
can produce a very sparse foreground prior for scale500 boxes.

## Decision

The scale500 adapter defaults to writing Stage 3 `crop.png`, `mask.png`,
`overlay.png`, and `metadata.json` from the exact high-resolution bbox crop,
with the Stage 3 mask set to all foreground.

## Rationale

The scale500 detector and crop classifier have already accepted these bboxes as
tissue-focused candidates. Seeding Stage 3 as all foreground avoids a low-
resolution thumbnail KMeans gate suppressing patches before Stage 6 VLM
classification. This keeps recall high and lets Stage 6 perform the foreground /
background decision over the full trusted bbox.

## Consequences

- `run_auto_context.py --resume` skips Stage 1, Stage 2, and Stage 3 for seeded
  cases; the next active step is Stage 4 point grounding.
- Stage 6 patch classification sees all patches inside the seeded bbox unless a
  later config explicitly disables or changes Stage 3 gating.
- Runtime and VLM cost may be higher than sparse KMeans gating because the full
  bbox is eligible for Stage 6.
- The adapter keeps `--seed-stage3 none` available when a real Stage 3 KMeans
  prior is desired.

## Evidence

Smoke export:

```bash
python scripts/export_scale500_detector_to_auto_context.py \
  --scale500-run-dir runs/detector_pipeline_scale500_v1/sv40_skip_odd \
  --output-root tmp/scale500_adapter_smoke \
  --run-id smoke_scale500_seeded_stage3 \
  --case anon_24fc1539_4651_4352_aab5_b6183e1bd56f \
  --overwrite
```

Resume check:

```bash
/vol/biomedic3/vj724/.conda/envs/path-agent/bin/python run_auto_context.py \
  --wsi /vol/biomedic2/bkainz/histopatho/win_share/2024-07-03/anon_24fc1539-4651-4352-aab5-b6183e1bd56f.svs \
  --output-root /data2/vj724/vlm-wsi-auto-context/tmp/scale500_adapter_smoke \
  --run-id smoke_scale500_seeded_stage3 \
  --resume \
  --skip-stage2 \
  --max-stage 3 \
  --skip-dvc-check
```

Observed resume output: Stage 1, Stage 2, and Stage 3 were all resume hits.
