# Foreground Pipeline

This repository uses a staged pipeline for foreground tissue segmentation in whole slide images.

Canonical orchestrator: `run_foreground_pipeline.py`

## Stage Map

1. Stage 1: `detect_foreground_regions_from_wsi_thumbnail.py`
2. Stage 2 (legacy/optional): `run_artifact_qc_pipeline.py`
3. Stage 3: `run_color_segmentation.py`
4. Stage 4: `find_icl_regions.py`
5. Stage 5: `reranker.py`
6. Stage 6: `run_vlm_bbox_inference.py`
7. Stage 7: `postprocess_mask.py`

## Current Run Policy

For production-like paper runs:
- Use `--skip-stage2`
- Use `--stage3-method kmeans`

Reviewer-guided Stage 3 reruns are external/manual (not currently auto-wired in the orchestrator).

## Why Stages Exist

- Stage 1 reduces search space cheaply at thumbnail resolution.
- Stage 2 optionally estimates which artifact classes should be modeled.
- Stage 3 creates a fast classical foreground prior inside Stage 1 bboxes.
- Stage 4 grounds candidate points for FG/BG/artifact exemplars.
- Stage 5 re-ranks/curates high-quality ICL examples.
- Stage 6 performs final patch-level VLM classification (optionally gated by Stage 3).
- Stage 7 cleans masks/class maps into final postprocessed tissue masks.

## Canonical Invocation

Use wrapper + args file for reproducibility:

```bash
bash scripts/run_paper_foreground.sh
```

Or call directly:

```bash
python run_foreground_pipeline.py \
  --wsi /path/to/slide.svs \
  --output-root runs/paper_foreground \
  --skip-stage2 \
  --stage3-method kmeans
```

## Related Docs

- `docs/scripts.md`
- `docs/stage_outputs.md`
- `docs/reviewer.md`
