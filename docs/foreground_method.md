# Foreground Method

This repository uses a staged method for foreground tissue segmentation in whole slide images.

Canonical orchestrator: `run_auto_context.py`

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

For the PER-188 distilled-classifier plan, route foreground segmentation in two
modes once a runnable distilled classifier exists:

1. Run the distilled foreground/background patch classifier first and review the
   resulting masks, using Tol Blue as the hard OOD stain gate.
2. If reviewer QC fails, is uncertain, or lacks adequate-resolution review
   inputs, fall back to `run_auto_context.py` with `--stage6-icl-k 1` and
   `--stage2-force-read-l0`.

MobileNetV3 student checkpoint artifacts and their original train/inference
scripts are available outside this repo under
`/vol/biomedic3/vj724/wsi-agents/distilled_student_models_20260225/`, mirrored
locally at `/data2/vj724/wsi-agents/distilled_student_models_20260225_package/`;
see `docs/data/distilled_student_models_20260225.md`. The package can run
student inference on a Stage6-like patch grid. This repo still does not have a
native `run_auto_context.py` Stage 6 adapter that swaps VLM patch calls for the
student model and then continues Stage 7 unchanged.

## Why Stages Exist

- Stage 1 reduces search space cheaply at thumbnail resolution.
- Stage 2 optionally estimates which artifact classes should be modeled.
- Stage 3 creates a fast classical foreground prior inside Stage 1 bboxes.
- Stage 4 grounds candidate points for FG/BG/artifact exemplars.
- Stage 5 re-ranks/curates high-quality ICL examples.
- Stage 6 performs final patch-level VLM classification (optionally gated by Stage 3).
- Stage 7 cleans masks/class maps into final postprocessed tissue masks.
- A future distilled runner should replace Stage 6 VLM calls while preserving
  the Stage 6/7 output contract.

For one-off distilled checks against existing reviewer masks, use:

- `scripts/build_distilled_stage6_grid_from_reviewer_masks.py` to create a
  Stage6-like `patches.csv` grid from reviewer `stage3/mask.png` files.
- The packaged
  `/data2/vj724/wsi-agents/distilled_student_models_20260225_package/scripts/student_patch_distill_export_test_overlays.py`
  script to run the checkpoint on that grid.
- `scripts/build_distilled_student_comparison_visuals.py` to assemble reviewer
  comparison PNGs, an HTML index, and by-bbox metrics.
- `scripts/build_trident_vs_distilled_comparison_visuals.py` to compare
  TRIDENT foreground overlays, auto-context Stage 6 overlays, and both
  distilled student overlays for the same Stage 1 bboxes.

## Canonical Invocation

Use wrapper + args file for reproducibility:

```bash
bash scripts/run_paper_method.sh
```

Or call directly:

```bash
python run_auto_context.py \
  --wsi /path/to/slide.svs \
  --output-root runs/paper_method \
  --skip-stage2 \
  --stage3-method kmeans
```

## Related Docs

- `docs/scripts.md`
- `docs/stage_outputs.md`
- `docs/reviewer.md`
