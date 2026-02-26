# Stage Outputs

Canonical output structures for kept foreground/reviewer scripts.

## Naming Conventions

- `{wsi_id}`: WSI basename
- `{model}`: model identifier sanitized for paths
- `{timestamp}`: `YYYYMMDD_HHMMSS`
- `{bbox_str}`: `x1_y1_x2_y2` in level-0 coordinates

Most runs emit:
- `metadata.json`
- `reproduce.txt`

## Stage 1
Script: `detect_foreground_regions_from_wsi_thumbnail.py`

Output root:

```text
stage1_output/{wsi_id}/{model}/{timestamp}/
```

Key files:
- `thumbnail.png`
- `bboxes.json`
- `bbox_overlay.png`
- `vlm_responses/`

## Stage 2 (optional)
Script: `run_artifact_qc_pipeline.py`

Output root:

```text
stage2_output/{wsi_id}/{model}/{timestamp}/
```

Per-bbox files under `{bbox_str}/` include:
- `bbox_region.png`
- `stage1_artifact_perception.json`
- `stage2_claim_evidence.json`
- `stage3_strength.json`
- `stage4_verdicts.json`

## Stage 3
Script: `run_color_segmentation.py`

Output root:

```text
stage3_output/{wsi_id}/{model}/{timestamp}/
```

Per-bbox files:
- `crop.png`
- `mask.png`
- `overlay.png` (unless disabled)

## Stage 4
Script: `find_icl_regions.py`

Output root:

```text
stage4_output/{wsi_id}/{bbox_str}/{model}/{timestamp}/
```

Key files:
- `region_thumbnail.png`
- `rot_*/points.json`
- `rot_*/points_overlay.png`
- `points_overlay_all.png` (if TTA aggregation used)

## Stage 5
Script: `reranker.py`

Output root:

```text
stage5_output/{wsi_id}/{bbox_str}/{timestamp}_{config_hash}/
```

Key files/directories:
- `{class_name}/` curated example patches
- `intermediate/candidate_manifest.json`
- `metadata.json`

Optional descriptor generation:
- `generate_stage5_descriptions.py` writes a class-description JSON into a Stage 5 run directory.

## Stage 6
Script: `run_vlm_bbox_inference.py`

Output root:

```text
stage6_output/{wsi_id}/{model}/{timestamp}_{config_hash}/
```

Key files:
- `mask.npy`, `mask.png`, `overlay.png`
- `patches.csv`
- `class_map.npy`, `quality_map.npy`
- `class_palette.json`

## Stage 7
Script: `postprocess_mask.py`

Pipeline-integrated outputs are written under the run directory, typically:

```text
{run_dir}/stage7/
{run_dir}/bboxes/{bbox_str}/stage7/
```

Key files:
- `mask.npy` (WSI-level postprocessed mask)
- `metadata.json`
- per-bbox postprocess artifacts

## Reviewer Batch Outputs
Script: `run_vlm_reviewer_batch.py`

Batch output root:

```text
<output_root>/<batch_name>/
```

Key files:
- `results.csv`
- `results.jsonl`
- `manifest.json`
- optional overlay/debug artifacts
