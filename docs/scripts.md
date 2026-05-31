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

### `scripts/run_detector_pipeline.py`
PER-207 arbitrary-WSI detector-oracle pipeline for bbox discovery and review.
It accepts exactly one input source:

- positional WSI path, WSI directory, or `.txt` worklist,
- `--wsi /path/to/slide.svs`,
- `--wsi-dir /path/to/wsis`,
- `--wsi-list /path/to/wsis.txt`.

Default execution is breadth-first: finish one stage across all WSIs before
moving to the next stage, with `--max-concurrent` controlling concurrent VLM
calls. `--batch-mode depth-first` processes one WSI through the full pipeline at
a time and parallelizes crop-level calls within that WSI.

Current integrated stage order:

1. Stage 1 thumbnail detection with the high-recall tissue-candidate prompt.
2. Stage 2a free-text missed/overcoverage review on the source thumbnail plus
   Stage 1 raw overlay.
3. Stage 2b text router over the Stage 2a review. By default, the first
   router pass is the Stage 3 trigger and the older adjudication pass is not
   run.
4. Optional Stage 3 feedback redetection on the original thumbnail plus raw
   overlay when the selected Stage 2b trigger is positive, or when
   `--force-stage3-redetect` is set.
5. Stage 4 deterministic merge/padding and optional high-resolution crop
   redetection.
6. Stage 5 classification-crop construction.
7. Stage 6 tissue/artifact classification.
8. Stage 7 comparative thumbnail-crop artifact filtering, unless
   `--skip-odd-one-out-filter` is set.

Current default pilot-100 command with the first-pass Stage 2b trigger:

```bash
eval "$(rg '^export (OPENROUTER_API_KEY|OPENAI_API_KEY)=' /homes/vj724/.zshrc)"
python scripts/run_detector_pipeline.py test-pipeline/pilot_100_wsis.txt \
  --output-dir test-pipeline-firstpass-default-skip-stage4 \
  --save-all-stage-artifacts \
  --batch-mode breadth-first \
  --max-concurrent 16 \
  --skip-repro \
  --skip-crop-redetect
```

The default Stage 2b routing now matches the reviewed first-pass-trigger
setting: `--stage2b-trigger-source first`. This is the setting used to produce
the current first-pass-trigger comparison packet:
`/data2/vj724/vlm-wsi-auto-context/test-pipeline-stage2b-firstpass-trigger-skip-stage4/visuals/stage2b_firstpass_trigger_downstream_comparison.pdf`.
Use `--stage2b-trigger-source adjudicated` only when reproducing the older
two-pass non-minor-failure gate.
That second pass was a false-positive-redetection reducer; it is no longer the
default because precision is now handled later by Stage 6 classification and,
for non-SV40 runs, Stage 7 odd-one-out filtering.
Some serialized Stage 2b fields still use the older `non_minor_detection_failure`
name for compatibility with existing PDFs/tables; in the current default flow,
read that value as the high-recall missed-tissue trigger for Stage 3.

`--skip-crop-redetect` skips the high-resolution Stage 4 redetection VLM calls.
The deterministic Stage 4 merge/normalization still runs because Stage 5 needs
normalized boxes.

`--skip-odd-one-out-filter` skips the final Stage 7 comparative thumbnail
filter. Final detections then include all Stage 6 tissue-positive boxes. The
alias `--skip-stage7-filter` is equivalent.
When Stage 7 runs, comparative removals are applied only if the odd-one-out
response parser returns an `ok*` status. Non-ok parses preserve all Stage 6
tissue-positive boxes, record the raw flags separately, and mark the case as
degraded instead of silently deleting detections.

Use this flag for SV40 runs and process SV40 cases as a separate worklist/output
root. SV40 control tissue can be real tissue while still looking different from
the target tissue, which breaks the Stage 7 homogeneous-tissue assumption and
can make valid control tissue look like an artifact outlier.

Single-WSI runs write `final_detected_bboxes.png`,
`detections.json`, and `reproduction.txt` directly under `--output-dir`.
Multi-WSI runs create one subdirectory per WSI filename stem, plus root-level
`summary.json`, `all_detections.json`, prompt copies, and aggregate JSONL/CSV
tables when `--save-all-stage-artifacts` is set.

`--reuse-existing` is conservative: a stage output is reused only when its
sidecar cache fingerprint matches the current pipeline version, model/backend,
prompt file hashes, relevant thresholds/skip flags, and stage inputs. Legacy
artifacts without a sidecar fingerprint are rerun rather than silently mixed
with new pipeline semantics.

The most recent full pilot-100 final-detections PDF predates the first-pass
default switch and used the older adjudicated trigger:
`/data2/vj724/vlm-wsi-auto-context/test-pipeline-integrated-skip-stage4/visuals/final_detections_pilot100.pdf`.

### `scripts/export_detector_training_dataset.py`
PER-240 exporter from detector-pipeline outputs into supervised thumbnail bbox
datasets. It treats COCO as the canonical detector interchange format and writes
YOLO labels as a derived view for Ultralytics-style training.

```bash
python scripts/export_detector_training_dataset.py export test-pipeline \
  --output-dir runs/detector_training_datasets/test_pipeline_pilot100_coco_yolo_v1 \
  --overwrite
```

Input root must contain `all_detections.json` or per-case `detections.json`.
The exporter reads each case's clean Stage 1 thumbnail from
`paths.thumbnail_path`, the aggregate Stage 1 case table, or the standard
per-case `intermediate_stage_artifacts/stage1_thumbnail_detection/thumbnail.png`
location. If thumbnails are missing, rerun the detector pipeline with
`--save-all-stage-artifacts`; the exporter does not reconstruct thumbnails by
reading the WSI.

Default output contract:

- `images/{train,val,test}/*.png`: copied clean thumbnails, the model inputs X.
- `annotations/instances_{train,val,test,all}.json`: COCO detection JSON with
  one class, `tissue_candidate`, and pixel-space `xywh` boxes.
- `labels/{train,val,test}/*.txt` and `dataset.yaml`: derived YOLO labels in
  `[class, x_center, y_center, width, height]` normalized image coordinates.
- `manifests/manifest.jsonl`: one JSON object per case preserving source WSI,
  split, group, thumbnail path, and all original normalized 0-1000
  `[y_min, x_min, y_max, x_max]` boxes alongside pixel and YOLO conversions.
- `manifests/manifest.csv`: one row per bbox for quick table inspection.
- `manifests/cases.csv`: one row per thumbnail/case.
- `summary.json`, `validation.json`, and `reproduction.txt`.

The default split policy is patient-slide aware: `--group-by auto` derives a
`patient_<id>_slide_<id>` group when available, keeping serial stains from the
same patient/slide in the same train/val/test split. Use `--group-by case` only
for smoke tests or deliberately image-level splits. Use `--image-mode symlink`
when the downstream trainer can safely dereference source thumbnails in place.

Validate an exported dataset with:

```bash
python scripts/export_detector_training_dataset.py validate \
  runs/detector_training_datasets/test_pipeline_pilot100_coco_yolo_v1
```

Framework ports should consume this export rather than re-parse the detector
pipeline directly. For Ultralytics, start from `dataset.yaml`. For RF-DETR or
other COCO-native trainers, start from `annotations/instances_*.json` and the
relative `file_name` fields. If a framework needs a different object, build a
thin adapter from the COCO or manifest outputs so the normalized source boxes
remain auditable.

### `scripts/train_yolo_detector.py`
PER-241 Ultralytics YOLO distillation runner over a dataset exported by
`scripts/export_detector_training_dataset.py`.

```bash
/data2/vj724/venvs/vlm-wsi-yolo-ultralytics-per241/bin/python \
  scripts/train_yolo_detector.py \
  --dataset-dir runs/detector_training_datasets/test_pipeline_pilot100_coco_yolo_v1 \
  --output-root runs/detector_distillation/yolo_pilot100_yolo11n_img1024_e60_v1 \
  --model /data2/vj724/model_weights/ultralytics/yolo11n.pt \
  --epochs 60 \
  --imgsz 1024 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --patience 25 \
  --cache \
  --conf 0.10 \
  --overwrite
```

The runner verifies the Ultralytics dataset layout, trains from the requested
pretrained YOLO weights, evaluates on the selected split, and writes:

- `metrics_summary.json`: Ultralytics metrics, project-specific metrics, run
  config, environment, and best-weight path.
- `project_error_metrics.json`: bbox agreement against exported labels at
  `--conf`.
- `project_error_metrics_by_conf.json`: the same project-specific metrics after
  filtering saved predictions at confidence thresholds from
  `--metric-conf-thresholds`.
- `per_case_project_metrics.csv`: per-thumbnail misses, false predictions,
  duplicate/fragment/overmerge counts, and stain labels.
- `review/prediction_overlays.pdf`: review packet with green exported boxes and
  red YOLO predictions.
- `reproduction.txt`: command, dataset path, environment, and key output paths.

PER-241 pilot runs are under
`/data2/vj724/vlm-wsi-auto-context/runs/detector_distillation/`. The summary
root is
`runs/detector_distillation/yolo_pilot100_per241_summary_v1/summary.csv`.
On the pilot-100 test split, `yolo11n` at `imgsz=1024` was the best practical
setting among the small grid. `yolo11s` overpredicted at low confidence and did
not improve test mAP, so model-size gains were second-order or negative on this
small dataset.

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
