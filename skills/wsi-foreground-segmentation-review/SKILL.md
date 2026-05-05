---
name: wsi-foreground-segmentation-review
description: Use when running or planning histopathology WSI foreground/background segmentation with TRIDENT, the vlm-wsi-auto-context foreground pipeline, VLM segmentation review, or future distilled foreground/background patch-classifier routes. Includes route selection, IO contracts, default TTA policies, reviewer inputs, and distillation gaps.
---

# WSI Foreground Segmentation + Review

Use this skill for repeat histopathology foreground/background work: segment tissue foreground, review mask quality, escalate hard cases to VLM, and plan VLM-to-student distillation.

## First Checks

- Start in `/data2/vj724/vlm-wsi-auto-context` unless the user specifies another checkout.
- Use the `path-agent` conda environment for this repo's scripts, including `run_auto_context.py`, `scripts/export_auto_context_reviewer_inputs.py`, `scripts/export_trident_reviewer_inputs.py`, `run_vlm_reviewer_batch.py`, and `vlm_reviewer.py`.
- Read `README.md`, `docs/foreground_method.md`, `docs/stage_outputs.md`, and `docs/reviewer.md` when commands or output paths matter.
- Check git/DVC/Linear per project instructions before edits or artifact-producing runs.
- Treat review outputs, exported masks, trained models, and generated datasets as persistent artifacts: write `reproduction.txt` next to output roots and track with DVC where the repo uses DVC.

## Route Selection

There are three foreground/background routes:

1. **TRIDENT route**: fastest classical/deep baseline with TRIDENT IO. Outputs GeoJSON foreground contours, contour thumbnails, and HDF5 foreground patch coordinates.
2. **Repo VLM route**: this repo's staged auto-context method. Outputs per-bbox Stage 3 masks, Stage 6 VLM patch classifications, and Stage 7 postprocessed tissue masks.
3. **Distilled route**: intended future route. Run bbox detection and context stages, then replace Stage 6 VLM patch classification with a trained lightweight FG/BG patch classifier. This repo did not contain that train/inference script when this skill was drafted; verify with `rg -i "distill|train|student|classifier"`.

Use TRIDENT first for a cheap baseline. Use repo VLM when TRIDENT masks fail quality review or when artifacts/background need VLM reasoning. Use distilled only after a teacher-labeled dataset and classifier runner exist.

## Route 1: TRIDENT

Local TRIDENT source has been seen at `/data2/vj724/TRIDENT`. TRIDENT has its own conda environment: run TRIDENT commands in `conda activate trident`, then switch back to this repo's environment (`path-agent` here) for exporter, reviewer, and VLM pipeline commands. Do not assume `import trident` works inside the repo env.

```bash
cd /data2/vj724/TRIDENT
conda activate trident

python run_batch_of_slides.py \
  --task seg \
  --wsi_dir /path/to/wsis \
  --wsi_ext .svs \
  --job_dir /path/to/trident_hest \
  --segmenter hest \
  --seg_conf_thresh 0.5 \
  --gpu 0 \
  --max_workers 0 \
  --skip_errors

python run_batch_of_slides.py \
  --task coords \
  --wsi_dir /path/to/wsis \
  --wsi_ext .svs \
  --job_dir /path/to/trident_hest \
  --mag 20 \
  --patch_size 512 \
  --overlap 0 \
  --max_workers 0
```

Key TRIDENT outputs:

- `contours_geojson/<slide>.geojson`: foreground polygons, QuPath-compatible.
- `contours/<slide>.jpg`: segmentation QC thumbnail.
- `20x_512px_0px_overlap/patches/<slide>_patches.h5`: foreground coordinates in level-0 `(x, y)`.

For reviewer QC, convert one slide's TRIDENT contours into this repo's
Stage3-compatible review layout. If the user says "review this segmentation"
and provides a path under `trident_output_*/contours/*.jpg` or
`trident_output_*/contours_geojson/*.geojson`, recognize it as a TRIDENT
contour output and use `scripts/export_trident_reviewer_inputs.py`.

If the user provides only a contour JPG path, pass it as `--contour`; the
exporter infers the sibling GeoJSON path. If the user provides only a GeoJSON
path, pass it as `--geojson`. If no `--wsi` is supplied, the exporter resolves
the `anon_<uuid>.svs` slide through known local manifests, especially
`/data2/vj724/wsi-agents/all_svs_fpaths.csv`, then loads the WSI through
`utils.wsi_backend`.

```bash
cd /data2/vj724/vlm-wsi-auto-context
conda activate path-agent

python scripts/export_trident_reviewer_inputs.py \
  --contour /data2/vj724/path-agent/outputs/trident_output_hest_task1/contours/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64.jpg \
  --output-root runs/trident_reviewer_inputs \
  --max-dim 2048 \
  --padding-frac 0.08
```

Then run the existing batch reviewer:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/trident_reviewer_inputs \
  --run-selection latest \
  --output-root runs/reviewer \
  --batch-name trident_hest_review_v1 \
  --prompt-file prompts/calibration_reviewer.txt \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --reasoning-effort high \
  --temperature 0.0 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

If TRIDENT includes too much background, try `--seg_conf_thresh 0.4`, `--remove_artifacts`, or `grandqc`; if it misses tissue, compare `hest` vs `grandqc` and inspect `contours/*.jpg`.

TRIDENT review building blocks:

- A contour JPG is only a QC thumbnail; the reviewable segmentation is the matching GeoJSON.
- The WSI pixels come from the source `.svs`, resolved by `anon_<uuid>.svs` through the WSI manifest.
- The exporter reads each contour bbox from the WSI, rasterizes the GeoJSON polygon at crop resolution, writes `crop.png`, `mask.png`, `overlay.png`, and `metadata.json`, then the batch reviewer consumes that Stage3-compatible layout.

## Route 2: Repo VLM Foreground Pipeline

Canonical entrypoint: `run_auto_context.py`. Wrapper: `scripts/run_paper_method.sh`.

User-preferred defaults for this workflow:

- Stage 1 bbox detection: Gemini Flash, four orientation TTA.
- Stage 4 point grounding: TTA enabled; do not pass `--stage4-no-tta`.
- If using ICL with `--stage6-icl-k 1`, add `--stage2-force-read-l0` so Stage 4 point grounding and review crops use higher-resolution bbox regions.
- If running a no-ICL baseline with `--stage6-icl-k 0`, leave `--stage2-force-read-l0` unset unless the user explicitly wants higher-res reviewer crops.
- Stage 6 patch-classification TTA is separate (`--stage6-rotations`) and defaults to `0`; do not enable it unless evaluating that cost/quality tradeoff.
- OpenRouter keys may be exported from `~/.zshrc`; non-interactive `bash` commands may need to source that file or pass `--stage*-api-key` explicitly. Never print the key value in logs.
- Default Stage 6 local patch classification expects a vLLM OpenAI-compatible server at `http://localhost:8000/v1`.

Start the default Qwen3-VL 8B server only when needed, from an environment that has `vllm` installed. Keep Hugging Face and vLLM caches on local `/data2` storage; putting model weights or compile caches on `/vol/biomedic3` can make startup appear hung in NFS wait before any GPU memory is allocated.

```bash
export HF_HOME=/data2/vj724/hf_cache
export HF_HUB_CACHE=/data2/vj724/hf_cache/hub
export VLLM_CACHE_ROOT=/data2/vj724/vllm_cache

vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype float16 \
  --trust-remote-code
```

Before a full run, verify it is reachable:

```bash
curl -sS http://localhost:8000/v1/models
```

To use multiple available GPUs, start one vLLM server per GPU on separate ports and point separate WSI or bbox-level jobs at different `--stage6-vllm-url` values. Check port availability first with `ss -ltn 'sport = :8001'`; do not assume 8001 is free.

Example:

```bash
python run_auto_context.py \
  --wsi /path/to/slide.svs \
  --output-root runs/foreground \
  --run-id foreground_flash_icl1 \
  --skip-stage2 \
  --stage3-method kmeans \
  --stage1-backend openrouter \
  --stage1-model google/gemini-3-flash-preview \
  --stage1-rotations 0 90 180 270 \
  --stage2-force-read-l0 \
  --stage4-backend openrouter \
  --stage4-model google/gemini-3-flash-preview \
  --stage5-vlm-backend gemini \
  --stage5-vlm-model gemini-3.1-pro \
  --stage5-gemini-thinking-level High \
  --stage6-icl-k 1 \
  --stage6-rotations 0
```

Review repo pipeline outputs directly:

```bash
python run_vlm_reviewer_batch.py \
  --baseline-dir runs/foreground \
  --run-selection latest \
  --output-root runs/reviewer \
  --batch-name foreground_review_v1 \
  --prompt-file prompts/calibration_reviewer.txt \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --reasoning-effort high \
  --temperature 0.0 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

When the user asks in natural language to review an auto-context pipeline output
directory, use the auto-context run directory as the input. A suitable input is
the directory that contains `stage7/tissue_mask_post.npy`, `stage1/`, and
`bboxes/`, for example:

```text
/data2/vj724/vlm-wsi-auto-context/runs/auto_context_pilot/he_patient_003_slide_003/he_p003_s003_icl0_20260430_175702
```

If the run used `--stage6-icl-k 0`, did not use `--stage2-force-read-l0`, or
the existing `bboxes/*/stage3/crop.png` images are visibly thumbnail-derived,
export high-resolution reviewer inputs from the final Stage 7 masks before
reviewing. This avoids sending blurry Stage 3 crops to the reviewer:

```bash
python scripts/export_auto_context_reviewer_inputs.py \
  --run-dir runs/foreground/<case>/<run_id> \
  --output-root runs/auto_context_reviewer_inputs \
  --max-dim 2048 \
  --padding-frac 0.02

python run_vlm_reviewer_batch.py \
  --baseline-dir runs/auto_context_reviewer_inputs \
  --run-selection latest \
  --output-root runs/reviewer \
  --batch-name foreground_stage7_review_v1 \
  --prompt-file prompts/calibration_reviewer.txt \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --reasoning-effort high \
  --max-concurrent-requests 2 \
  --qc-precision-threshold 0.9 \
  --qc-recall-threshold 0.9
```

If a run already used `--stage2-force-read-l0` and the saved Stage 3 bbox crops
are high-resolution enough for visual review, skip the exporter and point
`--baseline-dir` at the run root's parent or existing review-compatible baseline
root. The reviewer batch discovers Stage3-compatible bbox inputs under
`<baseline>/<case>/<run_id>/bboxes/<bbox_id>/stage3/`.

Batch reviewer discovers:

```text
<baseline>/<case>/<YYYYMMDD_HHMMSS>/bboxes/<bbox_id>/stage3/crop.png
<baseline>/<case>/<YYYYMMDD_HHMMSS>/bboxes/<bbox_id>/stage3/mask.png
<baseline>/<case>/<YYYYMMDD_HHMMSS>/bboxes/<bbox_id>/stage3/overlay.png
```

## Route 3: Distilled Patch Classifier

As drafted, this repo has VLM teacher outputs but no trainer for a distilled FG/BG patch classifier. Existing useful teacher artifacts:

- Stage 6 `patches.csv`: per-patch VLM labels and metadata.
- Stage 6 `class_map.npy` / `quality_map.npy`: per-grid labels.
- Stage 7 masks: postprocessed tissue masks.

Do not claim distilled inference is implemented until the repo has at least:

- A dataset exporter from Stage 6 teacher outputs to patch image paths + FG/BG labels.
- A train script for a lightweight patch classifier.
- An inference runner that writes Stage6-compatible `class_map.npy`, `patches.csv`, and metadata so Stage 7 can run unchanged.

The likely clean integration is: keep Stage 1 bbox detection, optional Stage 3 gating, optional Stage 4/5 context generation for teacher runs, then swap Stage 6 VLM calls for the distilled classifier runner.

## Review Guidance

A separate review skill is usually unnecessary: review is just a VLM call over a source crop plus a mask or overlay, using the prompt files in `prompts/`. Keep it in this skill unless users start asking for review-only workflows independent of foreground segmentation.

Use `prompts/calibration_reviewer.txt` for numeric precision/recall calibration and sortable QC. Use `prompts/subjective_reviewer.txt` for narrative expert judgment. Treat `prompts/objective_reviewer.txt` as a legacy prompt that should not be used for foreground review unless explicitly comparing old runs.

Default QC policy:

- Use OpenRouter `google/gemini-3-flash-preview` with reasoning effort `high`.
- Use `prompts/calibration_reviewer.txt` unless the user explicitly asks for a narrative subjective review.
- `precision_pass = precision > --qc-precision-threshold`.
- `recall_pass = recall > --qc-recall-threshold`.
- `overall_pass = precision_pass and recall_pass`.
- Default thresholds are `0.9` and `0.9`, but always expose them as args so they can be tuned as manual labels accumulate.

Natural-language call shape:

```text
Use the wsi-foreground-segmentation-review skill to review the auto-context run at /data2/vj724/vlm-wsi-auto-context/runs/auto_context_pilot/he_patient_003_slide_003/he_p003_s003_icl0_20260430_175702. If bbox crops are not high-res, export Stage 7 high-res reviewer inputs. Run Gemini 3 Flash calibration review with reasoning effort high, QC thresholds 0.9/0.9, and summarize precision_pass, recall_pass, and overall_pass per bbox.
```

TRIDENT natural-language call shape:

```text
Use the wsi-foreground-segmentation-review skill to review this segmentation: /data2/vj724/path-agent/outputs/trident_output_hest_task1/contours/anon_0c1699ad-e029-4ea6-91ea-8807a0fabb64.jpg. Treat it as a TRIDENT contour output, infer the GeoJSON, resolve the source WSI from the anon slide manifest, export Stage3 reviewer inputs, then run Gemini 3 Flash calibration review with QC thresholds 0.9/0.9.
```

Escalation policy:

- TRIDENT pass: keep contours/coords and record reviewer evidence.
- TRIDENT uncertain/fail: run repo VLM route on the case.
- VLM pass: optionally add teacher labels to distillation candidates.
- VLM fail: inspect Stage 1 bboxes, Stage 4 point overlays, Stage 5 exemplars, and Stage 6 `patches.csv`; rerun with higher-resolution Stage 2 crops or adjusted prompts before distilling.
