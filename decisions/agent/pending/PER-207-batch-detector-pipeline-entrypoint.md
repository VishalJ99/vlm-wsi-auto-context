# PER-207 Batch Detector Pipeline Entrypoint

## Status

Pending human review.

## Context

PER-207 needs an operational detector-oracle entrypoint that can run the updated
bbox pipeline on arbitrary WSI inputs instead of only the pilot manifest. The
recent accepted direction keeps the detailed Stage 1 prompt and adds
high-resolution crop redetection, classification, and comparative thumbnail-crop
artifact filtering.

## Decision

Add `scripts/run_detector_pipeline.py` as the arbitrary-WSI batch entrypoint.

The script accepts exactly one of:

- a single WSI path,
- a directory of WSIs,
- a text file containing WSI paths.

Default batching is breadth-first: complete each stage across all WSIs before
advancing to the next stage. This keeps OpenRouter/VLM concurrency saturated
across both WSIs and crops. A `--batch-mode depth-first` option processes one WSI
through the full pipeline at a time while still parallelizing crop-level calls
within that WSI.

All detector-control parameters that affect bbox generation or filtering are
CLI arguments, including Stage 1 prompt/rotations/padding/merge threshold,
post-source bbox padding, IoU merge threshold, overlap-over-smaller merge
threshold, crop max dimensions, classification padding, reasoning effort,
temperature, max tokens, and concurrency.

## Important Boundary

For arbitrary WSI inputs, the first version starts the postprocessing leg from
the raw Stage 1 source rotation boxes. The prior 10-case smoke script
`scripts/stage7_post_stage3_crop_redetect_pipeline.py` still handles the saved
pilot Stage 3 artifacts. If the arbitrary-WSI entrypoint should literally run
the Stage 2/3 reviewer-feedback redetection loop inline, add that as an explicit
source mode rather than hiding it inside the postprocessing step.

## Consequences

- Single-case output is review-friendly: `final_detected_bboxes.png`,
  `detections.json`, and `reproduction.txt` live directly in the output
  directory.
- Multi-case output creates one subdirectory per WSI filename stem.
- `--save-all-stage-artifacts` preserves intermediate crops, overlays, prompt
  copies, and per-stage JSONL summaries; otherwise successful generated
  intermediates are removed after final output.
- Every output root gets a `reproduction.txt` that records the stage prompt,
  image input type, output contract, and key hyperparameters.
