# PER-249 DINOv2 Small Linear Probe Fallback

Status: pending human review

## Context

PER-249 asked for a lightweight tissue/artifact classifier from Stage 5 and
Stage 6 detector-pipeline outputs, ideally using DINOv3-small as a frozen
foundation-model backbone and an MLP or linear head on top.

The runnable `path-agent` environment has PyTorch, timm, Transformers, sklearn,
and pretrained `vit_small_patch14_dinov2`. The separate `dinov3` environment was
slow to use from shared storage and did not expose an importable DINOv3 package
or timm model entry during this run.

## Decision

Use pretrained `vit_small_patch14_dinov2` as the frozen ViT-small backbone for
the first PER-249 readout, with `StandardScaler + LogisticRegression(liblinear)`
as the lightweight head.

Run the same training and scoring protocol for both image sources:

- `highres_crop`: Stage 5 WSI reread crop saved as `crop.png`.
- `thumbnail_bbox`: Stage 1 thumbnail cropped to the candidate bbox with 10%
  padding and the original bbox drawn in red.

## Evidence

The high-resolution crop source produced better 5-fold cross-validation metrics
on the balanced 51 rejected / 51 accepted training set:

- High-resolution crop: accuracy `0.912`, ROC AUC `0.968`, AP `0.968`,
  confusion `TN=46 FP=5 FN=4 TP=47`.
- Thumbnail bbox crop: accuracy `0.853`, ROC AUC `0.959`, AP `0.958`,
  confusion `TN=42 FP=9 FN=6 TP=45`.

Applied to the remaining 2,194 accepted crops at threshold `0.5`, high-resolution
crops proposed 260 rejections, while thumbnail crops proposed 339. The two lists
overlapped on 191 crops; 69 were high-resolution-only and 148 were
thumbnail-only.

## Consequences

- Treat the high-resolution crop model as the cleaner first-pass readout.
- Treat the thumbnail run as a useful comparison and a likely noisier
  candidate-mining view, especially because it proposed more H&E and PAS
  rejections.
- Re-run with actual DINOv3-small only after that backbone is available in a
  reproducible environment.
