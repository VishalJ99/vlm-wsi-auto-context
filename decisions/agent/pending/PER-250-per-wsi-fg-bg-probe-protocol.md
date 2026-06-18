# PER-250 per-WSI FG/BG probe protocol

Date: 2026-06-05

Status: pending human review

## Context

The user asked for a baseline experiment to estimate how many foreground and
background patches are needed to train a per-WSI DINO small linear probe on
selected detector crops, then test on other crops in the same WSI.

## Decision

For the initial baseline, use the completed PER-250 selector-seeded all500
foreground run as the source of patch grids and pseudo-labels:

- Patch labels come from `stage7/tissue_mask_post.npy`, aligned to
  `stage6/patches.csv` by `row` and `col`.
- The training/evaluation split is leave-one-selected-bbox-crop-out within each
  WSI: train on all selected crops except the held-out crop, and test on every
  patch in the held-out crop.
- Budgets are per class, not total: budget `n` means `n` foreground plus `n`
  background training patches.
- The script tries the requested DINOv3-small model first. If access is gated
  and the caller explicitly allows fallback, it runs the same protocol with
  timm DINOv2-small and records the access failure in the output summary.

## Rationale

Stage 7 masks are the closest available patch-level foreground/background
labels from the current selector-seeded pipeline, and leave-one-selected-crop-out
is a stricter test than a random same-crop split because it measures transfer to
other selected crops within the same WSI.

## Consequences

The first 2026-06-05 run is a reproducible DINOv2-small fallback baseline,
because Hugging Face access to `facebook/dinov3-vits16-pretrain-lvd1689m`
returned a gated-repo 401 before the model was available in the local cache.

After authenticated access populated the local Hugging Face cache, the same
protocol was rerun with true DINOv3-small via Transformers in offline mode. The
true DINOv3-small run used the same selected WSIs, split policy, budgets, and
sample seeds as the fallback run.

The patch-level overlay diagnostic for case `task=23`, bbox
`63942_8696_70237_20598`, showed that the earlier continuous overlay was
misaligned. The old visual blended Stage 7 labels over a Stage 3 detector-derived
crop thumbnail, while the labels are defined on the Stage 6 level-0 patch lattice
from `patches.csv`. The actual level-0 patch contact sheet and corrected
level-0 lattice overlay agree, so patch-level QA should use those visuals rather
than the Stage 3-thumbnail overlay.

Outputs:

- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1/`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1/reproduction.txt`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/reproduction.txt`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/metrics/dinov2_fallback_vs_dinov3small_comparison.csv`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1/visuals/patch_level_actual_patches_case023_bbox_63942_lower_left.png`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1/visuals/patch_overlay_vs_actual_grid_case023_bbox63942_side_by_side.png`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1/visuals/patch_level_overlay_case023_bbox63942_corrected_from_stage6_geometry.png`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/visuals/per_wsi_probe_fold_demo_case023_budget20_seed0/`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/visuals/per_wsi_probe_transfer_to_unselected_case023_allselected_train_v1/`
- `/data2/vj724/vlm-wsi-auto-context/runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small/visuals/per_wsi_probe_transfer_to_unselected_case023_10fg10bg_per_train_crop_v1/`

Follow-up transfer semantics:

The user clarified that "held out crops" means detector-pipeline candidate
bboxes not selected by the Stage 7/redundancy verifier, not leave-one selected
crop out. For case `task=23`, verifier-selected detector IDs `[6, 7, 8, 9, 10]`
trained the per-WSI probe and unselected detector IDs `[1, 2, 3, 4, 5]` were
scored by transfer. These unselected bboxes do not have Stage 7 pseudo-label
truth in this experiment, so the transfer packet reports prediction overlays
and per-candidate predicted FG fractions, not precision/recall.

A second transfer packet trains the same true DINOv3-small linear probe using
only `10` foreground and `10` background Stage 7 pseudo-labeled patches from
each selected training crop, `100` patches total. The sampled train patches are
outlined in the PDF and predictions are rendered on the same unselected detector
IDs `[1, 2, 3, 4, 5]`.

Pooled scale-500 protocol update:

For the all-scale500 pooled baseline, the sampling unit is WSI rather than
crop. Each WSI contributes at most `100` Stage 7 pseudo-labeled patches total,
targeting an even foreground/background split when the WSI has enough patches
for both classes. Each class quota is split across the WSI's verifier-selected
crop bboxes and crop shortfalls are backfilled from other selected crops in the
same WSI/class. This keeps large crops from dominating while still representing
every selected crop when possible.

The pooled run uses true DINOv3-small
`facebook/dinov3-vits16-pretrain-lvd1689m` via Transformers in offline mode,
DINO batch size `64`, and cuCIM WSI reads with `16` read workers for any newly
extracted patches. The current v1 cache is mixed-reader because the interrupted
OpenSlide-backed extraction wrote `375` reusable feature files before the user
asked to restart with cuCIM, and the resumed cuCIM run wrote the remaining
`125` feature files. Use a fresh output directory if the reader backend itself
needs to be controlled as an experimental variable.

The first pooled transfer packet trains `linear_logreg`, `mlp_1x64`, and
`mlp_2x64` on the all500 selected-crop feature cache, validates with a 20%
case-level holdout, final-fits on all selected pseudo-label patches, and applies
the chosen overlay model to unselected detector candidates for case
`anon_02665c40_cc43_42f3_8ab1_fb9a1416e3e6`. As with the per-WSI transfer
packet, unselected candidates have no Stage 7 truth here, so the packet reports
predicted foreground fractions and visual overlays rather than precision/recall.

Additional outputs:

- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/`
- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/reproduction.txt`
- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/visuals/pooled_probe_transfer_case023_v1/`
- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/visuals/pooled_probe_transfer_case023_v1/pooled_scale500_dinov3_probe_transfer.pdf`

The next review packet applies the same pooled probe to a broader 100-WSI
transfer set: `20` eligible cases per stain for EVG, H&E, JONES, PAS, and SV40.
Eligibility requires at least one detector candidate bbox not selected by the
verifier/Stage 7 policy. The output renders, for every transfer WSI, a
thumbnail-level detector overview, unselected-crop thumbnail panels, and
crop-level patch-grid prediction overlays. Feature extraction for these
unselected crops uses cuCIM with `16` read workers and DINOv3 batch size `64`.
The run completed with `100` unselected feature caches, `300` unselected bbox
rows for the chosen `mlp_1x64` overlay model, and a `301`-page PDF.
After review feedback, the thumbnail-level detector overview was regenerated
to draw `mlp_1x64` predicted foreground patch squares directly inside each
unselected detector bbox, so the overview now shows both selected/unselected
bbox status and patch-level transfer classifications.

The transfer candidate source is constrained to final detector orders from
`detections.json`. Stage 5 intermediate candidate metadata can contain boxes
that are not present in the final detector layout; those must not be scored or
rendered as unselected transfer crops. Cached unselected feature records are
filtered to final detector orders before writing CSVs or PDF overlays.

Additional 20-per-stain outputs:

- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/visuals/pooled_probe_transfer_20perstain_v1/`
- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/visuals/pooled_probe_transfer_20perstain_v1/pooled_scale500_dinov3_probe_transfer.pdf`
- `/data2/vj724/vlm-wsi-auto-context/runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1/visuals/pooled_probe_transfer_20perstain_v1/transfer_case_manifest_20perstain.csv`
