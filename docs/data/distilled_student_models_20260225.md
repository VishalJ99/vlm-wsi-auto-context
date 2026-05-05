# Distilled Student Models 20260225

Ticket: PER-188
Verified: 2026-05-05 on mnemosyne

## Locations

- Shared authoritative package: `/vol/biomedic3/vj724/wsi-agents/distilled_student_models_20260225/`
- Mnemosyne-local package mirror: `/data2/vj724/wsi-agents/distilled_student_models_20260225_package/`
- Older mnemosyne-local copy/exploration root: `/data2/vj724/wsi-agents/tmp/student_patch_distill_explore/`

The package roots contain the two checkpoint/result directories plus the
training script, checkpoint inference/export script, helper dependencies, split
manifests, eval wrapper, WSI manifest, dataset location notes, `FILES.txt`,
`SHA256SUMS`, and package-level `reproduction.txt`. The older
`tmp/student_patch_distill_explore/` path remains useful for local exploration
artifacts but is no longer the authoritative package.

## Files And Checksums

| Variant | Relative file | SHA-256 |
| --- | --- | --- |
| zero-shot-teacher student | `run_artifacts/train_zero_shot_t10k_e2_mnet_20260225/mobilenetv3_large_100_best.pt` | `ee83e44fe3f612105fa22f6cd8f29fc5cba5d2fed3037ed14b24d1ab9b7ab7e4` |
| zero-shot-teacher student | `run_artifacts/train_zero_shot_t10k_e2_mnet_20260225/results.json` | `6eb38f97c2c27e10c1860a576997e1b745a0668a7c053b6e31e08084f7993378` |
| harder qwen8b few-shot-teacher student | `run_artifacts/harder_qwen8bfew_t10k_e2_mnet_20260225_split22_3_7/mobilenetv3_large_100_best.pt` | `907cf16a2cf674f95edcdfec279d7cbcce35ec3c6332c4000d7a442dae939502` |
| harder qwen8b few-shot-teacher student | `run_artifacts/harder_qwen8bfew_t10k_e2_mnet_20260225_split22_3_7/results.json` | `0e71129202304384c56772528961ff6dc3779c9bc4aaf96bd2682eef22d609c4` |

The `.pt` and `results.json` hashes above matched the gallifrey source, and
`sha256sum -c SHA256SUMS` passed for the mnemosyne-local package on
2026-05-05.

## Result Summaries

From `results.json`, backbone `mobilenetv3_large_100`:

| Variant | Split cases | Test accuracy | Test precision | Test recall | Test F1 | Test AUROC | Test AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot | train 51, val 9, test 15 | 0.9864 | 0.9451913133402275 | 0.9838536060279871 | 0.9641350210970464 | 0.9991912921319348 | 0.9964350137362052 |
| harder qwen8b few-shot | train 22, val 3, test 7 | 0.9778 | 0.8622908622908623 | 0.9940652818991098 | 0.923501033769814 | 0.9974382005978618 | 0.9816082725669436 |

## Use In This Repo

These files are distilled foreground/background student model artifacts. The
package inference script is runnable against a Stage6-like patch grid, but it
does not by itself implement the native PER-188 route in
`/data2/vj724/vlm-wsi-auto-context`.

Before using them as the fast foreground mode inside `run_auto_context.py`, add
or locate an inference adapter that:

- loads the selected checkpoint and preprocessing from its `results.json` config;
- runs on the same WSI/bbox patch grid used by Stage 6;
- writes Stage6-compatible `class_map.npy`, `quality_map.npy`, `patches.csv`,
  masks/overlays, and metadata;
- lets Stage 7 postprocessing and the reviewer workflow run unchanged.

For the 2026-05-05 TRIDENT reviewer pilot, this repo used
`scripts/build_distilled_stage6_grid_from_reviewer_masks.py` to create a
Stage6-like grid from reviewer `mask.png` files, then ran the package inference
script for both checkpoints. The resulting comparison artifacts are under
`/data2/vj724/vlm-wsi-auto-context/runs/distilled_student_inference/trident_foreground_quality_flash_calibration_20260505/`.
