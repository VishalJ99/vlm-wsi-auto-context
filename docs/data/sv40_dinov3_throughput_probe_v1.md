# PER-272 SV40 DINOv3 Throughput Probe

Output root:
`/data2/vj724/vlm-wsi-auto-context/runs/sv40_dinov3_throughput_probe_v1/`

Parent run:
`/data2/vj724/vlm-wsi-auto-context/runs/stress32_n500_sv40_augmented_probe_v1_smoke/`

Purpose:
Benchmark fixed SV40 smoke-run patch coordinates to separate current DINOv3
feature-extraction implementation overhead from machine/storage effects.

Inputs:
- `anon_0f916c21_02b3_4b7f_a16e_260abfc2a664`: 2,202 fixed 512px level-0 patch cells.
- `anon_e60142ee_b5fd_44da_b63e_daa0a506e472`: 1,634 fixed 512px level-0 patch cells.
- WSI paths were resolved from the smoke run's `sv40_candidate_crops.csv`.
- Probe was the PER-269 stress `N=500` DINOv3-small logistic probe.

Measured mean throughput on mnemosyne `/vol` WSI paths:

| Variant | Mean patches/s | Notes |
| --- | ---: | --- |
| `baseline_current` | 35.0 | Current single-producer path; one `CuImage.read_region(..., num_workers=16)` call at a time; PIL plus HF `AutoImageProcessor`. |
| `readpool_pil_hf` | 262.8 | 16 concurrent patch reads, one `CuImage` per worker; same PIL plus HF preprocessing as baseline. Prediction-identical to baseline. |
| `readpool_tensor_preprocess` | 744.9 | 16 concurrent patch reads plus tensor/GPU resize-normalize and direct `AutoModel(pixel_values=...)`; faster but not feature-identical. |
| `gpu_probe_no_feature_cache` | 699.4 | Tensor/GPU preprocessing plus GPU logistic probe scoring; copies only probabilities back to CPU. |

Interpretation:
The same `/vol` SV40 patch coordinates are much faster once patch reads are
truly concurrent. The prediction-identical `readpool_pil_hf` variant is the
safe low-risk optimization candidate. The tensor/GPU preprocessing variants are
faster but changed features slightly, with about 99.1% prediction agreement
against baseline on these two cases, so they should be treated as a separate
quality/compatibility decision.

Primary artifacts:
- `throughput_summary.csv`
- `summary.json`
- `timings/*.json`
- `reproduction.txt`

Run caveat:
cuCIM emitted repeated low-level TIFF directory warnings while reading the
`/vol` SVS files, but the benchmark command exited successfully and wrote all
expected outputs.
