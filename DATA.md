# DATA

- `/data2/vj724/multistain/pilot_data/H&E/he_patient_003_slide_003.svs`: easy H&E pilot WSI used for PER-188 foreground auto-context runs.
- `/data2/vj724/multistain/pilot_data/TOL_BLUE/tol_blue_patient_054_slide_004.svs`: challenging Tol Blue pilot WSI used for PER-188 foreground auto-context runs.
- `/data2/vj724/vlm-wsi-auto-context/runs/auto_context_pilot/`: ignored local output root for PER-188 auto-context foreground pilot runs; each run directory has `reproduction.txt`.
- `/data2/vj724/vlm-wsi-auto-context/runs/auto_context_reviewer_inputs/`: ignored local output root for high-resolution Stage 7 auto-context reviewer inputs exported from WSI-level coordinates; each run directory has `reproduction.txt`.
- `/data2/vj724/vlm-wsi-auto-context/runs/trident_reviewer_inputs/`: ignored local output root for TRIDENT contour/GeoJSON reviewer inputs exported as Stage3-compatible crop/mask/overlay directories, preferably per Stage 1 VLM tissue-core bbox; each run directory has `reproduction.txt`.
- `/data2/vj724/vlm-wsi-auto-context/runs/reviewer_pilot/`: ignored local output root for PER-188 foreground reviewer pilot attempts; each batch directory should have `reproduction.txt`.
- `/data2/vj724/hf_cache/`: local Hugging Face cache for Qwen/Qwen3-VL-8B-Instruct-FP8 vLLM serving; see `/data2/vj724/hf_cache/reproduction.txt`.
- `/vol/biomedic3/histopatho/win_share/all_svs_fpaths.csv`: source-of-truth local WSI path manifest for resolving anonymized slide IDs such as `anon_<uuid>.svs` during TRIDENT reviewer exports; copied from the legacy `/data2/vj724/wsi-agents/all_svs_fpaths.csv` path.
- `/vol/biomedic3/vj724/wsi-agents/distilled_student_models_20260225/`: shared staging copy of two MobileNetV3 distilled foreground/background student checkpoints and `results.json` files; see `docs/data/distilled_student_models_20260225.md`.
- `/data2/vj724/wsi-agents/tmp/student_patch_distill_explore/`: mnemosyne-local copy/exploration tree for the same distilled student checkpoint directories, plus local smoke split files; see `docs/data/distilled_student_models_20260225.md`.
