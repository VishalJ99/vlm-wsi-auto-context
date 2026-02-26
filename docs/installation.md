# Installation

## Conda-First Setup (Recommended)

Use the project setup script. By default it targets the `path-agent` env name.

```bash
bash setup.sh
```

Useful setup knobs:

- `WSI_CONDA_ENV_NAME` (default: `path-agent`)
- `WSI_FORCE_ENV_INSTALL=1` to reinstall/verify deps in an existing env
- `WSI_INSTALL_VLLM=auto|require|skip`

The script installs from the project-level `requirements.txt` (falling back to
`requirements_pipeline.txt`), then installs optional runtime extras (for example `cucim`).

## Manual Conda Setup (Alternative)

```bash
conda env create -f environment.yml
conda activate path-agent
```

or:

```bash
conda create -n path-agent python=3.11 -y
conda activate path-agent
pip install -r requirements.txt
```

If you use OpenSlide-backed reads, install OpenSlide system libraries as required by your OS.

## Backend Credentials

### OpenRouter path
Set API key env vars used by your chosen scripts:

```bash
export OPENROUTER_API_KEY=...
# Optional compatibility fallback used by some paths:
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
```

### Gemini Vertex path
Set service-account credentials in environment:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

Most kept scripts also accept explicit credential flags; if omitted, they fall back to `GOOGLE_APPLICATION_CREDENTIALS`.

## Smoke Checks

```bash
python run_foreground_method.py --help
python run_vlm_bbox_inference.py --help
python run_vlm_reviewer_batch.py --help
```

If reviewer help fails with `ModuleNotFoundError: skimage`, install/repair `scikit-image` in the active environment.
