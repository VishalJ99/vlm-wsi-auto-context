#!/usr/bin/env bash
# WSI Foreground Pipeline - Environment Setup
# Usage: bash setup.sh
set -euo pipefail

ENV_NAME="${WSI_CONDA_ENV_NAME:-path-agent}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="${WSI_REQUIREMENTS_FILE:-$SCRIPT_DIR/requirements.txt}"
if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="$SCRIPT_DIR/requirements_pipeline.txt"
fi
CUCIM_PIP_VERSION="${WSI_CUCIM_PIP_VERSION:-}"
INSTALL_VLLM_MODE="${WSI_INSTALL_VLLM:-require}"
AUTO_SET_CUDA_HOME="${WSI_AUTO_SET_CUDA_HOME:-1}"
AUTO_INSTALL_NVCC="${WSI_AUTO_INSTALL_NVCC:-1}"
FORCE_ENV_INSTALL="${WSI_FORCE_ENV_INSTALL:-0}"
PERSIST_HF_TOKEN_TO_RC="${WSI_PERSIST_HF_TOKEN_TO_RC:-ask}"
ENV_BLOCK_START="# >>> WSI-AGENT ENV >>>"
ENV_BLOCK_END="# <<< WSI-AGENT ENV <<<"

is_truthy() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

normalize_path() {
    local p="$1"
    if [[ "$p" == "~"* ]]; then
        p="${HOME}${p:1}"
    fi
    if [[ "$p" != /* ]]; then
        p="$SCRIPT_DIR/$p"
    fi
    printf '%s\n' "$p"
}

sanitize_path_candidate() {
    local label="$1"
    local raw="${2:-}"
    local cleaned="$raw"
    local last_line=""

    [ -n "$cleaned" ] || {
        printf '\n'
        return 0
    }

    cleaned="${cleaned//$'\r'/}"

    if [[ "$cleaned" == *$'\n'* ]]; then
        while IFS= read -r line; do
            [ -n "$line" ] && last_line="$line"
        done <<< "$cleaned"
        cleaned="$last_line"
        echo "WARNING: $label had a multi-line value in env; using last line as path candidate." >&2
        echo "  $cleaned" >&2
    fi

    cleaned="$(printf '%s' "$cleaned" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    printf '%s\n' "$cleaned"
}

is_under_home_dir() {
    local p="$1"
    local h="${HOME%/}"
    p="${p%/}"
    [[ "$p" == "$h" || "$p" == "$h/"* ]]
}

confirm_home_cache_usage() {
    local label="$1"
    local dir="$2"
    if ! is_under_home_dir "$dir"; then
        return 0
    fi

    echo "WARNING: $label is under HOME:" >&2
    echo "  $dir" >&2
    echo "On HPC this may hit home quota or be slower than workspace/scratch." >&2

    if [ -t 0 ]; then
        local choice=""
        read -r -p "Use this HOME location anyway? [y/N]: " choice
        case "${choice,,}" in
            y|yes) return 0 ;;
            *) return 1 ;;
        esac
    fi

    return 0
}

echo "============================================"
echo "WSI Foreground Pipeline - Environment Setup"
echo "============================================"
echo ""

# Parse "conda config --show pkgs_dirs" into one path per line.
get_configured_pkgs_dirs() {
    conda config --show pkgs_dirs 2>/dev/null | awk '
        /^pkgs_dirs:/ {in_pkgs=1; next}
        in_pkgs && /^[^[:space:]-]/ {exit}
        in_pkgs && /^[[:space:]]*-[[:space:]]*/ {
            sub(/^[[:space:]]*-[[:space:]]*/, "", $0)
            print
        }
    '
}

print_pkgs_dirs_status() {
    local had_any=0
    echo "Current conda pkgs_dirs:"
    while IFS= read -r dir; do
        [ -z "$dir" ] && continue
        had_any=1
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            echo "  - $dir [exists, writable]"
        elif [ -d "$dir" ]; then
            echo "  - $dir [exists, not writable]"
        else
            echo "  - $dir [missing]"
        fi
    done < <(get_configured_pkgs_dirs)
    if [ "$had_any" -eq 0 ]; then
        echo "  (none configured)"
    fi
    echo ""
}

has_writable_configured_pkgs_dir() {
    while IFS= read -r dir; do
        [ -z "$dir" ] && continue
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            return 0
        fi
    done < <(get_configured_pkgs_dirs)
    return 1
}

has_writable_pkgs_in_list() {
    local list="$1"
    local dir=""
    local sep=""

    if [[ "$list" == *";"* ]]; then
        sep=";"
    else
        sep=":"
    fi

    IFS="$sep" read -r -a _pkgs_candidates <<< "$list"
    for dir in "${_pkgs_candidates[@]}"; do
        [ -z "$dir" ] && continue
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            return 0
        fi
    done
    return 1
}

first_writable_pkgs_in_list() {
    local list="$1"
    local dir=""
    local sep=""

    if [[ "$list" == *";"* ]]; then
        sep=";"
    else
        sep=":"
    fi

    IFS="$sep" read -r -a _pkgs_candidates <<< "$list"
    for dir in "${_pkgs_candidates[@]}"; do
        [ -z "$dir" ] && continue
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    return 1
}

has_writable_effective_pkgs_dir() {
    # If CONDA_PKGS_DIRS is set, conda will prioritize it over configured pkgs_dirs.
    if [ -n "${CONDA_PKGS_DIRS:-}" ]; then
        has_writable_pkgs_in_list "$CONDA_PKGS_DIRS"
        return $?
    fi

    has_writable_configured_pkgs_dir
}

first_writable_configured_pkgs_dir() {
    while IFS= read -r dir; do
        [ -z "$dir" ] && continue
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done < <(get_configured_pkgs_dirs)
    return 1
}

first_writable_effective_pkgs_dir() {
    if [ -n "${CONDA_PKGS_DIRS:-}" ]; then
        first_writable_pkgs_in_list "$CONDA_PKGS_DIRS"
        return $?
    fi
    first_writable_configured_pkgs_dir
}

configure_writable_pkgs_dir() {
    local default_pkgs_dir="$SCRIPT_DIR/.conda/pkgs"
    local chosen_dir="${WSI_CONDA_PKGS_DIR:-}"

    print_pkgs_dirs_status

    if [ -n "${CONDA_PKGS_DIRS:-}" ]; then
        if has_writable_pkgs_in_list "$CONDA_PKGS_DIRS"; then
            echo "Using CONDA_PKGS_DIRS from environment:"
            echo "  CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
            echo ""
            return
        fi

        echo "CONDA_PKGS_DIRS is set but has no writable directories:"
        echo "  CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
        echo "Ignoring it for this setup run."
        echo ""
    fi

    if [ -z "$chosen_dir" ]; then
        if [ -t 0 ]; then
            echo "No writable conda package cache directory was found."
            echo "Choose a cache path on your workspace/scratch (not HOME on quota-limited HPC)."
            read -r -p "Conda package cache dir [$default_pkgs_dir]: " chosen_dir
            chosen_dir="${chosen_dir:-$default_pkgs_dir}"
        else
            chosen_dir="$default_pkgs_dir"
            echo "No writable pkgs_dirs found (non-interactive mode)."
            echo "Defaulting to: $chosen_dir"
        fi
    fi

    while true; do
        chosen_dir="$(normalize_path "$chosen_dir")"
        mkdir -p "$chosen_dir"
        if [ ! -w "$chosen_dir" ]; then
            echo "ERROR: '$chosen_dir' is not writable."
            if [ -t 0 ]; then
                read -r -p "Choose another conda package cache dir: " chosen_dir
                chosen_dir="${chosen_dir:-$default_pkgs_dir}"
                continue
            fi
            echo "Set a writable path via WSI_CONDA_PKGS_DIR or CONDA_PKGS_DIRS and re-run setup.sh."
            exit 1
        fi

        if confirm_home_cache_usage "Conda package cache dir" "$chosen_dir"; then
            break
        fi
        if [ -t 0 ]; then
            read -r -p "Choose another conda package cache dir [$default_pkgs_dir]: " chosen_dir
            chosen_dir="${chosen_dir:-$default_pkgs_dir}"
        else
            break
        fi
    done

    export CONDA_PKGS_DIRS="$chosen_dir"
    echo "Using conda package cache for this setup:"
    echo "  CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
    echo ""

    if [ -t 0 ]; then
        read -r -p "Persist this cache path to ~/.condarc for future conda commands? [y/N]: " persist_choice
        case "${persist_choice,,}" in
            y|yes)
                conda config --prepend pkgs_dirs "$chosen_dir"
                echo "Updated ~/.condarc with pkgs_dirs entry: $chosen_dir"
                echo ""
                ;;
        esac
    fi
}

choose_writable_cache_dir() {
    local label="$1"
    local default_dir="$2"
    local candidate="${3:-}"

    while true; do
        if [ -z "$candidate" ]; then
            if [ -t 0 ]; then
                read -r -p "$label [$default_dir]: " candidate
                candidate="${candidate:-$default_dir}"
            else
                candidate="$default_dir"
                echo "No value set for $label (non-interactive mode)." >&2
                echo "Defaulting to: $candidate" >&2
            fi
        fi

        candidate="$(normalize_path "$candidate")"
        mkdir -p "$candidate" 2>/dev/null || true
        if [ ! -d "$candidate" ] || [ ! -w "$candidate" ]; then
            echo "WARNING: $label is not writable:" >&2
            echo "  $candidate" >&2
            if [ -t 0 ]; then
                echo "Provide another path, or press Enter to use default:" >&2
                echo "  $default_dir" >&2
                candidate=""
                continue
            fi
            echo "ERROR: Non-interactive mode requires a writable $label." >&2
            return 1
        fi

        if ! confirm_home_cache_usage "$label" "$candidate"; then
            candidate=""
            continue
        fi

        printf '%s\n' "$candidate"
        return 0
    done
}

configure_hf_cache_dirs() {
    local default_hf_home="$SCRIPT_DIR/.cache/huggingface"
    local default_hf_hub_cache=""
    local default_transformers_cache=""
    local requested_hf_home="${HF_HOME:-${WSI_HF_HOME:-}}"
    local requested_hf_hub_cache="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-${WSI_HUGGINGFACE_HUB_CACHE:-}}}"
    local requested_transformers_cache="${TRANSFORMERS_CACHE:-${WSI_TRANSFORMERS_CACHE:-}}"
    local chosen=""

    requested_hf_home="$(sanitize_path_candidate "HF_HOME" "$requested_hf_home")"
    requested_hf_hub_cache="$(sanitize_path_candidate "HUGGINGFACE_HUB_CACHE/HF_HUB_CACHE" "$requested_hf_hub_cache")"
    requested_transformers_cache="$(sanitize_path_candidate "TRANSFORMERS_CACHE" "$requested_transformers_cache")"

    echo "Checking Hugging Face cache paths..."
    echo "Default cache locations (press Enter to accept):"
    echo "  HF_HOME: $default_hf_home"
    echo "  HUGGINGFACE_HUB_CACHE: <HF_HOME>/hub"
    echo "  TRANSFORMERS_CACHE: <HF_HOME>/transformers"
    echo ""

    if [ -n "$requested_hf_home" ] || [ -n "$requested_hf_hub_cache" ] || [ -n "$requested_transformers_cache" ]; then
        echo "Detected existing HF cache env values:"
        [ -n "$requested_hf_home" ] && echo "  HF_HOME=$requested_hf_home"
        [ -n "$requested_hf_hub_cache" ] && echo "  HUGGINGFACE_HUB_CACHE/HF_HUB_CACHE=$requested_hf_hub_cache"
        [ -n "$requested_transformers_cache" ] && echo "  TRANSFORMERS_CACHE=$requested_transformers_cache"
        echo ""
    fi

    chosen="$(choose_writable_cache_dir "HF_HOME directory" "$default_hf_home" "$requested_hf_home")" || {
        echo "ERROR: Could not configure a writable HF_HOME directory."
        exit 1
    }
    export HF_HOME="$chosen"

    default_hf_hub_cache="${HF_HOME}/hub"
    chosen="$(choose_writable_cache_dir "HUGGINGFACE_HUB_CACHE directory" "$default_hf_hub_cache" "$requested_hf_hub_cache")" || {
        echo "ERROR: Could not configure a writable HUGGINGFACE_HUB_CACHE directory."
        exit 1
    }
    export HUGGINGFACE_HUB_CACHE="$chosen"
    export HF_HUB_CACHE="$chosen"

    default_transformers_cache="${HF_HOME}/transformers"
    chosen="$(choose_writable_cache_dir "TRANSFORMERS_CACHE directory" "$default_transformers_cache" "$requested_transformers_cache")" || {
        echo "ERROR: Could not configure a writable TRANSFORMERS_CACHE directory."
        exit 1
    }
    export TRANSFORMERS_CACHE="$chosen"

    echo "Using Hugging Face cache paths:"
    echo "  HF_HOME=$HF_HOME"
    echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
    echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
    echo ""
}

configure_hf_token() {
    local token="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

    if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ] && [ "$HUGGING_FACE_HUB_TOKEN" != "$HF_TOKEN" ]; then
        echo "WARNING: HUGGING_FACE_HUB_TOKEN and HF_TOKEN differ."
        echo "Using HUGGING_FACE_HUB_TOKEN for this setup session."
        token="$HUGGING_FACE_HUB_TOKEN"
    fi

    if [ -z "$token" ]; then
        echo "No Hugging Face token found in environment."
        if [ -t 0 ]; then
            read -r -s -p "Enter Hugging Face token (hf_...) [leave blank to skip]: " token
            echo ""
        else
            echo "Non-interactive mode: skipping token prompt."
        fi
    fi

    if [ -n "$token" ]; then
        export HUGGING_FACE_HUB_TOKEN="$token"
        export HF_TOKEN="$token"
        echo "Hugging Face token configured for this setup session."
    else
        echo "Hugging Face token not set. Gated model downloads may fail."
    fi
    echo ""
}

shell_single_quote() {
    printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\"'\"'/g")"
}

upsert_env_block_in_rc() {
    local rc_file="$1"
    local include_hf_token="$2"
    local tmp_file=""

    [ -f "$rc_file" ] || touch "$rc_file"

    tmp_file="$(mktemp)"
    awk -v start="$ENV_BLOCK_START" -v end="$ENV_BLOCK_END" '
        $0 == start {skip = 1; next}
        skip && $0 == end {skip = 0; next}
        !skip {print}
    ' "$rc_file" > "$tmp_file"

    {
        echo ""
        echo "$ENV_BLOCK_START"
        echo "# Managed by wsi-agents/setup.sh"
        if [ -n "${CONDA_PKGS_DIRS:-}" ]; then
            printf "export CONDA_PKGS_DIRS=%s\n" "$(shell_single_quote "$CONDA_PKGS_DIRS")"
        fi
        printf "export HF_HOME=%s\n" "$(shell_single_quote "$HF_HOME")"
        printf "export HUGGINGFACE_HUB_CACHE=%s\n" "$(shell_single_quote "$HUGGINGFACE_HUB_CACHE")"
        printf "export HF_HUB_CACHE=%s\n" "$(shell_single_quote "$HF_HUB_CACHE")"
        printf "export TRANSFORMERS_CACHE=%s\n" "$(shell_single_quote "$TRANSFORMERS_CACHE")"
        if [ "$include_hf_token" = "1" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
            printf "export HUGGING_FACE_HUB_TOKEN=%s\n" "$(shell_single_quote "$HUGGING_FACE_HUB_TOKEN")"
            printf "export HF_TOKEN=%s\n" "$(shell_single_quote "$HUGGING_FACE_HUB_TOKEN")"
        fi
        echo "$ENV_BLOCK_END"
    } >> "$tmp_file"

    mv "$tmp_file" "$rc_file"
}

should_persist_hf_token_to_rc() {
    local pref="${PERSIST_HF_TOKEN_TO_RC,,}"

    if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
        return 1
    fi

    case "$pref" in
        1|true|yes|y|on) return 0 ;;
        0|false|no|n|off) return 1 ;;
        ask|"")
            if [ -t 0 ]; then
                local choice=""
                read -r -p "Persist Hugging Face token to shell startup files? [y/N]: " choice
                case "${choice,,}" in
                    y|yes) return 0 ;;
                esac
            fi
            return 1
            ;;
        *)
            echo "WARNING: Unrecognized WSI_PERSIST_HF_TOKEN_TO_RC='$PERSIST_HF_TOKEN_TO_RC'."
            echo "Defaulting to: do not persist HF token."
            return 1
            ;;
    esac
}

persist_env_to_shell_rc_files() {
    local include_hf_token="$1"
    local bash_rc="$HOME/.bashrc"
    local zsh_rc="$HOME/.zshrc"

    upsert_env_block_in_rc "$bash_rc" "$include_hf_token"
    echo "Updated shell env block in: $bash_rc"

    upsert_env_block_in_rc "$zsh_rc" "$include_hf_token"
    echo "Updated shell env block in: $zsh_rc"
    echo ""
}

reload_shell_rc_in_setup_process() {
    if [ -f "$HOME/.bashrc" ]; then
        set +e +u
        # shellcheck disable=SC1090
        source "$HOME/.bashrc" >/dev/null 2>&1 || true
        set -e -u
        echo "Reloaded ~/.bashrc inside setup process."
    fi
    if [ -f "$HOME/.zshrc" ]; then
        echo "Updated ~/.zshrc (not sourced from bash)."
    fi
    echo "To apply in your current interactive shell, run:"
    echo "  source ~/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        echo "or (for zsh):"
        echo "  source ~/.zshrc"
    fi
    echo ""
}

install_cucim_pip() {
    local pkg="cucim"
    if [ -n "$CUCIM_PIP_VERSION" ]; then
        pkg="cucim==$CUCIM_PIP_VERSION"
    fi

    echo "Installing cuCIM via pip: $pkg"
    "$PIP" install "$pkg"
}

install_cucim() {
    echo "Installing cuCIM via pip (conda cuCIM install disabled in setup)..."
    install_cucim_pip
}

ensure_nvcc_available() {
    local auto_lower="${AUTO_INSTALL_NVCC,,}"
    local nvcc_path=""

    # Prefer env-local nvcc even when the env is not shell-activated.
    if [ -n "${ENV_PATH:-}" ] && [ -x "${ENV_PATH}/bin/nvcc" ]; then
        case ":$PATH:" in
            *":${ENV_PATH}/bin:"*) ;;
            *) export PATH="${ENV_PATH}/bin:${PATH}" ;;
        esac
        echo "Found nvcc in conda env:"
        echo "  ${ENV_PATH}/bin/nvcc"
        return 0
    fi

    nvcc_path="$(command -v nvcc || true)"
    if [ -n "$nvcc_path" ]; then
        echo "Found nvcc on PATH:"
        echo "  $nvcc_path"
        return 0
    fi

    if [ "$auto_lower" = "0" ] || [ "$auto_lower" = "false" ] || [ "$auto_lower" = "no" ]; then
        echo "nvcc not found and auto-install is disabled (WSI_AUTO_INSTALL_NVCC=$AUTO_INSTALL_NVCC)."
        return 1
    fi

    echo "nvcc not found. Installing CUDA compiler via conda (nvidia channel)..."
    conda install -n "$ENV_NAME" -c nvidia --freeze-installed cuda-nvcc cuda-cudart-dev -y

    if [ -n "${ENV_PATH:-}" ] && [ -x "${ENV_PATH}/bin/nvcc" ]; then
        case ":$PATH:" in
            *":${ENV_PATH}/bin:"*) ;;
            *) export PATH="${ENV_PATH}/bin:${PATH}" ;;
        esac
        echo "Installed nvcc in conda env:"
        echo "  ${ENV_PATH}/bin/nvcc"
        return 0
    fi

    nvcc_path="$(command -v nvcc || true)"
    if [ -n "$nvcc_path" ]; then
        echo "Found nvcc after install:"
        echo "  $nvcc_path"
        return 0
    fi

    echo "WARNING: nvcc is still unavailable after install attempt."
    return 1
}

configure_cuda_home_from_nvcc() {
    local auto_lower="${AUTO_SET_CUDA_HOME,,}"
    local nvcc_path=""
    local nvcc_real=""
    local cuda_home=""

    if [ "$auto_lower" = "0" ] || [ "$auto_lower" = "false" ] || [ "$auto_lower" = "no" ]; then
        echo "Skipping CUDA_HOME auto-configuration (WSI_AUTO_SET_CUDA_HOME=$AUTO_SET_CUDA_HOME)."
        return 1
    fi

    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "Using existing CUDA_HOME:"
        echo "  CUDA_HOME=$CUDA_HOME"
        export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
        return 0
    fi

    nvcc_path="$(command -v nvcc || true)"
    if [ -z "$nvcc_path" ]; then
        echo "nvcc not found on PATH; leaving CUDA_HOME unset."
        return 1
    fi

    nvcc_real="$(readlink -f "$nvcc_path" 2>/dev/null || echo "$nvcc_path")"
    cuda_home="$(cd "$(dirname "$nvcc_real")/.." && pwd)"
    if [ ! -x "${cuda_home}/bin/nvcc" ]; then
        echo "Found nvcc at '$nvcc_real' but inferred CUDA_HOME '$cuda_home' is invalid."
        return 1
    fi

    export CUDA_HOME="$cuda_home"
    case ":$PATH:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

    echo "Configured CUDA toolkit from nvcc:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  CMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER"
    return 0
}

install_vllm() {
    local mode="${INSTALL_VLLM_MODE,,}"

    case "$mode" in
        auto)
            ensure_nvcc_available || true
            configure_cuda_home_from_nvcc || true
            echo "Installing vLLM for local Qwen3-VL inference (auto mode; non-fatal if unavailable)..."
            set +e
            "$PIP" install vllm
            local rc=$?
            set -e
            if [ "$rc" -ne 0 ]; then
                echo "WARNING: vLLM installation failed (exit code $rc). Continuing setup without vLLM."
                echo "You can still run OpenRouter backend. Install vLLM later on a CUDA-capable node."
            fi
            ;;
        require)
            ensure_nvcc_available || true
            configure_cuda_home_from_nvcc || true
            echo "Installing vLLM for local Qwen3-VL inference (required mode)..."
            "$PIP" install vllm
            ;;
        skip)
            echo "Skipping vLLM install (WSI_INSTALL_VLLM=skip)."
            ;;
        *)
            echo "ERROR: Unsupported WSI_INSTALL_VLLM='$INSTALL_VLLM_MODE'."
            echo "Use one of: auto, require, skip."
            exit 1
            ;;
    esac
}

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check requirements file exists
if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements file not found."
    echo "Looked for:"
    echo "  $SCRIPT_DIR/requirements.txt"
    echo "  $SCRIPT_DIR/requirements_pipeline.txt"
    exit 1
fi

# Ensure conda has a writable package cache location (common HPC issue).
if ! has_writable_effective_pkgs_dir; then
    configure_writable_pkgs_dir
else
    effective_pkgs_dir="$(first_writable_effective_pkgs_dir || true)"
    print_pkgs_dirs_status
    if [ -n "$effective_pkgs_dir" ]; then
        if ! confirm_home_cache_usage "Conda package cache dir" "$effective_pkgs_dir"; then
            echo "Please choose a non-HOME conda package cache directory."
            configure_writable_pkgs_dir
            effective_pkgs_dir="${CONDA_PKGS_DIRS:-$effective_pkgs_dir}"
        fi
    fi
    if [ -n "${CONDA_PKGS_DIRS:-}" ]; then
        echo "Using CONDA_PKGS_DIRS from environment:"
        echo "  CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
        echo ""
    elif [ -n "$effective_pkgs_dir" ]; then
        echo "Using configured conda package cache:"
        echo "  $effective_pkgs_dir"
        echo ""
    fi
fi

# Ensure Hugging Face caches point to writable locations.
configure_hf_cache_dirs

# Ensure Hugging Face token is available when needed.
configure_hf_token

PERSIST_TOKEN_TO_RC=0
if should_persist_hf_token_to_rc; then
    PERSIST_TOKEN_TO_RC=1
fi
persist_env_to_shell_rc_files "$PERSIST_TOKEN_TO_RC"
reload_shell_rc_in_setup_process

# Create conda env (skip if exists)
SKIP_ENV_INSTALL=0
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists."
    if is_truthy "$FORCE_ENV_INSTALL"; then
        echo "WSI_FORCE_ENV_INSTALL=$FORCE_ENV_INSTALL -> re-running dependency install/verification."
    else
        SKIP_ENV_INSTALL=1
        echo "Assuming existing env is already set up; skipping dependency install/verification."
        echo "Set WSI_FORCE_ENV_INSTALL=1 to force reinstall."
    fi
else
    echo "Creating conda env '$ENV_NAME' with Python 3.11..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# Get env path
ENV_PATH="$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')"
PIP="$ENV_PATH/bin/pip"
PYTHON="$ENV_PATH/bin/python"

if [ "$SKIP_ENV_INSTALL" -eq 0 ]; then
    echo ""
    echo "Installing pip dependencies..."
    "$PIP" install -r "$REQ_FILE"

    echo ""
    install_cucim

    echo ""
    echo "Re-installing aiohttp (conda may have clobbered it)..."
    "$PIP" install aiohttp

    echo ""
    install_vllm

    echo ""
    echo "Verifying installation..."
    "$PYTHON" -c "
import numpy, PIL, openai, sklearn, matplotlib, aiohttp, requests, scipy
try:
    import cucim
    cucim_ok = True
except ImportError:
    cucim_ok = False
try:
    import vllm
    vllm_ok = vllm.__version__
except ImportError:
    vllm_ok = None
print(f'  numpy:       {numpy.__version__}')
print(f'  Pillow:      {PIL.__version__}')
print(f'  openai:      {openai.__version__}')
print(f'  scikit-learn:{sklearn.__version__}')
print(f'  matplotlib:  {matplotlib.__version__}')
print(f'  aiohttp:     {aiohttp.__version__}')
print(f'  requests:    {requests.__version__}')
print(f'  scipy:       {scipy.__version__}')
print(f'  cuCIM:       {\"OK\" if cucim_ok else \"MISSING (GPU WSI reading will not work)\"}')
print(f'  vLLM:        {vllm_ok if vllm_ok else \"MISSING (local inference will not work)\"}')
"
else
    echo ""
    echo "Skipping dependency install/verification for existing env '$ENV_NAME'."
fi

echo ""
echo "============================================"
echo "Environment '$ENV_NAME' is ready."
echo "Python: $PYTHON"
echo "============================================"
echo ""
echo "NEXT: Set your OpenRouter API key."
echo "Both variables should point to the same key:"
echo ""
echo "  export OPENAI_API_KEY=<your_openrouter_api_key>"
echo "  export OPENROUTER_API_KEY=<your_openrouter_api_key>"
echo ""
echo "Or add to your shell profile:"
echo "  echo 'export OPENAI_API_KEY=<your_openrouter_api_key>' >> ~/.bashrc"
echo "  echo 'export OPENROUTER_API_KEY=<your_openrouter_api_key>' >> ~/.bashrc"
echo ""
echo "Optional setup controls:"
echo "  WSI_CUCIM_PIP_VERSION=<version>              (optional pin for pip install)"
echo "  WSI_INSTALL_VLLM=auto|require|skip           (default: require)"
echo "  WSI_AUTO_SET_CUDA_HOME=1|0                   (default: 1)"
echo "  WSI_AUTO_INSTALL_NVCC=1|0                    (default: 1)"
