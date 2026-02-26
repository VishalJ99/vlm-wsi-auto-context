#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARGS_FILE="${1:-${REPO_ROOT}/configs/paper_method.args}"

if [[ ! -f "${ARGS_FILE}" ]]; then
  echo "Error: args file not found: ${ARGS_FILE}" >&2
  exit 1
fi

mapfile -t ARGS < <(grep -Ev '^[[:space:]]*(#|$)' "${ARGS_FILE}")
if [[ "${#ARGS[@]}" -eq 0 ]]; then
  echo "Error: no arguments loaded from ${ARGS_FILE}" >&2
  exit 1
fi

exec python "${REPO_ROOT}/run_auto_context.py" "${ARGS[@]}"
