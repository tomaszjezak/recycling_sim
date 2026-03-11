#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${ROOT}/.venv-isaacsim/bin/python" "${ROOT}/scripts/run_conveyor.py" "$@"
