#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  exit 0
fi

mkdir -p .cache

PYTHON="${PYTHON:-python3}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi

rm -rf "${VENV_DIR}"
"${PYTHON}" -m venv --without-pip "${VENV_DIR}"

GET_PIP_PATH=".cache/get-pip.py"

download_get_pip() {
  local url="https://bootstrap.pypa.io/get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${GET_PIP_PATH}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "${GET_PIP_PATH}" "${url}"
    return 0
  fi
  "${VENV_DIR}/bin/python" - <<'PY'
import urllib.request
url = "https://bootstrap.pypa.io/get-pip.py"
out = ".cache/get-pip.py"
with urllib.request.urlopen(url) as r, open(out, "wb") as f:
    f.write(r.read())
PY
}

download_get_pip

"${VENV_DIR}/bin/python" "${GET_PIP_PATH}" --disable-pip-version-check

