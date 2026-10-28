#!/usr/bin/env bash
# Integrity Theory demo launcher for macOS and Linux.
#
# First run: creates .venv, installs CPU-only PyTorch and the other packages, then starts the
# server on http://127.0.0.1:8000 and opens your browser. Later runs start straight away.
# If no Python 3.10-3.12 is installed, it downloads a private Python 3.12 into this folder.
#
# Options (environment variables):
#   PORT=8001 ./run.sh                 use another port
#   NO_BROWSER=1 ./run.sh              don't open a browser
#   TORCH_INDEX_URL=... ./run.sh       another PyTorch package index

set -u

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"
TOOLS="$ROOT/.tools"
STAMP="$VENV/.integrity-requirements.txt"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
UV_VERSION="0.13.6"
IMPORTS="import torch, torchvision, fastapi, uvicorn, python_multipart, PIL, numpy"
# Quick check that the packages are present, without the slow full import of torch.
PRESENT="import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ('torch', 'torchvision', 'fastapi', 'uvicorn', 'python_multipart', 'PIL', 'numpy')) else 1)"
export PIP_DISABLE_PIP_VERSION_CHECK=1

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n\n' "$*" >&2; exit 1; }

# Is "$1" a 64-bit Python 3.10 - 3.12 interpreter? (PyTorch has no 32-bit builds.)
supported() {
  "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32 else 1)' </dev/null >/dev/null 2>&1
}

find_python() {
  for c in python3.12 python3.11 python3.10 python3 python; do
    p="$(command -v "$c" 2>/dev/null)" || continue
    # macOS: /usr/bin/python3 is Apple's 3.9 stub and may pop up an installer; skip it.
    if [ "$(uname -s)" = "Darwin" ] && [ "$p" = "/usr/bin/python3" ]; then continue; fi
    if supported "$p"; then echo "$p"; return 0; fi
  done
  return 1
}

download() {  # download URL FILE
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
    return
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
    return
  fi
  # No curl or wget: any Python 3 (even an unsupported version) can still download.
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
      "$c" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
      return
    fi
  done
  return 1
}

# uv is a small standalone tool that can download Python. Used only when no usable
# Python 3.10 - 3.12 is installed. It is kept in .tools/ inside this folder.
get_uv() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)              target="x86_64-unknown-linux-musl" ;;
    Linux-aarch64|Linux-arm64) target="aarch64-unknown-linux-musl" ;;
    Darwin-x86_64)             target="x86_64-apple-darwin" ;;
    Darwin-arm64)              target="aarch64-apple-darwin" ;;
    *) fail "No Python 3.10-3.12 found, and this system ($(uname -s) $(uname -m)) can't use the automatic Python download. Install Python 3.12 from https://www.python.org/downloads/ and run this script again." ;;
  esac
  UV="$TOOLS/uv-$target/uv"
  if [ ! -x "$UV" ]; then
    say "Downloading uv $UV_VERSION (a small tool that fetches Python; it stays inside this folder)"
    mkdir -p "$TOOLS"
    download "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" "$TOOLS/uv.tar.gz" \
      || fail "Couldn't download uv. Check your internet connection, or install Python 3.12 from https://www.python.org/downloads/ and run this script again."
    tar -xzf "$TOOLS/uv.tar.gz" -C "$TOOLS" || fail "Couldn't unpack uv."
    rm -f "$TOOLS/uv.tar.gz"
  fi
}

# ---- 1. Python environment -------------------------------------------------------------
if [ -x "$VPY" ] && supported "$VPY" && "$VPY" -m pip --version >/dev/null 2>&1; then
  :  # existing .venv is fine
else
  rm -rf "$VENV"
  PY="$(find_python)" || PY=""
  if [ -n "$PY" ]; then
    say "Creating a virtual environment with $("$PY" -V 2>&1) ($PY)"
    if ! "$PY" -m venv "$VENV" >/dev/null 2>&1 || ! "$VPY" -m pip --version >/dev/null 2>&1; then
      echo "    That Python can't create virtual environments (on Debian/Ubuntu the python3-venv package is missing)."
      echo "    Using a private Python instead."
      rm -rf "$VENV"
      PY=""
    fi
  else
    say "No Python 3.10-3.12 found on this computer"
  fi
  if [ -z "$PY" ]; then
    get_uv
    say "Setting up Python 3.12 for this demo (downloaded into this folder if needed)"
    UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_CACHE_DIR="$TOOLS/uv-cache" \
      "$UV" venv --seed --python 3.12 "$VENV" \
      || fail "Couldn't set up Python 3.12. Install it from https://www.python.org/downloads/ and run this script again."
    "$VPY" -m pip --version >/dev/null 2>&1 || fail "The new Python environment has no pip. Delete the .venv folder and try again."
  fi
fi

# ---- 2. Packages (first run, or when requirements.txt changes) --------------------------
if ! cmp -s "$ROOT/requirements.txt" "$STAMP" || ! "$VPY" -c "$PRESENT" >/dev/null 2>&1; then
  say "Installing packages. First run only: PyTorch is a large download, so this can take several minutes"
  "$VPY" -m pip install --upgrade pip || fail "Couldn't update pip. Check your internet connection (see the messages above) and run this script again."

  say "Installing CPU-only PyTorch from $TORCH_INDEX_URL"
  if ! "$VPY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"; then
    echo "    Couldn't install from the PyTorch CPU index. Trying the standard package index (PyPI) instead..."
    "$VPY" -m pip install torch torchvision \
      || fail "Couldn't install PyTorch. The messages above say why (usually the internet connection or a full disk). Fix that and run this script again; finished downloads are kept."
  fi

  say "Installing the other packages from requirements.txt"
  "$VPY" -m pip install -r "$ROOT/requirements.txt" \
    || fail "Couldn't install the packages in requirements.txt. See the messages above."

  "$VPY" -c "$IMPORTS" || fail "The packages installed, but Python can't load them. See the error above."
  cp "$ROOT/requirements.txt" "$STAMP"
fi

# ---- 3. Start ---------------------------------------------------------------------------
say "Starting the demo at http://127.0.0.1:${PORT:-8000}"
exec "$VPY" -m app.launch
