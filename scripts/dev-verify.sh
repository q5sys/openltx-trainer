#!/usr/bin/env bash
# dev-verify.sh
# Quick build, install (unpacked), and run script for verifying the app state.
# Linux-first. No code signing, no installer packaging.
#
# Usage:
#   bash scripts/dev-verify.sh [options]
#
# Options:
#   --skip-setup     Skip pnpm install and uv sync (use existing deps)
#   --skip-build     Skip frontend build (use existing dist/)
#   --run-only       Skip everything, just run the unpacked app
#   --typecheck      Run typecheck before building
#   --backend-test   Run backend tests before building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_DIR/release"
BACKEND_DIR="$PROJECT_DIR/backend"
PYTHON_VERSION="3.13"

SKIP_SETUP=false
SKIP_BUILD=false
RUN_ONLY=false
DO_TYPECHECK=false
DO_BACKEND_TEST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)    SKIP_SETUP=true ;;
    --skip-build)    SKIP_BUILD=true ;;
    --run-only)      RUN_ONLY=true ;;
    --typecheck)     DO_TYPECHECK=true ;;
    --backend-test)  DO_BACKEND_TEST=true ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--skip-setup] [--skip-build] [--run-only] [--typecheck] [--backend-test]"
      exit 1
      ;;
  esac
  shift
done

cd "$PROJECT_DIR"

echo ""
echo "========================================"
echo "  OpenLTX Trainer - Dev Verify"
echo "========================================"
echo ""

# Ensure required tools are on PATH
check_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: $1 not found on PATH."
    echo "  PATH=$PATH"
    exit 1
  fi
}

# Find the unpacked app binary
find_unpacked_binary() {
  local binary=""
  if [ -f "$RELEASE_DIR/linux-unpacked/openltx-trainer" ]; then
    binary="$RELEASE_DIR/linux-unpacked/openltx-trainer"
  elif [ -f "$RELEASE_DIR/linux-unpacked/ltx-desktop" ]; then
    binary="$RELEASE_DIR/linux-unpacked/ltx-desktop"
  elif [ -d "$RELEASE_DIR/linux-unpacked" ]; then
    binary="$(find "$RELEASE_DIR/linux-unpacked" -maxdepth 1 -type f -executable | head -1)"
  fi
  echo "$binary"
}

# --run-only: just launch the existing unpacked app
if [ "$RUN_ONLY" = true ]; then
  binary="$(find_unpacked_binary)"
  if [ -z "$binary" ]; then
    echo "ERROR: No unpacked app found at $RELEASE_DIR/linux-unpacked/"
    echo "Run without --run-only to build first."
    exit 1
  fi
  echo "Launching: $binary"
  exec "$binary"
fi

check_tool pnpm
check_tool node
check_tool uv

echo "Using: node $(node -v), pnpm $(pnpm -v), uv $(uv --version)"
echo ""

# Step 1: Setup dependencies
if [ "$SKIP_SETUP" = false ]; then
  echo "[1] Installing dependencies..."

  # Install pnpm deps
  pnpm install

  # Ensure Python 3.13 is available via uv
  echo "  Ensuring Python $PYTHON_VERSION is installed via uv..."
  uv python install "$PYTHON_VERSION"

  # Create/sync the backend venv with the correct Python
  cd "$BACKEND_DIR"
  if [ ! -d ".venv" ]; then
    echo "  Creating backend venv with Python $PYTHON_VERSION..."
    uv venv --python "$PYTHON_VERSION"
  fi
  echo "  Syncing backend dependencies..."
  uv sync --frozen
  cd "$PROJECT_DIR"
  echo ""
else
  echo "[1] Skipping dependency setup (--skip-setup)"
  echo ""
fi

# Step 2: Typecheck (optional)
if [ "$DO_TYPECHECK" = true ]; then
  echo "[2] Running typecheck..."
  pnpm typecheck
  echo ""
else
  echo "[2] Skipping typecheck (use --typecheck to enable)"
  echo ""
fi

# Step 3: Backend tests (optional)
if [ "$DO_BACKEND_TEST" = true ]; then
  echo "[3] Running backend tests..."
  pnpm backend:test
  echo ""
else
  echo "[3] Skipping backend tests (use --backend-test to enable)"
  echo ""
fi

# Step 4: Build frontend + electron
if [ "$SKIP_BUILD" = false ]; then
  echo "[4] Building frontend..."
  pnpm run build:frontend
  echo ""
else
  echo "[4] Skipping build (--skip-build)"
  echo ""
fi

# Step 5: Package unpacked app (no installer, no python-embed needed on Linux)
echo "[5] Packaging unpacked app..."
pnpm exec electron-builder --linux --dir
echo ""

# Step 6: Launch
binary="$(find_unpacked_binary)"
if [ -z "$binary" ]; then
  echo "ERROR: Could not find unpacked binary in $RELEASE_DIR/linux-unpacked/"
  echo "Listing contents:"
  ls -la "$RELEASE_DIR/linux-unpacked/" 2>/dev/null || echo "  (directory does not exist)"
  exit 1
fi

echo "========================================"
echo "  Build complete. Launching app..."
echo "========================================"
echo ""
echo "Binary: $binary"
echo ""

exec "$binary"
