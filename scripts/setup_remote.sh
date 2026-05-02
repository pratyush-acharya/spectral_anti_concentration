#!/bin/bash
set -e

# Define project directory (passed from Makefile via REMOTE_DIR env var)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Setting up environment on remote pod ==="

# Ensure we are in the right directory (created by rsync)
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR not found. Did you run 'make push'?"
    exit 1
fi
cd "$PROJECT_DIR"

# 1. Install system dependencies
echo "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git tmux build-essential curl > /dev/null 2>&1

# 2. Install uv (handles Python version + venv + deps)
echo "Installing uv..."
if ! command -v uv &> /dev/null; then
    sudo mkdir -p "$HOME/.config" "$HOME/.local/bin"
    sudo chown -R "$(id -u):$(id -g)" "$HOME/.config" "$HOME/.local"
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv version: $(uv --version)"

# 3. Sync environment (uv auto-installs Python 3.12+ and all deps)
echo "Syncing environment with uv..."
uv sync

# 4. Install flash-attn for faster inference (optional, best-effort)
echo "Attempting flash-attn install (optional)..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn not available (OK, will use standard attention)"

# 5. Login to HuggingFace for gated models (Llama, etc.)
if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "hf_your_token_here" ]; then
    echo "Logging into HuggingFace..."
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    echo "  HF login complete."
else
    echo "  No HF_TOKEN provided. Gated models (e.g., Llama) will not be accessible."
fi

echo "=== Setup Complete! ==="
echo "  Python: $(uv run python --version)"
echo "  Torch: $(uv run python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
