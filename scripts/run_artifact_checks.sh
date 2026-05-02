#!/bin/bash
# Description: Sync local results folder up to the remote server, run all artifact_check_*.py 
# scripts on the remote server, and sync the generated results back down to local.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load config from .env.deploy ──
ENV_FILE="$PROJECT_DIR/.env.deploy"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# Expand ~ in KEY path
KEY="${KEY/#\~/$HOME}"

# Validate required variables
for var in HOST PORT KEY USER REMOTE_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set in $ENV_FILE"
        exit 1
    fi
done

RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
SSH_CMD="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
REMOTE="$USER@$HOST"

echo "=========================================================="
echo "  Deploying Data & Artifact Checks to Remote GPU"
echo "=========================================================="

echo "▶ [1/4] Preparing remote directories..."
if $SSH_CMD "$REMOTE" "[ -d $REMOTE_DIR/results ]"; then
    echo "  > Remote results directory exists. Skipping results sync..."
    SYNC_RESULTS=false
else
    SYNC_RESULTS=true
fi
$SSH_CMD "$REMOTE" "mkdir -p $REMOTE_DIR/results/_spectral/artifact_check $REMOTE_DIR/scripts"

echo "▶ [2/4] Pushing required results data and scripts to remote..."
if [ "$SYNC_RESULTS" = true ]; then
    # Only push the specific files needed for artifact checks to save bandwidth
    rsync -avz -e "$RSYNC_SSH" \
        --include="*/" \
        --include="cov_src.pt" \
        --include="cov_en.pt" \
        --include="eigen.pt" \
        --include="*.json" \
        --exclude="*" \
        "$PROJECT_DIR/results/" "$REMOTE:$REMOTE_DIR/results/"
fi

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/scripts/" "$REMOTE:$REMOTE_DIR/scripts/"

rsync -avz -e "$RSYNC_SSH" \
    "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$REMOTE:$REMOTE_DIR/"
echo "  ✓ Local files synced to remote"
echo ""

echo "▶ [3/4] Executing artifact checks on remote..."
# Run the pipeline directly so you can see live logs in this terminal.
$SSH_CMD "$REMOTE" "
    cd $REMOTE_DIR
    export PATH=\$HOME/.local/bin:\$PATH
    export HF_TOKEN=\"${HF_TOKEN:-}\"
    
    echo '======================================'
    echo '  Running artifact_check_probe.py'
    echo '======================================'
    uv run python scripts/artifact_check_probe.py || echo 'Warning: artifact_check_probe.py failed'
    echo ''
    
    echo '======================================'
    echo '  Running artifact_check_sae.py'
    echo '======================================'
    uv run python scripts/artifact_check_sae.py || echo 'Warning: artifact_check_sae.py failed'
    echo ''
    
    echo '======================================'
    echo '  Running artifact_check_unembed.py'
    echo '======================================'
    uv run python scripts/artifact_check_unembed.py || echo 'Warning: artifact_check_unembed.py failed'
    echo ''
    
    echo '======================================'
    echo '  ALL ARTIFACT CHECKS COMPLETE'
    echo '======================================'
"
echo "  ✓ Remote execution complete"
echo ""

echo "▶ [4/4] Pulling generated results back down..."
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/_spectral/artifact_check/" "$PROJECT_DIR/results/_spectral/artifact_check/"
echo "  ✓ Results downloaded and merged back into local results/"
echo ""
echo "Pipeline complete! Artifact check outputs are in results/_spectral/artifact_check/"
