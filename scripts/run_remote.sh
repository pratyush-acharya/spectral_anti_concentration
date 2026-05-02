#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Geometric Siege — Full Remote Experiment Pipeline
# ═══════════════════════════════════════════════════════════
#
#  End-to-end: push code → setup env → run 22 models → poll
#  until done → pull results → merge into local results/
#
#  Usage:
#    ./scripts/run_remote.sh              # full pipeline
#    ./scripts/run_remote.sh --skip-setup # skip setup (already done)
#    ./scripts/run_remote.sh --pull-only  # just download results
#
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments first (so --help works without config) ──
SKIP_SETUP=false
PULL_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --pull-only)  PULL_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-setup] [--pull-only]"
            echo "  --skip-setup  Skip remote environment setup (if already done)"
            echo "  --pull-only   Only download results (skip push/setup/run)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Load config from .env.deploy ──
ENV_FILE="$PROJECT_DIR/.env.deploy"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Copy .env.deploy.example to .env.deploy and fill in your SSH details."
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# Expand ~ in KEY path
KEY="${KEY/#\~/$HOME}"

# Validate required variables
for var in HOST PORT KEY USER REMOTE_DIR MODEL_LIST; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set in $ENV_FILE"
        exit 1
    fi
done

if [ "$HOST" = "123.45.67.89" ]; then
    echo "ERROR: HOST is still the placeholder value. Edit .env.deploy with your actual SSH details."
    exit 1
fi

# ── SSH / rsync helpers ──
SSH_CMD="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
RSYNC_SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"
REMOTE="$USER@$HOST"

ssh_run() {
    $SSH_CMD "$REMOTE" "$@"
}




echo "═══════════════════════════════════════════════════════"
echo "  Geometric Siege — Remote Experiment Pipeline"
echo "═══════════════════════════════════════════════════════"
echo "  Remote:  $REMOTE:$PORT"
echo "  Dir:     $REMOTE_DIR"
echo "  Models:  $(echo "$MODEL_LIST" | tr ',' '\n' | wc -l) models"
echo "═══════════════════════════════════════════════════════"
echo ""

# ════════════════════════════════════════════════════════════
# STEP 0: Verify SSH Connection
# ════════════════════════════════════════════════════════════
echo "▶ [0/5] Testing SSH connection..."
if ! ssh_run "echo 'SSH OK'" 2>/dev/null; then
    echo "ERROR: Cannot connect to $REMOTE on port $PORT"
    echo "Check your HOST, PORT, KEY, and USER in .env.deploy"
    exit 1
fi
echo "  ✓ Connected"
echo ""

if [ "$PULL_ONLY" = true ]; then
    # Jump straight to pull
    echo "▶ --pull-only mode: skipping to download..."
    echo ""
    # fall through to step 4
else

# ════════════════════════════════════════════════════════════
# STEP 1: Push Code
# ════════════════════════════════════════════════════════════
echo "▶ [1/5] Pushing code to remote..."
rsync -avz --delete -e "$RSYNC_SSH" \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'results' \
    --exclude 'remote_results' \
    --exclude '*.pt' \
    --exclude '.env' \
    --exclude '.env.deploy' \
    --exclude '.gemini' \
    --exclude '*.lock' \
    "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"
echo "  ✓ Code synced"
echo ""

# ════════════════════════════════════════════════════════════
# STEP 2: Setup Remote Environment
# ════════════════════════════════════════════════════════════
if [ "$SKIP_SETUP" = true ]; then
    echo "▶ [2/5] Skipping setup (--skip-setup)"
else
    echo "▶ [2/5] Setting up remote environment..."
    $SSH_CMD "$REMOTE" "REMOTE_DIR=$REMOTE_DIR HF_TOKEN=${HF_TOKEN:-} bash -s" < "$PROJECT_DIR/scripts/setup_remote.sh"
    echo "  ✓ Environment ready"
fi
echo ""

# ════════════════════════════════════════════════════════════
# STEP 3: Launch Experiment
# ════════════════════════════════════════════════════════════
echo "▶ [3/5] Launching experiment in tmux..."

# Kill any existing session
ssh_run "tmux kill-session -t raid 2>/dev/null || true"

# Start the full pipeline in tmux
ssh_run "cd $REMOTE_DIR && tmux new-session -d -s raid \
    'export PATH=\$HOME/.local/bin:\$PATH && \
    export MODEL_LIST=\"$MODEL_LIST\" && \
    export HF_TOKEN=\"${HF_TOKEN:-}\" && \
    echo \"Pipeline started at \$(date)\" && \
    uv run python scripts/perform_raid.py 2>&1 | tee raid.log; \
    echo \"\" && \
    echo \"═══ PIPELINE COMPLETE at \$(date) ═══\" | tee -a raid.log; \
    touch raid.done; \
    sleep 86400'"

echo "  ✓ Experiment launched in tmux session 'raid'"
echo ""

# ════════════════════════════════════════════════════════════
# STEP 3b: Poll Until Complete
# ════════════════════════════════════════════════════════════
echo "▶ [3b/5] Monitoring progress (Ctrl+C to detach — experiment keeps running)..."
echo "         Results will be pulled automatically when done."
echo ""

POLL_INTERVAL=10  # seconds between checks
LAST_LINES=""

while true; do
    # Check if done
    if ssh_run "test -f $REMOTE_DIR/raid.done" 2>/dev/null; then
        echo ""
        echo "  ✓ Experiment complete!"
        break
    fi
    
    # Check if tmux session is still alive
    if ! ssh_run "tmux has-session -t raid 2>/dev/null" 2>/dev/null; then
        echo ""
        echo "  ⚠ tmux session 'raid' ended (may have crashed)"
        echo "  Check logs: ssh -p $PORT -i $KEY $REMOTE 'cat $REMOTE_DIR/raid.log'"
        break
    fi

    # Show latest progress
    CURRENT_LINES=$(ssh_run "tail -3 $REMOTE_DIR/raid.log 2>/dev/null" 2>/dev/null || echo "")
    if [ "$CURRENT_LINES" != "$LAST_LINES" ] && [ -n "$CURRENT_LINES" ]; then
        echo "  [$(date '+%H:%M:%S')] $(echo "$CURRENT_LINES" | tail -1)"
        LAST_LINES="$CURRENT_LINES"
    fi
    
    sleep $POLL_INTERVAL
done

echo ""

fi  # end of PULL_ONLY skip

# ════════════════════════════════════════════════════════════
# STEP 4: Pull Results
# ════════════════════════════════════════════════════════════
echo "▶ [4/5] Downloading results..."
mkdir -p "$PROJECT_DIR/remote_results"
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/results/" "$PROJECT_DIR/remote_results/"

# Also grab the log
rsync -avz -e "$RSYNC_SSH" \
    "$REMOTE:$REMOTE_DIR/raid.log" "$PROJECT_DIR/remote_results/" 2>/dev/null || true

echo "  ✓ Results downloaded to remote_results/"
echo ""

# ════════════════════════════════════════════════════════════
# STEP 5: Merge into Local Results
# ════════════════════════════════════════════════════════════
echo "▶ [5/5] Merging into local results/..."
mkdir -p "$PROJECT_DIR/results"
rsync -av --ignore-existing "$PROJECT_DIR/remote_results/" "$PROJECT_DIR/results/"

# Summary
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Pipeline Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Results:"
for d in "$PROJECT_DIR"/results/*/; do
    model=$(basename "$d")
    if [ -f "$d/stats.txt" ]; then
        echo "    ✓ $model"
    elif [ -f "$d/error.txt" ]; then
        echo "    ✗ $model (failed — see results/$model/error.txt)"
    else
        echo "    ? $model (incomplete)"
    fi
done
echo ""
echo "  Log: remote_results/raid.log"
echo "  To re-run: $0 --skip-setup"
echo "═══════════════════════════════════════════════════════"
