# Load connection details from .env.deploy
-include .env.deploy
export

SSH_CMD = ssh -p $(PORT) -i $(KEY) -o StrictHostKeyChecking=no $(USER)@$(HOST)
RSYNC_SSH = ssh -p $(PORT) -i $(KEY) -o StrictHostKeyChecking=no

.PHONY: help push setup run pull ssh deploy status merge

help:
	@echo "Geometric Siege - Remote Deployment Automation"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make deploy   - Push code, setup env, run ALL models, pull results"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make push     - Sync code to remote pod"
	@echo "  make setup    - Install dependencies on remote pod"
	@echo "  make run      - Run the full experiment pipeline (RAID + analysis + ablation)"
	@echo "  make pull     - Download results from remote pod"
	@echo "  make merge    - Merge remote_results/ into local results/"
	@echo "  make ssh      - Open an interactive SSH session"
	@echo "  make status   - Check if the experiment is still running"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit .env.deploy to set HOST, PORT, KEY, HF_TOKEN, MODEL_LIST, etc."

push:
	@echo "═══ Syncing code to $(HOST) ═══"
	rsync -avz -e "$(RSYNC_SSH)" \
		--exclude '.venv' \
		--exclude '.git' \
		--exclude '__pycache__' \
		--exclude 'results' \
		--exclude '*.pt' \
		--exclude '.env' \
		--exclude '.env.deploy' \
		--exclude 'remote_results' \
		--exclude '.gemini' \
		. $(USER)@$(HOST):$(REMOTE_DIR)

setup:
	@echo "═══ Running setup on remote ═══"
	$(SSH_CMD) 'REMOTE_DIR=$(REMOTE_DIR) HF_TOKEN=$(HF_TOKEN) bash -s' < scripts/setup_remote.sh

run:
	@echo "═══ Starting full pipeline on $(HOST) in tmux session 'raid' ═══"
	$(SSH_CMD) \
		"cd $(REMOTE_DIR) && tmux kill-session -t raid 2>/dev/null || true && \
		tmux new-session -d -s raid \
		'export PATH=\$$HOME/.local/bin:\$$PATH && \
		export MODEL_LIST=\"$(MODEL_LIST)\" && \
		export HF_TOKEN=\"$(HF_TOKEN)\" && \
		uv run python scripts/perform_raid.py 2>&1 | tee raid.log; \
		echo \"=== ALL EXPERIMENTS COMPLETE ===\"; \
		sleep 86400'"
	@echo ""
	@echo "Pipeline started in tmux session 'raid'."
	@echo "  Monitor:  make ssh → tmux attach -t raid"
	@echo "  Status:   make status"
	@echo "  Logs:     make ssh → tail -f $(REMOTE_DIR)/raid.log"

pull:
	@echo "═══ Downloading results from $(HOST) ═══"
	mkdir -p remote_results
	rsync -avz -e "$(RSYNC_SSH)" \
		$(USER)@$(HOST):$(REMOTE_DIR)/results/ ./remote_results/
	@echo ""
	@echo "Results downloaded to remote_results/."
	@echo "Run 'make merge' to copy into local results/."

merge:
	@echo "═══ Merging remote_results/ into results/ ═══"
	rsync -av --ignore-existing remote_results/ results/
	@echo "Done."

status:
	@$(SSH_CMD) 'tmux has-session -t raid 2>/dev/null && echo "🔄 Experiment is RUNNING" || echo "✅ Experiment is DONE (or not started)"'
	@echo "--- Last 10 lines of log ---"
	@$(SSH_CMD) 'tail -10 $(REMOTE_DIR)/raid.log 2>/dev/null || echo "No log file yet"'

ssh:
	$(SSH_CMD)

deploy: push setup run
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo "  Deploy complete! Pipeline is running."
	@echo "  Use 'make status' to check progress."
	@echo "  Use 'make pull' when done."
	@echo "═══════════════════════════════════════════"
