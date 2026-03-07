PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

PARAMS ?=
_PARAMS_FLAG := $(if $(PARAMS),--params $(PARAMS),)
N_JOBS ?=
_N_JOBS_FLAG := $(if $(N_JOBS),--n-jobs $(N_JOBS),)
PARALLEL_TRIALS ?=
_PARALLEL_TRIALS_FLAG := $(if $(PARALLEL_TRIALS),--parallel-trials $(PARALLEL_TRIALS),)
PARALLEL_DOMAINS ?=
_PARALLEL_DOMAINS_FLAG := $(if $(PARALLEL_DOMAINS),--parallel-domains $(PARALLEL_DOMAINS),)
TRAIN_PARALLEL ?=
_TRAIN_PARALLEL_FLAG := $(if $(TRAIN_PARALLEL),-j$(TRAIN_PARALLEL),)
GPU ?=
_GPU_FLAG := $(if $(GPU),--gpu,)
# train-1 runs alone before train-2/3/4, so give it all cores (N_JOBS × TRAIN_PARALLEL)
_TRAIN1_NJOBS = $(if $(and $(TRAIN_PARALLEL),$(N_JOBS)),$(shell echo $$(( $(N_JOBS) * $(TRAIN_PARALLEL) ))),$(N_JOBS))
MODEL_DIR ?= models/reference
MODEL_DIR_NORM := $(patsubst %/,%,$(MODEL_DIR))
MODEL_NAME := $(notdir $(MODEL_DIR_NORM))
DATA_ROOT ?= data/processed
DATA_DIR_DEFAULT ?= $(DATA_ROOT)/ext_est
DATA_DIR_STRATIFIED ?= $(DATA_ROOT)/ext_est_opn
DATA_DIR ?=
ARTIFACTS_DIR ?= artifacts
SKIP_PROVENANCE ?=
FORCE ?=
RESET ?=
_UPLOAD_HF_DEPS := $(if $(SKIP_PROVENANCE),,provenance-check)
_RESET_FLAG := $(if $(RESET),--reset,)
ARTIFACTS_VARIANTS_DIR ?= $(ARTIFACTS_DIR)/variants
EVAL_DIR = $(ARTIFACTS_VARIANTS_DIR)/$(MODEL_NAME)
RESEARCH_SUMMARY_PATH ?= $(ARTIFACTS_DIR)/research_summary.json

TRAIN_DATA_DIR ?=
TRAIN_DATA_DIR_STRATIFIED ?=

ifeq ($(strip $(DATA_DIR)),)
ifeq ($(MODEL_NAME),ablation_stratified)
DATA_DIR := $(DATA_DIR_STRATIFIED)
else
DATA_DIR := $(DATA_DIR_DEFAULT)
endif
endif

ifeq ($(strip $(TRAIN_DATA_DIR)),)
TRAIN_DATA_DIR := $(DATA_DIR)
endif

ifeq ($(strip $(TRAIN_DATA_DIR_STRATIFIED)),)
ifneq ($(strip $(TRAIN_DATA_DIR)),)
ifneq ($(TRAIN_DATA_DIR),$(DATA_DIR_DEFAULT))
TRAIN_DATA_DIR_STRATIFIED := $(TRAIN_DATA_DIR)
else
TRAIN_DATA_DIR_STRATIFIED := $(DATA_DIR_STRATIFIED)
endif
else
TRAIN_DATA_DIR_STRATIFIED := $(DATA_DIR_STRATIFIED)
endif
endif

TRAIN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
TRAIN_RUN := $(firstword $(TRAIN_ARGS))
TRAIN_EXTRA_ARGS := $(wordlist 2,$(words $(TRAIN_ARGS)),$(TRAIN_ARGS))
VALID_TRAIN_RUNS := 1 2 3 4

.PHONY: all setup setup-python setup-typescript setup-web download load norms norms-check provenance-check provenance-check-full prepare prepare-default prepare-stratified correlations correlations-default correlations-stratified tune train train-1 train-2 train-3 train-4 check-model-data-pairing validate baselines simulate export export-all export-repo-readme export-reference export-ablation-none export-ablation-focused export-ablation-stratified figures research-eval research-eval-reference research-eval-ablation-none research-eval-ablation-focused research-eval-ablation-stratified research-summary research-summary-strict notes upload-hf upload-hf-reference test test-lib test-inference test-web archive clean web-setup web-dev web-build deploy-web

all: download load norms norms-check prepare correlations tune train research-eval export-all notes figures

setup: setup-python setup-typescript setup-web

setup-python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

setup-typescript:
	cd typescript && npm ci

setup-web:
	cd web && npm ci

download:
	$(PY) pipeline/01_download.py

load:
	$(PY) pipeline/02_load_sqlite.py

norms:
	$(PY) pipeline/03_compute_norms.py

norms-check:
	$(PY) pipeline/03_compute_norms.py --check

provenance-check:
	$(MAKE) norms-check
	$(PY) scripts/check_provenance.py --strict

provenance-check-full:
	$(MAKE) norms-check
	$(PY) scripts/check_provenance.py --strict --full

prepare: prepare-default prepare-stratified

prepare-default:
	$(PY) pipeline/04_prepare_data.py --stratification ext-est --output-dir $(DATA_DIR_DEFAULT)

prepare-stratified:
	$(PY) pipeline/04_prepare_data.py --stratification ext-est-opn --output-dir $(DATA_DIR_STRATIFIED)

correlations: correlations-default correlations-stratified

correlations-default:
	$(PY) pipeline/05_compute_correlations.py --data-dir $(DATA_DIR_DEFAULT)

correlations-stratified:
	$(PY) pipeline/05_compute_correlations.py --data-dir $(DATA_DIR_STRATIFIED)

tune:
	$(PY) pipeline/06_tune.py --config configs/reference.yaml --data-dir $(DATA_DIR_DEFAULT) --artifacts-dir $(ARTIFACTS_DIR) $(_N_JOBS_FLAG) $(_PARALLEL_TRIALS_FLAG) $(_GPU_FLAG)

train:
ifneq ($(strip $(TRAIN_EXTRA_ARGS)),)
	@echo "Too many train run arguments: $(TRAIN_ARGS). Use: make train [1|2|3|4]"
	@exit 2
else ifeq ($(TRAIN_RUN),)
	$(MAKE) train-1 PARAMS="$(PARAMS)" N_JOBS="$(_TRAIN1_NJOBS)"
	$(MAKE) $(_TRAIN_PARALLEL_FLAG) train-2 train-3 train-4 PARAMS="$(PARAMS)" N_JOBS="$(N_JOBS)"
else ifeq ($(filter $(TRAIN_RUN),$(VALID_TRAIN_RUNS)),$(TRAIN_RUN))
	$(MAKE) train-$(TRAIN_RUN) PARAMS="$(PARAMS)" N_JOBS="$(N_JOBS)"
else
	@echo "Invalid train run index: $(TRAIN_RUN). Use: make train [1|2|3|4]"
	@exit 2
endif

train-1:
	$(PY) pipeline/07_train.py --config configs/reference.yaml --data-dir $(TRAIN_DATA_DIR) --artifacts-dir $(ARTIFACTS_DIR) $(_PARAMS_FLAG) $(_N_JOBS_FLAG) $(_PARALLEL_DOMAINS_FLAG) $(_GPU_FLAG)

train-2:
	$(PY) pipeline/07_train.py --config configs/ablation_none.yaml --data-dir $(TRAIN_DATA_DIR) --artifacts-dir $(ARTIFACTS_DIR) $(_PARAMS_FLAG) $(_N_JOBS_FLAG) $(_PARALLEL_DOMAINS_FLAG) $(_GPU_FLAG)

train-3:
	$(PY) pipeline/07_train.py --config configs/ablation_focused.yaml --data-dir $(TRAIN_DATA_DIR) --artifacts-dir $(ARTIFACTS_DIR) $(_PARAMS_FLAG) $(_N_JOBS_FLAG) $(_PARALLEL_DOMAINS_FLAG) $(_GPU_FLAG)

train-4:
	$(PY) pipeline/07_train.py --config configs/ablation_stratified.yaml --data-dir $(TRAIN_DATA_DIR_STRATIFIED) --artifacts-dir $(ARTIFACTS_DIR) $(_PARAMS_FLAG) $(_N_JOBS_FLAG) $(_PARALLEL_DOMAINS_FLAG) $(_GPU_FLAG)

check-model-data-pairing:
	@expected=""; \
	case "$(MODEL_NAME)" in \
		reference|ablation_none|ablation_focused) expected="$(DATA_DIR_DEFAULT)" ;; \
		ablation_stratified) expected="$(DATA_DIR_STRATIFIED)" ;; \
	esac; \
	if [ -n "$$expected" ] && [ "$(DATA_DIR)" != "$$expected" ]; then \
		echo "Model/data mismatch: MODEL_DIR=$(MODEL_DIR_NORM) expects DATA_DIR=$$expected but got DATA_DIR=$(DATA_DIR)"; \
		echo "Override MODEL_DIR or DATA_DIR to a matching regime."; \
		exit 2; \
	fi

validate: check-model-data-pairing
	@mkdir -p $(EVAL_DIR)
	$(PY) pipeline/08_validate.py --model-dir $(MODEL_DIR_NORM) --data-dir $(DATA_DIR) --artifacts-dir $(EVAL_DIR)

baselines: check-model-data-pairing
	@mkdir -p $(EVAL_DIR)
	$(PY) pipeline/09_baselines.py --model-dir $(MODEL_DIR_NORM) --data-dir $(DATA_DIR) --artifacts-dir $(EVAL_DIR) --bootstrap-n 1000

simulate: check-model-data-pairing
	@mkdir -p $(EVAL_DIR)
	$(PY) pipeline/10_simulate.py --model-dir $(MODEL_DIR_NORM) --data-dir $(DATA_DIR) --artifacts-dir $(EVAL_DIR) --n-sample 5000

export: check-model-data-pairing
	$(PY) pipeline/11_export_onnx.py --model-dir $(MODEL_DIR_NORM) --data-dir $(DATA_DIR) --artifacts-dir $(EVAL_DIR) --output-dir output/$(MODEL_NAME)

export-all: export-reference export-ablation-none export-ablation-focused export-ablation-stratified export-repo-readme

export-repo-readme:
	$(PY) pipeline/11_export_onnx.py --repo-readme --output-dir output

export-reference:
	$(MAKE) export MODEL_DIR=models/reference DATA_DIR=$(DATA_DIR_DEFAULT)

export-ablation-none:
	$(MAKE) export MODEL_DIR=models/ablation_none DATA_DIR=$(DATA_DIR_DEFAULT)

export-ablation-focused:
	$(MAKE) export MODEL_DIR=models/ablation_focused DATA_DIR=$(DATA_DIR_DEFAULT)

export-ablation-stratified:
	$(MAKE) export MODEL_DIR=models/ablation_stratified DATA_DIR=$(DATA_DIR_STRATIFIED)

figures:
	$(PY) pipeline/12_generate_figures.py --artifacts-dir $(EVAL_DIR)

research-eval: research-eval-reference research-eval-ablation-none research-eval-ablation-focused research-eval-ablation-stratified

research-eval-reference:
	$(MAKE) validate baselines simulate MODEL_DIR=models/reference DATA_DIR=$(DATA_DIR_DEFAULT)

research-eval-ablation-none:
	$(MAKE) validate baselines simulate MODEL_DIR=models/ablation_none DATA_DIR=$(DATA_DIR_DEFAULT)

research-eval-ablation-focused:
	$(MAKE) validate baselines simulate MODEL_DIR=models/ablation_focused DATA_DIR=$(DATA_DIR_DEFAULT)

research-eval-ablation-stratified:
	$(MAKE) validate baselines simulate MODEL_DIR=models/ablation_stratified DATA_DIR=$(DATA_DIR_STRATIFIED)

research-summary:
	$(PY) scripts/build_research_summary.py --output $(RESEARCH_SUMMARY_PATH) --artifacts-variants-dir $(ARTIFACTS_VARIANTS_DIR)

research-summary-strict:
	$(PY) scripts/build_research_summary.py --strict --output $(RESEARCH_SUMMARY_PATH) --artifacts-variants-dir $(ARTIFACTS_VARIANTS_DIR)

notes:
	$(MAKE) research-summary-strict
	$(PY) scripts/generate_notes_data.py

upload-hf: $(_UPLOAD_HF_DEPS)
	$(PY) pipeline/13_upload_hf.py $(_RESET_FLAG)

upload-hf-reference: $(_UPLOAD_HF_DEPS)
	$(PY) pipeline/13_upload_hf.py --variant reference $(_RESET_FLAG)

test: test-lib test-inference test-web

test-lib:
	$(PY) -m pytest tests/ -v --tb=short

test-inference:
	cd python && ../$(PY) -m pytest -v
	cd typescript && npx vitest run

test-web:
	cd web && npx vitest run

archive:
	git archive --format=zip HEAD -o data/bffm-xgb-src.zip -- . ':!output/*.onnx'

clean:
	@found=""; \
	for t in \
		data/raw \
		data/processed \
		artifacts/ipip_bffm_norms.meta.json \
		artifacts/tuned_params.original.json \
		artifacts/research_summary.json \
		artifacts/variants \
		output/README.md \
		figures/manifest.json \
		.git-hash \
		pipeline.log \
		pipeline-timing.log \
		.pipeline-exit-code; \
	do \
		[ -e "$$t" ] && found="$$found  $$t\n"; \
	done; \
	for t in models/*/; do \
		[ -d "$$t" ] && found="$$found  $$t\n"; \
	done; \
	for t in output/*/; do \
		[ -d "$$t" ] && found="$$found  $$t\n"; \
	done; \
	for t in figures/*.png figures/*.pdf; do \
		[ -f "$$t" ] && found="$$found  $$t\n"; \
	done; \
	if [ -z "$$found" ]; then \
		echo "Nothing to clean."; \
		exit 0; \
	fi; \
	echo ""; \
	echo "WARNING: The following pipeline outputs will be deleted:"; \
	echo ""; \
	printf '%b' "$$found"; \
	echo ""; \
	echo "These files are all regenerated by running 'make all' (full pipeline)."; \
	echo ""; \
	if [ "$(FORCE)" != "1" ]; then \
		if [ ! -t 0 ]; then \
			echo "No TTY attached. Use FORCE=1 to skip confirmation."; \
			exit 1; \
		fi; \
		printf "Proceed? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]*) ;; *) echo "Aborted."; exit 1;; esac; \
	fi; \
	rm -rf data/raw data/processed \
		artifacts/ipip_bffm_norms.meta.json \
		artifacts/tuned_params.original.json \
		artifacts/research_summary.json \
		artifacts/variants \
		output/README.md \
		figures/manifest.json \
		.git-hash \
		pipeline.log \
		pipeline-timing.log \
		.pipeline-exit-code; \
	rm -rf models/*/; \
	rm -rf output/*/; \
	rm -f figures/*.png figures/*.pdf; \
	echo "Clean complete."

ifneq ($(filter train,$(firstword $(MAKECMDGOALS))),)
ifneq ($(strip $(TRAIN_ARGS)),)
.PHONY: $(TRAIN_ARGS)
$(TRAIN_ARGS):
	@:
endif
endif

# ---------------------------------------------------------------------------
# Web app targets
# ---------------------------------------------------------------------------

web-setup:
	cd web && npm ci

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

deploy-web: web-build
	$(PY) scripts/deploy_web.py

# ---------------------------------------------------------------------------
# Remote AWS targets
# ---------------------------------------------------------------------------
SSH_KEY       ?= ~/.ssh/id_rsa
SSH_OPTS       = -i $(SSH_KEY) -o StrictHostKeyChecking=no -o ConnectTimeout=10
CPU_USER      := ec2-user
CPU_HOST       = $(shell cd infra/cpu && terraform output -raw instance_ip 2>/dev/null)
CPU_DIR        = /home/$(CPU_USER)/bffm-xgb
CPU_SSH        = ssh $(SSH_OPTS) $(CPU_USER)@$(CPU_HOST)
CPU_RSYNC      = rsync -avz --progress -e "ssh $(SSH_OPTS)"
GPU_USER      := ubuntu
GPU_HOST       = $(shell cd infra/gpu && terraform output -raw instance_ip 2>/dev/null)
GPU_DIR        = /home/$(GPU_USER)/bffm-xgb
GPU_SSH        = ssh $(SSH_OPTS) $(GPU_USER)@$(GPU_HOST)
GPU_RSYNC      = rsync -avz --progress -e "ssh $(SSH_OPTS)"
# Backward compat defaults (point to CPU)
REMOTE_USER   ?= $(CPU_USER)
REMOTE_HOST    = $(CPU_HOST)
SSH            = $(CPU_SSH)
RSYNC          = $(CPU_RSYNC)
REMOTE_DIR     = $(CPU_DIR)
REMOTE_NJOBS  ?= 96
REMOTE_POLL_MAX ?= 360
REMOTE_PARALLEL_TRIALS ?= 4
REMOTE_PARALLEL_DOMAINS ?= 5

FILE ?=

.PHONY: infra-up infra-down infra-ssh infra-cpu-up infra-cpu-down infra-cpu-ssh infra-gpu-up infra-gpu-down infra-gpu-ssh remote-push remote-pull remote-setup remote-tune remote-train remote-research-eval remote-attach remote-status remote-all remote-all-1 remote-all-2 remote-1-gpu remote-2-cpu

infra-cpu-up:
	@echo "==> Initializing and applying Terraform (CPU)..."
	cd infra/cpu && terraform init -input=false && terraform apply -auto-approve
	@echo "==> Waiting for SSH to become available..."
	@IP=$$(cd infra/cpu && terraform output -raw instance_ip); \
	SSH_OK=0; \
	for i in $$(seq 1 30); do \
		ssh -i $(SSH_KEY) -o StrictHostKeyChecking=no -o ConnectTimeout=10 $(CPU_USER)@$$IP exit 2>/dev/null && SSH_OK=1 && break; \
		echo "    Attempt $$i/30 — waiting 10s..."; \
		sleep 10; \
	done; \
	if [ "$$SSH_OK" -ne 1 ]; then \
		echo "ERROR: SSH not available after 30 attempts. Aborting."; \
		exit 1; \
	fi; \
	echo "==> SSH available. Waiting for user-data (cloud-init) to finish..."; \
	SETUP_OK=0; \
	for i in $$(seq 1 60); do \
		ssh -i $(SSH_KEY) -o StrictHostKeyChecking=no -o ConnectTimeout=10 $(CPU_USER)@$$IP \
			'test -f /home/$(CPU_USER)/.setup-done' 2>/dev/null && SETUP_OK=1 && break; \
		echo "    cloud-init attempt $$i/60 — waiting 5s..."; \
		sleep 5; \
	done; \
	if [ "$$SETUP_OK" -ne 1 ]; then \
		echo "ERROR: user-data setup did not complete after 60 attempts. Aborting."; \
		exit 1; \
	fi; \
	echo ""; \
	echo "==> Instance ready!"; \
	echo "    IP:  $$IP"; \
	echo "    SSH: ssh -i $(SSH_KEY) $(CPU_USER)@$$IP"; \
	echo ""

infra-cpu-down:
	@echo "==> Destroying CPU infrastructure..."
	cd infra/cpu && terraform destroy -auto-approve

infra-cpu-ssh:
	$(CPU_SSH) -t

infra-gpu-up:
	@echo "==> Initializing and applying Terraform (GPU)..."
	cd infra/gpu && terraform init -input=false && terraform apply -auto-approve
	@echo "==> Waiting for SSH to become available..."
	@IP=$$(cd infra/gpu && terraform output -raw instance_ip); \
	SSH_OK=0; \
	for i in $$(seq 1 30); do \
		ssh -i $(SSH_KEY) -o StrictHostKeyChecking=no -o ConnectTimeout=10 $(GPU_USER)@$$IP exit 2>/dev/null && SSH_OK=1 && break; \
		echo "    Attempt $$i/30 — waiting 10s..."; \
		sleep 10; \
	done; \
	if [ "$$SSH_OK" -ne 1 ]; then \
		echo "ERROR: SSH not available after 30 attempts. Aborting."; \
		exit 1; \
	fi; \
	echo "==> SSH available. Waiting for user-data (cloud-init) to finish..."; \
	SETUP_OK=0; \
	for i in $$(seq 1 60); do \
		ssh -i $(SSH_KEY) -o StrictHostKeyChecking=no -o ConnectTimeout=10 $(GPU_USER)@$$IP \
			'test -f /home/$(GPU_USER)/.setup-done' 2>/dev/null && SETUP_OK=1 && break; \
		echo "    cloud-init attempt $$i/60 — waiting 5s..."; \
		sleep 5; \
	done; \
	if [ "$$SETUP_OK" -ne 1 ]; then \
		echo "ERROR: user-data setup did not complete after 60 attempts. Aborting."; \
		exit 1; \
	fi; \
	echo ""; \
	echo "==> GPU instance ready!"; \
	echo "    IP:  $$IP"; \
	echo "    SSH: ssh -i $(SSH_KEY) $(GPU_USER)@$$IP"; \
	echo ""

infra-gpu-down:
	@echo "==> Destroying GPU infrastructure..."
	cd infra/gpu && terraform destroy -auto-approve

infra-gpu-ssh:
	$(GPU_SSH) -t

# Backward compat aliases
infra-up: infra-cpu-up
infra-down: infra-cpu-down
infra-ssh: infra-cpu-ssh

remote-push:
ifdef FILE
	@echo "==> Uploading $(FILE) to $(REMOTE_HOST):$(REMOTE_DIR)/$(FILE)..."
	$(RSYNC) ./$(FILE) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/$(FILE)
else
	@printf '%s\n' '$(shell git rev-parse HEAD 2>/dev/null || echo unknown)' > .git-hash
	@echo "==> Uploading project to $(REMOTE_HOST):$(REMOTE_DIR)..."
	$(RSYNC) \
		--exclude='.venv/' \
		--exclude='__pycache__/' \
		--exclude='node_modules/' \
		--exclude='.git/' \
		--exclude='*.pyc' \
		--exclude='.terraform/' \
		--exclude='*.tfstate' \
		--exclude='*.tfstate.backup' \
		--exclude='.terraform.lock.hcl' \
		--exclude='terraform.tfvars' \
		--exclude='.env' \
		--exclude='.DS_Store' \
		--exclude='.claude/settings.local.json' \
		--exclude='data/' \
		--exclude='artifacts/variants/' \
		--exclude='artifacts/ipip_bffm_norms.meta.json' \
		--exclude='artifacts/research_summary.json' \
		--exclude='output/' \
		--exclude='figures/' \
		--exclude='notes/' \
		--exclude='web/' \
		--exclude='.pytest_cache/' \
		./ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/
endif
	@echo "==> Upload complete."

remote-pull:
ifdef FILE
	@echo "==> Downloading $(FILE) from $(REMOTE_HOST)..."
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/$(FILE) ./$(FILE)
else
	@echo "==> Downloading results from $(REMOTE_HOST)..."
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/data/ ./data/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/models/ ./models/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/tuned_params.json ./artifacts/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/tuned_params.original.json ./artifacts/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/ipip_bffm_norms.json ./artifacts/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/ipip_bffm_norms.meta.json ./artifacts/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/variants/ ./artifacts/variants/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/output/ ./output/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/artifacts/research_summary.json ./artifacts/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/notes/ ./notes/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/figures/ ./figures/ 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/pipeline.log ./pipeline.log 2>/dev/null || true
	$(RSYNC) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/pipeline-timing.log ./pipeline-timing.log 2>/dev/null || true
endif
	@echo "==> Download complete."

remote-setup:
	@echo "==> Installing dependencies on $(REMOTE_HOST)..."
	$(SSH) -t 'cd $(REMOTE_DIR) && make setup-python'

remote-tune:
	@echo "==> Running 'make tune' on $(REMOTE_HOST)..."
	@echo "    Logs stream live. If disconnected, run: make remote-attach"
	$(SSH) -t 'tmux kill-session -t pipeline 2>/dev/null; \
		cd $(REMOTE_DIR) && \
		tmux new-session -s pipeline \
			"make tune N_JOBS=$(REMOTE_NJOBS) PARALLEL_TRIALS=$(REMOTE_PARALLEL_TRIALS) 2>&1; \
			 echo; echo \">>> Done. Press Enter to close.\"; read"'

REMOTE_TRAIN_PARALLEL ?= 3
REMOTE_TRAIN_NJOBS = $(shell echo $$(( $(REMOTE_NJOBS) / $(REMOTE_TRAIN_PARALLEL) )) )

remote-train:
	@echo "==> Running 'make train' on $(REMOTE_HOST)..."
	@echo "    Config 1 first, then 2-4 in parallel. Domains train concurrently."
	@echo "    train-1: $(REMOTE_NJOBS) cores (all), train-2/3/4: $(REMOTE_TRAIN_NJOBS) each ($(REMOTE_NJOBS)/$(REMOTE_TRAIN_PARALLEL)), $(REMOTE_PARALLEL_DOMAINS) parallel domains"
	@echo "    If disconnected, run: make remote-attach"
	$(SSH) -t 'tmux kill-session -t pipeline 2>/dev/null; \
		cd $(REMOTE_DIR) && \
		tmux new-session -s pipeline \
			"make train N_JOBS=$(REMOTE_TRAIN_NJOBS) PARALLEL_DOMAINS=$(REMOTE_PARALLEL_DOMAINS) TRAIN_PARALLEL=$(REMOTE_TRAIN_PARALLEL) 2>&1; \
			 echo; echo \">>> Done. Press Enter to close.\"; read"'

remote-research-eval:
	@echo "==> Running 'make research-eval' on $(REMOTE_HOST)..."
	@echo "    Logs stream live. If disconnected, run: make remote-attach"
	$(SSH) -t 'tmux kill-session -t pipeline 2>/dev/null; \
		cd $(REMOTE_DIR) && \
		tmux new-session -s pipeline \
			"make research-eval 2>&1; \
			 echo; echo \">>> Done. Press Enter to close.\"; read"'

remote-attach:
	@echo "==> Attaching to tmux session on $(REMOTE_HOST)..."
	$(SSH) -t 'tmux attach -t pipeline'

remote-status:
	@echo "==> Instance status:"
	@$(SSH) 'uptime && echo && ps aux | head -20'

remote-all: remote-push remote-setup
	@echo "==> Starting full pipeline on $(REMOTE_HOST) (detached tmux)..."
	$(SSH) 'rm -f $(REMOTE_DIR)/.pipeline-exit-code && \
		tmux kill-session -t pipeline 2>/dev/null || true && \
		tmux new-session -d -s pipeline \
			"cd $(REMOTE_DIR) && \
			 TUNE_N_JOBS=$(REMOTE_NJOBS) \
			 TRAIN_N_JOBS=$(REMOTE_TRAIN_NJOBS) \
			 PARALLEL_TRIALS=$(REMOTE_PARALLEL_TRIALS) \
			 PARALLEL_DOMAINS=$(REMOTE_PARALLEL_DOMAINS) \
			 TRAIN_PARALLEL=$(REMOTE_TRAIN_PARALLEL) \
			 bash scripts/run-pipeline.sh; echo \$$? > .pipeline-exit-code"'
	@echo "==> Attaching to live pipeline output (Ctrl+B D to detach)..."
	@sleep 2
	-$(SSH) -t 'tmux attach -t pipeline'
	@echo "==> Polling for completion every 60s (timeout: $(REMOTE_POLL_MAX) min)..."
	@POLL_N=0; \
	while true; do \
		sleep 60; \
		POLL_N=$$((POLL_N + 1)); \
		if $(SSH) 'test -f $(REMOTE_DIR)/.pipeline-exit-code' 2>/dev/null; then \
			break; \
		fi; \
		if [ "$$POLL_N" -ge "$(REMOTE_POLL_MAX)" ]; then \
			echo "==> ERROR: Polling timed out after $(REMOTE_POLL_MAX) minutes."; \
			echo "    Instance still running — check: make infra-ssh"; \
			exit 1; \
		fi; \
	done
	@EXIT_CODE=$$($(SSH) 'cat $(REMOTE_DIR)/.pipeline-exit-code'); \
	if [ "$$EXIT_CODE" != "0" ]; then \
		echo "==> ERROR: Pipeline failed (exit code $$EXIT_CODE)."; \
		echo "    Check: make infra-ssh, then: cat $(REMOTE_DIR)/pipeline.log"; \
		exit 1; \
	fi
	@echo "==> Pipeline succeeded. Pulling results..."
	$(MAKE) remote-pull
	@echo "==> Tearing down infrastructure..."
	$(MAKE) infra-down
	@echo "==> Done. Infrastructure destroyed."

remote-all-1: remote-push remote-setup
	@echo "==> Starting phase 1 (download through tune) on $(REMOTE_HOST)..."
	$(SSH) 'rm -f $(REMOTE_DIR)/.pipeline-exit-code && \
		tmux kill-session -t pipeline 2>/dev/null || true && \
		tmux new-session -d -s pipeline \
			"cd $(REMOTE_DIR) && \
			 TUNE_N_JOBS=$(REMOTE_NJOBS) \
			 TRAIN_N_JOBS=$(REMOTE_TRAIN_NJOBS) \
			 PARALLEL_TRIALS=$(REMOTE_PARALLEL_TRIALS) \
			 PARALLEL_DOMAINS=$(REMOTE_PARALLEL_DOMAINS) \
			 TRAIN_PARALLEL=$(REMOTE_TRAIN_PARALLEL) \
			 bash scripts/run-pipeline.sh --end-stage tune; echo \$$? > .pipeline-exit-code"'
	@echo "==> Attaching to live pipeline output (Ctrl+B D to detach)..."
	@sleep 2
	-$(SSH) -t 'tmux attach -t pipeline'
	@echo "==> Polling for completion every 60s (timeout: $(REMOTE_POLL_MAX) min)..."
	@POLL_N=0; \
	while true; do \
		sleep 60; \
		POLL_N=$$((POLL_N + 1)); \
		if $(SSH) 'test -f $(REMOTE_DIR)/.pipeline-exit-code' 2>/dev/null; then \
			break; \
		fi; \
		if [ "$$POLL_N" -ge "$(REMOTE_POLL_MAX)" ]; then \
			echo "==> ERROR: Polling timed out after $(REMOTE_POLL_MAX) minutes."; \
			echo "    Instance still running — check: make infra-ssh"; \
			exit 1; \
		fi; \
	done
	@EXIT_CODE=$$($(SSH) 'cat $(REMOTE_DIR)/.pipeline-exit-code'); \
	if [ "$$EXIT_CODE" != "0" ]; then \
		echo "==> ERROR: Phase 1 failed (exit code $$EXIT_CODE)."; \
		echo "    Check: make infra-ssh, then: cat $(REMOTE_DIR)/pipeline.log"; \
		exit 1; \
	fi
	@echo "==> Phase 1 complete. Pulling results..."
	$(MAKE) remote-pull
	@echo "==> Tearing down infrastructure..."
	$(MAKE) infra-down
	@echo "==> Phase 1 done. Infrastructure destroyed."
	@echo "    To continue: make infra-up && make remote-all-2"

remote-all-2:
	@echo "==> Starting phase 2 (train through figures) on $(REMOTE_HOST)..."
	$(SSH) 'rm -f $(REMOTE_DIR)/.pipeline-exit-code && \
		tmux kill-session -t pipeline 2>/dev/null || true && \
		tmux new-session -d -s pipeline \
			"cd $(REMOTE_DIR) && \
			 TUNE_N_JOBS=$(REMOTE_NJOBS) \
			 TRAIN_N_JOBS=$(REMOTE_TRAIN_NJOBS) \
			 PARALLEL_TRIALS=$(REMOTE_PARALLEL_TRIALS) \
			 PARALLEL_DOMAINS=$(REMOTE_PARALLEL_DOMAINS) \
			 TRAIN_PARALLEL=$(REMOTE_TRAIN_PARALLEL) \
			 bash scripts/run-pipeline.sh --start-stage train; echo \$$? > .pipeline-exit-code"'
	@echo "==> Attaching to live pipeline output (Ctrl+B D to detach)..."
	@sleep 2
	-$(SSH) -t 'tmux attach -t pipeline'
	@echo "==> Polling for completion every 60s (timeout: $(REMOTE_POLL_MAX) min)..."
	@POLL_N=0; \
	while true; do \
		sleep 60; \
		POLL_N=$$((POLL_N + 1)); \
		if $(SSH) 'test -f $(REMOTE_DIR)/.pipeline-exit-code' 2>/dev/null; then \
			break; \
		fi; \
		if [ "$$POLL_N" -ge "$(REMOTE_POLL_MAX)" ]; then \
			echo "==> ERROR: Polling timed out after $(REMOTE_POLL_MAX) minutes."; \
			echo "    Instance still running — check: make infra-ssh"; \
			exit 1; \
		fi; \
	done
	@EXIT_CODE=$$($(SSH) 'cat $(REMOTE_DIR)/.pipeline-exit-code'); \
	if [ "$$EXIT_CODE" != "0" ]; then \
		echo "==> ERROR: Phase 2 failed (exit code $$EXIT_CODE)."; \
		echo "    Check: make infra-ssh, then: cat $(REMOTE_DIR)/pipeline.log"; \
		exit 1; \
	fi
	@echo "==> Phase 2 succeeded. Pulling results..."
	$(MAKE) remote-pull
	@echo "==> Tearing down infrastructure..."
	$(MAKE) infra-down
	@echo "==> Done. Infrastructure destroyed."

# ---------------------------------------------------------------------------
# Two-phase GPU/CPU workflow
# ---------------------------------------------------------------------------

remote-1-gpu:
	@echo "==> Phase 1 (GPU): Pushing source + artifacts to GPU instance..."
	@printf '%s\n' '$(shell git rev-parse HEAD 2>/dev/null || echo unknown)' > .git-hash
	$(GPU_RSYNC) \
		--exclude='.venv/' \
		--exclude='__pycache__/' \
		--exclude='node_modules/' \
		--exclude='.git/' \
		--exclude='*.pyc' \
		--exclude='.terraform/' \
		--exclude='*.tfstate' \
		--exclude='*.tfstate.backup' \
		--exclude='.terraform.lock.hcl' \
		--exclude='terraform.tfvars' \
		--exclude='.env' \
		--exclude='.DS_Store' \
		--exclude='.claude/settings.local.json' \
		--exclude='data/' \
		--exclude='artifacts/variants/' \
		--exclude='artifacts/ipip_bffm_norms.meta.json' \
		--exclude='artifacts/research_summary.json' \
		--exclude='output/' \
		--exclude='figures/' \
		--exclude='notes/' \
		--exclude='web/' \
		--exclude='.pytest_cache/' \
		./ $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/
	@echo "==> Setting up venv on GPU instance..."
	$(GPU_SSH) -t 'cd $(GPU_DIR) && make setup-python'
	@echo "==> Starting pipeline (download through train, GPU) in tmux..."
	$(GPU_SSH) 'rm -f $(GPU_DIR)/.pipeline-exit-code && \
		tmux kill-session -t pipeline 2>/dev/null || true && \
		tmux new-session -d -s pipeline \
			"cd $(GPU_DIR) && \
			 GPU=1 \
			 PARALLEL_TRIALS=1 \
			 PARALLEL_DOMAINS=1 \
			 TRAIN_PARALLEL=1 \
			 bash scripts/run-pipeline.sh --end-stage train --gpu; echo \$$? > .pipeline-exit-code"'
	@echo "==> Attaching to live pipeline output (Ctrl+B D to detach)..."
	@sleep 2
	-$(GPU_SSH) -t 'tmux attach -t pipeline'
	@echo "==> Polling for completion every 60s (timeout: $(REMOTE_POLL_MAX) min)..."
	@POLL_N=0; \
	while true; do \
		sleep 60; \
		POLL_N=$$((POLL_N + 1)); \
		if $(GPU_SSH) 'test -f $(GPU_DIR)/.pipeline-exit-code' 2>/dev/null; then \
			break; \
		fi; \
		if [ "$$POLL_N" -ge "$(REMOTE_POLL_MAX)" ]; then \
			echo "==> ERROR: Polling timed out after $(REMOTE_POLL_MAX) minutes."; \
			echo "    Instance still running — check: make infra-gpu-ssh"; \
			exit 1; \
		fi; \
	done
	@EXIT_CODE=$$($(GPU_SSH) 'cat $(GPU_DIR)/.pipeline-exit-code'); \
	if [ "$$EXIT_CODE" != "0" ]; then \
		echo "==> ERROR: GPU pipeline failed (exit code $$EXIT_CODE)."; \
		echo "    Instance left running for debugging: make infra-gpu-ssh"; \
		exit 1; \
	fi
	@echo "==> GPU pipeline succeeded. Pulling models and artifacts..."
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/models/ ./models/
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/artifacts/tuned_params.json ./artifacts/ 2>/dev/null || true
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/artifacts/tuned_params.original.json ./artifacts/ 2>/dev/null || true
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/artifacts/ipip_bffm_norms.json ./artifacts/ 2>/dev/null || true
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/pipeline.log ./pipeline-gpu.log 2>/dev/null || true
	$(GPU_RSYNC) $(GPU_USER)@$(GPU_HOST):$(GPU_DIR)/pipeline-timing.log ./pipeline-timing-gpu.log 2>/dev/null || true
	@echo "==> Tearing down GPU infrastructure..."
	$(MAKE) infra-gpu-down
	@echo "==> Phase 1 (GPU) complete. Models and artifacts pulled locally."
	@echo "    Continue with: make infra-cpu-up && make remote-2-cpu"

remote-2-cpu: remote-push remote-setup
	@echo "==> Phase 2 (CPU): Starting eval + export pipeline..."
	$(SSH) 'rm -f $(REMOTE_DIR)/.pipeline-exit-code && \
		tmux kill-session -t pipeline 2>/dev/null || true && \
		tmux new-session -d -s pipeline \
			"cd $(REMOTE_DIR) && \
			 . .venv/bin/activate && \
			 make download load norms norms-check prepare correlations && \
			 make research-eval export-all notes figures; \
			 echo \$$? > .pipeline-exit-code"'
	@echo "==> Attaching to live pipeline output (Ctrl+B D to detach)..."
	@sleep 2
	-$(SSH) -t 'tmux attach -t pipeline'
	@echo "==> Polling for completion every 60s (timeout: $(REMOTE_POLL_MAX) min)..."
	@POLL_N=0; \
	while true; do \
		sleep 60; \
		POLL_N=$$((POLL_N + 1)); \
		if $(SSH) 'test -f $(REMOTE_DIR)/.pipeline-exit-code' 2>/dev/null; then \
			break; \
		fi; \
		if [ "$$POLL_N" -ge "$(REMOTE_POLL_MAX)" ]; then \
			echo "==> ERROR: Polling timed out after $(REMOTE_POLL_MAX) minutes."; \
			echo "    Instance still running — check: make infra-cpu-ssh"; \
			exit 1; \
		fi; \
	done
	@EXIT_CODE=$$($(SSH) 'cat $(REMOTE_DIR)/.pipeline-exit-code'); \
	if [ "$$EXIT_CODE" != "0" ]; then \
		echo "==> ERROR: CPU pipeline failed (exit code $$EXIT_CODE)."; \
		echo "    Instance left running for debugging: make infra-cpu-ssh"; \
		exit 1; \
	fi
	@echo "==> CPU pipeline succeeded. Pulling results..."
	$(MAKE) remote-pull
	@echo "==> Tearing down CPU infrastructure..."
	$(MAKE) infra-cpu-down
	@echo "==> Phase 2 (CPU) complete. All results pulled locally."
