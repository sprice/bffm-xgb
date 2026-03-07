# Infrastructure Guide

## Overview

The pipeline can run on three types of infrastructure:

1. **Local** — Your laptop/desktop. Fine for eval and export, slow for tune/train.
2. **AWS CPU** — Spot instance (`c7a.24xlarge`, 96 cores). Good for everything but slower on tune/train.
3. **AWS GPU** — Spot instance (`g5.xlarge`, 1x A10G). Fast for tune/train, then hand off to CPU for eval.

There are two Terraform configurations under `infra/`:

| Directory    | Instance        | AMI                    | User       | Purpose              |
|-------------|-----------------|------------------------|------------|----------------------|
| `infra/cpu/` | `c7a.24xlarge`  | Amazon Linux 2023      | `ec2-user` | Eval, full pipeline  |
| `infra/gpu/` | `g5.xlarge`     | AWS Deep Learning (Ubuntu) | `ubuntu` | Tune + train         |

Each has its own VPC, subnet, security group, and Terraform state — fully independent.

---

## Option A: Full Pipeline on CPU (single instance)

Best when you want simplicity and don't mind longer tune/train times.

```bash
# 1. Spin up CPU instance
make infra-cpu-up

# 2. Run entire pipeline (push, setup, download through figures, pull, teardown)
make remote-all
```

Or in two phases with a pause to review tuned hyperparameters:

```bash
make infra-cpu-up
make remote-all-1          # download through tune, pulls tuned_params.json
# (optional) edit artifacts/tuned_params.json
make remote-all-2          # train through figures, pulls results, tears down
```

### Targets

| Target             | Steps                                                        |
|--------------------|--------------------------------------------------------------|
| `make remote-all`  | push, setup, download → figures, pull all, infra-cpu-down    |
| `make remote-all-1`| push, setup, download → tune, pull tuned_params.json         |
| `make remote-all-2`| train → figures, pull all, infra-cpu-down                    |

---

## Option B: GPU Tune/Train + CPU Eval (two instances)

Best when tune/train is the bottleneck. GPU accelerates XGBoost training significantly (5-10x), then a CPU instance handles the CPU-bound eval steps (bootstrap, simulation).

```bash
# Phase 1: GPU instance — tune + train
make infra-gpu-up
make remote-1-gpu          # push, setup, download → train (GPU), pull models+artifacts, infra-gpu-down

# Phase 2: CPU instance — eval + export
make infra-cpu-up
make remote-2-cpu          # push (with models+artifacts), setup, download → prepare, research-eval → figures, pull all, infra-cpu-down
```

### Phase 1: `make remote-1-gpu`

Runs on the GPU instance (`g5.xlarge` with Deep Learning AMI).

**Steps performed on the remote instance:**
1. `make remote-push` — upload source code + artifacts
2. `make remote-setup` — create venv, install `requirements.txt`
3. Pipeline stages (via `run-pipeline.sh --end-stage train --gpu`):
   - `download` — fetch IPIP-BFFM data
   - `load` — load into SQLite
   - `norms` — compute norms
   - `norms-check` — verify norms
   - `prepare` — prepare train/val/test splits
   - `correlations` — compute correlations
   - `tune` — Optuna hyperparameter search (GPU-accelerated)
   - `train` — train all 4 model variants (GPU-accelerated)

**After remote completion:**
- Pulls `models/`, `artifacts/tuned_params.json`, `artifacts/tuned_params.original.json`, `artifacts/ipip_bffm_norms.json`
- On successful pull: tears down GPU infrastructure (`make infra-gpu-down`)
- On failure: leaves instance running for debugging

### Phase 2: `make remote-2-cpu`

Runs on the CPU instance (`c7a.24xlarge` with Amazon Linux 2023).

**Steps performed on the remote instance:**
1. `make remote-push` — upload source code + models + artifacts (from Phase 1 pull)
2. `make remote-setup` — create venv, install `requirements.txt`
3. Data pipeline (via `make` directly):
   - `make download load norms norms-check prepare correlations`
4. Eval + export pipeline (via `make` directly):
   - `make research-eval export-all notes figures`

**After remote completion:**
- Pulls all results (artifacts/variants, output, notes, figures, logs)
- On successful pull: tears down CPU infrastructure (`make infra-cpu-down`)
- On failure: leaves instance running for debugging

---

## Infrastructure Targets

### GPU Infrastructure

| Target              | Description                                    |
|---------------------|------------------------------------------------|
| `make infra-gpu-up`  | Initialize and apply `infra/gpu/` Terraform    |
| `make infra-gpu-down`| Destroy GPU instance and networking            |
| `make infra-gpu-ssh` | SSH into the GPU instance                      |

### CPU Infrastructure

| Target              | Description                                    |
|---------------------|------------------------------------------------|
| `make infra-cpu-up`  | Initialize and apply `infra/cpu/` Terraform    |
| `make infra-cpu-down`| Destroy CPU instance and networking            |
| `make infra-cpu-ssh` | SSH into the CPU instance                      |

---

## GPU Training Details

When `GPU=1` is set (automatic in `remote-1-gpu`):

- `make tune GPU=1` — passes `--gpu` to `06_tune.py`
- `make train GPU=1` — passes `--gpu` to `07_train.py`
- XGBoost uses `device='cuda'`
- `N_JOBS` and `PARALLEL_DOMAINS` are ignored (GPU handles parallelism internally)
- Domains train sequentially (one GPU, one model at a time)

CPU training (default, unchanged):

- `make tune N_JOBS=96`
- `make train N_JOBS=96 PARALLEL_DOMAINS=5`

---

## Configuration

Each Terraform config reads from its own `terraform.tfvars` (gitignored):

**`infra/cpu/terraform.tfvars`:**
```hcl
key_name         = "my-macbook"
aws_region       = "us-east-1"
allowed_ssh_cidr = "YOUR_IP/32"
# instance_type  = "c7a.24xlarge"  # default
# spot_max_price = "2.50"          # default
```

**`infra/gpu/terraform.tfvars`:**
```hcl
key_name         = "my-macbook"
aws_region       = "us-east-1"
allowed_ssh_cidr = "YOUR_IP/32"
# instance_type  = "g5.xlarge"     # default
# spot_max_price = "1.50"          # default
```

---

## File Changes Summary

| File                      | Change                                              |
|---------------------------|-----------------------------------------------------|
| `infra/cpu/`              | Existing `infra/` moved here (unchanged config)     |
| `infra/gpu/`              | New Terraform config (DL AMI, Ubuntu, g5.xlarge)    |
| `pipeline/06_tune.py`     | Add `--gpu` flag → `device='cuda'`                  |
| `pipeline/07_train.py`    | Add `--gpu` flag → `device='cuda'`                  |
| `scripts/run-pipeline.sh` | Add `--gpu` flag, pass `GPU=1` to make tune/train   |
| `Makefile`                | New GPU/CPU infra targets, `remote-1-gpu`, `remote-2-cpu`, `GPU` variable |
| `.gitignore`              | Add `infra/gpu/` terraform patterns                 |
