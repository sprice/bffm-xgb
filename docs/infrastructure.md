# Infrastructure Guide

## Overview

The pipeline can run on three types of infrastructure:

1. **Local** — Your laptop/desktop. Fine for eval and export, slow for tune/train.
2. **AWS CPU** — `c7a.24xlarge`/`c7i.24xlarge`-class instance (spot by default, on-demand optional). Good for everything but slower on tune/train.
3. **AWS GPU** — `g5.xlarge`-class instance (spot by default, on-demand optional). Fast for tune/train, then hand off to CPU for eval.

There are two Terraform configurations under `infra/`:

| Directory    | Instance       | AMI                        | User       | Purpose             |
| ------------ | -------------- | -------------------------- | ---------- | ------------------- |
| `infra/cpu/` | `c7a.24xlarge` | Amazon Linux 2023          | `ec2-user` | Eval, full pipeline |
| `infra/gpu/` | `g5.xlarge`    | AWS Deep Learning (Ubuntu) | `ubuntu`   | Tune + train        |

Each has its own VPC, subnet, security group, and Terraform state — fully independent.

To provision on-demand instead of spot, prefix the infra command with `TF_VAR_use_spot=false`, for example:

```bash
TF_VAR_use_spot=false make infra-cpu-up
TF_VAR_use_spot=false make infra-gpu-up
```

To set the default remote Make core budget for an instance, use `TF_VAR_remote_njobs=<n>` or set `remote_njobs` in that Terraform root's `terraform.tfvars`. The Makefile reads this output and uses it as the default `REMOTE_NJOBS`.

---

## Option A: Full Pipeline on CPU (single instance)

Best when you want simplicity and don't mind longer tune/train times.

```bash
# 1. Spin up CPU instance
make infra-cpu-up

# 2. Run entire pipeline (push, setup, download through figures, pull, teardown)
make remote-all

# Reference-only variant of the same CPU workflow
make remote-reference
```

`make remote-all` checkpoint-pulls major artifacts during the run
(`norms`, `prepare`, `correlations`, `tune`, `train`, `research-eval`, `figures`)
so expensive outputs are synced locally before final teardown.

`make remote-reference` follows the same remote CPU path but only builds the
reference data/model path after load/norms: `prepare-default`,
`correlations-default`, `train 1`, `research-eval-reference`,
`export-reference`, `export-repo-readme`, and `figures`. It skips `notes`
because notes generation still requires all four variants.

`make remote-push` excludes `data/` by design. Use `make remote-push-data`
when you intentionally want to seed the remote box with your local `data/`
tree for a resume/recovery workflow.

By default, `make remote-reference` fails closed if the local workspace still
contains stale ablation outputs, `notes/NOTES.md`, or
`artifacts/research_summary.json` from an earlier full run. Use `FORCE=1` only
if you want those generated outputs removed before the reference-only pull.

Or in two phases with a pause to review tuned hyperparameters:

```bash
make infra-cpu-up
make remote-all-1          # download through tune, pulls tuned_params.json
# (optional) edit artifacts/tuned_params.json
make remote-all-2          # train through figures, pulls results, tears down
```

### Targets

| Target                  | Steps                                                                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make remote-all`       | push, setup, download → figures, checkpoint-pull major artifacts, pull all, infra-cpu-down                                                                                                 |
| `make remote-reference` | preflight local workspace, push, setup, download → reference-only prep/correlations/train/eval/export/figures, checkpoint-pull reference artifacts, pull reference results, infra-cpu-down |
| `make remote-all-1`     | push, setup, download → tune, pull tuned_params.json                                                                                                                                       |
| `make remote-all-2`     | train → figures, pull all, infra-cpu-down                                                                                                                                                  |

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

| Target                | Description                                 |
| --------------------- | ------------------------------------------- |
| `make infra-gpu-up`   | Initialize and apply `infra/gpu/` Terraform |
| `make infra-gpu-down` | Destroy GPU instance and networking         |
| `make infra-gpu-ssh`  | SSH into the GPU instance                   |

### CPU Infrastructure

| Target                | Description                                 |
| --------------------- | ------------------------------------------- |
| `make infra-cpu-up`   | Initialize and apply `infra/cpu/` Terraform |
| `make infra-cpu-down` | Destroy CPU instance and networking         |
| `make infra-cpu-ssh`  | SSH into the CPU instance                   |

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
# use_spot       = false            # optional; default true
# remote_njobs   = 64               # optional; default REMOTE_NJOBS for this host
# instance_type  = "c7a.24xlarge"  # default
# spot_max_price = "2.50"          # default
```

**`infra/gpu/terraform.tfvars`:**
```hcl
key_name         = "my-macbook"
aws_region       = "us-east-1"
allowed_ssh_cidr = "YOUR_IP/32"
# use_spot       = false            # optional; default true
# remote_njobs   = 4                # optional; default REMOTE_NJOBS for this host
# instance_type  = "g5.xlarge"     # default
# spot_max_price = "1.50"          # default
```

---

## File Changes Summary

| File                      | Change                                                                    |
| ------------------------- | ------------------------------------------------------------------------- |
| `infra/cpu/`              | Existing `infra/` moved here (unchanged config)                           |
| `infra/gpu/`              | New Terraform config (DL AMI, Ubuntu, g5.xlarge)                          |
| `pipeline/06_tune.py`     | Add `--gpu` flag → `device='cuda'`                                        |
| `pipeline/07_train.py`    | Add `--gpu` flag → `device='cuda'`                                        |
| `scripts/run-pipeline.sh` | Add `--gpu` flag, pass `GPU=1` to make tune/train                         |
| `Makefile`                | New GPU/CPU infra targets, `remote-1-gpu`, `remote-2-cpu`, `GPU` variable |
| `.gitignore`              | Add `infra/gpu/` terraform patterns                                       |
