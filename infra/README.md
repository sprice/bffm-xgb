# AWS Instance Infrastructure

Two Terraform configurations for running the IPIP-BFFM XGBoost pipeline on AWS.
Both use spot by default, and both can be switched to on-demand with
`TF_VAR_use_spot=false`.

| Directory | Instance | AMI | User | Purpose |
|-----------|----------|-----|------|---------|
| `cpu/` | `c7a.24xlarge` (default) | Amazon Linux 2023 | `ec2-user` | Full pipeline or eval-only |
| `gpu/` | `g5.xlarge` | AWS Deep Learning (Ubuntu) | `ubuntu` | Tune + train (GPU-accelerated) |

## Prerequisites

1. **AWS CLI** configured (`aws configure`)
2. **Terraform** >= 1.0 installed
3. **SSH key pair** registered in AWS (the key pair name, not the file)

## Setup

```bash
# CPU instance
cd infra/cpu
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set key_name and allowed_ssh_cidr at minimum

# GPU instance
cd infra/gpu
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set key_name and allowed_ssh_cidr at minimum
```

## Usage

Run everything from the project root via Make. See [docs/infrastructure.md](../docs/infrastructure.md) for the full guide.

If you want a different 96-vCPU Intel host such as `c7i.24xlarge`, override
`instance_type` in `infra/cpu/terraform.tfvars` before provisioning.

To force on-demand capacity for a run:

```bash
TF_VAR_use_spot=false make infra-cpu-up
TF_VAR_use_spot=false make infra-gpu-up
```

To set the default remote Make core budget for an instance, use
`TF_VAR_remote_njobs=<n>` or set `remote_njobs` in the corresponding
`terraform.tfvars`. The Makefile reads that Terraform output and uses it as the
default `REMOTE_NJOBS`.

```bash
# Option A: Full pipeline on CPU
make infra-cpu-up
make remote-all
make remote-reference   # same remote CPU path, reference variant only

# Option B: GPU tune/train + CPU eval (two-phase)
make infra-gpu-up
make remote-1-gpu       # tune+train on GPU, pulls models, tears down GPU

make infra-cpu-up
make remote-2-cpu       # eval+export on CPU, pulls results, tears down CPU
```

Remote pulls now include `logs/` alongside models, artifacts, notes, and figures,
so per-task train/eval logs from parallel runs are available locally after the job finishes.
`make remote-all` also checkpoint-pulls after major stages on the CPU path so
expensive artifacts are synced locally before the final teardown.
`make remote-reference` uses the same checkpoint-pull flow but only builds the
default reference data/model path, skips notes and the three ablation variants,
and fails closed if stale non-reference generated outputs are still present
locally. Use `FORCE=1` only if you want those stale generated outputs removed
before the run starts.
`make remote-push` excludes `data/` by design; use `make remote-push-data`
when you intentionally want to upload your local `data/` tree to resume work
on a fresh instance.

## Troubleshooting

See [docs/infrastructure.md](../docs/infrastructure.md) for detailed troubleshooting.
