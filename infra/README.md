# AWS Spot Instance Infrastructure

Provisions a spot instance for running the IPIP-BFFM XGBoost pipeline (~1-2 hours on a 96-vCPU instance vs ~2+ days on a laptop).

## Prerequisites

1. **AWS CLI** configured (`aws configure`)
2. **Terraform** >= 1.0 installed
3. **SSH key pair** registered in AWS (the key pair name, not the file)

## Setup

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — set key_name and allowed_ssh_cidr at minimum
```

## Usage

Run everything from the project root via Make:

```bash
# Full unattended run (push, setup, pipeline, pull, tear down)
make infra-up
make remote-all
```

Or run stages individually:

```bash
make infra-up
make remote-push
make remote-setup
make remote-tune           # 200 Optuna trials (live logs)
make remote-train          # 4 configs x 15 models (live logs)
make remote-research-eval  # validate + baselines + simulate
make remote-pull
make infra-down
```

## Reconnecting

If your SSH session drops, the pipeline keeps running in tmux:

```bash
make remote-attach    # reattach to the tmux session
```

## Troubleshooting

### SSH connection timeout
- Check that your `allowed_ssh_cidr` includes your current IP
- Verify the security group allows port 22
- Run `make remote-status` to check if the instance is up

### Spot capacity unavailable
- Try a different region or instance type in `terraform.tfvars`
- Increase `spot_max_price`
- Try `c7i.16xlarge` (64 vCPUs) as a fallback

### Instance terminated unexpectedly
- Spot instances can be reclaimed; check AWS console for termination reason
- Re-run `make infra-up` and `make remote-push` to restart
