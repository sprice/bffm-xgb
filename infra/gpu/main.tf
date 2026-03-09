terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  instance_id = one(concat(
    aws_spot_instance_request.pipeline[*].spot_instance_id,
    aws_instance.pipeline[*].id,
  ))
  instance_ip = one(concat(
    aws_spot_instance_request.pipeline[*].public_ip,
    aws_instance.pipeline[*].public_ip,
  ))
}

# --- AMI ---

data "aws_ami" "dl_ubuntu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }
}

# --- VPC ---

resource "aws_vpc" "pipeline" {
  cidr_block           = "10.1.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

resource "aws_internet_gateway" "pipeline" {
  vpc_id = aws_vpc.pipeline.id

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

data "aws_ec2_instance_type_offerings" "available_azs" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

resource "aws_subnet" "pipeline" {
  vpc_id                  = aws_vpc.pipeline.id
  cidr_block              = "10.1.1.0/24"
  availability_zone       = data.aws_ec2_instance_type_offerings.available_azs.locations[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

resource "aws_route_table" "pipeline" {
  vpc_id = aws_vpc.pipeline.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.pipeline.id
  }

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

resource "aws_route_table_association" "pipeline" {
  subnet_id      = aws_subnet.pipeline.id
  route_table_id = aws_route_table.pipeline.id
}

# --- Security Group ---

resource "aws_security_group" "pipeline" {
  name_prefix = "bffm-xgb-pipeline-gpu-"
  description = "SSH access for IPIP-BFFM pipeline"
  vpc_id      = aws_vpc.pipeline.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

# --- Compute ---

resource "aws_spot_instance_request" "pipeline" {
  count                  = var.use_spot ? 1 : 0
  ami                    = data.aws_ami.dl_ubuntu.id
  instance_type          = var.instance_type
  spot_price             = var.spot_max_price
  wait_for_fulfillment   = true
  spot_type              = "one-time"
  key_name               = var.key_name
  subnet_id              = aws_subnet.pipeline.id
  vpc_security_group_ids = [aws_security_group.pipeline.id]

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  user_data = <<EOF
#!/bin/bash
set -euo pipefail

apt-get update -y
apt-get install -y python3-venv tmux htop

mkdir -p /home/ubuntu/bffm-xgb
chown ubuntu:ubuntu /home/ubuntu/bffm-xgb

touch /home/ubuntu/.setup-done
chown ubuntu:ubuntu /home/ubuntu/.setup-done
EOF

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

resource "aws_instance" "pipeline" {
  count                       = var.use_spot ? 0 : 1
  ami                         = data.aws_ami.dl_ubuntu.id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  subnet_id                   = aws_subnet.pipeline.id
  vpc_security_group_ids      = [aws_security_group.pipeline.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  user_data = <<EOF
#!/bin/bash
set -euo pipefail

apt-get update -y
apt-get install -y python3-venv tmux htop

mkdir -p /home/ubuntu/bffm-xgb
chown ubuntu:ubuntu /home/ubuntu/bffm-xgb

touch /home/ubuntu/.setup-done
chown ubuntu:ubuntu /home/ubuntu/.setup-done
EOF

  tags = {
    Name = "bffm-xgb-pipeline-gpu"
  }
}

resource "aws_ec2_tag" "pipeline_name" {
  count       = var.use_spot ? 1 : 0
  resource_id = aws_spot_instance_request.pipeline[0].spot_instance_id
  key         = "Name"
  value       = "bffm-xgb-pipeline-gpu"
}
