variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c7a.24xlarge"
}

variable "remote_njobs" {
  description = "Default remote CPU core budget passed to Make targets on this instance"
  type        = number
  default     = 96
}

variable "use_spot" {
  description = "Whether to provision a spot instance request instead of an on-demand instance"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance (USD, ignored when use_spot=false)"
  type        = string
  default     = "2.50"
}

variable "key_name" {
  description = "Name of an existing AWS key pair for SSH access"
  type        = string
}

variable "ssh_private_key_path" {
  description = "Path to the private key for SSH"
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH in (restrict this to your IP, e.g. 1.2.3.4/32)"
  type        = string
}

variable "volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 50
}
