output "instance_ip" {
  description = "Public IP of the provisioned instance"
  value       = local.instance_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = local.instance_id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.ssh_private_key_path} ubuntu@${local.instance_ip}"
}

output "remote_njobs" {
  description = "Default remote N_JOBS budget for this GPU instance"
  value       = var.remote_njobs
}
