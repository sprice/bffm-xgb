output "instance_ip" {
  description = "Public IP of the spot instance"
  value       = aws_spot_instance_request.pipeline.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_spot_instance_request.pipeline.spot_instance_id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.ssh_private_key_path} ec2-user@${aws_spot_instance_request.pipeline.public_ip}"
}
