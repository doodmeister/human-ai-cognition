# Deployment region
variable "aws_region" {
  description = "AWS region where resources will be deployed"
  type        = string
  default     = "us-east-1"
}

# S3 Bucket for input and cognitive memory data
variable "s3_bucket_name" {
  description = "Name of the S3 bucket for uploading input files and logs"
  type        = string
  default     = "cognition-inputs"
}

# NASA API Key (used for the video stream Lambda)
variable "nasa_api_key" {
  description = "NASA API key for vision enrichment"
  type        = string
  sensitive   = true
}

# OpenSearch cluster endpoint (without https://)
variable "opensearch_domain" {
  description = "Domain endpoint for the OpenSearch instance (no https://)"
  type        = string
  default     = "humanai-cluster"
}

# Embedding vector dimension used across STM/LTM
variable "vector_dim" {
  description = "Embedding vector dimension (match your embedding model)"
  type        = number
  default     = 768
}

# Lambda memory size in MB
variable "lambda_memory_size" {
  description = "Default memory size for Lambda functions"
  type        = number
  default     = 512
}

# Lambda timeout in seconds
variable "lambda_timeout" {
  description = "Default timeout duration for Lambda functions"
  type        = number
  default     = 60
}
