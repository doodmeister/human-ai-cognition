# Deployment region
variable "aws_region" { ... }

# S3 Bucket for input and cognitive memory data
variable "s3_bucket_name" { ... }

# NASA API Key (used for the video stream Lambda)
variable "nasa_api_key" { ... }

# OpenSearch cluster endpoint (without https://)
variable "opensearch_domain" { ... }

# Embedding vector dimension used across STM/LTM
variable "vector_dim" { ... }

# Lambda memory size in MB
variable "lambda_memory_size" { ... }

# Lambda timeout in seconds
variable "lambda_timeout" { ... }

# Claude 3 model ID for AWS Bedrock
variable "claude_model_id" {
  description = "Claude model ID to use for Bedrock API interactions."
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

# Embedding model name for SentenceTransformer
variable "embedder_model" {
  description = "Model name for text embedding (e.g., all-MiniLM-L6-v2)."
  type        = string
  default     = "all-MiniLM-L6-v2"
}
