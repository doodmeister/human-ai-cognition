# -----------------------------
# CORE DEPLOYMENT VARIABLES
# -----------------------------

# Deployment region
variable "aws_region" {
  description = "AWS region where resources will be deployed."
  type        = string
  default     = "us-east-1"
}

# S3 Bucket for input and cognitive memory data
variable "s3_bucket_name" {
  description = "Name of the S3 bucket for uploading input files and logs."
  type        = string
  default     = "cognition-inputs"
}

# NASA API Key (for vision enrichment services)
variable "nasa_api_key" {
  description = "NASA API key for vision enrichment."
  type        = string
  sensitive   = true
}

# OpenSearch domain endpoint (without https://)
variable "opensearch_domain" {
  description = "Domain endpoint for the OpenSearch instance (no https://)."
  type        = string
  default     = "humanai-cluster"
}

# Embedding vector dimension
variable "vector_dim" {
  description = "Embedding vector dimension (must match your embedding model)."
  type        = number
  default     = 768
}

# Lambda memory size (in MB)
variable "lambda_memory_size" {
  description = "Default memory size for Lambda functions."
  type        = number
  default     = 512
}

# Lambda timeout (in seconds)
variable "lambda_timeout" {
  description = "Default timeout duration for Lambda functions."
  type        = number
  default     = 60
}

# -----------------------------
# NEW COGNITIVE SYSTEM VARIABLES
# -----------------------------

# Claude 3 Model ID
variable "claude_model_id" {
  description = "Claude model ID to use when calling AWS Bedrock."
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

# Embedder Model Name (for SentenceTransformer)
variable "embedder_model" {
  description = "SentenceTransformer model name used for text embeddings."
  type        = string
  default     = "all-MiniLM-L6-v2"
}

# New Variables
variable "training_schedule_expression" {
  description = "Schedule expression for triggering DPADRNN training (default: every 12 hours)"
  type        = string
  default     = "rate(12 hours)"
}

variable "log_retention_in_days" {
  description = "Number of days to retain logs in CloudWatch"
  type        = number
  default     = 14
}
# Meta-cognition index for Dream Trigger listening
variable "meta_index" {
  description = "Name of the OpenSearch index where meta-cognition events (like dream_needed) are stored."
  type        = string
  default     = "humanai-meta"
}

# URL to trigger the Dream State Processor Lambda (via API Gateway)
variable "dream_lambda_url" {
  description = "API Gateway URL to invoke the Dream State Processor Lambda."
  type        = string
}