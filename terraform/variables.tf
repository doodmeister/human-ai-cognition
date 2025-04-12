
variable "aws_region" {
  default = "us-east-1"
}

variable "s3_bucket_name" {
  default = "humanai-document-store"
}

variable "opensearch_domain_name" {
  default = "humanai-vector-store"
}

variable "meta_llm_model_id" {
  description = "Bedrock LLM model to use for meta-cognition"
  default     = "anthropic.claude-v2"  # Example
}
