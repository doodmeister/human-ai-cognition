
provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "document_store" {
  bucket = var.s3_bucket_name
}

resource "aws_s3_bucket_notification" "trigger_lambda" {
  bucket = aws_s3_bucket.document_store.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest_processor.arn
    events              = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_lambda_permission.allow_s3]
}

resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_exec_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Principal = {
        Service = "lambda.amazonaws.com"
      },
      Effect = "Allow",
      Sid    = ""
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "ingest_processor" {
  filename         = "lambda/ingest_processor.zip"
  function_name    = "ingest_processor"
  role             = aws_iam_role.lambda_exec_role.arn
  handler          = "ingest_processor.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = filebase64sha256("lambda/ingest_processor.zip")
}

resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest_processor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.document_store.arn
}

resource "aws_opensearch_domain" "vector_store" {
  domain_name = var.opensearch_domain_name

  cluster_config {
    instance_type = "t3.small.search"
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 10
  }

  engine_version = "OpenSearch_2.3"
}

output "s3_bucket_name" {
  value = aws_s3_bucket.document_store.bucket
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.vector_store.endpoint
}
