provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "us-east-1"
}

variable "s3_bucket_name" {
  default = "cognition-inputs"
}

variable "nasa_api_key" {
  default = "your-nasa-api-key"  # Set this securely in production
}

resource "aws_s3_bucket" "input_bucket" {
  bucket = var.s3_bucket_name
}

resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_execution_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# -----------------------------
# Lambda: Video Stream
# -----------------------------
resource "aws_lambda_function" "video_stream" {
  function_name = "video_stream_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "video_stream.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/video_stream.zip"
  source_code_hash = filebase64sha256("./build/video_stream.zip")
  timeout       = 60
  environment {
    variables = {
      S3_BUCKET    = var.s3_bucket_name
      NASA_API_KEY = var.nasa_api_key
    }
  }
}

# -----------------------------
# Lambda: Sensor Stream
# -----------------------------
resource "aws_lambda_function" "sensor_stream" {
  function_name = "sensor_stream_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "sensor_stream.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/sensor_stream.zip"
  source_code_hash = filebase64sha256("./build/sensor_stream.zip")
  timeout       = 60
  environment {
    variables = {
      S3_BUCKET = var.s3_bucket_name
    }
  }
}

# -----------------------------
# CloudWatch Scheduled Triggers
# -----------------------------
resource "aws_cloudwatch_event_rule" "video_schedule" {
  name                = "video_stream_schedule"
  schedule_expression = "rate(1 minute)"
}

resource "aws_cloudwatch_event_rule" "sensor_schedule" {
  name                = "sensor_stream_schedule"
  schedule_expression = "rate(10 minutes)"
}

resource "aws_cloudwatch_event_target" "video_trigger" {
  rule      = aws_cloudwatch_event_rule.video_schedule.name
  target_id = "videoLambdaTrigger"
  arn       = aws_lambda_function.video_stream.arn
}

resource "aws_cloudwatch_event_target" "sensor_trigger" {
  rule      = aws_cloudwatch_event_rule.sensor_schedule.name
  target_id = "sensorLambdaTrigger"
  arn       = aws_lambda_function.sensor_stream.arn
}

resource "aws_lambda_permission" "allow_cloudwatch_video" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.video_stream.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.video_schedule.arn
}

resource "aws_lambda_permission" "allow_cloudwatch_sensor" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sensor_stream.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sensor_schedule.arn
}
