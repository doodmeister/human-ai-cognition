provider "aws" {
  region = var.aws_region
}

# -----------------------------
# S3 BUCKET
# -----------------------------
resource "aws_s3_bucket" "input_bucket" {
  bucket = var.s3_bucket_name
}

# -----------------------------
# IAM ROLE FOR LAMBDA
# -----------------------------
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_inline_policy" {
  name = "lambda_cognition_policy"
  role = aws_iam_role.lambda_exec_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = "${aws_s3_bucket.input_bucket.arn}/*"
      },
      {
        Effect = "Allow",
        Action = [
          "es:ESHttpPost",
          "es:ESHttpPut",
          "es:ESHttpGet",
          "es:ESHttpDelete"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "*"
      }
    ]
  })
}

# -----------------------------
# LAMBDA FUNCTIONS
# -----------------------------

resource "aws_lambda_function" "video_stream" {
  function_name = "video_stream_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "video_stream.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/video_stream.zip"
  source_code_hash = filebase64sha256("./build/video_stream.zip")
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size
  environment {
    variables = {
      S3_BUCKET    = var.s3_bucket_name
      NASA_API_KEY = var.nasa_api_key
    }
  }
}

resource "aws_lambda_function" "sensor_stream" {
  function_name = "sensor_stream_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "sensor_stream.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/sensor_stream.zip"
  source_code_hash = filebase64sha256("./build/sensor_stream.zip")
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size
  environment {
    variables = {
      S3_BUCKET = var.s3_bucket_name
    }
  }
}

resource "aws_lambda_function" "text_stream" {
  function_name = "text_stream_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "text_stream.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/text_stream.zip"
  source_code_hash = filebase64sha256("./build/text_stream.zip")
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size
  environment {
    variables = {
      S3_BUCKET       = var.s3_bucket_name
      OPENSEARCH_HOST = var.opensearch_domain
      STM_INDEX       = "humanai-stm"
      VECTOR_DIM      = tostring(var.vector_dim)
      AWS_REGION      = var.aws_region
      CLAUDE_MODEL_ID = var.claude_model_id
      EMBEDDER_MODEL  = var.embedder_model
    }
  }
}

resource "aws_lambda_function" "dream_consolidator" {
  function_name = "dream_consolidator_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "dream_consolidator.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/dream_consolidator.zip"
  source_code_hash = filebase64sha256("./build/dream_consolidator.zip")
  timeout       = 90
  memory_size   = var.lambda_memory_size
  environment {
    variables = {
      OPENSEARCH_HOST = var.opensearch_domain
      STM_INDEX       = "humanai-stm"
      LTM_INDEX       = "humanai-ltm"
      VECTOR_DIM      = tostring(var.vector_dim)
    }
  }
}

resource "aws_lambda_function" "cognitive_agent" {
  function_name = "cognitive_agent_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/cognitive_agent.zip"
  source_code_hash = filebase64sha256("./build/cognitive_agent.zip")
  timeout       = 180
  memory_size   = 1024
  environment {
    variables = {
      S3_BUCKET       = var.s3_bucket_name
      OPENSEARCH_HOST = var.opensearch_domain
      STM_INDEX       = "humanai-stm"
      LTM_INDEX       = "humanai-ltm"
      VECTOR_DIM      = tostring(var.vector_dim)
      AWS_REGION      = var.aws_region
      CLAUDE_MODEL_ID = var.claude_model_id
      EMBEDDER_MODEL  = var.embedder_model
    }
  }
}

resource "aws_lambda_function" "dpad_trainer" {
  function_name = "dpad_trainer_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "train_handler.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/dpad_trainer.zip"
  source_code_hash = filebase64sha256("./build/dpad_trainer.zip")
  timeout       = 900
  memory_size   = 1024
  environment {
    variables = {
      S3_BUCKET       = var.s3_bucket_name
      OPENSEARCH_HOST = var.opensearch_domain
      STM_INDEX       = "humanai-stm"
      VECTOR_DIM      = tostring(var.vector_dim)
      AWS_REGION      = var.aws_region
      CLAUDE_MODEL_ID = var.claude_model_id
      EMBEDDER_MODEL  = var.embedder_model
    }
  }
}

# -----------------------------
# CLOUDWATCH LOG GROUPS
# -----------------------------
resource "aws_cloudwatch_log_group" "video_log" {
  name              = "/aws/lambda/${aws_lambda_function.video_stream.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "sensor_log" {
  name              = "/aws/lambda/${aws_lambda_function.sensor_stream.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "text_log" {
  name              = "/aws/lambda/${aws_lambda_function.text_stream.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "dream_log" {
  name              = "/aws/lambda/${aws_lambda_function.dream_consolidator.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "cognitive_log" {
  name              = "/aws/lambda/${aws_lambda_function.cognitive_agent.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "trainer_log" {
  name              = "/aws/lambda/${aws_lambda_function.dpad_trainer.function_name}"
  retention_in_days = var.log_retention_in_days
}

# -----------------------------
# CLOUDWATCH RULES
# -----------------------------
resource "aws_cloudwatch_event_rule" "training_schedule" {
  name                = "dpadrnn_training_schedule"
  schedule_expression = var.training_schedule_expression
}

resource "aws_cloudwatch_event_target" "training_trigger" {
  rule      = aws_cloudwatch_event_rule.training_schedule.name
  target_id = "trainingLambdaTrigger"
  arn       = aws_lambda_function.dpad_trainer.arn
}

resource "aws_lambda_permission" "training_perm" {
  statement_id  = "AllowCloudWatchTraining"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dpad_trainer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.training_schedule.arn
}

resource "aws_cloudwatch_event_rule" "video_schedule" {
  name                = "video_stream_schedule"
  schedule_expression = "rate(1 minute)"
}

resource "aws_cloudwatch_event_rule" "sensor_schedule" {
  name                = "sensor_stream_schedule"
  schedule_expression = "rate(10 minutes)"
}

resource "aws_cloudwatch_event_rule" "dream_schedule" {
  name                = "dream_state_trigger"
  schedule_expression = "rate(30 minutes)"
}

# -----------------------------
# EVENT TARGETS
# -----------------------------
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

resource "aws_cloudwatch_event_target" "dream_trigger" {
  rule      = aws_cloudwatch_event_rule.dream_schedule.name
  target_id = "dreamLambdaTrigger"
  arn       = aws_lambda_function.dream_consolidator.arn
}

# -----------------------------
# LAMBDA PERMISSIONS
# -----------------------------
resource "aws_lambda_permission" "video_perm" {
  statement_id  = "AllowCloudWatchVideo"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.video_stream.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.video_schedule.arn
}

resource "aws_lambda_permission" "sensor_perm" {
  statement_id  = "AllowCloudWatchSensor"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sensor_stream.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sensor_schedule.arn
}

resource "aws_lambda_permission" "dream_perm" {
  statement_id  = "AllowCloudWatchDream"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dream_consolidator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dream_schedule.arn
}
