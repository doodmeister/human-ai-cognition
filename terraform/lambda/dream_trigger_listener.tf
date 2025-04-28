resource "aws_lambda_function" "dream_trigger_listener" {
  function_name = "dream_trigger_listener_lambda"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "dream_trigger_listener.lambda_handler"
  runtime       = "python3.9"
  filename      = "./build/dream_trigger_listener.zip"
  source_code_hash = filebase64sha256("./build/dream_trigger_listener.zip")
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory_size

  environment {
    variables = {
      AWS_REGION       = var.aws_region
      OPENSEARCH_HOST  = var.opensearch_domain
      META_INDEX       = var.meta_index
      DREAM_LAMBDA_URL = var.dream_lambda_url
    }
  }
}

resource "aws_cloudwatch_log_group" "dream_trigger_log" {
  name              = "/aws/lambda/${aws_lambda_function.dream_trigger_listener.function_name}"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_event_rule" "dream_trigger_schedule" {
  name                = "dream_trigger_listener_schedule"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "dream_trigger_event_target" {
  rule      = aws_cloudwatch_event_rule.dream_trigger_schedule.name
  target_id = "dreamTriggerLambda"
  arn       = aws_lambda_function.dream_trigger_listener.arn
}

resource "aws_lambda_permission" "dream_trigger_permission" {
  statement_id  = "AllowCloudWatchDreamTrigger"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dream_trigger_listener.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dream_trigger_schedule.arn
}
