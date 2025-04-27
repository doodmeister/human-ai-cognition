# cognition/claude_client.py

"""
Claude Client.

Handles invoking Claude 3 models via AWS Bedrock Runtime.
"""

import boto3
import json

class ClaudeClient:
    def __init__(self, model_id: str, region_name: str = "us-east-1"):
        """
        Initialize the Claude client.

        Args:
            model_id (str): Claude model ID from Bedrock.
            region_name (str): AWS region.
        """
        self.model_id = model_id
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)

    def invoke(self, prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
        """
        Send a prompt to Claude and retrieve the response.

        Args:
            prompt (str): Full prompt string.
            max_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Assistant's reply text.
        """
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
