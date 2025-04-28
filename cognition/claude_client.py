'''
cognition/claude_client.py

A robust client for interacting with Claude 3 models via AWS Bedrock Runtime.
Includes error handling, input validation, caching, sync/async support, and dependency injection.
'''

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Any

import boto3
import aioboto3
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ClaudeConfig:
    model_id: str
    region_name: str = "us-east-1"
    max_tokens: int = 512
    temperature: float = 0.5
    api_version: str = "bedrock-2023-05-31"


class ClaudeError(Exception):
    """Base exception for Claude client errors."""


class ClaudeValidationError(ClaudeError):
    """Invalid input parameters."""


class ClaudeInvocationError(ClaudeError):
    """Error invoking the Claude model."""


class ClaudeParseError(ClaudeError):
    """Error parsing the Claude response."""


class ClaudeClient:
    """
    Client wrapper for Anthropic Claude via AWS Bedrock Runtime.
    Supports sync and async invocation, parameter validation, and
    dependency injection for testing.
    """

    MAX_ALLOWED_TOKENS = 4096
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 1.0

    def __init__(
        self,
        config: ClaudeConfig,
        client: Any = None,
        async_session: Any = None,
    ):
        """
        Args:
            config (ClaudeConfig): Configuration object.
            client: Optional boto3 Bedrock client for injection/testing.
            async_session: Optional aioboto3 Session for async.
        """
        if not config.model_id or not isinstance(config.model_id, str):
            raise ClaudeValidationError("model_id must be a non-empty string")

        self.config = config
        self.model_id = config.model_id
        self._client = client or boto3.client(
            "bedrock-runtime", region_name=config.region_name
        )
        self._async_session = async_session or aioboto3.Session()

    def _validate_parameters(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float,
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise ClaudeValidationError("messages must be a non-empty list")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ClaudeValidationError(
                    "Each message must be a dict with 'role' and 'content'"
                )
            if msg["role"] not in ("user", "assistant", "system"):
                raise ClaudeValidationError(
                    f"Unsupported role '{msg['role']}'"
                )
        if (
            not isinstance(max_tokens, int)
            or max_tokens <= 0
            or max_tokens > self.MAX_ALLOWED_TOKENS
        ):
            raise ClaudeValidationError(
                f"max_tokens must be a positive int <= {self.MAX_ALLOWED_TOKENS}"
            )
        if not (self.MIN_TEMPERATURE <= temperature <= self.MAX_TEMPERATURE):
            raise ClaudeValidationError(
                f"temperature must be between {self.MIN_TEMPERATURE} and {self.MAX_TEMPERATURE}"
            )

    def _make_payload(
        self,
        messages: List[dict],
        *,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        payload: dict = {
            "anthropic_version": self.config.api_version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        return payload

    def _parse_response(self, response: Any) -> str:
        try:
            body = json.loads(response["body"].read())
            return body["content"][0]["text"]
        except (KeyError, json.JSONDecodeError) as e:
            raise ClaudeParseError("Failed to parse response") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def invoke(
        self,
        messages: List[dict],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Synchronously invoke Claude with given messages.

        Args:
            messages: List of dict messages with 'role' and 'content'.
            system: Optional system prompt.
            max_tokens: Overrides default from config.
            temperature: Overrides default from config.
        Returns:
            Generated text from Claude.
        """
        max_toks = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        self._validate_parameters(messages, max_toks, temp)
        payload = self._make_payload(
            messages, system=system, max_tokens=max_toks, temperature=temp
        )

        try:
            response = self._client.invoke_model(
                ModelId=self.model_id,
                Body=json.dumps(payload),
                ContentType="application/json",
                Accept="application/json",
            )
            return self._parse_response(response)
        except (BotoCoreError, ClientError) as e:
            raise ClaudeInvocationError("AWS invocation failed") from e

    async def invoke_async(
        self,
        messages: List[dict],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Asynchronously invoke Claude with given messages.

        Args:
            messages: List of dict messages with 'role' and 'content'.
            system: Optional system prompt.
            max_tokens: Overrides default from config.
            temperature: Overrides default from config.
        Returns:
            Generated text from Claude.
        """
        max_toks = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        self._validate_parameters(messages, max_toks, temp)
        payload = self._make_payload(
            messages, system=system, max_tokens=max_toks, temperature=temp
        )

        async with self._async_session.client(
            "bedrock-runtime", region_name=self.config.region_name
        ) as client:
            try:
                response = await client.invoke_model(
                    ModelId=self.model_id,
                    Body=json.dumps(payload),
                    ContentType="application/json",
                    Accept="application/json",
                )
                return self._parse_response(response)
            except Exception as e:
                raise ClaudeInvocationError("Async invocation failed") from e


if __name__ == "__main__":
    # Example usage
    cfg = ClaudeConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    client = ClaudeClient(config=cfg)
    msgs = [{"role": "user", "content": "Hello, Claude!"}]
    try:
        print(client.invoke(msgs))
    except ClaudeError as err:
        logger.error("Error invoking Claude: %s", err)
