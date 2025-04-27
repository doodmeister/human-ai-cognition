from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.prospective_memory import check_due_reminders
from cognition.rag_utils import build_prompt
from cognition.memory_writer import store_to_stm
import boto3
import json
from sentence_transformers import SentenceTransformer
from botocore.exceptions import BotoCoreError, ClientError
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)

# -------------------------------
# Embedder Class (optimized)
# -------------------------------
class Embedder:
    """
    A class to handle text embedding using a SentenceTransformer model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> List[float]:
        """Embed a text string into a vector."""
        return self.model.encode(text).tolist()

# -------------------------------
# Bedrock Claude 3 Client
# -------------------------------
class ClaudeClient:
    """
    A client for interacting with the Bedrock Claude 3 model.
    """
    def __init__(self, model_id: str, region_name: str = "us-east-1"):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id

    def invoke(self, prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
        """Invoke the Claude model with a prompt and return the response."""
        logging.info(f"Invoking Claude model with prompt: {prompt}")
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError) as e:
            # Log the error and return a fallback response
            print(f"Error invoking Claude model: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

# -------------------------------
# Main Cognitive Loop
# -------------------------------
class CognitiveAgent:
    """
    The main cognitive agent that handles user input, memory queries, and model interactions.
    """
    def __init__(self):
        self.embedder = Embedder(model_name=os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2"))
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.claude = ClaudeClient(
            model_id=os.getenv("CLAUDE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

    def run_chat(self, user_input: str) -> str:
        """Run the main cognitive loop for a user input."""
        query_vec = self._embed_input(user_input)
        stm_hits, ltm_hits = self._query_memories(query_vec)
        reminders = check_due_reminders()
        prompt = build_prompt(user_input, stm_hits, ltm_hits, reminders)
        reply = self.claude.invoke(prompt)
        self._store_conversation(user_input, reply)
        return reply

    def _embed_input(self, user_input: str) -> list:
        """Embed the user input into a vector."""
        return self.embedder.embed(user_input)

    def _query_memories(self, query_vec: list) -> tuple:
        """Query short-term and long-term memory."""
        stm_hits = self.stm.query(query_vec, top_k=5)
        ltm_hits = self.ltm.query(query_vec, top_k=5)
        return stm_hits, ltm_hits

    def _store_conversation(self, user_input: str, reply: str):
        """Store the conversation into short-term memory."""
        conversation_text = f"User: {user_input}\nAssistant: {reply}"
        store_to_stm(conversation_text, self.embedder.embed, self.stm)
