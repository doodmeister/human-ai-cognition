from datetime import datetime
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.prospective_memory import check_due_reminders
from cognition.rag_utils import build_prompt
from cognition.memory_writer import store_to_stm
import boto3
import json
from sentence_transformers import SentenceTransformer

# -------------------------------
# Embedder Class (optimized)
# -------------------------------
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> list:
        """Embed a text string into a vector."""
        return self.model.encode(text).tolist()

# -------------------------------
# Bedrock Claude 3 Client
# -------------------------------
class ClaudeClient:
    def __init__(self, model_id: str, region_name: str = "us-east-1"):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id

    def invoke(self, prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
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

# -------------------------------
# Main Cognitive Loop
# -------------------------------
class CognitiveAgent:
    def __init__(self):
        self.embedder = Embedder()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.claude = ClaudeClient(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    def run_chat(self, user_input: str) -> str:
        # Embed user input
        query_vec = self.embedder.embed(user_input)

        # Query short-term and long-term memory
        stm_hits = self.stm.query(query_vec, top_k=5)
        ltm_hits = self.ltm.query(query_vec, top_k=5)

        # Check for due prospective memories
        reminders = check_due_reminders()

        # Build full RAG prompt
        prompt = build_prompt(user_input, stm_hits, ltm_hits, reminders)

        # Invoke Claude
        reply = self.claude.invoke(prompt)

        # Store conversation into STM
        conversation_text = f"User: {user_input}\nAssistant: {reply}"
        store_to_stm(conversation_text, self.embedder.embed, self.stm)

        return reply
