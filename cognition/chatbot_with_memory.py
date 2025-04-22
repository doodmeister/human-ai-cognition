from datetime import datetime
from sentence_transformers import SentenceTransformer
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.prospective_memory import check_due_reminders
from cognition.rag_utils import embed_text, build_prompt
from cognition.memory_writer import store_to_stm
import boto3

embedder = SentenceTransformer("all-MiniLM-L6-v2")
stm = ShortTermMemory()
ltm = LongTermMemory()
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def invoke_claude(prompt: str) -> str:
    response = bedrock.converse(
        modelId=CLAUDE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.5}
    )
    return response["output"]["message"]["content"][0]["text"]

def run_chat(user_input: str) -> str:
    query_vec = embed_text(user_input)

    stm_hits = stm.query(query_vec, top_k=5)
    ltm_hits = ltm.query(query_vec, top_k=5)
    reminders = check_due_reminders()

    prompt = build_prompt(user_input, stm_hits, ltm_hits, reminders)
    reply = invoke_claude(prompt)

    # Autonomously write response + query to STM
    store_to_stm(f"User: {user_input}\nAssistant: {reply}", embed_text, stm)

    return reply
