# cognition/cognitive_agent.py

"""
Cognitive Agent Orchestration.

This module defines the CognitiveAgent class, which manages the complete
cognitive loop: memory retrieval, prompt building, LLM invocation, and memory writing.
"""

from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.prospective_memory import check_due_reminders
from cognition.rag_utils import embed_text, build_prompt
from cognition.claude_client import ClaudeClient
from cognition.embedder import Embedder

class CognitiveAgent:
    def __init__(self):
        """
        Initialize the Cognitive Agent.
        """
        self.embedder = Embedder()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.claude = ClaudeClient(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    def run_chat(self, user_input: str) -> str:
        """
        Run the cognitive loop for a given user input.

        Args:
            user_input (str): User's query.

        Returns:
            str: Assistant's reply.
        """
        # Embed user input
        query_vec = self.embedder.embed(user_input)

        # Query memory systems
        stm_hits = self.stm.query(query_vec, top_k=5)
        ltm_hits = self.ltm.query(query_vec, top_k=5)
        reminders = check_due_reminders()

        # Build full prompt
        prompt = build_prompt(user_input, stm_hits, ltm_hits, reminders)

        # Invoke LLM
        reply = self.claude.invoke(prompt)

        # Store conversation into STM
        conversation_text = f"User: {user_input}\nAssistant: {reply}"
        self.stm.insert(memory=None, metadata={"text": conversation_text})

        return reply
