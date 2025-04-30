# cognition/cognitive_agent.py

"""
Cognitive Agent Orchestration.

This module defines the CognitiveAgent class, which manages the complete
cognitive loop: memory retrieval, prompt building, LLM invocation, and memory writing.
"""

from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from memory.prospective_memory import check_due_reminders
from memory.procedural_memory import ProceduralMemory, ProcedureValidationError
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
        self.procedural_memory = ProceduralMemory()
        self.claude = ClaudeClient(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    def run_chat(self, user_input: str) -> str:
        """
        Run the cognitive loop for a given user input.
        """
        # Embed user input
        query_vec = self.embedder.embed(user_input)

        # Query memory systems
        stm_hits = self.stm.query(query_vec, top_k=5)
        ltm_hits = self.ltm.query(query_vec, top_k=5)
        reminders = check_due_reminders()
        
        # Try to find relevant procedure
        procedure = self.procedural_memory.find_matching_procedure(user_input)
        
        # Build full prompt
        prompt = build_prompt(
            user_input, 
            stm_hits, 
            ltm_hits, 
            reminders,
            procedure=procedure  # Add procedure to prompt building
        )

        # Invoke LLM
        reply = self.claude.invoke(prompt)

        # Store conversation into STM
        conversation_text = f"User: {user_input}\nAssistant: {reply}"
        self.stm.insert(memory=None, metadata={"text": conversation_text})

        return reply

    def learn_procedure(self, name: str, steps: list) -> bool:
        """
        Learn a new procedure.
        
        Args:
            name: Procedure identifier
            steps: List of procedure steps
            
        Returns:
            bool: Success status
        """
        try:
            self.procedural_memory.add_procedure(name, steps)
            logger.info(f"Successfully learned procedure: {name}")
            return True
        except ProcedureValidationError as e:
            logger.error(f"Validation error learning procedure: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to learn procedure: {str(e)}")
            return False

    def recall_procedure(self, context: str) -> Optional[List[Any]]:
        """
        Recall a procedure based on context.
        
        Args:
            context: Context to match against procedures
            
        Returns:
            Optional[List[Any]]: Retrieved procedure steps or None
        """
        try:
            steps = self.procedural_memory.find_matching_procedure(context)
            if steps:
                logger.info(f"Successfully recalled procedure for context: {context}")
            return steps
        except Exception as e:
            logger.error(f"Error recalling procedure: {str(e)}")
            return None
