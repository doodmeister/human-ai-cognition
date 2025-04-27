# memory/short_term_memory.py

"""
Short Term Memory (STM) module.

Handles storing and retrieving recent memory entries for fast, temporary access.
Incorporates basic decay over time to simulate forgetting.
"""

from datetime import datetime, timedelta

class ShortTermMemory:
    def __init__(self, max_items=100, decay_minutes=60):
        """
        Initialize the short-term memory.

        Args:
            max_items (int): Maximum number of items to retain.
            decay_minutes (int): Time after which items are considered 'forgotten'.
        """
        self.max_items = max_items
        self.decay_minutes = decay_minutes
        self.memory = []  # List of (timestamp, text) tuples

    def add(self, content: str):
        """
        Add a new memory item.

        Args:
            content (str): The memory text to store.
        """
        timestamp = datetime.utcnow()
        self.memory.append((timestamp, content))

        # Keep only the most recent N items
        if len(self.memory) > self.max_items:
            self.memory = self.memory[-self.max_items:]

    def insert(self, memory: list, metadata: dict = None):
        """
        Insert a new memory from an embedded source (for compatibility).

        Args:
            memory (list): Vector representation (ignored here, we store text).
            metadata (dict): Metadata containing at least a 'text' field.
        """
        text = metadata.get("text") if metadata else str(memory)
        self.add(text)

    def query(self, query_vector: list, top_k: int = 5) -> list:
        """
        Query short-term memory for recent relevant items.

        Args:
            query_vector (list): Unused here (no real vector similarity yet).
            top_k (int): Number of top results to return.

        Returns:
            List[dict]: List of recent memory entries with 'text' field.
        """
        # Apply decay before returning
        self._apply_decay()

        # Return the most recent top_k memories in {"text": ...} format
        recent = self.memory[-top_k:]
        return [{"text": content} for (timestamp, content) in recent]

    def _apply_decay(self):
        """Forget memories older than decay_minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.decay_minutes)
        self.memory = [(ts, text) for (ts, text) in self.memory if ts >= cutoff]

    def get_recent_items(self, count: int = 5) -> list:
        """
        Get the most recent memory items (no decay check).

        Args:
            count (int): Number of items to retrieve.

        Returns:
            List[str]: List of recent memory texts.
        """
        return [content for (_, content) in self.memory[-count:]]
