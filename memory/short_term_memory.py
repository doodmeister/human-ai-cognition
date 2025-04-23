# memory/short_term_memory.py

from collections import deque
import time

class STMItem:
    """Container for an item in Short-Term Memory.
    
    Attributes:
        content: The raw data of the memory (e.g., text snippet, sensor reading).
        timestamp: The time when the memory was added (seconds since epoch).
        importance: A floating-point score representing attention/importance.
        prev_id: Link to the previous STM item ID for context chaining (if any).
    """
    def __init__(self, content, importance=1.0, prev_id=None):
        self.content = content
        self.timestamp = time.time()
        self.importance = importance  # initial attention level
        self.prev_id = prev_id  # link to previous event in sequence

class ShortTermMemory:
    """Short-Term Memory buffer for recent experiences (working memory).
    
    Simulates human working memory with limited capacity and decaying activation.
    New items can be added, while older items decay and are removed over time.
    Supports context chaining: related sequential items are linked.
    """
    def __init__(self, capacity=50, decay_half_life=30.0):
        """
        Args:
            capacity (int): Max number of items to hold at once.
            decay_half_life (float): Half-life in seconds for the importance decay.
        """
        self.capacity = capacity
        self.decay_half_life = decay_half_life
        self.items = deque()  # store STMItem objects
        self.last_item_id = 0  # simple counter for item IDs
        self.last_added_id = None  # track the most recently added item's ID
    
    def add(self, content, importance=1.0):
        """Add a new memory to STM, possibly evicting oldest if capacity exceeded.
        
        If a previous item exists, link this new item to it (context chaining).
        """
        # Decay existing items before adding new one
        self._apply_decay()
        # Create new item with link to previous
        new_item = STMItem(content, importance, prev_id=self.last_added_id)
        self.last_item_id += 1
        new_item.id = self.last_item_id
        # Add to STM
        self.items.append(new_item)
        self.last_added_id = new_item.id
        # Enforce capacity
        if len(self.items) > self.capacity:
            self.items.popleft()  # remove oldest item
        return new_item.id
    
    def get_recent_items(self, n=5):
        """Retrieve the N most recent items (after applying decay to update importances).
        
        Returns:
            List of (content, importance) for up to N latest items, sorted from newest to oldest.
        """
        self._apply_decay()
        recent = list(self.items)[-n:]  # take last n items
        # Sort by insertion order (newest last in deque, so already in order newest->oldest)
        recent_sorted = sorted(recent, key=lambda x: x.timestamp, reverse=True)
        return [(item.content, item.importance) for item in recent_sorted]
    
    def _apply_decay(self):
        """Internal: apply exponential decay to importance of all items based on time elapsed."""
        current_time = time.time()
        for item in list(self.items):
            # compute time since item added
            age = current_time - item.timestamp
            # exponential decay: importance decays by half every decay_half_life seconds
            decay_factor = 0.5 ** (age / self.decay_half_life)
            item.importance *= decay_factor
            # If importance falls very low, consider removing the item (forgetting)
            if item.importance < 0.01:
                # Remove item from deque if it's at either end
                try:
                    self.items.remove(item)
                except ValueError:
                    pass  # item might have already been removed if capacity trimming
