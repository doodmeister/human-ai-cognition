# memory/episodic_memory.py

class EpisodicMemory:
    """Long-Term Memory store for episodic (context-rich) memories.
    
    Stores events with contextual data like time and related items. Supports time-indexed retrieval.
    """
    def __init__(self):
        self.events = []  # list of episodic events (each could be a dict with 'timestamp', 'summary', 'details')
    
    def store_event(self, summary, details=None, timestamp=None, tags=None):
        """Store a new episodic memory.
        
        Args:
            summary (str): A concise summary of the event.
            details (dict or str): Additional details or raw data of the event.
            timestamp (float): Unix time of event (if None, current time used).
            tags (list of str): Optional tags or labels (e.g., emotional tone, location).
        """
        if timestamp is None:
            timestamp = time.time()
        event = {
            "summary": summary,
            "details": details,
            "timestamp": timestamp,
            "tags": tags or []
        }
        self.events.append(event)
        # (In a real implementation, also add embedding to vector index for semantic lookup)
    
    def query(self, query_text, top_k=5, time_range=None):
        """Retrieve relevant episodic memories (using semantic similarity and/or time filtering).
        
        Args:
            query_text (str): A description of what to recall.
            top_k (int): Number of top results to return.
            time_range (tuple): Optional (start, end) timestamps to filter events.
        
        Returns:
            List of event summaries matching the query.
        """
        # For now, a simple keyword search as placeholder for vector similarity:
        candidates = []
        for event in self.events:
            if time_range:
                if not (time_range[0] <= event["timestamp"] <= time_range[1]):
                    continue
            text = event["summary"] + " " + str(event.get("details", ""))
            if query_text.lower() in text.lower():
                candidates.append(event)
        # Sort by recency or relevance (placeholder: recency)
        candidates.sort(key=lambda e: e["timestamp"], reverse=True)
        results = [c["summary"] for c in candidates[:top_k]]
        return results
