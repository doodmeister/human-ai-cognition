# memory/semantic_memory.py

class SemanticMemory:
    """Long-Term Memory store for semantic knowledge (facts, concepts).
    
    Stores information in a vector database for similarity-based retrieval.
    """
    def __init__(self, vector_index=None):
        """
        Args:
            vector_index: An object providing add_document(text) and query(text, top_k) methods.
                          If None, uses a default in-memory list.
        """
        self.vector_index = vector_index
        if self.vector_index is None:
            self.documents = []  # fallback storage
    
    def add_knowledge(self, text):
        """Add a piece of knowledge to semantic memory."""
        if self.vector_index:
            self.vector_index.add_document(text)
        else:
            self.documents.append(text)
    
    def query(self, query_text, top_k=5):
        """Query semantic memory for relevant info."""
        if self.vector_index:
            return self.vector_index.query(query_text, top_k=top_k)
        else:
            # simple linear search placeholder
            results = []
            for doc in self.documents:
                if query_text.lower() in doc.lower():
                    results.append(doc)
            return results[:top_k]