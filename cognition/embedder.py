# cognition/embedder.py

"""
Embedder module.

Handles text embedding using a SentenceTransformer model.
Optimized for reuse (load model once, embed multiple times).
"""

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name (str): Huggingface model to load.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list:
        """
        Embed a single text input.

        Args:
            text (str): Input text.

        Returns:
            list: Embedding vector (as list, not numpy array).
        """
        return self.model.encode(text).tolist()
