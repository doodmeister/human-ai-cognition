import uuid
from datetime import datetime
from typing import List, Optional

from opensearchpy import OpenSearch

class ShortTermMemoryOpenSearch:
    def __init__(self, index_name="humanai-stm", host="localhost", port=9200, vector_dim=768):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True
        )
        self.index_name = index_name
        self.vector_dim = vector_dim
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "index": {
                            "knn": True
                        }
                    },
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {"type": "knn_vector", "dimension": self.vector_dim},
                            "timestamp": {"type": "date"},
                            "priority": {"type": "float"},
                            "emotion": {"type": "float"},
                            "tags": {"type": "keyword"},
                            "important": {"type": "boolean"},
                            "times_accessed": {"type": "integer"}
                        }
                    }
                }
            )

    def add_entry(self, content: str, embedding: List[float], base_priority: float = 1.0,
                  important: bool = False, emotion: float = 0.0, tags: Optional[List[str]] = None):
        doc = {
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "priority": base_priority,
            "important": important,
            "emotion": emotion,
            "tags": tags or [],
            "times_accessed": 0
        }
        self.client.index(index=self.index_name, id=str(uuid.uuid4()), body=doc)

    def get_top_entries(self, query_embedding: List[float], k: int = 5):
        """Retrieves top-k most similar entries based on vector similarity and boosts priority by access."""
        res = self.client.search(index=self.index_name, body={
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        })
        results = []
        for hit in res["hits"]["hits"]:
            doc_id = hit["_id"]
            source = hit["_source"]
            source["times_accessed"] += 1
            self.client.update(index=self.index_name, id=doc_id, body={
                "doc": {"times_accessed": source["times_accessed"]}
            })
            results.append(source)
        return results

    def prune_low_priority(self, threshold: float = 0.1):
        """Deletes entries with priority below the threshold."""
        self.client.delete_by_query(index=self.index_name, body={
            "query": {
                "range": {
                    "priority": {"lt": threshold}
                }
            }
        })

    def dump_all(self) -> List[dict]:
        """Returns all entries in the STM index (for debugging or transfer)."""
        results = []
        res = self.client.search(index=self.index_name, body={"query": {"match_all": {}}}, size=1000)
        for hit in res["hits"]["hits"]:
            results.append(hit["_source"])
        return results
