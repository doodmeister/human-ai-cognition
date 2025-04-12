import math, time

class ShortTermMemory:
    def __init__(self, half_life_seconds=300.0):
        self.half_life = half_life_seconds
        self.entries = [] 
    
    def add_entry(self, content, base_priority=1.0, important=False):
        entry = {
            "content": content,
            "time": time.time(),
            "base_priority": base_priority,
            "important": important,
            "emotion": 0.0  # future placeholder
        }
        self.entries.append(entry)
    
    def _current_priority(self, entry):
        age = time.time() - entry["time"]
        decayed = entry["base_priority"] * math.exp(-age * math.log(2) / self.half_life)
        if entry.get("important"):
            decayed *= 2.0
        if entry.get("emotion", 0.0) > 0.5:
            decayed *= 1.2
        return decayed
    
    def get_top_entries(self, top_n=5):
        scored_entries = [(self._current_priority(e), e) for e in self.entries]
        scored_entries.sort(reverse=True)
        return [e for _, e in scored_entries[:top_n]]

    def prune_old_entries(self, cutoff_priority=0.1):
        self.entries = [e for e in self.entries if self._current_priority(e) >= cutoff_priority]