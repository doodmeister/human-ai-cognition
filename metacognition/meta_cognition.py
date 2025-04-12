class MetaCognition:
    def __init__(self):
        self.attention_level = 1.0
        self.decay_rate = 0.05
        self.fatigue_threshold = 0.3
        self.emotional_state = 0.0
    
    def update_attention(self, novelty=0.0, task_difficulty=0.0):
        delta = -self.decay_rate + 0.1 * novelty + 0.1 * task_difficulty + 0.05 * self.emotional_state
        self.attention_level = max(0.0, min(1.0, self.attention_level + delta))
        if self.attention_level < self.fatigue_threshold:
            self.handle_fatigue()
    
    def handle_fatigue(self):
        print("Fatigue detected! Suggesting consolidation.")
        self.attention_level = 1.0
    
    def assess_importance(self, input_text, context_analysis=None):
        if self.attention_level > 0.8:
            return True
        if context_analysis and "important" in context_analysis.lower():
            return True
        if "remember this" in input_text.lower():
            return True
        if self.attention_level < self.fatigue_threshold:
            return False
        novelty_score = context_analysis.get("novelty_score", 0) if isinstance(context_analysis, dict) else 0
        return novelty_score > 0.7