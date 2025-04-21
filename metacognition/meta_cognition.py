import time

class MetaCognition:
    def __init__(self):
        # Core attributes
        self.attention_level = 1.0
        self.decay_rate = 0.05
        self.fatigue_threshold = 0.3

        # Emotional model (valence: positive/negative, arousal: intensity)
        self.emotional_state = {"valence": 0.0, "arousal": 0.0}

        # Prospective memory: [(trigger_time, action_description)]
        self.prospective_memory = []

    def update_attention(self, novelty=0.0, task_difficulty=0.0, priming_score=0.0):
        """
        Update attention level based on several cognitive variables.
        """
        emotion_mod = 0.05 * (self.emotional_state["valence"] + self.emotional_state["arousal"])
        delta = (-self.decay_rate +
                 0.1 * novelty +
                 0.1 * task_difficulty +
                 0.05 * priming_score +
                 emotion_mod)

        self.attention_level = max(0.0, min(1.0, self.attention_level + delta))

        print(f"[MetaCognition] Updated attention: {self.attention_level:.2f} (Δ={delta:.3f})")
        
        if self.attention_level < self.fatigue_threshold:
            self.handle_fatigue()

    def handle_fatigue(self):
        """
        Trigger memory consolidation or strategy shift when fatigued.
        """
        print("[MetaCognition] Fatigue detected! Initiating memory consolidation via dream state.")
        self.initiate_dream_state()
        self.attention_level = 1.0

    def initiate_dream_state(self):
        """
        Placeholder hook for triggering dream-state memory consolidation.
        """
        print("[MetaCognition] >>> Dream state triggered. Consolidating memory...")
        # This could call a DreamStateProcessor instance or send a signal

    def assess_importance(self, input_text, context_analysis=None):
        """
        Assess whether input should be flagged as important.
        """
        if self.attention_level > 0.8:
            return True
        if context_analysis:
            if isinstance(context_analysis, dict):
                if context_analysis.get("novelty_score", 0) > 0.7:
                    return True
            if "important" in str(context_analysis).lower():
                return True
        if "remember this" in input_text.lower():
            return True
        if self.attention_level < self.fatigue_threshold:
            return False
        return False

    def check_prospective_memory(self, current_time=None):
        """
        Check and trigger prospective memory tasks.
        """
        if current_time is None:
            current_time = time.time()

        triggered = []
        for trigger_time, action in self.prospective_memory:
            if trigger_time <= current_time:
                print(f"[MetaCognition] Triggering planned action: {action}")
                triggered.append((trigger_time, action))

        # Remove triggered items
        self.prospective_memory = [
            (t, a) for (t, a) in self.prospective_memory if (t, a) not in triggered
        ]

    def add_prospective_memory(self, trigger_time, action_description):
        """
        Schedule a future task.
        """
        self.prospective_memory.append((trigger_time, action_description))
        print(f"[MetaCognition] Scheduled future task: '{action_description}' at {trigger_time}")
