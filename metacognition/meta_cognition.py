import time
from datetime import datetime

class MetaCognition:
    def __init__(self, decay_rate=0.05, fatigue_threshold=0.3):
        self.attention_level = 1.0
        self.decay_rate = decay_rate
        self.fatigue_threshold = fatigue_threshold

        # Emotional model: valence [-1, 1], arousal [0, 1]
        self.emotional_state = {
            "valence": 0.0,
            "arousal": 0.0,
            "last_updated": datetime.utcnow().isoformat()
        }

        # Prospective memory: [(trigger_time, action_description)]
        self.prospective_memory = []

    def update_attention(self, novelty=0.0, task_difficulty=0.0, priming_score=0.0):
        """
        Update attention level based on decay and stimuli.
        """
        emotion_mod = 0.05 * (self.emotional_state["valence"] + self.emotional_state["arousal"])
        delta = (-self.decay_rate +
                 0.1 * novelty +
                 0.1 * task_difficulty +
                 0.05 * priming_score +
                 emotion_mod)

        self.attention_level = max(0.0, min(1.0, self.attention_level + delta))
        print(f"[MetaCognition] Updated attention: {self.attention_level:.2f} (Δ={delta:.3f})")

        return self.attention_level < self.fatigue_threshold

    def handle_fatigue(self):
        """
        Trigger dream-state memory consolidation.
        """
        print("[MetaCognition] Fatigue detected! Initiating dream-state consolidation...")
        self.initiate_dream_state()
        self.attention_level = 1.0

    def initiate_dream_state(self):
        """
        Placeholder hook — override this to call actual dream processor externally.
        """
        print("[MetaCognition] >>> Dream state signal sent.")

    def reset_attention(self):
        """
        Useful after a dream cycle or significant system event.
        """
        self.attention_level = 1.0
        print("[MetaCognition] Attention reset.")

    def assess_importance(self, input_text: str, context_analysis=None) -> bool:
        """
        Determine if input should be marked important.
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
        return False

    def update_emotion(self, valence: float, arousal: float):
        """
        Update the internal emotional state — should be called externally based on input.
        """
        self.emotional_state["valence"] = max(-1.0, min(1.0, valence))
        self.emotional_state["arousal"] = max(0.0, min(1.0, arousal))
        self.emotional_state["last_updated"] = datetime.utcnow().isoformat()
        print(f"[MetaCognition] Emotion updated: valence={valence:.2f}, arousal={arousal:.2f}")

    def check_prospective_memory(self, current_time=None):
        """
        Check and trigger planned actions.
        """
        if current_time is None:
            current_time = time.time()

        triggered = []
        for trigger_time, action in self.prospective_memory:
            if trigger_time <= current_time:
                print(f"[MetaCognition] Triggering future action: {action}")
                triggered.append((trigger_time, action))

        self.prospective_memory = [
            (t, a) for (t, a) in self.prospective_memory if (t, a) not in triggered
        ]

    def add_prospective_memory(self, trigger_time, action_description):
        self.prospective_memory.append((trigger_time, action_description))
        print(f"[MetaCognition] Future task scheduled: '{action_description}' at {trigger_time}")
