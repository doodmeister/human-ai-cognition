import math
import time
from datetime import datetime

class MetaCognition:
    """Comprehensive Meta-cognitive controller simulating human cognitive states."""

    def __init__(self, decay_rate=0.05, fatigue_threshold=1.0):
        self.fatigue = 0.0
        self.alert_threshold = fatigue_threshold
        self.attention_level = 1.0
        self.decay_rate = decay_rate

        # Emotional state model: valence [-1, 1], arousal [0, 1]
        self.emotional_state = {
            "valence": 0.0,
            "arousal": 0.0,
            "last_updated": datetime.utcnow().isoformat()
        }

        # Prospective memory management
        self.prospective_memory = []

    def update_fatigue(self, activity_level=1.0, novelty=0.0, task_difficulty=0.0, priming_score=0.0):
        """Update fatigue and attention influenced by cognitive factors."""
        increment = 0.1 * activity_level + 0.05 * novelty + 0.05 * task_difficulty
        self.fatigue = min(self.fatigue + increment, 5.0)

        self._update_attention(novelty, task_difficulty, priming_score)

    def recover(self, rest_interval):
        """Recover fatigue and reset attention after rest."""
        k = 0.5
        self.fatigue *= math.exp(-k * rest_interval)
        if self.fatigue < 0.01:
            self.fatigue = 0.0
        self.reset_attention()

    def _update_attention(self, novelty, task_difficulty, priming_score):
        """Detailed attention update influenced by cognitive and emotional stimuli."""
        emotion_influence = 0.05 * (self.emotional_state["valence"] + self.emotional_state["arousal"])
        attention_delta = (
            -self.decay_rate
            + 0.1 * novelty
            + 0.1 * task_difficulty
            + 0.05 * priming_score
            + emotion_influence
            - 0.05 * self.fatigue
        )
        self.attention_level = max(0.0, min(1.0, self.attention_level + attention_delta))

    def update_emotion(self, valence, arousal):
        """Update the emotional state."""
        self.emotional_state["valence"] = max(-1.0, min(1.0, valence))
        self.emotional_state["arousal"] = max(0.0, min(1.0, arousal))
        self.emotional_state["last_updated"] = datetime.utcnow().isoformat()

    def suggest_action(self):
        """Determine if dream-state consolidation is needed."""
        if self.fatigue >= self.alert_threshold or self.attention_level < 0.3:
            self.handle_fatigue()
            return "DREAM"
        return "CONTINUE"

    def handle_fatigue(self):
        """Explicitly handle fatigue state, initiating dream-state."""
        print("[MetaCognition] Fatigue detected! Initiating dream-state consolidation...")
        self.initiate_dream_state()
        self.reset_attention()

    def initiate_dream_state(self):
        """Placeholder for integrating external dream-state consolidation processes."""
        print("[MetaCognition] >>> Dream state signal sent.")

    def reset_attention(self):
        """Reset attention level to maximum."""
        self.attention_level = 1.0

    def assess_importance(self, input_text, context_analysis=None):
        """Determine if input should be marked important for memory consolidation."""
        if self.attention_level > 0.8:
            return True
        if context_analysis:
            if isinstance(context_analysis, dict) and context_analysis.get("novelty_score", 0) > 0.7:
                return True
            if "important" in str(context_analysis).lower():
                return True
        if "remember this" in input_text.lower():
            return True
        return False

    def add_prospective_memory(self, trigger_time, action_description):
        """Schedule a future-oriented memory task."""
        self.prospective_memory.append((trigger_time, action_description))
        print(f"[MetaCognition] Scheduled future task: '{action_description}' at {trigger_time}")

    def check_prospective_memory(self, current_time=None):
        """Check and trigger actions from prospective memory."""
        current_time = current_time or time.time()
        triggered = [(t, a) for (t, a) in self.prospective_memory if t <= current_time]
        for trigger_time, action in triggered:
            print(f"[MetaCognition] Prospective action triggered: {action}")
        self.prospective_memory = [(t, a) for (t, a) in self.prospective_memory if t > current_time]