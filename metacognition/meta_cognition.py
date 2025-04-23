import math
import time
from datetime import datetime

class MetaCognition:
    """Enhanced Meta-cognition integrating fatigue, emotions, and prospective memory."""
    def __init__(self, alert_threshold=1.0, decay_rate=0.05):
        self.fatigue = 0.0
        self.alert_threshold = alert_threshold
        self.decay_rate = decay_rate
        self.emotional_state = {"valence": 0.0, "arousal": 0.0, "last_updated": datetime.utcnow().isoformat()}
        self.prospective_memory = []

    def update_fatigue(self, activity_level=1.0):
        self.fatigue += 0.1 * activity_level
        self.fatigue = min(self.fatigue, 5.0)

    def recover(self, rest_interval):
        k = 0.5
        self.fatigue *= math.exp(-k * rest_interval)
        if self.fatigue < 0.01:
            self.fatigue = 0.0

    def update_emotion(self, valence: float, arousal: float):
        self.emotional_state["valence"] = max(-1.0, min(1.0, valence))
        self.emotional_state["arousal"] = max(0.0, min(1.0, arousal))
        self.emotional_state["last_updated"] = datetime.utcnow().isoformat()

    def need_break(self):
        return self.fatigue >= self.alert_threshold

    def initiate_dream_state(self):
        """Explicit external integration hook."""
        print("[MetaCognition] Initiating dream-state memory consolidation.")

    def suggest_action(self):
        if self.need_break():
            self.initiate_dream_state()
            return "DREAM"
        return "CONTINUE"

    def add_prospective_memory(self, trigger_time, action_description):
        self.prospective_memory.append((trigger_time, action_description))

    def check_prospective_memory(self, current_time=None):
        current_time = current_time or time.time()
        triggered = [action for (t, action) in self.prospective_memory if t <= current_time]
        for action in triggered:
            print(f"[MetaCognition] Prospective action triggered: {action}")
        self.prospective_memory = [(t, a) for (t, a) in self.prospective_memory if t > current_time]
