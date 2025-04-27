import math
import time
import logging
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaCognition:
    """Comprehensive Meta-cognitive controller simulating human cognitive states."""

    def __init__(self, decay_rate: float = 0.05, fatigue_threshold: float = 1.0):
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
        self._lock = threading.Lock()

    def update_fatigue(self, activity_level: float = 1.0, novelty: float = 0.0, 
                       task_difficulty: float = 0.0, priming_score: float = 0.0) -> None:
        """
        Update fatigue and attention influenced by cognitive factors.

        Args:
            activity_level (float): Level of activity [0.0, 1.0].
            novelty (float): Novelty of the task [0.0, 1.0].
            task_difficulty (float): Difficulty of the task [0.0, 1.0].
            priming_score (float): Priming score [0.0, 1.0].

        Raises:
            ValueError: If any input is outside the range [0.0, 1.0].
        """
        activity_level = max(0.0, min(1.0, activity_level))
        novelty = max(0.0, min(1.0, novelty))
        task_difficulty = max(0.0, min(1.0, task_difficulty))
        priming_score = max(0.0, min(1.0, priming_score))

        increment = 0.1 * activity_level + 0.05 * novelty + 0.05 * task_difficulty
        self.fatigue = min(self.fatigue + increment, 5.0)

        self._update_attention(novelty, task_difficulty, priming_score)

    def recover(self, rest_interval: float) -> None:
        """Recover fatigue and reset attention after rest."""
        k = 0.5
        self.fatigue *= math.exp(-k * rest_interval)
        if self.fatigue < 0.01:
            self.fatigue = 0.0
        self.reset_attention()

    def _update_attention(self, novelty: float, task_difficulty: float, priming_score: float) -> None:
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

    def update_emotion(self, valence: float, arousal: float) -> None:
        """Update the emotional state."""
        self.emotional_state["valence"] = max(-1.0, min(1.0, valence))
        self.emotional_state["arousal"] = max(0.0, min(1.0, arousal))
        self.emotional_state["last_updated"] = datetime.utcnow().isoformat()

    def suggest_action(self) -> str:
        """Determine if dream-state consolidation is needed."""
        if self.fatigue >= self.alert_threshold or self.attention_level < 0.3:
            self.handle_fatigue()
            return "DREAM"
        return "CONTINUE"

    def handle_fatigue(self) -> None:
        """Explicitly handle fatigue state, initiating dream-state."""
        logger.warning("[MetaCognition] Fatigue detected! Initiating dream-state consolidation...")
        self.initiate_dream_state()
        self.reset_attention()

    def initiate_dream_state(self) -> None:
        """Placeholder for integrating external dream-state consolidation processes."""
        logger.info("[MetaCognition] >>> Dream state signal sent.")

    def reset_attention(self) -> None:
        """Reset attention level to maximum."""
        self.attention_level = 1.0

    def assess_importance(self, input_text: str, context_analysis: Optional[Dict] = None) -> bool:
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

    def add_prospective_memory(self, trigger_time: float, action_description: str) -> None:
        """Schedule a future-oriented memory task."""
        with self._lock:
            self.prospective_memory.append((trigger_time, action_description))
            logger.info(f"[MetaCognition] Scheduled future task: '{action_description}' at {trigger_time}")

    def check_prospective_memory(self, current_time: Optional[float] = None) -> None:
        """Check and trigger actions from prospective memory."""
        current_time = current_time or time.time()
        with self._lock:
            remaining_memory = []
            for trigger_time, action in self.prospective_memory:
                if trigger_time <= current_time:
                    logger.info(f"[MetaCognition] Prospective action triggered: {action}")
                else:
                    remaining_memory.append((trigger_time, action))
            self.prospective_memory = remaining_memory