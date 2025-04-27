from datetime import datetime
import logging
from .state import meta_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for thresholds
LOW_SALIENCE_THRESHOLD = 0.3
HIGH_SALIENCE_THRESHOLD = 0.7
FATIGUE_DECREMENT = 0.1
ATTENTION_INCREMENT = 0.1

class MetaFeedbackManager:
    """
    Manages meta-cognitive feedback, including logging dream consolidation events
    and adjusting cognitive states like fatigue and attention.
    """
    def __init__(self, state=None):
        """
        Initialize the MetaFeedbackManager.

        Args:
            state (dict, optional): The meta-cognitive state to manage. Defaults to `meta_state`.
        """
        self.state = state or meta_state

    def send_feedback(self, event, count, avg_salience=None, timestamp=None):
        """
        Log feedback for a meta-cognitive event and adjust cognitive states.

        Args:
            event (str): The name of the event (e.g., 'dream_consolidation').
            count (int): The number of entries consolidated.
            avg_salience (float, optional): The average salience of the entries. Defaults to 0.0.
            timestamp (str, optional): The timestamp of the event. Defaults to the current UTC time.
        """
        try:
            # Validate inputs
            if not isinstance(event, str) or not event:
                raise ValueError("Event must be a non-empty string.")
            if not isinstance(count, int) or count < 0:
                raise ValueError("Count must be a non-negative integer.")
            if avg_salience is not None and (not isinstance(avg_salience, (int, float)) or not (0.0 <= avg_salience <= 1.0)):
                raise ValueError("Avg_salience must be a float between 0.0 and 1.0.")

            timestamp = timestamp or datetime.utcnow().isoformat()
            avg_salience = round(avg_salience or 0.0, 4)

            # Log the event
            log_entry = {
                "event": event,
                "timestamp": timestamp,
                "entries_consolidated": count,
                "avg_salience": avg_salience
            }
            self.state["dream_log"].append(log_entry)
            self.state["dream_count"] += 1

            # Adjust meta states
            self.state["fatigue_level"] = max(0.0, self.state["fatigue_level"] - FATIGUE_DECREMENT)
            self.state["attention_level"] = min(1.0, self.state["attention_level"] + ATTENTION_INCREMENT)

            # Simulated introspection logic
            if avg_salience < LOW_SALIENCE_THRESHOLD:
                logger.info("[Meta-Cognition] Low-value dream detected. Suggesting STM reheating.")
            elif avg_salience > HIGH_SALIENCE_THRESHOLD:
                logger.info("[Meta-Cognition] High-value dream — reinforce related memory pathways.")

            logger.info(
                f"[Meta-Cognition] Dream #{self.state['dream_count']} → {count} entries, "
                f"salience={avg_salience}, fatigue={self.state['fatigue_level']}, "
                f"attention={self.state['attention_level']}"
            )
        except Exception as e:
            logger.error(f"Error in send_feedback: {e}", exc_info=True)
