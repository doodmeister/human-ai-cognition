from datetime import datetime
from .state import meta_state


class MetaFeedbackManager:
    def __init__(self):
        self.state = meta_state

    def send_feedback(self, event, count, avg_salience=None, timestamp=None):
        timestamp = timestamp or datetime.utcnow().isoformat()
        avg_salience = round(avg_salience or 0.0, 4)

        log_entry = {
            "event": event,
            "timestamp": timestamp,
            "entries_consolidated": count,
            "avg_salience": avg_salience
        }

        self.state["dream_log"].append(log_entry)
        self.state["dream_count"] += 1

        # Adjust meta states
        self.state["fatigue_level"] = max(0.0, self.state["fatigue_level"] - 0.1)
        self.state["attention_level"] = min(1.0, self.state["attention_level"] + 0.1)

        # Simulated introspection logic
        if avg_salience < 0.3:
            print("[Meta-Cognition] Low-value dream detected. Suggesting STM reheating.")
        elif avg_salience > 0.7:
            print("[Meta-Cognition] High-value dream — reinforce related memory pathways.")

        print(f"[Meta-Cognition] Dream #{self.state['dream_count']} → {count} entries, salience={avg_salience}, fatigue={self.state['fatigue_level']}, attention={self.state['attention_level']}")
