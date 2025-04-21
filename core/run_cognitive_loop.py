import time
from metacognition.meta_cognition import MetaCognition
from memory.short_term_memory import ShortTermMemory  # Adjust if your STM class is named differently
from utils.sensory_input import get_sensory_input  # You’ll define this next
from utils.ltm import store_in_ltm  # Optional: placeholder for LTM integration

# Initialize system modules
meta = MetaCognition()
stm = ShortTermMemory()

# Simulation loop parameters
dream_interval = 10  # cycles
cycle = 0

def run_loop():
    global cycle
    while True:
        print(f"\n[Cycle {cycle}] ---------------------------")

        # 1. Simulated sensory input
        input_text = get_sensory_input()

        # 2. Contextual analysis (simple)
        context = {
            "novelty_score": 0.9 if "new" in input_text.lower() else 0.4
        }

        # 3. Optional priming score from LTM
        priming_score = 0.6  # Placeholder — you could pull this from LTM similarity

        # 4. Update attention state
        meta.update_attention(
            novelty=context["novelty_score"],
            task_difficulty=0.5,
            priming_score=priming_score
        )

        # 5. Check and trigger prospective memory
        meta.check_prospective_memory()

        # 6. Determine if input is important enough to store
        if meta.assess_importance(input_text, context):
            print("[STM] Storing input based on assessed importance.")
            stm.store(input_text)
        else:
            print("[STM] Input not stored (low importance or fatigue).")

        # 7. Optional: store to LTM during dream state
        if cycle > 0 and cycle % dream_interval == 0:
            print("[MetaCognition] Dream interval reached. Triggering consolidation.")
            meta.initiate_dream_state()
            store_in_ltm(stm.flush_high_priority())  # Optional LTM logic

        cycle += 1
        time.sleep(2)  # Simulate passage of time

if __name__ == "__main__":
    run_loop()
