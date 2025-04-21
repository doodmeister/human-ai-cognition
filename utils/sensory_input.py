import random

def get_sensory_input():
    examples = [
        "This is a new observation.",
        "Reminder: remember this later.",
        "Just another input.",
        "Important context shift detected.",
        "Repeat data stream detected."
    ]
    return random.choice(examples)
