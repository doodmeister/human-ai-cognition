# main.py

"""
Main launcher for the Human-AI Cognition Agent.

This script creates a CognitiveAgent instance and enables a simple terminal-based conversation loop,
where the agent utilizes short-term memory, long-term memory, and prospective memory to augment its responses.

Accounts for all current architecture improvements:
- Embedder class
- CognitiveAgent class
- Proper Bedrock Claude invocation
- Robust prompt building
- Error handling
"""

import logging
from cognition.cognitive_agent import CognitiveAgent
import signal
import sys

# Configure logging
logging.basicConfig(
    filename="cognition_chatbot.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Graceful shutdown handler
def signal_handler(sig, frame):
    print("\n👋 Exiting gracefully. Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("=" * 60)
    print("🤖  Welcome to Human-AI Cognition Chatbot")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    agent = CognitiveAgent()

    while True:
        try:
            # Timeout for user input (optional, set to None for no timeout)
            signal.alarm(120)  # 120 seconds timeout
            user_input = input("\nYou: ").strip()
            signal.alarm(0)  # Disable alarm after input

            # Exit command
            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break

            # Empty input guard
            if not user_input:
                print("⚠️  Please enter a message.")
                continue

            # Process input through the cognitive loop
            reply = agent.run_chat(user_input)
            print(f"\nAssistant: {reply}")

        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"\n🚨 An unexpected error occurred. Please try again.")
            continue

if __name__ == "__main__":
    main()
