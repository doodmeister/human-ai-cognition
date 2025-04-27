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

from cognition.cognitive_agent import CognitiveAgent

def main():
    print("=" * 60)
    print("🤖  Welcome to Human-AI Cognition Chatbot")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    agent = CognitiveAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

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

        except Exception as e:
            print(f"\n🚨 An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
