import streamlit as st
from cognition.cognitive_agent import CognitiveAgent

# Initialize the CognitiveAgent
if "agent" not in st.session_state:
    st.session_state.agent = CognitiveAgent()

# Title and description
st.title("🧠 Human-Like Memory-Augmented AI")
st.markdown("Engage in a conversation with an AI that uses memory and meta-cognition to enhance its responses.")

# Input field for user query
user_input = st.text_input("You:", placeholder="Type your message here...")

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if user_input:
    with st.spinner("Processing your input..."):
        try:
            # Get the response from the CognitiveAgent
            reply = st.session_state.agent.run_chat(user_input)
            # Append the conversation to the session state
            st.session_state.conversation.append({"user": user_input, "assistant": reply})
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display the conversation history
for message in st.session_state.conversation:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Claude:** {message['assistant']}")
