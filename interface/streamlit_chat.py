import streamlit as st
from cognition.chatbot_with_memory import run_chat

st.title("🧠 Human-Like Memory-Augmented AI")

user_input = st.text_input("You:")
if user_input:
    with st.spinner("Thinking..."):
        reply = run_chat(user_input)
        st.markdown(f"**Claude:** {reply}")
