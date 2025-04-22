def embed_text(text: str) -> list[float]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()

def build_prompt(user_input, stm_results, ltm_results, reminders, max_tokens=512):
    def format_section(title, docs):
        return f"### {title}\n" + "\n".join(d.get("text", "") for d in docs[:3]) + "\n"

    stm_section = format_section("Recent Short-Term Memory", stm_results)
    ltm_section = format_section("Relevant Long-Term Knowledge", ltm_results)
    reminder_section = format_section("Upcoming Tasks", [{"text": r} for r in reminders])

    context = f"{stm_section}\n{ltm_section}\n{reminder_section}"
    prompt = f"""{context}
You are a reflective and memory-aware assistant.
Answer the user's query using the information above, reasoning like a human.

User: {user_input}
Assistant:"""

    return prompt
