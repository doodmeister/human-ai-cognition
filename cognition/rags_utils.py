# cognition/rag_utils.py

"""
Retrieval-Augmented Generation (RAG) utilities.

Handles building prompts that combine user inputs with retrieved memories
(short-term, long-term, and prospective) to create context-rich inputs for the LLM.
"""

from typing import List, Union, Dict

def embed_text(text: str, embedder) -> list:
    """
    Embed a piece of text using the provided embedder.

    Args:
        text (str): Text to embed.
        embedder: SentenceTransformer or similar embedder instance.

    Returns:
        list: Embedding vector.
    """
    return embedder.embed(text)

def format_section(title: str, docs: List[Union[str, Dict[str, str]]]) -> str:
    """
    Format a memory section for the final prompt.

    Args:
        title (str): Section title (e.g., "Short Term Memory").
        docs (List): List of documents, either strings or dicts with a 'text' field.

    Returns:
        str: Formatted section text.
    """
    if not docs:
        return ""

    lines = []
    for d in docs[:3]:  # Limit to top 3 items per section
        if isinstance(d, dict):
            lines.append(d.get("text", ""))
        else:
            lines.append(str(d))
    
    section_text = f"### {title}\n" + "\n".join(lines) + "\n"
    return section_text

def build_prompt(user_input: str,
                 stm_hits: List[Union[str, Dict[str, str]]],
                 ltm_hits: List[Union[str, Dict[str, str]]],
                 reminders: List[Union[str, Dict[str, str]]]) -> str:
    """
    Build the complete prompt to send to the LLM.

    Args:
        user_input (str): The user's raw input query.
        stm_hits (list): Retrieved Short-Term Memory results.
        ltm_hits (list): Retrieved Long-Term Memory results.
        reminders (list): Retrieved Prospective Memory results.

    Returns:
        str: Complete prompt ready for LLM input.
    """
    prompt_parts = []

    # System behavior instruction
    prompt_parts.append("You are a helpful AI assistant with a structured memory system.")
    prompt_parts.append("You can recall memories, reminders, and past conversations to assist the user.\n")

    # Add memory sections
    prompt_parts.append(format_section("Short Term Memory", stm_hits))
    prompt_parts.append(format_section("Long Term Memory", ltm_hits))
    prompt_parts.append(format_section("Reminders", reminders))

    # Current user input
    prompt_parts.append(f"### Current Conversation\nUser: {user_input}\nAssistant:")

    # Combine everything cleanly
    full_prompt = "\n".join(part for part in prompt_parts if part)

    return full_prompt
