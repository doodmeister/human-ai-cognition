# cognition/rag_utils.py

"""
Retrieval-Augmented Generation (RAG) utilities.

Handles building prompts that combine user inputs with retrieved memories
(short-term, long-term, and prospective) to create context-rich inputs for the LLM.
"""

from typing import List, Union, Dict, Optional, Any

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
                stm_hits: list, 
                ltm_hits: list, 
                reminders: list,
                procedure: Optional[List[Any]] = None) -> str:
    """
    Build prompt incorporating all memory sources.
    """
    prompt_parts = []
    
    # Add relevant memories
    if stm_hits:
        prompt_parts.append("Recent context:")
        prompt_parts.extend(stm_hits)
        
    if ltm_hits:
        prompt_parts.append("Related memories:")
        prompt_parts.extend(ltm_hits)
        
    if reminders:
        prompt_parts.append("Active reminders:")
        prompt_parts.extend(reminders)
        
    if procedure:
        prompt_parts.append("Relevant procedure steps:")
        prompt_parts.extend([f"- {step}" for step in procedure])
    
    prompt_parts.append(f"User: {user_input}")
    return "\n".join(prompt_parts)
