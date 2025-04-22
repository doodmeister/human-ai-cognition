def store_to_stm(text, embed_func, stm_module):
    vec = embed_func(text)
    metadata = {
        "text": text,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "chat_response"
    }
    stm_module.insert(memory=vec, metadata=metadata)
