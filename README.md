# Human-AI Cognition

> **Where Memory Meets Meaning**  
> A modular framework for building a human-like AI system with memory, reasoning, and meta-cognitive abilities.

---

## 🧠 Overview

Human-AI Cognition is an open-source project to create a cognitive architecture that simulates aspects of human memory, attention, and thought.  
The system leverages **Short-Term Memory (STM)**, **Long-Term Memory (LTM)**, **Prospective Memory**, and Retrieval-Augmented Generation (**RAG**) to interact with a Large Language Model (Claude via AWS Bedrock).

The agent retrieves memories, builds context-rich prompts, and responds with reasoning augmented by its evolving memory.

---

## 📂 Project Structure

```
human-ai-cognition/
├── cognition/
│   ├── cognitive_agent.py         # Main agent orchestration
│   ├── rag_utils.py                # Prompt building utilities (RAG)
├── memory/
│   ├── short_term_memory.py        # Short-term memory (STM)
│   ├── long_term_memory.py         # Long-term memory (LTM - semantic focus)
│   ├── prospective_memory.py       # Scheduled future tasks (reminders)
│   ├── semantic_memory.py          # Semantic memory retrieval
│   ├── episodic_memory.py          # Episodic memory placeholder
├── model/
│   ├── dpad_transformer.py         # Dynamic Predictive Attention Transformer (WIP)
├── main.py                         # 🚀 Main launcher (new)
├── terraform/                      # (optional) AWS deployment infrastructure
├── README.md                       # (this file)
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You'll need:
- `boto3`
- `sentence-transformers`
- `torch`
- Any additional AWS authentication libraries if using Bedrock.

---

### 2. Set Environment Variables

You must have AWS credentials configured (`aws configure`) and access to:
- **AWS Bedrock Runtime** (Claude 3 model)
- (Optional) **S3** and **OpenSearch** if using memory consolidation features.

Set your environment variables (example):

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=cognition-inputs
export BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
```

---

### 3. Run the Cognitive Agent

```bash
python main.py
```

You should see:

```
🤖 Welcome to Human-AI Cognition Chatbot
Type 'exit' or 'quit' to end the conversation.
============================================================
You: 
```

Then you can start chatting!

The agent will:
- Retrieve memories from STM, LTM, and Reminders
- Build a prompt
- Send it to Claude on AWS Bedrock
- Save your conversation to STM for future use

---

## 🧠 How the Cognitive Loop Works

1. **User input** → embedded into a vector
2. **STM and LTM** → queried for relevant memories
3. **Reminders** → checked
4. **Prompt** → dynamically built (RAG style)
5. **Claude LLM** → invoked via AWS Bedrock
6. **Reply** → printed + conversation stored into STM
7. **Memory decay** → simulated over time

---

## 🛠 Key Improvements Over Earlier Versions

- 📚 **Short-Term Memory** now supports `query()` and `insert()`
- 📚 **Long-Term Memory** combines semantic memory retrieval
- 🧹 **Prompt Building** now safely handles dicts or strings
- 🚀 **Bedrock Integration** uses `invoke_model()` properly (no `.converse()`)
- 🧠 **Embedder Class** optimizes memory vector embedding
- 🔥 **CognitiveAgent** orchestrates everything cleanly
- 🧹 **Main CLI (`main.py`)** provides robust user interaction
- 📄 **Improved Documentation** (this README!)

---

## 📅 Future Enhancements (Roadmap)

- Dream-state consolidation of STM to LTM
- Fatigue, attention, and emotional state modeling
- Episodic + semantic memory blending
- Memory decay models based on time + relevance
- Context prioritization (importance scoring)
- Real-time sensory input buffering
- OpenSearch vector database for LTM (optional upgrade)
- Full web-based interactive UI (Phase 2)

---

## 🤝 Contributing

Contributions are welcome!  
Planned areas:
- Improving memory querying (semantic search)
- Better meta-cognition feedback loops
- Dream-state consolidation improvements
- More biologically-plausible memory decay models

---

## 📜 License

[MIT License](LICENSE)

---

## ✨ Credits

- Built with passion for human cognition, AI architecture, and open-source spirit.
- Thanks to AWS Bedrock, SentenceTransformers, and all contributors pushing the boundary of cognitive AI.

---

> "Memory isn't just storage. It's meaning. It's life." 🧠