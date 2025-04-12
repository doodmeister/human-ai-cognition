
# 🧠 System Design Document: Human-AI Cognition Platform

## Overview
This document outlines the system architecture, components, and logic of the Human-AI Cognition platform, including memory structures, meta-cognitive control, and attention modeling.

## 1. Modules

### 1.1 Ingestion Layer
- **Input**: Files (txt, pdf, mp3, mp4)
- **Tech**: S3 + Lambda + AWS Textract, Transcribe, Comprehend
- **Output**: Parsed data → Sensory Buffer

### 1.2 Sensory Buffer
- Evaluates incoming inputs with entropy score
- Only high-value data forwarded to STM

### 1.3 Short-Term Memory (STM)
- Stored in OpenSearch (STM domain)
- Contains embeddings + metadata
- Decays over time unless consolidated

### 1.4 Meta-Cognition Engine
- Bedrock LLM used to simulate internal reflection
- Prompt templates evaluate fatigue, attention, and retention
- Scored memories are either forgotten or pushed to LTM

### 1.5 Dream-State Processor
- Scheduled via EventBridge (nightly)
- Pulls recent STM, runs meta-cognition, updates LTM

### 1.6 Long-Term Memory (LTM)
- Stored in OpenSearch LTM domain
- Only retained and relevant memories
- Queryable and explorable via dashboard or API

## 2. Attention & Fatigue Modeling
- **Attention**: Scores assigned based on recency, relevance, frequency
- **Fatigue**: Simulated using load + STM volatility
- **Control Loop**: Meta-cognition adjusts focus or rest logic

## 3. Observability
- CloudWatch logs and dashboard monitor:
  - Lambda invocations
  - Uploads
  - Consolidation runs
- Streamlit dashboard visualizes:
  - Memory text
  - Attention score
  - Reason for retention

## 4. Deployment
- Terraform used to provision all infrastructure
- Bedrock model ID is configurable
- Streamlit and notebooks are locally executable

## 5. Extensibility
- Add more input modalities (chat, webcam, live docs)
- Introduce memory reinforcement via reward tagging
- Create an agent loop that uses LLM for real-time control
