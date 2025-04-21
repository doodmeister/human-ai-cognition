# 🧠 Human-Like AI Cognitive Architecture — Roadmap

This roadmap tracks key improvements and enhancements to align the architecture more closely with human cognitive models. Each task is categorized by module and priority.

---

## 🧩 Core Cognitive Modules

### 🔁 DPAD-Style RNN
- [ ] **Add training loop** for both behavior and residual prediction (cross-entropy/MSE loss)
- [ ] **Support nonlinear mappings** for all model elements (A′, K, Cy, Cz)
- [ ] Enable **adaptive latent dimension selection** based on behavioral salience
- [ ] Implement **flexible nonlinearity selection** as in original DPAD paper (auto model selection)

### 🧠 Short-Term Memory (STM)
- [ ] Fix `get_recent_items()` bug (`self.memory_data` reference)
- [ ] Implement **context chaining** (sequential episode links)
- [ ] Add **recency and attention decay scoring curve** logging
- [ ] Support **STM reheating** based on meta-cognitive boost

### 🗃️ Long-Term Memory (LTM)
- [ ] Add **contextual and temporal indexing** to episodic memory
- [ ] Implement **multi-hop semantic search** with concept chaining
- [ ] Support **emotion/motivation tagging** (optional metadata fields)
- [ ] Introduce **forgetting threshold** via attention + recency scores

### 🌙 Dream State Processor
- [ ] Allow optional **HDBSCAN clustering** for noise-tolerant memory compression
- [ ] Implement **reinforcement-based memory retention** using priority feedback
- [ ] Enable **replay-based consolidation** (e.g., latent state replay to retrain DPAD)
- [ ] Visualize STM → LTM transitions via PCA/t-SNE

### 🧭 Meta-Cognition Layer
- [ ] Replace linear fatigue decay with **sigmoid decay + recovery**
- [ ] Add **feedback loops** to boost STM entries during high salience
- [ ] Simulate **attention modulation** via task difficulty estimation
- [ ] Create a **strategy regulator** to prioritize perception, memory, or planning modules

---

## 🧠 Planning & Executive Function

- [ ] Introduce **Executive Planner Module** (simulates PFC)
  - SLM-based goal tracking and reasoning
  - Supports task breakdown and memory retrieval triggers
- [ ] Connect planner to LTM + STM to access relevant context
- [ ] Visualize goal stack and planned actions via dashboard

---

## 🧪 Evaluation & Testing

- [ ] Add unit tests for all modules with synthetic memory examples
- [ ] Create benchmarks for:
  - [ ] Recall fidelity (episodic vs semantic)
  - [ ] Attentional decay and recovery
  - [ ] Memory consolidation precision
- [ ] Simulate **biases** (recency, confirmation, salience-driven memory)
- [ ] Design **human-likeness scoring system** (Turing-style behavior metrics)

---

## 🧰 DevOps / Tooling

- [ ] Add GitHub Actions CI pipeline
  - Auto test on PRs
  - Code formatting (black, flake8)
- [ ] Generate module documentation with **MkDocs** or **Sphinx**
- [ ] Add API reference for Streamlit and cognitive loop methods

---

## 📊 Visualization & Interface

- [ ] Expand Streamlit dashboard:
  - [ ] STM memory map projection (PCA/UMAP)
  - [ ] Attention & fatigue over time
  - [ ] Trigger dream cycle manually
- [ ] Add **timeline view** of episodic memory formation
- [ ] Support input replay & memory path tracing

---

### 🔄 Update cadence
This roadmap will be reviewed and updated **monthly**. Contributions and suggestions welcome via Issues and PRs.

---

*Last updated: April 2025*

