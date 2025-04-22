# Human-AI Cognition

**Where Memory Meets Meaning**

This project simulates coordinated, human-like cognition by integrating multi-modal sensory ingestion, adaptive attention mechanisms, and memory filtering inspired by human short- and long-term memory systems.

## Features
- Real-time ingestion of video, audio, text, and sensor data from public APIs
- Short-Term Memory module with decaying relevance
- Meta-cognitive layer that monitors fatigue, novelty, and attention levels
- Factorized self-attention mechanism for efficient multi-modal event prioritization
- DPAD transformer model that learns what should be remembered
- Long-Term Memory consolidation via AWS OpenSearch (or simulated reflection)
- Fully deployable on AWS using Terraform

## Architecture
[video/api] --> [video_stream.py] --+ [audio/api] --> [audio_stream.py] --+--> [ingest_processor] --> [STM] [text/api] --> [text_stream.py] --+ [sensor/api]-> [sensor_stream.py] --+ | [MetaCognition] | [DPAD Transformer] | [Dream Consolidator] | [OpenSearch (LTM)]


## Getting Started

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/doodmeister/human-ai-cognition.git
   cd human-ai-cognition

2. Setup the environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prerequisites
- AWS CLI and Terraform installed
- AWS credentials configured
- API keys for:
  - [NASA API](https://api.nasa.gov/)
  - [NewsAPI](https://newsapi.org/)

### Deployment Steps
```bash
terraform init
terraform apply

## Quickstart

```python
from model.dpad_transformer import DPADTransformer

model = DPADTransformer(hidden_size=256)
dummy_input = torch.randn(1, 10, 256)
output = model(dummy_input)
print(output.shape)  # -> (1, 10, 256)


Environment Variables
Set the following environment variables in your AWS Lambda console or local .env:

S3_BUCKET

NASA_API_KEY

NEWS_API_KEY

Modules
lambda/video_stream.py: NASA video feed -> S3

lambda/audio_stream.py: LibriVox audio -> Transcribe -> S3

lambda/text_stream.py: News headlines -> S3

lambda/sensor_stream.py: Weather data -> S3

model/dpad_transformer.py: Multi-modal attention + LSTM classifier

memory/: Memory storage and decay management

metacognition/: Tracks cognitive attention and fatigue

Documentation
docs/design.md: Architecture and attention design

docs/api_usage.md: How each API works and is integrated

docs/architecture.png: System overview diagram

License
MIT
