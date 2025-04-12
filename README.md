
# 🧠 Human-Like AI Cognitive Architecture (Deployment Guide)

This document walks you through deploying and testing your full-stack Human-AI system using AWS and local tools.

---

## 🔧 AWS Infrastructure Setup (via Terraform)

### 1. Initialize Terraform
```bash
terraform init
terraform plan
terraform apply
```

### 2. Verify Deployments:
- S3 Bucket: `humanai-document-store`
- Lambda Functions:
  - `ingest_processor`
  - `dream_consolidator`
- OpenSearch Domains:
  - STM: `humanai-stm-store`
  - LTM: `humanai-ltm-store`
- CloudWatch Logs & Dashboard

---

## 📥 Ingest Data

Use `upload_test.py` to simulate file uploads:
```bash
python upload_test.py
```

Or manually upload:
```bash
aws s3 cp ./test_files/sample_doc.txt s3://humanai-document-store/
```

---

## 🌙 Nightly Dream State (Memory Consolidation)

- Triggered by EventBridge every night at 3AM UTC.
- Pulls from STM, runs prompt through Bedrock LLM, writes retained memories to LTM.

---

## 📊 Monitor & Debug

### AWS Console:
- Go to **CloudWatch > Dashboards**
- Import `cloudwatch_dashboard.json` for pre-built monitoring

### Logs:
- Check `/aws/lambda/ingest_processor`
- Check `/aws/lambda/dream_consolidator`

---

## 🧠 Explore Memory Data

### OpenSearch Index Explorer:
```bash
jupyter notebook explore_memory.ipynb
```

### Web Dashboard (Streamlit):
```bash
streamlit run streamlit_dashboard.py
```

---

## 📜 LLM Meta-Cognitive Prompts

Stored in: `prompts/metacognition_prompts.txt`

Used in:
- Bedrock prompt for dream state
- Scoring attention, fatigue, relevance

---

## ✅ Next Steps
- Integrate real Bedrock invoke_model call
- Use IAM auth for OpenSearch (secure)
- Add search/filter to Streamlit
- Automate weekly memory pruning (simulate forgetting)
