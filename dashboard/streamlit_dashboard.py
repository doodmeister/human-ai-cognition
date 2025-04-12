
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Human-AI Cognition Dashboard")
st.title("🧠 Human-Like AI Memory Dashboard")

# Placeholder endpoints — replace with your actual OpenSearch endpoints
STM_ENDPOINT = "https://your-stm-domain-name.us-east-1.es.amazonaws.com/humanai_stm_index/_search"
LTM_ENDPOINT = "https://your-ltm-domain-name.us-east-1.es.amazonaws.com/humanai_ltm_index/_search"

HEADERS = {
    "Content-Type": "application/json"
}

# Optional: Enable IAM-auth (requires requests-aws4auth and Boto session)
# from requests_aws4auth import AWS4Auth
# import boto3
# session = boto3.Session()
# credentials = session.get_credentials().get_frozen_credentials()
# region = session.region_name
# auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

def fetch_memories(endpoint, top_n=50):
    query = {
        "size": top_n,
        "query": {"match_all": {}},
        "sort": [{"timestamp": {"order": "desc"}}]
    }
    res = requests.get(endpoint, headers=HEADERS, data=json.dumps(query))  # , auth=auth if using IAM
    hits = res.json().get("hits", {}).get("hits", [])
    return [hit["_source"] for hit in hits]

# Sidebar selection
source_select = st.sidebar.radio("Memory Source", ["Short-Term Memory (STM)", "Long-Term Memory (LTM)"])
endpoint = STM_ENDPOINT if "STM" in source_select else LTM_ENDPOINT

st.sidebar.markdown("📥 Pulling recent memories...")

# Load data
memory_items = fetch_memories(endpoint)
if memory_items:
    df = pd.DataFrame(memory_items)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    # Chart: Attention Score over Time
    st.subheader("📈 Attention Scores Over Time")
    fig = px.line(df, x="timestamp", y="attention_score", text="text", markers=True,
                  title="Memory Attention Scores")
    st.plotly_chart(fig, use_container_width=True)

    # Memory Cards
    st.subheader("🧠 Recent Memory Items")
    for idx, row in df.iterrows():
        st.markdown(f"**Time:** {row['timestamp']}")
        st.markdown(f"**Memory:** {row['text']}")
        st.markdown(f"**Attention Score:** {row.get('attention_score', 'N/A')}")
        st.markdown(f"**Reason:** {row.get('reason', 'N/A')}")
        st.markdown("---")
else:
    st.warning("No memory data retrieved.")
