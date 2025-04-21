import time
from sentence_transformers import SentenceTransformer

from short_term_memory_opensearch import ShortTermMemoryOpenSearch
from meta_cognition import MetaCognition
from dream_state import process_dream_state  # assuming you have this function
from user_input_simulation import get_next_input  # replace with real input logic

# --- Config ---
DREAM_INTERVAL = 20  # cycles before triggering dream
CYCLE_SLEEP = 2      # seconds between cognitive cycles
VECTOR_DIM = 768

# --- Initialize ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

stm = ShortTermMemoryOpenSearch(
    index_name="humanai-stm",
    host="localhost",
    port=9200,
    vector_dim=VECTOR_DIM
)

meta = MetaCognition()
cycle_counter = 0

print("🧠 Starting cognitive loop...")

while True:
    # --- Sensory Input ---
    raw_input = get_next_input()  # This could be user text, file content, etc.
    if not raw_input:
        print("💤 No new input... idling.")
        time.sleep(CYCLE_SLEEP)
        continue

    # --- Embedding ---
    embedding = embedding_model.encode(raw_input).tolist()

    # --- Priority / Emotion Estimation (placeholder logic) ---
    base_priority = 1.0
    emotion_score = 0.3  # could be dynamically estimated
    tags = ["dialogue"]

    # --- Memory Encoding ---
    stm.add_entry(
        content=raw_input,
        embedding=embedding,
        base_priority=base_priority,
        emotion=emotion_score,
        tags=tags
    )

    print(f"💾 Encoded new input: {raw_input[:60]}...")

    # --- Memory Retrieval ---
    top_entries = stm.get_top_entries(query_embedding=embedding, k=3)
    print("🧠 Retrieved similar memories:")
    for e in top_entries:
        print(f"   ↪ {e['content'][:60]}... (priority={e['priority']:.2f})")

    # --- Meta-Cognition / Fatigue ---
    fatigue = meta.update(attention_delta=-0.03)
    print(f"⚙️ Attention Level: {meta.attention:.2f} | Fatigue: {fatigue}")

    # --- Dream Trigger ---
    cycle_counter += 1
    if cycle_counter % DREAM_INTERVAL == 0 or fatigue:
        print("🌙 Entering dream state...")
        entries = stm.dump_all()
        process_dream_state(entries)  # your existing consolidation logic
        stm.prune_low_priority(threshold=0.15)
        meta.reset_attention()

    time.sleep(CYCLE_SLEEP)
