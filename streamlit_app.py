import time
import streamlit as st
from hybrid_search import get_bm25_retriever, hybrid_search
from answer_generator import generate_answer
from evaluate_retrival import BENCHMARK_QUERIES, evaluate_query, calculate_weighted_rrf

st.set_page_config(
    page_title="Neuro Mates RAG Studio",
    page_icon="🧠",
    layout="wide"
)

# Custom header
st.title("🧠 Neuro Mates RAG — Interactive Testing Dashboard")
st.caption("Test hybrid retrieval (BM25 + Qdrant Dense), answer generation, and recall metrics.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Retrieval Parameters")
    bm25_weight = st.slider("BM25 Weight", min_value=0.5, max_value=4.0, value=2.0, step=0.5)
    dense_weight = st.slider("Dense Weight", min_value=0.5, max_value=2.0, value=1.0, step=0.5)
    top_k = st.slider("Top Chunks to Retrieve", min_value=2, max_value=8, value=4, step=1)

    st.divider()
    st.subheader("💡 Sample Test Queries")
    sample_queries = [
        "How does Neuro Mates monitor Parkinson's disease?",
        "What hardware and location features are used for Alzheimer's patients?",
        "How does the Epilepsy module detect seizures and send alerts?",
        "What physiotherapy exercises and sensors are used for rehabilitation?",
        "Can Neuro Mates diagnose neurological conditions or replace a doctor?",
    ]

    selected_sample = st.selectbox("Choose a sample query:", ["-- Select a query --"] + sample_queries)

    st.divider()
    st.subheader("📊 System Diagnostics")
    run_benchmark = st.button("Run 5-Query Benchmark Suite", use_container_width=True)

# Cache retriever across Streamlit reruns
@st.cache_resource
def load_retriever():
    return get_bm25_retriever()

retriever = load_retriever()

# Handle Benchmark run
if run_benchmark:
    st.subheader("📈 Live Retrieval Benchmark Evaluation")
    total_recall = 0.0
    total_mrr = 0.0
    
    with st.spinner("Evaluating benchmark queries across knowledge base..."):
        for idx, item in enumerate(BENCHMARK_QUERIES, 1):
            q = item["question"]
            expected = item["expected"]
            chunks = hybrid_search(q, bm25_retriever=retriever, bm25_weight=bm25_weight, dense_weight=dense_weight, top_k=5)
            retrieved_ids = [c["id"] for c in chunks]
            
            recall, mrr, top_5 = evaluate_query(retrieved_ids, expected)
            total_recall += recall
            total_mrr += mrr
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**Q{idx}:** {q}")
                st.caption(f"Top 5 IDs: `{top_5}` | Expected: `{sorted(list(expected))}`")
            with col2:
                st.metric("Recall@5", f"{recall:.2f}")
            with col3:
                st.metric("MRR", f"{mrr:.2f}")
            st.divider()

        mean_recall = total_recall / len(BENCHMARK_QUERIES)
        mean_mrr = total_mrr / len(BENCHMARK_QUERIES)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Overall Mean Recall@5", f"{mean_recall:.2f}")
        m_col2.metric("Overall Mean MRR", f"{mean_mrr:.2f}")

# Main Chat / Query Interface
st.subheader("🔍 Ask Neuro Mates Knowledge Base")

default_text = selected_sample if selected_sample != "-- Select a query --" else ""
user_question = st.text_input("Enter your medical care or project question:", value=default_text)

if st.button("🚀 Run RAG Query", type="primary", use_container_width=False):
    if not user_question.strip():
        st.warning("Please enter a question to query.")
    else:
        # Step 1: Retrieval
        t_start = time.time()
        with st.spinner("Retrieving relevant knowledge chunks..."):
            chunks = hybrid_search(
                user_question,
                bm25_retriever=retriever,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                top_k=top_k
            )
        t_retrieval = time.time() - t_start

        # Step 2: Generation
        t_gen_start = time.time()
        with st.spinner("Generating grounded response using Gemini..."):
            answer = generate_answer(user_question, chunks)
        t_generation = time.time() - t_gen_start

        # Display tabs
        tab_answer, tab_chunks, tab_stats = st.tabs(["💬 Grounded Answer", "📑 Retrieved Chunks", "⏱️ Performance"])

        with tab_answer:
            st.markdown("### Answer")
            st.markdown(answer)

            st.markdown("#### Source Citations")
            source_cols = st.columns(len(chunks))
            for i, chunk in enumerate(chunks):
                source_cols[i].info(f"**{chunk['id']}**\n\n*{chunk.get('topic', 'N/A')}*")

        with tab_chunks:
            st.markdown(f"**Top {len(chunks)} chunks retrieved (Weights: BM25={bm25_weight}, Dense={dense_weight})**")
            for rank, chunk in enumerate(chunks, 1):
                with st.expander(f"Rank {rank}: [{chunk['id']}] {chunk.get('topic', '')} — Score: {chunk.get('rrf_score', 0):.5f}"):
                    st.write(chunk.get("content", ""))

        with tab_stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieval Latency", f"{t_retrieval:.2f}s")
            col2.metric("Generation Latency", f"{t_generation:.2f}s")
            col3.metric("Total Latency", f"{(t_retrieval + t_generation):.2f}s")

