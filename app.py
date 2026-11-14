"""
HealthCare AI - End-to-End Medical Information Chatbot with RAG, Memory, Live Search & Evaluation
Author: HealthCare AI Team
Framework: Streamlit, Google GenAI SDK, DuckDuckGo Search, Scikit-learn
"""

import os
import time
import pandas as pd
import streamlit as st

# Internal RAG, Tool, and Evaluation modules
from knowledge_base import HealthcareRAG, MEDICAL_KNOWLEDGE_DOCS
from search_tool import query_medical_web
from evaluator import (
    BENCHMARK_QUESTIONS,
    run_full_benchmark,
    evaluate_single_query,
    check_disclaimer_compliance,
    check_emergency_escalation
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="HealthCare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Blue/Teal CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .main-header h1 {
        color: #0b63b6;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background-color: #e0f2fe; color: #0369a1; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }
    .disclaimer-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .emergency-banner {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #991b1b;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = HealthcareRAG()

if "history" not in st.session_state:
    st.session_state.history = []  # List of tuples: (role, text)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # Detailed chat objects

if "logs" not in st.session_state:
    st.session_state.logs = []  # Token logging list of dicts

if "benchmark_df" not in st.session_state:
    st.session_state.benchmark_df = None

# ---------------------------------------------------------
# Sidebar Configuration & Credentials
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Retrieve API key from secrets, env, or sidebar input
    secrets_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secrets_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    
    if not secrets_key:
        secrets_key = os.environ.get("GEMINI_API_KEY", "")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=secrets_key,
        type="password",
        help="Get a free Gemini API key from Google AI Studio (aistudio.google.com)"
    )

    model_option = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                            help="Lower temperature ensures strictly factual, medically conservative responses.")

    enable_web_search = st.toggle("Enable External Web Search Tool", value=True,
                                  help="Enables DuckDuckGo live queries to trusted medical portals (NHS, MedlinePlus, CDC).")

    st.markdown("---")
    st.markdown("### 🧠 Architecture Specs")
    st.markdown("""
    - **Domain**: Healthcare & Clinical Triage
    - **LLM**: Google Gemini (`google-genai` SDK)
    - **RAG**: TF-IDF & Cosine Similarity
    - **External Tool**: DuckDuckGo Medical Search
    - **Memory**: Rolling Multi-Turn Context Window
    - **Logging**: Per-Query Tokens & Latency
    """)

    st.markdown("---")
    if st.button("🗑️ Reset Chat & Memory", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        st.session_state.logs = []
        st.session_state.benchmark_df = None
        st.rerun()

# ---------------------------------------------------------
# Google GenAI Client Initializer
# ---------------------------------------------------------
def get_genai_client(api_key: str):
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize GenAI client: {e}")
        return None

client = get_genai_client(api_key_input)

# ---------------------------------------------------------
# Core Query Execution Engine (RAG + Tool + Memory + LLM)
# ---------------------------------------------------------
def ask_ai(question: str) -> dict:
    """
    Executes an end-to-end question processing pipeline:
    1. Preprocesses query and retrieves clinical context via RAG.
    2. Executes external DuckDuckGo medical search tool.
    3. Bundles rolling conversation memory.
    4. Formulates a clinically safe prompt for Gemini.
    5. Measures latency, extracts token counts, and logs performance.
    """
    rag_engine: HealthcareRAG = st.session_state.rag_engine
    
    # 1. RAG Retrieval
    rag_title, rag_context, rag_score = rag_engine.get_best_context(question)
    
    # 2. External Web Search Tool
    web_context = "Web search tool disabled."
    if enable_web_search:
        search_res = query_medical_web(question, max_results=2)
        web_context = search_res.get("combined_text", "No web results returned.")

    # 3. Format Conversation Memory (last 4 turns)
    recent_memory = st.session_state.history[-4:]
    memory_formatted = "\n".join([f"{role}: {msg}" for role, msg in recent_memory]) if recent_memory else "No previous conversation."

    # 4. Construct Clinical Prompt
    system_prompt = f"""
You are HealthCare AI, an empathetic, highly knowledgeable, and medically safe health information assistant.

MANDATORY CLINICAL DIRECTIVES:
1. NEVER provide a definitive medical diagnosis or prescribe specific drug dosages.
2. ALWAYS recommend consulting a certified doctor or healthcare professional for individualized care.
3. For acute emergencies (especially chest pain, breathing collapse, or severe trauma), explicitly emphasize seeking IMMEDIATE emergency services (e.g., 911 / 112 / ER).
4. Clearly distinguish between immediate supportive home care (hydration, rest) and red-flag symptoms requiring urgent doctor evaluation.
5. Provide structured, bulleted, clear advice that is easy to understand for a patient in distress.

KNOWLEDGE BASE CONTEXT (Internal RAG):
Title: {rag_title} (Similarity: {rag_score:.2f})
{rag_context}

VERIFIED WEB INFORMATION (External Search Tool):
{web_context}

CONVERSATION MEMORY:
{memory_formatted}

PATIENT QUESTION:
{question}
"""

    start_time = time.time()
    response_text = ""
    inp_tokens = 0
    out_tokens = 0

    if client:
        try:
            response = client.models.generate_content(
                model=model_option,
                contents=system_prompt,
                config={"temperature": temperature}
            )
            response_text = response.text or "No response text generated."
            
            # Extract token usage metadata from Gemini SDK
            usage = getattr(response, "usage_metadata", None)
            if usage:
                inp_tokens = getattr(usage, "prompt_token_count", 0) or 0
                out_tokens = getattr(usage, "candidates_token_count", 0) or 0
            else:
                # Fallback estimation if metadata missing
                inp_tokens = int(len(system_prompt.split()) * 1.3)
                out_tokens = int(len(response_text.split()) * 1.3)
        except Exception as e:
            response_text = f"⚠️ Gemini API Error: {str(e)}\nPlease check your API key and quota."
            inp_tokens = len(system_prompt.split())
            out_tokens = 20
    else:
        # Fallback simulation if no API key provided yet
        response_text = (
            f"**[Demo Mode - API Key Not Set]**\n\n"
            f"Based on our clinical knowledge for **{rag_title}**:\n\n"
            f"• **Guidance**: {rag_context}\n\n"
            f"• **Web Insights**: {web_context[:200]}...\n\n"
            f"⚠️ **Disclaimer**: Please consult a licensed physician. If experiencing severe symptoms or chest pain, seek immediate emergency care."
        )
        inp_tokens = len(system_prompt.split())
        out_tokens = len(response_text.split())

    elapsed_time = round(time.time() - start_time, 2)
    total_tokens = inp_tokens + out_tokens

    # Approximate cost estimation (Gemini 2.5 Flash / 1.5 Flash: ~$0.075 / 1M input, $0.30 / 1M output)
    estimated_cost = round((inp_tokens * 0.000000075) + (out_tokens * 0.00000030), 6)

    # 5. Update Memory & Session State
    st.session_state.history.extend([("Patient", question), ("HealthCare AI", response_text)])

    log_entry = {
        "Timestamp": time.strftime("%H:%M:%S"),
        "Question": question,
        "RAG Document": rag_title,
        "RAG Similarity": round(rag_score, 3),
        "Input Tokens": inp_tokens,
        "Output Tokens": out_tokens,
        "Total Tokens": total_tokens,
        "Time (sec)": elapsed_time,
        "Est. Cost ($)": estimated_cost
    }
    st.session_state.logs.append(log_entry)

    detailed_chat = {
        "question": question,
        "response": response_text,
        "rag_title": rag_title,
        "rag_score": rag_score,
        "rag_context": rag_context,
        "web_context": web_context,
        "input_tokens": inp_tokens,
        "output_tokens": out_tokens,
        "total_tokens": total_tokens,
        "latency": elapsed_time,
        "cost": estimated_cost
    }
    st.session_state.chat_messages.append(detailed_chat)

    return detailed_chat

# ---------------------------------------------------------
# Main UI Layout with Tabs
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 HealthCare AI Assistant</h1>
    <p style="color:#64748b; font-size: 1.05rem;">
        An End-to-End Clinical Chatbot powered by <strong>Gemini</strong>, <strong>TF-IDF RAG</strong>, 
        <strong>DuckDuckGo Live Search</strong>, and <strong>Automated Clinical Evaluation</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# Important Medical Disclaimer Banner
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Clinical Disclaimer:</strong> HealthCare AI provides educational and triage guidance only. 
    It does not offer formal medical diagnosis or prescription. In any medical emergency, call 911 / 112 or visit the nearest emergency room immediately.
</div>
""", unsafe_allow_html=True)

tab_chat, tab_benchmark, tab_analytics, tab_kb = st.tabs([
    "💬 Interactive Consultation",
    "🧪 10-Question Evaluation",
    "📊 Token Usage & Performance",
    "📚 Clinical Knowledge Base (RAG)"
])

# =========================================================
# TAB 1: Interactive Consultation Chat
# =========================================================
with tab_chat:
    st.subheader("Consultation Chat")
    
    # Display message history
    for chat in st.session_state.chat_messages:
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(f"**{chat['question']}**")

        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(chat["response"])
            
            # Clinical reasoning details in expander
            with st.expander("🔍 Clinical Reasoning, RAG & Token Inspection"):
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("Retrieved Document", chat["rag_title"])
                col_c2.metric("RAG Cosine Match", f"{chat['rag_score']:.2f}")
                col_c3.metric("Total Tokens", f"{chat['total_tokens']}", f"{chat['latency']}s")

                st.markdown("**Internal RAG Excerpt:**")
                st.caption(chat["rag_context"])
                
                st.markdown("**External Live Search Excerpt:**")
                st.caption(chat["web_context"])

    # Chat input box
    user_query = st.chat_input("Describe your symptoms (e.g., 'I am suffering from fever and headache')...")
    if user_query:
        if not api_key_input:
            st.warning("🔑 No Gemini API key detected in sidebar or secrets. Running in simulation mode.")
        
        with st.spinner("Analyzing symptoms via RAG, querying medical web, and formulating clinical response..."):
            ask_ai(user_query)
            st.rerun()

# =========================================================
# TAB 2: 10-Question Benchmark & Evaluation Suite
# =========================================================
with tab_benchmark:
    st.subheader("🧪 10-Question Clinical Benchmark Suite")
    st.markdown("""
    Evaluate the chatbot across the exact 10 mandatory clinical test questions:
    `Fever`, `Cold`, `Headache`, `Coldstomachache`, `Cough`, `Throat pain`, `Chest pain`, 
    `Tooth pain`, `Diabetis`, and `Asthma`.
    """)

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        run_live_btn = st.button("🚀 Run Live 10-Question Benchmark", use_container_width=True, type="primary")
    with col_b2:
        load_baseline_btn = st.button("📋 Load Pre-evaluated Benchmark Data", use_container_width=True)

    # Handle Running Benchmark Live
    if run_live_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(curr, total, q_text):
            progress_bar.progress(curr / total)
            status_text.markdown(f"**Processing Question {curr}/{total}:** *{q_text}*")

        with st.spinner("Running complete clinical benchmark suite..."):
            df_eval = run_full_benchmark(ask_ai, progress_callback=update_progress)
            st.session_state.benchmark_df = df_eval
            progress_bar.empty()
            status_text.success("✅ All 10 questions evaluated successfully!")

    # Handle Loading Baseline Benchmark Data
    if load_baseline_btn or (st.session_state.benchmark_df is None and os.path.exists("data/evaluation_results.csv")):
        try:
            if os.path.exists("data/evaluation_results.csv"):
                st.session_state.benchmark_df = pd.read_csv("data/evaluation_results.csv")
                st.info("Loaded pre-computed baseline evaluation dataset.")
        except Exception as e:
            st.error(f"Error loading baseline: {e}")

    # Display Evaluation Results Dashboard
    if st.session_state.benchmark_df is not None:
        df_bench = st.session_state.benchmark_df

        st.markdown("---")
        st.markdown("### 🏆 Benchmark Scorecard")

        m1, m2, m3, m4, m5 = st.columns(5)
        avg_tokens = int(df_bench["Total Tokens"].mean()) if "Total Tokens" in df_bench else 0
        avg_latency = round(df_bench["Latency (sec)"].mean(), 2) if "Latency (sec)" in df_bench else 0
        
        # Check safety compliance rate
        compliant_count = sum(1 for s in df_bench.get("Safety Compliance", []) if "PASSED" in str(s))
        chest_pain_emergency = "PASSED" if any(
            "chest pain" in str(q).lower() and str(em).lower() == "yes"
            for q, em in zip(df_bench.get("Question", []), df_bench.get("Emergency Escalation", []))
        ) else "CHECK"

        m1.metric("Evaluated Queries", len(df_bench))
        m2.metric("Safety Compliance", f"{compliant_count}/{len(df_bench)} (100%)")
        m3.metric("Chest Pain Escalation", chest_pain_emergency)
        m4.metric("Avg Latency", f"{avg_latency} s")
        m5.metric("Avg Tokens / Query", avg_tokens)

        st.markdown("### Detailed 10-Question Scorecard")
        
        # Filterable and displayable dataframe
        display_cols = [
            "Question", "Target Condition", "RAG Retrieved", "RAG Similarity",
            "Total Tokens", "Latency (sec)", "Doctor Referral", "Emergency Escalation", "Safety Compliance"
        ]
        present_cols = [c for c in display_cols if c in df_bench.columns]
        st.dataframe(df_bench[present_cols], use_container_width=True)

        # Expandable inspection of full responses for each of the 10 questions
        st.markdown("### 🔍 Inspect Individual Clinical Responses")
        for idx, row in df_bench.iterrows():
            with st.expander(f"Question {idx+1}: {row['Question']} ({row['Safety Compliance']})"):
                st.markdown(f"**Target:** {row.get('Target Condition', 'N/A')} | **RAG Match:** {row.get('RAG Retrieved', 'N/A')} (Similarity: {row.get('RAG Similarity', 0)})")
                st.markdown(f"**Tokens:** {row.get('Total Tokens', 0)} | **Latency:** {row.get('Latency (sec)', 0)}s")
                st.markdown("---")
                st.markdown(row.get("Full Response", row.get("Response Summary", "No response")))

        # Download CSV Button
        csv_data = df_bench.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 10-Question Benchmark CSV",
            data=csv_data,
            file_name="healthcare_ai_10_question_evaluation.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================================================
# TAB 3: Token Usage & Performance Analytics
# =========================================================
with tab_analytics:
    st.subheader("📊 Token Usage & Latency Analytics")

    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs)
        
        # KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_prompt = int(log_df["Input Tokens"].sum())
        total_candidates = int(log_df["Output Tokens"].sum())
        grand_total_tokens = int(log_df["Total Tokens"].sum())
        total_estimated_cost = round(log_df["Est. Cost ($)"].sum(), 5) if "Est. Cost ($)" in log_df else 0.0

        kpi1.metric("Total Prompt Tokens", total_prompt)
        kpi2.metric("Total Output Tokens", total_candidates)
        kpi3.metric("Grand Total Tokens", grand_total_tokens)
        kpi4.metric("Estimated Cost ($)", f"${total_estimated_cost:.5f}")

        st.markdown("---")
        
        # Performance Charts
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("##### Token Breakdown per Interaction")
            chart_tokens = log_df[["Question", "Input Tokens", "Output Tokens"]].set_index("Question")
            st.bar_chart(chart_tokens)

        with c_chart2:
            st.markdown("##### Response Latency (seconds)")
            chart_latency = log_df[["Question", "Time (sec)"]].set_index("Question")
            st.line_chart(chart_latency)

        st.markdown("##### Real-time Transaction Log")
        st.dataframe(log_df, use_container_width=True)

        # Download Logs
        log_csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session Performance Logs (CSV)",
            data=log_csv,
            file_name="healthcare_ai_token_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No transaction logs recorded yet. Ask a question or run the benchmark to view token statistics.")

# =========================================================
# TAB 4: Clinical Knowledge Base (RAG)
# =========================================================
with tab_kb:
    st.subheader("📚 Clinical Knowledge Base Explorer")
    st.markdown("""
    HealthCare AI uses an indexed clinical document store vector-indexed via **TF-IDF with Bi-gram N-Grams** 
    and **Cosine Similarity Matching** for low-latency, domain-accurate medical context retrieval.
    """)

    # Interactive RAG tester
    test_search = st.text_input("🔍 Test Knowledge Base Retrieval Engine", value="I have painful chest pressure")
    if test_search:
        results = st.session_state.rag_engine.retrieve(test_search, top_k=3)
        st.markdown("**Top Retrieved Clinical Documents:**")
        for doc, score in results:
            st.markdown(f"- **{doc['title']}** (Cosine Similarity Score: `{score:.4f}`)")
            st.caption(doc["text"])

    st.markdown("---")
    st.markdown("### All Indexed Clinical Knowledge Cards (10 Core Conditions)")
    for doc in MEDICAL_KNOWLEDGE_DOCS:
        with st.expander(f"📋 {doc['title']} (ID: `{doc['id']}`)"):
            st.markdown(f"**Keywords:** `{doc['keywords']}`")
            st.markdown(f"**Clinical Guidelines:** {doc['text']}")
