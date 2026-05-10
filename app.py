import os
import logging
import builtins

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from groq import Groq

from search import search
from ingest import (
    ingest_file,
    get_ingested_files,
    delete_file,
    CALL_GRAPH_STORE
)

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Environment Setup
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in environment variables.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# LLM System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are Drishti, an AI-powered code intelligence assistant.

Rules:
- Only answer using the provided code context
- If the answer is not present in context, say "I don't know"
- Always cite file names and line numbers
- Be concise, accurate, and technical
"""

# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="Drishti",
    page_icon="👁️",
    layout="wide"
)

# =========================================================
# Batman x Apple — Glassy Dark UI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600;700&family=Bebas+Neue&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ───────────────────────────────────────── */

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── App Background — deep carbon with subtle noise ─────── */

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(30,30,40,0.9) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(10,10,20,0.95) 0%, transparent 55%),
        linear-gradient(160deg, #0a0a0f 0%, #111118 40%, #0d0d14 100%);
    color: #e8e8f0;
    min-height: 100vh;
}

/* Subtle grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* ── Gold accent line at top ────────────────────────────── */

.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        #c9a84c 20%,
        #f0d060 50%,
        #c9a84c 80%,
        transparent 100%
    );
    z-index: 9999;
}

/* ── Main container ─────────────────────────────────────── */

.block-container {
    padding-top: 2.5rem !important;
    max-width: 1280px !important;
    position: relative;
    z-index: 1;
}

/* ── Typography ─────────────────────────────────────────── */

.drishti-wordmark {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg,
        #c9a84c 0%,
        #f5e070 35%,
        #e8c84a 60%,
        #a07830 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    filter: drop-shadow(0 0 18px rgba(201,168,76,0.35));
    margin: 0;
}

.drishti-subtitle {
    font-size: 0.82rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-top: 6px;
    font-weight: 300;
}

/* ── Bat icon orb ───────────────────────────────────────── */

.eye-orb {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%,
        rgba(201,168,76,0.25) 0%,
        rgba(10,10,15,0.95) 70%
    );
    border: 1px solid rgba(201,168,76,0.4);
    box-shadow:
        0 0 0 1px rgba(201,168,76,0.12),
        0 0 30px rgba(201,168,76,0.2),
        inset 0 1px 0 rgba(255,255,255,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* ── Glass card mixin ───────────────────────────────────── */

.glass-card {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    box-shadow:
        0 4px 24px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Metric cards ───────────────────────────────────────── */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.032) !important;
    border: 1px solid rgba(201,168,76,0.15) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow:
        0 4px 20px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(201,168,76,0.08) !important;
    padding: 1.1rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="metric-container"]:hover {
    border-color: rgba(201,168,76,0.35) !important;
    box-shadow:
        0 4px 30px rgba(0,0,0,0.4),
        0 0 20px rgba(201,168,76,0.1),
        inset 0 1px 0 rgba(201,168,76,0.12) !important;
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #f0d060 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

/* ── Sidebar ────────────────────────────────────────────── */

section[data-testid="stSidebar"] {
    background: rgba(8,8,14,0.92) !important;
    border-right: 1px solid rgba(201,168,76,0.12) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Buttons ────────────────────────────────────────────── */

.stButton > button {
    background: linear-gradient(135deg,
        rgba(201,168,76,0.15) 0%,
        rgba(201,168,76,0.08) 100%
    ) !important;
    color: #f0d060 !important;
    border: 1px solid rgba(201,168,76,0.35) !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg,
        rgba(201,168,76,0.28) 0%,
        rgba(201,168,76,0.15) 100%
    ) !important;
    border-color: rgba(201,168,76,0.65) !important;
    transform: translateY(-2px) !important;
    box-shadow:
        0 8px 28px rgba(0,0,0,0.4),
        0 0 20px rgba(201,168,76,0.2) !important;
}

/* ── Chat messages ──────────────────────────────────────── */

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.028) !important;
    border: 1px solid rgba(255,255,255,0.065) !important;
    border-radius: 20px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
    box-shadow:
        0 2px 16px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: border-color 0.25s ease !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(201,168,76,0.18) !important;
}

/* User message accent */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 2px solid rgba(201,168,76,0.5) !important;
}

/* ── Chat input ─────────────────────────────────────────── */

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow:
        0 4px 20px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(201,168,76,0.5) !important;
    box-shadow:
        0 4px 30px rgba(0,0,0,0.4),
        0 0 20px rgba(201,168,76,0.12) !important;
}

[data-testid="stChatInput"] textarea {
    color: rgba(255,255,255,0.9) !important;
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.25) !important;
}

/* ── File uploader ──────────────────────────────────────── */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px dashed rgba(201,168,76,0.25) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(201,168,76,0.45) !important;
    box-shadow: 0 0 20px rgba(201,168,76,0.08) !important;
}

/* ── Code blocks ────────────────────────────────────────── */

pre, code {
    font-family: 'DM Mono', 'SF Mono', 'Fira Code', monospace !important;
}

pre {
    background: rgba(0,0,0,0.45) !important;
    border: 1px solid rgba(201,168,76,0.12) !important;
    border-radius: 14px !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* ── Divider ────────────────────────────────────────────── */

hr {
    border: none !important;
    border-top: 1px solid rgba(201,168,76,0.1) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(201,168,76,0.25);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(201,168,76,0.45);
}

/* ── Spinner ────────────────────────────────────────────── */

.stSpinner > div {
    border-top-color: #c9a84c !important;
}

/* ── Source caption ─────────────────────────────────────── */

.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(201,168,76,0.08);
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: rgba(201,168,76,0.8);
    font-family: 'DM Mono', monospace;
    margin: 3px 4px 3px 0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# Application Header
# =========================================================

st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 28px;
        padding: 20px 24px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(201,168,76,0.15);
        border-radius: 24px;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    ">
        <div class="eye-orb">👁️</div>
        <div>
            <h1 class="drishti-wordmark">Drishti</h1>
            <p class="drishti-subtitle">AI-powered code intelligence &nbsp;·&nbsp; understand any codebase</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# System Metrics
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Files Indexed", len(get_ingested_files()))

with col2:
    st.metric("Vector Store", "ChromaDB")

with col3:
    st.metric("LLM", "LLaMA 3.3")

# =========================================================
# Session State Initialization
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

MAX_HISTORY = 10

# =========================================================
# File Upload Section
# =========================================================

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a code file",
    type=["py", "js", "ts", "java", "cpp", "c"]
)

if uploaded_file is not None:

    try:
        content = uploaded_file.read().decode("utf-8")
        filename = uploaded_file.name

        if st.button("⬆ Ingest File"):

            with st.spinner("Analyzing and indexing codebase..."):
                count = ingest_file(filename, content)

            if count > 0:
                st.success(f"✓ Successfully ingested {count} chunks from {filename}")
                logger.info(f"Ingested file: {filename}")
            else:
                st.warning("No valid chunks were generated.")

    except UnicodeDecodeError:
        st.error("Unable to decode uploaded file.")

    except Exception as e:
        logger.exception("File ingestion failed")
        st.error(f"Ingestion failed: {str(e)}")

# =========================================================
# Sidebar — Ingested Files
# =========================================================

st.sidebar.markdown(
    """
    <div style="
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.4rem;
        letter-spacing: 0.15em;
        background: linear-gradient(135deg, #c9a84c, #f5e070, #c9a84c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
    ">👁️ &nbsp;Ingested Files</div>
    """,
    unsafe_allow_html=True
)

ingested_files = get_ingested_files()

if ingested_files:

    for file in ingested_files:

        col1, col2 = st.sidebar.columns([4, 1])

        col1.markdown(
            f'<span style="color:rgba(255,255,255,0.75); font-size:0.85rem;">📄 {file}</span>',
            unsafe_allow_html=True
        )

        if col2.button("✕", key=f"delete_{file}"):

            try:
                delete_file(file)
                logger.info(f"Deleted file: {file}")
                st.rerun()

            except Exception as e:
                logger.exception("File deletion failed")
                st.sidebar.error(f"Deletion failed: {str(e)}")

else:
    st.sidebar.markdown(
        '<p style="color:rgba(255,255,255,0.3); font-size:0.82rem; font-style:italic;">No files ingested yet.</p>',
        unsafe_allow_html=True
    )

# =========================================================
# Empty State
# =========================================================

if not ingested_files:
    st.markdown(
        """
        <div style='
            text-align: center;
            padding: 70px 40px;
            color: rgba(255,255,255,0.25);
            background: rgba(255,255,255,0.018);
            border: 1px dashed rgba(201,168,76,0.15);
            border-radius: 24px;
            margin: 20px 0;
        '>
            <div style="font-size:3rem; margin-bottom:16px;">👁️</div>
            <h2 style="
                font-family: Bebas Neue, Impact, sans-serif;
                font-size: 2rem;
                letter-spacing: 0.12em;
                color: rgba(201,168,76,0.5);
                margin-bottom: 10px;
            ">Upload Your First Repository</h2>
            <p style="font-size:0.9rem; color:rgba(255,255,255,0.25); max-width:400px; margin:0 auto;">
                Drishti will analyse structure, embeddings, dependencies,
                and semantic meaning — then answer anything you ask.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# Chat History Rendering
# =========================================================
# FIX: use st.markdown with unsafe_allow_html=True so that any HTML in
# assistant responses renders correctly instead of showing raw markup.

for message in st.session_state.display_messages:

    with st.chat_message(message["role"]):

        # Render markdown/HTML properly
        st.markdown(message["content"], unsafe_allow_html=True)

        if "sources" in message and message["sources"]:

            badges = "".join([
                f'<span class="source-badge">📄 {s["filepath"]} — line {s["start_line"]}</span>'
                for s in message["sources"]
            ])

            st.markdown(
                f'<div style="margin-top:10px;">{badges}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# Call Graph Visualization
# =========================================================

st.markdown("---")

if st.button("📊 Show Call Graph"):

    if not CALL_GRAPH_STORE:
        st.warning("No Python call graph data available.")

    else:

        try:

            G = nx.DiGraph()
            MAX_EDGES = 100
            all_calls = []

            for calls in CALL_GRAPH_STORE.values():
                all_calls.extend(calls)

            for i, call in enumerate(all_calls):

                if i >= MAX_EDGES:
                    break

                if call["callee"] in dir(builtins):
                    continue

                G.add_edge(call["caller"], call["callee"])

            pos = nx.kamada_kawai_layout(G)

            node_colors = []

            for node in G.nodes():
                if G.in_degree(node) == 0:
                    node_colors.append("#c9a84c")   # gold — entry points
                elif G.out_degree(node) == 0:
                    node_colors.append("#4a9eff")   # blue — leaf nodes
                else:
                    node_colors.append("#2a2a3e")   # dark — intermediate

            fig, ax = plt.subplots(figsize=(16, 12))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0d0d16")

            nx.draw(
                G, pos, ax=ax,
                with_labels=True,
                node_color=node_colors,
                node_size=3200,
                font_size=9,
                font_weight="bold",
                font_color="#f0f0f8",
                edge_color="rgba(201,168,76,0.3)",
                width=1.5,
                arrows=True,
                arrowsize=16,
            )

            st.pyplot(fig)

        except Exception as e:
            logger.exception("Graph rendering failed")
            st.error(f"Graph generation failed: {str(e)}")

# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input("Ask anything about your code...")

# =========================================================
# Main Chat Pipeline
# =========================================================

if user_input:

    st.session_state.display_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=True)

    try:

        chunks = search(user_input)

        if not chunks:
            st.warning("No relevant code context found.")
            st.stop()

        MAX_CHUNKS = 3
        context_chunks = chunks[:MAX_CHUNKS]

        context = "\n\n".join([
            f"File: {c['filepath']} (line {c['start_line']}):\n{c['text']}"
            for c in context_chunks
        ])

        augmented_prompt = f"""
Use the following code context to answer the question.

Relevant Code:
{context}

User Question:
{user_input}
"""

        st.session_state.messages.append({
            "role": "user",
            "content": augmented_prompt
        })

        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        with st.spinner("Analysing codebase..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *st.session_state.messages
                ]
            )

        answer = response.choices[0].message.content

    except Exception as e:

        logger.exception("LLM request failed")
        st.error("⚠️ Failed to generate response.")
        answer = "Error generating response."
        chunks = []

    sources = [
        {"filepath": c["filepath"], "start_line": c["start_line"]}
        for c in chunks
    ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    with st.chat_message("assistant"):

        # FIX: render with markdown so HTML in responses displays correctly
        st.markdown(answer, unsafe_allow_html=True)

        if sources:

            badges = "".join([
                f'<span class="source-badge">📄 {s["filepath"]} — line {s["start_line"]}</span>'
                for s in sources
            ])

            st.markdown(
                f'<div style="margin-top:10px;">{badges}</div>',
                unsafe_allow_html=True
            )

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style='
        text-align: center;
        color: rgba(255,255,255,0.2);
        padding-bottom: 24px;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
    '>
        👁️ &nbsp; Built with Streamlit · ChromaDB · LLaMA 3.3
        &nbsp;·&nbsp;
        <span style="color:rgba(201,168,76,0.5);">Drishti</span>
    </div>
    """,
    unsafe_allow_html=True
)