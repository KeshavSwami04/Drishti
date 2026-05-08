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

# Validate API key before app starts
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in environment variables.")
    st.stop()

# Initialize Groq client
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
# Custom Styling
# =========================================================

st.markdown("""
<style>

/* Add your production CSS here */
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

# Validate API key before app starts
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in environment variables.")
    st.stop()

# Initialize Groq client
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
# Premium UI Styling
# =========================================================

st.markdown("""
<style>

/* =========================================================
   Global Styling
========================================================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================================================
   Main App Background
========================================================= */

.stApp {

    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #020617 100%
    );

    color: white;
}

/* =========================================================
   Main Container
========================================================= */

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* =========================================================
   Title Styling
========================================================= */

.main-title {

    font-size: 3.2rem;

    font-weight: 800;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #a78bfa
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0;
}

.subtitle {

    color: #94a3b8;

    margin-top: -10px;

    margin-bottom: 2rem;

    font-size: 1rem;
}

/* =========================================================
   Sidebar
========================================================= */

section[data-testid="stSidebar"] {

    background: rgba(15, 23, 42, 0.95);

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
   Buttons
========================================================= */

.stButton > button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 0.6rem 1.2rem;

    font-weight: 600;

    transition: 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.4);
}

/* =========================================================
   Chat Cards
========================================================= */

[data-testid="stChatMessage"] {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 14px;

    margin-bottom: 14px;

    backdrop-filter: blur(14px);
}

/* =========================================================
   File Uploader
========================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.03);

    border: 1px dashed rgba(255,255,255,0.12);

    border-radius: 16px;

    padding: 1rem;
}

/* =========================================================
   Metrics Cards
========================================================= */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    padding: 1rem;

    border-radius: 18px;

    backdrop-filter: blur(12px);
}

/* =========================================================
   Code Blocks
========================================================= */

pre {

    border-radius: 16px !important;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
   Scrollbar
========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {

    background: #334155;

    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# Application Header
# =========================================================

st.markdown(
    """
    <div>
        <h1 class="main-title">👁️ Drishti</h1>

        <p class="subtitle">
            AI-powered code intelligence for understanding unfamiliar codebases.
        </p>
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

                st.success(
                    f"✓ Successfully ingested {count} chunks from {filename}"
                )

                logger.info(f"Ingested file: {filename}")

            else:
                st.warning("No valid chunks were generated.")

    except UnicodeDecodeError:
        st.error("Unable to decode uploaded file.")

    except Exception as e:

        logger.exception("File ingestion failed")

        st.error(f"Ingestion failed: {str(e)}")

# =========================================================
# Sidebar - Ingested Files
# =========================================================

st.sidebar.markdown("## 📁 Ingested Files")

ingested_files = get_ingested_files()

if ingested_files:

    for file in ingested_files:

        col1, col2 = st.sidebar.columns([4, 1])

        col1.markdown(f"📄 {file}")

        if col2.button("✕", key=f"delete_{file}"):

            try:
                delete_file(file)

                logger.info(f"Deleted file: {file}")

                st.rerun()

            except Exception as e:

                logger.exception("File deletion failed")

                st.sidebar.error(
                    f"Deletion failed: {str(e)}"
                )

else:
    st.sidebar.info("No files ingested yet.")

# =========================================================
# Empty State
# =========================================================

if not ingested_files:

    st.markdown(
        """
        <div style='
            text-align:center;
            padding:60px;
            color:#94a3b8;
        '>

            <h2>🚀 Upload your first repository</h2>

            <p>
                Drishti will analyze structure, embeddings,
                dependencies, and semantic meaning.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# Chat History Rendering
# =========================================================

for message in st.session_state.display_messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if "sources" in message:

            st.caption("Sources:")

            for source in message["sources"]:

                st.caption(
                    f"📄 {source['filepath']} "
                    f"— line {source['start_line']}"
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

                G.add_edge(
                    call["caller"],
                    call["callee"]
                )

            pos = nx.kamada_kawai_layout(G)

            node_colors = []

            for node in G.nodes():

                if G.in_degree(node) == 0:
                    node_colors.append("lightgreen")

                elif G.out_degree(node) == 0:
                    node_colors.append("orange")

                else:
                    node_colors.append("lightblue")

            plt.figure(
                figsize=(16, 12),
                facecolor="#0f172a"
            )

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=node_colors,
                node_size=3000,
                font_size=10,
                font_weight="bold",
                edge_color="#64748b",
                width=2
            )

            st.pyplot(plt)

        except Exception as e:

            logger.exception("Graph rendering failed")

            st.error(
                f"Graph generation failed: {str(e)}"
            )

# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "Ask anything about your code..."
)

# =========================================================
# Main Chat Pipeline
# =========================================================

if user_input:

    st.session_state.display_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    try:

        # -------------------------------------------------
        # Semantic Retrieval
        # -------------------------------------------------

        chunks = search(user_input)

        if not chunks:

            st.warning("No relevant code context found.")

            st.stop()

        # -------------------------------------------------
        # Context Construction
        # -------------------------------------------------

        MAX_CHUNKS = 3

        context_chunks = chunks[:MAX_CHUNKS]

        context = "\n\n".join([
            f"File: {c['filepath']} "
            f"(line {c['start_line']}):\n{c['text']}"
            for c in context_chunks
        ])

        # -------------------------------------------------
        # Prompt Construction
        # -------------------------------------------------

        augmented_prompt = f"""
Use the following code context to answer the question.

Relevant Code:
{context}

User Question:
{user_input}
"""

        # -------------------------------------------------
        # Store Message History
        # -------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": augmented_prompt
        })

        st.session_state.messages = (
            st.session_state.messages[-MAX_HISTORY:]
        )

        # -------------------------------------------------
        # LLM Request
        # -------------------------------------------------

        with st.spinner("Analyzing codebase..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *st.session_state.messages
                ]
            )

        answer = response.choices[0].message.content

    except Exception as e:

        logger.exception("LLM request failed")

        st.error("⚠️ Failed to generate response.")

        answer = "Error generating response."

        chunks = []

    # -----------------------------------------------------
    # Source Attribution
    # -----------------------------------------------------

    sources = [
        {
            "filepath": c["filepath"],
            "start_line": c["start_line"]
        }
        for c in chunks
    ]

    # -----------------------------------------------------
    # Store Assistant Response
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # -----------------------------------------------------
    # Render Assistant Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

        if sources:

            st.caption("Sources:")

            for source in sources:

                st.caption(
                    f"📄 {source['filepath']} "
                    f"— line {source['start_line']}"
                )

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style='
        text-align:center;
        color:#64748b;
        padding-bottom:20px;
    '>

        Built with ❤️ using Streamlit, ChromaDB, and LLaMA 3.3

    </div>
    """,
    unsafe_allow_html=True
)
</style>
""", unsafe_allow_html=True)

# =========================================================
# Application Header
# =========================================================

st.markdown("# 👁️ Drishti")

st.markdown(
    """
    <p style='color: #8b949e; margin-top: -16px;'>
        AI-powered code intelligence for understanding unfamiliar codebases.
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Session State Initialization
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# Limit chat history to avoid memory growth
MAX_HISTORY = 10

# =========================================================
# File Upload Section
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a code file",
    type=["py", "js", "ts", "java", "cpp", "c"]
)

if uploaded_file is not None:

    try:
        content = uploaded_file.read().decode("utf-8")
        filename = uploaded_file.name

        if st.button("⬆ Ingest File"):

            with st.spinner("Ingesting file..."):

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
# Sidebar - Ingested Files
# =========================================================

st.sidebar.markdown("## 📁 Ingested Files")

ingested_files = get_ingested_files()

if ingested_files:

    for file in ingested_files:

        col1, col2 = st.sidebar.columns([4, 1])

        col1.markdown(f"📄 {file}")

        if col2.button("✕", key=f"delete_{file}"):

            try:
                delete_file(file)

                logger.info(f"Deleted file: {file}")

                st.rerun()

            except Exception as e:
                logger.exception("File deletion failed")
                st.sidebar.error(f"Deletion failed: {str(e)}")

else:
    st.sidebar.info("No files ingested yet.")

# =========================================================
# Chat History Rendering
# =========================================================

for message in st.session_state.display_messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if "sources" in message:

            st.caption("Sources:")

            for source in message["sources"]:

                st.caption(
                    f"📄 {source['filepath']} — line {source['start_line']}"
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

            # Flatten all stored call relationships
            all_calls = []

            for calls in CALL_GRAPH_STORE.values():
                all_calls.extend(calls)

            # Build graph edges
            for i, call in enumerate(all_calls):

                if i >= MAX_EDGES:
                    break

                # Ignore Python built-in functions
                if call["callee"] in dir(builtins):
                    continue

                G.add_edge(
                    call["caller"],
                    call["callee"]
                )

            # Generate graph layout
            pos = nx.kamada_kawai_layout(G)

            # Dynamic node coloring
            node_colors = []

            for node in G.nodes():

                if G.in_degree(node) == 0:
                    node_colors.append("lightgreen")

                elif G.out_degree(node) == 0:
                    node_colors.append("orange")

                else:
                    node_colors.append("lightblue")

            # Create graph figure
            plt.figure(figsize=(14, 10))

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=node_colors,
                node_size=2200,
                font_size=9,
                font_weight="bold"
            )

            st.pyplot(plt)

        except Exception as e:
            logger.exception("Graph rendering failed")
            st.error(f"Graph generation failed: {str(e)}")

# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "Ask anything about your code..."
)

# =========================================================
# Main Chat Pipeline
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Store user message
    # -----------------------------------------------------

    st.session_state.display_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    try:

        # -------------------------------------------------
        # Semantic Retrieval
        # -------------------------------------------------

        chunks = search(user_input)

        if not chunks:
            st.warning("No relevant code context found.")
            st.stop()

        # -------------------------------------------------
        # Context Construction
        # -------------------------------------------------

        MAX_CHUNKS = 3

        context_chunks = chunks[:MAX_CHUNKS]

        context = "\n\n".join([
            f"File: {c['filepath']} (line {c['start_line']}):\n{c['text']}"
            for c in context_chunks
        ])

        # -------------------------------------------------
        # Prompt Construction
        # -------------------------------------------------

        augmented_prompt = f"""
Use the following code context to answer the question.

Relevant Code:
{context}

User Question:
{user_input}
"""

        # -------------------------------------------------
        # Store Message History
        # -------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": augmented_prompt
        })

        # Prevent unbounded session growth
        st.session_state.messages = (
            st.session_state.messages[-MAX_HISTORY:]
        )

        # -------------------------------------------------
        # LLM Request
        # -------------------------------------------------

        with st.spinner("Generating response..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *st.session_state.messages
                ]
            )

        answer = response.choices[0].message.content

    except Exception as e:

        logger.exception("LLM request failed")

        st.error("⚠️ Failed to generate response.")

        answer = "Error generating response."

        chunks = []

    # -----------------------------------------------------
    # Source Attribution
    # -----------------------------------------------------

    sources = [
        {
            "filepath": c["filepath"],
            "start_line": c["start_line"]
        }
        for c in chunks
    ]

    # -----------------------------------------------------
    # Store Assistant Response
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # -----------------------------------------------------
    # Render Assistant Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

        if sources:

            st.caption("Sources:")

            for source in sources:

                st.caption(
                    f"📄 {source['filepath']} — line {source['start_line']}"
                )
