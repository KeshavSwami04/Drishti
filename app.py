import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from search import search
from ingest import ingest_file, get_ingested_files, delete_file
import networkx as nx
import matplotlib.pyplot as plt
from ingest import ALL_CALLS
import builtins

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Drishti, a code assistant.

Rules:
- Only answer using provided context
- If unsure, say "I don't know"
- Always cite file and line number
- Be concise and accurate
"""

st.set_page_config(
    page_title="Drishti",
    page_icon="👁️",
    layout="wide"
)

# ---------------- STYLING (UNCHANGED) ----------------
st.markdown("""<style>/* KEEP YOUR EXISTING CSS HERE EXACTLY */</style>""", unsafe_allow_html=True)

st.markdown("# 👁️ Drishti")
st.markdown("<p style='color: #8b949e; margin-top: -16px;'>See inside your codebase. Ask anything.</p>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a code file", type=["py", "js", "ts", "java", "cpp", "c"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name
    if st.button("⬆ Ingest File"):
        count = ingest_file(filename, content)
        st.success(f"✓ Ingested {count} chunks from {filename}")

# ---------------- SIDEBAR ----------------
ingested_files = get_ingested_files()
if ingested_files:
    st.sidebar.markdown("### 📁 Ingested Files")
    st.sidebar.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)
    for f in ingested_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"📄 {f}")
        if col2.button("✕", key=f"delete_{f}"):
            delete_file(f)
            st.rerun()
else:
    st.sidebar.markdown("No files ingested yet.")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# ---------------- CHAT HISTORY ----------------
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption("Sources:")
            for source in message["sources"]:
                st.caption(f"📄 {source['filepath']} — line {source['start_line']}")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask anything about your code...")

# ---------------- GRAPH BUTTON (ADDED, SAFE) ----------------
import builtins

if st.button("Show Call Graph"):

    if not ALL_CALLS:
        st.warning("No Python files ingested yet.")
    else:
        G = nx.DiGraph()

        MAX_EDGES = 100

        for i, c in enumerate(ALL_CALLS):
            if i > MAX_EDGES:
                break

            if c["callee"] in dir(builtins):
                continue

            G.add_edge(c["caller"], c["callee"])

        pos = nx.kamada_kawai_layout(G)

        node_colors = []
        for node in G.nodes():
            if G.in_degree(node) == 0:
                node_colors.append("lightgreen")
            elif G.out_degree(node) == 0:
                node_colors.append("orange")
            else:
                node_colors.append("lightblue")

        plt.figure(figsize=(14, 10))

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2000,
            font_size=9,
            font_weight="bold"
        )

        st.pyplot(plt)

# ---------------- MAIN LOGIC ----------------
if user_input:

    # 🔹 Store user message (display)
    st.session_state.display_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 🔹 Retrieve chunks
    chunks = search(user_input)

    MAX_CONTEXT = 3000

    context = "\n\n".join([
        f"File: {c['filepath']} (line {c['start_line']}):\n{c['text']}"
        for c in chunks
    ])

    context = context[:MAX_CONTEXT]

    # 🔹 Build prompt
    augmented_prompt = f"""You are a helpful coding assistant called Drishti. 
Use the following code chunks to answer the user's question.
Always mention which file and line number the answer comes from.
Be concise and clear.

Relevant code:
{context}

User question: {user_input}"""

    # 🔹 Store for LLM history (FIXED)
    st.session_state.messages.append({
        "role": "user",
        "content": augmented_prompt
    })

    # 🔹 LLM CALL (FIXED LOCATION)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages
            ]
        )

        answer = response.choices[0].message.content

    except Exception:
        st.error("⚠️ LLM request failed. Please try again.")
        answer = "Error generating response"

    # 🔹 Sources
    sources = [
        {"filepath": c["filepath"], "start_line": c["start_line"]}
        for c in chunks
    ]

    # 🔹 Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # 🔹 Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption("Sources:")
        for source in sources:
            st.caption(f"📄 {source['filepath']} — line {source['start_line']}")