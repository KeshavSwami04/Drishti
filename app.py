import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from search import search
from ingest import ingest_file, get_ingested_files, delete_file

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Drishti",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.stApp {
    background-color: #0d1117;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.stChatMessage {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

.stChatInputContainer {
    border-top: 1px solid #30363d;
    background-color: #0d1117;
    padding-top: 12px;
}

.stTextInput > div > div > input {
    background-color: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}

.stButton > button {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}

.stFileUploader {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 12px;
    padding: 8px;
}

.stSuccess {
    background-color: #0f2d1a;
    border: 1px solid #1a7f37;
    border-radius: 8px;
    color: #3fb950;
}

code {
    font-family: 'JetBrains Mono', monospace;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 6px;
    color: #79c0ff;
}

h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.5px;
}

.stCaption {
    color: #8b949e;
    font-size: 12px;
}

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #0d1117;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 👁️ Drishti")
st.markdown("<p style='color: #8b949e; margin-top: -16px; margin-bottom: 24px; font-size: 14px;'>See inside your codebase. Ask anything.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a code file", type=["py", "js", "ts", "java", "cpp", "c"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name
    if st.button("⬆ Ingest File"):
        count = ingest_file(filename, content)
        st.success(f"✓ Ingested {count} chunks from {filename}")

ingested_files = get_ingested_files()
if ingested_files:
    st.sidebar.markdown("### 📁 Ingested Files")
    st.sidebar.markdown("<hr style='border-color: #30363d; margin: 8px 0;'>", unsafe_allow_html=True)
    for f in ingested_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"<span style='font-size: 13px; color: #e6edf3;'>📄 {f}</span>", unsafe_allow_html=True)
        if col2.button("✕", key=f"delete_{f}"):
            delete_file(f)
            st.rerun()
else:
    st.sidebar.markdown("<p style='color: #8b949e; font-size: 13px;'>No files ingested yet.</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption("Sources:")
            for source in message["sources"]:
                st.caption(f"📄 {source['filepath']} — line {source['start_line']}")

user_input = st.chat_input("Ask anything about your code...")

if user_input:
    chunks = search(user_input)

    context = "\n\n".join([
        f"File: {c['filepath']} (line {c['start_line']}):\n{c['text']}"
        for c in chunks
    ])

    augmented_prompt = f"""You are a helpful coding assistant called Drishti. 
Use the following code chunks to answer the user's question.
Always mention which file and line number the answer comes from.
Be concise and clear.

Relevant code:
{context}

User question: {user_input}"""

    st.session_state.messages.append({"role": "user", "content": augmented_prompt})
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=st.session_state.messages
    )

    answer = response.choices[0].message.content
    sources = [{"filepath": c["filepath"], "start_line": c["start_line"]} for c in chunks]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption("Sources:")
        for source in sources:
            st.caption(f"📄 {source['filepath']} — line {source['start_line']}")