"""Streamlit UI for the AI HR Policy Agent.

Provides two views:
    1. Employee Chat — Ask questions about company policies
    2. Admin Panel — Upload and manage HR policy documents

Run with: streamlit run src/ui/app.py
"""

import requests
import streamlit as st

# --- Configuration ---
API_URL = "http://localhost:8000"


def get_api_url():
    """Get API URL from environment or default."""
    import os
    return os.getenv("API_URL", API_URL)


# --- Page Config ---
st.set_page_config(
    page_title="AI HR Policy Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    .user-message {
        background: rgba(102, 126, 234, 0.15);
        border-left: 4px solid #667eea;
    }
    .bot-message {
        background: rgba(118, 75, 162, 0.15);
        border-left: 4px solid #764ba2;
    }
    .source-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{get_api_url()}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏢 AI HR Policy Agent</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem;">
            Ask questions about company policies — powered by RAG & Gemini
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the employee chat interface."""
    st.subheader("💬 Ask a Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-message bot-message">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Show sources if available
            if msg.get("sources"):
                sources_html = " ".join(
                    f'<span class="source-badge">📄 {s["document"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(sources_html, unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("e.g., How many vacation days do I get?")

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})

        # Query the API
        with st.spinner("Searching policies..."):
            try:
                response = requests.post(
                    f"{get_api_url()}/query",
                    json={"question": question},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "metrics": data.get("metrics", {}),
                    })
                else:
                    st.error(f"API Error: {response.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to the API server. Is it running?")

        st.rerun()


def render_admin_panel():
    """Render the admin document upload panel."""
    st.subheader("📁 Document Management")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload HR Policy Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files containing your HR policies.",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{get_api_url()}/upload",
                        files=files,
                        timeout=120,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"✅ {data['filename']} — {data['chunks_created']} chunks created"
                        )
                    else:
                        st.error(f"❌ Failed to upload {uploaded_file.name}: {response.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to the API server.")

    # Knowledge base stats
    st.divider()
    st.subheader("📊 Knowledge Base Stats")
    try:
        response = requests.get(f"{get_api_url()}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Collection", stats["collection_name"])
            with col2:
                st.metric("Documents Indexed", stats["document_count"])
    except requests.ConnectionError:
        st.warning("Cannot fetch stats — API server not connected.")


def render_metrics_sidebar():
    """Show latest query metrics in the sidebar."""
    if st.session_state.get("messages"):
        last_bot_msg = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            None,
        )
        if last_bot_msg and last_bot_msg.get("metrics"):
            metrics = last_bot_msg["metrics"]
            st.sidebar.markdown("### 📈 Last Query Metrics")
            st.sidebar.metric("Latency", f"{metrics.get('latency_ms', 0):.0f} ms")
            st.sidebar.metric("Tokens Used", metrics.get("tokens_used", "N/A"))
            st.sidebar.metric(
                "Chunks",
                f"{metrics.get('chunks_retrieved', 0)} → {metrics.get('chunks_after_rerank', 0)}",
            )
            st.sidebar.metric("Prompt Version", metrics.get("prompt_version", "N/A"))


# --- Main App ---
def main():
    render_header()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["💬 Chat", "📁 Admin Panel"])

    # API status
    api_status = check_api_health()
    st.sidebar.markdown("---")
    if api_status:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.caption("Start the API server with:\n`uvicorn src.api.server:app`")

    # Render metrics
    render_metrics_sidebar()

    # Page routing
    if page == "💬 Chat":
        render_chat_interface()
    elif page == "📁 Admin Panel":
        render_admin_panel()


if __name__ == "__main__":
    main()
