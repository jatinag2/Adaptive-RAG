import streamlit as st
import os
import tempfile
from graph import app 
from ingest import load_and_chunk_pdf, embed_and_store
from config import settings

MAX_FILE_SIZE_MB = 10

# --- Configuration & Layout ---
st.set_page_config(page_title="Adaptive RAG Agent", layout="centered", initial_sidebar_state="expanded")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = "No document loaded"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set() 

# --- Helper Function: On-the-fly Ingestion ---
def process_uploaded_file(uploaded_file):
    """Saves the uploaded file temporarily, chunks it, and adds it to ChromaDB."""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File size ({file_size_mb:.2f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit.")
        return False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        progress_bar = st.progress(0)
        st.text("Loading PDF...")
        chunks = load_and_chunk_pdf(temp_path)
        progress_bar.progress(50)
        
        st.text(f"Embedding and storing {len(chunks)} chunks...")
        if not chunks:
           raise ValueError("No text chunks were created from this PDF.")
        
        embed_and_store(chunks)
        progress_bar.progress(100)
        
        return True
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Sidebar: Knowledge Base Management ---
with st.sidebar:
    st.markdown("###  Knowledge Base")
    st.caption("Upload documents to securely update the agent's memory.")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner(f"Ingesting `{uploaded_file.name}`..."):
            success = process_uploaded_file(uploaded_file)
            if success:
                st.session_state.processed_files.add(uploaded_file.name)
                st.session_state.current_doc = uploaded_file.name
                st.toast("Document successfully ingested! 🎉", icon="✅")
                st.rerun()

    st.divider()
    st.markdown("**System Settings**")
    st.caption("Vector DB: `Chroma`")
    st.caption(f"Embedding: `{settings.embedding_model}`")
    st.caption(f"Model: `{settings.llm_model}`")
    st.caption(f"Max Iterations: `{settings.max_iterations}`")

# --- UI Styling (CSS) ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .header-wrapper { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .title-wrapper { display: flex; align-items: center; gap: 12px; }
    .main-title { font-size: 1.8rem; font-weight: 700; margin: 0; color: #111827; }
    .subtitle { color: #6B7280; font-size: 1.05rem; margin-top: -5px; margin-bottom: 20px; }
    .doc-status { display: inline-flex; align-items: center; gap: 6px; background-color: #F3F4F6; color: #4B5563; padding: 4px 12px; border-radius: 16px; font-size: 0.85rem; font-weight: 500; border: 1px solid #E5E7EB; margin-bottom: 30px; }
    [data-testid="stChatInput"] { background-color: #FFFFFF !important; border: 1px solid #E5E7EB !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important; border-radius: 12px !important; padding-right: 5px !important; }
    [data-testid="stChatInputSubmitButton"] { background-color: #1E3A8A !important; color: white !important; border-radius: 8px !important; }
    [data-testid="stChatInputSubmitButton"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# --- Header & Dynamic Document Badge ---
st.markdown(f"""
<div class="header-wrapper">
    <div class="title-wrapper">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#111827" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
        </svg>
        <h1 class="main-title">Adaptive RAG Agent</h1>
    </div>
</div>
<p class="subtitle">Ask questions based on the ingested documents. The agent will route, grade, and self-correct.</p>
<div class="doc-status">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
    </svg>
    <span>Currently referencing: <strong>{st.session_state.current_doc}</strong></span>
</div>
""", unsafe_allow_html=True)

# --- Empty State & Suggested Prompts ---
active_prompt = None

if len(st.session_state.messages) == 0:
    st.write("##### Get started")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Summarize the document", use_container_width=True): active_prompt = "Can you provide a comprehensive summary of the referenced document?"
    with col2:
        if st.button("🔑 What are the key points?", use_container_width=True): active_prompt = "What are the core methodologies or key points discussed in the text?"
    with col3:
        if st.button("🌐 Search latest news", use_container_width=True): active_prompt = "Search the web for the latest news regarding this topic."
            
    st.markdown("<br>", unsafe_allow_html=True) 

# --- Chat Rendering ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Handling ---
chat_bar_input = st.chat_input("Ask a question about the document...")
if chat_bar_input: active_prompt = chat_bar_input

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"): st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.status("Agent is thinking...", expanded=True) as status_box:
            initial_state = {"question": active_prompt, "search_count": 0}
            final_generation = ""
            final_sources = []
            
            # Execute the LangGraph workflow
            for output in app.stream(initial_state):
                for node_name, state_data in output.items():
                    st.write(f"🔄 Executed Node: **{node_name}**")
                    if "generation" in state_data:
                        final_generation = state_data["generation"]
                    if "sources" in state_data:
                        final_sources = state_data["sources"]
            
            status_box.update(label="Response generated!", state="complete", expanded=False)
        
        st.markdown(final_generation)
        
        # Display Sources
        if final_sources:
            st.markdown("---")
            st.markdown("**Sources:**")
            for source in final_sources:
                st.markdown(f"- `{source}`")
                
        # Append full content to session state
        message_content = final_generation
        if final_sources:
            sources_text = "\n\n**Sources:**\n" + "\n".join([f"- `{s}`" for s in final_sources])
            message_content += sources_text
            
        st.session_state.messages.append({"role": "assistant", "content": message_content})
        st.rerun()
