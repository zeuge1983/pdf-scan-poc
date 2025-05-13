import streamlit as st
import os
import time
from datetime import datetime
import json
from engine import load_or_build_index, get_query_engine, DOC_DIR, GOOGLE_API_KEY, CHROMA_PERSIST_DIR

# --- Page Configuration ---
st.set_page_config(page_title="Doc Q&A with Gemini & LlamaIndex", layout="wide")

st.title("📄 Document Q&A with Gemini Pro 1.5 & LlamaIndex")
st.caption("Upload PDF or Markdown documents, build an index, and ask questions in natural language.")

# --- API Key Check ---
if not GOOGLE_API_KEY:
    st.error("🔴 Google API Key not found. Please create a .env file in the project root and add your GOOGLE_API_KEY.")
    st.info("Example .env file content:\nGOOGLE_API_KEY=\"YOUR_ACTUAL_API_KEY_HERE\"")
    st.stop()
else:
    st.sidebar.success("✅ Google API Key loaded.")

# --- Session State Initialization ---
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = set()
if "index_built_successfully" not in st.session_state:
    st.session_state.index_built_successfully = False
if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = None


# --- Sidebar for Document Upload and Indexing ---
with st.sidebar:
    st.header("⚙️ Setup & Document Management")

    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, MD)",
        type=["pdf", "md"],
        accept_multiple_files=True,
        help="Upload one or more PDF or Markdown files."
    )

    if uploaded_files:
        new_files_to_process_paths = []
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(DOC_DIR, uploaded_file.name)
            if uploaded_file.name not in st.session_state.processed_file_names:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_to_process_paths.append(file_path)
                st.session_state.processed_file_names.add(uploaded_file.name)
                progress_bar.progress((i + 1) / len(uploaded_files))
                st.success(f"Saved '{uploaded_file.name}'")
            else:
                st.info(f"'{uploaded_file.name}' was already uploaded.")

    col1, col2 = st.columns(2)
    with col1:
        build_index = st.button(
            "🔨 Build/Update Index",
            disabled=not st.session_state.processed_file_names and not os.listdir(DOC_DIR),
            help="Process uploaded documents and build the search index"
        )
    with col2:
        clear_data = st.button(
            "🗑️ Clear All Data",
            help="Reset the application and remove all uploaded documents"
        )

    if build_index:
        all_doc_paths = [os.path.join(DOC_DIR, fname) for fname in os.listdir(DOC_DIR)
                         if os.path.isfile(os.path.join(DOC_DIR, fname))]

        if all_doc_paths:
            with st.spinner("Building index..."):
                progress_bar = st.progress(0)
                try:
                    index = load_or_build_index(all_doc_paths)
                    if index:
                        st.session_state.query_engine = get_query_engine(index)
                        st.session_state.index_built_successfully = True
                        progress_bar.progress(1.0)
                        st.success("🎉 Index built successfully!")
                        st.session_state.messages = []
                    else:
                        st.error("Failed to build index.")
                        st.session_state.index_built_successfully = False
                except Exception as e:
                    st.error(f"Error building index: {e}")
                    st.session_state.index_built_successfully = False
        else:
            st.warning("No documents found. Please upload documents first.")

    if clear_data:
        if st.session_state.messages:  # Offer to export chat before clearing
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            st.download_button(
                "📥 Export Chat History Before Clearing",
                data=json.dumps(chat_export, indent=2),
                file_name="chat_history.json",
                mime="application/json"
            )

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        if os.path.exists(DOC_DIR):
            for filename in os.listdir(DOC_DIR):
                file_path = os.path.join(DOC_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f'Failed to delete {file_path}. Reason: {e}')

        if os.path.exists(CHROMA_PERSIST_DIR):
            import shutil

            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
            except Exception as e:
                st.error(f"Failed to delete ChromaDB store. Reason: {e}")

        st.success("Application reset complete.")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📚 Processed Documents")
    if st.session_state.processed_file_names:
        for name in sorted(list(st.session_state.processed_file_names)):
            st.markdown(f"- `{name}`")
    else:
        st.markdown("_No documents processed yet._")

# --- Main Chat Interface ---
st.header("💬 Ask Your Questions")

# Chat controls
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗑️ Clear Chat", disabled=not st.session_state.messages):
        st.session_state.messages = []
        if st.session_state.query_engine and hasattr(st.session_state.query_engine, 'chat_memory'):
            st.session_state.query_engine.chat_memory.clear()
        st.rerun()

with col2:
    if st.button("💾 Export Chat", disabled=not st.session_state.messages):
        chat_export = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        st.download_button(
            "📥 Download Chat History",
            data=json.dumps(chat_export, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )

if not st.session_state.index_built_successfully:
    st.info("ℹ️ Please upload documents and build the index to start asking questions.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            # Display confidence score
            confidence = message["metadata"].get("confidence_score", 0)
            st.progress(confidence, text=f"Confidence: {confidence:.2%}")

# Chat input and response handling
if prompt := st.chat_input("Ask a question...", disabled=not st.session_state.index_built_successfully):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        try:
            if st.session_state.query_engine:
                response = st.session_state.query_engine.query(prompt)

                # Stream the response
                for chunk in response.response_gen:
                    full_response_content += chunk
                    message_placeholder.markdown(full_response_content + "▌")
                message_placeholder.markdown(full_response_content)

                # Store and display metadata
                metadata = getattr(response, 'metadata', {})

                # Display confidence score
                if confidence := metadata.get('confidence_score'):
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                # Display sources
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("📚 Sources & References", expanded=False):
                        for i, node in enumerate(response.source_nodes, 1):
                            st.markdown(f"**Source {i}** (Relevance: {node.score:.2f})")
                            st.caption(f"From: `{node.metadata.get('file_name', 'Unknown')}`")
                            st.markdown("---")
                            st.markdown("**Content:**")
                            st.markdown(node.get_content(metadata_mode="all"))
                            st.markdown("---")

                # Store the response with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_content,
                    "metadata": metadata
                })
                
            else:
                st.error("Query engine not available. Please build the index first.")

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")