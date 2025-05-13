# app.py
import streamlit as st
import os
import time
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
    # Small visual confirmation that the key is loaded (doesn't validate the key itself)
    st.sidebar.success("✅ Google API Key loaded.")

# --- Session State Initialization ---
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # To store chat history
if "processed_file_names" not in st.session_state:
    # Stores names of files already processed to avoid re-saving and to list them.
    st.session_state.processed_file_names = set()
if "index_built_successfully" not in st.session_state:
    st.session_state.index_built_successfully = False


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
        # Ensure DOC_DIR exists (it should be created by engine.py, but good to double-check)
        os.makedirs(DOC_DIR, exist_ok=True)

        for uploaded_file in uploaded_files:
            # Save file to DOC_DIR if it's new
            file_path = os.path.join(DOC_DIR, uploaded_file.name)
            if uploaded_file.name not in st.session_state.processed_file_names:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_to_process_paths.append(file_path)
                st.session_state.processed_file_names.add(uploaded_file.name)
                st.success(f"Saved '{uploaded_file.name}'")
            else:
                st.info(f"'{uploaded_file.name}' was already uploaded.")

        if not new_files_to_process_paths and st.session_state.processed_file_names:
             st.info("All selected files are already processed. Click 'Build/Update Index' if you want to re-index.")


    if st.button("🔨 Build/Update Index", disabled=not st.session_state.processed_file_names and not os.listdir(DOC_DIR)):
        # We need paths of *all* documents in DOC_DIR to build a comprehensive index,
        # or at least those the user intends to query.
        # For this PoC, we'll use all files currently in DOC_DIR.
        all_doc_paths_in_doc_dir = [os.path.join(DOC_DIR, fname) for fname in os.listdir(DOC_DIR) if os.path.isfile(os.path.join(DOC_DIR, fname))]

        if all_doc_paths_in_doc_dir:
            with st.spinner("Building index... This may take a few moments. Please wait."):
                try:
                    index = load_or_build_index(all_doc_paths_in_doc_dir)
                    if index:
                        st.session_state.query_engine = get_query_engine(index)
                        st.session_state.index_built_successfully = True
                        st.success("🎉 Index built/updated successfully!")
                        st.session_state.messages = [] # Clear chat for new context
                    else:
                        st.error("🔴 Failed to build index. No documents were loaded or an error occurred.")
                        st.session_state.index_built_successfully = False
                except Exception as e:
                    st.error(f"🔴 Error building index: {e}")
                    st.session_state.index_built_successfully = False
        else:
            st.warning("⚠️ No documents found in the upload directory. Please upload documents first.")

    st.markdown("---")
    st.markdown("#### Currently Processed Files:")
    if st.session_state.processed_file_names:
        for name in sorted(list(st.session_state.processed_file_names)):
            st.markdown(f"- `{name}`")
    else:
        st.markdown("_No files uploaded in this session yet._")
    st.caption(f"Files are stored in: `{DOC_DIR}`")
    st.caption(f"Vector database is stored in: `{CHROMA_PERSIST_DIR}`")


    if st.button("🔄 Reset Application & Clear Data"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Delete files in uploaded_docs
        if os.path.exists(DOC_DIR):
            for filename in os.listdir(DOC_DIR):
                file_path = os.path.join(DOC_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f'Failed to delete {file_path}. Reason: {e}')

        # Delete ChromaDB store (BE CAREFUL WITH THIS IN PRODUCTION)
        # For ChromaDB PersistentClient, deleting the directory is the way to clear it.
        if os.path.exists(CHROMA_PERSIST_DIR):
            import shutil
            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
                st.success("ChromaDB store cleared.")
            except Exception as e:
                st.error(f"Failed to delete ChromaDB store directory. Reason: {e}")

        st.success("Application reset. Upload new documents or reload the page.")
        time.sleep(2) # Give time for the user to see a message
        st.rerun()


# --- Attempt to load an existing index on startup ---
if not st.session_state.query_engine and not uploaded_files: # Only if no engine and no new files uploaded in this session
    # Check if DOC_DIR has files from a previous session
    existing_doc_files = [os.path.join(DOC_DIR, fname) for fname in os.listdir(DOC_DIR) if os.path.isfile(os.path.join(DOC_DIR, fname))]
    if existing_doc_files:
        st.session_state.processed_file_names.update([os.path.basename(p) for p in existing_doc_files])

    if os.path.exists(CHROMA_PERSIST_DIR) and any(os.scandir(CHROMA_PERSIST_DIR)): # Check if Chroma dir exists and is not empty
        with st.spinner("Attempting to load existing index from ChromaDB..."):
            try:
                # Pass the empty list to signal loading existing index without adding new files yet
                index = load_or_build_index([])
                if index:
                    st.session_state.query_engine = get_query_engine(index)
                    st.session_state.index_built_successfully = True
                    st.sidebar.success("✅ Existing index loaded from ChromaDB.")
                else:
                    st.sidebar.info("No valid existing index found in ChromaDB or it's empty.")
            except Exception as e:
                st.sidebar.warning(f"Could not auto-load index: {e}")
    else:
        st.sidebar.info("No existing ChromaDB store found. Upload documents to build an index.")


# --- Main Chat Interface ---
st.header("💬 Ask Your Questions")

if not st.session_state.index_built_successfully:
    st.info("ℹ️ Please upload documents and build the index using the sidebar to enable Q&A.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.index_built_successfully):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        try:
            if st.session_state.query_engine:
                with st.spinner("Thinking..."):
                    streaming_response = st.session_state.query_engine.query(prompt)

                    # Stream the response
                    for chunk in streaming_response.response_gen:
                        full_response_content += chunk
                        message_placeholder.markdown(full_response_content + "▌")
                    message_placeholder.markdown(full_response_content) # Final response without cursor

                # Display source nodes if available
                if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, node in enumerate(streaming_response.source_nodes):
                            file_name = node.metadata.get('file_name', 'N/A')
                            # Truncate long node content for display
                            content_preview = node.get_content(metadata_mode="all")[:500] + "..." \
                                if len(node.get_content(metadata_mode="all")) > 500 \
                                else node.get_content(metadata_mode="all")

                            st.markdown(f"**Source {i+1} (Score: {node.score:.2f})**")
                            st.caption(f"File: `{file_name}`")
                            st.markdown(f"Content: _{content_preview}_")
                            st.divider()
            else:
                full_response_content = "🔴 Query engine not available. Please build the index."
                message_placeholder.markdown(full_response_content)

        except Exception as e:
            full_response_content = f"🔴 Error during query: {str(e)}"
            st.error(full_response_content)
            # Log the full traceback to the console for debugging
            import traceback
            print(f"Error during query: {traceback.format_exc()}")


    st.session_state.messages.append({"role": "assistant", "content": full_response_content})   