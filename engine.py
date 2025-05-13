from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Configuration Constants
CONFIG = {
    'DEFAULT_EMBEDDING': 'gemini',
    'GEMINI_MODEL': 'models/gemini-1.5-flash-latest',
    'EMBEDDING_MODEL': 'models/embedding-001',
    'COLLECTION_NAME': 'qna_collection'
}


# Export GOOGLE_API_KEY for compatibility
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Directory Configuration
BASE_DIR = Path(__file__).parent
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db_store"
DOC_DIR = BASE_DIR / "uploaded_docs"


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class SettingsInitializationError(Exception):
    """Custom exception for settings initialization errors."""
    pass


@dataclass
class SettingsManager:
    """Manages LlamaIndex settings and configurations."""
    api_key: str
    embedding_choice: str = CONFIG['DEFAULT_EMBEDDING']

    def initialize(self) -> bool:
        if not self.api_key:
            raise SettingsInitializationError("Missing Google API key")

        Settings.llm = Gemini(
            api_key=self.api_key,
            model_name=CONFIG['GEMINI_MODEL']
        )

        if self.embedding_choice == "gemini":
            Settings.embed_model = GeminiEmbedding(
                api_key=self.api_key,
                model_name=CONFIG['EMBEDDING_MODEL']
            )
        else:
            raise ValueError(f"Unsupported embedding choice: {self.embedding_choice}")
        return True


class VectorStoreManager:
    """Manages vector store operations and indexing."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.db = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.db.get_or_create_collection(CONFIG['COLLECTION_NAME'])
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self.vector_store)

    def create_index(self, doc_paths: Optional[List[str]] = None) -> Optional[VectorStoreIndex]:
        if not doc_paths:
            return self._load_existing_index()

        try:
            documents = self._load_documents(doc_paths)
            if not documents:
                return self._load_existing_index()

            return VectorStoreIndex.from_documents(
                documents,
                storage_context=self.get_storage_context()
            )
        except Exception as e:
            raise DocumentProcessingError(f"Failed to create index: {str(e)}")

    def _load_existing_index(self) -> Optional[VectorStoreIndex]:
        try:
            index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            return index if index.docstore.docs else None
        except Exception:
            return None

    def _load_documents(self, doc_paths: List[str]):
        reader = SimpleDirectoryReader(input_files=doc_paths)
        return reader.load_data()

# Add these functions to engine.py to maintain compatibility with app.py
def load_or_build_index(doc_file_paths: Optional[List[str]] = None) -> Optional[VectorStoreIndex]:
    """
    Compatibility wrapper for the existing app.py.
    Creates or loads an index using the new VectorStoreManager class.
    """
    try:
        engine = initialize_engine()
        return engine.create_index(doc_file_paths)
    except (DocumentProcessingError, SettingsInitializationError) as e:
        print(f"Error processing documents: {e}")
        return None


def initialize_engine():
    """Initialize the engine and create the necessary directories."""
    load_dotenv()
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

    settings_manager = SettingsManager(api_key=os.getenv("GOOGLE_API_KEY"))
    settings_manager.initialize()

    return VectorStoreManager(CHROMA_PERSIST_DIR)


def get_query_engine(index: Optional[VectorStoreIndex]):
    """Creates a query engine from the given index."""
    if not index:
        raise ValueError("Cannot create query engine: Index is None")

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=5
    )
    return query_engine