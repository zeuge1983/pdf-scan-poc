from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import time
from dotenv import load_dotenv
import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
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
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

class ChatMemory:
    def __init__(self, max_messages: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_chat_history(self) -> str:
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.messages[-self.max_messages:]
        ])

    def clear(self):
        self.messages.clear()

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
            model_name=CONFIG['GEMINI_MODEL'],
            generation_config={
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )

        if self.embedding_choice == "gemini":
            Settings.embed_model = GeminiEmbedding(
                api_key=self.api_key,
                model_name=CONFIG['EMBEDDING_MODEL']
            )
        else:
            raise ValueError(f"Unsupported embedding choice: {self.embedding_choice}")
        return True

class EnhancedQueryEngine:
    def __init__(self, base_query_engine, chat_memory: Optional[ChatMemory] = None):
        self.query_engine = base_query_engine
        self.chat_memory = chat_memory or ChatMemory()
        self.response_cache = {}

    def query(self, query_str: str, **kwargs):
        cache_key = hash(query_str)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        chat_history = self.chat_memory.get_chat_history()
        enhanced_prompt = f"""Previous conversation context:
{chat_history}

Current question: {query_str}

Please provide a comprehensive answer, using the document context and previous conversation if relevant."""

        response = self.query_engine.query(enhanced_prompt)
        
        full_response_text = ""
        response_gen = getattr(response, 'response_gen', None)
        
        if response_gen:
            def collecting_generator():
                nonlocal full_response_text
                for chunk in response_gen:
                    full_response_text += chunk
                    yield chunk
            
            response.response_gen = collecting_generator()
            first_chunk = next(response.response_gen, "")
            full_response_text = first_chunk
            
            response.metadata = {
                "confidence_score": self._calculate_confidence(response),
                "sources": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
            }
            
            self.chat_memory.add_message("user", query_str)
            self.chat_memory.add_message("assistant", first_chunk)
        else:
            full_response_text = response.response
            
            response.metadata = {
                "confidence_score": self._calculate_confidence(response),
                "sources": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
            }
            
            self.chat_memory.add_message("user", query_str)
            self.chat_memory.add_message("assistant", full_response_text)

        self.response_cache[cache_key] = response
        return response

    def _calculate_confidence(self, response) -> float:
        if not hasattr(response, 'source_nodes') or not response.source_nodes:
            return 0.5
        scores = [node.score for node in response.source_nodes if hasattr(node, 'score')]
        return sum(scores) / len(scores) if scores else 0.5

class VectorStoreManager:
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

def load_or_build_index(doc_file_paths: Optional[List[str]] = None) -> Optional[VectorStoreIndex]:
    try:
        engine = initialize_engine()
        return engine.create_index(doc_file_paths)
    except (DocumentProcessingError, SettingsInitializationError) as e:
        print(f"Error processing documents: {e}")
        return None

def initialize_engine():
    load_dotenv()
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

    settings_manager = SettingsManager(api_key=os.getenv("GOOGLE_API_KEY"))
    settings_manager.initialize()

    return VectorStoreManager(CHROMA_PERSIST_DIR)

def get_query_engine(index: Optional[VectorStoreIndex]):
    if not index:
        raise ValueError("Cannot create query engine: Index is None")

    system_prompt = """You are a highly knowledgeable assistant analyzing documents. 
    When answering questions:
    1. Provide comprehensive, well-structured responses
    2. Include relevant examples when applicable
    3. Break down complex information into digestible parts
    4. Cite specific parts of the documents when possible
    5. If information is ambiguous or unclear, acknowledge this and explain why
    6. Use markdown formatting to improve readability

    Context from previous messages: {chat_history}
    Relevant document sections: {context_str}
    User question: {query_str}

    Please provide a thorough response:"""

    custom_prompt = PromptTemplate(system_prompt)

    base_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=5,
        response_synthesizer=get_response_synthesizer(
            response_mode="compact",
            text_qa_template=custom_prompt,
            streaming=True
        ),
        verbose=True
    )

    return EnhancedQueryEngine(base_engine)