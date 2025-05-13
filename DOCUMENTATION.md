# Document Q&A System with Gemini Pro & LlamaIndex

## Overview
This application is a document question-answering system that uses RAG (Retrieval-Augmented Generation) architecture powered by Google's Gemini Pro 1.5 and LlamaIndex. It allows users to upload documents, process them, and ask questions about their content in natural language.

## Architecture Components

### 1. User Interface (Streamlit)
The UI is built using Streamlit and consists of several key components:

#### Sidebar
- Document upload interface supporting PDF and Markdown files
- Index building controls
- Document management (clear data, view processed files)
- Progress indicators for uploads and indexing

#### Main Interface
- Chat-like Q&A interface
- Message history display
- Confidence scores for answers
- Source references with relevance scores
- Chat export functionality

### 2. Document Processing Pipeline

#### Document Upload
1. Files are uploaded through Streamlit's file uploader
2. Uploaded files are saved to a local directory (DOC_DIR)
3. File names are tracked in session state to prevent duplicates
4. Supports PDF and Markdown formats

#### Document Indexing
1. Uses LlamaIndex for document processing
2. Creates vector embeddings of document content
3. Stores indexed content in ChromaDB (persistent vector store)
4. Handles document chunking and preprocessing

### 3. RAG (Retrieval-Augmented Generation) System

#### Vector Store (ChromaDB)
- Maintains persistent storage of document embeddings
- Enables efficient similarity search
- Stores metadata about document chunks

#### Query Processing
1. User question is received
2. Relevant document chunks are retrieved using similarity search
3. Retrieved context is combined with the question
4. Combined input is sent to LLM for answer generation

#### Response Generation
1. Gemini Pro 1.5 generates responses based on retrieved context
2. Responses are streamed back to the user
3. Confidence scores are calculated for answers
4. Source references are provided with relevance scores

### 4. Large Language Model Integration

#### Gemini Pro 1.5
- Handles natural language understanding
- Generates coherent responses
- Provides streaming capability for real-time response display

#### Query Engine
- Manages interaction between vector store and LLM
- Handles context window limitations
- Optimizes prompt construction

## Key Features

### 1. Document Management
- Multi-file upload support
- Document persistence
- Index rebuilding capability
- Complete data clearance option

### 2. Interactive Q&A
- Real-time response streaming
- Chat history maintenance
- Export functionality for conversations
- Confidence scoring

### 3. Source Attribution
- Reference tracking
- Relevance scoring
- Source content display
- File origin tracking

## Technical Implementation Details

### Session State Management
- Maintains query engine instance
- Tracks processed files
- Manages chat history
- Handles indexing status

### Error Handling
- Upload validation
- Index building error management
- Query processing error handling
- Clear data safety checks

### Performance Considerations
- Persistent vector store for quick reloading
- Efficient document chunking
- Optimized similarity search
- Streamed responses for better UX

## Security and Privacy

### Data Handling
- Local document storage
- Temporary file management
- Secure API key handling
- Chat history privacy

### System Requirements
- Google API Key for Gemini Pro access
- Local storage for documents
- ChromaDB for vector storage
- Python environment with required packages

## Limitations and Considerations

### Document Processing
- Limited to PDF and Markdown formats
- File size restrictions
- Processing time for large documents
- Index rebuild requirements

### Query Processing
- Context window limitations
- Response quality dependent on document content
- Confidence score reliability
- Processing time for complex queries

## Future Improvements

### Potential Enhancements
1. Additional document format support
2. Improved error handling
3. Better confidence scoring
4. Enhanced source attribution
5. Multi-language support
6. Advanced query optimization
7. User authentication
8. Cloud storage integration

This document Q&A system provides a robust solution for document-based question answering, combining modern LLM technology with efficient information retrieval techniques.