# AI-Powered RAG Ecosystem: Multimodal Chatbot & Knowledge Base

This repository houses two distinct Retrieval-Augmented Generation (RAG) implementations designed for advanced document analysis. It features a Multimodal RAG Chatbot capable of interpreting text and images within PDFs using local Vision-Language Models (VLMs), and a Knowledge-Base Search Engine leveraging quantized Large Language Models (LLMs) for high-precision semantic search.

## Project Components
### 1. Multimodal RAG Chatbot (Local Streamlit App)
A local, privacy-focused chatbot that processes PDF documents containing both text and complex visual data (images/tables). It utilizes Ollama to run models locally, ensuring no data leaves your machine.

Core Functionality:

Layout Parsing: Uses Unstructured to partition PDFs into text and image elements.

Vision Capabilities: Integrates the Moondream Vision-Language Model to generate detailed textual descriptions for images found within the PDF.

Vector Search: Indexes text chunks and image descriptions into an InMemoryVectorStore using Nomic embeddings for semantic retrieval.

Interactive UI: Built with Streamlit for seamless drag-and-drop file uploading and chatting.

### 2. Knowledge-Base Search Engine
A deep learning pipeline designed for creating a searchable knowledge base from heterogeneous document sources (Text & PDF). This implementation focuses on high-performance inference using quantized models.

#### Core Functionality:

Quantization: Implements 4-bit quantization (NF4) using BitsAndBytes to run the Llama-2-7b-chat model efficiently on consumer hardware (T4 GPU).

Embeddings: Utilizes sentence-transformers/all-mpnet-base-v2 for state-of-the-art sentence embeddings.

Vector Database: Persists document embeddings using ChromaDB for fast and scalable similarity search.

Retrieval Chain: Deploys a LangChain RetrievalQA chain to contextualize user queries and generate concise summaries.

| Category      | Technologies / Libraries |
| ------------- | ------------- |
| LLMs & VLMs |Llama-2-7b-chat, Moondream (Vision), Ollama |
| Embeddings  |Nomic-embed-text, all-mpnet-base-v2 |
| Orchestration| LangChain (Core, Community, Ollama)  |
| Vector Stores| ChromaDB, InMemoryVectorStore  |
| Document Processing| Unstructured (PDF Partitioning), PyPDFLoader  |
| Optimization| BitsAndBytes (4-bit Quantization), Accelerate  |
| Frontend| Streamlit  |
