# Knowledge-Based-Search-Engine
**Demo Video :**  https://drive.google.com/file/d/17aPCmxsR7Yuweqgc6vlNplSm7IXK2Xgm/view?usp=sharing

Knowledge-Base Search Engine with Llama 2 and RAG
This repository contains a Jupyter Notebook that demonstrates how to build an end-to-end knowledge-base search engine using the Llama 2 7B model and the Retrieval-Augmented Generation (RAG) technique.

### Objective
The primary goal of this project is to create a system that can answer user questions based on a specific set of private documents. Unlike a standard chatbot that relies on its general pre-trained knowledge, this RAG-based system ensures that its answers are accurate, verifiable, and grounded in the provided knowledge base.

### How It Works: The RAG Pipeline
The system follows a classic Retrieval-Augmented Generation workflow:

Document Ingestion: Local documents (such as .txt or .pdf files) are loaded, parsed, and split into smaller, manageable chunks.

Indexing: Each text chunk is converted into a numerical vector (embedding) that captures its semantic meaning. These embeddings are then stored in a ChromaDB vector store for efficient searching.

Retrieval: When a user asks a question, the query is also converted into an embedding. The system performs a similarity search in the vector store to find and retrieve the most relevant document chunks.

Generation: The retrieved chunks are combined with the user's original query into a detailed prompt. This context-rich prompt is then fed to the Llama 2 model, which generates a final, synthesized answer that is grounded in the retrieved information.

### Tech Stack
LLM: meta-llama/Llama-2-7b-chat-hf (quantized to 4-bit using bitsandbytes).

Framework: LangChain for orchestrating the RAG pipeline.

Embedding Model: sentence-transformers/all-mpnet-base-v2 for generating text embeddings.

Vector Store: ChromaDB for storing and retrieving document embeddings.

Core Libraries: PyTorch, Hugging Face Transformers.

### Getting Started
Prerequisites:
    Python 3.8+
    A Hugging Face account and an access token with permission to use Llama 2.
**Configuration**
Hugging Face Token: To access the Llama 2 model, you need to authenticate. The notebook is set up to use Google Colab's userdata secrets. Add your Hugging Face token as a secret named Rag.

Add Your Documents: Place your .txt or .pdf files in the /content/ directory (or modify the path in cell 19) to create your own custom knowledge base.

Running the Notebook
Execute the cells in the Jupyter Notebook sequentially to set up the environment, load the model, ingest documents, and test the RAG pipeline with your own queries.
Python 3.8+

A Hugging Face account and an access token with permission to use Llama 2.
