# RAG Document Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, FAISS, HuggingFace embeddings, and Claude as the LLM, served through a Streamlit UI.

## How it works

1. **Ingest** — a PDF is loaded, split into overlapping chunks, and embedded using a HuggingFace sentence transformer. The embeddings are stored in a local FAISS vector index.
2. **Retrieve** — when you ask a question, the top-k most relevant chunks are retrieved from the index via semantic similarity search.
3. **Generate** — the retrieved chunks are passed as context to Claude, which generates a grounded answer.

## Project structure

```
RAG_chatbot/
├── app.py            # Streamlit UI + chat logic
├── ingest.py         # PDF loading, chunking, embedding, FAISS index builder
├── faiss_index/      # Generated vector store (created after first ingest)
├── requirements.txt
└── .env              # ANTHROPIC_API_KEY goes here
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd RAG_chatbot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

## Usage

- Upload any PDF using the sidebar file uploader.
- The app will automatically chunk and index the document.
- Type questions in the chat input — answers are grounded in your document.
- Expand **Sources used** under any answer to see the exact chunks retrieved.
- Uploading a new PDF replaces the current index and clears the chat history.

## Pre-indexing a PDF (optional)

You can also build the index from the command line before launching the app:

```bash
python ingest.py your_document.pdf
```

This saves the index to `faiss_index/`. The app will use it automatically on startup.

## Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Claude via `langchain-anthropic` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (`faiss-cpu`) |
| PDF loading | LangChain `PyPDFLoader` |
| Orchestration | LangChain `RetrievalQA` |
