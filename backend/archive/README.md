# Learning artifacts

Code from weeks 1–3 while learning the NVIDIA NIM API, PDF processing,
and basic RAG. Superseded by the LangGraph agent in the parent directory.

Kept for reference and to show the progression of the project.
Not imported by any running code.

- `hello_nemotron.py` — first API call
- `chat_loop.py` — conversation memory without retrieval
- `doc_agent.py` — document persona prompt experiments
- `understand_embeddings.py` — embedding dimensionality exploration
- `pdf_loader.py`, `text_chunker.py` — early PDF pipeline, now in `core/ingest.py`
- `rag_chain.py`, `rag_chat.py` — linear RAG before LangGraph routing
- `setup_chromadb.py`, `query_chromadb.py` — first ChromaDB experiments
- `embed_and_store.py` — CLI ingest pipeline, superseded by the /upload endpoint