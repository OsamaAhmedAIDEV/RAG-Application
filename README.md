# RAG PDF Q&A — Streamlit + FastAPI (No OpenAI)

This project provides:
- PDF ingestion -> chunking with page-level metadata
- Embeddings with sentence-transformers + FAISS
- RAG pipeline with generator (Flan-T5) for short-answers and final synthesis
- Source attribution (page numbers + offsets)
- FastAPI backend with simple API-key auth and token-bucket rate limiting
- Streamlit frontend for upload and chat
- Dockerfiles for API and UI + docker-compose file

How to run locally:
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. Start API: uvicorn qa_api:app --reload --port 8000
5. Start UI: streamlit run streamlit_app.py

Docker Compose:
    docker compose up --build

Default demo API key: demo-key-123 (use X-API-Key header)
