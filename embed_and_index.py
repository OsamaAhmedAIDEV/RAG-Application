# embed_and_index.py
import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import pdf_to_chunks

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "index_store"

class Indexer:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadatas = []
        os.makedirs(INDEX_DIR, exist_ok=True)

    def build_index(self, chunks):
        docs = [c[0] for c in chunks]
        embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        self.metadatas = []
        for i, c in enumerate(chunks):
            text, page_no, start, end = c
            self.metadatas.append({
                "id": i,
                "text": text,
                "page": page_no,
                "char_start": start,
                "char_end": end
            })
        self._save_index()

    def _save_index(self):
        faiss.write_index(self.index, os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load_index(self):
        idx_path = os.path.join(INDEX_DIR, "faiss.index")
        meta_path = os.path.join(INDEX_DIR, "meta.json")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index files not found. Build index first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def query(self, query_text: str, top_k: int = 4):
        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadatas[idx]
            results.append({"score": float(score), "id": meta["id"], "text": meta["text"], "page": meta["page"], "char_start": meta["char_start"], "char_end": meta["char_end"]})
        return results

def build_index_from_pdf(pdf_path: str):
    chunks = pdf_to_chunks(pdf_path)
    idxr = Indexer()
    idxr.build_index(chunks)
    print(f"Built index with {len(chunks)} chunks.")
    return idxr

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_and_index.py file.pdf")
        raise SystemExit(1)
    build_index_from_pdf(sys.argv[1])
