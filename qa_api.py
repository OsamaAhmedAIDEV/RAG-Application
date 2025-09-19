# qa_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
import shutil, os, time, threading
from embed_and_index import build_index_from_pdf, Indexer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="RAG PDF Q&A - Streamlit Demo with Auth & Rate Limit")

INDEXER = None
GENERATOR = None
GEN_TOKENIZER = None
MODEL_NAME = "google/flan-t5-base"
INDEX_DIR = "index_store"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

API_KEYS = {"demo-key-123": {"name":"demo"}}

RATE_LIMIT = {"capacity": 10, "refill_rate": 1.0}
tokens = {}
lock = threading.Lock()

def allow_request(api_key: str) -> bool:
    with lock:
        state = tokens.get(api_key)
        now = time.time()
        if state is None:
            tokens[api_key] = {'tokens': RATE_LIMIT['capacity'], 'last': now}
            state = tokens[api_key]
        elapsed = now - state['last']
        refill = elapsed * RATE_LIMIT['refill_rate']
        state['tokens'] = min(RATE_LIMIT['capacity'], state['tokens'] + refill)
        state['last'] = now
        if state['tokens'] >= 1.0:
            state['tokens'] -= 1.0
            return True
        return False

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    max_length: int = 256

@app.on_event("startup")
def startup_event():
    global INDEXER, GENERATOR, GEN_TOKENIZER
    INDEXER = Indexer()
    try:
        INDEXER.load_index()
        print("Loaded existing index.")
    except Exception as e:
        print("No existing index; ingest first.", e)
    GEN_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    GENERATOR = pipeline("text2text-generation", model=GEN_MODEL, tokenizer=GEN_TOKENIZER, device=-1)

def check_api_key(x_api_key: str = Header(None)):
    if x_api_key is None or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not allow_request(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return x_api_key

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF supported.")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    idxr = build_index_from_pdf(dest)
    global INDEXER
    INDEXER = idxr
    return {"status":"ok","chunks":len(idxr.metadatas)}

@app.post("/query")
def query(req: QueryRequest, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    global INDEXER, GENERATOR
    if INDEXER is None or INDEXER.index is None:
        raise HTTPException(status_code=400, detail="Index not ready.")
    results = INDEXER.query(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant docs.")
    answers = []
    for r in results:
        snippet = r['text']
        prompt = f"""You are an assistant. Use the snippet below to answer the question. If snippet doesn't contain answer, say 'NOT_FOUND'.
Snippet:
{snippet}

Question: {req.question}
Answer (short):"""
        gen = GENERATOR(prompt, max_length=128, do_sample=False)[0]['generated_text']
        answers.append({'id': r['id'], 'page': r['page'], 'snippet': snippet, 'score': r['score'], 'answer': gen.strip()})
    synthesis = "You are a final answer synthesizer. Use the short answers below and the scores to produce a single concise final answer. Cite sources with page numbers in square brackets.\n\n"
    for a in answers:
        synthesis += f"Source (page {a['page']}, score {a['score']:.3f}):\n{a['answer']}\n\n"
    synthesis += f"Question: {req.question}\nFinal Answer:"
    final = GENERATOR(synthesis, max_length=req.max_length, do_sample=False)[0]['generated_text']
    return {
        'question': req.question,
        'answer': final.strip(),
        'sources': [{'page':a['page'], 'snippet_start': a['snippet'][:40], 'score':a['score']} for a in answers],
        'raw_retrieved': results,
        'short_answers': answers
    }
