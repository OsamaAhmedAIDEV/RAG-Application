# streamlit_app.py
import streamlit as st
import requests
import os
from pathlib import Path

API_URL = 'http://localhost:8000'
API_KEY = 'demo-key-123'

def ingest_file(file_path):
    url = f"{API_URL}/ingest"
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
        headers = {'X-API-Key': API_KEY}
        resp = requests.post(url, files=files, headers=headers)
    return resp.json()

def query_api(question, top_k=4):
    url = f"{API_URL}/query"
    headers = {'Content-Type':'application/json', 'X-API-Key': API_KEY}
    payload = {'question': question, 'top_k': top_k}
    resp = requests.post(url, json=payload, headers=headers)
    return resp.json()

st.set_page_config(page_title='RAG PDF Q&A', layout='wide')
st.title("RAG PDF Q&A — Streamlit UI")
st.write("Upload a PDF and ask questions. This UI talks to the FastAPI backend.")

uploaded = st.file_uploader("Upload PDF", type=['pdf'])
if uploaded:
    save_path = Path('uploads') / uploaded.name
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    if st.button("Ingest PDF"):
        with st.spinner("Uploading and indexing..."):
            res = ingest_file(str(save_path))
            st.success(str(res))

st.markdown('---')
st.header("Ask a question")
question = st.text_input("Your question")
top_k = st.slider("Top K retrieval", 1, 8, 4)
if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying..."):
            res = query_api(question, top_k=top_k)
        if res.get('answer'):
            st.subheader("Answer")
            st.write(res['answer'])
        if res.get('short_answers'):
            st.subheader("Retrieved snippets & short answers")
            for sa in res['short_answers']:
                st.markdown(f"**Page {sa['page']}** (score {sa['score']:.3f})")
                st.write(sa['snippet'])
                st.write(f"**Short answer:** {sa['answer']}")
