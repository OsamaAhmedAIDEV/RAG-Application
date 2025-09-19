# ingest.py
from typing import List, Tuple
import PyPDF2
from utils import clean_text, chunk_text_with_meta

def pdf_to_pages(pdf_path: str) -> List[Tuple[int,str]]:
    '''
    Return list of (page_no (1-indexed), text) for each page.
    '''
    reader = PyPDF2.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append((i+1, clean_text(page_text)))
    return pages

def pdf_to_chunks(pdf_path: str, chunk_size:int=900, overlap:int=200):
    pages = pdf_to_pages(pdf_path)
    all_chunks = []
    for page_no, text in pages:
        chs = chunk_text_with_meta(text, page_no, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chs)
    return all_chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py file.pdf")
        raise SystemExit(1)
    chunks = pdf_to_chunks(sys.argv[1])
    print(f'Created {len(chunks)} chunks from PDF')
    for c in chunks[:3]:
        print(c)
