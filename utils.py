# utils.py
import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text_with_meta(text: str, page_no:int, chunk_size: int = 900, overlap: int = 200) -> List[Tuple[str,int,int,int]]:
    '''
    Split text into overlapping chunks and return list of tuples:
    (chunk_text, page_no, char_start, char_end)
    '''
    text = text.strip()
    if len(text) <= chunk_size:
        return [(text, page_no, 0, len(text))]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, page_no, start, end))
        if end == L:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
