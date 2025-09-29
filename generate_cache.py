# generate_cache.py
from pathlib import Path
import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CONTEXT_PATH = os.getenv("CONTEXT_PATH", "context.txt")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600        # caracteres
CHUNK_OVERLAP = 100     # caracteres

def chunk_text(text: str, size=600, overlap=100):
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks

def main():
    ctx_path = Path(CONTEXT_PATH)
    if not ctx_path.exists():
        raise SystemExit(f"[ERRO] {ctx_path} não encontrado. Ajuste CONTEXT_PATH no .env.")

    raw = ctx_path.read_text(encoding="utf-8")
    parts = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[INFO] Total de chunks: {len(parts)}")

    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode(parts, convert_to_numpy=True, normalize_embeddings=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    chunks_path = ctx_path.with_suffix(ctx_path.suffix + ".chunks.jsonl")
    with chunks_path.open("w", encoding="utf-8") as f:
        for p in parts:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")

    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"[OK] Gerados:\n - {chunks_path.name}\n - {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
