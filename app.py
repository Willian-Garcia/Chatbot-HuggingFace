from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

# ------------------------
# FastAPI + CORS
# ------------------------
app = FastAPI(title="RAG + Hugging Face Inference API (Exemplo robusto)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ Em produção, restrinja domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Config simples (.env) + saneamento
# ------------------------
load_dotenv()  # carrega .env

def _clean_env(value: Optional[str], default: Optional[str] = None) -> str:
    """
    Remove espaços, aspas, quebras de linha e caracteres invisíveis comuns.
    """
    if value is None:
        return default or ""
    v = value.strip()                   # remove espaços nas pontas
    v = v.strip('"').strip("'")         # remove aspas
    v = v.replace("\r", "").replace("\n", "")  # remove CR/LF
    # Remove alguns caracteres invisíveis comuns (ex.: BOM / zero-width)
    v = v.replace("\ufeff", "").replace("\u200b", "")
    return v

HF_TOKEN = _clean_env(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
HF_MODEL = _clean_env(os.getenv("HF_MODEL"), "Qwen/Qwen2.5-0.5B-Instruct")

if not HF_TOKEN:
    raise RuntimeError(
        "Defina HUGGINGFACEHUB_API_TOKEN no ambiente ou no .env (sem aspas)."
    )

# Validação leve do formato owner/model
_model_pattern = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
if not _model_pattern.match(HF_MODEL):
    raise RuntimeError(
        f"HF_MODEL inválido: {HF_MODEL!r}. Exemplo válido: Qwen/Qwen2.5-0.5B-Instruct"
    )

# Monta caminho seguro preservando as barras (escapa espaços/aspas, etc.)
HF_MODEL_PATH = quote(HF_MODEL, safe="/A-Za-z0-9._-")

# ------------------------
# Modelo de embeddings
# ------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = embedder.get_sentence_embedding_dimension()  # 384
index = faiss.IndexFlatIP(dim)  # inner product com normalização = cos
documents: List[str] = []  # memória simples em RAM

# ------------------------
# Modelos de entrada
# ------------------------
class Ingest(BaseModel):
    text: str

class Ask(BaseModel):
    question: str
    k: Optional[int] = 3

def _encode_norm(texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

# ------------------------
# Rotas
# ------------------------
@app.post("/ingest")
def ingest(item: Ingest):
    vec = _encode_norm([item.text])
    index.add(vec)
    documents.append(item.text)
    return {"status": "added", "text": item.text, "total_docs": len(documents)}

@app.post("/ask")
def ask(item: Ask):
    if len(documents) == 0:
        return {
            "answer": "Sem contexto disponível. Por favor, ingira documentos primeiro em /ingest.",
            "context": None,
        }

    q_vec = _encode_norm([item.question])
    k = min(item.k or 3, len(documents))
    D, I = index.search(q_vec, k)

    retrieved = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        retrieved.append({"text": documents[idx], "score": float(score)})

    context = "\n---\n".join([r["text"] for r in retrieved])

    prompt = (
        "Você é um assistente que responde de forma objetiva usando APENAS o contexto fornecido.\n"
        "Se a resposta não estiver no contexto, diga que não há informações suficientes.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {item.question}\n"
        "Resposta:"
    )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "return_full_text": False},
        "options": {"wait_for_model": True, "use_cache": False},
    }

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_PATH}"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar HF API: {e}")

    # Tente JSON primeiro; se vier texto, repasse parte p/ diagnóstico
    content_type = resp.headers.get("content-type", "")
    text_body = resp.text
    try:
        data = resp.json() if "application/json" in content_type else None
    except ValueError:
        data = None

    if resp.status_code != 200:
        detail = data if data is not None else {"non_json_body": text_body[:500]}
        # Inclui a URL efetiva para depuração
        detail = {"error": detail, "url": url, "model": HF_MODEL}
        raise HTTPException(status_code=resp.status_code, detail=detail)

    # Formatos comuns do Inference API:
    answer = None
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        answer = data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        answer = data["generated_text"]
    elif data is not None:
        answer = str(data)
    else:
        raise HTTPException(status_code=502, detail="HF retornou corpo não-JSON: " + text_body[:500])

    return {"answer": answer, "retrieved": retrieved, "model": HF_MODEL}

# ------------------------
# Debug/diagnóstico
# ------------------------
@app.get("/health")
def health():
    tok = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    raw_model = os.getenv("HF_MODEL")
    return {
        "hf_model": HF_MODEL,           # limpo
        "hf_model_raw": raw_model,      # como veio do .env
        "hf_token_present": bool(tok),
        "hf_token_sample": tok[:6] + "..." + tok[-4:] if tok else None,
    }

@app.get("/debug_env")
def debug_env():
    raw_model = os.getenv("HF_MODEL") or ""
    codepoints = [ord(ch) for ch in raw_model]
    return {
        "raw": raw_model,
        "len": len(raw_model),
        "codepoints": codepoints[:64],  # primeiros 64 p/ inspeção
        "cleaned": HF_MODEL,
        "cleaned_path": HF_MODEL_PATH,
    }
