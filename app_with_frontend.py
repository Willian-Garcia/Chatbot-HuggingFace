# app_with_frontend.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import os, json, re, requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from urllib.parse import quote

# ------------------------
# App + Frontend estático
# ------------------------
app = FastAPI(title="RAG + HF (API + Frontend únicos)")

# CORS (deixe aberto só em dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).parent
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

@app.get("/", include_in_schema=False)
def _root():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"msg": "Abra /frontend para o chat."}

# ------------------------
# Env
# ------------------------
load_dotenv()

def _clean_env(v: Optional[str], default: Optional[str] = None) -> str:
    if v is None:
        return default or ""
    v = v.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    v = v.replace("\ufeff", "").replace("\u200b", "")
    return v

HF_TOKEN = _clean_env(os.getenv("HF_TOKEN"))
HF_MODEL = _clean_env(os.getenv("HF_MODEL"), "meta-llama/Llama-3.2-1B-Instruct")
CONTEXT_PATH = _clean_env(os.getenv("CONTEXT_PATH"), "context.txt")
FAISS_INDEX_PATH = _clean_env(os.getenv("FAISS_INDEX_PATH"), "faiss_index.bin")
ONNX_MODEL_PATH = _clean_env(os.getenv("ONNX_MODEL_PATH") or "")
LOCAL_QA_PATH = _clean_env(os.getenv("LOCAL_QA_PATH"), "context_qa.jsonl")

if not HF_TOKEN:
    raise RuntimeError("Defina HF_TOKEN no .env (sem aspas).")
if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", HF_MODEL):
    raise RuntimeError(f"HF_MODEL inválido: {HF_MODEL!r}")

HF_MODEL_PATH = quote(HF_MODEL, safe="/A-Za-z0-9._-")

# ------------------------
# Embeddings + índice
# ------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)
dim = embedder.get_sentence_embedding_dimension()

def _new_index():
    return faiss.IndexFlatIP(dim)

documents: List[str] = []
index = _new_index()

def _load_cache_if_any():
    """Carrega FAISS e chunks se existirem. Chunks = CONTEXT_PATH + '.chunks.jsonl'."""
    global documents, index
    docs_loaded = 0
    ctx_path = Path(CONTEXT_PATH)
    chunks_path = ctx_path.with_suffix(ctx_path.suffix + ".chunks.jsonl")
    idx_path = Path(FAISS_INDEX_PATH)

    if chunks_path.exists() and idx_path.exists():
        documents = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("text", "")
                    if txt:
                        documents.append(txt)
                except json.JSONDecodeError:
                    continue
        loaded = faiss.read_index(str(idx_path))
        if loaded.d != dim:
            index = _new_index()
            if documents:
                vecs = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
                index.add(vecs)
        else:
            index = loaded
        docs_loaded = len(documents)
    else:
        index = _new_index()
        documents = []
    return docs_loaded

def _encode_norm(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def _persist_new_doc(text: str, vec: np.ndarray):
    """Acrescenta doc no JSONL derivado do CONTEXT_PATH e atualiza FAISS_INDEX_PATH."""
    ctx_path = Path(CONTEXT_PATH)
    chunks_path = ctx_path.with_suffix(ctx_path.suffix + ".chunks.jsonl")
    documents.append(text)
    index.add(vec)
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    faiss.write_index(index, FAISS_INDEX_PATH)

loaded_count = _load_cache_if_any()
print(f"[INFO] Documentos carregados do cache: {loaded_count}")

# ------------------------
# Local Q&A (fallback fixo)
# ------------------------
def load_local_qa(path: str) -> List[dict]:
    qa_pairs: List[dict] = []
    p = Path(path)
    if not p.exists():
        return qa_pairs
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "pergunta" in obj and "resposta" in obj:
                    qa_pairs.append(obj)
            except json.JSONDecodeError:
                continue
    return qa_pairs

LOCAL_QA = load_local_qa(LOCAL_QA_PATH)

def match_local_answer(question: str) -> Optional[str]:
    """Encontra a melhor Q fixa por similaridade de embeddings; retorna a resposta se confiante."""
    if not LOCAL_QA:
        return None
    qv = _encode_norm([question])            # (1, dim)
    qs = [q["pergunta"] for q in LOCAL_QA]
    tv = _encode_norm(qs)                    # (N, dim)
    sims = (tv @ qv.T).ravel()
    best_i = int(np.argmax(sims))
    best_sim = float(sims[best_i])
    # limiar ajustável; 0.6 costuma ser seguro para Q curtas
    if best_sim >= 0.60:
        return LOCAL_QA[best_i]["resposta"]
    return None

# ------------------------
# Schemas
# ------------------------
class Ingest(BaseModel):
    text: str

class Ask(BaseModel):
    question: str
    k: Optional[int] = 3

# ------------------------
# Util: limpeza, intents e leitura inteligente
# ------------------------
def _clean_markers(text: str) -> str:
    return re.sub(r":contentreference\[.*?\]\{.*?\}", "", text, flags=re.IGNORECASE)

# Mini-intents (regex) para respostas bem pontuais quando a pergunta sugere o alvo
INTENT_PATTERNS = [
    (("semestres", "duração"),
     r"(\d+)\s*semestres?(?:\s*\((\d+)\s*anos?\))?",
     lambda m: f"O curso tem {m.group(1)} semestres" + (f" ({m.group(2)} anos)." if m.group(2) else ".")),
    (("modalidade",),
     r"(presencial(?:,?\s*com.*?remotas?)?|híbrida?|ead)",
     lambda m: m.group(1).strip().capitalize().rstrip(".") + "."),
    (("eixo", "tecnológico"),
     r"eixo\s+tecnológico:\s*([^\n]+)",
     lambda m: m.group(1).strip().rstrip(".") + "."),
    (("coordenador", "coordenação"),
     r"coordenador[a]?:\s*([^\n]+)",
     lambda m: m.group(1).strip().rstrip(".") + "."),
    (("carga", "horária"),
     r"carga\s*horária:\s*([0-9\.]+\s*h)",
     lambda m: m.group(1).strip().rstrip(".") + "."),
]

def try_intents_short_answer(question: str, context_clean: str) -> Optional[str]:
    q = question.lower()
    for keywords, rx, formatter in INTENT_PATTERNS:
        if all(k in q for k in keywords):
            m = re.search(rx, context_clean, flags=re.IGNORECASE)
            if m:
                return formatter(m)
    return None

def extract_short_answer(question: str, retrieved_texts: List[str],
                         min_sim: float = 0.30, max_chars: int = 240) -> Optional[str]:
    """
    Extrativo por sentença: escolhe a frase mais similar à pergunta (cosseno)
    entre as sentenças dos top-k trechos recuperados.
    """
    sents = []
    for txt in retrieved_texts:
        if not txt:
            continue
        parts = re.split(r'(?<=[\.\?!;])\s+', txt.strip())
        sents.extend([p for p in parts if p])

    if not sents:
        return None

    qv = _encode_norm([question])     # (1, dim)
    sv = _encode_norm(sents)          # (N, dim)
    sims = (sv @ qv.T).ravel()        # cossenos
    best_i = int(np.argmax(sims))
    best_sim = float(sims[best_i])
    if best_sim < min_sim:
        return None

    short = _clean_markers(sents[best_i].strip())
    if len(short) > max_chars:
        short = short[:max_chars].rstrip() + "…"
    return short

# ------------------------
# Rotas
# ------------------------
@app.post("/ingest")
def ingest(item: Ingest):
    vec = _encode_norm([item.text])
    _persist_new_doc(item.text, vec)
    return {"status": "added", "total_docs": len(documents)}

@app.post("/ask")
def ask(item: Ask):
    # 0) Se não há nada de contexto e nem Q&A local, avisa
    if len(documents) == 0 and not LOCAL_QA:
        return {
            "answer": "Sem contexto disponível. Gere o cache via generate_cache.py, use /ingest ou forneça um context_qa.jsonl.",
            "retrieved": [],
            "model": HF_MODEL,
            "mode": "no_context"
        }

    # 1) Q&A LOCAL (prioritário) — responde mesmo sem docs
    #    Aumenta a chance de resposta pontual e previsível
    local_ans = match_local_answer(item.question)
    if local_ans:
        return {"answer": local_ans, "retrieved": [], "model": "local-qa", "mode": "offline-fallback"}

    # 2) Recuperação semântica (se houver docs) para intents/extrativo/HF
    retrieved: List[dict] = []
    if documents:
        q_vec = _encode_norm([item.question])
        k = min(item.k or 3, len(documents))
        D, I = index.search(q_vec, k)
        for score, idx_ in zip(D[0].tolist(), I[0].tolist()):
            if idx_ == -1:
                continue
            retrieved.append({"text": documents[idx_], "score": float(score)})

    # 3) Contexto limpo (para intents / HF)
    context = "\n---\n".join([r["text"] for r in retrieved]) if retrieved else ""
    context_clean = _clean_markers(context)

    # 4) Intents (regex) — super pontual quando aplicável
    if context_clean:
        ans = try_intents_short_answer(item.question, context_clean)
        if ans:
            return {"answer": ans, "retrieved": retrieved, "model": "regex-intents", "mode": "extractive-intent"}

    # 5) Leitura inteligente por sentença (genérico)
    if retrieved:
        concise = extract_short_answer(item.question, [r["text"] for r in retrieved],
                                       min_sim=0.30, max_chars=240)
        if concise:
            return {"answer": concise, "retrieved": retrieved, "model": "miniLM sentence-extractive", "mode": "extractive-sent"}

    # 6) Hugging Face (se houver algum contexto)
    if context_clean:
        prompt = (
            "Você é um assistente que responde de forma objetiva usando APENAS o contexto fornecido.\n"
            "Se a resposta não estiver no contexto, diga que não há informações suficientes.\n\n"
            f"Contexto:\n{context_clean}\n\n"
            f"Pergunta: {item.question}\n"
            "Resposta:"
        )
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False},
            "options": {"wait_for_model": True, "use_cache": False}
        }
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL_PATH}"

        resp = None
        data = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            ct = resp.headers.get("content-type", "")
            data = resp.json() if "application/json" in ct else None
        except Exception:
            resp = None
            data = None

        if resp is not None and resp.status_code == 200 and isinstance(data, list) and data and "generated_text" in data[0]:
            return {"answer": data[0]["generated_text"], "retrieved": retrieved, "model": HF_MODEL, "mode": "hf"}

    # 7) Fallback curto final (sem despejar contexto)
    return {
        "answer": "Não encontrei essa informação no contexto.",
        "retrieved": retrieved,
        "model": HF_MODEL,
        "mode": "fallback"
    }

@app.post("/clear")
def clear():
    """Zera memória e remove arquivos de cache (chunks + FAISS)."""
    documents.clear()
    global index
    index = _new_index()
    ctx_path = Path(CONTEXT_PATH)
    chunks_path = ctx_path.with_suffix(ctx_path.suffix + ".chunks.jsonl")
    if chunks_path.exists():
        chunks_path.unlink()
    idx_path = Path(FAISS_INDEX_PATH)
    if idx_path.exists():
        idx_path.unlink()
    return {"status": "cleared"}

@app.get("/health")
def health():
    tok = os.getenv("HF_TOKEN")
    raw_model = os.getenv("HF_MODEL")
    return {
        "hf_model": HF_MODEL,
        "hf_model_raw": raw_model,
        "hf_token_present": bool(tok),
        "hf_token_sample": (tok[:6] + "..." + tok[-4:]) if tok else None,
        "docs_loaded": len(documents),
        "context_path": CONTEXT_PATH,
        "faiss_index_path": FAISS_INDEX_PATH,
        "local_qa_path": LOCAL_QA_PATH,
        "local_qa_count": len(LOCAL_QA),
    }

@app.get("/hf_test")
def hf_test():
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_PATH}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": "ping", "options": {"wait_for_model": True}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        ct = r.headers.get("content-type", "")
        body = r.text[:500]
        return {"url": url, "status": r.status_code, "content_type": ct, "body_preview": body}
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erro de rede ao chamar HF: {e}")
