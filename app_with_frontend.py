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
import unicodedata as ud

# ------------------------
# App + Frontend estático
# ------------------------
app = FastAPI(title="RAG (FAISS + MiniLM + HF opcional) — API + Frontend")

# CORS (dev)
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

# Normalização (coerente com generate_cache.py)
NORMALIZE_LEVEL = (_clean_env(os.getenv("NORMALIZE_LEVEL"), "light") or "light").lower()  # none|light|aggressive
STRIP_DIACRITICS = (_clean_env(os.getenv("STRIP_DIACRITICS"), "0") == "1")

# Chunking para /ingest (além do generate_cache.py)
def _to_int_env(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, default))
        return v if v > 0 else default
    except Exception:
        return default
INGEST_CHUNK_SIZE = _to_int_env("INGEST_CHUNK_SIZE", 600)
INGEST_OVERLAP    = _to_int_env("INGEST_OVERLAP", 100)

if HF_MODEL and not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", HF_MODEL):
    raise RuntimeError(f"HF_MODEL inválido: {HF_MODEL!r}")
HF_MODEL_PATH = quote(HF_MODEL, safe="/A-Za-z0-9._-")

# ------------------------
# Normalização para embeddings (query/texto)
# ------------------------
PUNCT_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "–": "-", "—": "-", "−": "-", "‐": "-",
    "…": "...", "·": ".", "•": "-",
    "\u00A0": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ", "\u2007": " ",
})
CONTROL_RX = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]")
SPACES_RX  = re.compile(r"[ \t]+")
MULTI_NL_RX = re.compile(r"\n{2,}")

def _strip_diacritics(s: str) -> str:
    nfd = ud.normalize("NFD", s)
    return "".join(ch for ch in nfd if ud.category(ch) != "Mn")

def _normalize_for_embedding(s: str) -> str:
    if not s:
        return s
    s = ud.normalize("NFKC", s)
    s = s.translate(PUNCT_MAP)
    s = CONTROL_RX.sub("", s)
    if STRIP_DIACRITICS:
        s = _strip_diacritics(s)
    if NORMALIZE_LEVEL == "aggressive":
        s = s.lower()
    s = SPACES_RX.sub(" ", s).strip()
    return s

def _clean_markers(text: str) -> str:
    """Remove marcadores tipo :contentReference[...] e normaliza um pouco para exibição/HF."""
    t = re.sub(r":contentreference\[.*?\]\{.*?\}", "", text, flags=re.IGNORECASE)
    t = t.replace("\ufeff", "").replace("\u200b", "")
    t = SPACES_RX.sub(" ", t)
    t = MULTI_NL_RX.sub("\n\n", t)
    return t.strip()

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
                vecs = embedder.encode(
                    [_normalize_for_embedding(t) for t in documents],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                index.add(vecs)
        else:
            index = loaded
        docs_loaded = len(documents)
    else:
        index = _new_index()
        documents = []
    return docs_loaded

def _encode_norm(texts: List[str]) -> np.ndarray:
    """Embeddings com normalização (alinhado ao generate_cache)."""
    normed = [_normalize_for_embedding(t) for t in texts]
    return embedder.encode(normed, convert_to_numpy=True, normalize_embeddings=True)

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
# Schemas
# ------------------------
class Ingest(BaseModel):
    text: str

class Ask(BaseModel):
    question: str
    k: Optional[int] = 3
    mode: Optional[str] = None   # "auto" | "hf"

# ------------------------
# Tokenização leve + overlap lexical (genérico)
# ------------------------
STOPWORDS_PT = {
    "a","o","os","as","um","uma","de","do","da","dos","das","em","no","na","nos","nas","para","por","com",
    "e","ou","que","se","sua","seu","suas","seus","ao","à","às","aos","como","qual","quais","é","ser","são",
    "daqui","dali","disso","neste","nesta","nisto","nesse","nessa","nesso","neste","nesta","nisto","isso","isto",
    "sobre","entre","até","desde","pela","pelo","pelas","pelos","um","uma","uns","umas","mais","menos","muito",
    "muita","muitos","muitas","também","já","só","apenas","quando","onde","depois","antes","sem","há","ter","tem",
}

WORD_RX = re.compile(r"[A-Za-zÁÂÃÀÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç0-9]+", flags=re.UNICODE)

def _tokenize(s: str) -> List[str]:
    s = _normalize_for_embedding(s)
    toks = [t.lower() for t in WORD_RX.findall(s)]
    return [t for t in toks if t not in STOPWORDS_PT and len(t) > 1]

def _lexical_overlap_score(q: str, s: str) -> float:
    q_tok = set(_tokenize(q))
    s_tok = set(_tokenize(s))
    if not q_tok or not s_tok:
        return 0.0
    inter = len(q_tok & s_tok)
    # Jaccard suave
    return inter / float(len(q_tok | s_tok))

# --------- Extrativo por sentença (híbrido: semântico + lexical) ----------
def extract_short_answer(question: str, retrieved_texts: List[str],
                         min_sim: float = 0.20, max_chars: int = 240,
                         alpha: float = 0.7) -> Optional[str]:
    """
    Extrativo híbrido:
      - cosine (MiniLM) + overlap lexical (Jaccard de tokens) → score combinado
      - "label-aware": se melhor sentença termina em ":" ou é muito curta, junta a próxima
    """
    # 1) limpar blocos
    cleaned_blocks = []
    for txt in retrieved_texts:
        if not txt:
            continue
        t = _clean_markers(txt)
        t = re.sub(r"^=+\s.*?$", "", t, flags=re.MULTILINE)  # ===== headings
        t = re.sub(r"^-{3,}$", "", t, flags=re.MULTILINE)    # ---- separators
        if t:
            cleaned_blocks.append(t)
    if not cleaned_blocks:
        return None

    # 2) tokenizar por sentenças guardando mapeamento (bloco, idx)
    sents, meta = [], []  # meta: (block_id, sent_idx, parts)
    for b_id, txt in enumerate(cleaned_blocks):
        parts = re.split(r'(?<=[\.\?!;:])\s+', txt)
        for i, p in enumerate(parts):
            p = p.strip()
            if not p or len(p) < 5 or len(p.split()) < 2:
                continue
            sents.append(p)
            meta.append((b_id, i, parts))

    if not sents:
        return None

    # 3) similaridade semântica
    qv = _encode_norm([question])
    sv = _encode_norm(sents)
    cos = (sv @ qv.T).ravel()

    # 4) overlap lexical
    lex = np.array([_lexical_overlap_score(question, s) for s in sents], dtype="float32")

    # 5) score combinado
    #   - alpha dá mais peso ao semântico; lex ajuda em "Label: Valor"
    score = alpha * cos + (1.0 - alpha) * lex
    best_i = int(np.argmax(score))
    best_sim = float(cos[best_i])  # checagem mínima em termos semânticos
    if best_sim < min_sim and score[best_i] < (min_sim * 0.6):
        return None

    # 6) "label-aware": se terminou com ":" OU é curta, pega a próxima do mesmo bloco
    chosen = sents[best_i].strip()
    _, s_idx, parts_ref = meta[best_i]
    needs_next = chosen.endswith(":") or len(chosen) < 30
    if needs_next and s_idx + 1 < len(parts_ref):
        nxt = parts_ref[s_idx + 1].strip()
        if nxt:
            combined = (chosen + " " + nxt).strip()
            combined = re.sub(r"^\s*[A-ZÁ-Úa-zá-ú0-9 ]{3,60}:\s*", "", combined)
            chosen = combined

    # 7) truncagem
    if len(chosen) > max_chars:
        chosen = chosen[:max_chars].rstrip() + "…"
    return chosen

# ------------------------
# HF response parsing helper
# ------------------------
def _parse_hf_response(resp):
    """Aceita list|dict e extrai generated_text ou mensagem de erro."""
    try:
        ct = resp.headers.get("content-type", "")
        data = resp.json() if "application/json" in ct else None
    except Exception:
        return None, f"Conteúdo inesperado (status {resp.status_code})."

    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"], None
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"], None
    if isinstance(data, dict) and "error" in data:
        return None, f"HF error: {data.get('error')} (status {resp.status_code})"
    return None, f"Resposta HF sem generated_text (status {resp.status_code})."

# ------------------------
# Utilidades de chunking (para /ingest)
# ------------------------
def _clean_for_storage(text: str) -> str:
    """Remove marcadores e normaliza espaços apenas para armazenar/exibir."""
    t = _clean_markers(text)
    return t

def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = _clean_for_storage(text)
    chunks = []
    i, n = 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

# ------------------------
# Rotas
# ------------------------
@app.post("/ingest")
def ingest(item: Ingest):
    """
    Divide o texto em chunks (caractere) com overlap e indexa cada chunk.
    Mantém um JSONL derivado de CONTEXT_PATH para reuso/carga futura.
    """
    raw = (item.text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Campo 'text' vazio.")

    parts = _chunk_text(raw, INGEST_CHUNK_SIZE, INGEST_OVERLAP)
    vecs = _encode_norm(parts)
    for txt, vec in zip(parts, vecs):
        _persist_new_doc(txt, vec[np.newaxis, :])  # _persist espera (1, dim)

    return {"status": "added", "chunks_added": len(parts), "total_docs": len(documents)}

@app.post("/ask")
def ask(item: Ask):
    """
    100% RAG:
      - Sempre recupera via FAISS
      - Responde por extrativo (sentença) OU gera via HF usando apenas o contexto
    Modo:
      - auto (padrão): extrativo → HF (se houver contexto) → fallbacks
      - hf   : força uso da HF; se falhar, cai para extrativo
    """
    mode = (item.mode or "auto").lower().strip()
    if mode not in {"auto", "hf"}:
        mode = "auto"

    # Sem documentos → nada a responder
    if len(documents) == 0:
        return {
            "answer": "Sem contexto disponível. Gere o cache via generate_cache.py ou use /ingest.",
            "retrieved": [],
            "model": HF_MODEL,
            "mode": "no_context"
        }

    # Recuperação semântica
    retrieved: List[dict] = []
    q_vec = _encode_norm([item.question])
    k = min(item.k or 3, len(documents))
    D, I = index.search(q_vec, k)
    for score, idx_ in zip(D[0].tolist(), I[0].tolist()):
        if idx_ == -1:
            continue
        retrieved.append({"text": documents[idx_], "score": float(score)})

    context = "\n---\n".join([r["text"] for r in retrieved]) if retrieved else ""
    context_clean = _clean_markers(context)

    # ====== HF forçada ======
    if mode == "hf":
        prompt = (
            "Você é um assistente que responde de forma objetiva usando APENAS o contexto fornecido, se houver.\n"
            "Se a resposta não estiver no contexto, diga que não há informações suficientes.\n\n"
            f"Contexto:\n{context_clean}\n\n"
            f"Pergunta: {item.question}\n"
            "Resposta:"
        )
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False, "do_sample": False},
            "options": {"wait_for_model": True, "use_cache": False}
        }
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL_PATH}"
        hf_err = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            text, err = _parse_hf_response(resp)
            if text:
                return {"answer": text, "retrieved": retrieved, "model": HF_MODEL, "mode": "hf"}
            hf_err = err or f"HTTP {resp.status_code}"
        except Exception as e:
            hf_err = str(e)

        # HF falhou → extrativo híbrido
        best = extract_short_answer(item.question, [r["text"] for r in retrieved], min_sim=0.20, max_chars=220, alpha=0.7)
        if best:
            return {"answer": best, "retrieved": retrieved, "model": "miniLM+lexical extractive", "mode": "hf-fallback-extractive"}
        return {"answer": f"Falha na HF e no extrativo. {hf_err}", "retrieved": retrieved, "model": HF_MODEL, "mode": "hf_error"}

    # ====== AUTO: extrativo → HF → fallbacks ======
    if retrieved:
        concise = extract_short_answer(item.question, [r["text"] for r in retrieved], min_sim=0.20, max_chars=240, alpha=0.7)
        if concise:
            return {"answer": concise, "retrieved": retrieved, "model": "miniLM+lexical extractive", "mode": "extractive-sent"}

    if context_clean and HF_TOKEN:
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
            "parameters": {"max_new_tokens": 200, "return_full_text": False, "do_sample": False},
            "options": {"wait_for_model": True, "use_cache": False}
        }
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL_PATH}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            text, _ = _parse_hf_response(resp)
            if text:
                return {"answer": text, "retrieved": retrieved, "model": HF_MODEL, "mode": "hf"}
        except Exception:
            pass

    if retrieved:
        best = extract_short_answer(item.question, [retrieved[0]["text"]], min_sim=0.15, max_chars=220, alpha=0.7)
        if best:
            return {"answer": best, "retrieved": retrieved, "model": "miniLM+lexical extractive", "mode": "emergency-extractive"}

    return {"answer": "Não encontrei essa informação no contexto.", "retrieved": retrieved, "model": HF_MODEL, "mode": "fallback"}

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
        "normalize_level": NORMALIZE_LEVEL,
        "strip_diacritics": STRIP_DIACRITICS,
        "ingest_chunk_size": INGEST_CHUNK_SIZE,
        "ingest_overlap": INGEST_OVERLAP,
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
