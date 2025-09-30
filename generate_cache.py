from __future__ import annotations
from pathlib import Path
import os, re, json, argparse
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import unicodedata as ud

load_dotenv()

CONTEXT_PATH      = os.getenv("CONTEXT_PATH", "context.txt")
FAISS_INDEX_PATH  = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "120"))
BATCH_SIZE        = int(os.getenv("EMBED_BATCH_SIZE", "32"))  # menor por padrão
WRITE_MAP         = os.getenv("WRITE_MAP", "1") == "1"
MIN_CHARS         = int(os.getenv("MIN_CHARS", "120"))
MAX_CHARS_HARD    = int(os.getenv("MAX_CHARS_HARD", "1200"))

NORMALIZE_LEVEL   = (os.getenv("NORMALIZE_LEVEL", "light") or "light").lower()  # none|light|aggressive
STRIP_DIACRITICS  = os.getenv("STRIP_DIACRITICS", "0") == "1"

PUNCT_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "–": "-", "—": "-", "−": "-", "‐": "-",
    "…": "...", "·": ".", "•": "-",
    "\u00A0": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ", "\u2007": " ",
})
CONTROL_RX   = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]")
SPACES_RX    = re.compile(r"[ \t]+")
MULTI_NL_RX  = re.compile(r"\n{2,}")
SENT_SPLIT_RX = re.compile(r"(?<=[\.\?\!…:;])\s+(?=[A-ZÁÂÃÀÉÊÍÓÔÕÚÜ0-9“\"'\(])")

def _strip_diacritics(s: str) -> str:
    nfd = ud.normalize("NFD", s)
    return "".join(ch for ch in nfd if ud.category(ch) == "Mn")

def normalize_for_embedding(s: str) -> str:
    if not s: return s
    s = ud.normalize("NFKC", s).translate(PUNCT_MAP)
    s = CONTROL_RX.sub("", s)
    if STRIP_DIACRITICS: s = _strip_diacritics(s)
    if NORMALIZE_LEVEL == "aggressive": s = s.lower()
    return SPACES_RX.sub(" ", s).strip()

def clean_text_for_display(s: str) -> str:
    if not s: return s
    s = s.replace("\ufeff","").replace("\u200b","")
    s = SPACES_RX.sub(" ", s).strip()
    return MULTI_NL_RX.sub("\n\n", s)

def split_sentences(block: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
    sents: List[str] = []
    for p in paras:
        parts = SENT_SPLIT_RX.split(p)
        for part in parts:
            t = part.strip()
            if t: sents.append(t)
    return sents

def pack_sentences_into_chunks(sents: List[str],
                               target: int = CHUNK_SIZE,
                               overlap: int = CHUNK_OVERLAP,
                               min_chars: int = MIN_CHARS,
                               max_chars_hard: int = MAX_CHARS_HARD) -> List[str]:
    if not sents: return []
    chunks: List[str] = []
    i = 0
    while i < len(sents):
        curr = []; total = 0; j = i
        while j < len(sents) and total + len(sents[j]) + (1 if total else 0) <= max(target, min_chars):
            curr.append(sents[j]); total += len(sents[j]) + (1 if total else 0); j += 1
        if not curr:
            big = sents[j] if j < len(sents) else sents[-1]
            chunks.append(big[:max_chars_hard].rstrip()); i = j + 1; continue
        chunk = " ".join(curr).strip()
        if len(chunk) > max_chars_hard:
            chunk = chunk[:max_chars_hard].rsplit(" ", 1)[0].rstrip()
        chunks.append(chunk)
        if j >= len(sents): break
        back_chars = 0; k = len(curr) - 1; keep = 0
        while k >= 0 and back_chars < overlap:
            back_chars += len(curr[k]) + 1; keep += 1; k -= 1
        i = j - min(keep, len(curr))
    return [c for c in chunks if len(c) >= min_chars or len(c.split()) >= 8]

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen=set(); out=[]
    for it in items:
        key = it.strip()
        if key and key not in seen:
            seen.add(key); out.append(it)
    return out

def build_chunks(text: str) -> List[str]:
    cleaned = clean_text_for_display(text)
    blocks = [b.strip() for b in re.split(r"\n{2,}", cleaned) if b.strip()]
    all_sents: List[str] = []
    for b in blocks: all_sents.extend(split_sentences(b))
    chunks = pack_sentences_into_chunks(all_sents, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS, MAX_CHARS_HARD)
    return dedupe_preserve_order(chunks)

def parse_args():
    ap = argparse.ArgumentParser(description="Gera chunks JSONL e índice FAISS (low-memory).")
    ap.add_argument("--context", default=CONTEXT_PATH)
    ap.add_argument("--index",   default=FAISS_INDEX_PATH)
    ap.add_argument("--model",   default=EMBED_MODEL)
    ap.add_argument("--chunk",   type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--batch",   type=int, default=BATCH_SIZE)
    ap.add_argument("--no-map",  action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()

    # aplique flags à config global utilizada pelo chunking
    global CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE
    CHUNK_SIZE    = args.chunk
    CHUNK_OVERLAP = args.overlap
    BATCH_SIZE    = args.batch

    ctx_path = Path(args.context)
    if not ctx_path.exists():
        raise SystemExit(f"[ERRO] {ctx_path} não encontrado.")

    raw = ctx_path.read_text(encoding="utf-8")
    chunks_display = build_chunks(raw)
    if not chunks_display:
        raise SystemExit("[ERRO] Nenhum chunk gerado.")

    # prepara arquivos de saída
    chunks_path = ctx_path.with_suffix(ctx_path.suffix + ".chunks.jsonl")
    map_path    = ctx_path.with_suffix(ctx_path.suffix + ".map.jsonl")

    # cria FAISS vazio e embeda em streaming
    print(f"[INFO] Chunks: {len(chunks_display)}  |  modelo: {args.model}  |  batch: {args.batch}")
    embedder = SentenceTransformer(args.model)
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    with chunks_path.open("w", encoding="utf-8") as fjsonl:
        if WRITE_MAP and not args.no_map:
            fmap = map_path.open("w", encoding="utf-8")
        else:
            fmap = None

        # embeda em lotes pequenos, escreve JSONL por linha e adiciona vetores direto no FAISS
        norm_buf: List[str] = []
        disp_buf: List[str] = []
        for i, chunk in enumerate(chunks_display, 1):
            disp_buf.append(chunk)
            norm_buf.append(normalize_for_embedding(chunk))

            if len(norm_buf) >= BATCH_SIZE or i == len(chunks_display):
                # encode
                vecs = embedder.encode(norm_buf, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                index.add(vecs.astype("float32"))

                # escreve este lote no .jsonl
                for disp, norm in zip(disp_buf, norm_buf):
                    fjsonl.write(json.dumps({"text": disp, "norm": norm}, ensure_ascii=False) + "\n")

                # escreve map (opcional)
                if fmap:
                    for j, disp in enumerate(disp_buf):
                        fmap.write(json.dumps({"i": i - len(disp_buf) + j, "chars": len(disp), "approx_words": len(disp.split())}, ensure_ascii=False) + "\n")

                # limpa buffers
                norm_buf.clear(); disp_buf.clear()

                # feedback leve
                if i % max(100, BATCH_SIZE) == 0:
                    print(f"[INFO] Processados {i}/{len(chunks_display)} chunks…")

        if fmap:
            fmap.close()

    faiss.write_index(index, args.index)
    print(f"[OK] {chunks_path.name} salvo")
    if WRITE_MAP and not args.no_map:
        print(f"[OK] {map_path.name} salvo")
    print(f"[OK] Índice FAISS salvo em {args.index}")

if __name__ == "__main__":
    # limitar threads (evita travar máquinas com pouca RAM/CPU)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
