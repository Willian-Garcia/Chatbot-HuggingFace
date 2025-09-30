# RAG Chat — FastAPI + FAISS + MiniLM (com Hugging Face opcional)

Um chatbot estilo “ChatGPT” que implementa **RAG** (Retrieval-Augmented Generation) com:

- **FastAPI** + **Uvicorn** (backend)  
- **Sentence-Transformers (MiniLM)** para **embeddings**  
- **FAISS** para busca semântica  
- **Hugging Face Inference API** (opcional, fallback generativo)  
- **Extrativo por sentença** (MiniLM + FAISS)  
- **Frontend** simples em HTML/CSS/JS (mesma porta do backend)  

O objetivo é funcionar **mesmo sem conexão com a Hugging Face**, mantendo respostas locais via embeddings.

---

## Sumário

- [Arquitetura](#arquitetura)  
- [Requisitos](#requisitos)  
- [Instalação](#instalação)  
- [Configuração (.env)](#configuração-env)  
- [Gerando contexto](#gerando-contexto)  
- [Executando](#executando)
- [Como funciona](#como-o-código-funciona)  
- [Frontend (chat)](#frontend-chat)  
- [Endpoints da API](#endpoints-da-api)  
- [Fluxo de resposta](#fluxo-de-resposta)  
- [Exemplos de uso](#exemplos-de-uso)  
- [Resolução de problemas](#resolução-de-problemas)  
- [Estrutura de pastas](#estrutura-de-pastas) 

---

## Arquitetura

```txt
┌───────────────────────────────────────────────┐
│ Frontend (index.html, script.js, styles.css)  │
│  - UI simples de chat                         │
│  - Consome API: /ask, /ingest, /clear, /health│
└───────────────▲───────────────────────────────┘
                │ HTTP
┌───────────────┴──────────── FastAPI ──────────┐
│ /ask     → Busca contexto + gera resposta      │
│ /ingest  → Indexa textos no FAISS              │
│ /clear   → Limpa memória e cache               │
│ /health  → Status geral do servidor            │
│ /hf_test → Testa integração com Hugging Face   │
└───────────────────────────────────────────────┘
```

---

## Requisitos

- Python 3.10+  
- Pip atualizado

Dependências (sugestão de `requirements.txt`):

```txt
fastapi
uvicorn
sentence-transformers
faiss-cpu
requests
numpy
python-dotenv
transformers
torch
```

---

## Instalação

```bash
# clone o seu repositório
git clone <seu-repo>.git
cd <seu-repo>

# crie venv (opcional, recomendado)
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

# instale dependências
pip install -r requirements.txt
```

---

## Configuração (.env)

Crie um arquivo `.env` na raiz:

```ini
HF_TOKEN=sua_chave_aqui
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct

CONTEXT_PATH=context.txt
FAISS_INDEX_PATH=faiss_index.bin

EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=600
CHUNK_OVERLAP=100
EMBED_BATCH_SIZE=64

NORMALIZE_LEVEL=light     
STRIP_DIACRITICS=0        

ONNX_MODEL_PATH=t5_small.onnx

---

## Gerando contexto em chunks

```bash
python generate_cache.py --context context.txt --index faiss_index.bin --chunk 600 --overlap 80 --batch 8 --no-map
```
---

## Executando

```bash
uvicorn app_with_frontend:app --reload --port 8000
```

Abra no navegador:  
**http://127.0.0.1:8000/frontend**

---

## Como o código funciona

1. **Embeddings & FAISS** → gera vetores com MiniLM e armazena no índice FAISS  
2. **/ask** → decide entre label-value, extrativo, Hugging Face ou fallback  
3. **Frontend** → interface simples de chat que consome a API  

---

## Frontend (chat)

- Campo de mensagem + botões: **Enviar**, **Limpar**, **Carregar** `.txt`
- **Carregar**: seleciona um ou mais `.txt` → chama `/ingest`.
- **Limpar**: chama `/clear`.

---

## Endpoints da API

- `GET /` → entrega `index.html` (chat)  
- `POST /ask` → pergunta (`{"question":"..."}`)  
- `POST /ingest` → adiciona novos textos  
- `POST /clear` → limpa memória/índice  
- `GET /health` → status do servidor  
- `GET /hf_test` → valida integração com Hugging Face  

---

## Fluxo de resposta (prioridade)

1. Busca semântica no **FAISS** (MiniLM)  
2. **Label-Value Extraction** (quando aplicável)  
3. **Extrativo por sentença**  
4. **Hugging Face API** (se configurado)  
5. Fallback: *“Não encontrei essa informação”*  

---

## Exemplos de uso

```bash
# Adicionar texto ao índice
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"text":"A FastAPI é um framework em Python."}'

# Fazer pergunta
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"O que é FastAPI?"}'
```

---

## Resolução de problemas

- **Erro 405 no /ask**: verifique se o método está correto (`POST`).  
- **Frontend não abre**: use `http://127.0.0.1:8000` (não `/frontend`).  
- **Sem resposta**: verifique se o contexto foi carregado (`/health`).  
- **Erro na HF API**: cheque `HF_TOKEN` no `.env`.  

---

## Estrutura de pastas

```txt
.
├─ app_with_frontend.py     # Backend + Frontend
├─ generate_cache.py        # Script para gerar FAISS
├─ requirements.txt
├─ .env.example
├─ context.txt              # Base de ingestão
├─ context.txt.chunks.jsonl # Cache dos chunks
├─ faiss_index.bin          # Índice FAISS
└─ frontend/
   ├─ index.html
   ├─ script.js
   └─ styles.css
```

---
