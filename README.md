# RAG Chat — FastAPI + FAISS + Hugging Face (com fallback local)

Um chatbot estilo “ChatGPT” que implementa **RAG** (Retrieval-Augmented Generation) com:

- **FastAPI** + **Uvicorn** (backend)
- **Sentence-Transformers (MiniLM)** para **embeddings**
- **FAISS** para busca semântica
- **Hugging Face Inference API** (geração — opcional)
- **Leitura inteligente por sentença** (respostas pontuais, local)
- **Q&A local** via `context_qa.jsonl` (fallback fixo, offline)
- **Frontend** simples em HTML/CSS/JS (mesma porta do backend)

O objetivo é funcionar **mesmo quando a API da Hugging Face/Meta Llama não estiver disponível**, sem travar a experiência do usuário.

---

## Sumário

- [Arquitetura](#arquitetura)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração (.env)](#configuração-env)
- [Executando](#executando)
- [Frontend (chat)](#frontend-chat)
- [Endpoints da API](#endpoints-da-api)
- [Fluxo de resposta (prioridade)](#fluxo-de-resposta-prioridade)
- [Conteúdos e ingestão](#conteúdos-e-ingestão)
- [Q&A local (context_qa.jsonl)](#qa-local-context_qajsonl)
- [Exemplos de uso (curl)](#exemplos-de-uso-curl)
- [Resolução de problemas](#resolução-de-problemas)
- [Boas práticas de segurança](#boas-práticas-de-segurança)
- [Estrutura de pastas](#estrutura-de-pastas)

---

## Arquitetura

```txt
┌─────────────────────────────────────────────────────────────────────┐
│ Frontend (index.html, script.js, styles.css)                        │
│  - Chat UI (input, enviar, limpar, carregar .txt)                   │
│  - Faz fetch para /ask, /ingest, /clear, /health                    │
└───────────────▲─────────────────────────────────────────────────────┘
                │
          HTTP (mesma porta)
                │
┌───────────────┴───────────────── FastAPI (app_with_frontend.py) ──────────────────────────┐
│ /ask           → Orquestra respostas                                                       │
│     1) Q&A local (context_qa.jsonl)                                                        │
│     2) Intents (regex pontuais)                                                            │
│     3) Extrativo por sentença (MiniLM + FAISS)                                             │
│     4) Hugging Face Inference API (se disponível)                                          │
│     5) Fallback curto (“Não encontrei…”)                                                   │
│ /ingest        → Indexa textos no FAISS                                                    │
│ /clear         → Limpa memória/índice e caches                                             │
│ /health        → Status geral (modelo HF, docs, arquivo local QA)                          │
│ /hf_test       → Ping à Inference API do modelo configurado                                 │
│ /qa_debug      → (opcional) Debug de match no Q&A local                                    │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Requisitos

- Python 3.10+  
- Pip atualizado

Dependências (sugestão de `requirements.txt`):

```txt
fastapi==0.111.*
uvicorn[standard]==0.30.*
sentence-transformers==2.7.*
faiss-cpu==1.8.*
numpy==1.26.*
requests==2.32.*
python-dotenv==1.0.*
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
HF_TOKEN=hf_xxx_sua_chave_aqui
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct
CONTEXT_PATH=context.txt
FAISS_INDEX_PATH=faiss_index.bin
LOCAL_QA_PATH=context_qa.jsonl
```

- **HF_TOKEN**: chave da Hugging Face (obrigatória para /hf_test e geração).  
- **HF_MODEL**: se a Inference API do Llama retornar 404, use temporariamente:
  - `Qwen/Qwen2.5-0.5B-Instruct` ou
  - `google/gemma-2-2b-it`

---

## Executando

```bash
uvicorn app_with_frontend:app --reload --port 8000
```

Abra no navegador:  
**http://127.0.0.1:8000/**

---

## Frontend (chat)

- Campo de mensagem + botões: **Enviar**, **Limpar**, **Carregar** `.txt`
- **Carregar**: seleciona um ou mais `.txt` → chama `/ingest`.
- **Limpar**: chama `/clear`.

---

## Endpoints da API

- `GET /` → entrega o frontend (`/frontend/index.html`)
- `GET /health` → status (modelo HF, docs carregados, Q&A local)
- `POST /ingest` → `{ "text": "conteúdo a indexar" }`
- `POST /ask` → `{ "question": "sua pergunta", "k": 3 }`
- `POST /clear` → zera memória/índice/cache
- `GET /hf_test` → testa a Inference API
- `GET /qa_debug?q=pergunta` → debug Q&A local

---

## Fluxo de resposta (prioridade)

1. **Q&A Local (`context_qa.jsonl`)**  
2. **Intents (regex)**  
3. **Extrativo por sentença** (MiniLM + FAISS)  
4. **Hugging Face Inference API**  
5. **Fallback curto**

---

## Conteúdos e ingestão

**Ingestão manual via UI**: botão **Carregar** aceita `.txt`.  
**Via API**:

```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{"text":"A FastAPI é um framework web moderno em Python."}'
```

---

## Q&A local (`context_qa.jsonl`)

Exemplo:

```jsonl
{"pergunta": "qual é o nome completo do curso dsm?", "resposta": "Curso Superior de Tecnologia em Desenvolvimento de Software Multiplataforma (DSM)."}
{"pergunta": "quantos semestres tem o curso de dsm?", "resposta": "O curso tem 6 semestres (3 anos)."}
```

Verifique `/health` → `local_qa_count`.

---

## Exemplos de uso (curl)

**Perguntar**

```bash
curl -s -X POST http://127.0.0.1:8000/ask   -H "Content-Type: application/json"   -d '{"question":"Quantos semestres tem o curso de DSM?","k":3}'
```

**Limpar**

```bash
curl -s -X POST http://127.0.0.1:8000/clear
```

---

## Resolução de problemas

- **404 no modelo HF**: use `Qwen/Qwen2.5-0.5B-Instruct` ou `google/gemma-2-2b-it`.  
- **Chat preso em “conectando”**: abra `/health`, faça hard refresh.  
- **Resposta despeja contexto**: verifique `mode` no retorno do `/ask`.  
- **Q&A local não carrega**: cheque `local_qa_count` no `/health`.

---

## Boas práticas de segurança

- Adicione `.env` ao `.gitignore`.  
- Se vazar token → remova do histórico (`git filter-repo`) e revogue no Hugging Face.  
- Em produção: restrinja CORS e use HTTPS.

---

## Estrutura de pastas

```txt
.
├─ app_with_frontend.py         # API + frontend
├─ requirements.txt
├─ .env                         # credenciais (NÃO commitar)
├─ context.txt                  # base de ingestão
├─ context.txt.chunks.jsonl     # cache
├─ faiss_index.bin              # índice
├─ context_qa.jsonl             # Q&A local (fallback)
└─ frontend/
   ├─ index.html
   ├─ script.js
   └─ styles.css
```

---

## Como o código funciona

1. **Embeddings & FAISS**: gera vetores com MiniLM e armazena no FAISS.  
2. **/ask**: decide entre Q&A local, intents, extrativo, Hugging Face ou fallback.  
3. **Frontend**: UI simples, mesma porta do backend.

---
