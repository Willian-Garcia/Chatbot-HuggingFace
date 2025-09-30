const $ = (s) => document.querySelector(s);

// ===== Seletores alinhados com seu HTML/CSS =====
const messagesEl = $("#messages");      // <div id="messages" class="messages">
const input      = $("#userInput");
const sendBtn    = $("#sendBtn");
const clearBtn   = $("#clearBtn");
const statusEl   = $("#status");
const modeBtn    = $("#modeBtn");       // botão no topo para alternar Auto/HF

// Upload .txt para ingestão
const txtBtn     = $("#txtBtn");
const fileInput  = $("#fileInput");

// Form principal do chat
const msgForm    = $("#msgForm");

// ===== Estado de Modo (Auto ↔︎ Llama/HF) =====
const MODES = ["auto", "hf"];
let currentMode = (localStorage.getItem("chat_mode") || "auto").toLowerCase();

function modeBadge() {
  return currentMode === "hf" ? "Llama" : "Auto";
}
function setModeLabel() {
  if (modeBtn) modeBtn.textContent = `Modo: ${modeBadge()}`;
}
modeBtn?.addEventListener("click", () => {
  const idx = MODES.indexOf(currentMode);
  currentMode = MODES[(idx + 1) % MODES.length];
  localStorage.setItem("chat_mode", currentMode);
  setModeLabel();
  refreshStatus(); // atualiza status com o modo atual
});
setModeLabel();

// ===== Helpers de UI =====
function appendMsg(who, text) {
  const row = document.createElement("div");
  row.className = `msg-row ${who === "user" ? "user" : "assist"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = who === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${who === "user" ? "user" : "assist"}`;
  bubble.textContent = text;

  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  bubble.appendChild(meta);

  if (who === "user") {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  messagesEl.appendChild(row);
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
  return row;
}

// Indicador “digitando…”
function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row assist";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble assist";
  bubble.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
  return row;
}
function removeTyping(row) {
  if (row && row.parentNode) row.parentNode.removeChild(row);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function refreshStatus(h) {
  const docs = h?.docs_loaded ?? "–";
  const offline = h ? !h.hf_token_present : false;
  const base = offline ? "pronto (offline)" : "pronto";
  setStatus(`${base} · modo: ${modeBadge()} · docs: ${docs}`);
}

// ===== API =====
async function apiHealth() {
  const r = await fetch("/health");
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

async function ask(question) {
  const r = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: 3, mode: currentMode }) // Auto ou HF
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

async function ingest(text) {
  const r = await fetch("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ===== Ping de saúde =====
(async function ping() {
  try {
    const h = await apiHealth();
    refreshStatus(h);
  } catch {
    setStatus("offline");
  }
})();

// ===== Enviar pergunta (chat) =====
msgForm?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = (input?.value || "").trim();
  if (!q) return;

  appendMsg("user", q);
  if (input) input.value = "";
  if (sendBtn) sendBtn.disabled = true;

  const typingRow = showTyping();
  try {
    const data = await ask(q);
    removeTyping(typingRow);
    appendMsg("assist", data?.answer || "⚠️ Sem resposta.");

    // Fontes (opcional: mostra top-3 com score)
    if (Array.isArray(data?.retrieved) && data.retrieved.length) {
      const tops = data.retrieved
        .slice(0, 3)
        .map((r, i) => `#${i + 1} score=${Number(r.score || 0).toFixed(3)} → ${String(r.text).slice(0, 120)}…`)
        .join("\n");
      appendMsg("assist", `Fontes:\n${tops}`);
    }
  } catch (err) {
    removeTyping(typingRow);
    appendMsg("assist", "❌ Erro: " + err.message);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    input?.focus();
  }
});

// ===== Limpar (servidor + UI) =====
clearBtn?.addEventListener("click", async () => {
  try {
    await fetch("/clear", { method: "POST" });
    messagesEl.innerHTML = "";
    appendMsg("assist", "🧹 Memória limpa.");
    try { const h = await apiHealth(); refreshStatus(h); } catch {}
  } catch {
    appendMsg("assist", "❌ Erro ao limpar.");
  }
});

// ===== Upload .txt → /ingest (para adicionar contexto ao FAISS) =====
txtBtn?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;

  appendMsg("assist", `Carregando ${files.length} arquivo(s) .txt…`);
  let ok = 0;

  for (const f of files) {
    try {
      const text = await f.text();
      if (text?.trim()) {
        await ingest(text);
        ok++;
      }
    } catch (err) {
      appendMsg("assist", `❌ Erro no arquivo ${f.name}: ${err.message}`);
    }
  }

  appendMsg("assist", `✅ Ingestão concluída: ${ok}/${files.length}.`);
  if (fileInput) fileInput.value = "";

  try { const h = await apiHealth(); refreshStatus(h); } catch {}
});
