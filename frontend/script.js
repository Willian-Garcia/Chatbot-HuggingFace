// Simple chat client for RAG backend (no explicit modes)
// Heuristic: long or multi-paragraph input -> ingest; otherwise -> ask
// Backend base URL (adjust if needed)
const BASE_URL = "http://localhost:8000";

const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const statusEl = $("#status");
const form = $("#msgForm");
const input = $("#msg");

let typingEl = null;

async function ping() {
  try {
    const res = await fetch(BASE_URL + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "ping", k: 1 }),
    });
    statusEl.textContent = res.ok ? "pronto" : "offline";
  } catch {
    statusEl.textContent = "offline";
  }
}
ping();

function addMsg(text, who = "assist") {
  const div = document.createElement("div");
  div.className = "msg " + (who === "user" ? "user" : "assist");
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addTyping() {
  if (typingEl) return typingEl;
  typingEl = document.createElement("div");
  typingEl.className = "msg assist typing";
  typingEl.textContent = "digitando…";
  messagesEl.appendChild(typingEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return typingEl;
}

function removeTyping() {
  if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

function isIngest(text) {
  // Heurística simples: texto muito longo ou vários parágrafos
  const tooLong = text.length > 280;
  const manyBreaks = (text.match(/\n/g) || []).length >= 2;
  return tooLong || manyBreaks;
}

async function sendIngest(text) {
  const res = await fetch(BASE_URL + "/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  return data;
}

async function sendAsk(question) {
  const res = await fetch(BASE_URL + "/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: 3 }),
  });
  const data = await res.json();
  return data;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMsg(text, "user");
  input.value = "";

  addTyping();
  try {
    if (isIngest(text)) {
      const data = await sendIngest(text);
      removeTyping();
      addMsg(`Adicionado ao contexto. Total de docs: ${data?.total_docs ?? "?"}`);
    } else {
      const data = await sendAsk(text);
      removeTyping();
      const answer = data?.answer || "(sem resposta)";
      addMsg(answer, "assist");

      if (Array.isArray(data?.retrieved) && data.retrieved.length) {
        const ctx = data.retrieved
          .map((r, i) => `#${i+1} (score ${Number(r.score).toFixed(3)}):\n${r.text}`)
          .join("\n\n—\n\n");
        const note = document.createElement("div");
        note.className = "msg assist small";
        const pre = document.createElement("pre");
        pre.className = "code";
        pre.textContent = ctx;
        note.appendChild(pre);
        messagesEl.appendChild(note);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
  } catch (err) {
    removeTyping();
    addMsg("Erro: " + err, "assist");
  }
});
