const $ = (s) => document.querySelector(s);
const chat = $("#chat");
const input = $("#userInput");
const sendBtn = $("#sendBtn");
const clearBtn = $("#clearBtn");
const statusEl = $("#status");
const fileBtn = $("#loadBtn");
const fileInput = $("#fileInput");

// ping de saúde
async function ping() {
  try {
    const r = await fetch("/health");
    if (!r.ok) throw new Error("HTTP " + r.status);
    await r.json();
    statusEl.textContent = "pronto";
  } catch (e) {
    statusEl.textContent = "offline";
  }
}
ping();

function appendMsg(who, text) {
  const div = document.createElement("div");
  div.className = who === "user" ? "msg user" : "msg assist";
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function ask(question) {
  const r = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question, k: 3 })
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

$("#msgForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  appendMsg("user", q);
  input.value = "";

  try {
    const data = await ask(q);
    appendMsg("assist", data?.answer || "⚠️ Sem resposta.");
  } catch (err) {
    appendMsg("assist", "❌ Erro: " + err.message);
  }
});

clearBtn.addEventListener("click", async () => {
  try {
    await fetch("/clear", { method: "POST" });
    chat.innerHTML = "";
    appendMsg("assist", "🧹 Memória limpa.");
  } catch {
    appendMsg("assist", "❌ Erro ao limpar.");
  }
});

fileBtn.addEventListener("click", ()=> fileInput.click());
fileInput.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  appendMsg("assist", `Carregando ${files.length} arquivo(s) .txt…`);
  let ok = 0;
  for (const f of files) {
    const text = await f.text();
    if (text?.trim()) {
      await fetch("/ingest", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ text })
      });
      ok++;
    }
  }
  appendMsg("assist", `✅ Ingestão concluída: ${ok}/${files.length}.`);
  fileInput.value = "";
});
