const AGENT_PORT = 8002;
const AGENT_NAME = "WSB Merito";

const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusText = document.getElementById("statusText");
const statusBar = document.getElementById("statusBar");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

const WELCOME_MESSAGE = "Cześć! Jestem asystentem WSB Merito. Jak mogę pomóc?";
const RETURNING_MESSAGE = "Witaj ponownie! W czym mogę Ci pomóc?";

let sessionId = localStorage.getItem('wsb_session_id');
let dataCollectionComplete = true;

async function clearChat() {
  chatMessages.innerHTML = "";

  // If we have a session with completed data collection, just clear history
  if (sessionId && dataCollectionComplete) {
    try {
      const response = await fetch(`/clear_history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });

      if (response.ok) {
        addMessage(RETURNING_MESSAGE, "assistant", "WSB Merito", false);
      }
    } catch (error) {
      console.error("Error clearing history:", error);
    }
  } else {
    // New session - show welcome message
    addMessage(WELCOME_MESSAGE, "assistant", "WSB Merito", false);
    sessionId = null;
    dataCollectionComplete = true;
    localStorage.removeItem('wsb_session_id');
    localStorage.removeItem('wsb_data_complete');
  }

  userInput.focus();
}

clearBtn.addEventListener("click", clearChat);
uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", uploadSelectedFile);

function addMessage(content, type, meta = "", isDataCollection = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}`;
  if (isDataCollection) {
    messageDiv.setAttribute("data-collection", "true");
  }

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = content;

  const metaDiv = document.createElement("div");
  metaDiv.className = "message-meta";
  metaDiv.textContent = meta || (type === "user" ? "Student" : AGENT_NAME);

  messageDiv.appendChild(contentDiv);
  messageDiv.appendChild(metaDiv);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearDataCollectionMessages() {
  // Remove all messages marked as data collection
  const dataCollectionMsgs = chatMessages.querySelectorAll('[data-collection="true"]');
  dataCollectionMsgs.forEach(msg => msg.remove());
}

function startStatusSequence() {
  const msgDiv = document.createElement("div");
  msgDiv.className = "message assistant";

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = "Szukam informacji...";

  const metaDiv = document.createElement("div");
  metaDiv.className = "message-meta";
  metaDiv.textContent = AGENT_NAME;

  msgDiv.appendChild(contentDiv);
  msgDiv.appendChild(metaDiv);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const timers = [];
  timers.push(setTimeout(() => {
    contentDiv.textContent = "Analizuję zebrane informacje...";
  }, 1800));
  timers.push(setTimeout(() => {
    contentDiv.textContent = "Generuję odpowiedź...";
  }, 3600));

  return { msgDiv, contentDiv, timers };
}

function clearStatusSequence(seq) {
  if (!seq) return;
  (seq.timers || []).forEach(t => clearTimeout(t));
}

async function uploadSelectedFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  uploadStatus.textContent = `Wysyłam: ${file.name}`;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`/documents/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const msg = data.status === "existing"
      ? `📁 Podobny dokument już w bazie (score ${(data.match_score || 0).toFixed(2)}). Źródło: ${data.match_source || "brak"}.`
      : `✅ Dokument dodany (kategoria: ${data.category || "nieznana"}). ID: ${data.doc_id || "n/a"}`;

    addMessage(msg, "assistant", AGENT_NAME, false);
    uploadStatus.textContent = data.status === "existing" ? "Dokument już istnieje" : "Dodano do bazy";
  } catch (error) {
    uploadStatus.textContent = "Błąd podczas wysyłki";
    addMessage(`Błąd wysyłki pliku: ${error.message}`, "assistant", "Error", false);
  } finally {
    fileInput.value = "";
  }
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user", "", !dataCollectionComplete);
  userInput.value = "";
  sendBtn.disabled = true;
  userInput.disabled = true;
  clearBtn.disabled = true;
  const statusSeq = startStatusSequence();

  try {
    const payload = { input: message };
    if (sessionId) {
      payload.session_id = sessionId;
    }

    const response = await fetch(`/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    sessionId = data.session_id;

    // Save session to localStorage
    if (sessionId) {
      localStorage.setItem('wsb_session_id', sessionId);
    }

    const result = data.response || JSON.stringify(data, null, 2);

    // Check if we should clear data collection messages
    if (data.clear_previous && !dataCollectionComplete) {
      clearDataCollectionMessages();
      dataCollectionComplete = true;
      localStorage.setItem('wsb_data_complete', 'true');
      if (statusSeq && statusSeq.contentDiv) {
        statusSeq.contentDiv.textContent = result;
      } else {
        addMessage(result, "assistant", "", false);
      }
    } else {
      if (statusSeq && statusSeq.contentDiv) {
        statusSeq.contentDiv.textContent = result;
      } else {
        addMessage(result, "assistant", "", !data.data_collection_complete);
      }
    }
    clearStatusSequence(statusSeq);

    // Update data collection status
    if (data.data_collection_complete) {
      dataCollectionComplete = true;
      localStorage.setItem('wsb_data_complete', 'true');
    }

    statusBar.classList.remove("error");
    statusText.textContent = `Ready - ${AGENT_NAME} (port ${AGENT_PORT})`;
  } catch (error) {
    clearStatusSequence(statusSeq);
    let errorMsg = error.message;
    if (
      error.message.includes("Load failed") ||
      error.message.includes("Failed to fetch")
    ) {
      errorMsg = `Cannot connect to agent. Please ensure the agent is running.`;
    }
    if (statusSeq && statusSeq.contentDiv) {
      statusSeq.contentDiv.textContent = errorMsg;
    } else {
      addMessage(errorMsg, "assistant", "Error");
    }
    statusBar.classList.add("error");
    statusText.textContent = `Connection failed`;
  } finally {
    clearStatusSequence(statusSeq);
    sendBtn.disabled = false;
    userInput.disabled = false;
    clearBtn.disabled = false;
    userInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.isComposing) return;
  e.preventDefault();
  if (userInput.disabled || sendBtn.disabled) return;
  sendMessage();
});

async function checkConnection() {
  try {
    const response = await fetch(`/health`, { method: "GET" });
    statusBar.classList.remove("error");
    statusText.textContent = `Connected - ${AGENT_NAME} (port ${AGENT_PORT})`;
  } catch (e) {
    statusBar.classList.add("error");
    statusText.textContent = `Waiting for agent...`;
  }
}

async function restoreSession() {
  // Check if we have a valid session
  if (sessionId && dataCollectionComplete) {
    try {
      const response = await fetch(`/session/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.data_collection_complete) {
          addMessage(RETURNING_MESSAGE, "assistant", "WSB Merito", false);
          return;
        }
      }
    } catch (error) {
      console.error("Error restoring session:", error);
    }
  }

  // No valid session - show welcome message only if chat is empty
  if (chatMessages.children.length === 0) {
    addMessage(WELCOME_MESSAGE, "assistant", "WSB Merito", false);
  }
}

checkConnection();
restoreSession();
