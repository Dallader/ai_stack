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

const WELCOME_MESSAGE = "Cześć! Witaj w systemie WSB Merito. Przed rozpoczęciem, potrzebuję kilku informacji. Jak masz na imię?";
const RETURNING_MESSAGE = "Witaj ponownie! W czym mogę Ci pomóc?";

let sessionId = localStorage.getItem('wsb_session_id');
let dataCollectionComplete = localStorage.getItem('wsb_data_complete') === 'true';

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
    addMessage(WELCOME_MESSAGE, "assistant", "WSB Merito", true);
    sessionId = null;
    dataCollectionComplete = false;
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

function addLoading() {
  const loadingDiv = document.createElement("div");
  loadingDiv.className = "message assistant";
  loadingDiv.id = "loadingMessage";
  loadingDiv.innerHTML = `
        <div class="loading">
            <span></span><span></span><span></span>
        </div>
    `;
  chatMessages.appendChild(loadingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoading() {
  const loading = document.getElementById("loadingMessage");
  if (loading) loading.remove();
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
  addLoading();

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

    removeLoading();

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
      addMessage(result, "assistant", "", false);
    } else {
      addMessage(result, "assistant", "", !data.data_collection_complete);
    }

    // Update data collection status
    if (data.data_collection_complete) {
      dataCollectionComplete = true;
      localStorage.setItem('wsb_data_complete', 'true');
    }

    statusBar.classList.remove("error");
    statusText.textContent = `Ready - ${AGENT_NAME} (port ${AGENT_PORT})`;
  } catch (error) {
    removeLoading();
    let errorMsg = error.message;
    if (
      error.message.includes("Load failed") ||
      error.message.includes("Failed to fetch")
    ) {
      errorMsg = `Cannot connect to agent. Please ensure the agent is running.`;
    }
    addMessage(errorMsg, "assistant", "Error");
    statusBar.classList.add("error");
    statusText.textContent = `Connection failed`;
  } finally {
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
    addMessage(WELCOME_MESSAGE, "assistant", "WSB Merito", true);
  }
}

checkConnection();
restoreSession();
