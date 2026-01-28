const AGENT_PORT = 8002;
const AGENT_NAME = 'WSB Merito';

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const statusText = document.getElementById('statusText');
const statusBar = document.getElementById('statusBar');

const WELCOME_MESSAGE = 'Cześć jestem Adam - Twój inteligętny pracownik BOS, w czym Ci mogę pomóc?';

function clearChat() {
    chatMessages.innerHTML = '';
    addMessage(WELCOME_MESSAGE, 'assistant', 'WSB Merito');
    userInput.focus();
}

clearBtn.addEventListener('click', clearChat);

function addMessage(content, type, meta = '') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = meta || (type === 'user' ? 'Student' : AGENT_NAME);

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loadingMessage';
    loadingDiv.innerHTML = `
        <div class="loading">
            <span></span><span></span><span></span>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoading() {
    const loading = document.getElementById('loadingMessage');
    if (loading) loading.remove();
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    userInput.value = '';
    sendBtn.disabled = true;
    addLoading();

    try {
        const response = await fetch(`/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input: message }),
        });

        removeLoading();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // Handle the ticket_classification response
        const result = data.ticket_classification || data.result || data.response || JSON.stringify(data, null, 2);
        addMessage(result, 'assistant');
        statusBar.classList.remove('error');
        statusText.textContent = `Ready - ${AGENT_NAME} (port ${AGENT_PORT})`;

    } catch (error) {
        removeLoading();
        let errorMsg = error.message;
        if (error.message.includes('Load failed') || error.message.includes('Failed to fetch')) {
            errorMsg = `Cannot connect to agent. Please ensure the agent is running.`;
        }
        addMessage(errorMsg, 'assistant', 'Error');
        statusBar.classList.add('error');
        statusText.textContent = `Connection failed`;
    }

    sendBtn.disabled = false;
    userInput.focus();
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Check connection on load
async function checkConnection() {
    try {
        const response = await fetch(`/health`, { method: 'GET' });
        statusBar.classList.remove('error');
        statusText.textContent = `Connected - ${AGENT_NAME} (port ${AGENT_PORT})`;
    } catch (e) {
        statusBar.classList.add('error');
        statusText.textContent = `Waiting for agent...`;
    }
}
checkConnection();
