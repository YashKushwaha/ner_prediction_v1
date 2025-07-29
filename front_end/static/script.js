function scrollToBottom(scrollContainer) {
  scrollContainer.scrollTop = scrollContainer.scrollHeight;
}

// Utility: Create and append a user message
function appendUserMessage(message, chatHistory, imageFile = null) {
  // --- Text Message ---
  if (message && message.trim() !== "") {
    const textDiv = document.createElement("div");
    textDiv.className = "user-message";
    textDiv.innerHTML = message.replace(/\n/g, "<br>");
    chatHistory.appendChild(textDiv);
  }
}

function appendServerMessage(markdownText, chatHistory) {
  const replyMsg = document.createElement("div");
  replyMsg.className = "server-message";

  // Step 1: Use marked to parse markdown
  let htmlContent = marked.parse(markdownText || "No response");

  // Step 3: Set HTML and highlight
  replyMsg.innerHTML = htmlContent;

    // Highlight any <pre><code> blocks after inserting HTML
   replyMsg.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
    });

  chatHistory.appendChild(replyMsg);
  scrollToBottom(scrollContainer);
}

async function sendMessageToBackend(message) {
  const formData = new FormData();
  formData.append("message", message);

  try {
    const response = await fetch('/ner_predict', {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    return data; 
  } catch (err) {
    console.error("Send failed", err);
    return null;
  }
}

async function sendMessageToBackendStream(message, chatHistory) {
  const formData = new FormData();
  formData.append("message", message);
 
  try {
    const response = await fetch('/ner_predict', {
      method: "POST",
      body: formData,
    });

    if (!response.ok || !response.body) {
      throw new Error("Network or server error");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let replyMsg = document.createElement("div");
    replyMsg.className = "server-message";
    chatHistory.appendChild(replyMsg);

    let markdownBuffer = "";

    // Read streamed data
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      markdownBuffer += chunk;

      // Render and update innerHTML with parsed markdown
      const htmlContent = marked.parse(markdownBuffer);
      replyMsg.innerHTML = htmlContent;

      // Highlight newly added code blocks
      replyMsg.querySelectorAll("pre code:not(.hljs)").forEach((block) => {
        hljs.highlightElement(block);
      });

      scrollToBottom(scrollContainer);
    }


  } catch (err) {
    console.error("Streaming failed", err);
  }
}

async function handleUserInput(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();

    const message = inputDiv.innerText.trim();    
    inputDiv.innerText = "";
    if (!message) return;
    console.log(message);
    appendUserMessage(message, chatHistory);  
    await sendMessageToBackendStream(message, chatHistory);


  }
}

async function getTagList() {
  try {
    const response = await fetch('/tag_list');
    if (!response.ok) {
      throw new Error('HTTP error! status: ' + response.status);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
  }
}


const tags = await getTagList();

const palette = chroma.scale('Set2').colors(tags.length);


const colorMap = {};
tags.forEach((tag, i) => {
  colorMap[tag] = palette[i];
});

const style = document.createElement('style');
document.head.appendChild(style);

Object.entries(colorMap).forEach(([tag, color]) => {
  if (tag === 'O') return;
  style.sheet.insertRule(
    `.ner-tag.${tag} { background-color: ${color}; color: #000; cursor:pointer;}`,
    style.sheet.cssRules.length
  );
});


const inputDiv = document.getElementById("user-input-div");
const chatHistory = document.getElementById("chat-history");
const scrollContainer = document.getElementById("chat-history-container");

inputDiv.addEventListener("keydown", handleUserInput);

