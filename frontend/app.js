const textField = document.getElementById("textField");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const chatBody = document.getElementById("chatBody");
const messageStack = document.getElementById("messageStack");

const CONVERSATION_ID = crypto.randomUUID();

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// Tracks the currently-playing Gemini audio response, if any, so a new
// recording can stop it immediately instead of letting it play over the
// user's next input.
let currentAudio = null;

function addBubble(text, role) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  messageStack.appendChild(bubble);
  chatBody.scrollTop = chatBody.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant typing-indicator";
  bubble.innerHTML = `<span></span><span></span><span></span>`;
  messageStack.appendChild(bubble);
  chatBody.scrollTop = chatBody.scrollHeight;
  return bubble;
}

function tryAutoplay(url) {
  // Stop whatever's currently playing before starting the new response 
  stopCurrentAudio();

  const audio = new Audio(url);
  currentAudio = audio;

  audio.addEventListener("ended", () => {
    if (currentAudio === audio) currentAudio = null;
  });

  audio.play().catch((err) => {
    // Expected sometimes -- browser autoplay policy
    console.warn("Autoplay was blocked (this is normal browser behavior):", err);
  });
}

function stopCurrentAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
}

async function sendText() {
  const text = textField.textContent.trim();
  if (!text) return;
  addBubble(text, "user");
  textField.textContent = "";

  const typingBubble = addTypingIndicator();

  try {
    const formData = new FormData();
    formData.append("conversation_id", CONVERSATION_ID);
    formData.append("message", text);

    const response = await fetch("/chat/text", { method: "POST", body: formData });
    if (!response.ok) throw new Error("Request failed");
    const data = await response.json();

    typingBubble.remove();
    addBubble(data.agent_text, "assistant");
  } catch (err) {
    typingBubble.remove();
    addBubble("Something went wrong sending that. Please try again.", "assistant");
  }
}

sendBtn.addEventListener("click", sendText);

textField.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});

async function startRecording() {
  // Interruption handling: if Gemini's still speaking when the user
  // starts a new recording, stop it immediately rather than letting it
  // keep playing under/over the new interaction.
  stopCurrentAudio();

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    addBubble("I need microphone access to record a voice note.", "assistant");
    return;
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  mediaRecorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  });

  mediaRecorder.addEventListener("stop", () => {
    stream.getTracks().forEach((track) => track.stop());
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    sendVoiceNote(audioBlob);
  });

  mediaRecorder.start();
  isRecording = true;
  micBtn.classList.add("recording");
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  micBtn.classList.remove("recording");
}

async function sendVoiceNote(audioBlob) {
  const placeholder = addBubble("Transcribing...", "user");
  const typingBubble = addTypingIndicator();

  const formData = new FormData();
  formData.append("conversation_id", CONVERSATION_ID);
  formData.append("audio", audioBlob, "voice-note.webm");

  try {
    const response = await fetch("/chat/voice", { method: "POST", body: formData });
    if (!response.ok) throw new Error("Request failed");
    const data = await response.json();

    typingBubble.remove();

    if (data.type === "fallback" || data.type === "error") {
      placeholder.remove();
      addBubble(data.agent_text, "assistant");
      return;
    }

    placeholder.textContent = data.user_text || "(voice note)";
    addBubble(data.agent_text, "assistant");

    if (data.audio_url) {
      tryAutoplay(data.audio_url);
    }
  } catch (err) {
    typingBubble.remove();
    placeholder.remove();
    addBubble("Something went wrong sending that. Please try again.", "assistant");
  }
}

micBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});