const textField = document.getElementById("textField");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const chatBody = document.getElementById("chatBody");
const messageStack = document.getElementById("messageStack");
const launcherBtn = document.getElementById("launcherBtn");
const minimizeBtn = document.getElementById("minimizeBtn");
const chatWidget = document.getElementById("chatWidget");
const connectionBanner = document.getElementById("connectionBanner");
const connectingBubble = document.getElementById("connectingBubble");
const liveCallBtn = document.getElementById("liveCallBtn");

const CONVERSATION_ID = crypto.randomUUID();

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

launcherBtn.addEventListener("click", () => {
  chatWidget.classList.add("open");
});

minimizeBtn.addEventListener("click", () => {
  chatWidget.classList.remove("open");
});

// ---------- Point 1: initial connectivity check ----------
async function checkConnection() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch("/", { method: "GET", cache: "no-store", signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error("Server responded with an error");

    connectingBubble.classList.remove("connecting");
    connectingBubble.innerHTML = "";
    connectingBubble.textContent =
      "Hi, I'm Anchor Logistics' assistant. Ask me about orders, shipping, or returns.";
  } catch (err) {
    clearTimeout(timeout);
    connectionBanner.classList.add("visible");
    connectingBubble.textContent = "Unable to reach support right now.";
    connectingBubble.classList.remove("connecting");
  }
}

checkConnection();

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

function setInputControlsEnabled(enabled) {
  // Prevents Path A (typed/voice-note chat) and Path B (live call) from
  // running at the same time, which would otherwise let two Gemini
  // sessions fight over the mic and the message log simultaneously.
  micBtn.disabled = !enabled;
  sendBtn.disabled = !enabled;
  textField.contentEditable = enabled ? "true" : "false";
}

function setHeaderStatusText(text) {
  const statusEl = document.querySelector(".header-status");
  for (const node of statusEl.childNodes) {
    if (node.nodeType === Node.TEXT_NODE) {
      node.nodeValue = ` ${text}`;
      break;
    }
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
    // Path A is text-only end to end now -- no audio_url ever comes back,
    // so there's nothing to autoplay here anymore (see Path B below for
    // where audio playback actually happens).
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

// =====================================================================
// Path B -- Live call
//
// Continuous mic audio streamed to /ws/live-call as raw 16-bit PCM at
// 16kHz (what the Live API requires), and Gemini's replies stream back
// the same way at 24kHz. Uses ScriptProcessorNode rather than the more
// modern AudioWorklet -- it's deprecated but still universally
// supported, and keeps this whole feature in one file instead of also
// needing a separately-loaded worklet module. Worth revisiting if this
// ever needs to be more production-grade.
// =====================================================================

const MIC_SAMPLE_RATE = 16000;
const PLAYBACK_SAMPLE_RATE = 24000;
const LIVE_WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live-call";

let liveCallState = "idle"; // idle | connecting | live
let liveSocket = null;
let liveMicStream = null;
let liveMicContext = null;
let liveMicSource = null;
let liveMicProcessor = null;
let livePlaybackContext = null;
let livePlaybackTime = 0;

liveCallBtn.addEventListener("click", () => {
  if (liveCallState === "idle") {
    startLiveCall();
  } else {
    endLiveCall();
  }
});

async function startLiveCall() {
  if (isRecording) stopRecording();
  setLiveCallState("connecting");

  try {
    liveMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    addBubble("I need microphone access to start a live call.", "assistant");
    setLiveCallState("idle");
    return;
  }

  liveSocket = new WebSocket(LIVE_WS_URL);
  liveSocket.binaryType = "arraybuffer";

  liveSocket.addEventListener("open", () => {
    setLiveCallState("live");
    startMicCapture();
  });

  liveSocket.addEventListener("message", handleLiveServerMessage);

  liveSocket.addEventListener("close", () => {
    if (liveCallState !== "idle") endLiveCall();
  });

  liveSocket.addEventListener("error", () => {
    addBubble("The live call connection ran into a problem.", "assistant");
    endLiveCall();
  });
}

function startMicCapture() {
  liveMicContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: MIC_SAMPLE_RATE });
  liveMicSource = liveMicContext.createMediaStreamSource(liveMicStream);

  // Buffer size 4096 at 16kHz is ~256ms of audio per chunk sent.
  liveMicProcessor = liveMicContext.createScriptProcessor(4096, 1, 1);

  liveMicProcessor.onaudioprocess = (event) => {
    if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
    const floatSamples = event.inputBuffer.getChannelData(0);
    liveSocket.send(floatTo16BitPCM(floatSamples));
  };

  // ScriptProcessorNode only fires onaudioprocess while connected into
  // the graph all the way to a destination -- this doesn't cause audible
  // echo, since we never route the *output* of this node anywhere audible,
  // it's just how the API requires the graph to be wired to keep pulling data.
  liveMicSource.connect(liveMicProcessor);
  liveMicProcessor.connect(liveMicContext.destination);
}

function floatTo16BitPCM(floatSamples) {
  const pcm16 = new Int16Array(floatSamples.length);
  for (let i = 0; i < floatSamples.length; i++) {
    const s = Math.max(-1, Math.min(1, floatSamples[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16.buffer;
}

function handleLiveServerMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    playLiveAudioChunk(event.data);
    return;
  }

  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch (err) {
    return; // ignore malformed control messages
  }

  if (msg.type === "transcript" && msg.text) {
    addBubble(msg.text, msg.role === "user" ? "user" : "assistant");
  } else if (msg.type === "interrupted") {
    stopLivePlaybackSchedule();
  } else if (msg.type === "error" && msg.message) {
    addBubble(msg.message, "assistant");
  }
}

function playLiveAudioChunk(arrayBuffer) {
  if (!livePlaybackContext) {
    livePlaybackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_SAMPLE_RATE });
    livePlaybackTime = livePlaybackContext.currentTime;
  }

  const pcm16 = new Int16Array(arrayBuffer);
  const floatSamples = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    floatSamples[i] = pcm16[i] / (pcm16[i] < 0 ? 0x8000 : 0x7fff);
  }

  const audioBuffer = livePlaybackContext.createBuffer(1, floatSamples.length, PLAYBACK_SAMPLE_RATE);
  audioBuffer.getChannelData(0).set(floatSamples);

  const source = livePlaybackContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(livePlaybackContext.destination);

  // Scheduling each chunk right after the previous one (rather than
  // starting all of them "now") is what makes streamed playback sound
  // continuous instead of stuttering between chunks.
  const startAt = Math.max(livePlaybackContext.currentTime, livePlaybackTime);
  source.start(startAt);
  livePlaybackTime = startAt + audioBuffer.duration;
}

function stopLivePlaybackSchedule() {
  // Barge-in: Gemini interrupted itself because the user started talking
  // over it. Resetting the schedule to "now" means any audio chunks that
  // were queued up behind this point effectively get skipped instead of
  // playing late.
  if (livePlaybackContext) {
    livePlaybackTime = livePlaybackContext.currentTime;
  }
}

function setLiveCallState(state) {
  liveCallState = state;
  const label = liveCallBtn.querySelector("span");

  if (state === "connecting") {
    liveCallBtn.classList.add("live-call-btn--active");
    label.textContent = "Connecting…";
    setInputControlsEnabled(false);
  } else if (state === "live") {
    liveCallBtn.classList.add("live-call-btn--active");
    label.textContent = "End call";
    setHeaderStatusText("Live call connected");
    setInputControlsEnabled(false);
  } else {
    liveCallBtn.classList.remove("live-call-btn--active");
    label.textContent = "Live call";
    setHeaderStatusText("Online now");
    setInputControlsEnabled(true);
  }
}

function endLiveCall() {
  if (liveSocket) {
    liveSocket.close();
    liveSocket = null;
  }
  if (liveMicProcessor) {
    liveMicProcessor.disconnect();
    liveMicProcessor.onaudioprocess = null;
    liveMicProcessor = null;
  }
  if (liveMicSource) {
    liveMicSource.disconnect();
    liveMicSource = null;
  }
  if (liveMicStream) {
    liveMicStream.getTracks().forEach((track) => track.stop());
    liveMicStream = null;
  }
  if (liveMicContext) {
    liveMicContext.close();
    liveMicContext = null;
  }
  if (livePlaybackContext) {
    livePlaybackContext.close();
    livePlaybackContext = null;
    livePlaybackTime = 0;
  }
  setLiveCallState("idle");
}