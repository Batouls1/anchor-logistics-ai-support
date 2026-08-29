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
const audioDeviceBar = document.getElementById("audioDeviceBar");
const audioOutputSelect = document.getElementById("audioOutputSelect");

// The server issues and signs conversation ids now. A browser-generated
// id would be rejected: any client could otherwise name someone else's
// conversation and read or append to its history.
let conversationTokenPromise = null;

function getConversationToken() {
  if (!conversationTokenPromise) {
    conversationTokenPromise = fetch("/conversation/start", { method: "POST" })
      .then((response) => {
        if (!response.ok) throw new Error("Could not start a conversation");
        return response.json();
      })
      .then((data) => data.conversation_id)
      .catch((err) => {
        // Clear the cached promise so the next message retries rather
        // than reusing a permanently rejected one.
        conversationTokenPromise = null;
        throw err;
      });
  }
  return conversationTokenPromise;
}

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

// Rejections (expired session, too long, rate limited) come back as
// non-2xx but still carry a written explanation. Treating every non-ok
// response as a generic failure would throw that away and show "something
// went wrong" when the server had already said something more useful.
async function readChatResponse(response) {
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }

  if (data && data.agent_text) return data;
  if (!response.ok) throw new Error("Request failed with status " + response.status);
  return data;
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
    formData.append("conversation_id", await getConversationToken());
    formData.append("message", text);

    const response = await fetch("/chat/text", { method: "POST", body: formData });
    const data = await readChatResponse(response);

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

  try {
    const formData = new FormData();
    formData.append("conversation_id", await getConversationToken());
    formData.append("audio", audioBlob, "voice-note.webm");

    const response = await fetch("/chat/voice", { method: "POST", body: formData });
    const data = await readChatResponse(response);

    typingBubble.remove();

    if (data.type === "fallback" || data.type === "error") {
      placeholder.remove();
      addBubble(data.agent_text, "assistant");
      return;
    }

    // Text-only end to end: the transcript replaces the placeholder and
    // the reply is read, not heard. Spoken replies are the live call's
    // job (Path B below).
    placeholder.textContent = data.user_text || "(voice note)";
    addBubble(data.agent_text, "assistant");
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
let liveMicSource = null;
let liveMicProcessor = null;
let liveMicSilencer = null;
// One AudioContext for BOTH directions -- see startMicCapture for why two
// of them (at different sample rates) silently broke playback on Windows.
let liveAudioContext = null;
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
  clearLiveCallError();
  setLiveCallState("connecting");

  // Must happen here, synchronously, while the click that triggered this
  // is still the running task -- that's what makes the browser treat the
  // AudioContext as user-initiated and let it produce sound. Creating it
  // after the `await` below, or later from the socket handler, is what
  // leaves it suspended and the call silent.
  ensureLiveAudioContext();

  try {
    liveMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showLiveCallError("Microphone access is needed to start a live call.");
    setLiveCallState("idle");
    return;
  }

  // Deliberately after getUserMedia: device labels stay blank until mic
  // permission is granted, and it's this very stream that triggers the
  // output re-routing we're correcting for. Awaited so the sink is
  // settled before the first audio chunk can arrive.
  await pinOutputToRealDevice(ensureLiveAudioContext());

  liveSocket = new WebSocket(LIVE_WS_URL);
  liveSocket.binaryType = "arraybuffer";

  liveSocket.addEventListener("open", () => {
    setLiveCallState("live");
    startMicCapture();
  });

  liveSocket.addEventListener("message", handleLiveServerMessage);

  liveSocket.addEventListener("close", () => {
    // The call ended on the server's side, not because the user asked --
    // so play out whatever audio is already buffered before tearing the
    // context down, instead of cutting the reply off mid-word.
    if (liveCallState !== "idle") endLiveCall({ drainAudio: true });
  });

  liveSocket.addEventListener("error", () => {
    showLiveCallError("The live call connection ran into a problem.");
    endLiveCall();
  });
}

function startMicCapture() {
  // Shares the ONE audio context with playback (see
  // ensureLiveAudioContext). A second context forced to 16kHz used to be
  // created here, and on Windows two contexts at different sample rates
  // fight over the output device: whichever opens it first wins, and the
  // other reports state "running" while never actually reaching the
  // speakers. That's what made the assistant inaudible even though every
  // byte arrived and was scheduled correctly.
  const ctx = ensureLiveAudioContext();
  liveMicSource = ctx.createMediaStreamSource(liveMicStream);

  // ~4096 frames per callback at the context's own rate.
  liveMicProcessor = ctx.createScriptProcessor(4096, 1, 1);

  liveMicProcessor.onaudioprocess = (event) => {
    if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
    const floatSamples = event.inputBuffer.getChannelData(0);
    // The context now runs at the hardware rate, so the 16kHz the Live
    // API requires is produced here instead of by the context.
    const resampled = downsampleTo(floatSamples, ctx.sampleRate, MIC_SAMPLE_RATE);
    if (resampled.length) liveSocket.send(floatTo16BitPCM(resampled));
  };

  // A ScriptProcessorNode only fires onaudioprocess while it's connected
  // through to a destination, so the chain has to reach one -- but the
  // mic must never be routed to the speakers. A zero gain node in between
  // satisfies the API without feeding your own voice back at you.
  liveMicSilencer = ctx.createGain();
  liveMicSilencer.gain.value = 0;

  liveMicSource.connect(liveMicProcessor);
  liveMicProcessor.connect(liveMicSilencer);
  liveMicSilencer.connect(ctx.destination);
}

function downsampleTo(input, inputRate, targetRate) {
  if (inputRate === targetRate) return input;
  if (inputRate < targetRate) return input; // never upsample; shouldn't happen

  const ratio = inputRate / targetRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    // Averaging the source window rather than picking one sample acts as
    // a cheap low-pass filter, which keeps high frequencies from folding
    // back as aliasing noise and confusing Gemini's speech detection.
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j++) {
      sum += input[j];
      count++;
    }
    output[i] = count ? sum / count : 0;
  }

  return output;
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

  // A live call is audio-only: nothing said on it is ever written into
  // the message stack. The server doesn't send transcripts at all any
  // more, and even if it did, there's deliberately no branch here that
  // would render them.
  if (msg.type === "turn_complete") {
    setLiveTurnStatus("listening");
  } else if (msg.type === "interrupted") {
    stopLivePlaybackSchedule();
    setLiveTurnStatus("listening");
  } else if (msg.type === "error" && msg.message) {
    // Connection/quota failures only -- never conversation content.
    // Shown in the alert banner rather than as a chat bubble so the live
    // call still leaves no trace in the message stack.
    showLiveCallError(msg.message);
  }
}

// The live call has no transcript, so the header status line is the only
// feedback that anything is happening. Without it the widget looks
// frozen while the assistant talks.
function setLiveTurnStatus(state) {
  if (liveCallState !== "live") return;
  setHeaderStatusText(state === "speaking" ? "Assistant speaking…" : "Listening…");
}

function showLiveCallError(message) {
  connectionBanner.textContent = message;
  connectionBanner.classList.add("visible");
}

function clearLiveCallError() {
  connectionBanner.classList.remove("visible");
}

// Created up front from the "Live call" click, NOT lazily when the first
// audio chunk arrives. Browsers start an AudioContext in the "suspended"
// state unless it's created during a user gesture, and a WebSocket
// message handler is not one -- a context built there can stay silently
// suspended for the whole call, which is exactly how you get a call that
// looks connected but plays nothing.
function ensureLiveAudioContext() {
  if (liveAudioContext) return liveAudioContext;

  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  // No forced sampleRate. The context runs at whatever the hardware uses;
  // mic audio is downsampled to 16kHz in JS on the way out, and incoming
  // 24kHz buffers are resampled by the graph on the way in. Forcing a
  // rate here is what required a second context in the first place.
  liveAudioContext = new AudioContextCtor();
  livePlaybackTime = liveAudioContext.currentTime;

  resumeLivePlayback();
  return liveAudioContext;
}

// Windows/Chrome re-routes an AudioContext's output the moment a
// microphone stream goes live -- it decides the page is "on a call" and
// hands playback to the communications sink, which is frequently a
// device nobody is listening to. Everything keeps reporting healthy: the
// context stays "running", its clock advances, buffers schedule and
// retire normally, and not a sound comes out.
//
// Pinning the context to the *concrete* device backing the system
// default (rather than the "default" alias, which is what gets
// re-routed) keeps audio where the user can hear it.
function normalizeDeviceLabel(label) {
  return (label || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

// Remembered by LABEL rather than deviceId: ids are per-origin and can be
// regenerated when site permissions are reset, whereas "Headset (soundcore
// ...)" survives that and still identifies the same hardware.
const OUTPUT_DEVICE_STORAGE_KEY = "anchor-live-output-device";

function readPreferredOutputLabel() {
  try {
    return localStorage.getItem(OUTPUT_DEVICE_STORAGE_KEY) || null;
  } catch (err) {
    return null; // private mode / storage blocked -- fall back to auto-pick
  }
}

function savePreferredOutputLabel(label) {
  try {
    if (label) localStorage.setItem(OUTPUT_DEVICE_STORAGE_KEY, label);
  } catch (err) {
    /* not being able to remember the choice shouldn't break the call */
  }
}

async function listConcreteOutputs() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  // "default" and "communications" are aliases, not hardware -- and
  // they're the ones the OS reassigns underneath us.
  return devices.filter(
    (d) => d.kind === "audiooutput" && d.deviceId !== "default" && d.deviceId !== "communications"
  );
}

// Auto-pick, used when the user hasn't chosen a device themselves.
function chooseOutputDevice(outputs, allDevices, micLabel) {
  // A Bluetooth headset cannot run its stereo (A2DP) output and its
  // microphone at the same time. Opening the mic forces the hands-free
  // profile, at which point the headset's *stereo* endpoint stops
  // producing sound entirely -- audio sent there vanishes with no error
  // anywhere. So when the mic belongs to a headset, playback has to go to
  // that same headset's hands-free endpoint.
  const micWords = normalizeDeviceLabel(micLabel)
    .split(" ")
    .filter((w) => w.length > 3);

  const sameDevice = micWords.length
    ? outputs.filter((d) => {
        const label = normalizeDeviceLabel(d.label);
        return micWords.filter((w) => label.includes(w)).length >= 2;
      })
    : [];

  const headsetEndpoint = sameDevice.find((d) => /hands.?free|headset/i.test(d.label));
  if (headsetEndpoint) return headsetEndpoint;
  if (sameDevice.length) return sameDevice[0];

  // Not a headset -- use the concrete device backing the system default,
  // matched through the "Default - <name>" alias so we follow the user's
  // own choice rather than an arbitrary device.
  const defaultAlias = allDevices.find(
    (d) => d.kind === "audiooutput" && d.deviceId === "default"
  );
  if (defaultAlias && defaultAlias.label) {
    const name = defaultAlias.label.replace(/^Default\s*-\s*/i, "").trim();
    const match = outputs.find((d) => d.label.trim() === name);
    if (match) return match;
  }

  return outputs[0] || null;
}

function renderOutputDeviceOptions(outputs, selectedDeviceId) {
  audioOutputSelect.innerHTML = "";
  for (const device of outputs) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || "Audio output";
    option.selected = device.deviceId === selectedDeviceId;
    audioOutputSelect.appendChild(option);
  }
  // Nothing to choose between, or no way to act on a choice -- don't show
  // a control that can't help.
  const usable =
    outputs.length > 1 && liveAudioContext && typeof liveAudioContext.setSinkId === "function";
  audioDeviceBar.classList.toggle("visible", Boolean(usable));
}

async function pinOutputToRealDevice(ctx) {
  if (typeof ctx.setSinkId !== "function") {
    // Firefox/Safari can't redirect an AudioContext. The call still works
    // on any setup the OS doesn't re-route, so this is a warning, not an
    // error -- but the picker would be useless, so it stays hidden.
    console.warn("Live call: this browser can't choose an audio output device.");
    return;
  }

  try {
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    const outputs = allDevices.filter(
      (d) =>
        d.kind === "audiooutput" &&
        d.deviceId !== "default" &&
        d.deviceId !== "communications"
    );
    if (!outputs.length) return; // only aliases available; leave the browser's choice alone

    const micTrack = liveMicStream && liveMicStream.getAudioTracks()[0];
    const micLabel = (micTrack && micTrack.label) || "";

    // An explicit choice from a previous call always wins over the
    // auto-pick -- the whole point of the picker is to override a guess
    // that got it wrong on this particular hardware.
    const preferredLabel = readPreferredOutputLabel();
    const remembered = preferredLabel
      ? outputs.find((d) => d.label === preferredLabel)
      : null;

    const target = remembered || chooseOutputDevice(outputs, allDevices, micLabel);
    if (!target) return;

    await ctx.setSinkId(target.deviceId);
    renderOutputDeviceOptions(outputs, target.deviceId);

    console.info(
      "Live call: audio output " + (remembered ? "restored to" : "pinned to") +
      " '" + (target.label || target.deviceId) +
      "' (microphone: '" + (micLabel || "unknown") + "')"
    );
  } catch (err) {
    // Non-fatal -- without this the call still works anywhere that isn't
    // affected by the re-routing behaviour.
    console.warn("Live call: could not pin the audio output device.", err);
  }
}

// The user picking a device by hand. Their choice is remembered and takes
// priority over the auto-pick on every later call.
audioOutputSelect.addEventListener("change", async () => {
  if (!liveAudioContext || typeof liveAudioContext.setSinkId !== "function") return;

  const deviceId = audioOutputSelect.value;
  const label = audioOutputSelect.selectedOptions[0]
    ? audioOutputSelect.selectedOptions[0].textContent
    : "";

  try {
    await liveAudioContext.setSinkId(deviceId);
    savePreferredOutputLabel(label);
    clearLiveCallError();
    console.info("Live call: audio output switched to '" + label + "'");
  } catch (err) {
    console.error("Live call: could not switch audio output.", err);
    showLiveCallError("Couldn't switch to that speaker. Try another one.");
  }
});

// Headphones plugged in or a headset connected mid-call: re-run the pick
// so audio follows the hardware instead of staying on a device that may
// have just disappeared.
if (navigator.mediaDevices && "ondevicechange" in navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener("devicechange", async () => {
    if (liveCallState !== "live" || !liveAudioContext) return;
    console.info("Live call: audio devices changed, re-selecting output.");
    await pinOutputToRealDevice(liveAudioContext);
  });
}

function resumeLivePlayback() {
  if (!liveAudioContext || liveAudioContext.state !== "suspended") return;

  liveAudioContext
    .resume()
    .then(() => {
      // currentTime is frozen while suspended, so any timestamp captured
      // before this point is now in the past. Re-baseline the schedule,
      // otherwise the first chunks get scheduled behind the clock.
      livePlaybackTime = liveAudioContext.currentTime;
    })
    .catch((err) => {
      console.error("Live call: could not start audio playback.", err);
      showLiveCallError("Audio playback was blocked. Click the page, then try the call again.");
    });
}

function playLiveAudioChunk(arrayBuffer) {
  try {
    const ctx = ensureLiveAudioContext();
    // A context can be suspended again by the browser (tab backgrounded,
    // device switch), so this is checked per chunk rather than just once.
    resumeLivePlayback();

    const pcm16 = new Int16Array(arrayBuffer);
    if (pcm16.length === 0) return;

    const floatSamples = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      floatSamples[i] = pcm16[i] / (pcm16[i] < 0 ? 0x8000 : 0x7fff);
    }

    const audioBuffer = ctx.createBuffer(1, floatSamples.length, PLAYBACK_SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(floatSamples);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    // Scheduling each chunk right after the previous one (rather than
    // starting all of them "now") is what makes streamed playback sound
    // continuous instead of stuttering between chunks.
    const startAt = Math.max(ctx.currentTime, livePlaybackTime);
    source.start(startAt);
    livePlaybackTime = startAt + audioBuffer.duration;

    setLiveTurnStatus("speaking");
  } catch (err) {
    // Without this the failure is completely invisible: an exception
    // thrown inside a WebSocket message handler kills only that handler,
    // so the call goes on looking healthy while producing no sound.
    console.error("Live call: failed to play an audio chunk.", err);
    showLiveCallError("Something went wrong playing the assistant's audio.");
  }
}

function stopLivePlaybackSchedule() {
  // Barge-in: Gemini interrupted itself because the user started talking
  // over it. Resetting the schedule to "now" means any audio chunks that
  // were queued up behind this point effectively get skipped instead of
  // playing late.
  if (liveAudioContext) {
    livePlaybackTime = liveAudioContext.currentTime;
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
    setHeaderStatusText("Listening…");
    setInputControlsEnabled(false);
  } else {
    liveCallBtn.classList.remove("live-call-btn--active");
    label.textContent = "Live call";
    setHeaderStatusText("Online now");
    setInputControlsEnabled(true);
  }
}

// Upper bound on how long teardown will wait for queued audio to finish,
// so a bad livePlaybackTime can never leave a context open indefinitely.
const MAX_AUDIO_DRAIN_SECONDS = 30;

function endLiveCall({ drainAudio = false } = {}) {
  // Device labels go blank once the mic stream is released, so the list
  // would degrade into "Audio output, Audio output" if left on screen.
  audioDeviceBar.classList.remove("visible");

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
  if (liveMicSilencer) {
    liveMicSilencer.disconnect();
    liveMicSilencer = null;
  }
  if (liveMicStream) {
    liveMicStream.getTracks().forEach((track) => track.stop());
    liveMicStream = null;
  }
  if (liveAudioContext) {
    const ctx = liveAudioContext;
    // Audio is scheduled a few seconds ahead of the clock, so closing the
    // context the instant the call ends silently destroys whatever hasn't
    // played yet -- typically the entire reply. When the call ended on
    // its own (the server hung up), let the queue drain first. Only an
    // explicit "End call" from the user cuts it off immediately, which is
    // what they're asking for by pressing it.
    const queued = Math.max(0, livePlaybackTime - ctx.currentTime);
    liveAudioContext = null;
    livePlaybackTime = 0;

    if (drainAudio && queued > 0.05) {
      setTimeout(() => ctx.close(), Math.min(queued + 0.3, MAX_AUDIO_DRAIN_SECONDS) * 1000);
    } else {
      ctx.close();
    }
  }
  setLiveCallState("idle");
}