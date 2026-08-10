"""
English-only speech-to-text using faster-whisper, loaded once at import
time and reused across every voice note. Safe to load eagerly here (no
live credentials needed), unlike gemini/tools.py's lazy Retriever.
"""

import os

from faster_whisper import WhisperModel

MODEL_SIZE = "small.en"  # meaningfully more accurate than base.en; costs
                         # roughly 5-10s per turn on CPU for that accuracy
COMPUTE_TYPE = "int8"

_model = WhisperModel(
    MODEL_SIZE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=os.cpu_count(),
)

AVG_LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


def transcribe(audio_path: str) -> dict:
    """
    Transcribes an audio file (any format faster-whisper's underlying
    decoder supports -- wav, webm, mp3, etc. -- no manual conversion needed).

    Returns: {"text": str, "confident": bool}
    `confident=False` is the signal for the fallback path: don't forward
    this to Gemini, ask the user to repeat or type instead.
    """
    try:
        segments, _info = _model.transcribe(
            audio_path,
            language="en",
            beam_size=5,  
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = list(segments)
    except Exception:
        # Malformed/empty/truncated audio (e.g. a recording stopped almost
        # immediately after starting). Treat this the same as a low-confidence 
        # transcription rather than crashing the request 
        return {"text": "", "confident": False}

    if not segments:
        return {"text": "", "confident": False}

    text = " ".join(s.text.strip() for s in segments).strip()
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    no_speech_prob = max(s.no_speech_prob for s in segments)

    confident = (
        bool(text)
        and avg_logprob > AVG_LOGPROB_THRESHOLD
        and no_speech_prob < NO_SPEECH_THRESHOLD
    )

    return {"text": text, "confident": confident}


if __name__ == "__main__":
    # Quick manual test 
    path = input("Path to an audio file to transcribe: ").strip()
    result = transcribe(path)
    print(f"\nTranscript: {result['text']}")
    print(f"Confident: {result['confident']}")