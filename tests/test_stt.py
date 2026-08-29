"""
Tests for whisper/stt.py's confidence gating and VAD settings. The
WhisperModel itself is mocked -- these assert the decision logic around
it, not transcription accuracy.
"""

from types import SimpleNamespace
from unittest.mock import patch

import whisper.stt as stt


def segment(text, avg_logprob=-0.2, no_speech_prob=0.1):
    return SimpleNamespace(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def _mock_transcribe(segments):
    return patch.object(stt._model, "transcribe", return_value=(iter(segments), None))


def test_clear_speech_is_confident():
    with _mock_transcribe([segment(" where is my order ")]):
        result = stt.transcribe("fake.webm")

    assert result == {"text": "where is my order", "confident": True}


def test_segments_are_joined_and_stripped():
    with _mock_transcribe([segment(" where is "), segment(" my order ")]):
        result = stt.transcribe("fake.webm")

    assert result["text"] == "where is my order"


def test_low_logprob_is_not_confident():
    with _mock_transcribe([segment("mumble", avg_logprob=-1.5)]):
        result = stt.transcribe("fake.webm")

    assert result["confident"] is False


def test_high_no_speech_probability_is_not_confident():
    with _mock_transcribe([segment("background noise", no_speech_prob=0.9)]):
        result = stt.transcribe("fake.webm")

    assert result["confident"] is False


def test_no_segments_is_not_confident():
    """VAD found no speech at all -- the fallback path, not a crash."""
    with _mock_transcribe([]):
        result = stt.transcribe("fake.webm")

    assert result == {"text": "", "confident": False}


def test_a_decoder_error_falls_back_instead_of_crashing_the_request():
    """
    Malformed/truncated audio (a recording stopped almost immediately)
    must return a low-confidence result, not raise into the endpoint.
    """
    with patch.object(stt._model, "transcribe", side_effect=RuntimeError("bad audio")):
        result = stt.transcribe("fake.webm")

    assert result == {"text": "", "confident": False}


def test_vad_is_tuned_for_short_quiet_clips():
    """
    The default VAD threshold of 0.5 was classifying short/quiet voice
    notes as pure silence. These are the settings that fixed it.
    """
    with _mock_transcribe([segment("hello")]) as mock:
        stt.transcribe("fake.webm")

    kwargs = mock.call_args.kwargs
    assert kwargs["vad_filter"] is True
    assert kwargs["language"] == "en"
    assert kwargs["vad_parameters"]["threshold"] == 0.3
    assert kwargs["vad_parameters"]["speech_pad_ms"] == 300
