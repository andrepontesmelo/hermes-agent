"""Gemini tail-noise trim in the core text_to_speech render path.

Gemini TTS appends ~130 ms of full-scale clipped noise (samples pinned near ±32767)
to every response. The generator now trims it (clip-anchor port from the tts-voice
skill's tts_trim.py) before the WAV wrap, so every container and every caller gets
clean audio. Contract:
  - a pinned-tail buffer is trimmed: no burst survives, tail peaks stay quiet (<= 2000)
  - a clean buffer passes through bit-identical
  - only the Gemini generator trims: a non-Gemini provider's bytes are untouched
"""

import base64
import json
import struct
from unittest.mock import MagicMock, patch

import pytest

SR = 24000


def _pcm(samples) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _speech_like(n: int, amp: int = 8000, seed: int = 7):
    """Deterministic speech-ish samples, well under the clip threshold."""
    x = seed
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(amp - (x % (2 * amp)))
    return out


def _dirty_gemini_pcm() -> bytes:
    """Speech + 60 ms quiet gap + ~130 ms pinned-noise burst: the real defect shape."""
    return _pcm(_speech_like(SR) + [0] * (SR * 60 // 1000) + _pinned_tail())


def _pinned_tail(duration_samples: int = SR * 130 // 1000):
    """Full-scale alternating ±32767 noise, as the Gemini defect produces."""
    return [32767 if i % 2 == 0 else -32767 for i in range(duration_samples)]


class TestTrimFunction:
    def test_pinned_tail_is_trimmed(self):
        from tools.tts_tool_delivery import _trim_gemini_tail_noise

        pcm = _dirty_gemini_pcm()
        out = _trim_gemini_tail_noise(pcm)
        assert len(out) > 0
        out_samples = struct.unpack(f"<{len(out) // 2}h", out)
        # the clipped burst is gone entirely
        assert max(abs(s) for s in out_samples) < 32000
        # fade-out drives the final sample (and the retained pad) near silence
        assert abs(out_samples[-1]) <= 500
        # removed = burst (130 ms) + 30 ms pad = ~160 ms; the pre-burst quiet gap is
        # retained by design (the 40 ms-gap rule stops the walk-back so speech survives)
        removed_ms = (len(pcm) - len(out)) / 2 / SR * 1000
        assert 150 <= removed_ms <= 170

    def test_clean_buffer_passes_through_untouched(self):
        from tools.tts_tool_delivery import _trim_gemini_tail_noise

        pcm = _pcm(_speech_like(SR))
        out = _trim_gemini_tail_noise(pcm)
        assert out is pcm  # no-op returns the same bytes object

    def test_short_and_empty_buffers_are_no_ops(self):
        from tools.tts_tool_delivery import _trim_gemini_tail_noise

        empty = b""
        assert _trim_gemini_tail_noise(empty) is empty
        partial = b"\x01"
        assert _trim_gemini_tail_noise(partial) is partial

    def test_loud_but_unclipped_speech_is_not_trimmed(self):
        from tools.tts_tool_delivery import _trim_gemini_tail_noise

        # Speech that peaks high but never pins >= 32000 must survive intact.
        pcm = _pcm(_speech_like(SR, amp=31000))
        out = _trim_gemini_tail_noise(pcm)
        assert out is pcm


class TestGeminiGeneratorTrims:
    """The generator output (post WAV-wrap) must carry no clipped tail."""

    @staticmethod
    def _mock_response(pcm_bytes: bytes) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"inlineData": {
                    "mimeType": "audio/L16;codec=pcm;rate=24000",
                    "data": base64.b64encode(pcm_bytes).decode(),
                }}]}
            }]
        }
        return resp

    def test_generator_output_has_no_pinned_tail(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        dirty = _dirty_gemini_pcm()
        out_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=self._mock_response(dirty)):
            _generate_gemini_tts("Hi", out_path, {})

        wav = (tmp_path / "test.wav").read_bytes()
        assert wav[:4] == b"RIFF"
        samples = struct.unpack(f"<{(len(wav) - 44) // 2}h", wav[44:])
        assert len(samples) > 0
        # the clipped burst never reached the file, and the fade closed the tail
        assert max(abs(s) for s in samples) < 32000
        assert abs(samples[-1]) <= 500

    def test_clean_gemini_output_is_untouched(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        clean = _pcm(_speech_like(SR))
        out_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=self._mock_response(clean)):
            _generate_gemini_tts("Hi", out_path, {})

        wav = (tmp_path / "test.wav").read_bytes()
        assert wav[44:] == clean  # bit-identical payload


class TestNonGeminiProvidersNeverTrim:
    def test_openai_path_bytes_untouched(self, tmp_path, monkeypatch):
        """A non-Gemini provider emitting the same dirty PCM must not be trimmed."""
        from tools import tts_tool as tts_mod

        dirty = _dirty_gemini_pcm()
        out_path = str(tmp_path / "dirty.wav")

        def fake_openai(text, file_str, tts_config, instructions=None):
            with open(file_str, "wb") as f:
                f.write(_wrap_pcm(dirty))
            return file_str

        monkeypatch.setattr(tts_mod, "_load_tts_config", lambda: {"provider": "openai"})
        monkeypatch.setattr(tts_mod, "_generate_openai_tts", fake_openai)

        result = json.loads(tts_mod.text_to_speech_tool("Hi there", output_path=out_path))
        assert result["success"] is True
        assert result["provider"] == "openai"
        # byte-identical: the pinned tail survives for a non-Gemini provider
        assert (tmp_path / "dirty.wav").read_bytes()[44:] == dirty

    def test_trim_not_called_for_openai(self, tmp_path, monkeypatch):
        from tools import tts_tool as tts_mod
        from tools import tts_tool_delivery as delivery

        called = []
        real = delivery._trim_gemini_tail_noise
        monkeypatch.setattr(
            delivery, "_trim_gemini_tail_noise",
            lambda pcm, **kw: called.append(1) or real(pcm, **kw))

        def fake_openai(text, file_str, tts_config, instructions=None):
            with open(file_str, "wb") as f:
                f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
            return file_str

        monkeypatch.setattr(tts_mod, "_load_tts_config", lambda: {"provider": "openai"})
        monkeypatch.setattr(tts_mod, "_generate_openai_tts", fake_openai)

        result = json.loads(tts_mod.text_to_speech_tool("Hi"))
        assert result["success"] is True
        assert called == []


def _wrap_pcm(pcm_bytes: bytes) -> bytes:
    """Minimal WAV wrapper for fixture bytes."""
    from tools.tts_tool_delivery import _wrap_pcm_as_wav
    return _wrap_pcm_as_wav(pcm_bytes)
