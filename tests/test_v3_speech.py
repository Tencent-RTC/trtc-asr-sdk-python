"""v3 streaming (/asr/v3) tests: start-frame wire format, sync ack error
handling, lifecycle and local validation (mirrors Go asr/v3
speech_recognizer_test.go)."""

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

import websockets.asyncio.client

from trtc_asr.errors import ASRError, ERR_INVALID_PARAM
from trtc_asr.v3 import (
    SPEAKER_DIARIZATION_VOICEPRINT,
    Context,
    ContextKV,
    SpeakerRole,
    SpeechRecognitionListener,
    SpeechRecognizer,
    new_credential,
)
from trtc_asr.v3.speech_recognizer import (
    MAX_VOICE_ID_LEN,
    STREAM_FRAME_MAX_BYTES,
)


class _Listener(SpeechRecognitionListener):
    def __init__(self):
        self.events = []
        self.failures = []

    def on_recognition_start(self, response):
        self.events.append(("start", response.voice_id))

    def on_sentence_begin(self, response):
        self.events.append(("begin", response.result.index))

    def on_recognition_result_change(self, response):
        self.events.append(("change", response.result.voice_text_str))

    def on_sentence_end(self, response):
        self.events.append(("end", response.result.voice_text_str))

    def on_recognition_complete(self, response):
        self.events.append(("complete", response.voice_id))

    def on_fail(self, response, error):
        self.failures.append(error)


class _FakeWS:
    """WebSocket double supporting the v3 handshake (recv for the ack) and
    async iteration for downlink frames.

    ``downlink`` frames are delivered as soon as the read loop asks for them;
    ``after_end`` frames are held back until the client sends
    ``{"type": "end"}``, mirroring a real server that only emits the terminal
    ``final=1`` response once the session is closed. Without that gate the
    fake would push the whole scripted session to the read loop before
    ``stop()`` runs, and the recognizer would already be stopped — making the
    ``end`` frame assertion below depend on event-loop scheduling.
    """

    def __init__(self, ack, downlink=None, after_end=None):
        self.sent = []
        self._ack = json.dumps(ack) if not isinstance(ack, str) else ack
        self._downlink = [json.dumps(m) if not isinstance(m, str) else m for m in downlink or []]
        self._after_end = [json.dumps(m) if not isinstance(m, str) else m for m in after_end or []]
        # Created lazily inside the read loop: on Python 3.8/3.9 constructing
        # asyncio.Event() outside a running loop resolves (and caches) the
        # thread's event loop, which raises once asyncio.run has torn it down.
        self._end_received = None
        self._end_sent = False
        self.closed = False

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, str):
            try:
                if json.loads(data).get("type") == "end":
                    self._end_sent = True
                    if self._end_received is not None:
                        self._end_received.set()
            except (ValueError, AttributeError):
                pass

    async def recv(self):
        return self._ack

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._downlink:
            return self._downlink.pop(0)
        if self._after_end:
            if self._end_received is None:
                self._end_received = asyncio.Event()
                if self._end_sent:
                    self._end_received.set()
            await self._end_received.wait()
            return self._after_end.pop(0)
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


def make_recognizer(listener=None):
    credential = new_credential(1400000000, "test-secret")
    return SpeechRecognizer(credential, "16k_zh_en", listener)


def _patch_connect(monkeypatch, ws):
    """Route every dial in this test to the given fake WS (never the real
    network)."""

    class _Connect:
        def __init__(self, captured):
            self._captured = captured

        async def __call__(self, url, **kwargs):
            if self._captured is not None:
                self._captured["url"] = url
            return ws

    captured = {}
    monkeypatch.setattr(websockets.asyncio.client, "connect", _Connect(captured))
    return captured


def _ack(voice_id="v1", **extra):
    frame = {"code": 0, "message": "success", "voice_id": voice_id}
    frame.update(extra)
    return frame


def _result_frame(slice_type, final=0, text="x"):
    return {
        "code": 0,
        "message": "success",
        "voice_id": "v1",
        "final": final,
        "result": {
            "slice_type": slice_type,
            "index": 0,
            "start_time": 0,
            "end_time": 1000,
            "voice_text_str": text,
        },
    }


# ---------------------------------------------------------------- wire


def test_start_frame_wire(monkeypatch):
    listener = _Listener()
    ws = _FakeWS(_ack("voice-1"))
    captured = _patch_connect(monkeypatch, ws)
    recognizer = make_recognizer(listener)
    recognizer.set_voice_id("voice-1")
    recognizer.set_hotword_list("深度学习|10")
    recognizer.set_vad_level(0)
    recognizer.set_noise_threshold(1.5)
    recognizer.set_filter_empty_result(0)
    recognizer.set_convert_num_mode(0)  # explicit 0 must be sent on v3
    recognizer.set_speaker_diarization(SPEAKER_DIARIZATION_VOICEPRINT)
    recognizer.set_speaker_roles(
        [SpeakerRole(role_name="teacher", audio_url="https://example.com/t.wav")]
    )
    recognizer.set_voiceprint_ids(["vp-1"])
    recognizer.set_context(
        Context(text="bg", terms=["ASR"], general=[ContextKV(key="domain", value="Meeting")])
    )

    async def run():
        await recognizer.start()
        await recognizer.stop()

    asyncio.run(run())

    # Handshake URL: path /asr/v3, query carries only voice_id.
    url = urlparse(captured["url"])
    assert url.path == "/asr/v3"
    query = parse_qs(url.query)
    assert query.get("voice_id") == ["voice-1"]
    for banned in ("signature", "usersig", "sdkappid", "secretid", "appid"):
        assert banned not in query, "v3 moves auth into the start frame"

    # Start frame structure.
    frame = json.loads(ws.sent[0])
    assert frame["type"] == "start"

    auth = frame["auth"]
    assert auth["sdkappid"] == "1400000000"
    assert auth["usersig"]
    # business is a server-side internal gray dimension; the SDK must not
    # send it.
    assert "business" not in auth

    params = frame["params"]
    assert params["voice_id"] == "voice-1"
    assert params["engine_model_type"] == "16k_zh_en"
    assert params["voice_format"] == 1
    assert params["needvad"] == 1
    assert params["convert_num_mode"] == 0  # explicit 0 preserved
    assert params["filter_empty_result"] == 0  # explicit 0 preserved
    assert params["vad_level"] == 0  # explicit 0 preserved
    assert params["noise_threshold"] == 1.5
    assert params["hotword_list"] == "深度学习|10"
    assert params["speaker_diarization"] == 3
    assert params["voiceprint_ids"] == ["vp-1"]
    # speaker_roles elements are snake_case (role_name / audio_url), not the
    # v2 CamelCase wire.
    assert params["speaker_roles"] == [
        {"role_name": "teacher", "audio_url": "https://example.com/t.wav"}
    ]
    assert params["context"] == {
        "text": "bg",
        "terms": ["ASR"],
        "general": [{"key": "domain", "value": "Meeting"}],
    }
    # sdk_info telemetry is present (survives in server dumps).
    assert params["sdk_info"]["sdk_lang"] == "python"
    assert params["sdk_info"]["version"]


def test_start_frame_defaults_tri_state():
    recognizer = make_recognizer()
    params = recognizer._build_params()

    for absent in (
        "vad_level",
        "noise_threshold",
        "filter_empty_result",
        "vad_silence_time",
        "input_sample_rate",
        "hotword_id",
        "hotword_list",
        "speaker_diarization",
        "context",
    ):
        assert absent not in params, "unset setter must omit the wire field"

    # SDK-managed defaults are always sent.
    assert params["needvad"] == 1
    assert params["convert_num_mode"] == 1
    assert params["voice_format"] == 1
    assert params["engine_model_type"] == "16k_zh_en"
    assert params["sdk_info"]["sdk_lang"] == "python"


# ---------------------------------------------------------------- ack


def test_start_auth_error_sync(monkeypatch):
    """v3 fails start() synchronously on a structured error frame."""

    async def run():
        listener = _Listener()
        ws = _FakeWS({"code": 4002, "message": "auth failed", "voice_id": "v1"})
        _patch_connect(monkeypatch, ws)
        recognizer = make_recognizer(listener)
        with pytest.raises(ASRError) as exc_info:
            await recognizer.start()
        assert exc_info.value.code == 4002
        # The failed start leaves no usable session.
        with pytest.raises(ASRError):
            await recognizer.write(b"abc")

    asyncio.run(run())


def test_start_gray_disabled_error_sync(monkeypatch):
    """A gray-disabled endpoint answers 4001 with a WS error frame."""

    async def run():
        listener = _Listener()
        ws = _FakeWS({"code": 4001, "message": "v3 interface not enabled", "voice_id": "v1"})
        _patch_connect(monkeypatch, ws)
        recognizer = make_recognizer(listener)
        with pytest.raises(ASRError) as exc_info:
            await recognizer.start()
        assert exc_info.value.code == 4001

    asyncio.run(run())


def test_normal_flow(monkeypatch):
    downlink = [
        _result_frame(0),
        _result_frame(1, text="你好"),
        _result_frame(2, text="你好。"),
    ]
    # The terminal final=1 frame only arrives after the client's end frame.
    after_end = [_result_frame(2, final=1, text="你好。")]

    async def run():
        listener = _Listener()
        ws = _FakeWS(_ack(), downlink, after_end=after_end)
        _patch_connect(monkeypatch, ws)
        recognizer = make_recognizer(listener)
        await recognizer.start()
        await recognizer.write(b"\x00" * 1280)
        await recognizer.stop()

        kinds = [e[0] for e in listener.events]
        # The standalone sentence-end (slice_type=2, final=0) and the terminal
        # final=1&slice_type=2 frame each dispatch on_sentence_end — mirroring
        # the Go SDK.
        assert kinds == ["start", "begin", "change", "end", "end", "complete"]
        assert listener.failures == []
        # Exactly one binary audio frame reached the wire.
        binary = [f for f in ws.sent if isinstance(f, bytes)]
        assert binary == [b"\x00" * 1280]
        # The end frame was sent.
        text_frames = [json.loads(f) for f in ws.sent if isinstance(f, str)]
        assert {"type": "end"} in text_frames

    asyncio.run(run())


def test_mid_session_error(monkeypatch):
    async def run():
        listener = _Listener()
        ws = _FakeWS(
            _ack(),
            [{"code": 4008, "message": "audio timeout", "voice_id": "v1"}],
        )
        _patch_connect(monkeypatch, ws)
        recognizer = make_recognizer(listener)
        await recognizer.start()

        # Drain the read loop.
        await recognizer._read_task
        assert len(listener.failures) == 1
        assert listener.failures[0].code == 4008

    asyncio.run(run())


def test_write_frame_too_large(monkeypatch):
    async def run():
        listener = _Listener()
        # Held back until the end frame so the session is still running when
        # write() performs its size check.
        ws = _FakeWS(_ack(), after_end=[_result_frame(2, final=1)])
        _patch_connect(monkeypatch, ws)
        recognizer = make_recognizer(listener)
        await recognizer.start()
        with pytest.raises(ASRError) as exc_info:
            await recognizer.write(b"\x00" * (STREAM_FRAME_MAX_BYTES + 1))
        assert exc_info.value.code == ERR_INVALID_PARAM

    asyncio.run(run())


# ---------------------------------------------------------------- validation


def test_validation_local():
    long_voice_id = "x" * (MAX_VOICE_ID_LEN + 1)
    cases = [
        ("voice_id too long", lambda r: r.set_voice_id(long_voice_id)),
        ("max_speak_time too small", lambda r: r.set_max_speak_time(1000)),
        ("max_speak_time too large", lambda r: r.set_max_speak_time(90001)),
        ("vad_silence_time too small", lambda r: r.set_vad_silence_time(100)),
        ("vad_silence_time too large", lambda r: r.set_vad_silence_time(2001)),
        ("needvad invalid", lambda r: r.set_need_vad(2)),
        ("convert_num_mode invalid", lambda r: r.set_convert_num_mode(2)),
        ("filter_dirty invalid", lambda r: r.set_filter_dirty(3)),
        ("filter_modal invalid", lambda r: r.set_filter_modal(3)),
        ("filter_punc invalid", lambda r: r.set_filter_punc(2)),
        ("word_info invalid", lambda r: r.set_word_info(3)),
        ("word_with_space invalid", lambda r: r.set_word_with_space(2)),
        ("voice_format invalid", lambda r: r.set_voice_format(2)),
        ("input_sample_rate invalid", lambda r: r.set_input_sample_rate(16000)),
        ("filter_empty_result invalid", lambda r: r.set_filter_empty_result(2)),
        ("vad_level invalid", lambda r: r.set_vad_level(2)),
        ("noise_threshold out of range", lambda r: r.set_noise_threshold(4.1)),
        ("diarization invalid", lambda r: r.set_speaker_diarization(2)),
    ]
    for name, apply in cases:
        recognizer = make_recognizer()
        apply(recognizer)
        with pytest.raises(ASRError) as exc_info:
            recognizer._validate_options()
        assert exc_info.value.code == ERR_INVALID_PARAM, name

    # Boundary values must pass validation.
    valid = [
        lambda r: r.set_max_speak_time(5000),
        lambda r: r.set_max_speak_time(90000),
        lambda r: r.set_vad_silence_time(240),
        lambda r: (r.set_need_vad(0), r.set_vad_silence_time(100)),  # only validated with needvad=1
        lambda r: r.set_voice_format(12),
        lambda r: r.set_word_info(100),
    ]
    for apply in valid:
        recognizer = make_recognizer()
        apply(recognizer)
        recognizer._validate_options()
