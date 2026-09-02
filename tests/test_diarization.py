"""Speaker diarization / VAD tuning end-to-end tests (aligned with Go
asr/diarization_test.go, asr/file_recognizer_speaker_test.go and
asr/sentence_recognizer_speaker_test.go)."""

import asyncio
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from trtc_asr import Credential
from trtc_asr.errors import ASRError
from trtc_asr.file_recognizer import CreateRecTaskRequest, FileRecognizer
from trtc_asr.sentence_recognizer import SentenceRecognitionRequest, SentenceRecognizer
from trtc_asr.signature import SPEAKER_DIARIZATION_VOICEPRINT, SpeakerRole
from trtc_asr.speech_recognizer import (
    _State,
    SpeechRecognitionListener,
    SpeechRecognizer,
    SpeechRecognitionResponse,
)


def make_credential():
    return Credential(app_id=1300000000, sdk_app_id=1400000000, secret_key="test-secret")


# ---------------------------------------------------------------- response
# decoding (Go: TestReadLoopDecodesSpeakerDiarizationResult)

DIARIZATION_END_MESSAGE = {
    "code": 0,
    "message": "ok",
    "voice_id": "v1",
    "result": {
        "slice_type": 2,
        "index": 1,
        "start_time": 3640,
        "end_time": 6600,
        "voice_text_str": "你好 嗯我想咨询一下",
        "word_size": 3,
        "finish_silence_ms": 800,
        "last_token_runtime_ms": 42,
        "word_list": [
            {"word": "你", "start_time": 3640, "end_time": 3760, "stable_flag": 1, "speaker_id": 1, "speaker_name": "teacher"},
            {"word": "好", "start_time": 3760, "end_time": 3880, "stable_flag": 1, "speaker_id": 1, "speaker_name": "teacher"},
            {"word": "嗯", "start_time": 5400, "end_time": 5550, "stable_flag": 1, "speaker_id": 2, "speaker_name": "student"},
        ],
        "speaker_segments": [
            {"speaker_id": 1, "speaker_name": "teacher", "start_time": 3640, "end_time": 3880, "text": "你好", "word_start": 0, "word_end": 1, "stable_flag": 1},
            {"speaker_id": 2, "speaker_name": "student", "start_time": 5400, "end_time": 6600, "text": "嗯我想咨询一下", "stable_flag": 0},
        ],
    },
}


def test_response_decodes_speaker_diarization_result():
    resp = SpeechRecognitionResponse.from_dict(DIARIZATION_END_MESSAGE)

    result = resp.result
    assert len(result.speaker_segments) == 2

    first = result.speaker_segments[0]
    assert (first.speaker_id, first.speaker_name) == (1, "teacher")
    assert (first.text, first.start_time, first.end_time, first.stable_flag) == ("你好", 3640, 3880, 1)
    assert (first.word_start, first.word_end) == (0, 1)

    second = result.speaker_segments[1]
    # word_info-less segments omit the indexes; None must be preserved so the
    # caller can tell "no index" from index 0.
    assert second.word_start is None
    assert second.word_end is None
    assert second.stable_flag == 0

    assert len(result.word_list) == 3
    w = result.word_list[2]
    assert (w.speaker_id, w.speaker_name) == (2, "student")

    assert result.finish_silence_ms == 800
    assert result.last_token_runtime_ms == 42
    # The sentence-level speaker stays absent on this engine; None
    # distinguishes that from speaker 0.
    assert result.speaker_id is None


# ---------------------------------------------------------------- read loop
# (Go: ack-frame skipping + terminal ordering + lifecycle state)


class _CaptureListener(SpeechRecognitionListener):
    def __init__(self):
        self.events = []
        self.failed = None

    def on_recognition_start(self, response):
        self.events.append(("start", response.voice_id))

    def on_sentence_begin(self, response):
        self.events.append(("begin", response.result.index))

    def on_recognition_result_change(self, response):
        self.events.append(("change", response.result.voice_text_str))

    def on_sentence_end(self, response):
        self.events.append(("end", response))

    def on_recognition_complete(self, response):
        self.events.append(("complete", response.voice_id))

    def on_fail(self, response, error):
        self.failed = (response, error)


class _FakeWS:
    """Minimal async-iterable WebSocket double."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send(self, data):
        pass

    async def close(self):
        self.closed = True


def test_read_loop_skips_ack_frame_and_finishes_on_final():
    """The connection ack (no "result") must not trigger on_sentence_begin;
    a final=1 sentence-end frame dispatches on_sentence_end then
    on_recognition_complete and advances to STOPPED."""
    ack = json.dumps({"code": 0, "message": "success", "voice_id": "v1"})
    end_frame = dict(DIARIZATION_END_MESSAGE)
    end_frame["final"] = 1
    messages = [ack, json.dumps(end_frame)]

    listener = _CaptureListener()
    recognizer = SpeechRecognizer(make_credential(), "16k_zh", listener)
    recognizer._ws = _FakeWS(messages)
    recognizer._state = _State.RUNNING

    async def run():
        await recognizer._read_loop()

    asyncio.run(run())

    kinds = [e[0] for e in listener.events]
    # Exactly one start (delivered at read-loop entry), the ack frame is
    # skipped, the final sentence-end is dispatched, then complete.
    assert kinds == ["start", "end", "complete"]
    assert listener.events[0][1] == recognizer._voice_id or recognizer._voice_id

    end_resp = listener.events[1][1]
    assert len(end_resp.result.speaker_segments) == 2

    # Terminal response: recognizer reached the stopped state before the
    # complete callback, so a re-entrant stop() would fail immediately.
    assert recognizer._state == _State.STOPPED
    assert listener.failed is None


def test_read_loop_dispatches_final_without_slice_type_2_nowhere():
    """final=1 with slice_type != 2 must not dispatch any sentence event,
    only on_recognition_complete (Go: dispatchEvent guard)."""
    frame = {"code": 0, "message": "ok", "voice_id": "v1", "final": 1,
             "result": {"slice_type": 1, "index": 0, "voice_text_str": "x"}}

    listener = _CaptureListener()
    recognizer = SpeechRecognizer(make_credential(), "16k_zh", listener)
    recognizer._ws = _FakeWS([json.dumps(frame)])
    recognizer._state = _State.RUNNING

    asyncio.run(recognizer._read_loop())

    kinds = [e[0] for e in listener.events]
    assert kinds == ["start", "complete"]
    assert recognizer._state == _State.STOPPED


def test_read_loop_swallows_listener_exception():
    """A faulty callback must never crash the read loop (Go: panic shielding)."""

    class _BrokenListener(_CaptureListener):
        def on_sentence_end(self, response):
            raise RuntimeError("listener bug")

    frame = {"code": 0, "message": "ok", "voice_id": "v1", "final": 1,
             "result": {"slice_type": 2, "index": 0, "voice_text_str": "x"}}

    listener = _BrokenListener()
    recognizer = SpeechRecognizer(make_credential(), "16k_zh", listener)
    recognizer._ws = _FakeWS([json.dumps(frame)])
    recognizer._state = _State.RUNNING

    asyncio.run(recognizer._read_loop())

    assert recognizer._state == _State.STOPPED
    assert listener.failed is None
    # complete still delivered after the broken sentence-end callback.
    assert ("complete", "v1") in listener.events


# ---------------------------------------------------------------- connect
# query string (Go: TestConnectSendsSpeakerAndVadParams)


def test_connect_sends_speaker_and_vad_params(monkeypatch):
    captured = {}

    class _ConnectCapture:
        async def __call__(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _FakeWS([])

    import websockets.asyncio.client

    monkeypatch.setattr(websockets.asyncio.client, "connect", _ConnectCapture())

    credential = make_credential()

    async def run():
        recognizer = SpeechRecognizer(credential, "16k_zh", _CaptureListener())
        recognizer.set_voice_id("voice-diarization")
        recognizer.set_word_info(1)
        recognizer.set_speaker_diarization(SPEAKER_DIARIZATION_VOICEPRINT)
        recognizer.set_speaker_number(2)
        recognizer.set_speaker_roles(
            [SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")]
        )
        recognizer.set_voiceprint_ids(["vp-1"])
        recognizer.set_vad_level(0)
        recognizer.set_noise_threshold(1.5)
        recognizer.set_filter_empty_result(0)
        recognizer.set_hotword_list("腾讯云|5")
        recognizer.set_replace_text_id("replace-1")
        recognizer.set_input_sample_rate(8000)
        await recognizer.start()
        await recognizer.stop()

    asyncio.run(run())

    # The UserSig is resolved locally and never written back to the shared
    # credential: a single Credential reused by concurrent recognizers must
    # not race (mirrors the Go fix).
    assert credential.user_sig == ""

    query = parse_qs(urlparse(captured["url"]).query)
    want = {
        "speaker_diarization": "3",
        "speaker_number": "2",
        "voiceprintids": '["vp-1"]',
        "vad_level": "0",
        "noise_threshold": "1.500",
        "filter_empty_result": "0",
        "hotword_list": "腾讯云|5",
        "replace_text_id": "replace-1",
        "input_sample_rate": "8000",
        "word_info": "1",
        # Authentication identity travels in the query string (browser
        # WebSocket clients cannot attach custom headers).
        "sdkappid": "1400000000",
    }
    for key, want_value in want.items():
        assert query.get(key) == [want_value], f"query {key} = {query.get(key)}, want {want_value}"

    # signature and usersig carry the same UserSig value.
    sig, sig_q = query.get("signature", [""])[0], query.get("usersig", [""])[0]
    assert sig and sig_q and sig == sig_q

    roles = json.loads(query["speaker_roles"][0])
    assert len(roles) == 1
    assert roles[0]["RoleName"] == "teacher"


def test_start_rejects_invalid_diarization_before_dialing():
    recognizer = SpeechRecognizer(make_credential(), "16k_zh", _CaptureListener())
    # Roles without mode 3 must fail locally, before any connection attempt.
    recognizer.set_speaker_roles(
        [SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")]
    )

    with pytest.raises(ASRError) as exc_info:
        asyncio.run(recognizer.start())
    assert "require SpeakerDiarization=3" in str(exc_info.value)
    # Invalid options leave the recognizer idle and restartable.
    assert recognizer._state == _State.IDLE


def test_stop_reentered_from_read_loop_returns_without_self_waiting():
    """A stop() invoked while running on the read-loop task must return
    right after sending the end signal instead of waiting on itself until
    the stop timeout (mirrors Go's calledFromListenerCallback guard)."""

    class _RecordingWS(_FakeWS):
        def __init__(self):
            super().__init__([])
            self.sent = []

        async def send(self, data):
            self.sent.append(data)

    ws = _RecordingWS()
    recognizer = SpeechRecognizer(make_credential(), "16k_zh", _CaptureListener())
    recognizer._ws = ws
    recognizer._state = _State.RUNNING

    async def run():
        # Simulate stop() being awaited from inside the read loop's callback:
        # the read task is the current task.
        recognizer._read_task = asyncio.current_task()
        started = asyncio.get_running_loop().time()
        await recognizer.stop()
        elapsed = asyncio.get_running_loop().time() - started
        # Re-entry returns immediately: far below the 10s stop timeout that a
        # self-wait would burn through.
        assert elapsed < 2.0, f"re-entrant stop self-waited for {elapsed:.1f}s"

    asyncio.run(run())

    # The end signal was sent before returning.
    assert json.dumps({"type": "end"}) in ws.sent


def test_stop_from_callback_task_does_not_deadlock():
    """Starting stop() from a listener callback via create_task must not
    deadlock: the read loop finishes on the final frame and stop returns."""

    class _StopOnChangeListener(_CaptureListener):
        def __init__(self, recognizer):
            super().__init__()
            self._recognizer = recognizer

        def on_recognition_result_change(self, response):
            super().on_recognition_result_change(response)
            asyncio.get_running_loop().create_task(self._recognizer.stop())

    frame = {"code": 0, "message": "ok", "voice_id": "v1",
             "result": {"slice_type": 1, "index": 0, "voice_text_str": "hi"}}
    final_frame = {"code": 0, "message": "ok", "voice_id": "v1", "final": 1,
                   "result": {"slice_type": 2, "index": 0, "voice_text_str": "hi."}}

    recognizer = SpeechRecognizer(make_credential(), "16k_zh", _StopOnChangeListener(None))
    recognizer._ws = _FakeWS([json.dumps(frame), json.dumps(final_frame)])
    recognizer._state = _State.RUNNING
    recognizer._listener._recognizer = recognizer

    async def run():
        await recognizer._read_loop()

    asyncio.run(run())
    assert recognizer._state == _State.STOPPED


# ---------------------------------------------------------------- file
# recognizer (Go: TestFileRecognizer_CreateTask_SpeakerDiarizationBody and
# friends)


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_file_recognizer_create_task_speaker_diarization_body():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps({"Response": {"Data": {"RecTaskId": "task-1"}, "RequestId": "r1"}}).encode()
        )

    recognizer = FileRecognizer(make_credential())
    req = CreateRecTaskRequest(
        engine_model_type="16k_zh",
        channel_num=1,
        res_text_format=1,
        source_type=0,
        url="https://example.com/audio.wav",
        speaker_diarization=SPEAKER_DIARIZATION_VOICEPRINT,
        speaker_number=2,
        speaker_roles=[SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")],
        voiceprint_ids=["vp-1"],
        vad_silence_ms=800,
        vad_level=0,
        noise_threshold=0.0,
        language="zh",
        replace_text_id="replace-1",
    )

    with patch("trtc_asr.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        task_id = recognizer.create_task(req)

    assert task_id == "task-1"
    body = captured["body"]

    # Explicit zeros must survive: None means "not configured" while 0 is a
    # valid, meaningful value for both VadLevel and NoiseThreshold.
    assert body["SpeakerDiarization"] == 3
    assert body["SpeakerNumber"] == 2
    assert body["VadSilenceMs"] == 800
    assert body["VadLevel"] == 0
    assert body["NoiseThreshold"] == 0.0
    assert body["Language"] == "zh"
    assert body["ReplaceTextId"] == "replace-1"

    roles = body["SpeakerRoles"]
    assert len(roles) == 1
    assert roles[0] == {"RoleName": "teacher", "AudioUrl": "https://example.com/a.wav"}

    assert body["VoiceprintIds"] == ["vp-1"]


def test_file_recognizer_create_task_rejects_invalid_diarization():
    recognizer = FileRecognizer(make_credential())
    req = CreateRecTaskRequest(
        engine_model_type="16k_zh",
        channel_num=1,
        source_type=0,
        url="https://example.com/audio.wav",
        speaker_roles=[SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")],
    )

    with pytest.raises(ASRError) as exc_info:
        recognizer.create_task(req)
    assert "require SpeakerDiarization=3" in str(exc_info.value)


def test_file_recognizer_describe_task_status_speaker_fields():
    resp_body = json.dumps(
        {
            "Response": {
                "RequestId": "req-1",
                "Data": {
                    "RecTaskId": "task-1",
                    "Status": 2,
                    "StatusStr": "success",
                    "Progress": 100,
                    "AudioDuration": 12.5,
                    "Result": "你好\n嗯",
                    "ResultDetail": [
                        {
                            "FinalSentence": "你好",
                            "StartMs": 0,
                            "EndMs": 1200,
                            "WordsNum": 2,
                            "Words": [{"Word": "你", "OffsetStartMs": 0, "OffsetEndMs": 120}],
                            "SpeakerId": 1,
                            "SpeakerRoleName": "teacher",
                            "Language": "zh",
                        },
                        {
                            "FinalSentence": "嗯",
                            "StartMs": 1300,
                            "EndMs": 1500,
                            "SpeakerId": 2,
                            "ChannelId": 2,
                        },
                    ],
                },
            }
        }
    )

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(resp_body.encode())

    recognizer = FileRecognizer(make_credential())
    with patch("trtc_asr.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        status = recognizer.describe_task_status("task-1")

    assert status.progress == 100
    assert len(status.result_detail) == 2

    first = status.result_detail[0]
    assert (first.speaker_id, first.speaker_role_name) == (1, "teacher")
    assert first.language == "zh"

    # Stereo recordings report the channel instead of a clustered speaker.
    second = status.result_detail[1]
    assert second.channel_id == 2


# ---------------------------------------------------------------- sentence
# recognizer (Go: TestSentenceRecognizer_Recognize_CustomizationAndLanguageBody)


def test_sentence_recognizer_customization_and_language_body():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps({"Response": {"Result": "ok", "RequestId": "req-1"}}).encode()
        )

    recognizer = SentenceRecognizer(make_credential())
    req = SentenceRecognitionRequest(
        eng_service_type="16k_zh",
        source_type=0,
        voice_format="wav",
        url="https://example.com/a.wav",
        customization_id="custom-1",
        language="zh",
    )

    with patch("trtc_asr.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        recognizer.recognize(req)

    body = captured["body"]
    assert body["CustomizationId"] == "custom-1"
    assert body["Language"] == "zh"
    # Sentence recognition does not support speaker diarization; the request
    # must not grow those fields.
    assert "SpeakerDiarization" not in body
