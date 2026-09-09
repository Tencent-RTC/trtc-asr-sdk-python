"""v3 offline interface tests (/v3/transcribe, /v3/create_transcription,
/v3/describe_transcription): wire format, flat responses, error mapping
(mirrors Go asr/v3 sentence_recognizer_test.go / file_recognizer_test.go)."""

import base64
import json
import urllib.error
from unittest.mock import patch

import pytest

from trtc_asr.errors import ASRError, ERR_INVALID_PARAM, ERR_SERVER_ERROR
from trtc_asr.v3 import (
    SOURCE_TYPE_DATA,
    SOURCE_TYPE_URL,
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_VOICEPRINT,
    AudioURLItem,
    Context,
    CreateTranscriptionRequest,
    FileRecognizer,
    SentenceRecognizer,
    SpeakerRole,
    TranscribeRequest,
    new_credential,
)
from trtc_asr.v3.file_recognizer import TASK_STATUS_SUCCESS
from trtc_asr.v3.wire import START_FRAME_MAX_BYTES


def make_credential():
    return new_credential(1400000000, "test-secret")


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(status: int, body: bytes):
    err = urllib.error.HTTPError(None, status, "err", None, None)
    err.read = lambda: body  # HTTPError.read reads from fp; patch directly
    return err


# ---------------------------------------------------------------- transcribe


def test_transcribe_wire():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "code": 0,
                    "message": "success",
                    "request_id": "req-1",
                    "result": "你好世界",
                    "audio_duration": 1500,
                    "language": "zh",
                    "word_size": 2,
                    "word_list": [
                        {"word": "你好", "start_time": 0, "end_time": 500},
                        {"word": "世界", "start_time": 500, "end_time": 1500},
                    ],
                }
            ).encode()
        )

    recognizer = SentenceRecognizer(make_credential())
    needvad = 0
    req = TranscribeRequest(
        engine_model_type="16k_zh_en",
        source_type=SOURCE_TYPE_DATA,
        voice_format="pcm",
        data=base64.b64encode(b"pcm-data").decode("ascii"),
        data_len=8,
        hotword_list="腾讯云|10",
        needvad=needvad,
        language="zh",
        convert_num_mode=1,
        word_info=1,
        filter_punc=1,
        input_sample_rate=8000,
        customization_id="cust-1",
        context=Context(text="bg", terms=["ASR"]),
    )

    with patch(
        "trtc_asr.v3.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        resp = recognizer.recognize(req)

    assert captured["url"].endswith("/v3/transcribe")
    assert "?" not in captured["url"], "v3 carries everything in the body"
    body = captured["body"]

    auth = body["auth"]
    assert auth["sdkappid"] == "1400000000"
    assert auth["usersig"]
    assert auth["request_id"]
    assert "business" not in auth

    params = body["params"]
    assert params["engine_model_type"] == "16k_zh_en"
    assert params["source_type"] == 1
    assert params["voice_format"] == "pcm"
    assert params["data"] == base64.b64encode(b"pcm-data").decode("ascii")
    assert params["data_len"] == 8
    assert params["hotword_list"] == "腾讯云|10"
    assert params["needvad"] == 0  # explicit 0 honored
    assert params["language"] == "zh"
    assert params["word_info"] == 1
    assert params["filter_punc"] == 1
    assert params["input_sample_rate"] == 8000
    assert params["customization_id"] == "cust-1"
    assert params["sdk_info"]["sdk_lang"] == "python"
    assert params["context"] == {"text": "bg", "terms": ["ASR"]}
    # snake_case everywhere — v1 PascalCase names must not appear.
    for bad in ("EngSerViceType", "SourceType", "VoiceFormat", "Data", "Url", "DataLen"):
        assert bad not in params

    assert resp.result == "你好世界"
    assert resp.audio_duration == 1500
    assert resp.language == "zh"
    assert resp.request_id == "req-1"
    assert resp.word_list[0].word == "你好"
    assert (resp.word_list[0].start_time, resp.word_list[0].end_time) == (0, 500)


def test_transcribe_auth_error_http200():
    """Auth failure arrives with HTTP 200 and code 4002 — judge by body."""

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(
            json.dumps({"code": 4002, "message": "auth failed", "request_id": "req-x"}).encode(),
            status=200,
        )

    recognizer = SentenceRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        with pytest.raises(ASRError) as exc_info:
            recognizer.recognize_url("https://example.com/a.wav", "wav", "16k_zh_en")

    assert exc_info.value.code == 4002
    assert "req-x" in exc_info.value.message


def test_transcribe_http_error_body_code():
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(
            json.dumps({"code": 5000, "message": "no worker available", "request_id": "r"}).encode(),
            status=503,
        )

    recognizer = SentenceRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        with pytest.raises(ASRError) as exc_info:
            recognizer.recognize_url("https://example.com/a.wav", "wav", "16k_zh_en")
    assert exc_info.value.code == 5000


def test_transcribe_non_v3_json_body():
    """A non-2xx JSON body without a numeric code must not be treated as
    success."""

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(json.dumps({"error": "upstream unavailable"}).encode(), status=502)

    recognizer = SentenceRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        with pytest.raises(ASRError) as exc_info:
            recognizer.recognize_url("https://example.com/a.wav", "wav", "16k_zh_en")
    assert exc_info.value.code == ERR_SERVER_ERROR
    assert "502" in exc_info.value.message


def test_transcribe_convenience_and_validation():
    recognizer = SentenceRecognizer(make_credential())
    with pytest.raises(ASRError):
        recognizer.recognize_data(b"", "pcm", "16k_zh_en")
    with pytest.raises(ASRError):
        recognizer.recognize(
            TranscribeRequest(source_type=SOURCE_TYPE_DATA, voice_format="pcm")
        )
    # nil request must raise, not crash.
    with pytest.raises(ASRError):
        recognizer.recognize_data_with_options(b"abc", None)
    bad_needvad = 2
    with pytest.raises(ASRError):
        recognizer.recognize(
            TranscribeRequest(
                engine_model_type="16k_zh_en",
                source_type=SOURCE_TYPE_URL,
                voice_format="wav",
                url="https://example.com/a.wav",
                needvad=bad_needvad,
            )
        )


# ---------------------------------------------------------------- create


def test_create_task_wire():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {"code": 0, "message": "success", "request_id": "r", "transcription_id": "tid-abc"}
            ).encode()
        )

    recognizer = FileRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        task_id = recognizer.create_task(
            CreateTranscriptionRequest(
                engine_model_type="16k_zh_en",
                channel_num=1,
                res_text_format=1,
                source_type=SOURCE_TYPE_URL,
                url="https://example.com/a.wav",
                callback_url="https://example.com/cb",
                hotword_id="hw-1",
                vad_silence_ms=600,
                vad_level=1,
                language="zh",
                speaker_diarization=SPEAKER_DIARIZATION_VOICEPRINT,
                speaker_roles=[SpeakerRole(role_name="teacher", audio_url="https://example.com/t.wav")],
                voiceprint_ids=["vp-1"],
            )
        )
    assert task_id == "tid-abc"
    assert captured["url"].endswith("/v3/create_transcription")

    auth = captured["body"]["auth"]
    assert auth["sdkappid"] == "1400000000" and auth["usersig"] and auth["request_id"]
    params = captured["body"]["params"]
    for key in (
        "engine_model_type",
        "channel_num",
        "res_text_format",
        "source_type",
        "url",
        "callback_url",
        "hotword_id",
        "vad_silence_ms",
        "vad_level",
        "language",
        "speaker_diarization",
        "speaker_roles",
        "voiceprint_ids",
        "sdk_info",
    ):
        assert key in params, key
    for bad in ("EngineModelType", "ChannelNum", "ResTextFormat", "Url", "CallbackUrl"):
        assert bad not in params
    assert params["speaker_roles"] == [
        {"role_name": "teacher", "audio_url": "https://example.com/t.wav"}
    ]


def test_create_task_audio_urls():
    """Distributed recording: audio_urls requires source_type=0 with
    url/data empty — the SDK must let it through and serialize audio_urls."""

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {"code": 0, "message": "success", "request_id": "r", "transcription_id": "tid-d"}
            ).encode()
        )

    recognizer = FileRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        task_id = recognizer.create_task(
            CreateTranscriptionRequest(
                engine_model_type="16k_zh_en",
                channel_num=1,
                res_text_format=1,
                source_type=SOURCE_TYPE_URL,  # zero value; url/data stay empty
                speaker_diarization=SPEAKER_DIARIZATION_CLUSTER,
                audio_urls=[
                    AudioURLItem(index=0, url="https://example.com/a.wav", label="a"),
                    AudioURLItem(index=1, url="https://example.com/b.wav"),
                ],
            )
        )
    assert task_id == "tid-d"


def test_create_task_audio_urls_reject():
    recognizer = FileRecognizer(make_credential())
    base = dict(
        engine_model_type="16k_zh_en",
        channel_num=1,
        res_text_format=1,
        audio_urls=[AudioURLItem(index=0, url="https://example.com/a.wav")],
    )
    cases = [
        ("with url", lambda r: setattr(r, "url", "https://example.com/x.wav")),
        ("with data", lambda r: (setattr(r, "data", "AAAA"), setattr(r, "data_len", 3))),
        ("non-zero source_type", lambda r: setattr(r, "source_type", SOURCE_TYPE_DATA)),
        ("negative index", lambda r: setattr(r, "audio_urls", [AudioURLItem(index=-1, url="https://x")])),
        ("duplicate index", lambda r: setattr(
            r, "audio_urls",
            [AudioURLItem(index=0, url="https://x"), AudioURLItem(index=0, url="https://y")],
        )),
        ("empty url", lambda r: setattr(r, "audio_urls", [AudioURLItem(index=0, url=" ")])),
        ("non-http url", lambda r: setattr(r, "audio_urls", [AudioURLItem(index=0, url="ftp://x/a")])),
    ]
    for name, mutate in cases:
        req = CreateTranscriptionRequest(**base)
        mutate(req)
        with pytest.raises(ASRError) as exc_info:
            recognizer.create_task(req)
        assert exc_info.value.code == ERR_INVALID_PARAM, name


def test_create_task_missing_transcription_id():
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(json.dumps({"code": 0, "message": "success"}).encode())

    recognizer = FileRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        with pytest.raises(ASRError):
            recognizer.create_task_from_url("https://example.com/a.wav", "16k_zh_en")


# ---------------------------------------------------------------- describe


def test_describe_task_poll_mapping():
    responses = [
        {"code": 0, "message": "success", "request_id": "r1", "transcription_id": "t1",
         "status": 1, "status_str": "executing", "progress": 40},
        {"code": 0, "message": "success", "request_id": "r2", "transcription_id": "t1",
         "status": 2, "status_str": "success", "progress": 100,
         "audio_duration": 6.312, "result": "全文",
         "result_detail": [
             {
                 "final_sentence": "第一句话。", "start_ms": 0, "end_ms": 1200,
                 "words_num": 1,
                 "words": [{"word": "第一句", "start_time": 0, "end_time": 900}],
                 "speaker_id": 1, "speaker_role_name": "teacher", "channel_id": 0,
                 "speech_speed": 3.2, "language": "zh", "language_b47": "zh-CN",
             }
         ]},
    ]
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(
            {
                "url": req.full_url,
                "body": json.loads(req.data.decode("utf-8")),
            }
        )
        return _FakeHTTPResponse(json.dumps(responses.pop(0)).encode())

    recognizer = FileRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        status = recognizer.wait_for_result_with_interval("t1", 0.005, 5.0)

    assert len(captured) >= 2
    assert captured[0]["body"]["params"]["transcription_id"] == "t1"
    assert captured[0]["url"].endswith("/v3/describe_transcription")

    assert status.status == TASK_STATUS_SUCCESS
    assert status.result == "全文"
    assert status.audio_duration == 6.312
    d = status.result_detail[0]
    assert d.final_sentence == "第一句话。"
    assert (d.speaker_id, d.speaker_role_name) == (1, "teacher")
    assert d.words[0].word == "第一句"
    assert (d.words[0].start_time, d.words[0].end_time) == (0, 900)
    assert d.language_b47 == "zh-CN"


def test_describe_ownership_error():
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(
            json.dumps(
                {"code": 4002, "message": "transcription_id does not belong to this sdkappid",
                 "request_id": "r"}
            ).encode(),
            status=403,
        )

    recognizer = FileRecognizer(make_credential())
    with patch(
        "trtc_asr.v3.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        with pytest.raises(ASRError) as exc_info:
            recognizer.describe_task("tid-foreign")
    assert exc_info.value.code == 4002


def test_with_options_nil_request():
    recognizer = FileRecognizer(make_credential())
    with pytest.raises(ASRError):
        recognizer.create_task_from_data_with_options(b"abc", None)
