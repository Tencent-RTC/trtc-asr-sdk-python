"""Tests for SDK self-identification reporting across the three transports."""

import asyncio
import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from trtc_asr import __version__
from trtc_asr.credential import Credential
from trtc_asr.file_recognizer import CreateRecTaskRequest, FileRecognizer
from trtc_asr.sdkinfo import (
    SDK_LANGUAGE,
    SDK_TYPE,
    SDK_VERSION,
    sdk_platform,
    sdk_report_params,
    sdk_report_query,
)
from trtc_asr.sentence_recognizer import (
    SentenceRecognitionRequest,
    SentenceRecognizer,
)
from trtc_asr.signature import SignatureParams
from trtc_asr.speech_recognizer import SpeechRecognizer


def make_credential():
    return Credential(app_id=1300000000, sdk_app_id=1400000000, secret_key="test-secret")


def assert_sdk_report_params(query: dict) -> None:
    """Check that a captured request query carries the SDK identification the
    service relies on for diagnostics."""
    assert query["sdk_lang"] == ["python"]
    assert query["sdk_type"] == [SDK_TYPE]
    assert query["version"] == [SDK_VERSION]
    assert query["platform"] == [sdk_platform()]


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


class _FakeWS:
    """Minimal WebSocket stub: the handshake URL is all these tests inspect."""

    def __init__(self):
        self.closed = False

    async def send(self, _data):
        return None

    async def recv(self):
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------- sdkinfo


def test_sdk_language_and_version_constants():
    assert SDK_LANGUAGE == "python"
    # All six language SDKs are server-side, so the type is a constant.
    assert SDK_TYPE == "server"
    # The version has a single source of truth: __init__ re-exports sdkinfo's.
    assert __version__ == SDK_VERSION


def test_sdk_platform_normalizes_known_systems():
    for system, want in [
        ("Darwin", "mac"),
        ("Windows", "windows"),
        ("Linux", "linux"),
        ("Android", "android"),
        ("iOS", "ios"),
    ]:
        with patch("trtc_asr.sdkinfo.platform.system", return_value=system):
            assert sdk_platform() == want


def test_sdk_platform_passes_unknown_system_through():
    # An unexpected platform must surface in telemetry rather than being
    # silently misattributed to one of the known values.
    with patch("trtc_asr.sdkinfo.platform.system", return_value="FreeBSD"):
        assert sdk_platform() == "freebsd"


def test_sdk_report_params_keys():
    assert set(sdk_report_params()) == {"platform", "sdk_lang", "sdk_type", "version"}


def test_sdk_report_query_is_sorted_and_encoded():
    query = sdk_report_query()
    assert not query.startswith("&")
    keys = [part.split("=")[0] for part in query.split("&")]
    assert keys == sorted(keys)
    assert parse_qs(query)["sdk_lang"] == ["python"]
    assert parse_qs(query)["sdk_type"] == ["server"]


# ---------------------------------------------------------- realtime (ws)


def test_speech_recognizer_handshake_reports_sdk_identity(monkeypatch):
    captured = {}

    class _ConnectCapture:
        async def __call__(self, url, **kwargs):
            captured["url"] = url
            return _FakeWS()

    import websockets.asyncio.client

    monkeypatch.setattr(websockets.asyncio.client, "connect", _ConnectCapture())

    async def run():
        recognizer = SpeechRecognizer(make_credential(), "16k_zh", None)
        recognizer.set_voice_id("voice-sdkinfo")
        await recognizer.start()
        await recognizer.stop()

    asyncio.run(run())

    query = parse_qs(urlparse(captured["url"]).query)
    assert_sdk_report_params(query)
    # The pre-existing protocol parameters must survive the addition.
    assert query["voice_id"] == ["voice-sdkinfo"]
    assert query["engine_model_type"] == ["16k_zh"]
    assert query["secretid"] == ["1300000000"]
    assert query["signature"] == query["usersig"]


def test_signature_params_report_sdk_identity():
    params = SignatureParams(
        app_id=1300000000,
        engine_model_type="16k_zh",
        voice_id="voice-1",
    )
    query = parse_qs(params.build_query_string_with_signature("sig"))
    assert_sdk_report_params(query)
    assert query["needvad"] == ["1"]


# ------------------------------------------------------------ sentence (http)


def test_sentence_recognizer_reports_sdk_identity():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeHTTPResponse(
            json.dumps({"Response": {"Result": "ok", "RequestId": "req-1"}}).encode()
        )

    recognizer = SentenceRecognizer(make_credential())
    req = SentenceRecognitionRequest(
        eng_service_type="16k_zh",
        source_type=0,
        voice_format="wav",
        url="https://example.com/a.wav",
    )

    with patch("trtc_asr.sentence_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        recognizer.recognize(req)

    query = parse_qs(urlparse(captured["url"]).query)
    assert_sdk_report_params(query)
    assert query["AppId"] == ["1300000000"]
    assert query["Secretid"] == ["1300000000"]
    assert query["RequestId"]
    assert query["Timestamp"]


# ---------------------------------------------------------------- file (http)


def test_file_recognizer_create_task_reports_sdk_identity():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeHTTPResponse(
            json.dumps(
                {"Response": {"Data": {"RecTaskId": "task-1"}, "RequestId": "r1"}}
            ).encode()
        )

    recognizer = FileRecognizer(make_credential())
    req = CreateRecTaskRequest(
        engine_model_type="16k_zh",
        channel_num=1,
        source_type=0,
        url="https://example.com/audio.wav",
    )

    with patch("trtc_asr.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        assert recognizer.create_task(req) == "task-1"

    query = parse_qs(urlparse(captured["url"]).query)
    assert_sdk_report_params(query)
    assert query["AppId"] == ["1300000000"]
    assert query["RequestId"]


def test_file_recognizer_describe_task_status_reports_sdk_identity():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "Response": {
                        "RequestId": "req-1",
                        "Data": {"RecTaskId": "task-1", "Status": 2, "Progress": 100},
                    }
                }
            ).encode()
        )

    recognizer = FileRecognizer(make_credential())
    with patch("trtc_asr.file_recognizer.urllib.request.urlopen", side_effect=fake_urlopen):
        recognizer.describe_task_status("task-1")

    query = parse_qs(urlparse(captured["url"]).query)
    assert_sdk_report_params(query)
    assert query["AppId"] == ["1300000000"]
