"""Credential site selection and endpoint resolution."""

from urllib.parse import urlparse

import pytest

from trtc_asr.credential import (
    HOST_CN,
    HOST_INTL,
    SITE_CN,
    SITE_INTL,
    Credential,
    host_for_site,
    resolve_http_endpoint,
    resolve_ws_endpoint,
)
from trtc_asr.errors import ASRError, ERR_INVALID_PARAM
from trtc_asr.file_recognizer import FileRecognizer
from trtc_asr.sentence_recognizer import SentenceRecognizer
from trtc_asr.speech_recognizer import SpeechRecognizer


def test_host_for_site():
    assert host_for_site("") == HOST_CN
    assert host_for_site(SITE_CN) == HOST_CN
    assert host_for_site("CN") == HOST_CN
    assert host_for_site(" cn ") == HOST_CN
    assert host_for_site(SITE_INTL) == HOST_INTL
    assert host_for_site("INTL") == HOST_INTL
    with pytest.raises(ASRError) as exc:
        host_for_site("mars")
    assert exc.value.code == ERR_INVALID_PARAM


def test_resolve_endpoints_honor_override_and_site():
    assert resolve_ws_endpoint("", SITE_INTL) == "wss://" + HOST_INTL
    assert resolve_http_endpoint("", "") == "https://" + HOST_CN
    assert resolve_ws_endpoint("wss://mock.local", SITE_INTL) == "wss://mock.local"


def test_credential_set_site():
    cred = Credential(1, 2, "k")
    assert cred.site == ""
    cred.set_site(SITE_INTL)
    assert cred.site == SITE_INTL


def test_recognizers_resolve_site_endpoints():
    cred = Credential(1300000000, 1400000000, "secret")
    cred.set_site(SITE_INTL)

    speech = SpeechRecognizer(cred, "16k_zh", None)
    assert resolve_ws_endpoint(speech._endpoint, speech._credential.site) == (
        "wss://asr-intl.cloud-rtc.com"
    )
    speech.set_endpoint("wss://127.0.0.1:9")
    assert resolve_ws_endpoint(speech._endpoint, speech._credential.site) == (
        "wss://127.0.0.1:9"
    )

    sent = SentenceRecognizer(cred)
    assert resolve_http_endpoint(sent._endpoint, sent._credential.site) == (
        "https://asr-intl.cloud-rtc.com"
    )

    file_r = FileRecognizer(cred)
    assert resolve_http_endpoint(file_r._endpoint, file_r._credential.site) == (
        "https://asr-intl.cloud-rtc.com"
    )


def test_default_site_is_domestic():
    cred = Credential(1, 2, "k")
    assert resolve_ws_endpoint("", cred.site) == "wss://asr.cloud-rtc.com"


def test_speech_recognizer_handshake_uses_intl_host(monkeypatch):
    import asyncio

    import websockets.asyncio.client

    captured = {}

    class _FakeWS:
        closed = False

        async def send(self, _data):
            return None

        async def recv(self):
            await asyncio.sleep(0)
            raise asyncio.CancelledError

        async def close(self):
            self.closed = True

    class _ConnectCapture:
        async def __call__(self, url, **kwargs):
            captured["url"] = url
            return _FakeWS()

    monkeypatch.setattr(websockets.asyncio.client, "connect", _ConnectCapture())

    cred = Credential(1300000000, 1400000000, "test-secret")
    cred.set_site(SITE_INTL)

    async def run():
        recognizer = SpeechRecognizer(cred, "16k_zh", None)
        await recognizer.start()
        await recognizer.stop()

    asyncio.run(run())
    assert urlparse(captured["url"]).netloc == HOST_INTL
