"""v2 断点续传：query 参数下发、本地校验与首响应 speaker_continue 解析。"""

import threading
import time

import pytest

from trtc_asr.errors import ERR_INVALID_PARAM
from trtc_asr.signature import SignatureParams
from trtc_asr.speech_recognizer import (
    SPEAKER_CONTEXT_ASYNC,
    SPEAKER_CONTEXT_SYNC,
    SpeechRecognitionResponse,
    SpeechRecognizer,
)


def test_signature_params_emit_context_query():
    p = SignatureParams(app_id=1, engine_model_type="bigmodel", voice_id="v")
    p.speaker_diarization = 1
    p.enable_speaker_context = SPEAKER_CONTEXT_SYNC
    p.speaker_context_id = "abc123"
    query = p.build_query_string()
    assert "enable_speaker_context=1" in query
    assert "speaker_context_id=abc123" in query


def test_signature_params_context_off_omits_params():
    p = SignatureParams(app_id=1, engine_model_type="bigmodel", voice_id="v")
    p.speaker_diarization = 1
    p.speaker_context_id = "abc123"  # 未开 context 时即使带了 id 也不下发
    assert "speaker_context" not in p.build_query_string()


def test_enable_speaker_context_requires_diarization():
    recognizer = SpeechRecognizer(_credential_stub(), "bigmodel", _listener_stub())
    recognizer.set_enable_speaker_context(SPEAKER_CONTEXT_SYNC)
    with pytest.raises(Exception, match="set_speaker_diarization"):
        recognizer._validate_options()


def test_enable_speaker_context_rejects_unknown_mode():
    recognizer = SpeechRecognizer(_credential_stub(), "bigmodel", _listener_stub())
    recognizer.set_speaker_diarization(1)
    recognizer.set_enable_speaker_context(3)
    with pytest.raises(Exception, match="EnableSpeakerContext"):
        recognizer._validate_options()


def test_response_parses_speaker_continue():
    resp = SpeechRecognitionResponse.from_dict({
        "code": 0,
        "message": "success",
        "voice_id": "v1",
        "speaker_continue": {
            "continue_status": "resumed",
            "speaker_context_id": "abc123",
        },
    })
    assert resp.speaker_continue is not None
    assert resp.speaker_continue.continue_status == "resumed"
    assert resp.speaker_continue.speaker_context_id == "abc123"

    # 未开启时字段缺省为 None
    plain = SpeechRecognitionResponse.from_dict({"code": 0, "voice_id": "v1"})
    assert plain.speaker_continue is None


def test_end_to_end_query_and_continue():
    """走真实 Start 路径：query 带上两个参数，首响应 speaker_continue 被捕获。"""
    import asyncio

    import websockets.asyncio.server

    received_query = {}
    captured = {}

    async def handler(ws):
        received_query["query"] = ws.request.path
        await ws.send(
            '{"code":0,"message":"success","voice_id":"v1",'
            '"speaker_continue":{"continue_status":"resumed","speaker_context_id":"abc"}}'
        )
        await ws.send(
            '{"code":0,"message":"success","voice_id":"v1","final":1,'
            '"result":{"slice_type":2,"index":0}}'
        )
        async for _ in ws:
            pass

    async def main():
        async with websockets.asyncio.server.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            recognizer = SpeechRecognizer(
                _credential_stub(), "bigmodel", _listener_stub()
            )
            recognizer._endpoint = f"ws://127.0.0.1:{port}"
            recognizer.set_speaker_diarization(1)
            recognizer.set_enable_speaker_context(SPEAKER_CONTEXT_SYNC)
            recognizer.set_speaker_context_id("abc")
            await recognizer.start()
            for _ in range(100):
                if recognizer.speaker_continue is not None:
                    break
                await asyncio.sleep(0.02)
            captured["continue"] = recognizer.speaker_continue
            await recognizer.stop()

    asyncio.run(main())

    assert "enable_speaker_context=1" in received_query["query"]
    assert "speaker_context_id=abc" in received_query["query"]
    assert captured["continue"] is not None
    assert captured["continue"].continue_status == "resumed"


# ---- 简易桩 ----

class _CredentialStub:
    app_id = 1400000000
    sdk_app_id = 1400000000
    secret_key = "test-secret"
    user_sig = ""
    site = ""  # 空 site 时 resolve_ws_endpoint 直接用 endpoint


class _ListenerStub:
    def on_recognition_start(self, response):
        pass

    def on_sentence_begin(self, response):
        pass

    def on_recognition_result_change(self, response):
        pass

    def on_sentence_end(self, response):
        pass

    def on_recognition_complete(self, response):
        pass

    def on_fail(self, response, error):
        pass


def _credential_stub():
    return _CredentialStub()


def _listener_stub():
    return _ListenerStub()
