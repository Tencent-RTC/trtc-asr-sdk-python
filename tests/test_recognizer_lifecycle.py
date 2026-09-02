import asyncio

import pytest

from trtc_asr.credential import Credential
from trtc_asr.errors import ASRError, ERR_NOT_STARTED, ERR_WRITE_FAILED
from trtc_asr.speech_recognizer import _State, SpeechRecognitionListener, SpeechRecognizer


class _Listener(SpeechRecognitionListener):
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


class _FailingWS:
    async def send(self, _data):
        raise RuntimeError("send failed")

    async def close(self):
        return None


def _new_recognizer() -> SpeechRecognizer:
    credential = Credential(1300000000, 1400000000, "secret")
    return SpeechRecognizer(credential, "16k_zh_en", _Listener())


def test_write_before_start_raises_not_started():
    recognizer = _new_recognizer()

    with pytest.raises(ASRError) as exc_info:
        asyncio.run(recognizer.write(b"abc"))

    assert exc_info.value.code == ERR_NOT_STARTED


def test_stop_when_already_stopped_is_noop():
    recognizer = _new_recognizer()
    recognizer._state = _State.STOPPED

    asyncio.run(recognizer.stop())

    assert recognizer._state == _State.STOPPED


def test_stop_without_connection_raises_not_started():
    recognizer = _new_recognizer()
    recognizer._state = _State.RUNNING
    recognizer._ws = None

    with pytest.raises(ASRError) as exc_info:
        asyncio.run(recognizer.stop())

    assert exc_info.value.code == ERR_NOT_STARTED


def test_stop_send_failure_sets_stopped_and_raises_write_failed():
    recognizer = _new_recognizer()
    recognizer._state = _State.RUNNING
    recognizer._ws = _FailingWS()

    with pytest.raises(ASRError) as exc_info:
        asyncio.run(recognizer.stop())

    assert exc_info.value.code == ERR_WRITE_FAILED
    assert recognizer._state == _State.STOPPED
