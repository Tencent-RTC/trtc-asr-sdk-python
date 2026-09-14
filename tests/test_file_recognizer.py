"""Tests for FileRecognizer."""

import base64
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from trtc_asr.credential import Credential
from trtc_asr.errors import ASRError, ERR_INVALID_PARAM, ERR_SERVER_ERROR, ERR_TIMEOUT
from trtc_asr.file_recognizer import (
    SOURCE_TYPE_DATA,
    SOURCE_TYPE_URL,
    TASK_STATUS_FAILED,
    TASK_STATUS_SUCCESS,
    CreateRecTaskRequest,
    FileRecognizer,
    SentenceDetail,
    SentenceWords,
    TaskStatus,
)


def make_credential():
    return Credential(app_id=1300000000, sdk_app_id=1400000000, secret_key="test-secret")


def make_recognizer():
    return FileRecognizer(make_credential())


# ---- Validation tests ----


class TestValidation:
    def test_create_task_from_data_empty_raises(self):
        r = make_recognizer()
        with pytest.raises(ASRError) as exc_info:
            r.create_task_from_data(b"", "16k_zh")
        assert exc_info.value.code == ERR_INVALID_PARAM

    def test_create_task_from_data_too_large_raises(self):
        r = make_recognizer()
        with pytest.raises(ASRError) as exc_info:
            r.create_task_from_data(b"x" * (5 * 1024 * 1024 + 1), "16k_zh")
        assert exc_info.value.code == ERR_INVALID_PARAM
        assert "5MB" in exc_info.value.message

    def test_create_task_from_url_empty_raises(self):
        r = make_recognizer()
        with pytest.raises(ASRError) as exc_info:
            r.create_task_from_url("", "16k_zh")
        assert exc_info.value.code == ERR_INVALID_PARAM

    def test_missing_engine_raises(self):
        r = make_recognizer()
        req = CreateRecTaskRequest(
            engine_model_type="",
            source_type=SOURCE_TYPE_DATA,
            data="abc",
            data_len=3,
        )
        with pytest.raises(ASRError) as exc_info:
            r.create_task(req)
        assert exc_info.value.code == ERR_INVALID_PARAM
        assert "engine_model_type" in exc_info.value.message

    def test_url_source_missing_url_raises(self):
        r = make_recognizer()
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            source_type=SOURCE_TYPE_URL,
            url="",
        )
        with pytest.raises(ASRError) as exc_info:
            r.create_task(req)
        assert exc_info.value.code == ERR_INVALID_PARAM
        assert "url" in exc_info.value.message

    def test_data_source_missing_data_raises(self):
        r = make_recognizer()
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            source_type=SOURCE_TYPE_DATA,
            data="",
        )
        with pytest.raises(ASRError) as exc_info:
            r.create_task(req)
        assert exc_info.value.code == ERR_INVALID_PARAM
        assert "data" in exc_info.value.message

    def test_describe_task_status_empty_id_raises(self):
        r = make_recognizer()
        with pytest.raises(ASRError) as exc_info:
            r.describe_task_status("")
        assert exc_info.value.code == ERR_INVALID_PARAM

    def test_create_task_from_data_with_options_empty_raises(self):
        r = make_recognizer()
        req = CreateRecTaskRequest(engine_model_type="16k_zh")
        with pytest.raises(ASRError) as exc_info:
            r.create_task_from_data_with_options(b"", req)
        assert exc_info.value.code == ERR_INVALID_PARAM

    def test_negative_channel_num_raises(self):
        r = make_recognizer()
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            channel_num=0,
            source_type=SOURCE_TYPE_DATA,
            data="abc",
            data_len=3,
        )
        with pytest.raises(ASRError) as exc_info:
            r.create_task(req)
        assert exc_info.value.code == ERR_INVALID_PARAM
        assert "channel_num" in exc_info.value.message


# ---- Request building tests ----


class TestRequestBuilding:
    def test_to_dict_data_source(self):
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh_en",
            source_type=SOURCE_TYPE_DATA,
            data="base64data",
            data_len=100,
        )
        d = req.to_dict()
        assert d["EngineModelType"] == "16k_zh_en"
        assert d["SourceType"] == 1
        assert d["Data"] == "base64data"
        assert d["DataLen"] == 100
        assert "Url" not in d

    def test_to_dict_url_source(self):
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            source_type=SOURCE_TYPE_URL,
            url="https://example.com/test.wav",
        )
        d = req.to_dict()
        assert d["Url"] == "https://example.com/test.wav"
        assert "Data" not in d
        assert "DataLen" not in d

    def test_to_dict_optional_fields(self):
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            source_type=SOURCE_TYPE_DATA,
            data="abc",
            data_len=3,
            callback_url="https://example.com/callback",
            filter_dirty=1,
            hotword_id="hw-001",
        )
        d = req.to_dict()
        assert d["CallbackUrl"] == "https://example.com/callback"
        assert d["FilterDirty"] == 1
        assert d["HotwordId"] == "hw-001"

    def test_to_dict_omits_defaults(self):
        req = CreateRecTaskRequest(
            engine_model_type="16k_zh",
            source_type=SOURCE_TYPE_DATA,
            data="abc",
            data_len=3,
        )
        d = req.to_dict()
        assert "CallbackUrl" not in d
        assert "FilterDirty" not in d
        assert "HotwordId" not in d


# ---- Result parsing tests ----


class TestResultParsing:
    def test_task_status_from_dict_basic(self):
        data = {
            "RecTaskId": "task-123",
            "Status": 2,
            "StatusStr": "success",
            "Result": "hello world",
            "ErrorMsg": "",
            "AudioDuration": 2.5,
            "ResultDetail": None,
        }
        status = TaskStatus.from_dict(data)
        assert status.rec_task_id == "task-123"
        assert status.status == 2
        assert status.result == "hello world"
        assert status.audio_duration == 2.5
        assert status.result_detail == []

    def test_task_status_from_dict_with_details(self):
        data = {
            "RecTaskId": "task-456",
            "Status": 2,
            "StatusStr": "success",
            "Result": "hello",
            "ErrorMsg": "",
            "AudioDuration": 1.0,
            "ResultDetail": [
                {
                    "FinalSentence": "hello",
                    "StartMs": 0,
                    "EndMs": 1000,
                    "WordsNum": 1,
                    "Words": [{"Word": "hello", "StartTime": 0, "EndTime": 500}],
                    "SpeechSpeed": 5.0,
                }
            ],
        }
        status = TaskStatus.from_dict(data)
        assert len(status.result_detail) == 1
        detail = status.result_detail[0]
        assert detail.final_sentence == "hello"
        assert detail.start_ms == 0
        assert detail.end_ms == 1000
        assert len(detail.words) == 1
        assert detail.words[0].word == "hello"
        assert detail.words[0].offset_start_ms == 0
        assert detail.words[0].offset_end_ms == 500

    def test_task_status_words_accept_offset_spelling(self):
        """OffsetStartMs / OffsetEndMs is the older spelling; keep accepting it
        as a fallback."""
        data = {
            "RecTaskId": "task-789",
            "Status": 2,
            "ResultDetail": [
                {
                    "FinalSentence": "hello",
                    "Words": [{"Word": "hello", "OffsetStartMs": 120, "OffsetEndMs": 640}],
                }
            ],
        }
        detail = TaskStatus.from_dict(data).result_detail[0]
        assert detail.words[0].offset_start_ms == 120
        assert detail.words[0].offset_end_ms == 640

    def test_task_status_words_explicit_zero_is_kept(self):
        """A leading word legitimately starts at 0 and must not fall through
        to the fallback key."""
        data = {
            "RecTaskId": "task-0",
            "Status": 2,
            "ResultDetail": [
                {
                    "FinalSentence": "你好",
                    "Words": [
                        {"Word": "你", "StartTime": 0, "EndTime": 160},
                        {"Word": "好", "StartTime": 160, "EndTime": 400},
                    ],
                }
            ],
        }
        words = TaskStatus.from_dict(data).result_detail[0].words
        assert [w.offset_start_ms for w in words] == [0, 160]
        assert [w.offset_end_ms for w in words] == [160, 400]
