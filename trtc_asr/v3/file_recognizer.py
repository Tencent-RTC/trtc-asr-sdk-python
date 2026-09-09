"""Async audio file recognition client for the TRTC-ASR v3 protocol.

Unlike :class:`v3.SentenceRecognizer` (one-shot, ≤60s), FileRecognizer
handles longer audio via an async workflow: submit a task
(POST /v3/create_transcription), then poll for results
(POST /v3/describe_transcription).

Usage::

    credential = v3.new_credential(sdk_app_id, secret_key)
    recognizer = v3.FileRecognizer(credential)
    task_id = recognizer.create_task_from_data(data, "16k_zh_en")
    status = recognizer.wait_for_result(task_id)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from trtc_asr.credential import Credential, resolve_http_endpoint
from trtc_asr.errors import (
    ASRError,
    ERR_CONNECT_FAILED,
    ERR_INVALID_PARAM,
    ERR_SERVER_ERROR,
    ERR_TIMEOUT,
)
from trtc_asr.params import validate_speaker_diarization, validate_vad_tuning
from trtc_asr.sdkinfo import sdk_report_params
from trtc_asr.v3.wire import (
    SOURCE_TYPE_DATA,
    SOURCE_TYPE_URL,
    TASK_STATUS_FAILED,
    TASK_STATUS_SUCCESS,
    SPEAKER_DIARIZATION_VOICEPRINT,
    AudioURLItem,
    Context,
    SpeakerRole,
    Word,
    decode_flat_response,
    offline_envelope,
    server_error,
    validate_audio_urls,
)

logger = logging.getLogger(__name__)

FILE_ENDPOINT = "https://asr.cloud-rtc.com"

_CREATE_PATH = "/v3/create_transcription"
_DESCRIBE_PATH = "/v3/describe_transcription"


@dataclass
class CreateTranscriptionRequest:
    """The params block of /v3/create_transcription. Field names map
    one-to-one to the snake_case wire names."""

    engine_model_type: str = ""
    channel_num: int = 1
    res_text_format: int = 1
    source_type: int = SOURCE_TYPE_DATA

    url: str = ""
    data: str = ""
    data_len: int = 0

    # audio_urls lists the pieces of a distributed recording task
    # (distributed speaker feature extraction). When non-empty, the task
    # runs the distributed path and source_type/url/data must be left at
    # their zero values (source_type=0, url/data empty) — the server rejects
    # any combination of audio_urls with a single-audio source.
    audio_urls: List[AudioURLItem] = field(default_factory=list)

    callback_url: str = ""

    speaker_diarization: int = 0
    speaker_number: int = 0
    voiceprint_ids: List[str] = field(default_factory=list)
    speaker_roles: List[SpeakerRole] = field(default_factory=list)

    hotword_id: str = ""
    customization_id: str = ""
    hotword_list: str = ""
    keyword_lib_id_list: List[str] = field(default_factory=list)
    replace_text_id: str = ""

    convert_num_mode: int = 0
    filter_dirty: int = 0
    filter_punc: int = 0
    filter_modal: int = 0

    sentence_max_length: int = 0
    extra: str = ""

    vad_silence_ms: int = 0
    vad_level: Optional[int] = None
    noise_threshold: Optional[float] = None

    language: str = ""

    context: Optional[Context] = None

    def to_wire(self) -> dict:
        d: dict = {
            "engine_model_type": self.engine_model_type,
            "channel_num": self.channel_num,
            "res_text_format": self.res_text_format,
            "source_type": self.source_type,
        }
        if self.source_type == SOURCE_TYPE_URL:
            d["url"] = self.url
        else:
            d["data"] = self.data
            d["data_len"] = self.data_len
        if self.audio_urls:
            d["audio_urls"] = [item.to_wire() for item in self.audio_urls]
        if self.callback_url:
            d["callback_url"] = self.callback_url
        if self.speaker_diarization:
            d["speaker_diarization"] = self.speaker_diarization
            if self.speaker_number:
                d["speaker_number"] = self.speaker_number
        if self.speaker_diarization == SPEAKER_DIARIZATION_VOICEPRINT:
            if self.voiceprint_ids:
                d["voiceprint_ids"] = list(self.voiceprint_ids)
            if self.speaker_roles:
                d["speaker_roles"] = [r.to_wire() for r in self.speaker_roles]
        if self.hotword_id:
            d["hotword_id"] = self.hotword_id
        if self.customization_id:
            d["customization_id"] = self.customization_id
        if self.hotword_list:
            d["hotword_list"] = self.hotword_list
        if self.keyword_lib_id_list:
            d["keyword_lib_id_list"] = list(self.keyword_lib_id_list)
        if self.replace_text_id:
            d["replace_text_id"] = self.replace_text_id
        if self.convert_num_mode:
            d["convert_num_mode"] = self.convert_num_mode
        if self.filter_dirty:
            d["filter_dirty"] = self.filter_dirty
        if self.filter_punc:
            d["filter_punc"] = self.filter_punc
        if self.filter_modal:
            d["filter_modal"] = self.filter_modal
        if self.sentence_max_length:
            d["sentence_max_length"] = self.sentence_max_length
        if self.extra:
            d["extra"] = self.extra
        if self.vad_silence_ms:
            d["vad_silence_ms"] = self.vad_silence_ms
        # Explicit zeros must survive: None means "not configured" while 0 is
        # a valid, meaningful value for both.
        if self.vad_level is not None:
            d["vad_level"] = self.vad_level
        if self.noise_threshold is not None:
            d["noise_threshold"] = self.noise_threshold
        if self.language:
            d["language"] = self.language
        if self.context is not None:
            wire = self.context.to_wire()
            if wire:
                d["context"] = wire
        return d


@dataclass
class SentenceDetail:
    """One sentence of a describe_transcription result."""

    final_sentence: str = ""
    slice_sentence: str = ""
    written_text: str = ""
    start_ms: int = 0
    end_ms: int = 0
    words_num: int = 0
    words: List[Word] = field(default_factory=list)
    speech_speed: float = 0.0
    speaker_id: int = 0
    channel_id: int = 0
    speaker_role_name: str = ""
    silence_time: int = 0
    language: str = ""
    language_b47: str = ""


@dataclass
class TranscriptionStatus:
    """The flat /v3/describe_transcription response."""

    code: int = 0
    message: str = ""
    request_id: str = ""
    transcription_id: str = ""
    status: int = 0
    status_str: str = ""
    progress: int = 0
    audio_duration: float = 0.0
    result: str = ""
    result_detail: List[SentenceDetail] = field(default_factory=list)
    error_msg: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionStatus":
        result_detail = []
        for sd in data.get("result_detail") or []:
            result_detail.append(
                SentenceDetail(
                    final_sentence=sd.get("final_sentence", ""),
                    slice_sentence=sd.get("slice_sentence", ""),
                    written_text=sd.get("written_text", ""),
                    start_ms=sd.get("start_ms", 0),
                    end_ms=sd.get("end_ms", 0),
                    words_num=sd.get("words_num", 0),
                    words=[Word.from_dict(w) for w in sd.get("words") or []],
                    speech_speed=sd.get("speech_speed", 0.0),
                    speaker_id=sd.get("speaker_id", 0),
                    channel_id=sd.get("channel_id", 0),
                    speaker_role_name=sd.get("speaker_role_name", ""),
                    silence_time=sd.get("silence_time", 0),
                    language=sd.get("language", ""),
                    language_b47=sd.get("language_b47", ""),
                )
            )
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            request_id=data.get("request_id", ""),
            transcription_id=data.get("transcription_id", ""),
            status=data.get("status", 0),
            status_str=data.get("status_str", ""),
            progress=data.get("progress", 0),
            audio_duration=data.get("audio_duration", 0.0),
            result=data.get("result", ""),
            result_detail=result_detail,
            error_msg=data.get("error_msg", ""),
        )


class FileRecognizer:
    """Async audio file recognition client (/v3/create_transcription +
    /v3/describe_transcription).

    The task ID (``transcription_id``) is valid for 24 hours and belongs to
    the v3 task space: a v1 ``RecTaskId`` cannot be queried through this
    client and vice versa.
    """

    def __init__(self, credential: Credential) -> None:
        self._credential = credential
        self._endpoint = ""
        self._timeout = 60.0

    def set_endpoint(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def set_timeout(self, timeout: float) -> None:
        self._timeout = timeout

    def create_task(self, req: CreateTranscriptionRequest) -> str:
        """Submit a file recognition task and return the transcription ID."""
        self._validate_create_request(req)

        params = req.to_wire()
        params["sdk_info"] = sdk_report_params()
        resp_data = self._post(_CREATE_PATH, params)

        resp_code = resp_data.get("code", 0)
        if resp_code != 0:
            raise server_error(
                resp_code, resp_data.get("message", ""), resp_data.get("request_id", "")
            )

        transcription_id = resp_data.get("transcription_id", "")
        if not transcription_id:
            raise ASRError(ERR_SERVER_ERROR, "empty transcription_id in response")
        return transcription_id

    def create_task_from_data(self, data: bytes, engine_model_type: str) -> str:
        """Submit local audio data for recognition (auto base64). Max 5MB."""
        if not data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(data) > 5 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 5MB limit")

        req = CreateTranscriptionRequest(
            engine_model_type=engine_model_type,
            channel_num=1,
            res_text_format=1,
            source_type=SOURCE_TYPE_DATA,
            data=base64.b64encode(data).decode("ascii"),
            data_len=len(data),
        )
        return self.create_task(req)

    def create_task_from_url(self, audio_url: str, engine_model_type: str) -> str:
        """Submit an audio URL for recognition. Audio ≤12h, ≤1GB."""
        if not audio_url:
            raise ASRError(ERR_INVALID_PARAM, "audio URL is empty")

        req = CreateTranscriptionRequest(
            engine_model_type=engine_model_type,
            channel_num=1,
            res_text_format=1,
            source_type=SOURCE_TYPE_URL,
            url=audio_url,
        )
        return self.create_task(req)

    def create_task_from_data_with_options(
        self, raw_data: bytes, req: CreateTranscriptionRequest
    ) -> str:
        """Submit local audio data with a pre-configured request (the
        request is mutated in place)."""
        if req is None:
            raise ASRError(ERR_INVALID_PARAM, "request is None")
        if not raw_data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(raw_data) > 5 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 5MB limit")

        req.source_type = SOURCE_TYPE_DATA
        req.data = base64.b64encode(raw_data).decode("ascii")
        req.data_len = len(raw_data)
        return self.create_task(req)

    def describe_task(self, transcription_id: str) -> TranscriptionStatus:
        """Query the status of a file recognition task."""
        if not transcription_id:
            raise ASRError(ERR_INVALID_PARAM, "transcription_id is empty")

        params = {
            "transcription_id": transcription_id,
            "sdk_info": sdk_report_params(),
        }
        resp_data = self._post(_DESCRIBE_PATH, params)

        resp_code = resp_data.get("code", 0)
        if resp_code != 0:
            raise server_error(
                resp_code, resp_data.get("message", ""), resp_data.get("request_id", "")
            )

        return TranscriptionStatus.from_dict(resp_data)

    def wait_for_result(self, transcription_id: str) -> TranscriptionStatus:
        """Poll for results with default interval (1s) and timeout (10min)."""
        return self.wait_for_result_with_interval(transcription_id, 1.0, 600.0)

    def wait_for_result_with_interval(
        self, transcription_id: str, interval: float, timeout: float
    ) -> TranscriptionStatus:
        """Poll for results with custom interval and timeout (in seconds)."""
        deadline = time.monotonic() + timeout

        while True:
            status = self.describe_task(transcription_id)

            if status.status == TASK_STATUS_SUCCESS:
                return status
            if status.status == TASK_STATUS_FAILED:
                raise ASRError(
                    ERR_SERVER_ERROR,
                    "task failed: {} (transcription_id: {})".format(
                        status.error_msg, status.transcription_id
                    ),
                )

            if time.monotonic() > deadline:
                raise ASRError(
                    ERR_TIMEOUT,
                    "task not completed within {}s (transcription_id: {}, Status: {})".format(
                        timeout, transcription_id, status.status_str
                    ),
                )

            time.sleep(interval)

    def _post(self, path: str, params: dict) -> dict:
        request_id = str(uuid.uuid4())
        body = offline_envelope(self._credential, request_id, params)

        req_url = "{}{}".format(
            resolve_http_endpoint(self._endpoint, self._credential.site), path
        )
        http_req = urllib.request.Request(
            req_url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        try:
            with urllib.request.urlopen(http_req, timeout=self._timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as e:
            resp_body = e.read().decode("utf-8") if e.fp else ""
            status = e.code
        except Exception as e:
            raise ASRError(ERR_CONNECT_FAILED, "http request failed: {}".format(e))

        return decode_flat_response(resp_body, status)

    @staticmethod
    def _validate_create_request(req: CreateTranscriptionRequest) -> None:
        if req is None:
            raise ASRError(ERR_INVALID_PARAM, "request is None")
        if not req.engine_model_type:
            raise ASRError(ERR_INVALID_PARAM, "engine_model_type is required")
        if req.channel_num not in (1, 2):
            raise ASRError(ERR_INVALID_PARAM, "channel_num must be 1 or 2")
        if req.res_text_format not in (0, 1, 2, 3):
            raise ASRError(ERR_INVALID_PARAM, "res_text_format must be one of [0, 1, 2, 3], got {}".format(req.res_text_format))
        if req.channel_num == 2 and req.speaker_diarization:
            raise ASRError(
                ERR_INVALID_PARAM,
                "speaker_diarization is not supported for stereo (channel_num=2); sentences carry channel_id instead",
            )
        if req.audio_urls:
            # Distributed path: the server requires source_type=0 with
            # url/data empty (rectask_cluster.go
            # validateDistributedAudioUrlsRequest).
            validate_audio_urls(req.source_type, req.url, req.data, req.audio_urls)
        else:
            if req.source_type == SOURCE_TYPE_URL and not req.url:
                raise ASRError(ERR_INVALID_PARAM, "url is required when source_type=0")
            if req.source_type == SOURCE_TYPE_DATA and not req.data:
                raise ASRError(ERR_INVALID_PARAM, "data is required when source_type=1")
        validate_speaker_diarization(
            req.speaker_diarization,
            req.speaker_number,
            req.speaker_roles,
            req.voiceprint_ids,
        )
        # vad_level uses the plain-int wire field: 0 is both the default and
        # the high-recall value, so only 1 needs an explicit check.
        vad_level = req.vad_level if req.vad_level is not None else 0
        validate_vad_tuning(vad_level if vad_level else None, req.noise_threshold)
