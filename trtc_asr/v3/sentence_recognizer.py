"""One-shot sentence recognition client for the TRTC-ASR v3 protocol.

POST /v3/transcribe with a {"auth": ..., "params": ...} body; request and
response are flat snake_case structures with numeric codes.

Usage::

    credential = v3.new_credential(sdk_app_id, secret_key)
    recognizer = v3.SentenceRecognizer(credential)
    result = recognizer.recognize_data(data, "pcm", "16k_zh_en")
"""

from __future__ import annotations

import base64
import json
import logging
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
    ERR_READ_FAILED,
    ERR_SERVER_ERROR,
)
from trtc_asr.sdkinfo import sdk_report_params
from trtc_asr.v3.wire import (
    SOURCE_TYPE_DATA,
    SOURCE_TYPE_URL,
    Context,
    Word,
    decode_flat_response,
    offline_envelope,
    server_error,
)

logger = logging.getLogger(__name__)

SENTENCE_ENDPOINT = "https://asr.cloud-rtc.com"

_OFFLINE_PATH = "/v3/transcribe"


@dataclass
class TranscribeRequest:
    """The params block of /v3/transcribe. Field names map one-to-one to the
    snake_case wire names."""

    engine_model_type: str = ""
    source_type: int = SOURCE_TYPE_DATA
    voice_format: str = "pcm"

    url: str = ""
    data: str = ""
    data_len: int = 0

    word_info: int = 0
    filter_dirty: int = 0
    filter_modal: int = 0
    filter_punc: int = 0
    convert_num_mode: int = 0
    hotword_id: str = ""
    customization_id: str = ""
    hotword_list: str = ""
    input_sample_rate: int = 0

    # needvad / vad_silence_time: None leaves the server default; an explicit
    # 0/1 is honored (pointer semantics).
    needvad: Optional[int] = None
    vad_silence_time: Optional[int] = None

    language: str = ""

    speaker_diarization: int = 0
    speaker_number: int = 0

    context: Optional[Context] = None

    def to_wire(self) -> dict:
        d: dict = {
            "engine_model_type": self.engine_model_type,
            "source_type": self.source_type,
            "voice_format": self.voice_format,
        }
        if self.source_type == SOURCE_TYPE_URL:
            d["url"] = self.url
        else:
            d["data"] = self.data
            d["data_len"] = self.data_len
        if self.word_info:
            d["word_info"] = self.word_info
        if self.filter_dirty:
            d["filter_dirty"] = self.filter_dirty
        if self.filter_modal:
            d["filter_modal"] = self.filter_modal
        if self.filter_punc:
            d["filter_punc"] = self.filter_punc
        if self.convert_num_mode:
            d["convert_num_mode"] = self.convert_num_mode
        if self.hotword_id:
            d["hotword_id"] = self.hotword_id
        if self.customization_id:
            d["customization_id"] = self.customization_id
        if self.hotword_list:
            d["hotword_list"] = self.hotword_list
        if self.input_sample_rate:
            d["input_sample_rate"] = self.input_sample_rate
        if self.needvad is not None:
            d["needvad"] = self.needvad
        if self.vad_silence_time is not None:
            d["vad_silence_time"] = self.vad_silence_time
        if self.language:
            d["language"] = self.language
        if self.speaker_diarization:
            d["speaker_diarization"] = self.speaker_diarization
            if self.speaker_number:
                d["speaker_number"] = self.speaker_number
        if self.context is not None:
            wire = self.context.to_wire()
            if wire:
                d["context"] = wire
        return d


@dataclass
class TranscribeResponse:
    """The flat /v3/transcribe response."""

    code: int = 0
    message: str = ""
    request_id: str = ""
    result: str = ""
    audio_duration: int = 0
    language: str = ""
    language_b47: str = ""
    word_size: int = 0
    word_list: List[Word] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscribeResponse":
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            request_id=data.get("request_id", ""),
            result=data.get("result", ""),
            audio_duration=data.get("audio_duration", 0),
            language=data.get("language", ""),
            language_b47=data.get("language_b47", ""),
            word_size=data.get("word_size", 0),
            word_list=[Word.from_dict(w) for w in data.get("word_list") or []],
        )


class SentenceRecognizer:
    """One-shot sentence recognition client (POST /v3/transcribe).

    A non-zero server code is raised as an :class:`ASRError` carrying that
    code. Note that v3 answers auth failures with HTTP 200 + code 4002, so
    the body code — not the HTTP status — is authoritative (handled here).
    """

    def __init__(self, credential: Credential) -> None:
        self._credential = credential
        self._endpoint = ""
        self._timeout = 30.0

    def set_endpoint(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def set_timeout(self, timeout: float) -> None:
        self._timeout = timeout

    def recognize(self, req: TranscribeRequest) -> TranscribeResponse:
        """Send a sentence recognition request and return the result."""
        self._validate_request(req)

        request_id = str(uuid.uuid4())
        params = req.to_wire()
        params["sdk_info"] = sdk_report_params()
        body = offline_envelope(self._credential, request_id, params)

        resp_data = self._post(_OFFLINE_PATH, body)

        resp = TranscribeResponse.from_dict(resp_data)
        if resp.code != 0:
            raise server_error(resp.code, resp.message, resp.request_id)
        return resp

    def recognize_data(
        self, data: bytes, voice_format: str, engine_model_type: str
    ) -> TranscribeResponse:
        """Recognize local audio data (auto base64). Max 3MB / 60s."""
        if not data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(data) > 3 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 3MB limit")

        req = TranscribeRequest(
            engine_model_type=engine_model_type,
            source_type=SOURCE_TYPE_DATA,
            voice_format=voice_format,
            data=base64.b64encode(data).decode("ascii"),
            data_len=len(data),
        )
        return self.recognize(req)

    def recognize_data_with_options(
        self, data: bytes, req: TranscribeRequest
    ) -> TranscribeResponse:
        """Recognize local audio data with a pre-configured request (the
        request is mutated in place)."""
        if req is None:
            raise ASRError(ERR_INVALID_PARAM, "request is None")
        if not data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(data) > 3 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 3MB limit")

        req.source_type = SOURCE_TYPE_DATA
        req.data = base64.b64encode(data).decode("ascii")
        req.data_len = len(data)
        return self.recognize(req)

    def recognize_url(
        self, audio_url: str, voice_format: str, engine_model_type: str
    ) -> TranscribeResponse:
        """Recognize audio from a URL."""
        if not audio_url:
            raise ASRError(ERR_INVALID_PARAM, "audio URL is empty")

        req = TranscribeRequest(
            engine_model_type=engine_model_type,
            source_type=SOURCE_TYPE_URL,
            voice_format=voice_format,
            url=audio_url,
        )
        return self.recognize(req)

    @staticmethod
    def _validate_request(req: TranscribeRequest) -> None:
        if req is None:
            raise ASRError(ERR_INVALID_PARAM, "request is None")
        if not req.engine_model_type:
            raise ASRError(ERR_INVALID_PARAM, "engine_model_type is required")
        if not req.voice_format:
            raise ASRError(ERR_INVALID_PARAM, "voice_format is required")
        if req.source_type == SOURCE_TYPE_URL and not req.url:
            raise ASRError(ERR_INVALID_PARAM, "url is required when source_type=0")
        if req.source_type == SOURCE_TYPE_DATA and not req.data:
            raise ASRError(ERR_INVALID_PARAM, "data is required when source_type=1")
        if req.speaker_diarization or req.speaker_number:
            from trtc_asr.params import validate_speaker_diarization

            validate_speaker_diarization(req.speaker_diarization, req.speaker_number, [], [])
        if req.needvad is not None:
            validate_enum("needvad", req.needvad, (0, 1))
        # 8000 is the only supported override; 0 means "use the engine rate".
        validate_enum("input_sample_rate", req.input_sample_rate, (0, 8000))

    def _post(self, path: str, body: dict) -> dict:
        """POST the {"auth","params"} envelope and decode the flat response."""
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


def validate_enum(name: str, value: int, allowed) -> None:
    if value not in allowed:
        raise ASRError(
            ERR_INVALID_PARAM,
            "{} must be one of {}, got {}".format(name, list(allowed), value),
        )
