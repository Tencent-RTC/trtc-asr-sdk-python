"""One-shot sentence recognition client for TRTC-ASR.

Usage::

    credential = Credential(app_id, sdk_app_id, secret_key)
    recognizer = SentenceRecognizer(credential)
    result = recognizer.recognize_data(data, "pcm", "16k_zh_en")
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from trtc_asr.credential import Credential, resolve_http_endpoint
from trtc_asr.errors import (
    ASRError,
    ERR_AUTH_FAILED,
    ERR_CONNECT_FAILED,
    ERR_INVALID_PARAM,
    ERR_READ_FAILED,
    ERR_SERVER_ERROR,
)
from trtc_asr.sdkinfo import sdk_report_query
from trtc_asr.usersig import gen_user_sig

logger = logging.getLogger(__name__)

SENTENCE_ENDPOINT = "https://asr.cloud-rtc.com"

# SourceType constants
SOURCE_TYPE_URL = 0
SOURCE_TYPE_DATA = 1


@dataclass
class SentenceRecognitionRequest:
    """JSON request body for sentence recognition."""

    eng_service_type: str = ""
    source_type: int = SOURCE_TYPE_DATA
    voice_format: str = "pcm"

    # Conditional fields
    url: str = ""
    data: str = ""
    data_len: int = 0

    # Optional fields
    word_info: int = 0
    filter_dirty: int = 0
    filter_modal: int = 0
    filter_punc: int = 0
    convert_num_mode: int = 1
    hotword_id: str = ""
    hotword_list: str = ""
    customization_id: str = ""
    input_sample_rate: int = 0
    language: str = ""

    def to_dict(self) -> dict:
        """Convert to the JSON body dict matching API field names."""
        d: dict = {
            "EngSerViceType": self.eng_service_type,
            "SourceType": self.source_type,
            "VoiceFormat": self.voice_format,
        }

        if self.source_type == SOURCE_TYPE_URL:
            d["Url"] = self.url
        else:
            d["Data"] = self.data
            d["DataLen"] = self.data_len

        # Optional — only include non-default values
        if self.word_info:
            d["WordInfo"] = self.word_info
        if self.filter_dirty:
            d["FilterDirty"] = self.filter_dirty
        if self.filter_modal:
            d["FilterModal"] = self.filter_modal
        if self.filter_punc:
            d["FilterPunc"] = self.filter_punc
        if self.convert_num_mode != 1:
            d["ConvertNumMode"] = self.convert_num_mode
        if self.hotword_id:
            d["HotwordId"] = self.hotword_id
        if self.hotword_list:
            d["HotwordList"] = self.hotword_list
        if self.customization_id:
            d["CustomizationId"] = self.customization_id
        if self.input_sample_rate:
            d["InputSampleRate"] = self.input_sample_rate
        if self.language:
            d["Language"] = self.language

        return d


@dataclass
class SentenceWord:
    """Word-level timing information."""

    word: str = ""
    start_time: int = 0
    end_time: int = 0


@dataclass
class SentenceRecognitionResult:
    """Recognition result from the API."""

    result: str = ""
    audio_duration: int = 0
    word_size: int = 0
    word_list: List[SentenceWord] = field(default_factory=list)
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "SentenceRecognitionResult":
        word_list = [
            SentenceWord(
                word=w.get("Word", ""),
                start_time=w.get("StartTime", 0),
                end_time=w.get("EndTime", 0),
            )
            for w in data.get("WordList", []) or []
        ]
        return cls(
            result=data.get("Result", ""),
            audio_duration=data.get("AudioDuration", 0),
            word_size=data.get("WordSize", 0),
            word_list=word_list,
            request_id=data.get("RequestId", ""),
        )


class SentenceRecognizer:
    """One-shot sentence recognition client using HTTP POST.

    This is a synchronous client (no async). For the real-time streaming
    recognizer, see :class:`SpeechRecognizer`.
    """

    def __init__(self, credential: Credential) -> None:
        self._credential = credential
        self._endpoint = ""
        self._timeout = 30.0  # seconds

    def set_endpoint(self, endpoint: str) -> None:
        """Override the default API endpoint (for testing)."""
        self._endpoint = endpoint

    def set_timeout(self, timeout: float) -> None:
        """Set HTTP request timeout in seconds."""
        self._timeout = timeout

    def recognize(
        self, req: SentenceRecognitionRequest
    ) -> SentenceRecognitionResult:
        """Send a sentence recognition request and return the result."""
        self._validate_request(req)

        request_id = str(uuid.uuid4())

        # Generate UserSig using request_id as user ID
        user_sig = self._credential.user_sig
        if not user_sig:
            try:
                user_sig = gen_user_sig(
                    self._credential.sdk_app_id,
                    self._credential.secret_key,
                    request_id,
                    86400,
                )
            except Exception as e:
                raise ASRError(ERR_AUTH_FAILED, "generate user sig failed: {}".format(e))

        # Build URL with query parameters
        req_url = (
            "{}/v1/SentenceRecognition"
            "?AppId={}"
            "&Secretid={}"
            "&RequestId={}"
            "&Timestamp={}"
            "&{}".format(
                resolve_http_endpoint(self._endpoint, self._credential.site),
                self._credential.app_id,
                self._credential.app_id,  # Secretid uses AppID per protocol
                request_id,
                int(time.time()),
                sdk_report_query(),
            )
        )

        body = json.dumps(req.to_dict()).encode("utf-8")

        http_req = urllib.request.Request(
            req_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-TRTC-SdkAppId": str(self._credential.sdk_app_id),
                "X-TRTC-UserSig": user_sig,
            },
        )

        try:
            with urllib.request.urlopen(http_req, timeout=self._timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                status_code = resp.status
        except urllib.error.HTTPError as e:
            resp_body = e.read().decode("utf-8") if e.fp else ""
            raise ASRError(
                ERR_SERVER_ERROR,
                "http status {}: {}".format(e.code, resp_body),
            )
        except Exception as e:
            raise ASRError(ERR_CONNECT_FAILED, "http request failed: {}".format(e))

        # Parse response
        try:
            resp_data = json.loads(resp_body)
        except (json.JSONDecodeError, ValueError) as e:
            raise ASRError(ERR_READ_FAILED, "unmarshal response failed: {}".format(e))

        response = resp_data.get("Response")
        if response is None:
            raise ASRError(ERR_SERVER_ERROR, "empty response from server")

        # Check for API-level errors
        error = response.get("Error")
        if error:
            raise ASRError(
                ERR_SERVER_ERROR,
                "server error [{}]: {} (RequestId: {})".format(
                    error.get("Code", ""),
                    error.get("Message", ""),
                    response.get("RequestId", ""),
                ),
            )

        return SentenceRecognitionResult.from_dict(response)

    def recognize_data(
        self,
        data: bytes,
        voice_format: str,
        engine_model_type: str,
    ) -> SentenceRecognitionResult:
        """Convenience method: recognize local audio data (auto base64 encoding).

        Args:
            data: Raw audio bytes (max 3MB).
            voice_format: Audio format ("pcm", "wav", "ogg-opus", "mp3", "m4a").
            engine_model_type: Engine model ("16k_zh", "16k_zh_en").
        """
        if not data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(data) > 3 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 3MB limit")

        req = SentenceRecognitionRequest(
            eng_service_type=engine_model_type,
            source_type=SOURCE_TYPE_DATA,
            voice_format=voice_format,
            data=base64.b64encode(data).decode("ascii"),
            data_len=len(data),
        )
        return self.recognize(req)

    def recognize_data_with_options(
        self,
        data: bytes,
        req: SentenceRecognitionRequest,
    ) -> SentenceRecognitionResult:
        """Recognize local audio data with a pre-configured request.

        Data and DataLen fields are set automatically from the raw data.
        """
        if not data:
            raise ASRError(ERR_INVALID_PARAM, "audio data is empty")
        if len(data) > 3 * 1024 * 1024:
            raise ASRError(ERR_INVALID_PARAM, "audio data exceeds 3MB limit")

        req.source_type = SOURCE_TYPE_DATA
        req.data = base64.b64encode(data).decode("ascii")
        req.data_len = len(data)
        return self.recognize(req)

    def recognize_url(
        self,
        audio_url: str,
        voice_format: str,
        engine_model_type: str,
    ) -> SentenceRecognitionResult:
        """Convenience method: recognize audio from a URL."""
        if not audio_url:
            raise ASRError(ERR_INVALID_PARAM, "audio URL is empty")

        req = SentenceRecognitionRequest(
            eng_service_type=engine_model_type,
            source_type=SOURCE_TYPE_URL,
            voice_format=voice_format,
            url=audio_url,
        )
        return self.recognize(req)

    @staticmethod
    def _validate_request(req: SentenceRecognitionRequest) -> None:
        if req is None:
            raise ASRError(ERR_INVALID_PARAM, "request is None")
        if not req.eng_service_type:
            raise ASRError(ERR_INVALID_PARAM, "eng_service_type is required")
        if not req.voice_format:
            raise ASRError(ERR_INVALID_PARAM, "voice_format is required")
        if req.source_type == SOURCE_TYPE_URL and not req.url:
            raise ASRError(ERR_INVALID_PARAM, "url is required when source_type=0")
        if req.source_type == SOURCE_TYPE_DATA and not req.data:
            raise ASRError(ERR_INVALID_PARAM, "data is required when source_type=1")
