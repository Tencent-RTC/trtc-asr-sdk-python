"""URL query parameter building for the ASR WebSocket request."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote

from trtc_asr.sdkinfo import sdk_report_params

# Speaker diarization modes for the speaker_diarization parameter.
SPEAKER_DIARIZATION_OFF = 0
SPEAKER_DIARIZATION_CLUSTER = 1
SPEAKER_DIARIZATION_VOICEPRINT = 3


@dataclass
class SpeakerRole:
    """Temporary voiceprint enrollment entry used with speaker_diarization=3.

    ``role_name`` is echoed back by the server as ``speaker_name`` on the
    matched words / speaker segments.

    The serialized field names intentionally match the server-side contract
    (CamelCase) for both the streaming ``speaker_roles`` query parameter and
    the CreateRecTask ``SpeakerRoles`` body field.
    """

    role_name: str = ""
    audio_url: str = ""

    def to_dict(self) -> dict:
        return {"RoleName": self.role_name, "AudioUrl": self.audio_url}

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerRole":
        return cls(
            role_name=data.get("RoleName", ""),
            audio_url=data.get("AudioUrl", ""),
        )


@dataclass
class SignatureParams:
    """Holds URL query parameters for the ASR WebSocket request.

    The ``secretid`` URL parameter is required by the protocol but internally
    populated with AppID — users do not need to provide a separate SecretID.
    The ``signature`` parameter is set to the UserSig value per protocol spec.

    Authentication identity travels in the URL instead of HTTP headers: the
    gateway accepts the ``sdkappid`` / ``usersig`` query parameters, and
    browsers cannot attach custom headers to a native WebSocket handshake.
    """

    app_id: int
    engine_model_type: str
    voice_id: str
    timestamp: int = field(default_factory=lambda: int(time.time()))
    expired: int = 0
    nonce: int = field(default_factory=lambda: random.randint(1, 9999999))
    voice_format: int = 1  # PCM
    need_vad: int = 1
    convert_num_mode: int = 1

    # SdkAppID is the TRTC application ID, sent as the "sdkappid" query
    # parameter. 0 means not configured.
    sdk_app_id: int = 0

    # Optional parameters
    hotword_id: str = ""
    hotword_list: str = ""  # temporary inline hotwords: "word|weight,word|weight"
    customization_id: str = ""
    replace_text_id: str = ""  # replacement word table ID
    filter_dirty: int = 0
    filter_modal: int = 0
    filter_punc: int = 0
    word_info: int = 0
    vad_silence_time: int = 0
    max_speak_time: int = 0
    input_sample_rate: int = 0  # 8000: feed 8kHz PCM to a 16k engine (upsampled server-side)
    language: str = ""  # bigmodel engine language hint (e.g. "ms", "zh", "auto")

    # filter_empty_result controls empty-result callbacks: 0=deliver empty
    # results, 1=skip them (server default). None leaves the parameter out.
    filter_empty_result: Optional[int] = None

    # vad_level selects the VAD profile: 0=high recall, 1=far-field filtering
    # (server default). None leaves the parameter out, so an explicit 0 is
    # distinguishable from "not configured".
    vad_level: Optional[int] = None

    # noise_threshold fine-tunes VAD noise suppression, range [0, 4]. When set
    # it overrides the profile selected by vad_level. None leaves the
    # parameter out (0 is a valid, meaningful value).
    noise_threshold: Optional[float] = None

    # speaker_diarization enables speaker diarization: 0=off (default),
    # 1=anonymous clustering, 3=voiceprint role authentication.
    speaker_diarization: int = 0

    # speaker_number hints the expected speaker count; 0=auto detection
    # (default). Sent whenever diarization is enabled: the server feeds it
    # into online clustering for both modes.
    speaker_number: int = 0

    # speaker_roles carries temporary voiceprint enrollment audio, serialized
    # into the speaker_roles JSON array. Only sent when speaker_diarization
    # is 3.
    speaker_roles: List[SpeakerRole] = field(default_factory=list)

    # voiceprint_ids lists pre-registered voiceprint IDs, serialized into the
    # voiceprintids JSON array. Only sent when speaker_diarization is 3.
    voiceprint_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expired == 0:
            self.expired = self.timestamp + 86400

    def build_query_string(self) -> str:
        """Build URL query string without signature."""
        params = self._to_map()
        return _encode_params(params)

    def build_query_string_with_signature(self, user_sig: str) -> str:
        """Build URL query string with signature set to the given UserSig.

        Per protocol: the ``signature`` value equals the UserSig. The same
        value is also sent as the ``usersig`` query parameter, which the
        gateway reads (e.g. browser WebSocket clients that cannot attach
        custom headers).
        """
        params = self._to_map()
        params["signature"] = user_sig
        params["usersig"] = user_sig
        return _encode_params(params)

    def _to_map(self) -> dict:
        # "secretid" is required by protocol; internally use AppID as its value.
        m: dict = {
            "secretid": str(self.app_id),
            "timestamp": str(self.timestamp),
            "expired": str(self.expired),
            "nonce": str(self.nonce),
            "engine_model_type": self.engine_model_type,
            "voice_id": self.voice_id,
            "voice_format": str(self.voice_format),
            "needvad": str(self.need_vad),
        }
        # SDK self-identification for server-side diagnostics. Not part of the
        # signature (the signature is the UserSig), so it is safe to append.
        m.update(sdk_report_params())
        if self.sdk_app_id > 0:
            m["sdkappid"] = str(self.sdk_app_id)

        if self.hotword_id:
            m["hotword_id"] = self.hotword_id
        if self.hotword_list:
            m["hotword_list"] = self.hotword_list
        if self.customization_id:
            m["customization_id"] = self.customization_id
        if self.replace_text_id:
            m["replace_text_id"] = self.replace_text_id
        if self.filter_dirty:
            m["filter_dirty"] = str(self.filter_dirty)
        if self.filter_modal:
            m["filter_modal"] = str(self.filter_modal)
        if self.filter_punc:
            m["filter_punc"] = str(self.filter_punc)
        if self.filter_empty_result is not None:
            m["filter_empty_result"] = str(self.filter_empty_result)
        if self.convert_num_mode:
            m["convert_num_mode"] = str(self.convert_num_mode)
        if self.word_info:
            m["word_info"] = str(self.word_info)
        if self.vad_silence_time:
            m["vad_silence_time"] = str(self.vad_silence_time)
        if self.max_speak_time:
            m["max_speak_time"] = str(self.max_speak_time)
        if self.input_sample_rate:
            m["input_sample_rate"] = str(self.input_sample_rate)
        # vad_level and noise_threshold are tri-state: an explicit 0 differs
        # from "not configured" (the server defaults vad_level to 1), so they
        # are only emitted when the caller set them.
        if self.vad_level is not None:
            m["vad_level"] = str(self.vad_level)
        if self.noise_threshold is not None:
            m["noise_threshold"] = f"{self.noise_threshold:.3f}"
        if self.speaker_diarization != 0:
            m["speaker_diarization"] = str(self.speaker_diarization)
            if self.speaker_number != 0:
                m["speaker_number"] = str(self.speaker_number)
        # speaker_roles / voiceprintids only apply to the voiceprint role
        # authentication mode.
        if self.speaker_diarization == SPEAKER_DIARIZATION_VOICEPRINT:
            if self.speaker_roles:
                m["speaker_roles"] = json.dumps(
                    [r.to_dict() for r in self.speaker_roles],
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            if self.voiceprint_ids:
                m["voiceprintids"] = json.dumps(
                    self.voiceprint_ids, separators=(",", ":"), ensure_ascii=False
                )
        if self.language:
            m["language"] = self.language

        return m


def _encode_params(params: dict) -> str:
    """Encode parameters into a sorted URL query string."""
    return "&".join(
        f"{k}={quote(str(v), safe='')}" for k, v in sorted(params.items())
    )
