"""Shared wire types and helpers for the TRTC-ASR v3 protocol.

The v3 protocol restructures the wire format around two separated blocks: an
"auth" block (sdkappid / usersig / request_id) consumed by the gateway's
authentication layer, and a "params" block (engine, VAD, hotwords, filters,
...) consumed by the recognition layer. All field names are snake_case and
responses are flat (no Response envelope) with numeric codes.

v3 uses SdkAppID as the only customer dimension; the Tencent Cloud AppID is
not needed (see :func:`new_credential`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trtc_asr.credential import Credential
from trtc_asr.errors import ASRError, ERR_AUTH_FAILED, ERR_INVALID_PARAM, ERR_SERVER_ERROR
from trtc_asr.usersig import gen_user_sig

# Wire size limits, mirroring the server side. Checked locally so an
# oversized frame fails before touching the network.
START_FRAME_MAX_BYTES = 64 * 1024
STREAM_FRAME_MAX_BYTES = 256 * 1024

# ack_timeout caps how long start() waits for the server's start-frame
# acknowledgement (in seconds).
ACK_TIMEOUT = 5.0

# SourceType for the HTTP interfaces.
SOURCE_TYPE_URL = 0
SOURCE_TYPE_DATA = 1

# Task status values returned by describe_transcription.
TASK_STATUS_WAITING = 0
TASK_STATUS_RUNNING = 1
TASK_STATUS_SUCCESS = 2
TASK_STATUS_FAILED = 3

# Speaker diarization modes.
SPEAKER_DIARIZATION_OFF = 0
SPEAKER_DIARIZATION_CLUSTER = 1
SPEAKER_DIARIZATION_VOICEPRINT = 3


def new_credential(sdk_app_id: int, secret_key: str) -> Credential:
    """Create a credential for the v3 API.

    v3 uses SdkAppID as the only customer dimension and does not need the
    Tencent Cloud AppID.
    """
    return Credential(0, sdk_app_id, secret_key)


@dataclass
class ContextKV:
    """A domain key-value pair of Context.general."""

    key: str = ""
    value: str = ""

    def to_wire(self) -> Dict[str, Any]:
        return {"key": self.key, "value": self.value}


@dataclass
class Context:
    """Recognition context, aligned with Soniox / Volcengine semantics.

    LLM-class engines can consume all three parts; traditional engines
    degrade terms to hotwords and ignore text/general.
    """

    text: str = ""
    terms: List[str] = field(default_factory=list)
    general: List[ContextKV] = field(default_factory=list)

    def to_wire(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.text:
            d["text"] = self.text
        if self.terms:
            d["terms"] = list(self.terms)
        if self.general:
            d["general"] = [kv.to_wire() for kv in self.general]
        return d


@dataclass
class SpeakerRole:
    """A temporary voiceprint enrollment entry used with
    speaker_diarization=3. role_name is echoed back as speaker_name /
    speaker_role_name on matched results.
    """

    role_name: str = ""
    audio_url: str = ""

    def to_wire(self) -> Dict[str, Any]:
        return {"role_name": self.role_name, "audio_url": self.audio_url}


@dataclass
class AudioURLItem:
    """One audio piece of a distributed recording task."""

    index: int = 0
    url: str = ""
    label: str = ""

    def to_wire(self) -> Dict[str, Any]:
        d = {"index": self.index, "url": self.url}
        if self.label:
            d["label"] = self.label
        return d


@dataclass
class Word:
    """A word-level timing entry (shared by transcribe and describe
    responses)."""

    word: str = ""
    start_time: int = 0
    end_time: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Word":
        return cls(
            word=data.get("word", ""),
            start_time=data.get("start_time", 0),
            end_time=data.get("end_time", 0),
        )


def build_auth_block(credential: Credential, request_id: str = "") -> Dict[str, Any]:
    """Build the v3 auth block. request_id is the UserSig identifier for the
    offline interfaces; the server binds the signature to it.
    """
    user_sig = credential.user_sig
    if not user_sig:
        try:
            user_sig = gen_user_sig(
                credential.sdk_app_id, credential.secret_key, request_id, 86400
            )
        except Exception as exc:
            raise ASRError(ERR_AUTH_FAILED, "generate user sig failed: {}".format(exc))
    auth = {"sdkappid": str(credential.sdk_app_id), "usersig": user_sig}
    if request_id:
        auth["request_id"] = request_id
    return auth


def offline_envelope(credential: Credential, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap params into the {"auth": ..., "params": ...} offline body."""
    return {
        "auth": build_auth_block(credential, request_id),
        "params": params,
    }


def server_error(code: int, message: str, request_id: str = "") -> ASRError:
    """Convert a v3 flat error into an ASRError carrying the server code.

    Server codes (4xxx client errors / 5xxx server errors) are disjoint from
    the SDK-local 10xx codes, so callers can distinguish them.
    """
    if request_id:
        return ASRError(code, "{} (request_id: {})".format(message, request_id))
    return ASRError(code, message)


def decode_flat_response(resp_body: str, status: int) -> Dict[str, Any]:
    """Unmarshal a v3 flat response body.

    Two failure shapes are rejected here:

    - the body is not a JSON object (cannot be a v3 response);
    - the HTTP status is not 2xx while the body carries no numeric code — a
      gateway/LB JSON error page would otherwise decode to code==0 and be
      mistaken for success.

    A legitimate v3 endpoint always sends 2xx for code==0, so the second
    guard never misfires on a real response.
    """
    try:
        data = json.loads(resp_body)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ASRError(
            ERR_SERVER_ERROR,
            "invalid response (http {}): {}".format(status, exc),
        )
    if not isinstance(data, dict):
        raise ASRError(
            ERR_SERVER_ERROR,
            "invalid response (http {}): not a JSON object".format(status),
        )
    if (status < 200 or status > 299) and data.get("code", 0) == 0:
        preview = resp_body[:256] + ("..." if len(resp_body) > 256 else "")
        raise ASRError(
            ERR_SERVER_ERROR,
            "http {} with non-v3 response body: {}".format(status, preview),
        )
    return data


def validate_audio_urls(source_type: int, url: str, data: str, audio_urls: List[AudioURLItem]) -> None:
    """Check the distributed-recording invariants the server enforces
    (asr-local-manager rectask_cluster.go validateDistributedAudioUrlsRequest):
    audio_urls requires source_type=0 with url/data left empty, and each item
    needs a unique non-negative index and an absolute http(s) URL.
    """
    from urllib.parse import urlparse

    if source_type != SOURCE_TYPE_URL or url or data:
        raise ASRError(
            ERR_INVALID_PARAM,
            "AudioURLs cannot be used together with non-zero source_type, url or data",
        )
    seen = set()
    for i, item in enumerate(audio_urls):
        if item.index < 0:
            raise ASRError(ERR_INVALID_PARAM, "AudioURLs[{}].Index must be non-negative".format(i))
        if item.index in seen:
            raise ASRError(ERR_INVALID_PARAM, "AudioURLs index duplicated: {}".format(item.index))
        seen.add(item.index)
        raw = (item.url or "").strip()
        if not raw:
            raise ASRError(ERR_INVALID_PARAM, "AudioURLs[{}].URL is required".format(i))
        try:
            parsed = urlparse(raw)
        except Exception as exc:
            raise ASRError(ERR_INVALID_PARAM, "AudioURLs[{}].URL is not a valid URL: {}".format(i, exc))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ASRError(
                ERR_INVALID_PARAM, "AudioURLs[{}].URL must be a valid http/https URL".format(i)
            )
