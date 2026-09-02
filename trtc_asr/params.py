"""Shared parameter validation for the recognizers.

The service validates every parameter as well, but rejecting an obviously
invalid value locally turns a remote 4001 ("参数不合法") into an immediate,
descriptive error and avoids burning a connection or a task quota.
"""

from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

from trtc_asr.errors import ASRError, ERR_INVALID_PARAM
from trtc_asr.signature import (
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_VOICEPRINT,
    SpeakerRole,
)

# Server-side accepted ranges, kept in one place so streaming and file
# recognition validate identically.
MIN_NOISE_THRESHOLD = 0.0
MAX_NOISE_THRESHOLD = 4.0

# A "%" not followed by two hex digits is an invalid percent-escape, which
# Go's url.ParseRequestURI rejects as bad syntax.
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def validate_speaker_diarization(
    mode: int,
    speaker_number: int,
    roles: List[SpeakerRole],
    voiceprint_ids: List[str],
) -> None:
    """Check the diarization mode and its enrollment input.

    roles/voiceprint_ids are only meaningful with mode 3, but supplying
    them for another mode is a caller mistake worth surfacing.
    """
    if mode not in (SPEAKER_DIARIZATION_OFF, 1, SPEAKER_DIARIZATION_VOICEPRINT):
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerDiarization must be 0 (off), 1 (cluster) or 3 (voiceprint), got {}".format(mode),
        )

    if speaker_number < 0:
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerNumber must be >= 0 (0 = auto detection), got {}".format(speaker_number),
        )

    if mode != SPEAKER_DIARIZATION_VOICEPRINT and (roles or voiceprint_ids):
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerRoles/VoiceprintIds require SpeakerDiarization=3",
        )

    for i, role in enumerate(roles or []):
        if not role.role_name:
            raise ASRError(ERR_INVALID_PARAM, "SpeakerRoles[{}].RoleName is empty".format(i))
        validate_enrollment_url(i, role.audio_url)

    for i, voiceprint_id in enumerate(voiceprint_ids or []):
        if not voiceprint_id:
            raise ASRError(ERR_INVALID_PARAM, "VoiceprintIds[{}] is empty".format(i))


def validate_enrollment_url(index: int, raw_url: str) -> None:
    """Require an absolute http(s) URL for enrollment audio.

    The URL is fetched by the ASR service, not by the SDK: this is a
    customer-facing client library, so it only rejects inputs that can never
    work (bad syntax, non-http scheme, missing host). Reachability and network
    policies belong to the service-side allow list.
    """
    if not raw_url.strip():
        raise ASRError(ERR_INVALID_PARAM, "SpeakerRoles[{}].AudioUrl is empty".format(index))
    # Reject bad syntax locally the way Go's url.ParseRequestURI does:
    # control characters / spaces and invalid percent-escapes must never
    # reach the wire.
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in raw_url):
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerRoles[{}].AudioUrl is not a valid URL: contains control characters or spaces".format(index),
        )
    if _INVALID_PERCENT_ESCAPE.search(raw_url):
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerRoles[{}].AudioUrl is not a valid URL: invalid percent-escape".format(index),
        )
    try:
        parsed = urlparse(raw_url)
    except Exception as exc:  # pragma: no cover - urlparse rarely raises
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerRoles[{}].AudioUrl is not a valid URL: {}".format(index, exc),
        )
    if parsed.scheme not in ("http", "https"):
        raise ASRError(
            ERR_INVALID_PARAM,
            "SpeakerRoles[{}].AudioUrl must use http or https, got '{}'".format(index, parsed.scheme),
        )
    if not parsed.hostname:
        raise ASRError(ERR_INVALID_PARAM, "SpeakerRoles[{}].AudioUrl has no host".format(index))


def validate_vad_tuning(
    vad_level: Optional[int], noise_threshold: Optional[float]
) -> None:
    """Check the VAD profile and noise threshold."""
    if vad_level is not None and vad_level not in (0, 1):
        raise ASRError(
            ERR_INVALID_PARAM,
            "VadLevel must be 0 (high recall) or 1 (far-field filtering), got {}".format(vad_level),
        )
    if noise_threshold is not None:
        v = noise_threshold
        # NaN fails every comparison, so test the valid range positively.
        if not (MIN_NOISE_THRESHOLD <= v <= MAX_NOISE_THRESHOLD):
            raise ASRError(
                ERR_INVALID_PARAM,
                "NoiseThreshold must be between {} and {}, got {}".format(
                    MIN_NOISE_THRESHOLD, MAX_NOISE_THRESHOLD, v
                ),
            )


def validate_enum_option(name: str, value: int, allowed) -> None:
    """Check a small enumerated option such as input_sample_rate."""
    if value not in allowed:
        raise ASRError(
            ERR_INVALID_PARAM,
            "{} must be one of {}, got {}".format(name, list(allowed), value),
        )
