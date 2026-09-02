"""Tests for shared parameter validation (aligned with Go asr/params_test.go)."""

import math

import pytest

from trtc_asr.errors import ASRError, ERR_INVALID_PARAM
from trtc_asr.params import (
    validate_enum_option,
    validate_speaker_diarization,
    validate_vad_tuning,
)
from trtc_asr.signature import (
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_VOICEPRINT,
    SpeakerRole,
)

VALID_ROLE = SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")


class TestValidateSpeakerDiarization:
    def test_off(self):
        validate_speaker_diarization(SPEAKER_DIARIZATION_OFF, 0, [], [])

    def test_cluster(self):
        validate_speaker_diarization(SPEAKER_DIARIZATION_CLUSTER, 0, [], [])

    def test_cluster_with_number_hint(self):
        validate_speaker_diarization(SPEAKER_DIARIZATION_CLUSTER, 2, [], [])

    def test_voiceprint_with_enrollment(self):
        validate_speaker_diarization(
            SPEAKER_DIARIZATION_VOICEPRINT,
            2,
            [VALID_ROLE],
            ["vp-1"],
        )

    def test_unsupported_mode(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(2, 0, [], [])
        assert "SpeakerDiarization must be 0" in str(exc_info.value)
        assert exc_info.value.code == ERR_INVALID_PARAM

    def test_negative_speaker_number(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(SPEAKER_DIARIZATION_CLUSTER, -1, [], [])
        assert "SpeakerNumber must be >= 0" in str(exc_info.value)

    def test_roles_without_voiceprint_mode(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(SPEAKER_DIARIZATION_CLUSTER, 0, [VALID_ROLE], [])
        assert "require SpeakerDiarization=3" in str(exc_info.value)

    def test_voiceprint_ids_without_voiceprint_mode(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(SPEAKER_DIARIZATION_OFF, 0, [], ["vp-1"])
        assert "require SpeakerDiarization=3" in str(exc_info.value)

    def test_empty_role_name(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(
                SPEAKER_DIARIZATION_VOICEPRINT,
                0,
                [SpeakerRole(role_name="", audio_url="https://example.com/a.wav")],
                [],
            )
        assert "RoleName is empty" in str(exc_info.value)

    def test_empty_audio_url(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(
                SPEAKER_DIARIZATION_VOICEPRINT,
                0,
                [SpeakerRole(role_name="teacher", audio_url="")],
                [],
            )
        assert "AudioUrl is empty" in str(exc_info.value)

    def test_non_http_scheme(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(
                SPEAKER_DIARIZATION_VOICEPRINT,
                0,
                [SpeakerRole(role_name="teacher", audio_url="file:///etc/passwd")],
                [],
            )
        assert "must use http or https" in str(exc_info.value)

    def test_url_without_host(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(
                SPEAKER_DIARIZATION_VOICEPRINT,
                0,
                [SpeakerRole(role_name="teacher", audio_url="https:///a.wav")],
                [],
            )
        assert "has no host" in str(exc_info.value)

    def test_internal_host_allowed(self):
        # This SDK is customer-facing: internal hosts belong to the caller's
        # own network and stay fetchable for the service, so no SSRF-style
        # blocking (mirrors the Go decision).
        validate_speaker_diarization(
            SPEAKER_DIARIZATION_VOICEPRINT,
            0,
            [SpeakerRole(role_name="teacher", audio_url="http://192.168.1.10/a.wav")],
            [],
        )

    @pytest.mark.parametrize(
        "bad_url",
        [
            "http://exa mple.com/a.wav",  # space in host
            "http://\nexample.com/a.wav",  # control character
            "http://%zz/a.wav",  # invalid percent-escape
        ],
    )
    def test_rejects_malformed_url(self, bad_url):
        # Bad syntax must be rejected locally, mirroring Go's
        # url.ParseRequestURI behavior.
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(
                SPEAKER_DIARIZATION_VOICEPRINT,
                0,
                [SpeakerRole(role_name="teacher", audio_url=bad_url)],
                [],
            )
        assert "is not a valid URL" in str(exc_info.value)

    def test_empty_voiceprint_id(self):
        with pytest.raises(ASRError) as exc_info:
            validate_speaker_diarization(SPEAKER_DIARIZATION_VOICEPRINT, 0, [], [""])
        assert "VoiceprintIds[0] is empty" in str(exc_info.value)


class TestValidateVadTuning:
    def test_unset(self):
        validate_vad_tuning(None, None)

    def test_valid_levels(self):
        validate_vad_tuning(0, None)
        validate_vad_tuning(1, None)

    def test_invalid_level(self):
        with pytest.raises(ASRError) as exc_info:
            validate_vad_tuning(2, None)
        assert "VadLevel must be 0 (high recall) or 1 (far-field filtering)" in str(exc_info.value)

    def test_threshold_bounds(self):
        validate_vad_tuning(None, 0.0)
        validate_vad_tuning(None, 4.0)
        validate_vad_tuning(None, 1.5)

    def test_threshold_below_range(self):
        with pytest.raises(ASRError) as exc_info:
            validate_vad_tuning(None, -0.1)
        assert "NoiseThreshold must be between 0.0 and 4.0" in str(exc_info.value)

    def test_threshold_above_range(self):
        with pytest.raises(ASRError) as exc_info:
            validate_vad_tuning(None, 4.1)
        assert "NoiseThreshold must be between 0.0 and 4.0" in str(exc_info.value)

    def test_threshold_nan_rejected(self):
        # NaN fails every comparison, so the valid range must be tested
        # positively (mirrors the Go implementation).
        with pytest.raises(ASRError):
            validate_vad_tuning(None, math.nan)


class TestValidateEnumOption:
    def test_valid(self):
        validate_enum_option("InputSampleRate", 0, (0, 8000))
        validate_enum_option("InputSampleRate", 8000, (0, 8000))

    def test_invalid(self):
        with pytest.raises(ASRError) as exc_info:
            validate_enum_option("InputSampleRate", 16000, (0, 8000))
        assert "InputSampleRate must be one of [0, 8000], got 16000" in str(exc_info.value)
