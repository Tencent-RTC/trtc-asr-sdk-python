"""Speaker diarization / VAD tuning signature tests (aligned with Go
common/signature_speaker_test.go)."""

from urllib.parse import parse_qs

from trtc_asr.signature import (
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_VOICEPRINT,
    SignatureParams,
    SpeakerRole,
)


def _params() -> SignatureParams:
    return SignatureParams(
        app_id=1300403317,
        engine_model_type="16k_zh",
        voice_id="voice-001",
    )


def _query(qs: str) -> dict:
    return parse_qs(qs, keep_blank_values=True)


def test_omits_unset_optional_params():
    qs = _params().build_query_string()

    for key in [
        "speaker_diarization",
        "speaker_number",
        "speaker_roles",
        "voiceprintids",
        "noise_threshold",
        "vad_level",
        "filter_empty_result",
        "hotword_list",
        "replace_text_id",
        "input_sample_rate",
    ]:
        assert key + "=" not in qs, f"query should not contain {key} when unset: {qs}"


def test_speaker_diarization_cluster():
    params = _params()
    params.speaker_diarization = SPEAKER_DIARIZATION_CLUSTER
    # The speaker count hint feeds online clustering in both modes.
    params.speaker_number = 2
    # Enrollment input only applies to mode 3 and must not leak into mode 1.
    params.speaker_roles = [SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav")]
    params.voiceprint_ids = ["vp-1"]

    q = _query(params.build_query_string())

    assert q["speaker_diarization"] == ["1"]
    assert q["speaker_number"] == ["2"]
    assert "speaker_roles" not in q
    assert "voiceprintids" not in q


def test_speaker_diarization_voiceprint():
    params = _params()
    params.speaker_diarization = SPEAKER_DIARIZATION_VOICEPRINT
    params.speaker_roles = [
        SpeakerRole(role_name="teacher", audio_url="https://example.com/a.wav"),
        SpeakerRole(role_name="student", audio_url="https://example.com/b.wav"),
    ]
    params.voiceprint_ids = ["vp-1", "vp-2"]

    params.speaker_number = 0  # auto detection stays the server default

    q = _query(params.build_query_string())

    assert q["speaker_diarization"] == ["3"]
    # 0 means auto detection; the server applies the same default, so the
    # parameter is omitted instead of being pinned to zero.
    assert "speaker_number" not in q

    import json

    roles = json.loads(q["speaker_roles"][0])
    assert len(roles) == 2
    assert roles[0]["RoleName"] == "teacher"
    assert roles[1]["AudioUrl"] == "https://example.com/b.wav"

    ids = json.loads(q["voiceprintids"][0])
    assert ids == ["vp-1", "vp-2"]


def test_tri_state_vad_tuning():
    import json

    params = _params()
    params.vad_level = 0
    params.noise_threshold = 0.0
    params.filter_empty_result = 0

    q = _query(params.build_query_string())

    # An explicit 0 differs from "unset": the server defaults vad_level to 1
    # and filter_empty_result to 1, so both must reach the wire.
    assert q["vad_level"] == ["0"]
    assert q["filter_empty_result"] == ["0"]
    assert q["noise_threshold"] == ["0.000"]

    params.noise_threshold = 1.5
    q = _query(params.build_query_string())
    assert q["noise_threshold"] == ["1.500"]


def test_advanced_optional_params():
    params = _params()
    params.hotword_list = "腾讯云|5,ASR|11"
    params.replace_text_id = "replace-1"
    params.input_sample_rate = 8000
    params.language = "zh"

    q = _query(params.build_query_string())

    assert q["hotword_list"] == ["腾讯云|5,ASR|11"]
    assert q["replace_text_id"] == ["replace-1"]
    assert q["input_sample_rate"] == ["8000"]
    assert q["language"] == ["zh"]


def test_sdkappid_emitted_when_configured():
    params = _params()
    params.sdk_app_id = 1400000000
    q = _query(params.build_query_string())
    assert q["sdkappid"] == ["1400000000"]

    # 0 means not configured: omitted.
    params.sdk_app_id = 0
    q = _query(params.build_query_string())
    assert "sdkappid" not in q


def test_signature_and_usersig_both_carry_user_sig():
    params = _params()
    params.sdk_app_id = 1400000000
    user_sig = "eJyrVgrxCdYrLkksyczPs1KyUkqpTM4sSgUAR94HgQ--"
    q = _query(params.build_query_string_with_signature(user_sig))

    # Per protocol the signature parameter equals the UserSig, and the same
    # value is sent as usersig so the gateway authenticates without headers.
    assert q["signature"] == [user_sig]
    assert q["usersig"] == [user_sig]
