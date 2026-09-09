"""TRTC-ASR v3 protocol client.

The v3 protocol restructures the wire format around two separated blocks: an
"auth" block consumed by the gateway's authentication layer, and a "params"
block consumed by the recognition layer. Field names are snake_case and
responses are flat with numeric codes.

v3 uses SdkAppID as the only customer dimension; the Tencent Cloud AppID is
not needed (see :func:`new_credential`).

Interfaces:

- :class:`SpeechRecognizer`: WebSocket ``/asr/v3`` — the URL carries only
  voice_id; auth and params travel in a single start frame sent right after
  the handshake. ``start()`` waits synchronously for the server ack.
- :class:`SentenceRecognizer`: HTTP ``POST /v3/transcribe`` — one-shot (≤60s).
- :class:`FileRecognizer`: HTTP ``/v3/create_transcription`` +
  ``/v3/describe_transcription`` — async task for long audio.

Usage::

    from trtc_asr import v3

    credential = v3.new_credential(sdk_app_id=140xxx, secret_key="xxx")
    listener = MyListener()
    recognizer = v3.SpeechRecognizer(credential, "16k_zh_en", listener)

    await recognizer.start()
    await recognizer.write(audio_data)
    await recognizer.stop()

The v2/v1 clients in the parent ``trtc_asr`` package remain fully supported
and unchanged.
"""

from trtc_asr.v3.speech_recognizer import (
    ENDPOINT,
    SpeechRecognizer,
    SpeechRecognitionListener,
    SpeechRecognitionResponse,
    SpeakerSegment,
    Result,
    WordInfo,
)
from trtc_asr.v3.sentence_recognizer import (
    SentenceRecognizer,
    TranscribeRequest,
    TranscribeResponse,
)
from trtc_asr.v3.file_recognizer import (
    FileRecognizer,
    CreateTranscriptionRequest,
    TranscriptionStatus,
    SentenceDetail,
)
from trtc_asr.v3.wire import (
    SOURCE_TYPE_URL,
    SOURCE_TYPE_DATA,
    TASK_STATUS_WAITING,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILED,
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_VOICEPRINT,
    AudioURLItem,
    Context,
    ContextKV,
    SpeakerRole,
    Word,
    new_credential,
)

__all__ = [
    "ENDPOINT",
    "new_credential",
    "SpeechRecognizer",
    "SpeechRecognitionListener",
    "SpeechRecognitionResponse",
    "SpeakerSegment",
    "Result",
    "WordInfo",
    "SentenceRecognizer",
    "TranscribeRequest",
    "TranscribeResponse",
    "FileRecognizer",
    "CreateTranscriptionRequest",
    "TranscriptionStatus",
    "SentenceDetail",
    "SOURCE_TYPE_URL",
    "SOURCE_TYPE_DATA",
    "TASK_STATUS_WAITING",
    "TASK_STATUS_RUNNING",
    "TASK_STATUS_SUCCESS",
    "TASK_STATUS_FAILED",
    "SPEAKER_DIARIZATION_OFF",
    "SPEAKER_DIARIZATION_CLUSTER",
    "SPEAKER_DIARIZATION_VOICEPRINT",
    "AudioURLItem",
    "Context",
    "ContextKV",
    "SpeakerRole",
    "Word",
]
