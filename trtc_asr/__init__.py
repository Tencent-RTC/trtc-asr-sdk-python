"""Tencent TRTC ASR SDK for Python."""

from trtc_asr.credential import Credential
from trtc_asr.signature import (
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_VOICEPRINT,
    SpeakerRole,
)
from trtc_asr.speech_recognizer import (
    SpeechRecognizer,
    SpeechRecognitionListener,
    SpeechRecognitionResponse,
    SpeakerSegment,
    Result,
    WordInfo,
)
from trtc_asr.sentence_recognizer import (
    SentenceRecognizer,
    SentenceRecognitionRequest,
    SentenceRecognitionResult,
)
from trtc_asr.file_recognizer import (
    FileRecognizer,
    CreateRecTaskRequest,
    TaskStatus,
    SentenceDetail,
)
from trtc_asr.errors import ASRError
from trtc_asr.sdkinfo import SDK_VERSION

__all__ = [
    "Credential",
    "SpeakerRole",
    "SPEAKER_DIARIZATION_OFF",
    "SPEAKER_DIARIZATION_CLUSTER",
    "SPEAKER_DIARIZATION_VOICEPRINT",
    "SpeechRecognizer",
    "SpeechRecognitionListener",
    "SpeechRecognitionResponse",
    "SpeakerSegment",
    "Result",
    "WordInfo",
    "SentenceRecognizer",
    "SentenceRecognitionRequest",
    "SentenceRecognitionResult",
    "FileRecognizer",
    "CreateRecTaskRequest",
    "TaskStatus",
    "SentenceDetail",
    "ASRError",
]

# Single source of truth lives in sdkinfo, which also reports it to the
# service; re-exported here so ``trtc_asr.__version__`` keeps working.
__version__ = SDK_VERSION
