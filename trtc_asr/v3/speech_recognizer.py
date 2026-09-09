"""Real-time speech recognition client for the TRTC-ASR v3 protocol.

Protocol recap (WebSocket /asr/v3):

1. Dial ``wss://{host}/asr/v3?voice_id=<id>`` — the URL carries only
   voice_id.
2. Within 3s of the handshake, send one text frame:
   ``{"type":"start","auth":{...},"params":{...}}`` (≤64KB).
3. The server acks ``{"code":0,...}`` or fails with a structured error frame
   followed by a normal close.
4. Stream audio as binary frames (≤256KB each), then send ``{"type":"end"}``.
5. Downlink result frames are identical to v2:
   ``{code, message, voice_id, message_id, result{slice_type,...}, final}``.

``start()`` waits synchronously for the acknowledgement, so authentication
(4002) and parameter errors (4001) surface from ``start()`` directly instead
of only via ``on_fail``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import websockets

from trtc_asr.credential import Credential, resolve_ws_endpoint
from trtc_asr.errors import (
    ASRError,
    ERR_ALREADY_STARTED,
    ERR_AUTH_FAILED,
    ERR_CONNECT_FAILED,
    ERR_INVALID_PARAM,
    ERR_NOT_STARTED,
    ERR_READ_FAILED,
    ERR_WRITE_FAILED,
)
from trtc_asr.params import (
    validate_enum_option,
    validate_speaker_diarization,
    validate_vad_tuning,
)
from trtc_asr.sdkinfo import sdk_report_params
from trtc_asr.usersig import gen_user_sig
from trtc_asr.v3.wire import (
    ACK_TIMEOUT,
    START_FRAME_MAX_BYTES,
    STREAM_FRAME_MAX_BYTES,
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_VOICEPRINT,
    Context,
    SpeakerRole,
)

logger = logging.getLogger(__name__)

ENDPOINT = "wss://asr.cloud-rtc.com"

DEFAULT_WRITE_TIMEOUT = 5.0
MIN_WRITE_TIMEOUT = 0.05
MAX_WRITE_TIMEOUT = 30.0

DEFAULT_STOP_TIMEOUT = 10.0
MIN_STOP_TIMEOUT = 1.0
MAX_STOP_TIMEOUT = 60.0

# Server-side accepted ranges (asr-proxy validator_v3.go).
MAX_VOICE_ID_LEN = 128
MIN_MAX_SPEAK_TIME = 5000
MAX_MAX_SPEAK_TIME = 90000
MIN_VAD_SILENCE_TIME_MS = 240
MAX_VAD_SILENCE_TIME_MS = 2000

# Server-side voice_format whitelist.
VALID_VOICE_FORMATS = (1, 4, 6, 8, 10, 11, 12, 14, 16)


class _State:
    IDLE = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4


@dataclass
class WordInfo:
    """Word-level recognition details."""

    word: str = ""
    start_time: int = 0
    end_time: int = 0
    stable_flag: int = 0
    speaker_id: int = 0
    speaker_name: str = ""


@dataclass
class SpeakerSegment:
    """A contiguous section of one result attributed to a single speaker."""

    speaker_id: int = 0
    speaker_name: str = ""
    start_time: int = 0
    end_time: int = 0
    text: str = ""
    word_start: Optional[int] = None
    word_end: Optional[int] = None
    stable_flag: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSegment":
        return cls(
            speaker_id=data.get("speaker_id", 0),
            speaker_name=data.get("speaker_name", ""),
            start_time=data.get("start_time", 0),
            end_time=data.get("end_time", 0),
            text=data.get("text", ""),
            word_start=data.get("word_start"),
            word_end=data.get("word_end"),
            stable_flag=data.get("stable_flag", 0),
        )


@dataclass
class Result:
    """Speech recognition result details (shape identical to v2)."""

    slice_type: int = 0
    index: int = 0
    start_time: int = 0
    end_time: int = 0
    voice_text_str: str = ""
    word_size: int = 0
    word_list: List[WordInfo] = field(default_factory=list)
    language: str = ""
    speaker_segments: List[SpeakerSegment] = field(default_factory=list)
    speaker_id: Optional[int] = None
    finish_silence_ms: int = 0
    last_token_runtime_ms: int = 0


@dataclass
class SpeechRecognitionResponse:
    """Response message from the ASR service (shape identical to v2)."""

    code: int = 0
    message: str = ""
    voice_id: str = ""
    message_id: str = ""
    final: int = 0
    result: Result = field(default_factory=Result)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeechRecognitionResponse":
        result_data = data.get("result") or {}
        word_list = [
            WordInfo(
                word=w.get("word", ""),
                start_time=w.get("start_time", 0),
                end_time=w.get("end_time", 0),
                stable_flag=w.get("stable_flag", 0),
                speaker_id=w.get("speaker_id", 0),
                speaker_name=w.get("speaker_name", ""),
            )
            for w in result_data.get("word_list") or []
        ]
        result = Result(
            slice_type=result_data.get("slice_type", 0),
            index=result_data.get("index", 0),
            start_time=result_data.get("start_time", 0),
            end_time=result_data.get("end_time", 0),
            voice_text_str=result_data.get("voice_text_str", ""),
            word_size=result_data.get("word_size", 0),
            word_list=word_list,
            language=result_data.get("language", ""),
            speaker_segments=[
                SpeakerSegment.from_dict(s)
                for s in result_data.get("speaker_segments") or []
            ],
            speaker_id=result_data.get("speaker_id"),
            finish_silence_ms=result_data.get("finish_silence_ms", 0),
            last_token_runtime_ms=result_data.get("last_token_runtime_ms", 0),
        )
        return cls(
            code=data.get("code", 0),
            message=data.get("message", ""),
            voice_id=data.get("voice_id", ""),
            message_id=data.get("message_id", ""),
            final=data.get("final", 0),
            result=result,
        )


class SpeechRecognitionListener:
    """Callback interface for speech recognition events.

    Every method has a default no-op implementation. Subclass and override
    only the events you care about.
    """

    def on_recognition_start(self, response: SpeechRecognitionResponse) -> None:
        """Called when the recognition session starts."""

    def on_sentence_begin(self, response: SpeechRecognitionResponse) -> None:
        """Called when a new sentence begins."""

    def on_recognition_result_change(self, response: SpeechRecognitionResponse) -> None:
        """Called when intermediate results are available."""

    def on_sentence_end(self, response: SpeechRecognitionResponse) -> None:
        """Called when a sentence ends with the final result."""

    def on_recognition_complete(self, response: SpeechRecognitionResponse) -> None:
        """Called when the entire recognition session completes."""

    def on_fail(self, response: Optional[SpeechRecognitionResponse], error: Exception) -> None:
        """Called when an error occurs during recognition."""


class SpeechRecognizer:
    """Real-time speech recognition client for the v3 protocol.

    Lifecycle and concurrency mirror the v2 client: single-use instances,
    options configured before start(), callbacks delivered on the internal
    read-loop task with exception shielding, and re-entrant stop() support.

    The v3 difference: ``start()`` waits synchronously for the server's
    start-frame acknowledgement, so authentication and parameter errors are
    raised directly from ``start()``.

    Example::

        credential = v3.new_credential(sdk_app_id=140xxx, secret_key="xxx")
        recognizer = v3.SpeechRecognizer(credential, "16k_zh_en", listener)

        await recognizer.start()   # auth/param errors raise here
        await recognizer.write(audio_data)
        await recognizer.stop()
    """

    def __init__(
        self,
        credential: Credential,
        engine_model_type: str,
        listener: Optional[SpeechRecognitionListener] = None,
    ) -> None:
        self._credential = credential
        self._listener = listener if listener is not None else SpeechRecognitionListener()
        self._engine_model_type = engine_model_type
        self._endpoint = ""

        # Configuration (defaults match the Go v3 SDK).
        self._voice_format = 1  # PCM
        self._need_vad = 1
        self._convert_num_mode = 1
        self._hotword_id = ""
        self._hotword_list = ""
        self._filter_dirty = 0
        self._filter_modal = 0
        self._filter_punc = 0
        self._filter_empty_result: Optional[int] = None
        self._word_info = 0
        self._word_with_space = 0
        self._vad_silence_time = 0
        self._vad_level: Optional[int] = None
        self._noise_threshold: Optional[float] = None
        self._max_speak_time = 0
        self._input_sample_rate = 0
        self._speaker_diarization = 0
        self._speaker_number = 0
        self._speaker_roles: List[SpeakerRole] = []
        self._voiceprint_ids: List[str] = []
        self._voice_id = ""
        self._language = ""
        self._context: Optional[Context] = None

        self._write_timeout = DEFAULT_WRITE_TIMEOUT
        self._stop_timeout = DEFAULT_STOP_TIMEOUT

        self._state = _State.IDLE
        self._ws = None
        self._read_task: Optional[asyncio.Task] = None
        self._callback_failed = False

    # ---- Configuration setters ----

    def set_voice_format(self, fmt: int) -> None:
        self._voice_format = fmt

    def set_need_vad(self, need_vad: int) -> None:
        """0: disable, 1: enable (default). Unlike the v2 transport, v3
        honors an explicit 0 (it is sent on the wire, not dropped)."""
        self._need_vad = need_vad

    def set_convert_num_mode(self, mode: int) -> None:
        """0: no conversion, 1: smart conversion (default), 3: math
        conversion. Unlike the v2 transport, v3 honors an explicit 0."""
        self._convert_num_mode = mode

    def set_hotword_id(self, hotword_id: str) -> None:
        self._hotword_id = hotword_id

    def set_hotword_list(self, hotword_list: str) -> None:
        self._hotword_list = hotword_list

    def set_filter_dirty(self, mode: int) -> None:
        self._filter_dirty = mode

    def set_filter_modal(self, mode: int) -> None:
        self._filter_modal = mode

    def set_filter_punc(self, mode: int) -> None:
        self._filter_punc = mode

    def set_filter_empty_result(self, mode: int) -> None:
        self._filter_empty_result = mode

    def set_word_info(self, mode: int) -> None:
        """0: no (default), 1: yes, 2: with punctuation timing, 100: caption."""
        self._word_info = mode

    def set_word_with_space(self, mode: int) -> None:
        """Set whether English words are joined with spaces. 0: no
        (default), 1: yes."""
        self._word_with_space = mode

    def set_vad_silence_time(self, ms: int) -> None:
        self._vad_silence_time = ms

    def set_vad_level(self, level: int) -> None:
        self._vad_level = level

    def set_noise_threshold(self, threshold: float) -> None:
        self._noise_threshold = threshold

    def set_max_speak_time(self, ms: int) -> None:
        self._max_speak_time = ms

    def set_input_sample_rate(self, rate: int) -> None:
        self._input_sample_rate = rate

    def set_speaker_diarization(self, mode: int) -> None:
        self._speaker_diarization = mode

    def set_speaker_number(self, n: int) -> None:
        self._speaker_number = n

    def set_speaker_roles(self, roles: List[SpeakerRole]) -> None:
        """Register temporary voiceprints (v3.SpeakerRole, serialized as
        snake_case role_name/audio_url). The list is copied."""
        self._speaker_roles = list(roles or [])

    def set_voiceprint_ids(self, ids: List[str]) -> None:
        self._voiceprint_ids = list(ids or [])

    def set_voice_id(self, voice_id: str) -> None:
        self._voice_id = voice_id

    def set_language(self, lang: str) -> None:
        self._language = lang

    def set_context(self, context: Optional[Context]) -> None:
        """Set the recognition context (background text, terms, domain
        key-values)."""
        self._context = context

    def set_endpoint(self, endpoint: str) -> None:
        self._endpoint = endpoint or ""

    def set_write_timeout(self, timeout: float) -> None:
        if timeout <= 0:
            timeout = DEFAULT_WRITE_TIMEOUT
        self._write_timeout = min(max(timeout, MIN_WRITE_TIMEOUT), MAX_WRITE_TIMEOUT)

    def set_stop_timeout(self, timeout: float) -> None:
        if timeout <= 0:
            timeout = DEFAULT_STOP_TIMEOUT
        self._stop_timeout = min(max(timeout, MIN_STOP_TIMEOUT), MAX_STOP_TIMEOUT)

    # ---- Core operations ----

    async def start(self) -> None:
        """Connect, send the start frame and wait for the server ack.

        Authentication (4002) and parameter errors (4001) are raised here
        synchronously instead of surfacing later via on_fail.
        """
        if self._state != _State.IDLE:
            raise ASRError(ERR_ALREADY_STARTED, "recognizer already started")

        self._validate_options()

        self._state = _State.STARTING
        try:
            first_result = await self._connect()
        except ASRError:
            self._state = _State.IDLE
            raise
        except Exception as exc:
            self._state = _State.IDLE
            raise ASRError(ERR_CONNECT_FAILED, "websocket connect failed: {}".format(exc)) from exc

        self._state = _State.RUNNING
        self._read_task = asyncio.create_task(self._read_loop(first_result))

    async def write(self, data: bytes) -> None:
        """Send one audio frame (binary, ≤256KB) to the ASR service."""
        if self._state != _State.RUNNING:
            raise ASRError(ERR_NOT_STARTED, "recognizer not running")
        if self._ws is None:
            raise ASRError(ERR_NOT_STARTED, "connection not established")
        if len(data) > STREAM_FRAME_MAX_BYTES:
            raise ASRError(
                ERR_INVALID_PARAM,
                "audio frame exceeds {} bytes".format(STREAM_FRAME_MAX_BYTES),
            )

        if self._state != _State.RUNNING:
            raise ASRError(ERR_NOT_STARTED, "recognizer not running")

        try:
            await asyncio.wait_for(self._ws.send(data), timeout=self._write_timeout)
        except Exception as exc:
            raise ASRError(ERR_WRITE_FAILED, "write audio data failed: {}".format(exc)) from exc

    async def stop(self) -> None:
        """Gracefully stop the recognition session (sends ``{"type":"end"}``
        and waits for the server's final response)."""
        if self._state == _State.STOPPED:
            return
        if self._state != _State.RUNNING:
            raise ASRError(ERR_NOT_STARTED, "recognizer not running")

        self._state = _State.STOPPING

        if self._ws is None:
            self._state = _State.STOPPED
            raise ASRError(ERR_NOT_STARTED, "connection not established")

        try:
            end_msg = json.dumps({"type": "end"})
            await asyncio.wait_for(self._ws.send(end_msg), timeout=self._write_timeout)
        except Exception as exc:
            if self._state == _State.STOPPED:
                return
            await self._close()
            self._state = _State.STOPPED
            raise ASRError(ERR_WRITE_FAILED, "send end signal failed: {}".format(exc)) from exc

        # Re-entrant stop from a listener callback must not self-block on the
        # read-loop task it is running on; a detached watchdog preserves the
        # timeout semantics.
        if self._read_task is not None and asyncio.current_task() is self._read_task:
            asyncio.create_task(self._wait_for_read_loop())
            return

        await self._wait_for_read_loop()
        self._state = _State.STOPPED

    # ---- Internal methods ----

    def _validate_options(self) -> None:
        """Check the options that have a documented server-side range,
        mirroring the proxy's v3 online validator so an invalid value fails
        locally instead of coming back as a remote 4001."""
        if len(self._voice_id) > MAX_VOICE_ID_LEN:
            raise ASRError(
                ERR_INVALID_PARAM,
                "VoiceID length must not exceed {}".format(MAX_VOICE_ID_LEN),
            )
        validate_speaker_diarization(
            self._speaker_diarization,
            self._speaker_number,
            self._speaker_roles,
            self._voiceprint_ids,
        )
        validate_vad_tuning(self._vad_level, self._noise_threshold)
        if self._max_speak_time and not (
            MIN_MAX_SPEAK_TIME <= self._max_speak_time <= MAX_MAX_SPEAK_TIME
        ):
            raise ASRError(
                ERR_INVALID_PARAM,
                "MaxSpeakTime must be between {} and {} ms, got {}".format(
                    MIN_MAX_SPEAK_TIME, MAX_MAX_SPEAK_TIME, self._max_speak_time
                ),
            )
        # vad_silence_time==0 means "not set" (the field is then omitted on
        # the wire); the range check only applies to an explicitly set value
        # with VAD enabled, matching the server rule.
        if (
            self._vad_silence_time
            and self._need_vad == 1
            and not (MIN_VAD_SILENCE_TIME_MS <= self._vad_silence_time <= MAX_VAD_SILENCE_TIME_MS)
        ):
            raise ASRError(
                ERR_INVALID_PARAM,
                "VadSilenceTime must be between {} and {} ms (needvad=1), got {}".format(
                    MIN_VAD_SILENCE_TIME_MS, MAX_VAD_SILENCE_TIME_MS, self._vad_silence_time
                ),
            )
        checks = (
            ("NeedVad", self._need_vad, (0, 1)),
            ("ConvertNumMode", self._convert_num_mode, (0, 1, 3)),
            ("FilterDirty", self._filter_dirty, (0, 1, 2)),
            ("FilterModal", self._filter_modal, (0, 1, 2)),
            ("FilterPunc", self._filter_punc, (0, 1)),
            ("WordInfo", self._word_info, (0, 1, 2, 100)),
            ("WordWithSpace", self._word_with_space, (0, 1)),
            ("VoiceFormat", self._voice_format, VALID_VOICE_FORMATS),
            # 8000 is the only supported override; 0 means "use the engine rate".
            ("InputSampleRate", self._input_sample_rate, (0, 8000)),
        )
        for name, value, allowed in checks:
            validate_enum_option(name, value, allowed)
        if self._filter_empty_result is not None:
            validate_enum_option("FilterEmptyResult", self._filter_empty_result, (0, 1))

    def _build_params(self) -> dict:
        """Assemble the snake_case params block. Pointer fields (None) are
        omitted so the server default applies; SDK-managed defaults
        (needvad/convert_num_mode/voice_format) are always sent — including
        an explicit 0, which v3 honors (the v2 query transport dropped it)."""
        params = {
            "voice_id": self._voice_id,
            "engine_model_type": self._engine_model_type,
            "voice_format": self._voice_format,
            "needvad": self._need_vad,
            "convert_num_mode": self._convert_num_mode,
            # SDK telemetry: the gateway replays the start frame byte-for-byte
            # to the worker, which ignores unknown keys, so it survives in
            # server-side dumps without disturbing the protocol.
            "sdk_info": sdk_report_params(),
        }
        if self._language:
            params["language"] = self._language
        if self._hotword_id:
            params["hotword_id"] = self._hotword_id
        if self._hotword_list:
            params["hotword_list"] = self._hotword_list
        if self._filter_dirty:
            params["filter_dirty"] = self._filter_dirty
        if self._filter_modal:
            params["filter_modal"] = self._filter_modal
        if self._filter_punc:
            params["filter_punc"] = self._filter_punc
        if self._filter_empty_result is not None:
            params["filter_empty_result"] = self._filter_empty_result
        if self._word_info:
            params["word_info"] = self._word_info
        if self._word_with_space:
            params["word_with_space"] = self._word_with_space
        if self._vad_silence_time:
            params["vad_silence_time"] = self._vad_silence_time
        if self._vad_level is not None:
            params["vad_level"] = self._vad_level
        if self._noise_threshold is not None:
            params["noise_threshold"] = self._noise_threshold
        if self._max_speak_time:
            params["max_speak_time"] = self._max_speak_time
        if self._input_sample_rate:
            params["input_sample_rate"] = self._input_sample_rate
        if self._speaker_diarization:
            params["speaker_diarization"] = self._speaker_diarization
            if self._speaker_number:
                params["speaker_number"] = self._speaker_number
        if self._speaker_diarization == SPEAKER_DIARIZATION_VOICEPRINT:
            if self._speaker_roles:
                params["speaker_roles"] = [r.to_wire() for r in self._speaker_roles]
            if self._voiceprint_ids:
                params["voiceprint_ids"] = list(self._voiceprint_ids)
        if self._context is not None:
            wire = self._context.to_wire()
            if wire:
                params["context"] = wire
        return params

    def _build_start_frame(self, user_sig: str) -> str:
        """Return the start frame as a JSON string — it must be sent as a
        TEXT frame (a binary first frame is rejected by the server with
        4010)."""
        frame = {
            "type": "start",
            "auth": {"sdkappid": str(self._credential.sdk_app_id), "usersig": user_sig},
            "params": self._build_params(),
        }
        data = json.dumps(frame)
        if len(data.encode("utf-8")) > START_FRAME_MAX_BYTES:
            raise ASRError(
                ERR_INVALID_PARAM,
                "start frame exceeds {} bytes ({})".format(
                    START_FRAME_MAX_BYTES, len(data.encode("utf-8"))
                ),
            )
        return data

    async def _connect(self) -> Optional[bytes]:
        """Dial /asr/v3, send the start frame and wait for the server ack.

        Returns the first downlink frame when that frame already carries a
        result (the ack frame itself is consumed here and not reported
        again).
        """
        voice_id = self._voice_id or str(uuid.uuid4())
        self._voice_id = voice_id

        # Resolve UserSig locally without mutating the shared credential.
        # The v3 signature identifier is the voice_id.
        user_sig = self._credential.user_sig
        if not user_sig:
            try:
                user_sig = gen_user_sig(
                    self._credential.sdk_app_id,
                    self._credential.secret_key,
                    voice_id,
                    86400,
                )
            except Exception as exc:
                raise ASRError(ERR_AUTH_FAILED, "generate user sig failed: {}".format(exc)) from exc

        frame = self._build_start_frame(user_sig)

        base = resolve_ws_endpoint(self._endpoint, self._credential.site)
        # The URL carries only voice_id; auth and params travel in the start
        # frame, so the handshake stays header-free and browser-friendly.
        ws_url = "{}/asr/v3?voice_id={}".format(base, voice_id)

        self._ws = await websockets.asyncio.client.connect(ws_url, open_timeout=10)

        # Send the start frame immediately (the server enforces a 3s deadline).
        try:
            await asyncio.wait_for(self._ws.send(frame), timeout=self._write_timeout)
        except Exception as exc:
            await self._close()
            raise ASRError(ERR_WRITE_FAILED, "send start frame failed: {}".format(exc)) from exc

        # Wait for the ack. A failure arrives as a structured error frame
        # ({code,message,voice_id}) followed by a normal close.
        try:
            message = await asyncio.wait_for(self._ws.recv(), timeout=ACK_TIMEOUT)
        except Exception as exc:
            await self._close()
            raise ASRError(ERR_READ_FAILED, "read start ack failed: {}".format(exc)) from exc
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        try:
            ack = json.loads(message)
        except (json.JSONDecodeError, ValueError) as exc:
            await self._close()
            raise ASRError(ERR_SERVER_ERROR, "invalid start ack: {}".format(exc)) from exc

        code = ack.get("code", 0)
        if code != 0:
            await self._close()
            raise ASRError(code, ack.get("message", ""))

        if ack.get("result") is not None:
            return message
        return None

    async def _read_loop(self, first_result: Optional[bytes]) -> None:
        try:
            self._callback_failed = False
            self._safe_callback(
                self._listener.on_recognition_start,
                SpeechRecognitionResponse(code=0, message="success", voice_id=self._voice_id),
            )
            if self._callback_failed:
                return

            if first_result is not None:
                if not self._handle_message(first_result):
                    return

            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                if not self._handle_message(message):
                    return
        except websockets.ConnectionClosed:
            if self._state < _State.STOPPING:
                self._finish()
                self._safe_on_fail(
                    None, ASRError(ERR_READ_FAILED, "websocket connection closed unexpectedly")
                )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._state < _State.STOPPING:
                self._finish()
                self._safe_on_fail(
                    None, ASRError(ERR_READ_FAILED, "read message failed: {}".format(exc))
                )
            return
        finally:
            self._finish()
            await self._close()

    def _handle_message(self, message: str) -> bool:
        """Dispatch one downlink frame. Returns False when the loop must
        stop (terminal response or error)."""
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, ValueError) as exc:
            # Non-terminal: the session continues, so do not finish here.
            self._safe_on_fail(
                None, ASRError(ERR_READ_FAILED, "unmarshal response failed: {}".format(exc))
            )
            return True

        resp = SpeechRecognitionResponse.from_dict(data)

        if resp.code != 0:
            # Terminal: finish the lifecycle before notifying, so a
            # stop()/write() call from inside on_fail sees the stopped state.
            self._finish()
            self._safe_on_fail(resp, ASRError(resp.code, resp.message))
            return False

        if resp.final == 1:
            self._finish()
            self._dispatch_event(resp)
            if not self._callback_failed:
                self._safe_callback(
                    self._listener.on_recognition_complete, resp, _shield=True
                )
            return False

        # Skip frames without a "result" object (defensive: the ack frame is
        # consumed by _connect, but a frame lacking result must not be
        # misread as a slice_type=0 sentence begin).
        if data.get("result") is None:
            return True

        self._dispatch_event(resp)
        return not self._callback_failed

    def _dispatch_event(self, resp: SpeechRecognitionResponse) -> None:
        if resp.final == 1 and resp.result.slice_type != 2:
            return

        if resp.result.slice_type == 0:
            self._safe_callback(self._listener.on_sentence_begin, resp)
        elif resp.result.slice_type == 1:
            self._safe_callback(self._listener.on_recognition_result_change, resp)
        elif resp.result.slice_type == 2:
            self._safe_callback(self._listener.on_sentence_end, resp)

    def _finish(self) -> None:
        """Advance the recognizer to the terminal stopped state (idempotent).

        The WebSocket is closed by the read loop's finally block via
        ``_close``: closing from here with ``create_task`` races the
        still-running ``async for``.
        """
        self._state = _State.STOPPED

    async def _wait_for_read_loop(self) -> None:
        if self._read_task is None:
            return
        try:
            await asyncio.wait_for(self._read_task, timeout=self._stop_timeout)
        except asyncio.TimeoutError:
            await self._close()

    def _safe_callback(self, callback, *args, _shield: bool = False) -> None:
        """Deliver a listener callback while shielding the read loop from an
        exception raised inside the user-supplied listener."""
        try:
            callback(*args)
        except Exception as exc:
            if _shield or self._callback_failed:
                logger.exception("listener callback raised, ignored")
                return
            logger.exception("listener callback raised")
            self._callback_failed = True
            self._finish()
            self._safe_on_fail(
                None,
                ASRError(
                    ERR_READ_FAILED,
                    "recovered from panic in readLoop: {}\n{}".format(
                        exc, traceback.format_exc()
                    ),
                ),
            )

    def _safe_on_fail(self, response, error) -> None:
        self._safe_callback(self._listener.on_fail, response, error, _shield=True)

    async def _close(self) -> None:
        if self._ws is not None:
            ws = self._ws
            self._ws = None
            try:
                await ws.close()
            except Exception:
                pass
