"""Real-time speech recognition client for TRTC-ASR."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import websockets
import websockets.asyncio.client

from trtc_asr.credential import Credential, resolve_ws_endpoint
from trtc_asr.errors import (
    ASRError,
    ERR_ALREADY_STARTED,
    ERR_AUTH_FAILED,
    ERR_CONNECT_FAILED,
    ERR_NOT_STARTED,
    ERR_READ_FAILED,
    ERR_WRITE_FAILED,
)
from trtc_asr.params import (
    validate_enum_option,
    validate_speaker_diarization,
    validate_vad_tuning,
)
from trtc_asr.signature import (
    SPEAKER_DIARIZATION_CLUSTER,
    SPEAKER_DIARIZATION_OFF,
    SPEAKER_DIARIZATION_VOICEPRINT,
    SignatureParams,
    SpeakerRole,
)
from trtc_asr.usersig import gen_user_sig

logger = logging.getLogger(__name__)

ENDPOINT = "wss://asr.cloud-rtc.com"

# Write-timeout bounds. A single write is bounded by write_timeout, so
# stop()'s worst-case wait to acquire the writer for the end signal is
# bounded as well. Clamping keeps stop()'s exit time predictable.
DEFAULT_WRITE_TIMEOUT = 5.0  # seconds
MIN_WRITE_TIMEOUT = 0.05
MAX_WRITE_TIMEOUT = 30.0

# Stop-timeout bounds. stop_timeout caps how long stop() waits for the
# server's final response after the end signal before forcing the
# connection closed.
DEFAULT_STOP_TIMEOUT = 10.0  # seconds
MIN_STOP_TIMEOUT = 1.0
MAX_STOP_TIMEOUT = 60.0


class _State(IntEnum):
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

    # speaker_id is the speaker of this word, filled when speaker diarization
    # is enabled together with word_info != 0. Valid IDs start at 1, -1 means
    # unknown, 0 means absent.
    speaker_id: int = 0

    # speaker_name is the enrolled role name, returned only with
    # speaker_diarization=3.
    speaker_name: str = ""


@dataclass
class SpeakerSegment:
    """A contiguous section of one result attributed to a single speaker.

    Returned when speaker diarization is enabled.
    """

    # speaker_id is the speaker number within the current session. Valid IDs
    # start at 1, -1 means unknown, 0 is reserved.
    speaker_id: int = 0

    # speaker_name is the enrolled role name, returned only with
    # speaker_diarization=3. It equals the requested SpeakerRole.role_name.
    speaker_name: str = ""

    start_time: int = 0
    end_time: int = 0
    text: str = ""

    # word_start / word_end are inclusive indexes into Result.word_list,
    # i.e. word_list[word_start : word_end + 1]. Both are None when
    # word_info=0 (no word list to index into); 0 is a valid index.
    word_start: Optional[int] = None
    word_end: Optional[int] = None

    # stable_flag reports whether this segment is stable: 1=stable, 0=not.
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
    """Speech recognition result details."""

    slice_type: int = 0
    index: int = 0
    start_time: int = 0
    end_time: int = 0
    voice_text_str: str = ""
    word_size: int = 0
    word_list: List[WordInfo] = field(default_factory=list)

    # language is the detected language (bigmodel engine, e.g. "Malay").
    language: str = ""

    # speaker_segments lists the speaker attribution of this result, split by
    # speaker turn. It is the recommended entry point for speaker diarization:
    # one result may contain several speakers, so a sentence-level speaker is
    # ambiguous by design. Empty when diarization is disabled.
    #
    # A result is single-speaker when len(speaker_segments) == 1.
    speaker_segments: List[SpeakerSegment] = field(default_factory=list)

    # speaker_id is the legacy sentence-level speaker attribution. It is None
    # when the field is absent (0 is a reserved value and the field is absent
    # on most engines). Prefer speaker_segments / WordInfo.speaker_id.
    speaker_id: Optional[int] = None

    # finish_silence_ms is the trailing silence (ms) that triggered the
    # sentence break. Zero when the server does not report it.
    finish_silence_ms: int = 0

    # last_token_runtime_ms is the server-side decoding time (ms) of the last
    # token. Zero when the server does not report it.
    last_token_runtime_ms: int = 0


@dataclass
class SpeechRecognitionResponse:
    """Response message from the ASR service."""

    code: int = 0
    message: str = ""
    voice_id: str = ""
    message_id: str = ""
    final: int = 0
    result: Result = field(default_factory=Result)

    @classmethod
    def from_dict(cls, data: dict) -> SpeechRecognitionResponse:
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
    """Real-time speech recognition client using WebSocket.

    Lifecycle and concurrency:

    - A SpeechRecognizer is single-use: once it reaches the stopped state
      (via stop() or a terminal error) it cannot be restarted. Create a new
      instance to reconnect.
    - All set_xxx options must be configured before start().
    - Recognition callbacks are delivered on the internal read-loop task. A
      faulty callback never crashes the loop: the exception is recovered,
      the session is finished, and the failure is surfaced via on_fail
      (mirroring the Go SDK's panic shielding).
    - stop() is safe to call from a recognition callback: it detects the
      re-entry and returns without waiting, so it cannot self-block.
      Calling stop() after the session has already stopped is a no-op.

    Example::

        credential = Credential(app_id=130xxx, sdk_app_id=140xxx, secret_key="xxx")
        listener = MyListener()
        recognizer = SpeechRecognizer(credential, "16k_zh", listener)

        await recognizer.start()
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

        # Configuration (defaults match Go SDK)
        self._voice_format = 1  # PCM
        self._need_vad = 1
        self._convert_num_mode = 1
        self._hotword_id = ""
        self._hotword_list = ""
        self._customization_id = ""
        self._replace_text_id = ""
        self._filter_dirty = 0
        self._filter_modal = 0
        self._filter_punc = 0
        self._filter_empty_result: Optional[int] = None
        self._word_info = 0
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

        self._write_timeout = DEFAULT_WRITE_TIMEOUT
        self._stop_timeout = DEFAULT_STOP_TIMEOUT

        self._state = _State.IDLE
        self._ws: Optional[websockets.asyncio.client.ClientConnection] = None
        self._read_task: Optional[asyncio.Task] = None
        # Set when a user callback raises; the read loop then skips further
        # dispatch (so a panicking on_sentence_end on a final frame does not
        # still deliver on_recognition_complete).
        self._callback_failed = False

    # ---- Configuration setters ----

    def set_voice_format(self, fmt: int) -> None:
        self._voice_format = fmt

    def set_need_vad(self, need_vad: int) -> None:
        self._need_vad = need_vad

    def set_convert_num_mode(self, mode: int) -> None:
        self._convert_num_mode = mode

    def set_hotword_id(self, hotword_id: str) -> None:
        self._hotword_id = hotword_id

    def set_hotword_list(self, hotword_list: str) -> None:
        """Set a temporary inline hotword list, which does not require
        creating a hotword table on the console.

        Format: "word1|weight1,word2|weight2". Each word is at most 30 bytes
        and the weight must be 1-11 (11 = super hotword) or 100 (homophone
        replacement).
        """
        self._hotword_list = hotword_list

    def set_customization_id(self, customization_id: str) -> None:
        self._customization_id = customization_id

    def set_replace_text_id(self, replace_text_id: str) -> None:
        """Set the replacement word table ID used for forced text
        replacement on the recognized result."""
        self._replace_text_id = replace_text_id

    def set_filter_dirty(self, mode: int) -> None:
        self._filter_dirty = mode

    def set_filter_modal(self, mode: int) -> None:
        self._filter_modal = mode

    def set_filter_punc(self, mode: int) -> None:
        self._filter_punc = mode

    def set_filter_empty_result(self, mode: int) -> None:
        """Set whether empty recognition results are delivered.

        0: deliver empty results, 1: skip them (server default).

        Calling this method makes the choice explicit on the wire, so passing
        0 is honored instead of falling back to the server default.
        """
        self._filter_empty_result = mode

    def set_word_info(self, mode: int) -> None:
        """Set whether to show word-level timing information.

        0: no (default), 1: yes, 2: include punctuation timing.

        Word-level speaker attribution (WordInfo.speaker_id) requires a
        non-zero value together with set_speaker_diarization.
        """
        self._word_info = mode

    def set_vad_silence_time(self, ms: int) -> None:
        self._vad_silence_time = ms

    def set_vad_level(self, level: int) -> None:
        """Select the VAD profile: 0 = high recall, 1 = far-field noise
        filtering (server default).

        Calling this method makes the choice explicit on the wire, so passing
        0 is honored instead of falling back to the server default.
        """
        self._vad_level = level

    def set_noise_threshold(self, threshold: float) -> None:
        """Fine-tune VAD noise suppression. Valid range: [0, 4]; larger
        values suppress more noise at the cost of recall. When set, it
        overrides the profile selected by set_vad_level.

        The value is only sent when this method is called, because 0 is a
        valid, meaningful threshold and cannot be distinguished from "unset"
        otherwise.
        """
        self._noise_threshold = threshold

    def set_max_speak_time(self, ms: int) -> None:
        self._max_speak_time = ms

    def set_input_sample_rate(self, rate: int) -> None:
        """Declare the sample rate of the incoming PCM audio.

        Only 8000 is supported, which lets an 8kHz stream be fed to a 16k
        engine (the server upsamples it).
        """
        self._input_sample_rate = rate

    def set_speaker_diarization(self, mode: int) -> None:
        """Enable real-time speaker diarization.

        - SPEAKER_DIARIZATION_OFF (0): disabled (default)
        - SPEAKER_DIARIZATION_CLUSTER (1): anonymous clustering; speakers are
          numbered from 1 within the session, -1 = unknown
        - SPEAKER_DIARIZATION_VOICEPRINT (3): voiceprint role authentication;
          combine with set_speaker_roles / set_voiceprint_ids to get role
          names back in speaker_name

        Results are reported through Result.speaker_segments, and additionally
        through WordInfo.speaker_id when word_info is non-zero.
        """
        self._speaker_diarization = mode

    def set_speaker_number(self, n: int) -> None:
        """Hint the expected number of speakers. 0 means auto detection
        (default). It applies to both diarization modes: the server feeds it
        into the online clustering."""
        self._speaker_number = n

    def set_speaker_roles(self, roles: List[SpeakerRole]) -> None:
        """Register temporary voiceprints for this session. Each role carries
        a name and the URL of its enrollment audio; the name is echoed back
        as speaker_name on matched words and speaker segments.

        Only used when speaker diarization is set to voiceprint mode. The
        list is copied, so later mutations by the caller do not affect the
        session.
        """
        self._speaker_roles = list(roles or [])

    def set_voiceprint_ids(self, ids: List[str]) -> None:
        """Register previously enrolled voiceprints by ID for this session.

        Only used when speaker diarization is set to voiceprint mode. The
        list is copied.
        """
        self._voiceprint_ids = list(ids or [])

    def set_voice_id(self, voice_id: str) -> None:
        self._voice_id = voice_id

    def set_language(self, lang: str) -> None:
        """Set the language hint for the bigmodel engine (e.g. "ms", "zh",
        "auto"). It is transparently forwarded to the server as the
        "language" query parameter."""
        self._language = lang

    def set_endpoint(self, endpoint: str) -> None:
        """Override the WebSocket endpoint (for testing against a mock server)."""
        self._endpoint = endpoint or ""

    def set_write_timeout(self, timeout: float) -> None:
        """Set the timeout for a single audio write, in seconds.

        The value is clamped to [0.05, 30]; a non-positive value resets it to
        the default. Because stop() must send the end signal after any
        in-flight write, an unbounded write timeout would let a blocked write
        delay stop() indefinitely — clamping keeps stop()'s worst-case exit
        time predictable.
        """
        if timeout <= 0:
            timeout = DEFAULT_WRITE_TIMEOUT
        self._write_timeout = min(max(timeout, MIN_WRITE_TIMEOUT), MAX_WRITE_TIMEOUT)

    def set_stop_timeout(self, timeout: float) -> None:
        """Set how long stop() waits for the server's final response after
        sending the end signal before forcing the connection closed, in
        seconds.

        The value is clamped to [1, 60]; a non-positive value resets it to
        the default.
        """
        if timeout <= 0:
            timeout = DEFAULT_STOP_TIMEOUT
        self._stop_timeout = min(max(timeout, MIN_STOP_TIMEOUT), MAX_STOP_TIMEOUT)

    # ---- Core operations ----

    async def start(self) -> None:
        """Initiate the WebSocket connection and begin the recognition session."""
        if self._state != _State.IDLE:
            raise ASRError(ERR_ALREADY_STARTED, "recognizer already started")

        # Validate before dialing so an invalid option fails locally instead
        # of costing a connection and coming back as a server-side 4001.
        try:
            self._validate_options()
        except ASRError:
            raise

        self._state = _State.STARTING

        try:
            await self._connect()
        except ASRError:
            self._state = _State.IDLE
            raise
        except Exception as exc:
            self._state = _State.IDLE
            raise ASRError(ERR_CONNECT_FAILED, "websocket connect failed: {}".format(exc)) from exc

        self._state = _State.RUNNING
        self._read_task = asyncio.create_task(self._read_loop())

    async def write(self, data: bytes) -> None:
        """Send audio data to the ASR service."""
        if self._state != _State.RUNNING:
            raise ASRError(ERR_NOT_STARTED, "recognizer not running")
        if self._ws is None:
            raise ASRError(ERR_NOT_STARTED, "connection not established")

        # Re-check the state after acquiring the send slot. Between the entry
        # check above and this point, stop() may have transitioned the state
        # and sent the end signal. Writing audio after end would violate the
        # protocol, so bail out instead.
        if self._state != _State.RUNNING:
            raise ASRError(ERR_NOT_STARTED, "recognizer not running")

        try:
            await asyncio.wait_for(self._ws.send(data), timeout=self._write_timeout)
        except Exception as exc:
            raise ASRError(ERR_WRITE_FAILED, "write audio data failed: {}".format(exc)) from exc

    async def stop(self) -> None:
        """Gracefully stop the recognition session.

        It sends the end signal and waits for the server's final response
        (up to stop_timeout) before forcing the connection closed.

        stop() is safe to call from a recognition callback: it detects the
        re-entry and returns after sending the end signal, so it cannot
        self-block on the read-loop task it is running on.

        For terminal callbacks (on_recognition_complete / terminal
        on_fail), the recognizer has already advanced to stopped before
        callback dispatch, so stop() is a no-op. Fire-and-forget
        ``create_task(stop())`` from those callbacks is therefore safe.
        """
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

        # If stop() is called from within a listener callback (which runs on
        # the read-loop task), waiting on the read task here would self-block
        # until timeout. In that case, return after sending end; the read
        # loop will continue and finish. A detached watchdog preserves
        # stop()'s timeout semantics if the server never sends a terminal
        # response after receiving end.
        if self._read_task is not None and asyncio.current_task() is self._read_task:
            asyncio.create_task(self._wait_for_read_loop())
            return

        # Wait for read loop to finish with timeout
        await self._wait_for_read_loop()

        self._state = _State.STOPPED

    # ---- Internal methods ----

    def _validate_options(self) -> None:
        """Check the options that have a documented server-side range."""
        validate_speaker_diarization(
            self._speaker_diarization,
            self._speaker_number,
            self._speaker_roles,
            self._voiceprint_ids,
        )
        validate_vad_tuning(self._vad_level, self._noise_threshold)
        if self._filter_empty_result is not None:
            validate_enum_option("FilterEmptyResult", self._filter_empty_result, (0, 1))
        # 8000 is the only supported override; 0 means "use the engine rate".
        validate_enum_option("InputSampleRate", self._input_sample_rate, (0, 8000))

    async def _connect(self) -> None:
        voice_id = self._voice_id or str(uuid.uuid4())
        self._voice_id = voice_id

        # Resolve UserSig locally without mutating the shared credential.
        # Writing back to credential.user_sig would race when a single
        # Credential is shared by multiple recognizers started concurrently.
        # This mirrors how the sentence / file recognizers resolve the
        # signature.
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

        # Build request parameters (AppID is used for the secretid URL
        # parameter). Authentication identity (sdkappid + usersig) travels in
        # the query string instead of headers, so browser WebSocket clients
        # work without header support; the gateway reads these query
        # parameters when the corresponding headers are absent.
        sig_params = SignatureParams(
            app_id=self._credential.app_id,
            engine_model_type=self._engine_model_type,
            voice_id=voice_id,
            voice_format=self._voice_format,
            need_vad=self._need_vad,
            convert_num_mode=self._convert_num_mode,
            sdk_app_id=self._credential.sdk_app_id,
            hotword_id=self._hotword_id,
            hotword_list=self._hotword_list,
            customization_id=self._customization_id,
            replace_text_id=self._replace_text_id,
            filter_dirty=self._filter_dirty,
            filter_modal=self._filter_modal,
            filter_punc=self._filter_punc,
            filter_empty_result=self._filter_empty_result,
            word_info=self._word_info,
            vad_silence_time=self._vad_silence_time,
            vad_level=self._vad_level,
            noise_threshold=self._noise_threshold,
            max_speak_time=self._max_speak_time,
            input_sample_rate=self._input_sample_rate,
            speaker_diarization=self._speaker_diarization,
            speaker_number=self._speaker_number,
            speaker_roles=self._speaker_roles,
            voiceprint_ids=self._voiceprint_ids,
            language=self._language,
        )

        query_string = sig_params.build_query_string_with_signature(user_sig)
        base = resolve_ws_endpoint(self._endpoint, self._credential.site)
        ws_url = "{}/asr/v2/{}?{}".format(base, self._credential.app_id, query_string)

        # No custom headers: the handshake relies on the query string only,
        # which also keeps native browser WebSocket usable.
        self._ws = await websockets.asyncio.client.connect(
            ws_url,
            open_timeout=10,
        )

    async def _read_loop(self) -> None:
        try:
            self._callback_failed = False
            self._safe_callback(self._listener.on_recognition_start, SpeechRecognitionResponse(
                code=0,
                message="success",
                voice_id=self._voice_id,
            ))
            if self._callback_failed:
                return

            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, ValueError) as exc:
                    # Non-terminal: the session continues, so do not finish here.
                    self._safe_on_fail(None, ASRError(ERR_READ_FAILED, "unmarshal response failed: {}".format(exc)))
                    continue

                resp = SpeechRecognitionResponse.from_dict(data)

                if resp.code != 0:
                    # Terminal: finish the lifecycle before notifying, so a
                    # stop()/write() call from inside on_fail sees the stopped
                    # state instead of waiting on the read loop.
                    self._finish()
                    self._safe_on_fail(resp, ASRError(resp.code, resp.message))
                    return

                # Check if recognition is complete before dispatching the
                # terminal response. A final=1 response can still carry
                # slice_type=2, which dispatches on_sentence_end; finish
                # first so stop()/write() from that callback observe the
                # stopped state.
                if resp.final == 1:
                    self._finish()
                    self._dispatch_event(resp)
                    if not self._callback_failed:
                        self._safe_callback(
                            self._listener.on_recognition_complete, resp, _shield=True
                        )
                    return

                # Skip the connection acknowledgement frame. After connect,
                # the server sends an ack that carries no "result" object
                # (e.g. {"code":0,"message":"success","voice_id":"v1"}).
                # Decoding such a frame into SpeechRecognitionResponse yields
                # a zero-valued Result whose slice_type=0 would otherwise be
                # misread as a slice_type=0 "sentence begin", emitting a
                # spurious on_sentence_begin. The session start is already
                # signaled via on_recognition_start at read-loop entry.
                if "result" not in data or data["result"] is None:
                    continue

                self._dispatch_event(resp)
                if self._callback_failed:
                    return
        except websockets.ConnectionClosed:
            if self._state < _State.STOPPING:
                self._finish()
                self._safe_on_fail(None, ASRError(ERR_READ_FAILED, "websocket connection closed unexpectedly"))
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._state < _State.STOPPING:
                self._finish()
                self._safe_on_fail(None, ASRError(ERR_READ_FAILED, "read message failed: {}".format(exc)))
            return
        finally:
            self._finish()
            await self._close()

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
        """Advance the recognizer to the terminal stopped state.

        It is invoked before terminal callbacks (so a stop()/write() from
        inside a callback returns immediately). The WebSocket is closed by
        the read loop's finally block via ``_close``: closing from here with
        ``create_task`` races the still-running ``async for`` and surfaces as
        ``aclose(): asynchronous generator is already running``.
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
        exception raised inside the user-supplied listener.

        A faulty non-terminal callback never crashes the host process: the
        session is finished and the error is surfaced via on_fail, matching
        the Go SDK's readLoop recover. ``_shield=True`` is used for on_fail
        and on_recognition_complete so a second exception cannot recurse.
        """
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
