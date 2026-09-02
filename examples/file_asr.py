"""Example: Async audio file recognition.

Usage:
    python examples/file_asr.py -f test.wav
    python examples/file_asr.py -u https://example.com/test.wav
    python examples/file_asr.py -f audio.mp3 -e 16k_zh
    python examples/file_asr.py -u https://example.com/call.wav -diarization 1
    python examples/file_asr.py -u https://example.com/call.wav -diarization 3 \
        -roles "teacher=https://example.com/teacher.wav"

Prerequisites:
    1. Get Tencent Cloud APPID: https://console.cloud.tencent.com/cam/capi
    2. Create a TRTC application: https://console.cloud.tencent.com/trtc/app
    3. Get SDKAppID and SDK secret key from the application overview page
    4. Prepare an audio file (local ≤5MB, URL ≤1GB / ≤12h)
"""

import argparse
import os
import logging
import sys
import time

from trtc_asr import Credential, SpeakerRole
from trtc_asr.file_recognizer import FileRecognizer, CreateRecTaskRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ===== Configuration =====
# Fill in your credentials before running.
APP_ID = int(os.environ.get("TRTC_ASR_APP_ID", "0"))
SDK_APP_ID = int(os.environ.get("TRTC_ASR_SDK_APP_ID", "0"))
SECRET_KEY = os.environ.get("TRTC_ASR_SECRET_KEY", "")


def main():
    parser = argparse.ArgumentParser(description="TRTC File ASR Example")
    parser.add_argument("-f", "--file", default="", help="path to local audio file (≤5MB)")
    parser.add_argument("-u", "--url", default="", help="URL of audio file (≤1GB, ≤12h)")
    parser.add_argument("-e", "--engine", default="16k_zh_en", help="engine model type")
    parser.add_argument("--res", type=int, default=1, help="result format: 0=basic, 1=detailed, 2=with punctuation timing")
    parser.add_argument("--callback", default="", help="callback URL for receiving results")
    parser.add_argument("--diarization", type=int, default=0, help="speaker diarization: 0=off, 1=cluster, 3=voiceprint roles")
    parser.add_argument("--speakers", type=int, default=0, help="expected speaker count hint (0=auto)")
    parser.add_argument("--roles", default="", help='voiceprint roles for --diarization=3: "name=https://url,name2=https://url2"')
    parser.add_argument("--vad-level", type=int, default=-1, help="VAD profile: 0=high recall, 1=far-field; negative means unset")
    parser.add_argument("--noise-threshold", type=float, default=-1.0, help="VAD noise threshold [0,4]; negative means unset")
    parser.add_argument("--poll", type=float, default=1.0, help="poll interval in seconds")
    parser.add_argument("--timeout", type=float, default=600.0, help="max wait time in seconds")
    args = parser.parse_args()

    if APP_ID == 0 or SDK_APP_ID == 0 or not SECRET_KEY:
        print(
            "Error: Please set APP_ID, SDK_APP_ID and SECRET_KEY in the code.\n\n"
            "Steps:\n"
            "  1. Get APPID from CAM Console: https://console.cloud.tencent.com/cam/capi\n"
            "  2. Open TRTC Console: https://console.cloud.tencent.com/trtc/app\n"
            "  3. Create or select an application\n"
            "  4. Copy SDKAppID and SDK secret key from the application overview\n"
            "  5. Fill in the credentials at the top of this file.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.file and not args.url:
        print(
            "Error: Please specify either -f (local file) or -u (audio URL).\n\n"
            "Examples:\n"
            "  python examples/file_asr.py -f test.wav\n"
            "  python examples/file_asr.py -u https://example.com/test.wav\n",
            file=sys.stderr,
        )
        sys.exit(1)

    credential = Credential(APP_ID, SDK_APP_ID, SECRET_KEY)
    recognizer = FileRecognizer(credential)

    common_kwargs = {}
    if args.diarization:
        common_kwargs["speaker_diarization"] = args.diarization
        common_kwargs["speaker_number"] = args.speakers
        roles = parse_roles(args.roles)
        if roles:
            common_kwargs["speaker_roles"] = roles
    if args.vad_level >= 0:
        common_kwargs["vad_level"] = args.vad_level
    if args.noise_threshold >= 0:
        common_kwargs["noise_threshold"] = args.noise_threshold

    if args.url:
        log.info("Submitting URL task: %s", args.url)
        req = CreateRecTaskRequest(
            engine_model_type=args.engine,
            channel_num=1,
            res_text_format=args.res,
            source_type=0,
            url=args.url,
            callback_url=args.callback,
            **common_kwargs
        )
        task_id = recognizer.create_task(req)
    else:
        with open(args.file, "rb") as f:
            data = f.read()
        log.info("Submitting file task: %s (%d bytes)", args.file, len(data))

        req = CreateRecTaskRequest(
            engine_model_type=args.engine,
            channel_num=1,
            res_text_format=args.res,
            callback_url=args.callback,
            **common_kwargs
        )
        task_id = recognizer.create_task_from_data_with_options(data, req)

    print("Task created: {}".format(task_id))
    print("Polling for result (interval={}s, timeout={}s)...".format(args.poll, args.timeout))

    status = recognizer.wait_for_result_with_interval(task_id, args.poll, args.timeout)

    print("\n=== Recognition Result ===")
    print("Task ID: {}".format(status.rec_task_id))
    print("Status: {}".format(status.status_str))
    print("Audio Duration: {:.2f} s".format(status.audio_duration))
    print("Result: {}".format(status.result))

    if status.result_detail:
        print("\n=== Sentence Details ===")
        for i, detail in enumerate(status.result_detail):
            speaker = ""
            if detail.speaker_role_name:
                speaker = " [{}]".format(detail.speaker_role_name)
            elif detail.speaker_id:
                speaker = " [spk{}]".format(detail.speaker_id)
            elif detail.channel_id:
                speaker = " [ch{}]".format(detail.channel_id)
            print("[{}]{} {} ({}-{} ms, speed={:.1f} words/s)".format(
                i, speaker, detail.final_sentence, detail.start_ms, detail.end_ms, detail.speech_speed))

            if detail.words:
                for j, w in enumerate(detail.words):
                    print("    [{}] {} ({}-{} ms)".format(
                        j, w.word, w.offset_start_ms, w.offset_end_ms))


def parse_roles(spec):
    """Convert "name=url,name2=url2" into voiceprint enrollment roles."""
    if not spec:
        return []
    roles = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            print("Invalid --roles entry {!r}, expected name=https://url".format(entry), file=sys.stderr)
            sys.exit(1)
        name, audio_url = entry.split("=", 1)
        roles.append(SpeakerRole(role_name=name.strip(), audio_url=audio_url.strip()))
    return roles


if __name__ == "__main__":
    main()
