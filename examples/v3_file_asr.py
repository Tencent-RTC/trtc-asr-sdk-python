#!/usr/bin/env python3
"""Async audio file recognition example over the v3 protocol
(POST /v3/create_transcription + /v3/describe_transcription).

Credentials come from environment variables:
  TRTC_ASR_SDK_APP_ID, TRTC_ASR_SECRET_KEY
(v3 does not need the Tencent Cloud APPID.)

Usage: python examples/v3_file_asr.py [-f local.wav | -u https://.../a.wav]
       [--diarization 1] [engine]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trtc_asr.v3 import (
    CreateTranscriptionRequest,
    FileRecognizer,
    new_credential,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="", help="local audio file (<=5MB)")
    parser.add_argument("-u", "--url", default="", help="audio URL (<=12h, <=1GB)")
    parser.add_argument("--diarization", type=int, default=0,
                        help="speaker diarization: 0=off, 1=cluster, 3=voiceprint roles")
    parser.add_argument("engine", nargs="?", default="16k_zh_en")
    args = parser.parse_args()

    sdk_app_id = int(os.environ.get("TRTC_ASR_SDK_APP_ID", "0"))
    secret_key = os.environ.get("TRTC_ASR_SECRET_KEY", "")
    if not sdk_app_id or not secret_key:
        raise SystemExit("Set TRTC_ASR_SDK_APP_ID and TRTC_ASR_SECRET_KEY first.")
    if bool(args.file) == bool(args.url):
        raise SystemExit("Pass exactly one of -f (local file) or -u (URL).")

    # v3 credentials need only SdkAppID + SecretKey (no Tencent Cloud APPID).
    credential = new_credential(sdk_app_id, secret_key)
    recognizer = FileRecognizer(credential)

    req = CreateTranscriptionRequest(
        engine_model_type=args.engine,
        channel_num=1,
        res_text_format=1,  # include word-level timestamps
        speaker_diarization=args.diarization,
    )
    if args.url:
        req.source_type = 0  # SOURCE_TYPE_URL
        req.url = args.url
        task_id = recognizer.create_task(req)
    else:
        with open(args.file, "rb") as f:
            task_id = recognizer.create_task_from_data_with_options(f.read(), req)
    print(f"Task created: {task_id}")

    status = recognizer.wait_for_result(task_id)
    print(f"Status: {status.status_str}  Duration: {status.audio_duration:.2f} s")
    print(f"Result: {status.result}")
    for d in status.result_detail or []:
        speaker = ""
        if d.speaker_role_name:
            speaker = f" [{d.speaker_role_name}]"
        elif d.speaker_id > 0:
            speaker = f" [spk{d.speaker_id}]"
        print(f"  [{d.start_ms:>6} - {d.end_ms:>6}]{speaker} {d.final_sentence}")


if __name__ == "__main__":
    main()
