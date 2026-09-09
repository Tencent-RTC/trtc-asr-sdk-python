#!/usr/bin/env python3
"""Sentence (one-shot) speech recognition example over the v3 protocol
(POST /v3/transcribe).

Recognizes a local audio file (<=60s, <=3MB).

Credentials come from environment variables:
  TRTC_ASR_SDK_APP_ID, TRTC_ASR_SECRET_KEY
(v3 does not need the Tencent Cloud APPID.)

Prerequisite: the server has enabled the EnableV3Route gray switch for
your SDKAppID, otherwise requests fail with 404/4001.

Usage: python examples/v3_sentence_asr.py [-f audio.pcm] [--format pcm]
       [--word-info 1] [engine]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trtc_asr.v3 import SentenceRecognizer, TranscribeRequest, new_credential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default=os.path.join(
        os.path.dirname(__file__), "test.pcm"))
    parser.add_argument("--format", default="pcm",
                        help="wav|pcm|ogg-opus|mp3|m4a")
    parser.add_argument("--word-info", type=int, default=0,
                        help="word-level timestamps: 0=off, 1=on, 2=with punctuation")
    parser.add_argument("engine", nargs="?", default="16k_zh_en")
    args = parser.parse_args()

    sdk_app_id = int(os.environ.get("TRTC_ASR_SDK_APP_ID", "0"))
    secret_key = os.environ.get("TRTC_ASR_SECRET_KEY", "")
    if not sdk_app_id or not secret_key:
        raise SystemExit("Set TRTC_ASR_SDK_APP_ID and TRTC_ASR_SECRET_KEY first.")

    # v3 credentials need only SdkAppID + SecretKey (no Tencent Cloud APPID).
    credential = new_credential(sdk_app_id, secret_key)
    recognizer = SentenceRecognizer(credential)

    with open(args.file, "rb") as f:
        data = f.read()

    if args.word_info:
        resp = recognizer.recognize_data_with_options(
            data,
            TranscribeRequest(
                engine_model_type=args.engine,
                voice_format=args.format,
                word_info=args.word_info,
            ),
        )
    else:
        resp = recognizer.recognize_data(data, args.format, args.engine)

    print(f"Result: {resp.result}")
    print(f"Duration: {resp.audio_duration} ms  RequestId: {resp.request_id}")
    for w in resp.word_list or []:
        print(f"  [{w.start_time:>6} - {w.end_time:>6}] {w.word}")


if __name__ == "__main__":
    main()
