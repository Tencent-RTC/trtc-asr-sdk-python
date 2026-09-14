"""Real-time speech recognition over the TRTC-ASR v3 protocol.

Usage::

    python v3_realtime_asr.py -f ../test.pcm
    python v3_realtime_asr.py -f ../test.pcm -e bigmodel -lang zh

Prerequisites:
  1. Create a TRTC application (SDKAppID + SDK secret key; v3 does not need
     the Tencent Cloud APPID)
  2. Prepare a PCM audio file (16kHz, 16bit, mono)
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from trtc_asr import v3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class MyListener(v3.SpeechRecognitionListener):
    def on_recognition_start(self, response):
        logging.info("recognition started, voice_id=%s", response.voice_id)

    def on_sentence_begin(self, response):
        logging.info("sentence begin, index=%d", response.result.index)

    def on_recognition_result_change(self, response):
        logging.info("result change, index=%d, text=%s", response.result.index, response.result.voice_text_str)

    def on_sentence_end(self, response):
        logging.info("sentence end, index=%d, text=%s", response.result.index, response.result.voice_text_str)
        for seg in response.result.speaker_segments:
            label = seg.speaker_name or "spk{}".format(seg.speaker_id)
            logging.info("  [%s] %s (%d-%d ms)", label, seg.text, seg.start_time, seg.end_time)

    def on_recognition_complete(self, response):
        logging.info("recognition complete, voice_id=%s", response.voice_id)

    def on_fail(self, response, error):
        logging.error("recognition failed: %s", error)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", default="test.pcm", help="PCM audio file")
    parser.add_argument(
        "-e", "--engine", required=True, help="engine model type, required (e.g. bigmodel)"
    )
    parser.add_argument(
        "-lang", default="", help="language hint; when omitted, the bigmodel engine uses zh"
    )
    parser.add_argument("-diarization", type=int, default=0, help="0=off 1=cluster 3=voiceprint")
    parser.add_argument("-word-info", type=int, default=0, help="word-level timestamps")
    args = parser.parse_args()

    # The bigmodel engine is best used with an explicit language; every other
    # engine falls back to server-side detection unless -lang is given.
    if not args.lang and args.engine == "bigmodel":
        args.lang = "zh"

    # Credentials come from the environment: TRTC_ASR_SDK_APP_ID and
    # TRTC_ASR_SECRET_KEY (v3 does not need the Tencent Cloud APPID).
    sdk_app_id = int(os.environ.get("TRTC_ASR_SDK_APP_ID", "0"))
    secret_key = os.environ.get("TRTC_ASR_SECRET_KEY", "")
    if not sdk_app_id or not secret_key:
        print("Set TRTC_ASR_SDK_APP_ID and TRTC_ASR_SECRET_KEY first.")
        return 1

    # v3 credentials need only SdkAppID + SecretKey.
    credential = v3.new_credential(sdk_app_id=sdk_app_id, secret_key=secret_key)
    # credential.set_site("intl")  # 国际站

    recognizer = v3.SpeechRecognizer(credential, args.engine, MyListener())
    if args.lang:
        recognizer.set_language(args.lang)
    if args.word_info:
        recognizer.set_word_info(args.word_info)
    if args.diarization:
        recognizer.set_speaker_diarization(args.diarization)

    # v3 start() waits for the server ack: auth/param errors raise here.
    await recognizer.start()

    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: open(args.file, "rb").read())
    chunk = 6400  # 200ms of 16kHz 16bit mono PCM
    for i in range(0, len(data), chunk):
        await recognizer.write(data[i : i + chunk])
        await asyncio.sleep(0.2)  # simulate real-time pacing

    await recognizer.stop()
    print("Processing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
