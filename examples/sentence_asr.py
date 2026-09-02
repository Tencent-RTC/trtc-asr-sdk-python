"""Example: One-shot sentence recognition.

Usage:
    python examples/sentence_asr.py -f test.pcm
    python examples/sentence_asr.py -f test.wav -fmt wav
    python examples/sentence_asr.py -u https://example.com/test.wav -fmt wav

Prerequisites:
    1. Get Tencent Cloud APPID: https://console.cloud.tencent.com/cam/capi
    2. Create a TRTC application: https://console.cloud.tencent.com/trtc/app
    3. Get SDKAppID and SDK secret key from the application overview page
    4. Prepare an audio file (duration <= 60s, size <= 3MB)
"""

import argparse
import os
import logging
import sys

from trtc_asr import Credential
from trtc_asr.sentence_recognizer import SentenceRecognizer, SentenceRecognitionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ===== Configuration =====
# Fill in your credentials before running.
APP_ID = int(os.environ.get("TRTC_ASR_APP_ID", "0"))
SDK_APP_ID = int(os.environ.get("TRTC_ASR_SDK_APP_ID", "0"))
SECRET_KEY = os.environ.get("TRTC_ASR_SECRET_KEY", "")


def main():
    parser = argparse.ArgumentParser(description="TRTC Sentence ASR Example")
    parser.add_argument("-f", "--file", default="", help="path to local audio file")
    parser.add_argument("-u", "--url", default="", help="URL of audio file")
    parser.add_argument("-e", "--engine", default="16k_zh_en", help="engine model type")
    parser.add_argument("-fmt", "--format", default="pcm", help="audio format (pcm, wav, mp3, ...)")
    parser.add_argument("-w", "--word-info", type=int, default=0, help="word-level timing: 0=hide, 1=show, 2=with punctuation")
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
            "  python examples/sentence_asr.py -f test.pcm\n"
            "  python examples/sentence_asr.py -f test.wav -fmt wav\n"
            "  python examples/sentence_asr.py -u https://example.com/test.wav -fmt wav\n",
            file=sys.stderr,
        )
        sys.exit(1)

    credential = Credential(APP_ID, SDK_APP_ID, SECRET_KEY)
    recognizer = SentenceRecognizer(credential)

    if args.url:
        log.info("Recognizing from URL: %s", args.url)
        result = recognizer.recognize_url(args.url, args.format, args.engine)
    else:
        with open(args.file, "rb") as f:
            data = f.read()
        log.info("Recognizing from file: %s (%d bytes)", args.file, len(data))

        if args.word_info > 0:
            req = SentenceRecognitionRequest(
                eng_service_type=args.engine,
                source_type=1,
                voice_format=args.format,
                word_info=args.word_info,
            )
            result = recognizer.recognize_data_with_options(data, req)
        else:
            result = recognizer.recognize_data(data, args.format, args.engine)

    print("Result: {}".format(result.result))
    print("Audio Duration: {} ms".format(result.audio_duration))
    print("Request ID: {}".format(result.request_id))

    if result.word_list:
        print("Word Count: {}".format(result.word_size))
        for i, w in enumerate(result.word_list):
            print("  [{}] {} ({}-{} ms)".format(i, w.word, w.start_time, w.end_time))


if __name__ == "__main__":
    main()
