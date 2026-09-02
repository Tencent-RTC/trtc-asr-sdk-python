"""SDK self-identification carried by every request.

Every request (WebSocket handshake and HTTP API calls) reports which SDK
language, version and OS platform produced it. Without this, a customer issue
can only be traced to an AppID — not to the concrete client build that
triggered it, which is what makes cross-version regressions diagnosable.

The values travel as URL query parameters rather than headers because a
browser-originated WebSocket handshake cannot set custom headers, and the three
transports must report identically.

This module deliberately depends on the standard library only: ``__init__``
re-exports :data:`SDK_VERSION` as ``__version__``, so importing anything from
the package here would create a cycle.
"""

from __future__ import annotations

import platform
from urllib.parse import quote

# SDK_VERSION is the released version of this SDK. It is the single source of
# truth for the version: ``trtc_asr.__version__`` re-exports it, and it must be
# kept in sync with the ``version`` field in pyproject.toml.
SDK_VERSION = "1.0.0"

# SDK_LANGUAGE identifies the SDK implementation language.
SDK_LANGUAGE = "python"

# SDK_TYPE distinguishes this family of SDKs from the client-side ones. All six
# language bindings here run server-side, so the value is constant; it exists so
# server-side telemetry can bucket traffic the same way it does for the
# mobile/desktop client SDKs.
SDK_TYPE = "server"

# Normalized platform vocabulary the service expects. Keys are the lowercased
# platform.system() values; anything absent is reported verbatim so a new
# platform shows up in telemetry instead of being silently misattributed.
_PLATFORM_ALIASES = {
    "darwin": "mac",
    "windows": "windows",
    "linux": "linux",
    "android": "android",
    "ios": "ios",
    "ipados": "ios",
}


def sdk_platform() -> str:
    """Report the OS platform, normalized to windows/linux/mac/android/ios."""
    system = platform.system().lower()
    return _PLATFORM_ALIASES.get(system, system)


def sdk_report_params() -> dict:
    """Return the SDK identification parameters shared by every transport."""
    return {
        "platform": sdk_platform(),
        "sdk_lang": SDK_LANGUAGE,
        "sdk_type": SDK_TYPE,
        "version": SDK_VERSION,
    }


def sdk_report_query() -> str:
    """Return the SDK identification parameters as an encoded query fragment.

    There is no leading ``&``, for the transports that build their URL by
    string concatenation.
    """
    params = sdk_report_params()
    return "&".join(
        "{}={}".format(k, quote(str(v), safe="")) for k, v in sorted(params.items())
    )
