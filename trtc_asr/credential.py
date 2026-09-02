"""Credential management for TRTC-ASR authentication."""

from trtc_asr.errors import ASRError, ERR_INVALID_PARAM

SITE_CN = "cn"
SITE_INTL = "intl"
HOST_CN = "asr.cloud-rtc.com"
HOST_INTL = "asr-intl.cloud-rtc.com"


def host_for_site(site: str) -> str:
    """Return the ASR hostname for site. Empty / cn is domestic; intl is international."""
    normalized = (site or "").strip().lower()
    if normalized in ("", SITE_CN):
        return HOST_CN
    if normalized == SITE_INTL:
        return HOST_INTL
    raise ASRError(
        ERR_INVALID_PARAM,
        'unsupported site "{}", want "{}" or "{}"'.format(site, SITE_CN, SITE_INTL),
    )


def ws_endpoint_for_site(site: str) -> str:
    return "wss://" + host_for_site(site)


def http_endpoint_for_site(site: str) -> str:
    return "https://" + host_for_site(site)


def resolve_ws_endpoint(override: str, site: str) -> str:
    if override:
        return override
    return ws_endpoint_for_site(site)


def resolve_http_endpoint(override: str, site: str) -> str:
    if override:
        return override
    return http_endpoint_for_site(site)


class Credential:
    """Holds the authentication information for the TRTC-ASR service.

    Three values are needed:
        - app_id: Tencent Cloud account APPID, from https://console.cloud.tencent.com/cam/capi
        - sdk_app_id: TRTC application ID, from https://console.cloud.tencent.com/trtc/app
        - secret_key: TRTC SDK secret key, from TRTC console > Application Overview > SDK Key

    Call :meth:`set_site` with :data:`SITE_INTL` to use the international
    cluster. The default is the China site.

    Example::

        credential = Credential(
            app_id=1300403317,
            sdk_app_id=1400188366,
            secret_key="your-sdk-secret-key",
        )
        # credential.set_site(SITE_INTL)  # 国际站；不调用则走国内站
    """

    def __init__(self, app_id: int, sdk_app_id: int, secret_key: str) -> None:
        self.app_id = app_id
        self.sdk_app_id = sdk_app_id
        self.secret_key = secret_key
        self.user_sig: str = ""
        self.site: str = ""

    def set_user_sig(self, user_sig: str) -> None:
        """Set a pre-computed UserSig. If not set, the SDK will auto-generate it."""
        self.user_sig = user_sig

    def set_site(self, site: str) -> None:
        """Select the ASR cluster: SITE_CN (default) or SITE_INTL."""
        self.site = site or ""
