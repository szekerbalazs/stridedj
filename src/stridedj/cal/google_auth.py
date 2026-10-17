"""One-time Google login (`stridedj auth google`): OAuth for installed apps, refresh token written to `.env`.

* Loopback redirect (`http://127.0.0.1:<random port>/`), the method Google recommends for desktop apps; the old
  copy-paste "out-of-band" flow is gone.
* Least privilege: `calendar.freebusy` can read availability but no event details; `calendar.events` is needed for
  the one write (`events.insert`). Neither allows sharing calendars or changing their settings.
* `access_type=offline` + `prompt=consent` makes Google return a refresh token even if the app was authorised
  before. While the OAuth consent screen is in "Testing" status, that refresh token expires after 7 days, so this
  command has to be re-run weekly until the app is published.

Docs: https://developers.google.com/identity/protocols/oauth2/native-app,
https://developers.google.com/identity/protocols/oauth2#expiration, https://developers.google.com/calendar/api/auth
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from stridedj.config import GoogleSettings
from stridedj.envfile import store_secret
from stridedj.errors import AuthError, ConfigError

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"  # noqa: S105 (an endpoint, not a secret)
LOOPBACK_HOST = "127.0.0.1"
CALENDAR_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar.freebusy",
    "https://www.googleapis.com/auth/calendar.events",
)
REFRESH_TOKEN_KEY = "GOOGLE_REFRESH_TOKEN"  # noqa: S105 (the env var name, not its value)


class OAuthCredentials(Protocol):
    """The part of `google.oauth2.credentials.Credentials` we use."""

    @property
    def refresh_token(self) -> str | None: ...

    @property
    def granted_scopes(self) -> Sequence[str] | str | None: ...


class OAuthFlow(Protocol):
    """The part of `google_auth_oauthlib.flow.InstalledAppFlow` we use."""

    def run_local_server(self, **kwargs: Any) -> OAuthCredentials: ...


FlowFactory = Callable[[Mapping[str, Mapping[str, Any]], Sequence[str]], OAuthFlow]
StoreSecret = Callable[[str, str], None]


def client_config(settings: GoogleSettings) -> dict[str, dict[str, Any]]:
    """The client config in Google's "installed app" format, built from settings instead of a JSON file on disk."""
    client_secret = settings.client_secret.get_secret_value() if settings.client_secret else ""
    if not settings.client_id or not client_secret:
        raise ConfigError(
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set first (a Desktop-app OAuth client: docs/SETUP.md §4)"
        )
    return {
        "installed": {
            "client_id": settings.client_id,
            "client_secret": client_secret,
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "redirect_uris": [f"http://{LOOPBACK_HOST}"],
        }
    }


def missing_scopes(granted: Sequence[str] | str | None, required: Sequence[str] = CALENDAR_SCOPES) -> list[str]:
    """Required scopes the user did not grant. Google's consent screen lets users untick individual permissions.

    `granted` may be a list or a space-separated string; `None` means "unknown" and is treated as nothing missing.
    """
    if granted is None:
        return []
    granted_set = set(granted.split()) if isinstance(granted, str) else set(granted)
    return [scope for scope in required if scope not in granted_set]


def interactive_login(
    settings: GoogleSettings,
    *,
    flow_factory: FlowFactory | None = None,
    store: StoreSecret = store_secret,
) -> None:
    """Open the browser for Google consent and store the resulting refresh token in `.env` (never printed)."""
    flow = (flow_factory or _installed_app_flow)(client_config(settings), CALENDAR_SCOPES)
    try:
        credentials = flow.run_local_server(
            host=LOOPBACK_HOST, port=0, prompt="consent", access_type="offline", open_browser=True
        )
    except Warning as exc:  # oauthlib raises Warning("Scope has changed ...") when a permission was unticked
        raise AuthError("google", "not all calendar permissions were granted; run it again and tick both") from exc

    missing = missing_scopes(credentials.granted_scopes)
    if missing:
        raise AuthError("google", f"permissions not granted: {', '.join(missing)}; run it again and tick all boxes")
    if not credentials.refresh_token:
        raise AuthError(
            "google", "Google returned no refresh token; remove StrideDJ at myaccount.google.com/permissions and retry"
        )
    store(REFRESH_TOKEN_KEY, credentials.refresh_token)


def _installed_app_flow(config: Mapping[str, Mapping[str, Any]], scopes: Sequence[str]) -> OAuthFlow:
    from google_auth_oauthlib.flow import InstalledAppFlow  # lazy: only the login command needs it

    flow: OAuthFlow = InstalledAppFlow.from_client_config(dict(config), scopes=list(scopes))
    return flow
