"""Spotify login: Authorization Code with PKCE (RFC 7636), plus the token-endpoint calls shared with the API client.

Why PKCE: a CLI/bot cannot keep a client secret (anyone with the code has it), so we prove possession of a one-time
`code_verifier` instead. The authorize request carries only its SHA-256 (`code_challenge`); the token request carries
the verifier. An attacker who intercepts the redirect gets a code that is useless without the verifier.

Why `state`: it binds the callback to the login *we* started (CSRF protection). The callback server rejects any
request whose state does not match, compared in constant time.

The callback server is one-shot and loopback-only: it binds to the redirect URI's host (127.0.0.1 by default), serves
exactly one successful callback on the exact path, never logs the query string (it carries the code), and gives up
after a timeout.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import secrets
import threading
import webbrowser
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from pydantic import Field, SecretStr

from stridedj.config import SpotifySettings
from stridedj.domain import Frozen
from stridedj.envfile import store_secret
from stridedj.errors import AuthError, ConfigError, ExternalAPIError
from stridedj.http import RetryPolicy, Sleep, build_client, request_with_retries

logger = logging.getLogger(__name__)

SERVICE = "spotify"
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"  # noqa: S105 (a URL, not a secret)
REFRESH_TOKEN_ENV_KEY = "SPOTIFY_REFRESH_TOKEN"  # noqa: S105 (the name of the variable, not its value)

# Least privilege: read the running playlists (private + collaborative), write *private* playlists only, read plays,
# saved tracks and top tracks. No public-playlist write, no playback control, no account data.
SCOPES: tuple[str, ...] = (
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "user-read-recently-played",
    "user-library-read",
    "user-top-read",
)

_VERIFIER_RE = re.compile(r"^[A-Za-z0-9\-._~]{43,128}$")  # RFC 7636 §4.1: unreserved characters, 43..128 long
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
DEFAULT_LOGIN_TIMEOUT_S = 300.0

TokenBytes = Callable[[int], bytes]


# ─────────────────────────────── pure PKCE helpers ───────────────────────────────
def _b64url(raw: bytes) -> str:
    """Base64url without padding (RFC 7636 Appendix A)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_code_verifier(*, n_bytes: int = 64, token_bytes: TokenBytes = secrets.token_bytes) -> str:
    """A high-entropy verifier: `n_bytes` random bytes, base64url-encoded (64 bytes → 86 characters)."""
    verifier = _b64url(token_bytes(n_bytes))
    if not _VERIFIER_RE.fullmatch(verifier):
        raise ValueError("code_verifier must be 43..128 unreserved characters; use 32..96 random bytes")
    return verifier


def code_challenge_s256(code_verifier: str) -> str:
    """`BASE64URL(SHA256(ASCII(code_verifier)))`, the S256 method of RFC 7636 §4.2."""
    if not _VERIFIER_RE.fullmatch(code_verifier):
        raise ValueError("code_verifier must be 43..128 unreserved characters")
    return _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())


def generate_state(*, token_bytes: TokenBytes = secrets.token_bytes) -> str:
    return _b64url(token_bytes(32))


def build_authorize_url(
    *, client_id: str, redirect_uri: str, state: str, code_challenge: str, scopes: tuple[str, ...] = SCOPES
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": " ".join(scopes),
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def code_from_callback(query: str | Mapping[str, str], *, expected_state: str) -> str:
    """Validate the redirect's query and return the authorization code.

    Raises `AuthError` if the state does not match (checked first: an attacker-initiated callback must not even be
    read further), if the user denied access, or if no code is present.
    """
    params = _first_values(query) if isinstance(query, str) else dict(query)
    state = params.get("state", "")
    if not secrets.compare_digest(state.encode(), expected_state.encode()):
        raise AuthError(SERVICE, "login callback rejected: state mismatch (possible CSRF); run the login again")
    if "error" in params:
        reason = re.sub(r"[^a-z_]", "", params["error"].lower())[:40]  # echo a sanitised reason only
        raise AuthError(SERVICE, f"login was not completed ({reason or 'unknown error'})")
    code = params.get("code")
    if not code:
        raise AuthError(SERVICE, "login callback carried no authorization code")
    return code


def _first_values(query: str) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items()}


# ─────────────────────────────── token endpoint ───────────────────────────────
class TokenResponse(Frozen):
    """The allowlisted part of a token-endpoint response."""

    access_token: SecretStr
    token_type: str
    expires_in_s: int = Field(gt=0)
    scope: str = ""
    refresh_token: SecretStr | None = Field(
        default=None, description="present on login; on refresh only when Spotify rotates it"
    )


def parse_token_response(payload: Mapping[str, Any]) -> TokenResponse:
    refresh = payload.get("refresh_token")
    return TokenResponse(
        access_token=SecretStr(str(payload["access_token"])),
        token_type=str(payload.get("token_type", "Bearer")),
        expires_in_s=int(payload.get("expires_in", 3600)),
        scope=str(payload.get("scope", "")),
        refresh_token=SecretStr(str(refresh)) if refresh else None,
    )


async def exchange_code(
    http: httpx.AsyncClient, *, client_id: str, code: str, redirect_uri: str, code_verifier: str
) -> TokenResponse:
    """Authorization code → tokens. Not retried: a code is single-use, so a retry after a lost response would fail."""
    response = await request_with_retries(
        http,
        "POST",
        TOKEN_URL,
        service=SERVICE,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        },
    )
    return parse_token_response(response.json())


async def refresh_access_token(
    http: httpx.AsyncClient,
    *,
    client_id: str,
    refresh_token: SecretStr,
    policy: RetryPolicy | None = None,
    sleep: Sleep = asyncio.sleep,
) -> TokenResponse:
    """Refresh token → new access token (and maybe a rotated refresh token). PKCE refreshes need `client_id` only.

    A 400 from the token endpoint means the refresh token was revoked or expired (`invalid_grant`): that is an
    authentication problem the user fixes by logging in again, so it is raised as `AuthError`.
    """
    try:
        response = await request_with_retries(
            http,
            "POST",
            TOKEN_URL,
            service=SERVICE,
            retry_unsafe=True,  # 429/5xx: re-sending a refresh is harmless
            policy=policy,
            sleep=sleep,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token.get_secret_value(),
                "client_id": client_id,
            },
        )
    except ExternalAPIError as exc:
        if exc.status_code == 400:
            raise AuthError(SERVICE, "refresh token rejected: run `stridedj auth spotify`", status_code=400) from None
        raise
    return parse_token_response(response.json())


# ─────────────────────────────── one-shot loopback callback server ───────────────────────────────
class _CallbackResult:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.code: str | None = None
        self.error: AuthError | None = None


_SUCCESS_HTML = b"<html><body><h3>StrideDJ: Spotify login complete. You can close this tab.</h3></body></html>"
_FAILURE_HTML = b"<html><body><h3>StrideDJ: Spotify login failed. Check the terminal.</h3></body></html>"


class CallbackServer:
    """Receives exactly one OAuth redirect on the loopback interface.

    Requests to other paths (e.g. the browser's `/favicon.ico`) get a 404 and do not end the wait.
    """

    def __init__(self, redirect_uri: str, *, expected_state: str) -> None:
        parts = urlsplit(redirect_uri)
        host = parts.hostname or ""
        if parts.scheme != "http" or host not in _LOOPBACK_HOSTS or parts.port is None:
            raise ConfigError(
                "SPOTIFY_REDIRECT_URI must be http://127.0.0.1:<port>/<path> (loopback IP; Spotify rejects 'localhost')"
            )
        self._host = host
        self._requested_port = parts.port
        self._path = parts.path or "/"
        self._expected_state = expected_state
        self._result = _CallbackResult()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port (differs from the configured one only when it was 0, which tests use)."""
        if self._server is None:
            raise RuntimeError("server not started")
        return int(self._server.server_address[1])

    def start(self) -> None:
        handler = self._handler_class()
        self._server = HTTPServer((self._host, self._requested_port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="spotify-callback", daemon=True)
        self._thread.start()

    def wait_for_code(self, *, timeout_s: float = DEFAULT_LOGIN_TIMEOUT_S) -> str:
        try:
            if not self._result.done.wait(timeout_s):
                raise AuthError(SERVICE, f"no login callback within {timeout_s:.0f} s; run the login again")
            if self._result.error is not None:
                raise self._result.error
            if self._result.code is None:  # unreachable: `done` is set only after code or error
                raise AuthError(SERVICE, "login callback carried no authorization code")
            return self._result.code
        finally:
            self.close()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        expected_path, expected_state, result = self._path, self._expected_state, self._result

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parts = urlsplit(self.path)
                if parts.path != expected_path or result.done.is_set():
                    self._reply(404, b"not found")
                    return
                try:
                    result.code = code_from_callback(parts.query, expected_state=expected_state)
                    self._reply(200, _SUCCESS_HTML)
                except AuthError as exc:
                    result.error = exc
                    self._reply(400, _FAILURE_HTML)
                result.done.set()

            def _reply(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                """Silenced: the default access log would print the query string, which carries the code."""

        return Handler


# ─────────────────────────────── the interactive login (`stridedj auth spotify`) ───────────────────────────────
def interactive_login(
    settings: SpotifySettings,
    *,
    open_browser: Callable[[str], object] = webbrowser.open,
    store: Callable[[str, str], None] = store_secret,
    notify: Callable[[str], None] = print,
    timeout_s: float = DEFAULT_LOGIN_TIMEOUT_S,
    http_factory: Callable[[], httpx.AsyncClient] = build_client,
) -> None:
    """Open the browser for consent, catch the redirect, exchange the code, and store the refresh token in `.env`.

    The token value is never printed. Raises `ConfigError` if the client id is missing and `AuthError` on any
    login failure (denied, state mismatch, timeout, rejected exchange).
    """
    if not settings.client_id:
        raise ConfigError("SPOTIFY_CLIENT_ID is not set (docs/SETUP.md §3)")
    verifier = generate_code_verifier()
    state = generate_state()
    url = build_authorize_url(
        client_id=settings.client_id,
        redirect_uri=settings.redirect_uri,
        state=state,
        code_challenge=code_challenge_s256(verifier),
    )
    server = CallbackServer(settings.redirect_uri, expected_state=state)
    server.start()  # listen *before* the browser can redirect
    notify(f"Opening Spotify in your browser. If nothing happens, open this URL:\n{url}")
    open_browser(url)
    code = server.wait_for_code(timeout_s=timeout_s)
    tokens = asyncio.run(_exchange(http_factory, settings.client_id, code, settings.redirect_uri, verifier))
    if tokens.refresh_token is None:
        raise AuthError(SERVICE, "token response carried no refresh token")
    store(REFRESH_TOKEN_ENV_KEY, tokens.refresh_token.get_secret_value())
    notify("Spotify login complete: refresh token stored in .env.")


async def _exchange(
    http_factory: Callable[[], httpx.AsyncClient], client_id: str, code: str, redirect_uri: str, verifier: str
) -> TokenResponse:
    async with http_factory() as http:
        return await exchange_code(
            http, client_id=client_id, code=code, redirect_uri=redirect_uri, code_verifier=verifier
        )
