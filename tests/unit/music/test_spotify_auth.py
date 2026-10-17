from __future__ import annotations

import socket
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from pydantic import SecretStr

from stridedj.config import SpotifySettings
from stridedj.errors import AuthError, ConfigError
from stridedj.music import spotify_auth as sa

# RFC 7636 Appendix B: the 32 random octets, the resulting verifier and its S256 challenge.
RFC_OCTETS = bytes(
    [116, 24, 223, 180, 151, 153, 224, 37, 79, 250, 96, 125, 216, 173, 187, 186,
     22, 212, 37, 77, 105, 214, 191, 240, 91, 88, 5, 88, 83, 132, 141, 121]
)  # fmt: skip
RFC_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
RFC_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_rfc7636_appendix_b_challenge() -> None:
    assert sa.code_challenge_s256(RFC_VERIFIER) == RFC_CHALLENGE


def test_rfc7636_appendix_b_verifier_from_octets() -> None:
    assert sa.generate_code_verifier(n_bytes=32, token_bytes=lambda _: RFC_OCTETS) == RFC_VERIFIER


def test_default_verifier_is_long_random_and_unreserved() -> None:
    first, second = sa.generate_code_verifier(), sa.generate_code_verifier()
    assert len(first) == 86  # 64 bytes, base64url without padding
    assert first != second
    assert set(first) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


@pytest.mark.parametrize("bad", ["short", "x" * 129, "a" * 42 + "!"])
def test_invalid_verifier_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="code_verifier"):
        sa.code_challenge_s256(bad)


def test_authorize_url_carries_pkce_state_and_least_privilege_scopes() -> None:
    url = sa.build_authorize_url(
        client_id="cid", redirect_uri="http://127.0.0.1:8888/callback", state="st", code_challenge=RFC_CHALLENGE
    )
    parts = urlsplit(url)
    params = {key: values[0] for key, values in parse_qs(parts.query).items()}
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == sa.AUTHORIZE_URL
    assert params == {
        "client_id": "cid",
        "response_type": "code",
        "redirect_uri": "http://127.0.0.1:8888/callback",
        "state": "st",
        "scope": " ".join(sa.SCOPES),
        "code_challenge_method": "S256",
        "code_challenge": RFC_CHALLENGE,
    }
    assert "playlist-modify-public" not in params["scope"]


def test_callback_returns_code_when_state_matches() -> None:
    assert sa.code_from_callback("code=abc&state=st", expected_state="st") == "abc"
    assert sa.code_from_callback({"code": "abc", "state": "st"}, expected_state="st") == "abc"


def test_callback_state_mismatch_is_rejected_before_anything_else() -> None:
    with pytest.raises(AuthError, match="state mismatch"):
        sa.code_from_callback("code=abc&state=attacker", expected_state="st")
    with pytest.raises(AuthError, match="state mismatch"):
        sa.code_from_callback("error=access_denied", expected_state="st")


def test_callback_error_and_missing_code() -> None:
    with pytest.raises(AuthError, match=r"access_denied"):
        sa.code_from_callback("error=access_denied&state=st", expected_state="st")
    with pytest.raises(AuthError, match=r"\(scriptalertxscript\)"):  # attacker-controlled text is sanitised
        sa.code_from_callback("error=<script>alert(x)</script>&state=st", expected_state="st")
    with pytest.raises(AuthError, match="no authorization code"):
        sa.code_from_callback("state=st", expected_state="st")


async def test_exchange_code_sends_verifier_and_no_secret() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "at",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "rt",
                "scope": "x",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        tokens = await sa.exchange_code(
            http, client_id="cid", code="abc", redirect_uri="http://127.0.0.1:8888/callback", code_verifier=RFC_VERIFIER
        )
    form = {key: values[0] for key, values in parse_qs(seen[0].content.decode()).items()}
    assert str(seen[0].url) == sa.TOKEN_URL
    assert form == {
        "grant_type": "authorization_code",
        "code": "abc",
        "redirect_uri": "http://127.0.0.1:8888/callback",
        "client_id": "cid",
        "code_verifier": RFC_VERIFIER,
    }
    assert "authorization" not in seen[0].headers
    assert tokens.refresh_token is not None
    assert tokens.refresh_token.get_secret_value() == "rt"
    assert "rt" not in repr(tokens)
    assert tokens.expires_in_s == 3600


async def test_refresh_with_rejected_token_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(AuthError, match="stridedj auth spotify"):
            await sa.refresh_access_token(http, client_id="cid", refresh_token=SecretStr("old"))


@pytest.mark.parametrize(
    "uri", ["http://localhost:8888/callback", "https://127.0.0.1:8888/callback", "http://example.com:8888/cb"]
)
def test_callback_server_only_accepts_loopback_http(uri: str) -> None:
    with pytest.raises(ConfigError):
        sa.CallbackServer(uri, expected_state="st")


def test_callback_server_ignores_other_paths_and_returns_code() -> None:
    server = sa.CallbackServer("http://127.0.0.1:0/callback", expected_state="st")
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    with httpx.Client(trust_env=False) as browser:
        assert browser.get(f"{base}/favicon.ico").status_code == 404
        assert browser.get(f"{base}/callback?code=abc&state=st").status_code == 200
    assert server.wait_for_code(timeout_s=5) == "abc"


def test_callback_server_rejects_state_mismatch() -> None:
    server = sa.CallbackServer("http://127.0.0.1:0/callback", expected_state="st")
    server.start()
    with httpx.Client(trust_env=False) as browser:
        assert browser.get(f"http://127.0.0.1:{server.port}/callback?code=abc&state=evil").status_code == 400
    with pytest.raises(AuthError, match="state mismatch"):
        server.wait_for_code(timeout_s=5)


def test_callback_server_times_out() -> None:
    server = sa.CallbackServer("http://127.0.0.1:0/callback", expected_state="st")
    server.start()
    with pytest.raises(AuthError, match="no login callback"):
        server.wait_for_code(timeout_s=0.05)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_interactive_login_stores_refresh_token_without_printing_it() -> None:
    settings = SpotifySettings(client_id="cid", redirect_uri=f"http://127.0.0.1:{_free_port()}/callback")
    stored: dict[str, str] = {}
    printed: list[str] = []

    def fake_browser(url: str) -> bool:  # the "user" approves: Spotify redirects back with code + state
        params = {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}
        with httpx.Client(trust_env=False) as browser:
            browser.get(params["redirect_uri"], params={"code": "the-code", "state": params["state"]})
        return True

    def token_endpoint(request: httpx.Request) -> httpx.Response:
        assert b"code=the-code" in request.content
        return httpx.Response(200, json={"access_token": "at", "expires_in": 3600, "refresh_token": "secret-rt"})

    sa.interactive_login(
        settings,
        open_browser=fake_browser,
        store=stored.__setitem__,
        notify=printed.append,
        timeout_s=5,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(token_endpoint)),
    )
    assert stored == {"SPOTIFY_REFRESH_TOKEN": "secret-rt"}
    assert not any("secret-rt" in line for line in printed)


def test_interactive_login_requires_client_id() -> None:
    with pytest.raises(ConfigError, match="SPOTIFY_CLIENT_ID"):
        sa.interactive_login(SpotifySettings(), open_browser=lambda _: True, store=lambda *_: None)
