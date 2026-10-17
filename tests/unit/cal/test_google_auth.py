from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import SecretStr

from stridedj.cal.google_auth import (
    CALENDAR_SCOPES,
    _installed_app_flow,
    client_config,
    interactive_login,
    missing_scopes,
)
from stridedj.config import GoogleSettings
from stridedj.errors import AuthError, ConfigError

FREEBUSY = "https://www.googleapis.com/auth/calendar.freebusy"
EVENTS = "https://www.googleapis.com/auth/calendar.events"


def settings(**overrides: Any) -> GoogleSettings:
    values: dict[str, Any] = {"client_id": "cid.apps.googleusercontent.com", "client_secret": SecretStr("csecret")}
    values.update(overrides)
    return GoogleSettings(**values)


@dataclass
class FakeCredentials:
    refresh_token: str | None = "1//refresh"
    granted_scopes: Sequence[str] | str | None = (FREEBUSY, EVENTS)


@dataclass
class FakeFlow:
    result: FakeCredentials | Exception = field(default_factory=FakeCredentials)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def run_local_server(self, **kwargs: Any) -> FakeCredentials:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FlowFactory:
    def __init__(self, flow: FakeFlow) -> None:
        self.flow = flow
        self.config: Mapping[str, Mapping[str, Any]] | None = None
        self.scopes: Sequence[str] | None = None

    def __call__(self, config: Mapping[str, Mapping[str, Any]], scopes: Sequence[str]) -> FakeFlow:
        self.config, self.scopes = config, scopes
        return self.flow


class Store:
    def __init__(self) -> None:
        self.saved: dict[str, str] = {}

    def __call__(self, key: str, value: str) -> None:
        self.saved[key] = value


def test_scopes_are_least_privilege() -> None:
    assert CALENDAR_SCOPES == (FREEBUSY, EVENTS)


def test_client_config_is_an_installed_app_with_loopback_redirect() -> None:
    assert client_config(settings()) == {
        "installed": {
            "client_id": "cid.apps.googleusercontent.com",
            "client_secret": "csecret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://127.0.0.1"],
        }
    }


@pytest.mark.parametrize(
    "overrides", [{"client_id": None}, {"client_id": ""}, {"client_secret": None}, {"client_secret": SecretStr("")}]
)
def test_client_config_requires_client_id_and_secret(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="GOOGLE_CLIENT_ID"):
        client_config(settings(**overrides))


def test_real_installed_app_flow_accepts_our_config_offline() -> None:
    flow = _installed_app_flow(client_config(settings()), CALENDAR_SCOPES)
    assert flow.client_type == "installed"  # type: ignore[attr-defined]
    assert flow.oauth2session.scope == list(CALENDAR_SCOPES)  # type: ignore[attr-defined]


def test_login_runs_the_loopback_flow_and_stores_the_refresh_token() -> None:
    factory, store = FlowFactory(FakeFlow()), Store()
    interactive_login(settings(), flow_factory=factory, store=store)
    assert factory.scopes == CALENDAR_SCOPES
    assert factory.config is not None and factory.config["installed"]["client_id"] == "cid.apps.googleusercontent.com"
    assert factory.flow.kwargs == {
        "host": "127.0.0.1",
        "port": 0,
        "prompt": "consent",
        "access_type": "offline",
        "open_browser": True,
    }
    assert store.saved == {"GOOGLE_REFRESH_TOKEN": "1//refresh"}


def test_login_without_refresh_token_fails_and_stores_nothing() -> None:
    store = Store()
    with pytest.raises(AuthError, match="no refresh token"):
        interactive_login(
            settings(), flow_factory=FlowFactory(FakeFlow(FakeCredentials(refresh_token=None))), store=store
        )
    assert store.saved == {}


def test_login_with_an_unticked_permission_fails() -> None:
    store = Store()
    partial = FakeFlow(FakeCredentials(granted_scopes=[FREEBUSY]))
    with pytest.raises(AuthError, match=r"calendar\.events"):
        interactive_login(settings(), flow_factory=FlowFactory(partial), store=store)
    assert store.saved == {}


def test_oauthlib_scope_change_warning_becomes_auth_error() -> None:
    flow = FakeFlow(Warning('Scope has changed from "a b" to "a".'))
    with pytest.raises(AuthError, match="permissions"):
        interactive_login(settings(), flow_factory=FlowFactory(flow), store=Store())


def test_login_requires_client_settings_before_opening_a_browser() -> None:
    factory = FlowFactory(FakeFlow())
    with pytest.raises(ConfigError):
        interactive_login(settings(client_id=None), flow_factory=factory, store=Store())
    assert factory.config is None


def test_missing_scopes() -> None:
    assert missing_scopes([FREEBUSY, EVENTS]) == []
    assert missing_scopes(f"{EVENTS} {FREEBUSY}") == []
    assert missing_scopes([EVENTS]) == [FREEBUSY]
    assert missing_scopes(None) == []
