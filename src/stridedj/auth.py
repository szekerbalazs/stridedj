"""One-time interactive logins: `stridedj auth garmin | spotify | google`.

* Spotify: Authorization Code + PKCE in the browser; the refresh token is written into `.env`.
* Google: installed-app OAuth (loopback redirect) in the browser; the refresh token is written into `.env`.
* Garmin: email + password from `.env` (+ an MFA code typed here if enabled); garminconnect stores its session in
  `GARMIN_TOKEN_DIR` (0600).
Nothing secret is printed.
"""

from __future__ import annotations

from enum import StrEnum

import typer

from stridedj.config import load_settings


class Service(StrEnum):
    GARMIN = "garmin"
    SPOTIFY = "spotify"
    GOOGLE = "google"


def login(service: Service) -> None:
    settings = load_settings()
    if service is Service.GARMIN:
        from stridedj.runs.sources.garmin_live import interactive_login as garmin_login

        garmin_login(settings.garmin, prompt_mfa=lambda: typer.prompt("Garmin MFA code", hide_input=True))
    elif service is Service.SPOTIFY:
        from stridedj.music.spotify_auth import interactive_login as spotify_login

        spotify_login(settings.spotify)
    else:
        from stridedj.cal.google_auth import interactive_login as google_login

        google_login(settings.google)
    typer.echo(f"✓ {service.value}: logged in. Run `stridedj doctor` to verify.")
