"""One-time interactive logins (`stridedj auth garmin|spotify|google`).

Spotify and Google: OAuth in the browser; the refresh token is written into `.env`. Garmin: credentials + MFA prompt;
garminconnect keeps its own token store (0600)."""
