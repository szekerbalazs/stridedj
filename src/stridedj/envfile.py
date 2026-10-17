"""Write values into `.env` (used by the one-time login commands to store refresh tokens).

The file is created with mode 0600 if missing. Values are never echoed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import set_key

from stridedj.config import ENV_FILE


def store_secret(key: str, value: str, *, env_file: Path = ENV_FILE) -> None:
    """Set `key=value` in the env file (adds or replaces the line) and keep the file private (0600)."""
    if not env_file.exists():
        env_file.touch(mode=0o600)
    set_key(str(env_file), key, value, quote_mode="never")
    os.chmod(env_file, 0o600)  # noqa: PTH101
