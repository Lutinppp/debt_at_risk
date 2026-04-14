"""
IMF API key configuration.

Resolution order:
  1. Environment variable  IMF_API_KEY
  2. File                  ~/.imf_api_key  (one line, plain text)
  3. Interactive prompt    (falls back to no-key if stdin is not a tty)

The IMF DataMapper REST API is publicly accessible without a key for most
endpoints; supply a key to raise rate-limits or access restricted datasets.
The key is forwarded as the query parameter  ?api_key=<key>  on every request.
"""

import os
import sys
from pathlib import Path
from functools import lru_cache

_KEY_FILE = Path.home() / ".imf_api_key"
_ENV_VAR  = "IMF_API_KEY"


@lru_cache(maxsize=1)
def get_imf_api_key() -> str | None:
    """
    Return the IMF API key string, or None if the user skips.

    The result is cached so the prompt appears at most once per process.
    """
    # 1. Environment variable
    key = os.environ.get(_ENV_VAR, "").strip()
    if key:
        print(f"  [api_config] Using IMF API key from ${_ENV_VAR}.")
        return key

    # 2. Key file
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
        if key:
            print(f"  [api_config] Using IMF API key from {_KEY_FILE}.")
            return key

    # 3. Interactive prompt (only when running in a terminal)
    if sys.stdin.isatty():
        print()
        print("─" * 60)
        print("  IMF API Key Setup")
        print("  The IMF DataMapper API works without a key for public data,")
        print("  but a key raises rate limits and unlocks restricted datasets.")
        print("  Get a free key at: https://www.imf.org/en/Data")
        print("─" * 60)
        key = input("  Enter your IMF API key (or press Enter to skip): ").strip()
        print()
        if key:
            # Offer to save for future runs
            save = input(f"  Save key to {_KEY_FILE} for future runs? [y/N]: ").strip().lower()
            if save == "y":
                _KEY_FILE.write_text(key)
                _KEY_FILE.chmod(0o600)
                print(f"  Key saved to {_KEY_FILE}.")
            return key
        else:
            print("  No key provided — proceeding without authentication.")
            return None

    # Non-interactive: no key
    return None


def add_api_key(params: dict | None = None) -> dict:
    """
    Return a query-parameter dict with  api_key  appended if a key is available.

    Usage:
        resp = requests.get(url, params=add_api_key({"startPeriod": "2000"}))
    """
    key = get_imf_api_key()
    result = dict(params) if params else {}
    if key:
        result["api_key"] = key
    return result
