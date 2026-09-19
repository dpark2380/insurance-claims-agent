"""Shared HTTP fetch helper for the ingest scripts (stdlib urllib only).

Used by both discover_sources.py (scraping insurer pages) and download.py
(downloading PDFs) so the User-Agent, timeout handling, and error reporting
stay in one place instead of drifting apart between the two scripts.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from email.message import Message

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
}


def fetch(url: str, timeout: int = 30) -> tuple[bytes, Message] | tuple[None, None]:
    """GET url, returning (raw bytes, response headers), or (None, None) on error."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ERROR fetching {url}: {type(e).__name__} {e}")
        return None, None
