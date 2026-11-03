"""Minimal, rate-limited client for the OpenAQ v3 API.

Docs (checked 2026-11-03):
- API key header ``X-API-Key``: https://docs.openaq.org/using-the-api/api-key
- Rate limits 60/min, 2000/h: https://docs.openaq.org/using-the-api/rate-limits
- Parameters per endpoint: https://api.openaq.org/openapi.json
"""

import os
import time
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

from src.utils.paths import PROJECT_ROOT


class OpenAQClient:
    """Paginating GET client that respects OpenAQ's rate limits."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        min_seconds_between_requests: float,
        page_limit: int,
        max_retries: int = 5,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Copy .env.example to .env and add your OpenAQ key."
            )
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.min_interval = min_seconds_between_requests
        self.page_limit = page_limit
        self.max_retries = max_retries
        self._last_request = 0.0
        self.request_count = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET one page, sleeping as needed and retrying on 429 / 5xx errors."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            self.request_count += 1
            resp = self.session.get(url, params=params, timeout=60)

            if resp.status_code == 429 or resp.status_code >= 500:
                # Respect the server's reset hint if present, else back off exponentially.
                reset = resp.headers.get("x-ratelimit-reset")
                delay = float(reset) if reset and reset.isdigit() else 2.0 ** attempt
                time.sleep(min(delay, 120))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Giving up on {url} {params} after {self.max_retries} attempts")

    def get_all(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Yield every result across all pages of a list endpoint."""
        page = 1
        while True:
            query = {**(params or {}), "limit": self.page_limit, "page": page}
            results = self.get(path, query).get("results", [])
            yield from results
            if len(results) < self.page_limit:
                return
            page += 1
