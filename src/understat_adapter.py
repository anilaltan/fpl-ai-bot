"""
Lightweight Understat adapter inspired by collinb9/understatAPI.

Provides synchronous helpers to fetch league players/fixtures with retry and
HTML fallback so we don't depend on the external understatapi package.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class UnderstatAdapter:
    """
    Minimal Understat client (sync).

    First tries the documented-like JSON endpoint:
        https://understat.com/getLeagueData/{league}/{season}
    Falls back to scraping JSON blobs from the HTML league page if needed.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: int = 2,
        timeout: int = 15,
        user_agent: str = "Mozilla/5.0",
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.session = requests.Session()
        self.default_headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://understat.com/",
        }
        self.ajax_headers = {
            **self.default_headers,
            "X-Requested-With": "XMLHttpRequest",
        }

    def _get_with_retry(self, url: str, headers: Dict[str, str]) -> requests.Response:
        """HTTP GET with exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                wait = self.backoff_base ** attempt
                if attempt < self.max_retries - 1:
                    logger.debug("Retry %s/%s for %s (wait %ss)", attempt + 1, self.max_retries, url, wait)
                    time.sleep(wait)
        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without exception.")

    @staticmethod
    def _extract_json_blob(html: str, var_name: str) -> Optional[Any]:
        """
        Extract and decode JSON.parse('...') blob assigned to a JS variable.
        Example pattern on understat pages: playersData = JSON.parse('...json...');
        """
        pattern = rf"{var_name}\s*=\s*JSON.parse\('([^']+)'\);"
        match = re.search(pattern, html)
        if not match:
            return None

        raw = match.group(1)
        try:
            decoded_str = bytes(raw, "utf-8").decode("unicode_escape")
            return json.loads(decoded_str)
        except Exception as exc:
            logger.debug("Failed to decode %s blob: %s", var_name, str(exc))
            return None

    def fetch_league(self, league: str, season: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch league players + fixtures (dates) for given season.
        Returns (players, fixtures). Empty lists on failure.
        """
        json_url = f"https://understat.com/getLeagueData/{league}/{season}"
        try:
            resp = self._get_with_retry(json_url, headers=self.ajax_headers)
            payload = resp.json()
            players = payload.get("players", []) if isinstance(payload, dict) else []
            fixtures = payload.get("dates", []) if isinstance(payload, dict) else []
            if players:
                return players, fixtures
        except Exception as exc:
            logger.warning("JSON endpoint failed (%s, %s): %s", league, season, str(exc))

        # HTML fallback
        html_url = f"https://understat.com/league/{league}/{season}"
        try:
            resp = self._get_with_retry(html_url, headers=self.default_headers)
            html = resp.text
            players = self._extract_json_blob(html, "playersData") or []
            fixtures = self._extract_json_blob(html, "datesData") or []
            if players:
                return players, fixtures
        except Exception as exc:
            logger.warning("HTML fallback failed (%s, %s): %s", league, season, str(exc))

        return [], []

