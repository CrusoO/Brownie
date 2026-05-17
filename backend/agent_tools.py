"""Tool executors for Brownie's agent loop."""

from __future__ import annotations

import urllib.parse
from typing import Any


def execute_web_search(query: str) -> dict[str, Any]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "ok": False,
            "query": query,
            "error": "Install duckduckgo-search to enable web search.",
        }

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        results = [
            {
                "title": hit.get("title", ""),
                "snippet": hit.get("body", ""),
                "url": hit.get("href", ""),
            }
            for hit in hits
        ]
        return {"ok": True, "query": query, "results": results}
    except Exception as exc:
        return {"ok": False, "query": query, "error": str(exc)}


def execute_play_music(query: str) -> dict[str, Any]:
    encoded = urllib.parse.quote_plus(query.strip())
    return {
        "ok": True,
        "query": query,
        "spotify_url": f"https://open.spotify.com/search/{encoded}",
        "youtube_url": f"https://www.youtube.com/results?search_query={encoded}",
        "message": f"Open Spotify or YouTube to play: {query}",
    }
