"""
Unit tests for data freshness and staleness detection.

Tests the freshness logic without requiring a Supabase connection.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from modules.supabase_cache import (
    _parse_ts,
    data_staleness_info,
)


# ── _parse_ts ────────────────────────────────────────────────────────────────

class TestParseTs:
    def test_iso_with_tz(self):
        result = _parse_ts("2026-03-27T10:21:38.894125+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.hour == 10

    def test_iso_without_tz(self):
        result = _parse_ts("2026-03-27T10:21:38")
        assert result is not None
        assert result.hour == 10

    def test_date_only(self):
        result = _parse_ts("2026-03-27")
        assert result is not None
        assert result.year == 2026
        assert result.hour == 0

    def test_none(self):
        assert _parse_ts(None) is None

    def test_garbage(self):
        assert _parse_ts("not-a-date") is None

    def test_already_datetime(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _parse_ts(dt) is dt


# ── data_staleness_info ──────────────────────────────────────────────────────

class TestStalenessInfo:
    def test_no_supabase(self):
        result = data_staleness_info({"supabase_connected": False})
        assert result["status"] == "unknown"
        assert "not connected" in result["message"].lower() or "local" in result["message"].lower()

    def test_no_data(self):
        result = data_staleness_info({
            "supabase_connected": True,
            "options_updated_at": None,
            "price_updated_at": None,
        })
        assert result["status"] == "unknown"

    def test_fresh_data_market_closed(self):
        now = datetime.now(timezone.utc)
        freshness = {
            "supabase_connected": True,
            "options_updated_at": now - timedelta(minutes=30),
            "price_updated_at": now - timedelta(minutes=30),
        }
        with patch("modules.supabase_cache.is_market_open", return_value=False):
            result = data_staleness_info(freshness)
        assert result["status"] == "fresh"
        assert not result["is_stale"]

    def test_stale_during_market_hours(self):
        now = datetime.now(timezone.utc)
        freshness = {
            "supabase_connected": True,
            "options_updated_at": now - timedelta(hours=3),
            "price_updated_at": now - timedelta(hours=3),
        }
        with patch("modules.supabase_cache.is_market_open", return_value=True):
            result = data_staleness_info(freshness)
        assert result["is_stale"]
        assert result["status"] == "stale"
        assert "market" in result["message"].lower()

    def test_old_data_market_closed(self):
        now = datetime.now(timezone.utc)
        freshness = {
            "supabase_connected": True,
            "options_updated_at": now - timedelta(hours=24),
            "price_updated_at": now - timedelta(hours=24),
        }
        with patch("modules.supabase_cache.is_market_open", return_value=False):
            result = data_staleness_info(freshness)
        assert result["is_stale"]
        assert result["status"] == "stale"

    def test_age_strings_present(self):
        now = datetime.now(timezone.utc)
        freshness = {
            "supabase_connected": True,
            "options_updated_at": now - timedelta(hours=2),
            "price_updated_at": now - timedelta(minutes=5),
        }
        with patch("modules.supabase_cache.is_market_open", return_value=False):
            result = data_staleness_info(freshness)
        assert "2h" in result["options_age_str"]
        assert "5m" in result["price_age_str"]
