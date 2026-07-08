"""Tests for the time helpers in code/utils/analyzeOutputResults.py."""
import numpy as np

import analyzeOutputResults as aor


def test_parse_hms_format():
    assert aor.parse_time_to_seconds("0:01:04") == 64
    assert aor.parse_time_to_seconds("1:23:45") == 5025


def test_parse_mm_ss_colon_format():
    assert aor.parse_time_to_seconds("2:30") == 150


def test_parse_minutes_seconds_format():
    assert aor.parse_time_to_seconds("5m 33s") == 333
    assert aor.parse_time_to_seconds("47s") == 47


def test_parse_missing_values_return_none():
    assert aor.parse_time_to_seconds("") is None
    assert aor.parse_time_to_seconds(np.nan) is None


def test_format_time_ranges():
    assert aor.format_time(47) == "47s"
    assert aor.format_time(64) == "1m 04s"
    assert aor.format_time(3661) == "1h 01m 01s"


def test_format_time_handles_missing():
    assert aor.format_time(None) == "N/A"
    assert aor.format_time(np.nan) == "N/A"
