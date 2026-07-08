"""Tests for the CLI configuration in code/main.py.

These invariants guard against typos in the area list and against menu entries
that point at missing scripts or drop required metadata.
"""
import re

import main


def test_all_oxford_areas_count_and_uniqueness():
    assert len(main.ALL_OXFORD_AREAS) == 17
    assert len(set(main.ALL_OXFORD_AREAS)) == 17, "area codes must be unique"


def test_all_oxford_areas_format():
    # Every area code is 'E' followed by 8 digits (MSOA 2021 codes).
    assert all(re.fullmatch(r"E\d{8}", code) for code in main.ALL_OXFORD_AREAS)


def test_all_oxford_areas_sorted():
    # The list is maintained in ascending order; E02005952 is intentionally absent.
    assert main.ALL_OXFORD_AREAS == sorted(main.ALL_OXFORD_AREAS)
    assert "E02005952" not in main.ALL_OXFORD_AREAS


def test_script_options_structure():
    for key, cfg in main.SCRIPT_OPTIONS.items():
        assert cfg.get("name"), f"option {key} missing a name"
        assert cfg.get("script", "").endswith(".py"), f"option {key} script must be a .py file"
        assert cfg.get("description"), f"option {key} missing a description"


def test_script_options_keys_are_numeric_strings():
    for key in main.SCRIPT_OPTIONS:
        assert key.isdigit()
