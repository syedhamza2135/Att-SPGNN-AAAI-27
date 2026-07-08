"""Tests for the column-name parser in data/sort_crosstable_columns.py."""
import sort_crosstable_columns as scc


def test_parse_individual_format():
    assert scc.parse_column_name("M 0_4 C") == ("M", "0_4", "C")
    assert scc.parse_column_name("F 25_29 Single") == ("F", "25_29", "Single")


def test_parse_household_format():
    assert scc.parse_column_name("1PE W1") == ("1PE", "W1", None)
    assert scc.parse_column_name("1FM-0C C") == ("1FM-0C", "C", None)


def test_parse_unrecognised_returns_all_none():
    assert scc.parse_column_name("") == (None, None, None)
    assert scc.parse_column_name("solo") == (None, None, None)


def test_parse_is_whitespace_tolerant():
    assert scc.parse_column_name("  M   0_4   C  ") == ("M", "0_4", "C")
