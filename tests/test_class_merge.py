"""Tests for class merging (thin-pole classes have near-identical visual form)."""

from railintrude.dataset import merge_classes


def test_merge_classes_single_group():
    assert merge_classes("catenary_pole,signal_pole>pole") == {
        "catenary_pole": "pole",
        "signal_pole": "pole",
    }


def test_merge_classes_multiple_groups():
    spec = "catenary_pole,signal_pole>pole;animal,crotchet>misc"
    assert merge_classes(spec) == {
        "catenary_pole": "pole",
        "signal_pole": "pole",
        "animal": "misc",
        "crotchet": "misc",
    }


def test_merge_classes_empty_spec_returns_empty_mapping():
    assert merge_classes("") == {}
    assert merge_classes("  ") == {}


def test_merge_classes_ignores_blank_entries():
    assert merge_classes("catenary_pole , signal_pole > pole ;") == {
        "catenary_pole": "pole",
        "signal_pole": "pole",
    }
