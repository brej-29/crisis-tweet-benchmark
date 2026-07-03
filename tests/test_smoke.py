"""Smoke test: confirms the dtc package installs and imports cleanly."""

import dtc


def test_package_imports():
    assert dtc is not None
