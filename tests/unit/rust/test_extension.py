"""The Rust extension is present, importable, and reports the versions everything pins."""

from __future__ import annotations

from types import ModuleType

from ticket_to_ride import engine


def test_extension_imports(rust: ModuleType) -> None:
    assert rust.__name__ == "ttr_rust"


def test_the_engine_package_re_exports_the_extension(rust: ModuleType) -> None:
    """`ticket_to_ride.engine` is a thin alias, so agent, CLI and UI import paths never move."""
    assert engine.Game is rust.Game
    assert engine.State is rust.State
    assert engine.CONTRACT_VERSION == rust.contract_version() == rust.CONTRACT_VERSION
    assert rust.obs_version() == engine.OBS_VERSION
