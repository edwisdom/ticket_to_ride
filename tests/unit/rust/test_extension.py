"""The Rust extension is present, importable, and speaks the same contract version.

These are the checks that make every later differential test meaningful. If the two
engines disagree about `CONTRACT_VERSION` they are not implementing the same game, and
comparing their state hashes step by step would be comparing two different games and
calling the mismatch a bug.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from conftest import rust_or_skip

from ticket_to_ride.engine.contract import CONTRACT_VERSION


def test_extension_imports(rust: ModuleType) -> None:
    assert rust.__name__ == "ttr_rust"


def test_contract_versions_agree(rust: ModuleType) -> None:
    """Drift here means every recorded replay is unreplayable by one of the two engines."""
    assert rust.contract_version() == CONTRACT_VERSION, (
        f"Rust implements contract version {rust.contract_version()}, Python implements "
        f"{CONTRACT_VERSION}. See docs/CONTRACT.md -- this is never a rebuild, it is a "
        "contract change."
    )


def test_a_missing_extension_skips_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locally, an unbuilt extension must not fail the suite."""
    # `None` in sys.modules is the documented way to make an import fail without touching
    # the filesystem: the import machinery raises rather than falling through to a finder.
    monkeypatch.setitem(sys.modules, "ttr_rust", None)
    monkeypatch.delenv("TTR_REQUIRE_RUST", raising=False)
    with pytest.raises(pytest.skip.Exception):
        rust_or_skip()


def test_require_rust_turns_that_skip_into_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard the guard, and this one earns its keep.

    A differential harness that silently skips reports green while comparing nothing --
    strictly worse than having no harness, because it manufactures confidence. CI sets
    `TTR_REQUIRE_RUST=1` so an unbuilt extension is a red build. A guard nothing exercises
    proves nothing, so both branches are tested here.
    """
    monkeypatch.setitem(sys.modules, "ttr_rust", None)
    monkeypatch.setenv("TTR_REQUIRE_RUST", "1")
    with pytest.raises(pytest.fail.Exception, match="TTR_REQUIRE_RUST"):
        rust_or_skip()
