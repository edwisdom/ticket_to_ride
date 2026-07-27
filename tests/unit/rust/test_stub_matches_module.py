"""The hand-written `.pyi` stub must describe the module that actually exists.

`crates/ttr-py/ttr_rust.pyi` is what makes `ty` useful for every consumer of the Rust
core. It is hand-written, so it can drift from `src/lib.rs` -- and a stub that promises a
method the module does not have turns a build failure back into a runtime one, which is
the failure mode it exists to prevent.

This compares names in both directions. Missing-from-stub is the more likely drift (add a
`#[pymethod]`, forget the stub); stubbed-but-absent is the more dangerous one.
"""

from __future__ import annotations

import ast
import pathlib
from types import ModuleType

import pytest

STUB = pathlib.Path("crates/ttr-py/ttr_rust.pyi")

#: Dunders and inherited machinery the stub deliberately does not describe.
IGNORED = {"__init__", "__repr__", "__doc__", "__module__", "__new__", "__dict__"}


def _stub_tree() -> ast.Module:
    if not STUB.exists():
        pytest.fail(f"{STUB} is missing; the Rust core would be untyped for every consumer")
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _stub_classes() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name not in IGNORED
            }
    return out


def _stub_functions() -> set[str]:
    return {n.name for n in _stub_tree().body if isinstance(n, ast.FunctionDef)}


def _public(obj: object) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


def test_every_module_function_is_stubbed(rust: ModuleType) -> None:
    actual = {
        name
        for name in _public(rust)
        if callable(getattr(rust, name)) and not isinstance(getattr(rust, name), type)
    }
    missing = actual - _stub_functions()
    assert not missing, f"module functions absent from {STUB}: {sorted(missing)}"


def test_no_stubbed_function_is_missing_from_the_module(rust: ModuleType) -> None:
    extra = _stub_functions() - _public(rust)
    assert not extra, f"{STUB} promises functions the module does not have: {sorted(extra)}"


@pytest.mark.parametrize("class_name", ["Game", "State", "VecEnv", "Rng"])
def test_class_members_agree(class_name: str, rust: ModuleType) -> None:
    stub = _stub_classes()
    assert class_name in stub, f"{STUB} does not describe {class_name}"
    actual = _public(getattr(rust, class_name))

    missing = actual - stub[class_name]
    assert not missing, f"{class_name} members absent from {STUB}: {sorted(missing)}"

    extra = stub[class_name] - actual
    assert not extra, (
        f"{STUB} promises {class_name} members that do not exist: {sorted(extra)}. "
        "This is the dangerous direction: it type checks and fails at runtime."
    )


def test_the_module_ships_py_typed(rust: ModuleType) -> None:
    """Without `py.typed`, type checkers ignore the stub entirely and say nothing."""
    assert rust.__file__ is not None, "ttr_rust has no __file__; it is not a real package"
    package = pathlib.Path(rust.__file__).parent
    assert (package / "py.typed").exists(), f"no py.typed beside {package}"
    assert (package / "__init__.pyi").exists(), f"no __init__.pyi beside {package}"
