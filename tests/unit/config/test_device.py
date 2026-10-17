"""The `device` setting: `auto` picks the best accelerator, an absent one is an error."""

from __future__ import annotations

import pytest

from ticket_to_ride.config.device import DEVICES, resolve_device

pytestmark = pytest.mark.torch


def test_an_unknown_setting_is_rejected_before_torch_is_touched() -> None:
    with pytest.raises(ValueError, match="one of auto, cuda, mps, cpu"):
        resolve_device("tpu")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_takes_the_best_accelerator_present(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto") == expected


def test_naming_an_absent_accelerator_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="'cuda' is not available"):
        resolve_device("cuda")


def test_cpu_is_always_there() -> None:
    pytest.importorskip("torch")
    assert resolve_device("cpu") == "cpu"
    assert DEVICES[0] == "auto"
