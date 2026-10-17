"""Where tensors live: one setting, resolved in one place.

PLAN.md §14.1, Phase 3.3: training code takes its device from configuration from its first
line, so the same code runs on the M2 Max (MPS), a Modal GPU container (CUDA) and CI (CPU).
`auto` takes the best accelerator present. Naming one that is absent is an error rather
than a quiet fallback -- a run that silently trained on the CPU is a wasted run.
"""

from __future__ import annotations

from typing import Literal, get_args

Device = Literal["auto", "cuda", "mps", "cpu"]

#: Every accepted setting, `auto` first.
DEVICES: tuple[str, ...] = get_args(Device)

#: What `auto` tries, best first.
PREFERENCE = ("cuda", "mps", "cpu")


def resolve_device(setting: str = "auto") -> str:
    """The torch device for `setting`, one of `auto|cuda|mps|cpu`.

    Raises `ValueError` for an unknown setting, or for an accelerator this machine lacks.
    Imports torch, so it needs the `rl` extra.
    """
    if setting not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, not {setting!r}")
    import torch  # the `rl` extra; nothing outside training needs torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if setting == "auto":
        return next(device for device in PREFERENCE if available[device])
    if not available[setting]:
        raise ValueError(f"device {setting!r} is not available on this machine")
    return setting
