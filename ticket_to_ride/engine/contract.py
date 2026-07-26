"""The contract version, and the one place it is defined.

`CONTRACT_VERSION` covers the three things docs/CONTRACT.md freezes: the PRNG, the draw
procedure, and the state serialization that `state_hash()` digests. Bump it and every
replay recorded under the old version becomes unreplayable -- which is the point. Replays
carry it, so an old file fails loudly instead of silently replaying a different game.

It is *not* a version for the board data (that is `DATA_HASH`), nor for the rules
(`rules_hash`), nor for the observation encoding (`OBS_VERSION`). Those move independently
and much more often.
"""

from __future__ import annotations

from typing import Final

#: Bumped only by a deliberate change to docs/CONTRACT.md plus `gen_vectors.py --write`.
CONTRACT_VERSION: Final = 1
