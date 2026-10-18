"""Lineup scheduling. No games played."""

from __future__ import annotations

import pytest

from ticket_to_ride.eval.arena import round_robin


def test_a_round_robin_lists_each_group_once() -> None:
    """Rotations are the arena's job, so a lineup is an unordered choice -- listing (a, b)
    and (b, a) separately would double the work and measure nothing new."""
    assert round_robin(5, 2) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    assert len(round_robin(5, 4)) == 5
    assert len(round_robin(4, 4)) == 1


def test_too_few_agents_for_the_seats_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        round_robin(2, 4)
