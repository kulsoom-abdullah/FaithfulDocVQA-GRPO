"""Pin the paired exact two-sided binomial against every p-value already committed.

Why this test exists. The first implementation compared point masses with an
ABSOLUTE epsilon (`p/total <= obs + 1e-12`). For moderate p-values that is
harmless, and it reproduced four of the five committed figures exactly. At
b=144, c=0 the observed mass is 2^-144 ≈ 5.5e-44, far below the epsilon, so the
condition degenerated to `p/total <= 1e-12` and swept in most of the
distribution — returning **9.93e-13** where the true value is
**2^-143 = 8.97e-44**. Wrong by 31 orders of magnitude, in the direction that
makes a result look less significant than it is, and invisible unless a test
carries an extreme case.

The fix compares exact integer binomial weights, so no float rounding enters.
The extreme case below is the one that matters; the moderate ones guard against
a "fix" that breaks the ordinary path.

Sources for the expected values:
  - run3_test_results.md — seed ck-642 -> ck-1296, and ck-1296 vs run-2 ck-150
  - run3_checkpoint_selection.md §3c — the three ck-648/ck-1296 partitions
"""

from __future__ import annotations

import pytest

from scripts.score_run4a_test import exact_binomial_two_sided as p2

# (label, b, c, committed p, source)
COMMITTED = [
    ("seed ck-642 -> run-3 ck-1296", 144, 0, 8.97e-44, "run3_test_results.md"),
    ("run-3 ck-1296 vs run-2 ck-150", 14, 6, 0.115, "run3_test_results.md"),
    ("ck-648 vs ck-1296, answerable", 14, 9, 0.4049, "run3_checkpoint_selection.md §3c"),
    ("ck-648 vs ck-1296, abstain", 1, 1, 1.0000, "run3_checkpoint_selection.md §3c"),
    ("ck-648 vs ck-1296, all 753", 15, 10, 0.4244, "run3_checkpoint_selection.md §3c"),
]


@pytest.mark.parametrize("label,b,c,expected,source", COMMITTED)
def test_reproduces_committed_p_value(label, b, c, expected, source):
    """Every committed p-value must come back out of the implementation."""
    got = p2(b, c)
    assert got == pytest.approx(expected, rel=0.02), (
        f"{label} ({source}): b={b} c={c} -> {got:.4g}, committed {expected:.4g}")


def test_extreme_tail_is_exact_not_swamped():
    """The regression case: b=144, c=0 is exactly 2*2^-144 = 2^-143.

    An absolute-epsilon comparison returns ~9.9e-13 here. Assert the true value
    directly rather than only via the committed rounding.
    """
    assert p2(144, 0) == pytest.approx(2.0 ** -143, rel=1e-9)


def test_symmetry_and_bounds():
    """p is symmetric in (b, c) and always a probability."""
    for b, c in [(0, 0), (1, 0), (5, 3), (30, 30), (100, 7), (200, 0)]:
        f, r = p2(b, c), p2(c, b)
        assert f == pytest.approx(r), f"asymmetric at b={b} c={c}: {f} vs {r}"
        assert 0.0 < f <= 1.0, f"out of range at b={b} c={c}: {f}"


def test_no_discordant_pairs_is_one():
    """b=c=0 means the two models never disagreed — p must be 1.0, not a crash."""
    assert p2(0, 0) == 1.0


def test_even_split_is_one():
    """A perfectly even split cannot be evidence of a difference."""
    for n in (2, 10, 50):
        assert p2(n, n) == pytest.approx(1.0)


def test_monotone_in_imbalance():
    """Holding the discordant total fixed, more imbalance must not raise p."""
    n = 40
    ps = [p2(b, n - b) for b in range(n // 2, n + 1)]
    for earlier, later in zip(ps, ps[1:]):
        assert later <= earlier + 1e-12, f"p rose with imbalance: {ps}"
