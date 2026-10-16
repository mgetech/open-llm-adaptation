"""Bootstrap confidence intervals and the McNemar exact test."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

RESAMPLES = 1000
CONFIDENCE = 0.95
SEED = 0


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class McNemarResult:
    """`only_a` counts documents that system A got right and B got wrong, `only_b` the reverse."""

    only_a: int
    only_b: int
    p_value: float


def bootstrap_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> Interval | None:
    """Percentile interval for sum(numerator) / sum(denominator), resampling documents.

    Every rate in the eval is such a ratio: the numerator and denominator hold one value per
    document (for example 1 if the document is automated and correct, and 1 if it is
    automated). Resamples whose denominator is zero are left out. Returns None if the
    denominator of the whole sample is zero.
    """
    if len(numerator) != len(denominator):
        raise ValueError("numerator and denominator must have the same length")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if den.sum() == 0:
        return None

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(num), size=(resamples, len(num)))
    sums_num = num[picks].sum(axis=1)
    sums_den = den[picks].sum(axis=1)
    usable = sums_den != 0
    if not usable.any():
        return None
    ratios = sums_num[usable] / sums_den[usable]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(ratios, [tail, 100 - tail])
    return Interval(float(num.sum() / den.sum()), float(low), float(high))


def mcnemar_exact(a: Mapping[str, bool], b: Mapping[str, bool]) -> McNemarResult:
    """Exact McNemar test on two systems' per-document correctness, matched by document id.

    Only documents where exactly one system is correct carry information. Under the null
    hypothesis each of them is equally likely to favour either system, so the p-value is a
    two-sided binomial test with probability 0.5. Two systems that agree everywhere give 1.
    """
    if a.keys() != b.keys():
        raise ValueError("both systems must be scored on the same documents")
    only_a = sum(a[key] and not b[key] for key in a)
    only_b = sum(b[key] and not a[key] for key in a)
    discordant = only_a + only_b
    p_value = binomtest(only_a, discordant, 0.5).pvalue if discordant else 1.0
    return McNemarResult(only_a, only_b, float(p_value))
