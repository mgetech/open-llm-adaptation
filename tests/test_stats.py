import random

import pytest

from adapt.eval.stats import bootstrap_ratio, mcnemar_exact


def correct(keys, wrong):
    """Per-document results where the documents in `wrong` are wrong."""
    return {key: key not in wrong for key in keys}


def test_identical_systems_give_a_p_value_of_one():
    results = correct(range(50), wrong={3, 7, 11})
    outcome = mcnemar_exact(results, dict(results))
    assert (outcome.only_a, outcome.only_b) == (0, 0)
    assert outcome.p_value == 1.0


def test_two_systems_with_the_same_errors_in_different_places_are_not_different():
    a = correct(range(20), wrong={0, 1, 2})
    b = correct(range(20), wrong={3, 4, 5})
    outcome = mcnemar_exact(a, b)
    assert (outcome.only_a, outcome.only_b) == (3, 3)
    assert outcome.p_value == 1.0


def test_the_p_value_matches_the_hand_computed_binomial_tails():
    keys = range(30)
    # 10 against 0: 2 * 0.5**10
    a = correct(keys, wrong=set())
    b = correct(keys, wrong=set(range(10)))
    assert mcnemar_exact(a, b).p_value == pytest.approx(2 * 0.5**10)
    # 3 against 0: 2 * 0.5**3
    b = correct(keys, wrong={0, 1, 2})
    assert mcnemar_exact(a, b).p_value == pytest.approx(0.25)
    # 8 against 2: 2 * (1 + 10 + 45) / 1024
    a = correct(keys, wrong=set(range(0, 2)))
    b = correct(keys, wrong=set(range(2, 10)))
    outcome = mcnemar_exact(a, b)
    assert (outcome.only_a, outcome.only_b) == (8, 2)
    assert outcome.p_value == pytest.approx(112 / 1024)


def test_the_test_does_not_care_which_system_is_first():
    a = correct(range(30), wrong=set(range(10)))
    b = correct(range(30), wrong=set())
    assert mcnemar_exact(a, b).p_value == mcnemar_exact(b, a).p_value


def test_agreement_on_wrong_documents_does_not_count():
    a = correct(range(30), wrong={0, 1, 2, 3, 4})
    b = correct(range(30), wrong={0, 1, 2, 3, 4, 5, 6})
    outcome = mcnemar_exact(a, b)
    assert (outcome.only_a, outcome.only_b) == (2, 0)
    assert outcome.p_value == pytest.approx(0.5)


def test_systems_scored_on_different_documents_are_refused():
    with pytest.raises(ValueError, match="same documents"):
        mcnemar_exact({"a": True}, {"b": True})


def flags(n, ones, seed=1):
    values = [1] * ones + [0] * (n - ones)
    random.Random(seed).shuffle(values)
    return values


def test_the_estimate_is_the_plain_ratio():
    numerator, denominator = [1, 0, 1, 0], [1, 1, 1, 0]
    interval = bootstrap_ratio(numerator, denominator)
    assert interval.estimate == pytest.approx(2 / 3)


def test_the_interval_contains_the_estimate_for_many_shapes_of_data():
    for n, ones in [(10, 5), (20, 1), (20, 19), (50, 25), (100, 90), (200, 100), (500, 7)]:
        values = flags(n, ones)
        interval = bootstrap_ratio(values, [1] * n)
        assert interval.low <= interval.estimate <= interval.high, (n, ones)


def test_the_interval_contains_the_estimate_for_a_ratio_of_subsets():
    rng = random.Random(3)
    den = [rng.random() < 0.6 for _ in range(300)]
    num = [d and rng.random() < 0.9 for d in den]
    interval = bootstrap_ratio([int(x) for x in num], [int(x) for x in den])
    assert interval.low <= interval.estimate <= interval.high


def test_the_interval_has_a_sensible_width():
    interval = bootstrap_ratio(flags(200, 100), [1] * 200)
    # The standard error is about 0.035, so a 95% interval is about 0.14 wide.
    assert interval.low < 0.5 < interval.high
    assert 0.09 < interval.high - interval.low < 0.20


def test_more_documents_give_a_narrower_interval():
    small = bootstrap_ratio(flags(50, 25), [1] * 50)
    large = bootstrap_ratio(flags(800, 400), [1] * 800)
    assert large.high - large.low < small.high - small.low


def test_all_correct_gives_a_point_interval():
    interval = bootstrap_ratio([1] * 30, [1] * 30)
    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)


def test_seed_zero_is_deterministic():
    values = flags(120, 80)
    assert bootstrap_ratio(values, [1] * 120) == bootstrap_ratio(values, [1] * 120, seed=0)
    assert bootstrap_ratio(values, [1] * 120) == bootstrap_ratio(values, [1] * 120)


def test_another_seed_gives_another_interval_with_the_same_estimate():
    values = flags(120, 80)
    first = bootstrap_ratio(values, [1] * 120, seed=0)
    second = bootstrap_ratio(values, [1] * 120, seed=1)
    assert first.estimate == second.estimate
    assert (first.low, first.high) != (second.low, second.high)


def test_the_default_is_one_thousand_resamples_at_95_percent():
    values = flags(120, 80)
    assert bootstrap_ratio(values, [1] * 120) == bootstrap_ratio(
        values, [1] * 120, resamples=1000, confidence=0.95
    )
    wide = bootstrap_ratio(values, [1] * 120, confidence=0.99)
    narrow = bootstrap_ratio(values, [1] * 120, confidence=0.80)
    default = bootstrap_ratio(values, [1] * 120)
    assert narrow.high - narrow.low < default.high - default.low < wide.high - wide.low


def test_nothing_to_divide_by_gives_none():
    assert bootstrap_ratio([0, 0, 0], [0, 0, 0]) is None
    assert bootstrap_ratio([], []) is None


def test_resamples_without_a_denominator_are_left_out():
    # One document in the denominator out of 40: most resamples miss it or hit it several times.
    den = [1] + [0] * 39
    num = [1] + [0] * 39
    interval = bootstrap_ratio(num, den)
    assert interval.estimate == 1.0
    assert interval.low == interval.high == 1.0


def test_wrong_arguments_are_refused():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ratio([1, 0], [1])
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ratio([1], [1], resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ratio([1], [1], confidence=1.0)
