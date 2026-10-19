import random
import re
from difflib import SequenceMatcher

import pytest

from adapt.config import load_platform
from adapt.data.generate import generate_invoices, make_suppliers
from adapt.data.layouts import NUMBER_STYLES, format_money, render_document
from adapt.data.noise import NOISE_LEVELS, STRAY_MARKS, add_noise, confuse_letters

SEED = 13
CANONICAL = str.maketrans("OlS", "015")
LOOK_ALIKE = {"0": "[0O]", "1": "[1l]", "5": "[5S]"}


@pytest.fixture(scope="module")
def documents():
    rules = load_platform("tax-booking").rules
    invoices = generate_invoices(rules, make_suppliers(rules, SEED), SEED, 300)
    return [
        (invoice, render_document(invoice, random.Random(f"{SEED}/render/{n}")))
        for n, invoice in enumerate(invoices)
    ]


def similarity(documents, level):
    """Average similarity between the clean and the noisy text. Lower means more noise."""
    scores = []
    for n, (_, text) in enumerate(documents):
        noisy = add_noise(text, level, random.Random(f"{SEED}/noise/{n}"))
        scores.append(SequenceMatcher(None, text, noisy, autojunk=False).ratio())
    return sum(scores) / len(scores)


def test_three_levels_exist():
    assert list(NOISE_LEVELS) == ["low", "medium", "high"]


def test_unknown_level_is_an_error():
    with pytest.raises(ValueError, match="unknown noise level"):
        add_noise("text", "extreme", random.Random(0))


def test_noise_grows_with_the_level(documents):
    low, medium, high = (similarity(documents, level) for level in NOISE_LEVELS)
    assert 1 > low > medium > high


def test_same_rng_gives_same_noise(documents):
    for _, text in documents[:50]:
        first = add_noise(text, "high", random.Random(5))
        assert add_noise(text, "high", random.Random(5)) == first


def test_different_rng_gives_different_noise(documents):
    text = documents[0][1]
    results = {add_noise(text, "high", random.Random(n)) for n in range(20)}
    assert len(results) > 5


def undo(text):
    """Remove what the noise can change: swaps, whitespace, stray marks and letter confusions."""
    text = "".join(text.split()).translate(CANONICAL)
    text = "".join(char for char in text if char not in STRAY_MARKS)
    for pair, other in (("rn", "m"), ("vv", "w")):
        text = text.replace(pair, other)
    return text


@pytest.mark.parametrize("level", list(NOISE_LEVELS))
def test_only_the_six_kinds_of_noise_happen(documents, level):
    # Swaps are mapped back, lost spaces, merged and broken lines are ignored with the
    # whitespace, and stray marks and letter confusions are undone. What is left must be
    # the clean text.
    for n, (_, text) in enumerate(documents):
        noisy = add_noise(text, level, random.Random(n))
        assert undo(noisy) == undo(text)


def test_the_new_kinds_of_noise_do_happen_at_the_high_level(documents):
    noisy = [add_noise(text, "high", random.Random(n)) for n, (_, text) in enumerate(documents)]
    clean = [text for _, text in documents]
    assert any(
        STRAY_MARKS[0] in b and STRAY_MARKS[0] not in a for a, b in zip(clean, noisy, strict=True)
    )
    assert any(b.count("rn") > a.count("rn") for a, b in zip(clean, noisy, strict=True))
    assert any(b.count("m") < a.count("m") for a, b in zip(clean, noisy, strict=True))
    # A line is broken where a space between two letters was.
    words = " ".join(["alpha", "beta", "gamma", "delta"] * 20)
    assert any("\n" in add_noise(words, "high", random.Random(s)) for s in range(20))


def test_letter_confusions_never_touch_a_character_next_to_a_digit():
    text = "5m 3rn 9rn rn7 m1 vv2 w8 4vv"
    for seed in range(200):
        assert confuse_letters(text, 1.0, random.Random(seed)) == text


def test_letter_confusions_happen_inside_words():
    outcomes = {confuse_letters("modern", 0.5, random.Random(s)) for s in range(50)}
    assert {"modern", "rnodern", "modem"} <= outcomes
    assert {confuse_letters("wow", 0.5, random.Random(s)) for s in range(50)} >= {"wow", "vvow"}


@pytest.mark.parametrize("level", list(NOISE_LEVELS))
def test_stray_marks_and_broken_lines_leave_amounts_alone(level):
    text = "Office rent 1.234,56 EUR\nparking space rent 45,00 EUR\nTotal amount due 1.279,56 EUR"
    amounts = ["1.234,56", "45,00", "1.279,56"]
    for seed in range(300):
        noisy = add_noise(text, level, random.Random(seed)).translate(CANONICAL)
        assert all(amount in noisy for amount in amounts), noisy


@pytest.mark.parametrize("level", list(NOISE_LEVELS))
def test_numbers_are_never_joined(level):
    text = "12 34 56 78\n10\n20\n30"
    for seed in range(300):
        noisy = add_noise(text, level, random.Random(seed))
        assert len(noisy.split()) == 7, noisy


@pytest.mark.parametrize("level", list(NOISE_LEVELS))
def test_amounts_stay_readable(documents, level):
    for n, (invoice, text) in enumerate(documents):
        noisy = add_noise(text, level, random.Random(n))
        amounts = [item.amount for item in invoice.items] + [invoice.total]
        assert any(
            all(re.search(_pattern(format_money(a, style)), noisy) for a in amounts)
            for style in NUMBER_STYLES
        ), noisy


def _pattern(amount: str) -> str:
    return "".join(LOOK_ALIKE.get(char, re.escape(char)) for char in amount)


def test_lines_are_merged_more_at_higher_levels():
    text = "\n".join(f"line {n}" for n in range(40))
    kept = {
        level: sum(add_noise(text, level, random.Random(s)).count("\n") for s in range(100))
        for level in NOISE_LEVELS
    }
    assert kept["low"] > kept["medium"] > kept["high"] > 0


def test_empty_and_single_line_text():
    assert add_noise("", "high", random.Random(0)) == ""
    assert "\n" not in add_noise("one line only", "high", random.Random(0))
