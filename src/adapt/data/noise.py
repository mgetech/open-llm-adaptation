"""OCR-style noise for document text.

Three things go wrong, as in scanned documents: look-alike characters are swapped
(0/O, 1/l, 5/S), spaces are lost, and lines run together. The noise never joins
two numbers into one, so the amounts stay readable.
"""

import random
from dataclasses import dataclass

SWAPS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5"}


@dataclass(frozen=True)
class NoiseLevel:
    swap: float  # chance to swap each look-alike character
    drop_space: float  # chance to drop each space
    merge_lines: float  # chance to join each line to the one before


NOISE_LEVELS = {
    "low": NoiseLevel(swap=0.01, drop_space=0.03, merge_lines=0.05),
    "medium": NoiseLevel(swap=0.03, drop_space=0.08, merge_lines=0.15),
    "high": NoiseLevel(swap=0.06, drop_space=0.15, merge_lines=0.30),
}


def add_noise(text: str, level: str, rng: random.Random) -> str:
    """Return `text` with noise at the given level ("low", "medium" or "high")."""
    if level not in NOISE_LEVELS:
        raise ValueError(f"unknown noise level {level!r}")
    params = NOISE_LEVELS[level]

    lines = text.split("\n")
    merged = lines[0]
    for line in lines[1:]:
        merged += (" " if rng.random() < params.merge_lines else "\n") + line

    out: list[str] = []
    for index, char in enumerate(merged):
        if char == " ":
            between_digits = (
                0 < index < len(merged) - 1
                and merged[index - 1].isdigit()
                and merged[index + 1].isdigit()
            )
            if not between_digits and rng.random() < params.drop_space:
                continue
        elif char in SWAPS and rng.random() < params.swap:
            char = SWAPS[char]
        out.append(char)
    return "".join(out)
