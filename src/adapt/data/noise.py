"""OCR-style noise for document text.

Six things go wrong, as in scanned documents:
- look-alike characters are swapped (0/O, 1/l, 5/S);
- spaces are lost;
- lines run together;
- letter pairs are confused (rn/m, vv/w);
- a long line is broken in two at a space;
- stray marks (|, ', _, `) appear between words.

The noise never joins two numbers into one, and the last three never touch a character next to
a digit, so the amounts stay readable.
"""

import random
from dataclasses import dataclass

SWAPS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5"}
# Letter pairs the scanner reads as another pair. Only lower-case letters, so the placeholders
# (<IBAN>, <EMAIL>, ...) are not changed by these.
CONFUSIONS = (("rn", "m"), ("m", "rn"), ("vv", "w"), ("w", "vv"))
STRAY_MARKS = ("|", "'", "_", "`")


@dataclass(frozen=True)
class NoiseLevel:
    swap: float  # chance to swap each look-alike character
    drop_space: float  # chance to drop each space
    merge_lines: float  # chance to join each line to the one before
    confuse: float  # chance to confuse each letter pair
    split_line: float  # chance to break each space between two letters into a line break
    stray: float  # chance to put a stray mark after each space before a letter


NOISE_LEVELS = {
    "low": NoiseLevel(
        swap=0.01, drop_space=0.03, merge_lines=0.05, confuse=0.01, split_line=0.005, stray=0.005
    ),
    "medium": NoiseLevel(
        swap=0.03, drop_space=0.08, merge_lines=0.15, confuse=0.03, split_line=0.015, stray=0.015
    ),
    "high": NoiseLevel(
        swap=0.06, drop_space=0.15, merge_lines=0.30, confuse=0.06, split_line=0.03, stray=0.03
    ),
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

    letters = confuse_letters(merged, params.confuse, rng)

    out: list[str] = []
    for index, char in enumerate(letters):
        before = letters[index - 1] if index > 0 else ""
        after = letters[index + 1] if index < len(letters) - 1 else ""
        if char == " ":
            if before.isdigit() and after.isdigit():
                out.append(char)
                continue
            if rng.random() < params.drop_space:
                continue
            if _is_letter(before) and _is_letter(after) and rng.random() < params.split_line:
                char = "\n"
            if char == " " and _is_letter(after) and rng.random() < params.stray:
                out.append(char)
                char = rng.choice(STRAY_MARKS) + " "
        elif char in SWAPS and rng.random() < params.swap:
            char = SWAPS[char]
        out.append(char)
    return "".join(out)


def _is_letter(char: str) -> bool:
    return char.isalpha()


def confuse_letters(text: str, chance: float, rng: random.Random) -> str:
    """Replace letter pairs by their look-alikes, never next to a digit."""
    out: list[str] = []
    index = 0
    while index < len(text):
        for pair, other in CONFUSIONS:
            end = index + len(pair)
            if text[index:end] != pair:
                continue
            before = text[index - 1] if index > 0 else ""
            after = text[end] if end < len(text) else ""
            if not before.isdigit() and not after.isdigit() and rng.random() < chance:
                out.append(other)
                index = end
                break
        else:
            out.append(text[index])
            index += 1
    return "".join(out)
