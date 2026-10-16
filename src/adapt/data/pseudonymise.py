"""Replace personal identifiers in document text with typed placeholders.

This runs on the training data and, in the gateway, before the input is hashed for
the audit log. It matches IBANs, email addresses, phone numbers and 11-digit tax
IDs. Digits that OCR noise swapped for letters (0/O, 1/l, 5/S) are not recognised
in phone numbers and tax IDs, so run it on the text before noise is added.
"""

import re

# Order matters. An email is replaced before the digit patterns see its digits. A tax ID
# comes before a phone number, because a group of the tax ID can start with 0.
_PATTERNS = (
    ("<EMAIL>", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    ("<IBAN>", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("<TAX_ID>", re.compile(r"\b\d{2}(?: ?\d{3}){3}\b")),
    (
        "<PHONE>",
        re.compile(
            r"(?<![\w+(-])\+\d{1,3}[ ./-]?\(?\d{1,5}\)?(?:[ ./-]?\d{2,}){1,4}"  # +49 30 23125 123
            r"|(?<![\w+(-])\(0\d{2,4}\) ?\d{3,}(?:[ -]\d+)*"  # (030) 23125 123
            r"|(?<![\w+(-])0\d{2,4}[ /-]\d{3,}(?:[ /-]\d+)*"  # 030 23125 123
        ),
    ),
)
PLACEHOLDERS = tuple(placeholder for placeholder, _ in _PATTERNS)


def pseudonymise(text: str) -> str:
    """Return `text` with every identifier replaced by <IBAN>, <EMAIL>, <PHONE> or <TAX_ID>."""
    for placeholder, pattern in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
