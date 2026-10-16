"""The JSON schema of the model output. The vLLM server enforces it while decoding."""

from typing import Any, get_args

from adapt.config import Rules, VatRate

DOCUMENT_TYPES = ["invoice", "receipt", "credit_note"]
MAX_REASON_LENGTH = 120


def build_output_schema(rules: Rules) -> dict[str, Any]:
    """Return the output schema, with the account enum taken from the rules."""
    item = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "amount": {"type": "number"},
            "account": {"type": "string", "enum": [a.number for a in rules.accounts]},
            "vat_rate": {"type": "string", "enum": list(get_args(VatRate))},
        },
        "required": ["description", "amount", "account", "vat_rate"],
        "additionalProperties": False,
    }
    properties = {
        "supplier": {"type": "string"},
        "document_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "document_type": {"type": "string", "enum": DOCUMENT_TYPES},
        "currency": {"type": "string", "pattern": r"^[A-Z]{3}$"},
        "total": {"type": "number"},
        "items": {"type": "array", "items": item},
        "reverse_charge": {"type": "boolean"},
        "needs_review": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": MAX_REASON_LENGTH},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
