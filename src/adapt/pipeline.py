"""The processing pipeline shared by the eval runner and the gateway.

A document goes through checks, pseudonymisation, the model, the booking checks and a
confidence check. Any failure, and any exception, ends in `needs_review` with a reason.
Nothing on a failure path is returned as automated.
"""

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from adapt.booking import BookingLine, ReasonCode, compute_booking
from adapt.client import ModelClient, ModelError, TokenLogprob
from adapt.config import Platform
from adapt.data.pseudonymise import pseudonymise
from adapt.prompting import build_messages, system_prompt
from adapt.schema import build_output_schema

log = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 6000

# A word for the total, then an amount with cents. The amount may hold the look-alike
# characters that OCR swaps for digits (O, o, l, I, S), as the training data does.
_DIGIT = r"[\dOolIS]"
_TOTAL = re.compile(
    r"(tota[l1I]|amount due|amount payable|balance due|to pay|payable|sum)"
    rf"[^\d]{{0,40}}{_DIGIT}[\dOolIS., ]*[.,]{_DIGIT}{_DIGIT}(?!\d)",
    re.IGNORECASE,
)
_ACCOUNT_VALUE = re.compile(r'"account"\s*:\s*"([^"]*)"')

ReviewReason = (
    ReasonCode
    | Literal[
        "empty_text",
        "text_too_long",
        "no_total",
        "model_unavailable",
        "invalid_output",
        "model_flagged",
        "no_items",
        "low_confidence",
        "no_confidence",
        "internal_error",
    ]
)


@dataclass(frozen=True)
class Result:
    """What the pipeline returns. `output` is the model answer, or None if there was none."""

    request_id: str
    needs_review: bool
    reasons: list[ReviewReason]
    items: list[dict[str, Any]]
    booking_lines: list[BookingLine]
    adapter: str
    adapter_version: str | None
    confidence: float | None = None
    output: dict[str, Any] | None = field(default=None, repr=False)


def document_problems(text: Any, chart: Any, platform: Platform) -> list[ReviewReason]:
    """Step 1: the checks that need no model."""
    problems: list[ReviewReason] = []
    present = isinstance(text, str) and bool(text.strip())
    if not present:
        problems.append("empty_text")
    elif len(text) > MAX_TEXT_LENGTH:
        problems.append("text_too_long")
    if chart not in platform.config.charts:
        problems.append("unsupported_chart")
    if present and not _TOTAL.search(text):
        problems.append("no_total")
    return problems


def account_confidences(
    content: str, tokens: list[TokenLogprob], item_count: int
) -> list[float] | None:
    """The probability of each item's account value, from the log-probabilities of its tokens.

    The characters of the answer are matched to tokens by joining the token texts. Returns None
    when that does not line up, because then the confidence cannot be trusted.
    """
    if not tokens or "".join(t.token for t in tokens) != content:
        return None
    spans = [match.span(1) for match in _ACCOUNT_VALUE.finditer(content)]
    if len(spans) != item_count:
        return None
    bounds = []
    position = 0
    for t in tokens:
        bounds.append((position, position + len(t.token), t.logprob))
        position += len(t.token)
    confidences = []
    for start, end in spans:
        covered = [logprob for s, e, logprob in bounds if s < end and e > start]
        if not covered:
            return None
        confidences.append(math.exp(sum(covered)))
    return confidences


class Pipeline:
    """One platform's pipeline: its rules, its prompt, and the model that serves its adapter."""

    def __init__(
        self,
        platform: Platform,
        client: ModelClient,
        *,
        adapter: str,
        threshold: float,
        adapter_version: str | None = None,
    ) -> None:
        self.platform = platform
        self.client = client
        self.adapter = adapter
        self.adapter_version = adapter_version
        self.threshold = threshold
        self._schema = build_output_schema(platform.rules)
        self._system = system_prompt(platform.config.name, platform.rules.chart, platform.rules)

    def process(self, document_text: str, chart: str, request_id: str | None = None) -> Result:
        request_id = request_id or uuid.uuid4().hex
        try:
            return self._process(document_text, chart, request_id)
        except Exception:
            log.exception("request %s failed", request_id)
            return self._review(request_id, ["internal_error"])

    def _review(
        self,
        request_id: str,
        reasons: list[ReviewReason],
        output: dict[str, Any] | None = None,
        lines: list[BookingLine] | None = None,
        confidence: float | None = None,
    ) -> Result:
        items = output["items"] if output else []
        return Result(
            request_id=request_id,
            needs_review=bool(reasons),
            reasons=reasons,
            items=items,
            booking_lines=lines or [],
            adapter=self.adapter,
            adapter_version=self.adapter_version,
            confidence=confidence,
            output=output,
        )

    def _process(self, document_text: str, chart: str, request_id: str) -> Result:
        problems = document_problems(document_text, chart, self.platform)
        if problems:
            return self._review(request_id, problems)

        messages = build_messages(self._system, pseudonymise(document_text))
        try:
            completion = self.client.complete(self.adapter, messages, self._schema)
        except ModelError as error:
            reason = "model_unavailable" if error.kind == "unavailable" else "invalid_output"
            return self._review(request_id, [reason])

        output = completion.output
        reasons: list[ReviewReason] = []
        lines: list[BookingLine] = []
        confidence = None
        if output["needs_review"]:
            reasons.append("model_flagged")
        if output["items"]:
            booking = compute_booking(output, chart, self.platform)
            lines, reasons = booking.lines, reasons + booking.reasons
            confidence = self._confidence(completion.content, completion.tokens, output)
            if confidence is None:
                reasons.append("no_confidence")
            elif confidence < self.threshold:
                reasons.append("low_confidence")
        elif not output["needs_review"]:
            reasons.append("no_items")
        return self._review(request_id, reasons, output, lines, confidence)

    @staticmethod
    def _confidence(
        content: str, tokens: list[TokenLogprob], output: dict[str, Any]
    ) -> float | None:
        confidences = account_confidences(content, tokens, len(output["items"]))
        return min(confidences) if confidences else None
