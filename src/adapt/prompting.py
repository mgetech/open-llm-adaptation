"""The prompt the model sees. Training and serving both build it here."""

from collections.abc import Sequence

import jinja2

from adapt.config import PLATFORMS_DIR, Rules


def system_prompt(platform: str, chart: str, rules: Rules, *, few_shot: bool = False) -> str:
    """Render the platform's system prompt for `chart` with the accounts of `rules`.

    Training and the fine-tuned model use the zero-shot prompt. `few_shot` adds the three
    examples, for the baseline that is not fine-tuned."""
    template = (PLATFORMS_DIR / platform / "prompt.jinja").read_text(encoding="utf-8")
    return jinja2.Template(template, undefined=jinja2.StrictUndefined).render(
        chart=chart, accounts=rules.accounts, few_shot=few_shot
    )


def build_messages(system: str, document_text: str) -> Sequence[dict[str, str]]:
    """Zero-shot chat: the system prompt, then the document as the user message."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": document_text},
    ]
