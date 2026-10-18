"""Render a chat with the Qwen3 chat template, without the tokenizer library.

`fixtures/qwen3_chat_template.jinja` is the chat template of the Qwen3 models (Apache-2.0).
The 1.7B and 4B models ship the same template. It is rendered the way the tokenizer
library renders it: sandboxed, with trim_blocks and lstrip_blocks.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3_chat_template.jinja"


def _raise(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _template() -> jinja2.Template:
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    environment.globals["raise_exception"] = _raise
    return environment.from_string(TEMPLATE.read_text(encoding="utf-8"))


_COMPILED = _template()


def render_chat(
    messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool, **kwargs: Any
) -> str:
    """The text the model sees. `kwargs` are template variables such as enable_thinking."""
    return _COMPILED.render(
        messages=list(messages), add_generation_prompt=add_generation_prompt, **kwargs
    )
