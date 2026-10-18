"""Turn the training records into tokens, with the loss on the answer only.

The prompt is the one serving builds: the zero-shot system prompt and the document, rendered
with the chat template of the model, with the generation prompt at the end. The answer is the
target as JSON, followed by the end token. The prompt and the answer are tokenized apart, as the
server does when it tokenizes the prompt and the model then writes the answer. The labels of
the prompt tokens are -100, so only the answer and the end token are learned.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adapt.config import Rules
from adapt.eval.runner import DATA_DIR, split_path
from adapt.prompting import build_messages, system_prompt

IGNORE = -100


def load_train_records(
    platform: str, data_dir: Path = DATA_DIR, data_round: int | None = None
) -> list[dict[str, Any]]:
    path = split_path(platform, "train", data_dir, data_round)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def render_prompt(tokenizer: Any, system: str, document_text: str, enable_thinking: bool) -> str:
    """The text the model sees before it answers."""
    return tokenizer.apply_chat_template(
        build_messages(system, document_text),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def answer_text(tokenizer: Any, record: dict[str, Any]) -> str:
    return json.dumps(record["target"]) + tokenizer.eos_token


def encode_example(
    tokenizer: Any,
    system: str,
    record: dict[str, Any],
    *,
    enable_thinking: bool,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(
        render_prompt(tokenizer, system, record["document_text"], enable_thinking),
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(answer_text(tokenizer, record), add_special_tokens=False)["input_ids"]
    length = len(prompt_ids) + len(answer_ids)
    if length > max_length:
        # A cut answer would teach the model to stop in the middle of the JSON.
        raise ValueError(f"{record['id']} has {length} tokens, the limit is {max_length}")
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE] * len(prompt_ids) + answer_ids,
    }


def encode_records(
    tokenizer: Any,
    platform: str,
    rules: Rules,
    records: Sequence[dict[str, Any]],
    *,
    enable_thinking: bool,
    max_length: int,
) -> list[dict[str, list[int]]]:
    systems: dict[str, str] = {}
    examples = []
    for record in records:
        chart = record["chart"]
        if chart not in systems:
            systems[chart] = system_prompt(platform, chart, rules)
        examples.append(
            encode_example(
                tokenizer,
                systems[chart],
                record,
                enable_thinking=enable_thinking,
                max_length=max_length,
            )
        )
    return examples
