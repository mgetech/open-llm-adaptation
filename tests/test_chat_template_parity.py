"""Training and serving must show the model the same prompt.

Serving sends chat messages with `chat_template_kwargs`, and the server renders them with the
chat template and adds the generation prompt. Training renders the same messages plus the
answer. The text of the prompt must be identical, character by character, and the answer must
follow it directly. The template writes an empty thinking block before the answer, so serving
only matches when it asks for thinking to be disabled, which the last test shows.
"""

import json

import pytest

from adapt.client import ModelClient
from adapt.config import load_platform
from adapt.data.gold import gold_path
from adapt.data.pseudonymise import pseudonymise
from adapt.data.split import build_splits
from adapt.pipeline import Pipeline, document_problems
from adapt.prompting import build_messages, system_prompt
from chat_template import render_chat
from fake_server import FakeServer

END_OF_TURN = "<|im_end|>\n"
EMPTY_THINKING = "<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def serving():
    """A pipeline whose requests are recorded by a fake server."""
    platform = load_platform("tax-booking")
    with FakeServer() as server, ModelClient(server.url, timeout=5.0) as client:
        yield server, Pipeline(platform, client, adapter="tax-booking", threshold=0.0)


def serving_request(serving, record):
    """The request body the pipeline really sends for `record`."""
    server, pipeline = serving
    server.reset()
    server.reply_with_output(record["target"])
    pipeline.process(record["document_text"], record["chart"])
    return server.requests[0].body


def serving_prompt(serving, record):
    """The prompt the server renders for that request."""
    body = serving_request(serving, record)
    return render_chat(body["messages"], add_generation_prompt=True, **body["chat_template_kwargs"])


def training_text(platform, record, **kwargs):
    """The text of one training example: the prompt messages plus the answer."""
    system = system_prompt(platform.config.name, record["chart"], platform.rules)
    messages = build_messages(system, record["document_text"])
    messages = [*messages, {"role": "assistant", "content": json.dumps(record["target"])}]
    return render_chat(messages, add_generation_prompt=False, **kwargs)


def reaches_the_model(platform, records):
    """Documents that fail the first checks never reach the model, so serving has no prompt."""
    return [r for r in records if not document_problems(r["document_text"], r["chart"], platform)]


def gold_records():
    lines = gold_path("tax-booking").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_the_renderer_matches_the_known_qwen3_format():
    messages = [{"role": "user", "content": "Hi"}]
    assert render_chat(messages, add_generation_prompt=True, enable_thinking=False) == (
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" + EMPTY_THINKING
    )
    assert render_chat(messages, add_generation_prompt=True) == (
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )
    messages = [{"role": "system", "content": "Be brief."}, *messages]
    assert render_chat(messages, add_generation_prompt=False).startswith(
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n"
    )


def test_serving_asks_for_thinking_to_be_disabled(serving):
    server, _ = serving
    serving_prompt(serving, gold_records()[0])
    assert server.requests[0].body["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_serving_prompt_ends_with_an_empty_thinking_block(serving):
    prompt = serving_prompt(serving, gold_records()[0])
    assert prompt.endswith("<|im_start|>assistant\n" + EMPTY_THINKING)


def test_a_training_example_is_the_serving_prompt_followed_by_the_answer(serving, platform):
    record = gold_records()[0]
    prompt = serving_prompt(serving, record)
    text = training_text(platform, record, enable_thinking=False)
    assert text == prompt + json.dumps(record["target"]) + END_OF_TURN


def test_the_prompts_match_for_documents_of_every_kind(serving, platform):
    records = build_splits(platform, 0)["dev"][::4] + gold_records()[::4]
    records = reaches_the_model(platform, records)
    assert len(records) > 80
    for record in records:
        prompt = serving_prompt(serving, record)
        text = training_text(platform, record, enable_thinking=False)
        assert text.startswith(prompt), record["id"]
        assert text[len(prompt) :] == json.dumps(record["target"]) + END_OF_TURN, record["id"]


def test_serving_sends_the_training_document_text_unchanged(serving, platform):
    for record in reaches_the_model(platform, build_splits(platform, 0)["dev"][::10]):
        body = serving_request(serving, record)
        assert body["messages"][1]["content"] == record["document_text"]


def test_pseudonymising_a_stored_document_changes_nothing():
    platform = load_platform("tax-booking")
    for records in build_splits(platform, 0).values():
        for record in records:
            assert pseudonymise(record["document_text"]) == record["document_text"], record["id"]
    for record in gold_records():
        assert pseudonymise(record["document_text"]) == record["document_text"], record["id"]


def test_serving_without_the_thinking_flag_would_not_match_training(serving, platform):
    record = gold_records()[0]
    body = serving_request(serving, record)
    text = training_text(platform, record, enable_thinking=False)
    flagged = render_chat(body["messages"], add_generation_prompt=True, enable_thinking=False)
    unflagged = render_chat(body["messages"], add_generation_prompt=True)
    assert text.startswith(flagged)
    # Without the flag the prompt stops earlier, and the model would have to write the
    # empty thinking block itself before the answer.
    assert text[len(unflagged) :].startswith(EMPTY_THINKING)
