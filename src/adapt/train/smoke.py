"""A tiny tokenizer and a tiny random model, built here, so a smoke run needs no download.

The chat template is the shape of the Qwen3 one for the cases training uses: a system message, a
user message, an answer, and the empty thinking block when thinking is disabled. The real
template is checked against serving by a separate test. This one only has to make the code path
run.
"""

from typing import Any

from adapt.train.config import Smoke

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n"
    "{% endif %}"
    "{% endif %}"
)


def tiny_tokenizer(texts: list[str], vocab_size: int) -> Any:
    """A byte-level BPE tokenizer trained on `texts`. The same texts give the same tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
    )


def tiny_model(tokenizer: Any, smoke: Smoke, seed: int) -> Any:
    """A random Qwen3 model of a few thousand parameters, with the vocabulary of `tokenizer`."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config

    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=smoke.hidden_size,
        intermediate_size=smoke.intermediate_size,
        num_hidden_layers=smoke.layers,
        num_attention_heads=smoke.heads,
        num_key_value_heads=smoke.kv_heads,
        head_dim=smoke.head_dim,
        max_position_embeddings=smoke.max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(config)
