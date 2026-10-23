---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
license: cc-by-nc-4.0
language:
  - en
tags:
  - lora
  - qlora
  - peft
  - vllm
  - bookkeeping
  - invoices
  - synthetic-data
---

# Qwen3-4B adapter for tax booking (SKR03)

A LoRA adapter for [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). It reads the text of an invoice, credit note or receipt and answers with one JSON object: supplier, date, total, and the booking lines (account of the SKR03 chart, VAT rate of 19, 7 or 0), and a flag that says the document needs a person to look at it. It is trained to set that flag when it is not sure instead of guessing.

The adapter is one part of a system. The numbers below are for the adapter together with the pipeline rules and the confidence threshold of the repository [open-llm-adaptation](https://github.com/mgetech/open-llm-adaptation). The adapter alone does not reach them, and I have not measured it alone.

## Intended use

For research and demonstration only. Not for commercial use.

- A demonstration of a fine-tuning and evaluation pipeline: a small open-weight model fine-tuned for a structured task and checked with a gate.
- The booking behaviour it demonstrates: suggestions that a person approves when the system flags them, and the system never books a flagged document by itself.

Not for:

- Tax advice. The chart of accounts and the rules are an illustrative subset of SKR03, written for this project.
- Documents of a real client without review. I tested on synthetic documents and on small sets I wrote myself, never on real client data.
- Other charts of accounts, other countries' VAT, or languages other than English with German terms. The pipeline sends any other chart to review.

## Base model and licence

The base model is Qwen/Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c`, licensed under Apache 2.0. That licence is the base model's. The adapter weights (`adapter_model.safetensors` and `adapter_config.json`) are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): you may share and adapt them with credit and for non-commercial purposes. The code of the repository stays under Apache 2.0.

The adapter was trained with `enable_thinking` off, and serving must use the same chat template setting.

## Files and checksum

| file | sha256 |
|---|---|
| `adapter_model.safetensors` | `9b876add2e053167207e22cef9555043db6790faa88396a39a4602ad19221331` |
| `adapter_config.json` | `fd1d335f9142fc1712f1210fa76815cc8f6e59ca9a0a99ac0bd89398315aaf1a` |

The repository identifies the adapter by one hash over the names and contents of these two files, `2de6c7087c2b1abe1c48f21c14d247d24d9a752c06facb96130361f83194478c`, and the launcher refuses to start, and the gateway reports not ready, when it differs.

## How to use it

Serve the base model with the adapter as a LoRA module (vLLM 0.30.0 was used):

```
vllm serve Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name qwen3-4b --dtype half --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 --lora-modules tax-booking=<path to this adapter>
```

Then send the prompt of `platforms/tax-booking/prompt.jinja` in the repository (the adapter was trained with it, its hash is `2f3789f30dd9234c912df016d6c5687b85920d701c0bdf5d4dffe1f1e13edcb7`) and request the output with the JSON schema of the platform. The repository has the pipeline that applies the rules after the model and the gateway that serves it.

## Training

- **Method:** QLoRA with supervised fine-tuning. 4-bit NF4 with double quantisation, float16 compute (a T4 has no bf16). LoRA rank 16, alpha 32, dropout 0.05, on all linear layers (33,030,144 trainable parameters).
- **Optimiser:** learning rate 2e-4 with a cosine schedule and 3% warm-up, 8-bit paged AdamW, effective batch size 16, one epoch (375 steps). Seed 0. One run; I trained no other seed of this size on this data.
- **Hardware:** two Tesla T4 GPUs, 3.6 GPU hours (6,517 seconds).
- **Checkpoint:** a checkpoint every 75 steps. I scored steps 150, 225, 300 and 375 on `dev_real` with the whole pipeline and kept step 375, the one with the lowest critical-error rate (0.053 on `dev_real` without a confidence threshold, 0.122 at step 150).
- **Versions:** transformers 5.20.0, trl 1.16.0, peft 0.21.2, bitsandbytes 0.50.2, torch 2.13.0.

## Training data

6,000 synthetic documents (round 3 of the repository), all made by code from a seed: the suppliers are invented, the amounts and the labels come from the generator's rules, and the data contains no real client data. The accounts and VAT rates are the generator's reading of an illustrative subset of SKR03, so the labels are not checked by a tax expert.

2,942 of the 6,000 documents were then rewritten by language models run outside the repository, to make the wording and layout less regular. The labels did not change, and a check rejected any rewrite in which an amount, total, rate or item name changed. The repository does not name the rewriting models, and I did not check whether the terms of their providers limit the use of their output to train other models. Check this before you reuse the adapter or the data in a product.

## Results

The labels are the same as in the repository's README. The pipeline rules include the check for orders in the text and the check of the VAT printed on the document, and the confidence threshold is 0.9817.

**Validation, not a test.** I chose the threshold on `dev` and `dev_real`, and I wrote some of the checks after I read errors on `dev`. On these two sets the configuration passes every check they can show:

| | dev | dev_real |
|---|---|---|
| automation rate (>= 0.6) | 0.761 | 0.649 |
| accuracy on automated documents (>= 0.95) | 0.995 | 0.960 |
| critical-error rate (<= 0.005) | 0.0049 (1 of 205 automated) | 0.000 (0 of 50) |

**Held-out: it did not pass the gate.** In the round 3 run, on the round 3 test sets and `gold_v2`, it passed 7 of 10 checks. It beat the few-shot baseline of the same size (p < 0.001) and the generalisation gap was 0.027 (limit 0.05). It failed accuracy on automated documents (0.938 against 0.95), the critical-error rate (0.043 against 0.005) and prompt-injection success. I found the causes afterwards (documents with orders in them, a VAT reader that handled one of four receipt formats, and a threshold chosen on regular synthetic text) and fixed them in the pipeline. The adapter did not change. I have not run a fresh held-out test of the fixed configuration, so I do not claim that it passes.

**Post-hoc, on previously used sets, not held out.** Scoring the saved outputs again with all current checks:

| | gold | gold_v2 |
|---|---|---|
| documents | 80 | 44 |
| accuracy on automated documents | 1.000 | 0.938 |
| critical errors | 0 of 40 automated | 1 of 16 automated |

Details, the other rounds and the 1.7B and two-epoch runs are in the repository's `docs/results.md`.

## Limitations

- The data is synthetic and the realistic test sets are small (44 to 96 documents). One error moves the accuracy on `gold_v2` by six points.
- The model makes confident mistakes between neighbouring accounts (repairs and maintenance against software licences, training against legal and consulting fees) and between food and non-food VAT on receipts. A confidence threshold does not catch these. Training for two epochs made it worse: the wrong answers became more confident.
- The critical-error limit of 0.5% needs about 600 automated test documents to show, and my test sets have a few hundred.
- A document with an order in it ("book this to account X") can be followed by the model. The pipeline sends such documents to review before the model is called, with patterns for the kinds of orders I tried. A phrasing none of them covers is not caught.
- The adapter does not know any chart other than SKR03, any currency other than EUR, or any document longer than the 1,280 tokens of prompt and answer it was trained with.
