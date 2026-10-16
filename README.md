# open-llm-adaptation

I fine-tune a small open-weight LLM to turn the text of an invoice or receipt into booking lines, and to flag anything uncertain for review instead of guessing. I evaluate it against a few-shot baseline and serve it with vLLM on GPUs I control.

**What works today:** the project setup only. The package installs, `make lint` and `make test` pass, and CI runs both. There is no data generator, model or evaluation yet.
