# Results

The details behind the numbers in the README. Every number comes from `results/`, and `results/README.md` has the commands that write them.

The configuration I ship is Qwen3-4B fine-tuned with QLoRA on the round 3 data for one epoch (the checkpoint of step 375), with the pipeline rules and a confidence threshold of 0.9817. The pipeline rules include the check for orders in the text and the check of the VAT printed on the document.

## From the saved results

These tables and the chart are written from the files in `results/` by `make report SYSTEM=qwen3-4b-sft-r3-0 BASELINE=qwen3-4b-fewshot-r3 ROUND=3`. The sections after them explain the numbers.

<!-- results:start -->
**Held out: the claim run on the round 3 test sets.** The gate passed with `gold` 7 of 10 checks; with `gold_v2` 7 of 10 checks. It did not pass the gate.

| failed check | limit | gate with `gold` | gate with `gold_v2` |
|---|---|---|---|
| accuracy on automated documents | >= 0.95 | 0.9385 FAIL | 0.9327 FAIL |
| critical-error rate | <= 0.005 | 0.0431 FAIL | 0.0505 FAIL |
| prompt-injection success | <= 0 | no value, saved with the old definition FAIL | no value, saved with the old definition FAIL |

**Post hoc, on sets that were used before, not held out.** The saved outputs of the same configuration scored again with every check of today.

|  | gold | gold_v2 |
|---|---|---|
| documents | 80 | 44 |
| automation rate | 0.625 | 0.471 |
| accuracy on automated documents | 1.000 | 0.938 |
| critical errors | 0 of 40 automated | 1 of 16 automated |
| review recall | 1.000 | 1.000 |

**Training runs**, with the number of GPUs each one used.

| run | round | steps | GPUs | GPU hours |
|---|---|---|---|---|
| tax-booking-qwen3-1.7b-0 | 1 | 376 | not stated | 1.3 |
| tax-booking-qwen3-4b-0 | 1 | 376 | 2 | 3.4 |
| tax-booking-qwen3-1.7b-r2-0 | 2 | 376 | 2 | 1.4 |
| tax-booking-qwen3-4b-r2-0 | 2 | 376 | 2 | 3.3 |
| tax-booking-qwen3-1.7b-r3-0 | 3 | 375 | 2 | 1.5 |
| tax-booking-qwen3-4b-r3-0 | 3 | 375 | 2 | 3.6 |
| tax-booking-qwen3-4b-r3-e2-0 | 3 | 750 | 2 | 6.3 |

**Azure T4 against Kaggle**, the same adapter, prompt and threshold: `dev` passes, 299 of 300 answers identical; `dev_real` passes, 96 of 96 answers identical. Details are in `results/azure/README.md`.

![Automation rate and accuracy on automated documents at every confidence threshold](img/coverage_accuracy.png)

Accuracy against cost per 1,000 documents: not run, because the load benchmark is future work. One figure exists as a by-product of an evaluation run. The Azure evaluation processed 300 `dev` documents in 686 seconds on one T4, with 8 requests in flight (0.44 documents per second, about 1,600 an hour). At 0.66 USD an hour, the on-demand price of the machine, that is about 0.42 USD per 1,000 documents. It is not a load benchmark, and the price is for the VM alone.
<!-- results:end -->

## Validation: dev and dev_real

The shipped configuration passes every gate check that dev and `dev_real` can show:

| check | limit | dev | dev_real |
|---|---|---|---|
| automation rate | >= 0.6 | 0.761 | 0.649 |
| accuracy on automated documents | >= 0.95 | 0.995 | 0.960 |
| critical-error rate | <= 0.005 | 0.0049 (1 of 205 automated) | 0.000 (0 of 50) |
| review-flag recall | >= 0.95 | 0.969 | 1.000 |
| invalid JSON rate | 0 | 0 | 0 |
| invented account rate | 0 | 0 | 0 |
| prompt-injection success | 0 | 0 | no injection documents |
| canary leak | 0 | 0 | 0 |

This is validation, not a test. I chose the threshold on these two splits, and I wrote the checks after I read errors on dev. The margin is thin: the critical-error rate on dev is 0.0049 against a limit of 0.005, and with 205 automated documents the interval around it is wide. `dev_real` is a set of realistic documents that I never train on.

## Fine-tuned against few-shot, and the generalisation gap

On the round 3 test sets the fine-tuned model beat the few-shot baseline of the same size on `test_ood`: it was right on 224 documents where the baseline was wrong, and the baseline was right on 2 where it was wrong (p < 0.001). The generalisation gap, `test_iid` minus `test_ood`, was 0.027 against a limit of 0.05. Review recall on the edge cases was 0.964 against 0.95, and the automation rate was 0.938. These come from the round 3 run in which the gate was evaluated once.

## Realistic documents (post-hoc, previously used gold sets)

`gold` and `gold_v2` were evaluated before, and I wrote some of the checks after I saw those results, so these numbers are post-hoc and not held out. I scored the saved outputs of the shipped configuration again with all current checks and changed nothing because of them.

| | gold | gold_v2 |
|---|---|---|
| documents | 80 | 44 |
| automation rate | 0.625 | 0.471 |
| accuracy on automated documents | 1.000 | 0.938 |
| critical errors | 0 of 40 automated | 1 of 16 automated |

`gold_v2` is small: one error moves its accuracy by six points. `results/post_hoc/gold_rescored.json` has the full report.

## Held-out result and the gate

On the held-out data of the round 3 claim run (the round 3 test sets and the gold sets; 20 of the 44 `gold_v2` documents had already been evaluated in round 2) the model passed 7 of 10 checks with `gold` and 7 of 10 with `gold_v2`. It beat few-shot (p < 0.001), the generalisation gap was 0.027 and review recall was 0.964. Automation was 0.938 with `gold` and 0.971 with `gold_v2`. It failed the same three checks with both: accuracy on automated documents was 0.938 against 0.95 with `gold` and 0.933 with `gold_v2`, the critical-error rate was 0.043 with `gold` and 0.051 with `gold_v2`, against 0.005, and prompt-injection success was 0.167 against 0 on `test_ood` (under the definition I used then, see below). The table below uses the gate with `gold`.

I set the thresholds strict on purpose, before I trained anything. The critical-error limit also cannot be shown with a few hundred automated documents: zero errors in n documents only shows a rate below about 3/n at 95% confidence, so showing a rate of 0.5% or less needs about 600 automated documents. That is a lesson for sizing the next test sets.

The four-billion-parameter model on the held-out data of each round:

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| checks failed (of 10) | 3 | 5 | 3 |
| accuracy on automated documents (>= 0.95) | 0.928 | 0.914 | 0.938 |
| critical-error rate (<= 0.005) | 0.058 | 0.068 | 0.043 |
| generalisation gap (<= 0.05) | 0.015 | 0.051 | 0.027 |
| review-flag recall (>= 0.95) | 0.964 | 0.853 | 0.964 |
| automation rate (>= 0.6) | 0.984 | 0.953 | 0.938 |

Round 3 is better than round 2 on every row except automation, and it is the round with the lowest critical-error rate. It is only a small step from round 1: the critical-error rate fell from 0.058 to 0.043, accuracy rose from 0.928 to 0.938, and the gap grew from 0.015 to 0.027, still inside its limit. The rounds do not compare exactly: the gold set grew from round to round and I changed the data and the generator between them.

The fixes after round 3 (below) pass every check on validation. A fresh held-out run on newly generated test sets and `gold_v3`, which is written and sealed (its hash and the date are in `results/final.lock.json`), is the next step. I have not run it, so the shipped configuration has not passed the gate on held-out data.

## Rounds and what each taught me

| run | what changed | result |
|---|---|---|
| round 1 | first generator, seed 0 | both sizes failed the gate; I read the errors on dev and test, so those test splits could no longer count as unseen |
| round 2 | changed generator, seed 1, `gold_v2` written before any result | both sizes failed; the generalisation gap and review recall also failed for the 4B |
| round 3 | model-rewritten wording for part of every split, `dev_real`, checkpoint chosen on `dev_real` | the 4B failed accuracy on automated documents, the critical-error rate and injection |
| round 3, 1.7B | same data, one epoch, dev and `dev_real` only | critical errors on 14% of automated documents without a threshold; at a safe threshold automation is 0.05; not shipped |
| round 3, two epochs | same data, 750 steps | better on dev, worse on `dev_real`: accuracy on automated 0.908 against 0.960, with confident wrong answers; not shipped |

After the round 3 failure I found three causes and fixed them:
- **Followed injections.** A document with an order inside it ("book this to account X", "skip the review") could be followed by the model. A rule now sends such a document to review before the model is called. I also changed how I count injection success: it counts only an output that did what the sentence asked.
- **The VAT reader.** The check that compares the printed VAT with the booked VAT read only one of the four ways a receipt prints included VAT. It now reads all four. This does not change the dev or `dev_real` numbers of the shipped configuration.
- **A threshold tuned on synthetic data.** I chose the confidence threshold on dev, whose text is regular. It is now chosen on dev and `dev_real` together, with the limit on critical errors first.

The round 3 claim and its lock are unchanged, and the round 3 test sets and `gold_v2` cannot back another claim.

## Model selection

I fine-tuned both sizes in every round. The 1.7B failed the gate in round 1 (5 checks failed) and round 2 (4 failed). On the round 3 data I ran it for one epoch on dev and `dev_real` only, and it is far behind the 4B there: critical errors on 14% of automated documents without a threshold, and an automation rate of 0.05 at the threshold that keeps them out (the 4B: 0.76 and 0.65). The 4B failed 3 checks on held-out data in round 3. Nothing passed on held-out data, so the rule for that case applies: the configuration with the fewest failed checks ships, which is the 4B (the rounds do not use the same data, so this is a rough comparison). I did not try a pair of two adapters.

## Future work

- A fresh held-out run on newly generated `test_ood` and `gold_v3`.
- DPO on the confident mistakes.
- Two-seed agreement.
- Training data aimed at the account pairs the model confuses (repairs and maintenance against software licences, training against legal and consulting fees) and at food against non-food VAT on receipts.
- Measuring the loss on the account and VAT tokens, to choose checkpoints.
- Test sets sized to the critical-error limit (about 600 automated documents).
- The serving benchmark on Kaggle and the run on Azure, if I do not get to them.
- A gold set built from real anonymised documents with a tax expert.

## How I count prompt injection

An injection document has one sentence that tells the model to do something else. The gate counts an attack as a success only when the output did what the sentence asked: it booked to the named account, used the named VAT rate, wrote the requested word or kept the review from happening. A review flag alone is not a success. The limit is 0.

I changed this definition after the round 3 result failed the check. Before, I counted every injection document whose outcome differed from the expected one, so a correct booking that went to review because of low confidence counted as a successful attack. The limit did not change. The old figure is still in the results as `injection_unexpected_rate`, and the gate prints it next to the new one. Results saved before the change have only the old figure, and the gate does not read them until `make rescore` has run on them.

## Post-hoc re-scoring of round 3

`results/post_hoc/round3_rescored.json` re-scores the saved outputs of the round 3 claim run with the injection figure and the confidence threshold I changed after that run. It is post-hoc and not a claim: the test splits and `gold_v2` were already evaluated in that run. The threshold is chosen on dev only. The report lists every gate check and shows that the model still does not pass. `results/README.md` has the details.
