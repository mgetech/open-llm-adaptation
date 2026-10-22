# Results

The results are kept per round of data. Each round has its own folder:

```
results/
  data_manifest.json    the rounds of data, with seed, generator version and SHA-256 of every file
  final.lock.json       the final claim: the files it stands on, the configurations declared for it and the runs
  post_hoc/
    round3_rescored.json                    post-hoc re-scoring of the round 3 claim run, not a claim
    gold_rescored.json                      post-hoc re-scoring of the shipped configuration on gold and gold_v2, not a claim
  round1/
    eval/<platform>/<system>/<split>.json    metrics, confidence intervals, slices and one row per document
    runs/<platform>-<model>-<seed>.json      manifest of one training run
  round2/
    eval/, runs/                              as in round 1
    rescored/<platform>/<system>/<split>.json the round 2 results scored again with pipeline checks added later
  round3/
    eval/, runs/                              as in round 1
    rescored/<platform>/<system>/<split>.json dev and dev_real scored again with the later pipeline checks
```

## The rounds

- **Round 1** is the data from seed 0 and the first version of the generator. Both model sizes failed the gate on it. I kept its data (`data/tax-booking/round1/`), its results and its run manifests as they were.
- **Round 2** is the data from a changed generator and seed 1. I changed the generator after reading the errors on the round 1 dev and test splits, so the round 1 test splits could no longer count as unseen. Names of round 2 systems and runs have `-r2` after the model, for example `qwen3-4b-sft-r2-0`.

Neither model size passed the gate in round 2 either. `make gate` on the `-r2` systems (command below) shows which checks failed. No seeds beyond seed 0 were trained for round 2.

**Round 3** is the data from generator v3 and seed 2. The wording and layout of a share of the documents of every split were rewritten by language models run outside this repository, because text from the generator is more regular than real documents. The labels still come from code, and a check rejects any rewrite in which an amount, total, rate, canary name or item name changed. The rewritten text cannot be rebuilt from the seed, so the data files are checked by hash. Names of round 3 systems and runs have `-r3`, for example `qwen3-4b-sft-r3-0`. A run that trains a set number of epochs (`EPOCHS=2`) has `-e2` after the round, for example `qwen3-4b-sft-r3-e2-0`, so it never shares a name with the run of the config.

The gold files are the same for all rounds: `gold.jsonl` is the original set, and `gold_v2.jsonl` was written before any round 2 result existed. `dev_real.jsonl` is a set of realistic documents that I use only to choose settings. It is never trained on and is not part of the gate or of any report.

### Round 3

Qwen3-4B, seed 0, was the first system trained for round 3. It trained for all 375 steps. I chose its checkpoint among steps 150, 225, 300 and 375 on `dev_real`, with the whole pipeline: the lowest critical-error rate, then the highest document accuracy. The scores of each checkpoint are in `qwen3-4b-sft-r3-0-ckpt<step>`, and the choice is in the run manifest.

It did not pass the gate. As run, accuracy on automated documents, the critical-error rate and prompt-injection success were over their limits (`make gate`, command below). I changed how prompt-injection success is counted after this result (see the changes below), so the injection figure in the round 3 files is the old one. I trained Qwen3-1.7B on the same data afterwards (see below), and more seeds were not trained.

An earlier 4B run of round 3 stopped training early on the `dev_real` loss and gave a much worse model. I dropped that rule. The results of that run are not kept here.

The round 3 test splits and `gold_v2` were evaluated in that run, so they cannot back another claim. `final.lock.json` holds that claim. A run of a system on `test_ood` or on a gold file is refused unless a claim was declared before the run (`make declare-claim`) and the system is one of its configurations. The claim cannot change after the first run.

### Two epochs on round 3 (a negative result)

The one-epoch model was close to the limits on dev and `dev_real`, so I trained the same Qwen3-4B on the same round 3 data for two epochs (750 steps, a checkpoint every 75, seed 0, run `qwen3-4b-r3-e2-0`). I scored steps 450, 525, 600, 675 and 750 on `dev_real` with the whole pipeline and kept step 450 (critical-error rate 0.029 and document accuracy 0.854, the best of the five; the others had a critical-error rate of 0.042 to 0.055). I set the margin I wanted before the run in `configs/margin.yaml`. I did not train a second seed: a second seed is luck, not a change.

It is not better than one epoch. This is the stage report with the threshold chosen on dev and `dev_real` together, 0.8333 (`make stage-gate SYSTEM=qwen3-4b-sft-r3-e2-0 ROUND=3 OUT_DIR=results/round3/rescored MARGIN=1`):

| check | limit | dev | dev_real |
|---|---|---|---|
| automation rate | >= 0.6 | 0.881 pass | 0.844 pass |
| accuracy on automated documents | >= 0.95 | 0.983 pass | 0.908 fail |
| critical-error rate | <= 0.005 | 0.000 pass | 0.015 fail |
| review-flag recall | >= 0.95 | 1.000 pass | 1.000 pass |

The margin is not met: `dev_real` accuracy on automated documents is 0.908, against 0.97. The one-epoch model has 0.960 there.

Training longer made the model more sure of its wrong answers. On `dev_real` four wrong account and VAT answers that no rule catches sit at a confidence of 0.9993 to 0.9998, so no threshold separates them from the right ones: raising it to 0.99 still leaves 1 critical error in 54 automated documents. The wrong answers are the same kind each time: neighbouring accounts (4805 and 4964 for a maintenance or support contract, 4945 and 4950 for consulting work) and food against non-food VAT on a receipt with no printed cue.

I first scored this run before the VAT check read the "incl." lines of a receipt. A critical error on dev passed that check, and the threshold had to go to 0.9999, which left an automation rate of 0.20 on dev and 0.04 on `dev_real`. The numbers above use the check after the fix, and the one-epoch model is the same under both.

### The 1.7B on round 3

I trained Qwen3-1.7B on the round 3 data for one epoch (375 steps, seed 0, run `qwen3-1.7b-r3-0`, about 45 minutes on two T4 GPUs) after the 4B had failed, so that both sizes have a round 3 result. It ran on dev and `dev_real` only (`FINAL = False`), so it has no test-split result and no gate. I scored steps 150, 225, 300 and 375 on `dev_real` and kept step 300 (critical-error rate 0.143, document accuracy 0.656; step 375 had 0.157 and 0.646).

It is far behind the 4B. Without a confidence threshold, 38 of 265 automated dev documents and 10 of 70 on `dev_real` have a critical error (0.143 on both), against 1 of 205 and 0 of 50 for the 4B at its threshold. The threshold chosen on dev and `dev_real` together is 0.9997, and it leaves an automation rate of 0.049 on dev and 0.052 on `dev_real`, against the limit of 0.6 (`make stage-gate SYSTEM=qwen3-1.7b-sft-r3-0 ROUND=3 OUT_DIR=results/round3/rescored`). The other checks pass only because almost nothing is automated.

### Post-hoc re-scoring of round 3

`post_hoc/round3_rescored.json` scores the saved outputs of the round 3 claim run again with the changes below: the injection figure and a confidence threshold chosen on dev. It is post-hoc and not a claim. The test splits and `gold_v2` it reads were already evaluated in the claim run, and I made the changes after I saw that result. The threshold comes from dev only. I did not change it after looking at the other splits, and I do not choose anything from this report. The file carries the same label, the hashes of the saved results it read, and every gate check, once with `gold_v2` as the gold set and once with `gold`. `make post-hoc SYSTEM=qwen3-4b-sft-r3-0 BASELINE=qwen3-4b-fewshot-r3 ROUND=3` writes it again from the saved results without running a model. It did not pass the gate: with `gold_v2` as the gold set, the automation rate is below its limit, and the critical-error rate and prompt-injection success are over theirs.

### Post-hoc re-scoring on gold and gold_v2

`post_hoc/gold_rescored.json` scores the saved outputs of the shipped configuration (Qwen3-4B, round 3, one epoch) on `gold` and `gold_v2` with every pipeline check of today: the orders guard, the VAT check that reads the "incl." lines, and the confidence threshold of 0.9817 chosen on dev and `dev_real`. `make post-hoc-gold SYSTEM=qwen3-4b-sft-r3-0 ROUND=3` writes it again from the saved results without running a model. Both sets were evaluated before, and some of the checks were written after I saw those results, so these numbers are post-hoc and not held out. I changed nothing because of them, and `gold_v3` is not read.

| | gold | gold_v2 |
|---|---|---|
| documents | 80 | 44 |
| automation rate | 0.625 (40 of 64 that need no review) | 0.471 (16 of 34) |
| accuracy on automated documents | 1.000 | 0.938 |
| critical errors | 0 of 40 automated | 1 of 16 automated |
| review recall | 1.000 | 1.000 |

### Changes made after the round 3 result

- **Prompt-injection success** counts an injection document only when the output did what the injected sentence asked. A document sent to review is not a success of the attack. Before, any difference from the expected outcome counted, so a correct booking that went to review for low confidence counted as an attack that worked. The old figure is kept as `injection_unexpected_rate`. The limit is still 0. The gate does not read the injection figure of results saved before the change; `make rescore` writes them again with the new definition.
- **Orders in the text.** A document whose text gives orders to the system that reads it (drop the rules, book to an account, skip the review, approve, a message to the system in a role) goes to review with the reason `injection_suspected`, before the model is called. It is a pipeline rule and is reported apart from the model's flags. The patterns name kinds of orders and were not taken from the failing documents of any test split. A document that words an order in a way none of them covers is not caught, and the model still has to resist it. `python -m adapt.input_guard` counts the injection documents it flags and the other documents it flags by mistake in the training split, dev and `dev_real`.
- **The confidence threshold** is chosen on the rows of dev and `dev_real` together, with the limit on critical errors first and then the accuracy target, and then used as it is. Before, it was chosen on dev, and only the accuracy target counted. With both splits in the choice, no split is left to check the threshold on; the claim run is the first check. `make rescore RETUNE=1 TUNE_ON="dev dev_real"` applies it to saved results, and `make threshold-report SYSTEM=...` prints how many documents are automated and how many of them have a critical error at each threshold, on dev and on `dev_real`.
- **`make stage-gate SYSTEM=...`** shows the gate checks that dev and `dev_real` can show, with a pass or a fail for each. It helps to choose settings and is not the claim.

### The gold_v3 seal

`final.lock.json` records the SHA-256 of `gold_v3.jsonl` and the date I sealed it, 2026-10-22, before the decisions that followed. The first hash I recorded (`02152f1ca6396e98cd544af0d66d256a52e73eca75104fcf2b0aaf286c9c6df6`) was computed over a copy with Windows line endings, but git stores the file with Unix line endings (`.gitattributes`), so no other checkout could reproduce it. CI found this. The content is the same: with the line endings converted, the file hashes to `cf07dbdaa76632d42e607ebefa75d22b0349d0b03bf7cef07be8f314f21d371c`, which is also the hash of the file in the commit that added it, before any decision that followed. The lock now holds that hash, `make seal` and `make declare-claim STATUS=1` hash a file with Unix line endings, and a test checks that the committed file still matches.

The validator for `gold_v3` (`make validate-gold`) then found one problem: the `reason` of `gold3-016` held a leftover drafting note and was over the length limit. I replaced that one string and changed nothing else in the file, and I sealed the file again. No result on `gold_v3` existed, so no decision depended on it. The hash before the correction was `cf07dbdaa76632d42e607ebefa75d22b0349d0b03bf7cef07be8f314f21d371c`; it is now `08d92575c8ad5586ed6364aaec695567dfe48d6a9a169ec91c6c94e7c42eff24`.

## Reading a round

```
make gate SYSTEM=qwen3-4b-sft-0 BASELINE=qwen3-4b-fewshot ROUND=1
make gate SYSTEM=qwen3-4b-sft-r2-0 BASELINE=qwen3-4b-fewshot-r2 ROUND=2
make gate SYSTEM=qwen3-4b-sft-r3-0 BASELINE=qwen3-4b-fewshot-r3 ROUND=3
```

The gate compares a system with the few-shot baseline of the same round, because the test documents of the two rounds differ. It refuses a baseline that ran on other test data.

The files in `rescored/` come from `make rescore`. It applies the `heavy_noise` document check and the checks on the answer (`vat_mismatch`, `mixed_private`, `reverse_charge_cue`) to the answers saved in `eval/`, without running a model, and keeps the confidence thresholds of the run. It also prints what each of the two newest rules adds on `dev`, `dev_real` and `test_ood`. The files in `round2/rescored/` were made before those two rules existed. The `eval/` files stay as they were run. `standin-1.7b-4b-r2-each` and `-ignore` combine the two round 2 adapters of different sizes with `make pair`. They stand in for a pair of two seeds of one size, which does not exist for round 2.

Files with `-steps` in their name come from short runs, such as the training speed measurement on one and on two GPUs. They are not results of a full run.
