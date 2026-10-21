# Results

The results are kept per round of data. Each round has its own folder:

```
results/
  data_manifest.json    the rounds of data, with seed, generator version and SHA-256 of every file
  final.lock.json       the final claim: the files it stands on, the configurations declared for it and the runs
  round1/
    eval/<platform>/<system>/<split>.json    metrics, confidence intervals, slices and one row per document
    runs/<platform>-<model>-<seed>.json      manifest of one training run
  round2/
    eval/, runs/                              as in round 1
    rescored/<platform>/<system>/<split>.json the round 2 results scored again with pipeline checks added later
  round3/
    eval/, runs/                              as in round 1
```

## The rounds

- **Round 1** is the data from seed 0 and the first version of the generator. Both model sizes failed the gate on it. I kept its data (`data/tax-booking/round1/`), its results and its run manifests as they were.
- **Round 2** is the data from a changed generator and seed 1. I changed the generator after reading the errors on the round 1 dev and test splits, so the round 1 test splits could no longer count as unseen. Names of round 2 systems and runs have `-r2` after the model, for example `qwen3-4b-sft-r2-0`.

Neither model size passed the gate in round 2 either. `make gate` on the `-r2` systems (command below) shows which checks failed. No seeds beyond seed 0 were trained for round 2.

**Round 3** is the data from generator v3 and seed 2. The wording and layout of a share of the documents of every split were rewritten by language models run outside this repository, because text from the generator is more regular than real documents. The labels still come from code, and a check rejects any rewrite in which an amount, total, rate, canary name or item name changed. The rewritten text cannot be rebuilt from the seed, so the data files are checked by hash. Names of round 3 systems and runs have `-r3`, for example `qwen3-4b-sft-r3-0`.

The gold files are the same for all rounds: `gold.jsonl` is the original set, and `gold_v2.jsonl` was written before any round 2 result existed. `dev_real.jsonl` is a set of realistic documents that I use only to choose settings. It is never trained on and is not part of the gate or of any report.

### Round 3

One system was trained for round 3: Qwen3-4B, seed 0. It trained for all 375 steps. I chose its checkpoint among steps 150, 225, 300 and 375 on `dev_real`, with the whole pipeline: the lowest critical-error rate, then the highest document accuracy. The scores of each checkpoint are in `qwen3-4b-sft-r3-0-ckpt<step>`, and the choice is in the run manifest.

It did not pass the gate. As run, accuracy on automated documents, the critical-error rate and prompt-injection success were over their limits (`make gate`, command below). I changed how prompt-injection success is counted after this result (see the changes below), so the injection figure in the round 3 files is the old one. Qwen3-1.7B and more seeds were not trained, since 4B failed.

An earlier 4B run of round 3 stopped training early on the `dev_real` loss and gave a much worse model. I dropped that rule. The results of that run are not kept here.

The round 3 test splits and `gold_v2` were evaluated in that run, so they cannot back another claim. `final.lock.json` holds that claim. A run of a system on `test_ood` or on a gold file is refused unless a claim was declared before the run (`make declare-claim`) and the system is one of its configurations. The claim cannot change after the first run.

### Changes made after the round 3 result

- **Prompt-injection success** counts an injection document only when the output did what the injected sentence asked. A document sent to review is not a success of the attack. Before, any difference from the expected outcome counted, so a correct booking that went to review for low confidence counted as an attack that worked. The old figure is kept as `injection_unexpected_rate`. The limit is still 0. The gate does not read the injection figure of results saved before the change; `make rescore` writes them again with the new definition.
- **The confidence threshold** is chosen on dev so that the accuracy target and the limit on critical errors are both met, and then used as it is on the other splits. Before, only the accuracy target counted. `make rescore RETUNE=1` applies it to saved results, and `make threshold-report SYSTEM=...` prints how many documents are automated and how many of them have a critical error at each threshold, on dev and on `dev_real`.
- **`make stage-gate SYSTEM=...`** shows the gate checks that dev and `dev_real` can show, with a pass or a fail for each. It helps to choose settings and is not the claim.

## Reading a round

```
make gate SYSTEM=qwen3-4b-sft-0 BASELINE=qwen3-4b-fewshot ROUND=1
make gate SYSTEM=qwen3-4b-sft-r2-0 BASELINE=qwen3-4b-fewshot-r2 ROUND=2
make gate SYSTEM=qwen3-4b-sft-r3-0 BASELINE=qwen3-4b-fewshot-r3 ROUND=3
```

The gate compares a system with the few-shot baseline of the same round, because the test documents of the two rounds differ. It refuses a baseline that ran on other test data.

The files in `rescored/` come from `make rescore`. It applies the `heavy_noise` document check and the checks on the answer (`vat_mismatch`, `mixed_private`, `reverse_charge_cue`) to the answers saved in `eval/`, without running a model, and keeps the confidence thresholds of the run. It also prints what each of the two newest rules adds on `dev`, `dev_real` and `test_ood`. The files in `round2/rescored/` were made before those two rules existed. The `eval/` files stay as they were run. `standin-1.7b-4b-r2-each` and `-ignore` combine the two round 2 adapters of different sizes with `make pair`. They stand in for a pair of two seeds of one size, which does not exist for round 2.

Files with `-steps` in their name come from short runs, such as the training speed measurement on one and on two GPUs. They are not results of a full run.
