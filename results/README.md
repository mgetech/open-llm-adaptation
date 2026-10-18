# Results

The results are kept per round of data. Each round has its own folder:

```
results/
  data_manifest.json    the rounds of data, with seed, generator version and SHA-256 of every file
  round1/
    eval/<platform>/<system>/<split>.json    metrics, confidence intervals, slices and one row per document
    runs/<platform>-<model>-<seed>.json      manifest of one training run
  round2/
    ...
```

## The rounds

- **Round 1** is the data from seed 0 and the first version of the generator. Both model sizes failed the gate on it. I kept its data (`data/tax-booking/round1/`), its results and its run manifests as they were.
- **Round 2** is the data from a changed generator and seed 1. I changed the generator after reading the errors on the round 1 dev and test splits, so the round 1 test splits could no longer count as unseen. Names of round 2 systems and runs have `-r2` after the model, for example `qwen3-4b-sft-r2-0`.

The gold files are the same for both rounds: `gold.jsonl` is the original set, and `gold_v2.jsonl` was written before any round 2 result existed.

## Reading a round

```
make gate SYSTEM=qwen3-4b-sft-0 BASELINE=qwen3-4b-fewshot ROUND=1
make gate SYSTEM=qwen3-4b-sft-r2-0 BASELINE=qwen3-4b-fewshot-r2 ROUND=2
```

The gate compares a system with the few-shot baseline of the same round, because the test documents of the two rounds differ. It refuses a baseline that ran on other test data.

Files with `-steps` in their name come from short runs, such as the training speed measurement on one and on two GPUs. They are not results of a full run.
