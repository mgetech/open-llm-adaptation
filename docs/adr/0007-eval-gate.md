# 0007 Eval gate

Status: accepted

Context: I need to decide which adapter ships, and to show that the decision was not made after seeing the test results.

Decision: A gate of ten checks with thresholds in `platforms/tax-booking/platform.yaml`, written before training. `results/final.lock.json` fixes the test files, the configurations and the shipped choice before the final run, and a test file that backs one claim cannot back another. The adapter the gate judged is identified by the hash of its two files, which is in the run manifests, `configs/serve.yaml` and the audit log.

Alternatives: Judging by eye. Thresholds chosen after seeing the results.

Consequences: The gate did its job: it showed where the model fell short on held-out data (7 of 10 checks at best) and each round's failures told me what to change. It also keeps the claim honest. After I read errors, a test file could no longer count as unseen, so the fixes after round 3 wait for a fresh held-out run, which is the next step.
