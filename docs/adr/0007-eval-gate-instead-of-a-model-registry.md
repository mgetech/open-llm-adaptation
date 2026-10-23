# 0007 Eval gate instead of a model registry

Status: accepted

Context: I need to decide which adapter ships, and to show that the decision was not made after seeing the test results.

Decision: A gate of ten checks with thresholds in `platforms/tax-booking/platform.yaml`, written before training. `results/final.lock.json` fixes the test files, the configurations and the shipped choice before the final run, and a test file that backs one claim cannot back another. An adapter is identified by the hash of its two files, which is in the run manifests, `configs/serve.yaml` and the audit log.

Alternatives: An experiment tracker and a model registry. Judging by eye.

Consequences: The shipped configuration did not pass on held-out data in any round (7 of 10 checks at best). After I read errors, a test file could no longer count as unseen, so the fixes after round 3 have no held-out result yet. I label that openly.
