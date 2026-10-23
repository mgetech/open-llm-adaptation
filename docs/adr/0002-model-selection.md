# 0002 Model selection

Status: accepted. The rule for the case that nothing passes decided the result.

Context: The candidates are Qwen3-1.7B and Qwen3-4B. The shipped model has to be small, openly licensed and trainable on one 16 GB T4.

Decision: `make models-check` gates every candidate before any GPU time: an Apache-2.0 or MIT licence, a QLoRA memory estimate that fits a T4, and the pinned vLLM version serving the architecture with LoRA. The smallest configuration that passes the eval gate ships. If none passes, the one with the fewest failed checks ships.

Alternatives: Take the largest model that fits. Compare models by one headline accuracy.

Consequences: Neither size passed on held-out data in any round. The 1.7B failed 5 and 4 checks in rounds 1 and 2 and is far behind on round 3, and the 4B failed 3 in round 3, so the 4B ships under the fallback rule. I removed the Qwen3-8B fallback from the registry and did not try a pair of adapters.
