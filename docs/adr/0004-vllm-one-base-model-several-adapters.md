# 0004 vLLM with one base model and several adapters

Status: accepted. One adapter is shipped.

Context: Each platform needs its own behaviour. A full model per platform costs about 8 GB and needs its own evaluation.

Decision: One vLLM server holds the base model at a pinned revision and the adapters as LoRA modules, registered by name in `configs/serve.yaml` (up to two, rank 16). The gateway serves a platform with the adapter of the same name, and a request cannot name another model.

Alternatives: A merged model per platform. One vLLM process per adapter.

Consequences: A new platform costs one adapter of about 132 MB. What I evaluated is what I serve: the base model at half precision with the adapter on top. A merged model would be a different artifact that I would have to evaluate again. The second adapter slot is unused.
