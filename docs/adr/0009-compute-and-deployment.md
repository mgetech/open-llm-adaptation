# 0009 Compute and deployment

Status: accepted. Run on Kaggle and on Azure.

Context: I develop on free GPUs, and the data is synthetic. A firm that uses this would need the model in a region and on a machine it controls.

Decision: Kaggle (two T4) is where I develop: baselines, training, and evaluation on two vLLM servers in parallel. Kaggle has no Docker, so `make serve-local` runs vLLM and the gateway as processes from `configs/serve.yaml`. For a deployment, `deploy/compose.yaml` runs the pinned vLLM image and the gateway on one GPU VM, with vLLM on an internal network. The planned VM is an Azure `NC4as_T4_v3` in Germany West Central, with the same T4.

Alternatives: A hosted model API, which breaks professional secrecy. Kubernetes, which is too much for one GPU.

Consequences: Training and evaluation are reproducible on a free GPU, and the served stack started on a Kaggle T4 in a smoke test. The compose stack ran on an Azure T4: the smoke test passed, and `dev` and `dev_real` gave the same results as on Kaggle (`results/azure/`). The serving benchmark is future work.
