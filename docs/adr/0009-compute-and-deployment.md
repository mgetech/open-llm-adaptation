# 0009 Compute and deployment

Status: accepted for Kaggle. The Azure run is not done.

Context: I develop on free GPUs, and the data is synthetic. A firm that uses this would need the model in a region and on a machine it controls.

Decision: Kaggle (two T4) is where I develop: baselines, training, and evaluation on two vLLM servers in parallel. Kaggle has no Docker, so `make serve-local` runs vLLM and the gateway as processes from `configs/serve.yaml`. For a deployment, `deploy/compose.yaml` runs the pinned vLLM image and the gateway on one GPU VM, with vLLM on an internal network. The planned VM is an Azure `NC4as_T4_v3` in Germany West Central, with the same T4.

Alternatives: A hosted model API, which breaks professional secrecy. Kubernetes, which is too much for one GPU.

Consequences: Training and evaluation are reproducible on a free GPU. The VM run, which should match Kaggle within the confidence intervals, is future work, and so is the serving benchmark.
