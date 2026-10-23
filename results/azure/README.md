# The run on Azure

The shipped adapter served by the compose stack of `deploy/compose.yaml` on one Tesla T4, and the same evaluation as on Kaggle. It checks that the stack starts and answers, and that the adapter gives the same results on another machine. It is not a benchmark and it is not a new claim: it ran on `dev` and `dev_real` only.

| | |
|---|---|
| VM | Azure `Standard_NC4as_T4_v3` (4 vCPUs, 28 GiB, one Tesla T4 with 16 GB), Germany West Central, created 2026-10-23 at 19:44 UTC |
| System | Ubuntu 22.04.5, NVIDIA driver 595.99.02, Docker 29.9.2 with Compose 5.6.1 |
| Serving | `vllm/vllm-openai:v0.30.0` and the gateway image built from `deploy/gateway.Dockerfile`, at commit `c4d1212` plus the fix to the adapter hash (see below) |
| Model | Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c`, adapter `qwen3-4b-r3-0`, hash `2de6c7087c2b1abe1c48f21c14d247d24d9a752c06facb96130361f83194478c` |

## Files

- `smoke-test.json`: the serving smoke test (`make serve-check NO_VLLM=1`).
- `vm.txt`: the machine, the driver and `nvidia-smi` with vLLM loaded, the versions and the containers.
- `eval/tax-booking/qwen3-4b-sft-r3-0-azure/dev.json` and `dev_real.json`: the evaluation results, from inside the compose network.
- `compare-dev.json` and `compare-dev_real.json`: the comparison with the Kaggle results of the same adapter (`make compare`).
- `screenshots/overview.jpg`: the VM in the portal while it ran. `screenshots/request.jpg`: one request to the gateway, an invoice of `dev` through the Postman collection `deploy/gateway.postman_collection.json`, with the answer, the adapter hash and the confidence.

I deleted the VM and its resources after I copied these files.

## The smoke test

8 of 8 checks passed: the gateway was ready, no key and a wrong key gave 401, a key without the platform gave 403, six `dev` documents got an answer, a document without a total went to review, the metrics counted the requests and the audit log held a record for each, with hashes and no text. The two checks that call vLLM directly are skipped (`NO_VLLM=1`), because vLLM is on a network without a route out. `/readyz` still checks that vLLM serves the base model and the adapter, and vLLM reported version 0.30.0.

## The comparison with Kaggle

| | dev | dev_real |
|---|---|---|
| documents | 300 | 96 |
| identical answers | 299 | 96 |
| same review decision | 300 | 96 |
| largest difference in confidence | 0.011 | 0.008 |
| document accuracy | 0.7833 | 0.6979 |
| automation rate | 0.7612 | 0.6494 |
| accuracy on automated documents | 0.9951 | 0.9600 |
| critical-error rate | 0.0049 | 0.0000 |
| review recall | 0.9688 | 1.0000 |

Both comparisons pass: the base model revision, the adapter hash, the prompt hash and the threshold (0.9817) are the same, the documents are the same, and every metric of the VM is equal to the Kaggle value and inside its confidence interval. The one `dev` answer that differs is `dev-00100`: both runs send it to review, and only the reason that the model wrote differs. The comparison also lists the GPU and the commit as different, because the evaluation ran in a container without `nvidia-smi` and without the `.git` folder, so those two fields are empty in the files here. They do not count towards the result.

The evaluation took about 13 minutes for `dev` and about 4 minutes for `dev_real`. I report this because it happened. I did not design the run to measure speed, and the load benchmark is still future work.

## The first run failed the comparison

The first evaluation, on the same VM, compared as `FAIL` on both splits, because the adapter hash was different: `3808cac3…` against `2de6c708…`. The evaluation hashed the whole adapter folder, and the download tool had left its notes in `.cache/huggingface/`. The two adapter files were the same (`make fetch-adapters` had checked them), the metrics of that run were the same as in the second, and on `dev_real` all 96 answers were identical. I changed the evaluation to hash only `adapter_config.json` and `adapter_model.safetensors`, like every other place, added a test, copied the changed file to the VM and ran the evaluation again. The files of the first run are not kept. `vm.txt` lists the changed file as modified, because the VM held it before I committed it.
