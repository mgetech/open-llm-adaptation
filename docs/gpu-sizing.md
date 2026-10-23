# GPU sizing

What the shipped model needs on a GPU and what one GPU can serve, from the runs I made. A number I did not measure says "not measured".

## Serving memory on one T4

The calculated column comes from the model files and its config. The other column is what vLLM 0.30.0 reported in its start-up log on a Kaggle T4 (`results/serving/vllm-memory.txt`), started with `max_model_len` 4096, `gpu_memory_utilization` 0.9 and `max_loras` 2, with one adapter loaded.

| part | calculated | vLLM reports |
|---|---|---|
| base weights, fp16 | 7.49 GiB (8,044,936,192 bytes in the weight index of the pinned revision) | 7.7 GiB |
| KV cache | 147,456 bytes per token (2 × 36 layers × 8 KV heads × 128 × 2 bytes), so 24,784 tokens are 3.40 GiB | 3.4 GiB, 24,784 tokens, 6.05 requests of 4,096 tokens |
| adapter | 33,030,144 parameters, 63 MiB in fp16 (the file is 126 MiB in 32 bits) | not reported |
| activations and CUDA graphs | not calculated | 1.65 GiB and 0.45 GiB |
| whole GPU | | 13.11 GiB wanted, which is 0.9 of the 14.56 GiB the T4 offers |

On the Azure T4, `nvidia-smi` showed 14,079 MiB in use of 16,384 MiB with the model loaded (`results/azure/vm.txt`). The KV cache holds at least 19 documents of the 1,280 tokens that the adapter was trained for, at the same time.

## One adapter or two

The server ran with `max_loras: 2` and one adapter loaded, on Kaggle and on Azure. Traffic with two adapters at once is not measured. It needs the load benchmark, which is future work.

## Training on one GPU or two

Two runs of Qwen3-1.7B of 50 steps each, with an effective batch size of 16 (`results/round1/runs/`):

| | 1 GPU | 2 GPUs |
|---|---|---|
| time for 50 steps | 696.5 s | 353.8 s |
| samples per second | 1.15 | 2.26 |
| GPU hours | 0.193 | 0.197 |
| peak memory per GPU | 2.69 GiB | 2.81 GiB |
| one epoch of 375 steps, calculated from the 50 steps | 87 min | 44 min |

Two GPUs were 1.97 times as fast, a scaling efficiency of 98%, for 2% more GPU hours. The shipped Qwen3-4B trained 375 steps on two GPUs in 6,517 seconds (3.6 GPU hours) with a peak of 5.0 GiB per GPU. The 4B model with QLoRA fits a T4 with room to spare.

## Capacity

The only measurement is the evaluation of `dev` on the Azure T4: 300 documents in 685.8 seconds with 8 requests in flight, which is 0.437 documents per second. Per request, the median time was 18.6 seconds and the 95th percentile 27.9 seconds (`results/azure/eval/`). With one request at a time, the Kaggle smoke test took 6.8 to 19.6 seconds per document for 6 documents, and vLLM generated about 11.6 tokens per second. I did not measure other numbers of requests in flight, so this is not a load benchmark.

At 0.437 documents per second one GPU would process about 37,800 documents in 24 hours of continuous work. Peaks and headroom are not included:

| documents a day | GPU busy time | GPUs |
|---|---|---|
| 5,000 | 13% | 1 |
| 20,000 | 53% | 1 |
| 100,000 | 265% | 3 |

The cost per 1,000 documents from the same run is in `docs/results.md`.
