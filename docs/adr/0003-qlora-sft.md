# 0003 QLoRA SFT

Status: accepted

Context: Training has to fit a T4 (16 GB, no bf16). Kaggle gives two of them.

Decision: 4-bit NF4 QLoRA with LoRA rank 16 on all linear layers, supervised fine-tuning with the loss on the answer only, data-parallel on two GPUs. The shipped adapter trained for one epoch on 6,000 documents (375 steps, 3.6 GPU hours, seed 0). I chose its checkpoint on `dev_real` with the whole pipeline.

Alternatives: Full fine-tuning does not fit. A few-shot prompt is the baseline, and the adapter beat it on `test_ood` (p < 0.001). DPO on the confident mistakes is planned and not run. Two epochs were better on `dev` and worse on `dev_real`, with confident wrong answers.

Consequences: The adapter is 132 MB. Longer training makes wrong answers more confident, which the confidence threshold cannot catch. Only one seed of the shipped size was trained.
