# Reproducing the main results

This document lists the exact configuration behind the reported numbers and
the checks to run at each stage.

## 0. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
bash scripts/0_setup.sh
```

Required external assets (see `DATA.md`): the base model, `train-sft.json`,
`token_labels_v7_ord.jsonl`, `filtered-test_cases.json`, and a CWEval
checkout.  Everything defaults to a path inside the repository and can be
overridden with `MODEL_DIR`, `SFT_DATA`, `TOKEN_LABELS`, `SECPLT_CASES`,
`CWEVAL_REPO`, `WORK_DIR`.

## 1. Bottleneck SFT (Stage I)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/1_sft.sh
```

| Setting | Value |
|---|---|
| Data | 40,360 triples, seed-42 shuffle |
| Epochs | 1 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Embedding LR multiplier / init | 50.0 / mean |
| LoRA rank | 0 (full fine-tuning) |
| Bottleneck tokens `n_vuln` | 4 |
| Max sequence length | 1024 |
| Secure loss weight `sec_up` | 3.0 |
| Optimizer | paged AdamW 8-bit |
| Seed | 42 |

Output: `$SFT_OUT/merged_hf_model`.

The validation split used during training is a deterministic 525-example
sample of the training file (seed 42) when `VAL_DATA` is not provided.  It is
used for loss logging only; the final checkpoint is always kept.

## 2. Token head (Stage II)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/2_token_head.sh
```

| Setting | Value |
|---|---|
| Data | `token_labels_v7_ord.jsonl` |
| Architecture | LayerNorm -> Linear(H,256) -> GELU -> Dropout(0.1) -> Linear(256,1) |
| Objective | ordinal hinge (`--loss ordinal`, `lam_ord=5.0`, `margin_ord=0.3`) |
| Epochs / batch / LR | 40 / 32 / 1e-3 |
| Seed | 42 |

The script copies epoch 40 to `token_head_rl.pt`; this frozen head is used in
`scripts/3_rl_token_reward.sh`.

## 3. Token-reward dual-arm RL (Stage III)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/3_rl_token_reward.sh
```

| Setting | Value |
|---|---|
| Seed model | the SFT checkpoint from Stage I |
| RL pool | SecCodePLT+ `filtered-test_cases.json` (400 tasks) |
| Steps | 768 |
| Batch / samples per arm (`k`) | 4 / 4 |
| Learning rate | 3e-6 |
| KL coefficient | 0.3 |
| Sampling temperature | 0.8 |
| LoRA rank | 0 |
| Bottleneck tokens | 4 |
| Seed | 42 |
| Secure arm input | prompt + `<vuln>`x4 + `<secu>` |
| Vulnerable arm input | prompt + `<vuln>`x4 (task input masked) |
| Rewards | `R_sec = Pass_func * Pass_sec`, `R_vul = Pass_func * (1 - Pass_sec)` |
| Token-level reward | `alpha * (sigma(w^T h_t + b) - 0.5)` added to the secure arm, `w_head=0.5` |
| Policy optimisation | PPO with `gamma=1.0`, `lambda=0.95`, `clip_eps=0.2`, `vf_coef=0.5` |

The frozen head scores the hidden states that the policy already computes, so
the position-level reward adds no extra forward pass.  The run exports
`merged_hf_model` at the end of training.

## 4. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/4_eval_cweval.sh work/rl_token_reward/merged_hf_model 0 8123
```

The script starts the bottleneck vLLM server, generates greedy completions for
all 119 CWEval tasks, scores them with the official harness inside
`co1lin/cweval:latest`, and writes `report.json`, `res_all.json`, and
`metric.json`.

Reference numbers for the released checkpoint:

| Metric | Value |
|---|---|
| CWEval Func@1 | 67.23 |
| CWEval Sec@1 | 66.39 |
| CWEval Func-Sec@1 | 57.98 |
| CWEval correct-secure@1 (of functionally correct) | 86.25 |
| HumanEval+ Pass@1 | 83.54 |
| MBPP+ Pass@1 | 79.89 |

HumanEval+/MBPP+ are optional:

```bash
pip install evalplus
CUDA_VISIBLE_DEVICES=0 bash scripts/5_eval_functional.sh work/rl_token_reward/merged_hf_model 0 8124
```

## 5. Expected variation

* CWEval has 119 tasks, so one task is worth ~0.84 percentage points; small
  differences in the sampling environment can move a single task.
* Generation is greedy (`temperature=0`) and seeds are fixed, so the main
  variation comes from library versions and compile/test environment inside
  the docker image.
* RL runs are stochastic by design (temperature 0.8, 4 samples per task per
  arm).  The seed is fixed to 42; reruns with different seeds are expected to
  vary by a few tasks.
* The token-level reward is computed from the frozen head, so reruns with the
  same seed model produce the same reward definition; remaining variation
  comes from the stochastic rollouts.

## 6. Troubleshooting

* **CUDA OOM during SFT**: keep `--lora_rank 0` only on 80 GB GPUs; otherwise
  set `SFT_LORA_RANK` in the script to a positive rank (the released numbers
  use full fine-tuning).
* **vLLM startup timeout**: the script logs to `$EVAL_ROOT/vllm_cweval.log`;
  the first load can take several minutes on shared storage.
* **Missing docker image**: `docker pull co1lin/cweval:latest`.
* **CWEval merge warns about missing directories**: some generated samples
  may not compile; missing directories are merged as failing samples so the
  denominator stays at 119.
