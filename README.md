# SecBaRT

Official code release for **SecBaRT**, a bottleneck-based reinforcement
learning framework for secure code generation.  The repository contains the
core training and evaluation pipeline needed to reproduce the main results;
baselines are intentionally not included.

SecBaRT learns a task-conditioned *bottleneck* state before decoding.  During
training, the vulnerable and secure implementations of the same task are
reconstructed through an asymmetric attention layout: the vulnerable branch
can only see the task through the bottleneck, and the secure branch cannot copy
the vulnerable code.  A token-level scoring head is trained on the same hidden
states, frozen, and reused during dual-arm RL, where functional and security
tests provide the executable reward.

## Main results

Evaluated on CWEval (119 tasks, greedy decoding) with Qwen2.5-Coder-7B:

| Benchmark | Metric | Score |
|---|---|---|
| CWEval | Func@1 | 67.23 |
| CWEval | Sec@1 | 66.39 |
| CWEval | Func-Sec@1 | 57.98 |
| HumanEval+ | Pass@1 | 83.54 |
| MBPP+ | Pass@1 | 79.89 |

`scripts/collect_metrics.py` writes these numbers into `metric.json` after an
evaluation run.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| I | `scripts/1_sft.sh` | Bottleneck SFT: vulnerable reconstruction + secure generation on the 40,360-triple pool |
| II | `scripts/2_token_head.sh` | Train the token-level scoring head (ordinal objective) on the frozen SFT seed |
| III | `scripts/3_rl_dualarm.sh` | Dual-arm group-normalised RL with functional + security execution feedback |
| III (token-shaped) | `scripts/3b_rl_dualarm_tkh.sh` | Dual-arm PPO that adds the frozen head's token-level security reward |
| IV | `scripts/4_eval_cweval.sh` | Generate on CWEval with the bottleneck server and score with the official harness |
| optional | `scripts/5_eval_functional.sh` | HumanEval+ / MBPP+ with EvalPlus |

Stage III has two entry points that share the same data, rewards, and
evaluation path: the sequence-level dual-arm objective and the PPO variant that
adds the frozen head's token-level reward (the full method).  Reproducing the
ablation table means running both.

## Quick start

```bash
git clone https://github.com/scgen/SecBaRT && cd SecBaRT

# 1. environment
bash scripts/0_setup.sh

# 2. assets (see DATA.md for download links / reconstruction)
#    data/wcstatic_synthref_merge/train-sft.json               (40,360 triples)
#    data/wcstatic_synthref_merge/token_labels_v7_ord.jsonl    (token labels)
#    models/Qwen2.5-Coder-7B                                   (base model)
#    third_party/SecCodePLT_Plus/filtered-test_cases.json      (RL tasks)
#    third_party/CWEval/benchmark                              (evaluation)

# 3. train
CUDA_VISIBLE_DEVICES=0 bash scripts/1_sft.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/2_token_head.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/3_rl_dualarm.sh

# 4. evaluate (requires docker with the public co1lin/cweval:latest image)
CUDA_VISIBLE_DEVICES=0 bash scripts/4_eval_cweval.sh
```

All entry scripts read their configuration from environment variables with
repository-relative defaults, so an existing data/model directory can be
plugged in without editing any file:

```bash
MODEL_DIR=/data/models/Qwen2.5-Coder-7B \
SFT_DATA=/data/SecBaRT/train-sft.json \
WORK_DIR=/scratch/secbart bash scripts/1_sft.sh
```

## Repository layout

```
secbart/                 core package (training, RL, token head, vLLM server)
secbart/annotate/        reference implementation of the token-label pipeline
scripts/                 end-to-end entry points
third_party/cweval/      official-harness scoring driver (docker based)
data/samples/            small schema examples for the released data files
DATA.md                  data construction, downloads, and licensing
REPRODUCE.md             step-by-step reproduction notes and expected numbers
```

## Hardware

* Training (SFT and RL) was run on a single 80 GB GPU (A100/H100 class) with
  `--lora_rank 0` (full fine-tuning) and the paged 8-bit optimizer.
* Evaluation serves the 7B model with vLLM at
  `--gpu_memory_utilization 0.9` on one GPU.
* Smaller GPUs can be used with a LoRA rank above zero, but the released
  numbers correspond to the full fine-tuning configuration above.

## Notes on the paper

* The main checkpoint is produced by the **dual-arm GRPO-style** objective
  (`scripts/3_rl_dualarm.sh`), which normalises advantages within each
  (arm, task) group and uses no value head.  `scripts/3b_rl_dualarm_tkh.sh`
  runs the PPO + GAE variant with the frozen token head; both share the same
  data, reward definitions, and evaluation path.
* The scoring head is trained once on the frozen SFT seed and is not updated
  during RL, so the reward definition does not drift with the policy.
* CWEval evaluation counts every task in the denominator.  If generation
  fails, a placeholder file is written so the task is scored as a failure
  instead of silently dropping out of the benchmark.

## Third-party components

The repository depends on, but does not redistribute, CWEval
(<https://github.com/Co1lin/CWEval>, Apache-2.0), SecCodePLT+
(<https://github.com/uiuc-kang-lab/SecCodePLT>), EvalPlus, and the public
vulnerability datasets listed in `DATA.md`.  Their licenses apply to those
assets.
