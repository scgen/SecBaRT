# Data, checkpoints, and third-party assets

The repository ships small schema samples under `data/samples/`.  The full
files are too large for git and are distributed separately (Hugging Face
dataset / release assets).  Everything can also be rebuilt from public sources
with the recipe below.

## 1. Supervised pool (Stage I)

`train-sft.json` is a JSON list of 40,360 triples:

```json
{"prompt": "<task description>",
 "vul-code": "<vulnerable implementation>",
 "sec-code": "<secure implementation>"}
```

Composition:

| Part | Rows | Source |
|---|---|---|
| Vulnerability-repair triples | 20,360 | nine public vulnerability datasets (below) |
| General-code triples | 20,000 | code-focused subset of AM-DeepSeek-R1-Distilled-1.4M; `vul-code == sec-code`, every token neutral |

The vulnerability-repair triples were reconstructed with a reasoning model
from pre-fix code, post-fix code, and commit evidence, because raw commits
depend on repository context.  The reconstruction normalises the generated
fields, rejects malformed triples, runs functional checks on both
implementations, and keeps only pairs for which the security tests distinguish
the vulnerable and repaired versions.

Source datasets (see the paper's reference list for the exact versions):
PrimeVul, DiverseVul, SecVulEval-BL, CrossVul, MegaVul, LPO, PreciseBug,
SafeCoder, and TitanVul.  The general-code samples come from the
AM-DeepSeek-R1-Distilled-1.4M corpus.

## 2. Token labels (Stage II)

`token_labels_v7_ord.jsonl` contains one row per (pair, side):

```json
{"pair_id": 6, "prompt": "...", "side": 0, "code": "...",
 "labels": [0, 0, ..., 1, 1, ...], "weights": [0.0, ..., 1.0, ...],
 "spans": [{"quote": "verbatim code", "weight": 1.0, "why": "..."}],
 "cwe": "CWE-89", "ord_cls": ["irrel", ..., "vuln", ...]}
```

* `side = 0` is the vulnerable implementation, `side = 1` the secure one.
* `labels` are the three-valued targets: `0` = UNSAFE token (causal flaw span),
  `1` = SAFE token (effective defense span), `0.5` = NEUTRAL.  Tokens touched
  by a span inherit its label; `weights` carry the span weight
  (`0.5` for supporting context).
* The annotator receives the task description and both implementations.  It
  returns the **smallest causal flaw spans** in the vulnerable code and the
  **smallest effective defense spans** in the secure code as *verbatim code
  quotations*, not token indices.  The pipeline locates each quotation in the
  source, maps its character offsets to tokenizer offsets, and derives the
  labels.  Locating quotations instead of trusting indices makes every
  annotation verifiable; the released files record the quotations and their
  occurrence counts.
* Sparse annotations keep the supervision focused on the security-relevant
  minority, while the neutral targets still give the head a target at every
  position.  Before an annotation is accepted the pipeline checks index
  bounds, uniqueness, ordering, token counts, and correspondence with the
  source sequence.

The reference implementation of this pipeline is in `secbart/annotate/`.
It requires an OpenAI-compatible endpoint (`ARK_KEY`, `ARK_BASE`,
`ARK_MODEL`); the released label file lets you skip this stage entirely.

## 3. Reinforcement-learning pool (Stage III)

RL uses SecCodePLT+ tasks rewritten into self-contained programs with inlined
dependencies and explicit expected outputs.  Every task must pass both test
suites with the reference implementation, and a vulnerable variant must
trigger the intended vulnerability.  The released pool is
`filtered-test_cases.json` (400 Python tasks, 17 CWE families); the untouched
official `test_cases.json` can be used with `RL_DATASET=secodeplt`.

The executable feedback is obtained by running each sampled program against
the task's capability (functional) and safety (security) tests with
`secbart/secodeplt_run.py`; no docker is required for this step.

**No CWEval task is used for training**, so the reported CWEval numbers are
not affected by train/test overlap.  Note that the filtered SecCodePLT+ pool
overlaps the official SecCodePLT train split, which is why SecCodePLT is not
used for evaluation here.

## 4. Evaluation assets

* **CWEval** (main result): clone <https://github.com/Co1lin/CWEval> into
  `third_party/CWEval`.  Scoring runs in the public docker image
  `co1lin/cweval:latest`; see `third_party/cweval/official_reeval.sh`.
* **HumanEval+ / MBPP+** (functional retention): install `evalplus` and run
  `scripts/5_eval_functional.sh`.

## 5. Downloading the prepared assets

The prepared data is split across three Hugging Face dataset repositories:

| Repository | File | Status |
|---|---|---|
| `scgen/SecBaRT-sft` | `train-sft.json` | available |
| `scgen/SecBaRT-token-labels` | `token_labels_v7_ord.jsonl` | published separately |
| `scgen/SecBaRT-rl-tasks` | `filtered-test_cases.json` | published separately |

```bash
# supervised pool
hf download scgen/SecBaRT-sft train-sft.json --repo-type dataset \
    --local-dir data/wcstatic_synthref_merge

# token labels
hf download scgen/SecBaRT-token-labels token_labels_v7_ord.jsonl --repo-type dataset \
    --local-dir data/wcstatic_synthref_merge

# RL tasks
hf download scgen/SecBaRT-rl-tasks filtered-test_cases.json --repo-type dataset \
    --local-dir third_party/SecCodePLT_Plus

# base model
hf download Qwen/Qwen2.5-Coder-7B --local-dir models/Qwen2.5-Coder-7B

# evaluation harness
git clone https://github.com/Co1lin/CWEval third_party/CWEval
```

`scripts/prepare_data.sh` wraps these commands.  Repositories that are not
published yet are reported and skipped, and each can be mirrored with
`SECBART_SFT_REPO`, `SECBART_LABELS_REPO`, or `SECBART_SCPLT_REPO`.

## 6. Licensing

The vulnerability datasets, AM-DeepSeek-R1-Distilled-1.4M, SecCodePLT+,
CWEval, and EvalPlus are third-party assets and remain under their original
licenses.  Check each upstream repository before redistributing derived
files.  The reconstructed task descriptions, token annotations, and
checkpoints released with this repository are provided for research use.
