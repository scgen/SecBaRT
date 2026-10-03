# Third-party assets

Nothing in this directory is redistributed with the repository except the
small scoring driver in `cweval/`.

## CWEval

* Upstream: <https://github.com/Co1lin/CWEval> (Apache-2.0)
* Required for the main result.  Clone it as `third_party/CWEval`; the
  generation driver (`scripts/cweval_gen.py`) reads
  `third_party/CWEval/benchmark`, and scoring runs in the public docker image
  `co1lin/cweval:latest`.
* `cweval/official_reeval.sh` and `cweval/cweval_stages_driver.py` are thin
  drivers around the official harness: they stage generations into the
  container and run parse -> compile -> tests -> merge -> report with
  per-stage timeouts.  The harness itself comes from the docker image.

## SecCodePLT+

* Upstream: <https://github.com/uiuc-kang-lab/SecCodePLT>
* Required for RL.  Place the released `filtered-test_cases.json` (400 Python
  tasks) under `third_party/SecCodePLT_Plus/`.  The RL loop executes the
  capability and safety unittests with `secbart/secodeplt_run.py` on the host
  (no docker needed).

## Datasets and models

The vulnerability datasets, the general-code corpus, the Qwen2.5-Coder-7B base
model, and EvalPlus are downloaded separately; see `DATA.md`.
