"""Reference implementation of the LLM token-label construction pipeline.

The scripts in this package rebuild the supervision used by the scoring head:
an LLM annotator reads a task together with its vulnerable and secure
implementations and returns *verbatim code quotations* for the causal flaw
spans (vulnerable code) and the effective defense spans (secure code).  The
quotations are programmatically located in the source and mapped to tokenizer
offsets, producing the three-valued labels (0 = unsafe, 1 = safe,
0.5 = neutral).

Running these scripts requires an OpenAI-compatible endpoint (``ARK_KEY``,
``ARK_BASE``, ``ARK_MODEL``) plus the corresponding third-party checkouts
(CWEval / SecCodePLT+).  The released label files let you skip this stage.
"""
