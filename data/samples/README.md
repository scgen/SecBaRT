# Schema samples

Small excerpts of the released data files, kept in git so that the formats are
documented even when the full files are stored externally.

| File | Full file | Notes |
|---|---|---|
| `sft_sample.json` | `data/wcstatic_synthref_merge/train-sft.json` | three triples of `{prompt, vul-code, sec-code}` |
| `token_labels_sample.jsonl` | `data/wcstatic_synthref_merge/token_labels_v7_ord.jsonl` | two rows (one per side); the per-token arrays are truncated to 60 entries with `"..."` |
| `secodeplt_case_sample.json` | `third_party/SecCodePLT_Plus/filtered-test_cases.json` | one RL task; the unittest strings are truncated to 400 characters |

See `DATA.md` for the full field semantics.
