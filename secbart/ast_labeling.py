"""AST-based token-level security labels for (vul-code, sec-code) pairs.

Motivation
----------
The previous labeler (``token_diff_labels`` in the cotrain trainer) aligns the
*BPE token id lists* of vul/sec code with difflib.  That is noisy in three
ways:

  1. formatting / whitespace / indentation changes show up as token diffs even
     though they carry no security signal;
  2. comment / docstring rewrites ("Insecure implementation: ..." ->
     "Secure implementation: ...") are labeled SAFE/UNSAFE although they are
     not executable;
  3. a single BPE token is the diff unit, so identifiers/operators are split
     at subword boundaries and the same semantic change is inconsistently
     labeled on the two sides.

This module replaces the diff unit with tree-sitter *leaf tokens* (AST
leaves), which have well-defined node types and byte spans.  Alignment happens
on ``(node_type, text)`` tuples, so formatting noise disappears by
construction and comment/docstring/string leaves can be excluded explicitly.
Leaf labels are then mapped back to BPE token ids through the tokenizer's
offset mapping, so the function remains a drop-in replacement for
``token_diff_labels(vul_ids, sec_ids)``.

Design choices
--------------
- Leaves are nodes with no children (named or unnamed), plus the content
  inside string literals (``string_content``) so a string change is visible.
- A docstring is a ``string`` whose parent ``expression_statement`` is the
  first statement of module/function/class body; its ``string_content`` leaf
  is classified as docstring noise.
- ``skip_modes``:  "comment" skips comment leaves only; "comment+string"
  additionally skips all string leaves; "none" keeps everything.
- If either side fails to parse (syntax error), falls back to the raw BPE
  difflib labels.
"""

from __future__ import annotations

import difflib
import functools
import importlib
from dataclasses import dataclass
from typing import Iterable, Optional

from tree_sitter import Language, Parser


# ---------------------------------------------------------------------------
# language registry (lazily imported, only what is installed)
# ---------------------------------------------------------------------------
_LANGUAGE_MODULES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "java": ("tree_sitter_java", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "go": ("tree_sitter_go", "language"),
}

_PARSER_CACHE: dict[str, Optional[Parser]] = {}


def parser_for(language: str) -> Optional[Parser]:
    """Return a cached tree-sitter Parser for ``language`` (None if absent)."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    entry = _LANGUAGE_MODULES.get(language)
    if entry is None:
        _PARSER_CACHE[language] = None
        return None
    module_name, factory_name = entry
    try:
        module = importlib.import_module(module_name)
        _PARSER_CACHE[language] = Parser(Language(getattr(module, factory_name)()))
    except Exception:
        _PARSER_CACHE[language] = None
    return _PARSER_CACHE[language]


def detect_language(code: str, fallback: str = "python") -> str:
    """Cheap heuristic; caller can pass an explicit language instead."""
    stripped = code.lstrip()
    if stripped.startswith(("#include", "int main", "void main")):
        return "cpp"
    if stripped.startswith("package ") and ";" in code[:200]:
        return "go"
    if stripped.startswith(("import java.", "public class", "public static")):
        return "java"
    return fallback


# ---------------------------------------------------------------------------
# leaf extraction
# ---------------------------------------------------------------------------
# language aliases used by dataset "language" fields -> parser names
LANGUAGE_ALIASES = {
    "py": "python",
    "js": "javascript",
    "c/c++": "cpp",
}

COMMENT_NODE_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "doc_comment",
    "documentation_comment",
}
STRING_NODE_TYPES = {
    "string",
    "string_content",
    "string_start",
    "string_end",
    "raw_string_literal",
    "interpreted_string_literal",
    "char_literal",
}
ALL_NOISE_NODE_TYPES = COMMENT_NODE_TYPES | STRING_NODE_TYPES


@dataclass(frozen=True)
class Leaf:
    node_type: str
    text: str
    start_byte: int
    end_byte: int
    noise_class: str  # "", "comment", "string", "docstring"


@functools.lru_cache(maxsize=8192)
def _cached_leaves(text: str, language: str) -> tuple[Leaf, ...]:
    parser = parser_for(language)
    if parser is None:
        return ()
    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node
    if root.has_error:
        return ()
    leaves: list[Leaf] = []
    _walk(root, text.encode("utf-8"), leaves)
    return tuple(leaves)


def _walk(node, code_bytes: bytes, out: list[Leaf]):
    nt = node.type
    if nt == "string":
        # Keep f-string / formatted-string interpolations (they are executable
        # code and security relevant, e.g. XSS: f"<b>{name}</b>"), while the
        # literal quote/content parts become noise leaves.
        for child in node.children:
            if child.type == "interpolation":
                _walk(child, code_bytes, out)
            elif child.type in ("string_start", "string_content", "string_end"):
                if child.end_byte > child.start_byte:
                    out.append(
                        Leaf(
                            node_type=child.type,
                            text=code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace"),
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            noise_class=_noise_class(child),
                        )
                    )
            else:
                _walk(child, code_bytes, out)
        return
    if nt in ALL_NOISE_NODE_TYPES:
        # comments and string fragments (string_start/content/end reached only
        # via interpolation-less paths) become single leaves.
        out.append(
            Leaf(
                node_type=nt,
                text=code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                noise_class=_noise_class(node),
            )
        )
        return
    if len(node.children) == 0:
        if not node.is_missing and node.end_byte > node.start_byte:
            out.append(
                Leaf(
                    node_type=nt,
                    text=code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    noise_class="",
                )
            )
        return
    for child in node.children:
        _walk(child, code_bytes, out)


def _noise_class(node) -> str:
    nt = node.type
    if nt in COMMENT_NODE_TYPES:
        return "comment"
    if nt in STRING_NODE_TYPES:
        # docstring: parent chain string -> expression_statement -> (block)? ->
        # (module|function_definition|class_definition|decorated_definition)
        cur = node.parent
        for _ in range(3):
            if cur is None:
                break
            if cur.type == "expression_statement":
                # docstring statement contains the string as its only child
                if len(cur.children) != 1 or cur.children[0] != node:
                    return "string"
                nxt = cur.parent
                if nxt is not None and nxt.type in {
                    "module",
                    "function_definition",
                    "class_definition",
                    "decorated_definition",
                }:
                    return "docstring"
                if nxt is not None and nxt.type == "block":
                    nxt = nxt.parent
                    if nxt is not None and nxt.type in {
                        "function_definition",
                        "class_definition",
                        "decorated_definition",
                    }:
                        return "docstring"
            cur = cur.parent
        return "string"
    return ""


def leaves_for(text: str, language: str = "python") -> list[Leaf]:
    return list(_cached_leaves(text, language))


# ---------------------------------------------------------------------------
# labeling
# ---------------------------------------------------------------------------
def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def leaf_diff_labels(
    vul_code: str,
    sec_code: str,
    language: str = "python",
    skip_modes: str = "comment+string",
    align: str = "difflib",
) -> tuple[Optional[list[Leaf]], Optional[list[Leaf]], Optional[list[int]], Optional[list[int]]]:
    """Align vul/sec leaf sequences and label leaves.

    Returns (vul_leaves, sec_leaves, vul_labels, sec_labels) where labels are
    per-leaf: 0 neutral, 1 unsafe(vul side), 2 safe(sec side).  None on parse
    failure.  Labeled leaves whose node type is filtered by ``skip_modes`` are
    reset to neutral.
    """
    vul_leaves = leaves_for(vul_code, language)
    sec_leaves = leaves_for(sec_code, language)
    if not vul_leaves or not sec_leaves:
        return None, None, None, None

    skip: set[str] = set()
    if "comment" in skip_modes:
        skip |= COMMENT_NODE_TYPES
    if "string" in skip_modes:
        skip |= STRING_NODE_TYPES

    def key(leaf: Leaf):
        return (leaf.node_type, leaf.text)

    vkeys = [key(l) for l in vul_leaves]
    skeys = [key(l) for l in sec_leaves]

    vul_labels = [0] * len(vul_leaves)
    sec_labels = [0] * len(sec_leaves)
    if align == "dp":
        _vlab, _slab = dp_diff_labels(vkeys, skeys)
        # per-leaf masks: 1 (vul side), 2 (sec side)
        for i, lab in enumerate(_vlab):
            if lab == 1:
                if vul_leaves[i].node_type not in skip:
                    vul_labels[i] = 1
        for j, lab in enumerate(_slab):
            if lab == 2:
                if sec_leaves[j].node_type not in skip:
                    sec_labels[j] = 2
        return vul_leaves, sec_leaves, vul_labels, sec_labels
    if align != "difflib":
        raise ValueError(f"unknown align: {align}")
    matcher = difflib.SequenceMatcher(None, vkeys, skeys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            for i in range(i1, i2):
                if vul_leaves[i].node_type not in skip:
                    vul_labels[i] = 1
        elif tag == "insert":
            for j in range(j1, j2):
                if sec_leaves[j].node_type not in skip:
                    sec_labels[j] = 2
        elif tag == "replace":
            for i in range(i1, i2):
                if vul_leaves[i].node_type not in skip:
                    vul_labels[i] = 1
            for j in range(j1, j2):
                if sec_leaves[j].node_type not in skip:
                    sec_labels[j] = 2
    return vul_leaves, sec_leaves, vul_labels, sec_labels


def _tokens_overlapping_leaves(
    code: str,
    tokenizer,
    labeled_leaves: Iterable[Leaf],
) -> set[int]:
    """Return BPE token indices whose char span overlaps any labeled leaf."""
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    hits: set[int] = set()
    for leaf in labeled_leaves:
        for idx, (cs, ce) in enumerate(offsets):
            if cs >= leaf.end_byte or ce <= leaf.start_byte:
                continue
            hits.add(idx)
    return hits


def _tokens_covered_by_leaves(
    code: str,
    tokenizer,
    labeled_leaves: Iterable[Leaf],
    min_ratio: float = 0.0,
) -> set[int]:
    """BPE token indices whose char span is covered by labeled leaves.

    ``min_ratio`` in [0,1]: a token [cs,ce) is labeled when the fraction of its
    characters overlapped by *any* labeled leaf (merged, no double counting)
    is >= min_ratio.  ``0.0`` reproduces the legacy "any overlap" behavior.

    Intended for ablation: min_ratio=1.0 requires the token to be fully inside
    labeled leaves (kills boundary artifacts where a token straddles a labeled
    leaf and an unlabeled one); intermediate values trade recall vs precision.
    """
    if min_ratio <= 0.0:
        return _tokens_overlapping_leaves(code, tokenizer, labeled_leaves)
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    spans = sorted((l.start_byte, l.end_byte) for l in labeled_leaves)
    hits: set[int] = set()
    for idx, (cs, ce) in enumerate(offsets):
        if cs >= ce:
            continue
        # merge leaf spans overlapping [cs, ce)
        covered = 0
        cur = cs
        for a, b in spans:
            if b <= cs:
                continue
            if a >= ce:
                break
            if a > cur:
                cur = a
            if b > cur:
                covered += b - cur
                cur = b
        if covered / (ce - cs) >= min_ratio:
            hits.add(idx)
    return hits


def ast_token_diff_labels(
    vul_code: str,
    sec_code: str,
    tokenizer,
    vul_ids: Optional[list[int]] = None,
    sec_ids: Optional[list[int]] = None,
    language: str = "python",
    skip_modes: str = "comment+string",
    align: str = "difflib",
    overlap_ratio: float = 0.0,
):
    """Drop-in replacement for ``token_diff_labels``.

    Returns (vul_labels, sec_labels) aligned with the BPE token id lists
    (``vul_ids``/``sec_ids`` used only for length/fallback).  Labels:
    0 neutral, 1 unsafe, 2 safe.
    """
    v_leaves, s_leaves, v_labels, s_labels = leaf_diff_labels(
        vul_code, sec_code, language=language, skip_modes=skip_modes, align=align
    )
    if v_leaves is None or s_leaves is None:
        # fallback: raw BPE difflib (existing behavior)
        return token_diff_labels(vul_ids or [], sec_ids or [])

    vul_hits = _tokens_covered_by_leaves(
        vul_code, tokenizer,
        (l for l, lab in zip(v_leaves, v_labels) if lab == 1),
        min_ratio=overlap_ratio,
    )
    sec_hits = _tokens_covered_by_leaves(
        sec_code, tokenizer,
        (l for l, lab in zip(s_leaves, s_labels) if lab == 2),
        min_ratio=overlap_ratio,
    )
    enc_v = tokenizer(vul_code, add_special_tokens=False)
    enc_s = tokenizer(sec_code, add_special_tokens=False)
    nv, ns = len(enc_v["input_ids"]), len(enc_s["input_ids"])
    return (
        [1 if i in vul_hits else 0 for i in range(nv)],
        [2 if i in sec_hits else 0 for i in range(ns)],
    )


def token_diff_labels(vul_ids, sec_ids):
    """Original BPE-level difflib labeler (kept for comparison / fallback)."""
    A, B = vul_ids, sec_ids
    if A == B:
        return [0] * len(A), [0] * len(B)
    vul_labels, sec_labels = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, A, B, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            vul_labels.extend([0] * (i2 - i1))
            sec_labels.extend([0] * (j2 - j1))
        elif tag == "delete":
            vul_labels.extend([1] * (i2 - i1))
        elif tag == "insert":
            sec_labels.extend([2] * (j2 - j1))
        elif tag == "replace":
            vul_labels.extend([1] * (i2 - i1))
            sec_labels.extend([2] * (j2 - j1))
    return vul_labels, sec_labels


# ---------------------------------------------------------------------------
# DP (Levenshtein) alignment, matching the reward_evaluator algorithm but
# vectorized with numpy.  Produces the globally minimal-edit alignment
# (difflib's SequenceMatcher is greedy and can be suboptimal on repeats).
# ---------------------------------------------------------------------------
def dp_diff_labels(A, B):
    """Exact edit-distance DP alignment of two sequences.

    Reproduces the *effective* ``reward_evaluator.label_elements_with_changes``
    semantics (the second definition, which shadows the first):

      - substitution cost 1, insertion/deletion cost 1;
      - when A[i-1] == B[j-1] the U operation is forced (never replaced by a
        D/A tie at equal cost);
      - otherwise ties are broken D -> A -> S (``min`` over [D, A, S]).

    Returns (labels_A, labels_B): 0 unchanged, 1 deleted/substituted (A side),
    2 inserted/substituted (B side).  Same contract as ``token_diff_labels``.
    """
    import numpy as np

    n, m = len(A), len(B)
    if n == 0:
        return [], [2] * m
    if m == 0:
        return [1] * n, []
    if A == B:
        return [0] * n, [0] * m

    # op matrix: 0=U,1=D(consume A),2=A(consume B),3=S (int8)
    ops = np.zeros((n + 1, m + 1), dtype=np.int8)
    ops[1:, 0] = 1  # D
    ops[0, 1:] = 2  # A
    prev = np.arange(n + 1, dtype=np.int64)  # cost column j-1
    a = list(A)
    b = list(B)
    for j in range(1, m + 1):
        cur = np.empty(n + 1, dtype=np.int64)
        cur[0] = j
        bj = b[j - 1]
        for i in range(1, n + 1):
            eq = a[i - 1] == bj
            if eq:
                # Original forces U on equal chars (U <= D and U <= A always
                # hold for a proper edit-distance DP, so this is a tie-break).
                cur[i] = prev[i - 1]
                ops[i, j] = 0
                continue
            # old D = dp[i-1][j] + 1 (cell above, current column)
            # old A = dp[i][j-1] + 1 (cell left, previous column)
            # old S = dp[i-1][j-1] + 1
            cost_d = cur[i - 1] + 1
            cost_a = prev[i] + 1
            cost_s = prev[i - 1] + 1
            # tie-break order D -> A -> S, matching the original
            # reward_evaluator implementation (min over [D, A, S]).
            best, op = cost_d, 1
            if cost_a < best:
                best, op = cost_a, 2
            if cost_s < best:
                best = cost_s
                op = 3
            cur[i] = best
            ops[i, j] = op
        prev = cur

    lab_a, lab_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        op = int(ops[i, j])
        if op == 0:
            lab_a.append(0); lab_b.append(0); i -= 1; j -= 1
        elif op == 1:
            lab_a.append(1); i -= 1
        elif op == 2:
            lab_b.append(2); j -= 1
        else:
            lab_a.append(1); lab_b.append(2); i -= 1; j -= 1
    lab_a.reverse(); lab_b.reverse()
    return lab_a, lab_b
