"""Locate CWEval (task, unsafe) pairs in a CWEval checkout.

Minimal replacement for the internal ``verify_cweval_pairs.find_cases`` helper
(which also carries a full dual-oracle docker verifier that is not needed for
token annotation).

The CWEval repository ships, for each task, a secure reference
``*_task.<ext>`` and an ``*_unsafe.<ext>`` variant that the official test suite
must reject.  Python tasks have no ``_unsafe.py`` file in the upstream repo;
the original pipeline took their vulnerable variant from a synthesized
``data.json``.  Set ``CWEVAL_SYNTH_JSON`` to point at such a file (same layout
as the internal one) when you want to annotate the Python tasks too.

Environment variables
---------------------
CWEVAL_REPO      path to the CWEval checkout (default: ``third_party/CWEval``)
CWEVAL_SYNTH_JSON optional synthesized vulnerable Python code (list of objects
                 with ``language``, ``case_path`` and ``code_before``)
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

CORE_LANGS = ["c", "cpp", "go", "js", "py"]
EXT = {"c": ".c", "cpp": ".cpp", "go": ".go", "js": ".js", "py": ".py"}


@dataclass
class Case:
    case_id: str          # e.g. cwe_022_0_c (py: cwe_022_0)
    language: str
    origin: str           # official | llm_synth
    group: str            # core | lang
    src_dir: Path
    task_name: str
    test_name: str
    unsafe_name: str
    unsafe_text: str | None = None

    @property
    def key(self):
        return f"{self.group}/{self.language}/{self.case_id}"


def _cweval_root():
    env = os.environ.get("CWEVAL_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "third_party" / "CWEval"


def _load_synth_py():
    path = os.environ.get("CWEVAL_SYNTH_JSON")
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for x in json.loads(Path(path).read_text(encoding="utf-8")):
        if x.get("language") != "py":
            continue
        rel = x["case_path"].split("/CWEval/benchmark/", 1)[-1]
        out[Path(rel).name] = x["code_before"]
    return out


def find_cases(bench=None):
    """Return the CWEval cases that have both a secure and a vulnerable file."""
    root = Path(bench) if bench else _cweval_root() / "benchmark"
    synth = _load_synth_py()
    cases = []
    for lang in CORE_LANGS:
        d = root / "core" / lang
        if not d.is_dir():
            continue
        for task in sorted(d.glob("*_task.*")):
            if task.suffix != EXT[lang]:
                continue
            stem = task.name[: -len(task.suffix)]
            base = stem[:-5] if stem.endswith("_task") else stem
            test = d / f"{base}_test.py"
            if not test.exists():
                continue
            unsafe = d / f"{base}_unsafe{task.suffix}"
            if lang == "py":
                txt = synth.get(task.name)
                cases.append(Case(base, lang, "llm_synth" if txt else "missing",
                                  "core", d, task.name, test.name, unsafe.name, txt))
            else:
                cases.append(Case(base, lang, "official", "core", d,
                                  task.name, test.name, unsafe.name))
    d = root / "lang" / "c"
    if d.is_dir():
        for task in sorted(d.glob("*_task.c")):
            stem = task.name[: -len(task.suffix)]
            base = stem[:-5] if stem.endswith("_task") else stem
            test = d / f"{base}_test.py"
            unsafe = d / f"{base}_unsafe.c"
            if test.exists() and unsafe.exists():
                cases.append(Case(base, "c", "official", "lang", d,
                                  task.name, test.name, unsafe.name))
    return cases
