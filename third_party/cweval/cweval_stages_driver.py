#!/usr/bin/env python3
"""Stage-wise driver for the official CWEval harness.

Why this exists (2026-09-23, cost = one window):
    `python cweval/evaluate.py pipeline` runs parse -> compile -> run_tests ->
    merge -> report inside ONE process: no per-stage progress, no per-stage
    timeout.  The init checkpoint's K=64 paper-protocol eval printed
    "Parsing 7616 files"/"7616/7616" and then nothing for ~55 min (one core at
    100 %).  Serial re-runs of the two preceding stages on the same container
    were clean (7616 parses in 60 s, 4544 compiles with no stall), so the stall
    lives in the test stage, which is exactly the stage whose workers spawn
    child processes (node/go servers, subprocesses) that the harness never
    reaps.

    Splitting the stages lets the caller wrap every test batch in `timeout`,
    resume per directory (res.json is written as soon as a directory finishes),
    and fall back to per-test-file runs for the pathological ones - so one bad
    sample costs one sample instead of a whole measurement.

Usage (inside the cweval container, cwd=/home/ubuntu/CWEval):
    python /tmp/cweval_stages_driver.py --stage parse      --eval_path evals/eval_reeval
    python /tmp/cweval_stages_driver.py --stage compile    --eval_path evals/eval_reeval
    python /tmp/cweval_stages_driver.py --stage print-batches --num_proc 8
    python /tmp/cweval_stages_driver.py --stage tests      --eval_path ... --dirs "d1 d2"
    python /tmp/cweval_stages_driver.py --stage one-file   --file <test.py> --out <json>
    python /tmp/cweval_stages_driver.py --stage merge      --eval_path ... [--allow-missing-dirs]
    python /tmp/cweval_stages_driver.py --stage report     --eval_path ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = '/home/ubuntu/CWEval'


def log(msg: str) -> None:
    print(f'[stages {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def test_path_of(task_path: str) -> str:
    return os.path.splitext(task_path.replace('_task.', '_test.'))[0] + '.py'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['parse', 'compile', 'print-batches', 'tests',
                             'one-file', 'one-dir', 'merge', 'report'])
    ap.add_argument('--eval_path', default='evals/eval_reeval')
    ap.add_argument('--num_proc', type=int, default=8)
    ap.add_argument('--dirs', default='')
    ap.add_argument('--file', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--progress_dir', default='/tmp/cweval_progress')
    ap.add_argument('--per_file_timeout', type=float, default=90,
                    help='one-dir: seconds allowed per single test file')
    ap.add_argument('--force', action='store_true',
                    help='re-run directories that already have res.json')
    ap.add_argument('--allow-missing-dirs', action='store_true',
                    help='merge even if some generated_* dirs have no res.json '
                         '(their samples are synthesized as failing and listed)')
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    os.makedirs(args.progress_dir, exist_ok=True)
    from cweval.evaluate import Evaler

    if args.stage == 'print-batches':
        ev = Evaler(eval_path=args.eval_path, num_proc=args.num_proc)
        for i in range(0, len(ev.generated_paths), args.num_proc):
            print(' '.join(ev.generated_paths[i:i + args.num_proc]))
        return 0

    if args.stage == 'one-file':
        # Run a single test file in-process (the caller bounds it with `timeout`).
        from cweval.run_tests import TestResultCollector
        import pytest
        out_path, test_file = args.out, args.file
        collector = TestResultCollector(timeout_per_test=3)
        _exit, os._exit = os._exit, lambda *a: None
        pytest.main([test_file, '--tb=short', '--continue-on-collection-errors',
                     '-k', 'not _unsafe'], plugins=[collector])
        os._exit = _exit
        res = {}
        for fr in collector.file_results.values():
            fr.functional = all(tc.passed for tc in fr.test_cases
                                if tc.marker == 'functionality' and '_unsafe' not in tc.name)
            fr.secure = all(tc.passed for tc in fr.test_cases
                            if tc.marker == 'security' and '_unsafe' not in tc.name)
            res[fr.file] = {'functional': fr.functional, 'secure': fr.secure}
        if out_path:
            json.dump(res, open(out_path, 'w'), indent=4)
        print(f'[one-file] {test_file} -> {json.dumps(res)[:200]}', flush=True)
        return 0

    if args.stage == 'one-dir':
        # Last resort for a directory that stalls even alone: run its test files
        # one at a time in a fresh, timeout-bounded subprocess.  A file that
        # still stalls costs that sample (counted as failing) and is listed.
        import subprocess
        d = (args.dirs.split() or [''])[0]
        files = sorted(os.path.join(root, f)
                       for root, _dirs, fs in os.walk(d) for f in fs
                       if f.endswith('_test.py'))
        fdir = os.path.join(args.progress_dir, 'file_results',
                            os.path.basename(d.rstrip('/')))
        os.makedirs(fdir, exist_ok=True)
        res, stalled = {}, []
        log(f'per-file fallback: {len(files)} test files in {d}')
        for f in files:
            out = os.path.join(fdir, os.path.basename(f) + '.json')
            cmd = ['timeout', '-k', '5', str(args.per_file_timeout), sys.executable,
                   os.path.abspath(__file__), '--stage', 'one-file', '--file', f,
                   '--out', out]
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc == 0 and os.path.exists(out):
                res.update(json.load(open(out)))
            else:
                log(f'  STALLED (rc={rc}) {f}')
                stalled.append(f)
                res[f] = {'functional': False, 'secure': False}
        json.dump(res, open(os.path.join(d, 'res.json'), 'w'), indent=4)
        if stalled:
            json.dump(stalled, open(os.path.join(fdir, 'stalled.json'), 'w'), indent=2)
        log(f'per-file fallback done: {len(res)} cases, {len(stalled)} stalled')
        return 0

    ev = Evaler(eval_path=args.eval_path, num_proc=args.num_proc)

    if args.stage == 'parse':
        # Serial: identical output to the harness, 60 s for 7616 files, and no
        # pool to hang in.
        np0, ev.num_proc = ev.num_proc, 1
        log(f'serial parse of {len(ev.raw_files)} raw files')
        ev.parse_generated()
        ev.num_proc = np0
        log('parse done')
        return 0

    if args.stage == 'compile':
        log(f'compile (pooled, num_proc={ev.num_proc})')
        ev.compile_parsed()
        log('compile done')
        return 0

    if args.stage == 'tests':
        import multiprocessing as mp
        from cweval.run_tests import run_tests
        dirs = args.dirs.split() or ev.generated_paths
        todo = [d for d in dirs
                if args.force or not os.path.exists(os.path.join(d, 'res.json'))]
        skipped = [d for d in dirs if d not in todo]
        if skipped:
            log(f'skip {len(skipped)} dirs that already have res.json')
        if not todo:
            log('nothing to run')
            return 0
        ev._copy_test_files()
        mp.set_start_method('spawn', force=True)
        log(f'tests: {len(todo)} dirs with {ev.num_proc} workers')
        prog = os.path.join(args.progress_dir, 'tests_progress.txt')
        with open(prog, 'a') as prog_f:
            with mp.Pool(ev.num_proc, maxtasksperchild=1) as pool:
                # imap (not map) so each finished dir is persisted immediately:
                # a `timeout`-killed batch then loses only its in-flight dirs.
                for gp, file_res_list in zip(todo, pool.imap(run_tests, todo, chunksize=1)):
                    res = {fr.file: {'functional': fr.functional, 'secure': fr.secure}
                           for fr in file_res_list}
                    json.dump(res, open(os.path.join(gp, 'res.json'), 'w'), indent=4)
                    print(f'OK {gp} n_files={len(res)} {time.strftime("%H:%M:%S")}',
                          file=prog_f, flush=True)
                    log(f'done {gp} ({len(res)} files)')
        log('tests done')
        return 0

    if args.stage == 'merge':
        missing = [gp for gp in ev.generated_paths
                   if not os.path.exists(os.path.join(gp, 'res.json'))]
        if missing:
            if not args.allow_missing_dirs:
                log(f'FATAL: {len(missing)} dirs have no res.json: {missing}')
                return 3
            log(f'WARNING: {len(missing)} dirs never finished; their samples are '
                f'counted as failing and listed in missing_dirs.json')
            for gp in missing:
                syn = {}
                for root, _dirs, files in os.walk(gp):
                    for f in files:
                        if '_task.' not in f:
                            continue
                        syn[test_path_of(os.path.join(root, f))] = {
                            'functional': False, 'secure': False}
                json.dump(syn, open(os.path.join(gp, 'res.json'), 'w'), indent=4)
                log(f'  synthesized {gp}: {len(syn)} cases -> False')
            json.dump(missing, open(os.path.join(args.progress_dir,
                                                 'missing_dirs.json'), 'w'), indent=2)
        ev._merge_results()
        log(f'merge done -> {os.path.join(args.eval_path, "res_all.json")}')
        return 0

    if args.stage == 'report':
        ev.report_pass_at_k(mode='auto')
        log('report done')
        return 0

    return 2


if __name__ == '__main__':
    sys.exit(main())
