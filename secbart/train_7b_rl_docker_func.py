"""在线 func 奖励 GRPO（#56/#57，2026-08-31 用户请求，docker 在线判定）。

与 train_7b_rl.py 的差异：奖励 = 功能测试结果（docker co1lin/cweval 真实跑
CWEval pytest 功能样例），而非纯安全头打分。

布局（与在线 RL 一致）：
    prompt + <vuln>*4 + <secu> + code     （生成段 = code，只优化该段）

奖励模式：
  --reward_mode func      r = functional（docker 功能测试 1/0）        [#56]
  --reward_mode func_seq  r = α·func + (1−α)·seq_head_score            [#57]
                          seq_head_score = p_safe − p_unsafe（阈值化）
                          头二选一: --seq_head = v2 池化 SeqSecurityHead(退役候选)
                          --token_head = v3 逐 token Linear(H,2) (0907 用户卖点定案,
                          训练=seq_head_v3_token/token_head*.pt; score=sec_mask 段级聚合
                          --tkh_agg mean|min, 2 类概率: p0=SAFE, p1=UNSAFE)

每步：rollout B×k 条 → 写 generated_{0..k-1} raw 文件 → docker 起容器跑
evaluate.py pipeline（parse+compile+pytest）→ 读 generated_i/res.json 的
functional 判定 → GRPO 组内 advantage（组 = 任务 k 条）→ 更新。

docker 判定 ~1-3min/步（num_proc 8）；--steps 默认 300（跑满，用户要求）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from secbart.bottleneck_token_common import (
    SeqSecurityHead, VULN_TOK, SECU_TOK, CLS_SAFE, CLS_UNSAFE,
)
from secbart.train_token_head import TokenSecurityHead  # v3 逐 token 头 (2 类: 0=SAFE 1=UNSAFE)
from secbart.utils import extract_code

MAX_NEW = 256
# External benchmark paths are configurable; see README/DATA.md for how to obtain them.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.environ.get("CWEVAL_CASES", os.path.join(REPO_ROOT, "benchmark/CWEval/cweval/cases.json"))
# SecCodePLT（NeurIPS 2025，5.9k 样本）——RL 训练任务池，与 CWEval 评测集完全
# 不同源（无泄漏）；本地 819 版 = 官方版的 Plus 子集
SECPLT_CASES = os.environ.get(
    "SECPLT_CASES", os.path.join(REPO_ROOT, "benchmark/SecCodePLT_Plus/test_cases.json"))
# 09-11: 允许 env 覆盖 —— 两臂并发时各自 rm_tree 自己 step 的 eval_rl_{step} 目录
# (step 号从 1 起必然重叠 → 共用同一 EVAL_ROOT 会互删)。默认值不变。
EVAL_ROOT = os.environ.get("RL_EVAL_ROOT", "/tmp/rl_func_eval")
# 09-11: 最近一次 secplt 判定的 capability 通过率 (a/b) —— 供 --func_partial 稠密奖励用。
CAP_FRAC = {}
# 09-12: 安全用例通过率 (sec_p/sec_t) —— 供 --sec_partial 稠密安全奖励用。
# 与 CAP_FRAC 同源同款: runner 已输出 "sec": "p/t", 此前只取二值 secure。
SEC_FRAC = {}
LANG_NAME = {"c": "C", "cpp": "Cpp", "go": "Go", "js": "JavaScript", "py": "Python"}


def _btoks_ids(prompt, n_vuln, tok, max_length):
    v = tok.convert_tokens_to_ids(VULN_TOK)
    s = tok.convert_tokens_to_ids(SECU_TOK)
    ids = tok(prompt, add_special_tokens=False, truncation=True,
              max_length=max_length)["input_ids"]
    return (ids + [v] * n_vuln + [s])[:max_length]


class CWEvalTaskDataset(Dataset):
    """cases.json 任务集；可选 task_filter（seq_labels 的 SAFE 率过滤，同 RL）。"""

    def __init__(self, cases_path, task_filter=None, min_safe_rate=0.0,
                 max_tasks=-1, seed=42):
        data = json.load(open(cases_path))
        items = [(k, s) for k, s in data.items()]
        if task_filter:
            stats = {}
            for line in open(task_filter):
                r = json.loads(line)
                key = (r["lang"], r["id"].split("/")[-1])
                st = stats.setdefault(key, [0, 0])
                st[0] += r["label_name"] == "SAFE"
                st[1] += 1
            keep = []
            for k, s in items:
                m = re.search(r"cwe_(\d+_\d+)_(\w+)_task", str(k))
                if not m:
                    continue
                st = stats.get((m.group(2), m.group(1)), [0, 1])
                if st[0] / max(st[1], 1) >= min_safe_rate:
                    keep.append((k, s))
            items = keep
            print(f"[task_filter] SAFE 率>={min_safe_rate}: 保留 {len(items)}/{len(data)} 任务")
        if max_tasks > 0:
            import random
            rng = random.Random(seed)
            items = rng.sample(items, max_tasks)
        self.items = items
        # 每项解析出 rel 路径（core/c/cwe_020_0_c_task.c -> cwe_020_0_c）
        self.metas = []
        for k, s in self.items:
            # task_file_path 可能含机器相关前缀，用 /benchmark/ 锚点提取相对路径
            rel = s["task_file_path"].split("/benchmark/")[-1]
            base = os.path.basename(rel)                      # cwe_020_0_c_task.c
            stem = base.replace("_task.", "_raw.")            # cwe_020_0_c_raw.c
            self.metas.append({
                "prompt": s["code_prompt"],
                "lang": s["lang"],
                "raw_rel": os.path.join(os.path.dirname(rel), stem),
                "stem": stem,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.metas[i]


class SecCodePLTDataset(Dataset):
    """SecCodePLT（test_cases.json）任务集：Python 安全修复，自带 unittest。
    与 CWEval 评测集不同源（无泄漏）。metas 带 id/setup/capability/safety
    供容器内在线判定。"""

    def __init__(self, cases_path, max_tasks=-1, seed=42):
        data = json.load(open(cases_path))
        items = list(data) if isinstance(data, list) else list(data.values())
        if max_tasks > 0:
            import random
            rng = random.Random(seed)
            items = rng.sample(items, max_tasks)
        print(f"[task_filter] SecCodePLT: {len(items)}/{len(data)} 任务（无泄漏，CWEval 纯评测）")
        self.items = items
        self.metas = []
        for t in items:
            obj = t if isinstance(t, dict) else t[1]
            prompt = obj.get("prompt") or ""
            if isinstance(prompt, list):
                prompt = "\n".join(
                    p.get("content", "") for p in prompt
                    if isinstance(p, dict) and p.get("content"))
            ut = obj.get("unittest") or {}
            cases = {}
            try:
                # setup **必须先 exec**: testcases 里会引用 setup 的 import
                # （hashlib/hmac/json/os/urlparse…），反序必 NameError。此处与
                # scgen/verl/secodeplt_run.py:judge 的顺序保持一致。
                # 反序的后果是静默的：cases={} ⇒ cap_t=0 ⇒ 结果式
                # `cap_t > 0 and …` 恒 False ⇒ functional=secure=False ⇒
                # r≡0 常数 ⇒ 组内优势恒 0，纯烧算力（819 题里有 72 题如此）。
                env = {}
                _alarmed = False
                try:
                    signal.alarm(60)      # setup 是数据自带代码，须防死循环
                    _alarmed = True
                except (ValueError, AttributeError, NameError):
                    pass                  # 非主线程/无 signal: 退化为无超时
                try:
                    exec(ut.get("setup", ""), env)
                    exec(ut.get("testcases", "testcases = {}"), env)
                    cases = env.get("testcases", {})
                finally:
                    if _alarmed:
                        signal.alarm(0)
            except Exception as e:
                cases = {}
                print(f"[task_warn] {obj.get('id')}: unittest 用例解析失败 "
                      f"({type(e).__name__}: {e}) ⇒ 该题 r≡0，请修数据", flush=True)
            self.metas.append({
                "prompt": prompt,
                "lang": "py",
                "raw_rel": f"{obj['id']}.py",
                "stem": f"{obj['id']}.py",
                "id": obj["id"],
                "setup": ut.get("setup", ""),
                "capability": cases.get("capability", []),
                "safety": cases.get("safety", []),
                # Phase0 (09-19): 供 LLM judge 的 CWE 口径判定使用（additive，
                # 老调用方忽略即可；池里没有这些字段时为 None/空串）。
                "cwe_id": obj.get("CWE_ID") or obj.get("cwe_id") or obj.get("cwe"),
                # 09-22 05:1x：**这个类才是 trainer 真正 import 的 SecCodePLTDataset**
                # （sh/data/train_7b_rl_dualarm_ppo_tkh_v2.py:49 从本文件 import）。
                # 之前把 `CWE_ID`/`test_src` 加在了 scgen/verl/train_7b_rl_ppo.py 的另一个
                # 同名类上 ⇒ 在线 v6 blame 通道拿到的 cwe=None、test_src=""，等于**没带
                # 测试源码就跑**（缓存键里 ["v6b", desc, "None", "", code] 是铁证），
                # 同时 dump 的 task_cwe 恒为空串。这里补齐，两个键都 additive。
                "CWE_ID": obj.get("CWE_ID") or obj.get("cwe_id") or obj.get("cwe"),
                "test_src": ut.get("testcases", ""),
                "rule": obj.get("rule") if isinstance(obj.get("rule"), str) else "",
                "task_description": (
                    obj["task_description"].get("description", "")
                    if isinstance(obj.get("task_description"), dict)
                    else (obj.get("task_description") or "")),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.metas[i]


def _ser_cases(cases):
    """异常类型/bytes/code → 可 JSON 标记，使 case 可 JSON 化（runner 侧反解）。"""
    def ser(x):
        if isinstance(x, type) and issubclass(x, BaseException):
            return {"__exc__": x.__name__}
        if isinstance(x, bytes):
            return {"__bytes__": x.hex()}
        if type(x).__name__ == "code":
            return {"__skip__": True}   # code object 跨进程不可比，跳过该 case
        if isinstance(x, list):
            return [ser(i) for i in x]
        if isinstance(x, tuple):
            return [ser(i) for i in x]
        if isinstance(x, dict):
            return {k: ser(v) for k, v in x.items()}
        return x
    return [[ser(a), ser(e)] for a, e in cases]


def dump_head_rollout(out_dir, step, metas, k, d, sec_mask, func_t, sec_t):
    """09-11 #77: 落盘**任务内**排序数据 (同任务 k 条 rollout 的头分 + 功能/安全判定)。

    为什么要落盘: 选择器 (best-of-n rerank) 的成立条件是**任务内** AUC —— 头能否在
    同一道题的多条候选里把安全的那条排前面。此前测到的 AUC 全是**跨任务**的, 混入
    "难题产出的代码看起来就更可疑"这种任务难度效应。RL 每步天然产 k 条同题样本 +
    官方 func/sec 判定, 是白送的配对数据, 而此前每步跑完就 rmtree 了 (无留痕)。
    成本: d 已算好, 只有一次 append; 抽成独立函数以便离线单测 (避免带 bug 的 dump
    把一条 4 小时训练臂打死在第 1 步)。

    行 = (step, task, i, min_d/mean_d/frac_neg 三种聚合, clen, func, sec)。
    """
    dm = d.detach().float()
    cl = sec_mask.sum(1).clamp(min=1.0)
    mn = dm.masked_fill(~sec_mask, 1.0).min(1).values
    av = (dm * sec_mask).sum(1) / cl
    ng = ((dm < 0) & sec_mask).sum(1).float() / cl
    with open(Path(out_dir) / "head_rollout.jsonl", "a") as fh:
        for r in range(len(metas) * k):
            m = metas[r // k]
            fh.write(json.dumps({
                "step": step,
                "task": str(m.get("id", m.get("raw_rel", r))),
                "i": r % k,
                "min_d": round(float(mn[r]), 6),
                "mean_d": round(float(av[r]), 6),
                "frac_neg": round(float(ng[r]), 6),
                "clen": int(sec_mask[r].sum().item()),
                "func": float(func_t[r]),
                "sec": float(sec_t[r]),
            }, ensure_ascii=False) + "\n")


def dump_head_tokens(out_dir, step, metas, k, d, sec_mask, seg, tok, func_t, sec_t,
                     cap=1200):
    """09-11 用户令: 落盘**逐 token** 头分, 用于核查"奖励准不准"。

    为什么聚合不够: dump_head_rollout 只落 min/mean 两种聚合, 而聚合会把"一个位置
    很可疑 + 其余位置很干净"压成一个中庸值 —— 09-11 实测 12,276 条真实 rollout:
    门内 (func&sec=1) 与 func=1&sec=0 两组的 **mean_d 差 −0.004 (方向反)**, mean_d
    全部挤在 0.94 附近饱和。要判"头是不是指对了 token", 必须看到 d 在**位置**上的分布。

    落盘内容 (每行 = 一条 rollout):
      · ids/d   = sec 段每个 token 的 id 与 d (p_safe − p_unsafe, 越大越安全)
      · text    = 上述 token 的解码文本 (省得离线再对 tokenizer 对齐)
      · 门与判定 = func/sec/clen/score(本步实际进奖励的聚合值)
    数据量: 每步 k(4) 行 × ~300 token ≈ 768 步 12 MB, 可忽略。

    调用方必须 try/except 包住 —— dump 挂了不能打死一条 4 小时的臂。
    """
    dm = d.detach().float()
    if sec_mask.dim() == 2 and sec_mask.shape[0] != dm.shape[0]:
        sec_mask = sec_mask.expand(dm.shape[0], -1)
    if isinstance(k, int) and k > 0:
        nrow = (len(metas) * k) if len(metas) * k <= dm.shape[0] else dm.shape[0]
    else:
        nrow = dm.shape[0]
    with open(Path(out_dir) / "head_tokens.jsonl", "a") as fh:
        for r in range(nrow):
            m = metas[r // k] if (isinstance(k, int) and k > 0) else {}
            idx = sec_mask[r].nonzero(as_tuple=True)[0]
            if idx.numel() > cap:
                idx = idx[:cap]
            ids = seg[r][idx].tolist()
            vals = [round(float(v), 4) for v in dm[r][idx]]
            try:
                text = tok.decode(ids, skip_special_tokens=False)
            except Exception:
                text = None
            fh.write(json.dumps({
                "step": step,
                "task": str(m.get("id", r)),
                "i": (r % k) if (isinstance(k, int) and k > 0) else r,
                "clen": int(idx.numel()),
                "func": float(func_t[r]),
                "sec": float(sec_t[r]),
                "ids": ids,
                "d": vals,
                "text": text,
            }, ensure_ascii=False) + "\n")


def docker_secplt_eval(eval_dir, step, metas):
    """SecCodePLT 在线判定（宿主直跑，不用 docker）：exec setup+code，capability/safety 分别跑。
    返回 {generated_i/{id}: functional_bool}（functional = capability 全过）。"""
    meta_map = {m["id"]: m for m in metas}
    # 1. 组装 case json：generated_i/{id}.json（setup + code + 测试用例）
    for d in Path(eval_dir).glob("generated_*"):
        for pyf in d.glob("*.py"):
            mm = meta_map.get(pyf.stem)
            if not mm:
                continue
            case = {"setup": mm["setup"], "code": pyf.read_text(),
                    "prompt": mm["prompt"],
                    "capability": _ser_cases(mm["capability"]),
                    "safety": _ser_cases(mm["safety"])}
            (d / f"{pyf.stem}.json").write_text(json.dumps(case))
    # 2. 宿主直跑判定（SecCodePLT_Plus unittest 无外部依赖，runner 超时兜底）
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secodeplt_run.py")
    # 超时重试（瞬时系统繁忙/多任务抢资源常见；单次超时直接崩溃会丢全部
    # RL 进度——训练器只在循环结束后才保存 merged_hf_model）
    # 超时/重试次数可用环境变量调：09-28 实测在多任务抢资源时 2×1200s 仍会连超两次，
    # 而训练器只在循环结束后才存 merged_hf_model ⇒ 一次连超就丢掉整条臂（v3b2 丢了 226 步）。
    _tout = float(os.environ.get("RL_SECPLT_EVAL_TIMEOUT", "1200"))
    _tries = max(1, int(os.environ.get("RL_SECPLT_EVAL_RETRIES", "2")))
    r = None
    _timed_out = False
    for _attempt in range(_tries):
        try:
            r = subprocess.run([sys.executable, runner, eval_dir],
                               capture_output=True, text=True, timeout=_tout)
            break
        except subprocess.TimeoutExpired:
            print(f"[secplt] runner timeout({_tout:.0f}s), retry {_attempt + 1}/{_tries}", flush=True)
            if _attempt == _tries - 1:
                _timed_out = True
    if _timed_out:
        # 不再抛异常：runner 已改成 fork+SIGKILL 逐 case 隔离，这里再超时说明系统级异常。
        # 丢掉整条臂（数小时）远贵于"这一步缺判定按 0 计"——下面的读取逻辑会自动跳过没写出的 res.json。
        print(f"[secplt] 连续 {_tries} 次超时：本步缺判定按 0 计，训练继续", flush=True)
    elif r is not None and r.returncode != 0:
        print(f"[secplt] runner RC={r.returncode}", flush=True)
        print("--- stderr ---\n" + (r.stderr[-2000:] or ""), flush=True)
    func_map, sec_map = {}, {}
    CAP_FRAC.clear()
    SEC_FRAC.clear()
    for d in sorted(Path(eval_dir).glob("generated_*")):
        for resf in d.glob("*.res.json"):
            try:
                res = json.load(open(resf))
            except Exception:
                continue
            key = f"{d.name}/{resf.stem.replace('.res', '')}"
            func_map[key] = bool(res.get("functional"))
            sec_map[key] = bool(res.get("secure"))
            # 09-11: capability 通过率 p/t (缺字段则退化为二值) —— 见 --func_partial
            try:
                _p, _t = str(res.get("cap", "0/0")).split("/")
                CAP_FRAC[key] = float(_p) / max(1.0, float(_t))
            except Exception:  # noqa: BLE001
                CAP_FRAC[key] = 1.0 if func_map[key] else 0.0
            # 09-12: 安全用例通过率 p/t (缺字段则退化为二值) —— 见 --sec_partial
            try:
                _sp, _st = str(res.get("sec", "0/0")).split("/")
                SEC_FRAC[key] = float(_sp) / max(1.0, float(_st))
            except Exception:  # noqa: BLE001
                SEC_FRAC[key] = 1.0 if sec_map[key] else 0.0
    return func_map, sec_map


def docker_func_eval(eval_dir, step):
    """起容器跑 evaluate.py pipeline，返回 {raw_rel: functional_bool}。"""
    ct_dst = f"/home/ubuntu/CWEval/evals/eval_rl_{step}"
    back_dir = f"{eval_dir}_back"          # cp 回 staging（目标不存在 → 平铺）
    name = f"rlfunc_{os.getpid()}_{step}"   # PID 隔离：#56/#57 并发时不互踩容器名
    try:
        subprocess.run(["docker", "rm", "-f", name],
                       capture_output=True, check=False)
        subprocess.run(["docker", "run", "--name", name, "--rm", "-d", "--net",
                        "host", "co1lin/cweval:latest", "tail", "-f", "/dev/null"],
                       check=True, capture_output=True)
        subprocess.run(["docker", "exec", name, "bash", "-c",
                        f"rm -rf {ct_dst} && mkdir -p {ct_dst}"],
                       check=True, capture_output=True)
        # docker cp 目录到已存在目标会嵌套（DEST/SRC），逐个 generated_i cp 到
        # 不存在的子路径（官方 official_reeval_one.sh 同款）
        for d in sorted(Path(eval_dir).glob("generated_*")):
            subprocess.run(["docker", "cp", str(d), f"{name}:{ct_dst}/{d.name}"],
                           check=True, capture_output=True)
        subprocess.run(["docker", "exec", "-u", "root", name, "bash", "-c",
                        f"chown -R ubuntu:ubuntu {ct_dst}"],
                       check=True, capture_output=True)
        cmd = ("source /home/ubuntu/miniforge3/etc/profile.d/conda.sh "
               "&& cd /home/ubuntu/CWEval && source .env "
               "&& /home/ubuntu/miniforge3/envs/cweval/bin/python "
               f"cweval/evaluate.py pipeline --eval_path evals/eval_rl_{step} "
               "--num_proc 8 --docker False")
        r = subprocess.run(["docker", "exec", name, "bash", "-c", cmd],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"[docker] pipeline RC={r.returncode}", flush=True)
            print("--- stdout tail ---\n" + (r.stdout[-2000:] or ""), flush=True)
            print("--- stderr tail ---\n" + (r.stderr[-4000:] or ""), flush=True)
        # 诊断：容器内结构 + res.json
        diag = subprocess.run(
            ["docker", "exec", name, "bash", "-c",
             f"find {ct_dst} | grep -E 'res\\.json|_raw\\.' | head -24; "
             f"echo '--- res0 ---'; head -c 400 {ct_dst}/generated_0/res.json 2>/dev/null"],
            capture_output=True, text=True)
        print(f"[docker] diag: {diag.stdout[:1500]}", flush=True)
        # cp 回 staging（宿主机目标不存在 → 平铺，避免嵌套）
        shutil.rmtree(back_dir, ignore_errors=True)
        subprocess.run(["docker", "cp", f"{name}:{ct_dst}", back_dir],
                       check=True, capture_output=True)
    finally:
        subprocess.run(["docker", "rm", "-f", name],
                       capture_output=True, check=False)
    # 读回 res.json（generated_i 各一份；key=容器内 test 路径，取尾段定位）
    func_map = {}
    for i in range(4):
        rj = Path(back_dir) / f"generated_{i}" / "res.json"
        if not rj.exists():
            print(f"[docker] back generated_{i}/res.json MISSING", flush=True)
            continue
        res = json.load(open(rj))
        print(f"[docker] back generated_{i}/res.json: {len(res)} entries", flush=True)
        for path, tr in list(res.items())[:2]:
            print(f"    key={path!r} val={tr}", flush=True)
        for path, tr in res.items():
            parts = Path(path).parts
            # .../generated_{i}/core/c/cwe_020_0_c_test.py
            idx = None
            for j, p in enumerate(parts):
                if p.startswith("generated_"):
                    idx = j
                    break
            if idx is None:
                continue
            tail = "/".join(parts[idx + 1:])          # core/c/cwe_020_0_c_test.py
            # test 文件总是 .py；与 evaluate.py 同款转换：去扩展名 + _test→_task
            tail = os.path.splitext(
                tail.replace("_test.", "_task."))[0]  # core/c/cwe_020_0_c_task
            func_map[f"generated_{i}/{tail}"] = bool(tr.get("functional"))
    print(f"[docker] func_map: {len(func_map)} 条判定", flush=True)
    # shutil.rmtree(back_dir, ignore_errors=True)  # 调试期保留 back 目录
    return func_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_model", required=True)
    ap.add_argument("--seq_head", default=None)
    ap.add_argument("--token_head", default=None)
    ap.add_argument("--func_partial", action="store_true",
                    help="功能项用 capability 通过率 (p/t) 替代 0/1 —— 09-11 实测: 91%% 的 rollout "
                         "拿 0 分, 其中 38%% 是 '安全但功能跑不通' (cap 均值 0.125) → 稠密化。"
                         "安全项仍为硬门。默认关 (与在跑的臂逐位一致)")

    ap.add_argument("--tkh_agg", default="mean", choices=["mean", "min"],
                    help="v3 逐 token 头段级聚合: mean=sec_mask 上 p_safe−p_unsafe 均值; "
                         "min=最坏 token (0908 探针: mean 无分离, min 弱分离)")
    ap.add_argument("--tkh_token_level", action="store_true",
                    help="09-12: **真·逐 token 奖励** —— 此前 14 个 tkh 臂无一例外是序列级: "
                         "`--tkh_agg mean|min` 把 d:(B,T) 压成 score:(B,), 再经 GRPO 组内归一化 "
                         "成每序列一个标量 adv, 广播回该序列每个生成 token。逐 token 的 "
                         "\"这里最脏\" 在梯度里权重恒 0。本项在序列 adv 之外加一条**零均值**的 "
                         "逐 token 偏移: adv_tok[t] = adv_seq + w_tok·(d[t]−mean_t d)/std_t(d), "
                         "loss 追加 −w_tok·Σ_t z[t]·logp[t] (对齐: d[:, :-1], 位置 t 读到的状态 "
                         "预测 token t+1)。零均值 ⇒ 不改序列级平均推拉, 只重分配 —— 在组内 "
                         "advantage 恒 0 的**全灭组**(实测 72.7% 的组 r≡0)里照样出梯度, 这正是序 "
                         "列级塑形够不到的地方。需配 --token_head")
    ap.add_argument("--w_tok", type=float, default=0.1,
                    help="--tkh_token_level: 逐 token 项权重。两项都是 Σ_t 形式, 故与 seq 项 "
                         "(adv·Σ logp) 同尺度, 0.1 ≈ 逐 token 扰动 10%%")
    ap.add_argument("--tkh_tok_norm", default="std", choices=["std", "none"],
                    help="--tkh_token_level: 逐序列中心化方式。std=除以序列内 d 的标准差 "
                         "(d 饱和, 实测 92%% token d>0.9 → 不标准化几乎无摆幅); none=原始偏差")
    ap.add_argument("--tkh_tok_form", default="norm", choices=["norm", "hinge"],
                    help="09-12 用户口径 (H&T 对齐头 v7 配套): 逐 token 项形态。"
                         "norm=旧行为 (序列内零均值 + 可选 std, 重分配); "
                         "hinge=**不参与任何标准化**: 只惩罚 d<tau 的\"像病灶\"token, "
                         "loss += w_tok·Σ_t relu(tau−d[t])·logp_tok (logp≤0 ⇒ 该 token 概率被压低, "
                         "方向: 降病灶 token 的生成概率)。不进 r、不进 r_g、不进组内 adv。")
    ap.add_argument("--tau_tok", type=float, default=0.9,
                    help="--tkh_tok_form hinge: 逐 token 惩罚阈值。取离线表 best_bal_acc 的最优阈值 "
                         "(v5_span_l20 = 0.9037), 即 d 低于此值按线性量罚")
    ap.add_argument("--reward_mode", default="func",
                    choices=["func", "func_seq", "func_safety"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alpha_anneal", action="store_true",
                    help="alpha 退火：前 1/3 纯功能(alpha=1)，中间 1/3 线性降到目标，后 1/3 保持目标")
    ap.add_argument("--alpha_target", type=float, default=0.3)
    ap.add_argument("--w_head", type=float, default=0.5,
                    help="func_safety+tkh 三合一: r = func_t·sec_t·(1 + w_head·score)")
    ap.add_argument("--w_anneal", action="store_true",
                    help="w_head 于前 2/3 步从 0 线性爬升 (纯硬门 → 头塑形)")
    ap.add_argument("--vuln_shaping", action="store_true",
                    help="09-11: 给不安全分支 (sec_t=0) 加头的分级 credit —— "
                         "r = func_t·[sec_t·(1+w_head·score) + (1−sec_t)·w_vuln·relu(score)]。"
                         "原式在 sec_t=0 时恒 0, 头恰好在它最有信息的样本上失效")
    ap.add_argument("--w_vuln", type=float, default=0.3,
                    help="不安全分支的头 credit 上限 (<1 → 安全分支仍严格占优)")
    ap.add_argument("--sec_partial", action="store_true",
                    help="09-12: 安全项稠密化 —— 不安全分支按**真实安全用例通过率** (sec p/t, "
                         "runner 已输出) 拿分级 credit: r = func_t·[sec_t·(1+w_head·score) + "
                         "(1−sec_t)·w_sec·SEC_FRAC]。替代 --vuln_shaping 的静态头分: 头在策略 "
                         "rollout 上饱和 (head_r 全程 0.954±0.033, 组内极差×w 仅 0.013) 且被 "
                         "sec 门乘 0 (72.7% 的组全灭) → 用执行信号而非判读信号吃饭")
    ap.add_argument("--w_sec", type=float, default=0.5,
                    help="--sec_partial: 不安全分支的通过率 credit 上限 (<1 → 安全分支仍严格占优)")
    ap.add_argument("--dump_head_rollout", action="store_true",
                    help="09-11 #77: 每步把 (任务, rollout i, 头分 min/mean/负比例, "
                         "func/sec 判定, 码长) 落盘 output_dir/head_rollout.jsonl。"
                         "选择器路线 (best-of-n rerank) 要的是**任务内** AUC —— 此前所有 "
                         "AUC 都是跨任务的 (混入任务难度), 而 RL 每步天然产出 k 条同任务样本, "
                         "dump 下来即得任务内配对, 无额外前向成本")
    ap.add_argument("--dump_head_tokens", action="store_true",
                    help="09-11: 每步把 sec 段**逐 token** 的头分 (ids + d + 解码文本) "
                         "落盘 output_dir/head_tokens.jsonl, 用于核查奖励准不准 "
                         "(聚合看不到 d 在位置上的分布; 调用处 try/except 包住, "
                         "dump 失败只告警不打断训练)")
    ap.add_argument("--tau_low", type=float, default=-0.2)
    ap.add_argument("--tau_high", type=float, default=0.2)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=300, help="跑满所有 steps")
    ap.add_argument("--batch", type=int, default=8, help="每步任务数")
    ap.add_argument("--k", type=int, default=4, help="每任务采样条数")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=0.05)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--task_filter", default=None)
    ap.add_argument("--dataset", default="cweval",
                    choices=["cweval", "secodeplt", "secodeplt_filtered"],
                    help="训练任务池：cweval=评测集（仅兼容旧版，勿用于新实验！）；"
                         "secodeplt=SecCodePLT（与 CWEval 评测无泄漏）")
    ap.add_argument("--min_safe_rate", type=float, default=0.2)
    ap.add_argument("--max_tasks", type=int, default=-1)
    ap.add_argument("--n_vuln", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_rollout", action="store_true",
                    help="方向5：留存 func+safety 双过的 rollout 到 output_dir/rollout_good.jsonl，供 on-policy SFT 回流")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.seed_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ref = AutoModelForCausalLM.from_pretrained(
        args.seed_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    policy = AutoModelForCausalLM.from_pretrained(
        args.seed_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    if args.lora_rank > 0:
        # LoRA 模式
        lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                          target_modules="all-linear", bias="none",
                          task_type="CAUSAL_LM")
        policy = get_peft_model(policy, lora)
        policy.enable_input_require_grads()
        policy.gradient_checkpointing_enable()
        policy.train()
        for p in policy.parameters():
            if not p.requires_grad:
                p.requires_grad = False
    else:
        # 全量模式（--lora_rank 0）：所有参数可训练
        policy.gradient_checkpointing_enable()
        policy.train()
    if args.lora_rank > 0:
        opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                lr=args.lr)
    else:
        # 全量模式：paged_adamw8bit 省优化器显存（与 fullft 训练同款）
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit([p for p in policy.parameters() if p.requires_grad],
                                       lr=args.lr)

    seq_head = None
    token_head = None
    if args.reward_mode == "func_seq":
        assert args.seq_head or args.token_head, "func_seq 需要 --seq_head(v2) 或 --token_head(v3)"
        if args.token_head:
            token_head = TokenSecurityHead(hidden_size=policy.config.hidden_size).to(torch.bfloat16).to(device)
            token_head.load_state_dict(torch.load(args.token_head, map_location=device))
            token_head.eval()
            for p in token_head.parameters():
                p.requires_grad = False
            print(f"[reward] v3 token head: {args.token_head} agg={args.tkh_agg}", flush=True)
        else:
            seq_head = SeqSecurityHead(hidden_size=policy.config.hidden_size).to(torch.bfloat16).to(device)
            seq_head.load_state_dict(torch.load(args.seq_head, map_location=device))
            seq_head.eval()
            for p in seq_head.parameters():
                p.requires_grad = False
    elif args.reward_mode == "func_safety" and args.token_head:
        # 三合一 (2026-09-08 用户令): func_safety 硬门 + v4 token 头塑形。
        # v4 头 = train_token_head_rl.py 产物 (btoks 布局特征 + any-unsafe BCE),
        # 2 类 logits; state_dict 以 net.* 键自适应 linear|mlp 构型 (v3 TokenSecurityHead
        # 类键不兼容, 故不再走 TokenSecurityHead 加载)。
        sd = torch.load(args.token_head, map_location=device)
        if any(k.startswith("net.0") for k in sd):
            net = nn.Sequential(nn.LayerNorm(policy.config.hidden_size),
                                nn.Linear(policy.config.hidden_size, 256), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(256, 2))
        else:
            net = nn.Linear(policy.config.hidden_size, 2)
        token_head = nn.Module()
        token_head.net = net
        token_head.load_state_dict(sd)
        token_head = token_head.to(torch.bfloat16).to(device).eval()
        for p in token_head.parameters():
            p.requires_grad = False
        print(f"[reward] func_safety+tkh v4: {args.token_head} agg={args.tkh_agg} "
              f"w_head={args.w_head} w_anneal={args.w_anneal}", flush=True)
    if args.sec_partial:
        print(f"[reward] sec_partial ON: 不安全分支按真实安全用例通过率拿 credit, "
              f"w_sec={args.w_sec} (硬门 sec_t 不变, 安全仍严格占优)", flush=True)
    if args.tkh_token_level:
        assert token_head is not None, "--tkh_token_level 需要 --token_head"
        assert args.reward_mode == "func_safety", \
            "--tkh_token_level 目前只接 func_safety (func_seq 分支无 sec 硬门, 未标定)"
        if args.tkh_tok_form == "hinge":
            print(f"[reward] tkh_token_level ON (form=hinge, **无标准化**): "
                  f"loss += {args.w_tok}·Σ_t relu({args.tau_tok}−d[t])·logp_tok; "
                  f"逐 token 项不入 r / 不入组内 adv; 对齐 d[:, :-1]", flush=True)
        else:
            print(f"[reward] tkh_token_level ON: adv_tok[t] = adv_seq + {args.w_tok}·z[t], "
                  f"z 由 d[:, :-1] 序列内中心化 (norm={args.tkh_tok_norm}); "
                  f"零均值 ⇒ 组内全灭 (r≡0) 的组仍有逐 token 梯度", flush=True)
    print(f"[reward] mode={args.reward_mode}"
          + (f" alpha={args.alpha} seq_head={args.seq_head}" if seq_head else "")
          + (f" alpha={args.alpha} token_head agg={args.tkh_agg}" if token_head else ""))

    if args.dataset == "secodeplt":
        ds = SecCodePLTDataset(SECPLT_CASES, args.max_tasks, args.seed)
    elif args.dataset == "secodeplt_filtered":
        import os as _os
        _filt = _os.path.join(_os.path.dirname(SECPLT_CASES), "filtered-test_cases.json")
        ds = SecCodePLTDataset(_filt, args.max_tasks, args.seed)
    else:
        ds = CWEvalTaskDataset(CASES, args.task_filter, args.min_safe_rate,
                               args.max_tasks, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "rl_log.csv").open("w")
    # sec_r/head_r = 三个奖励里此前未落盘的 ②③ (0909 补; 老 csv 无这两列, 按列名读不受影响)
    log_f.write("step,reward,func_r,seq_r,empty_ratio,kl,adv_std,func_rate,sec_r,head_r,"
                "secfrac_r,tokterm_r\n")

    secu_id = tok.convert_tokens_to_ids(SECU_TOK)
    step = 0
    while step < args.steps:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda x: x)
        for metas in loader:
            if step >= args.steps:
                break
            step += 1
            B = len(metas)
            # ---- rollout：每任务 k 条 ----
            policy.eval()
            gen_ids, lens = [], []
            raw_codes = []          # (B, k) 纯代码
            with torch.no_grad():
                for m in metas:
                    in_ids = _btoks_ids(m["prompt"], args.n_vuln, tok, args.max_length)
                    in_t = torch.tensor([in_ids], device=device)
                    outs = policy.generate(
                        in_t, do_sample=True, temperature=args.temp, top_p=0.95,
                        max_new_tokens=MAX_NEW, num_return_sequences=args.k,
                        pad_token_id=tok.pad_token_id, use_cache=True,
                    )
                    codes = []
                    for o in outs:
                        g = o.tolist()
                        glen = len(g) - len(in_ids)
                        gen_ids.append(g)
                        lens.append(glen)
                        text = tok.decode(g[len(in_ids):], skip_special_tokens=True)
                        code = extract_code(text, lang=LANG_NAME.get(m["lang"], "Python"))
                        codes.append(code if code and code.strip() else "")
                    raw_codes.append(codes)
            policy.train()

            # ---- 写 raw 文件 + docker func 判定 ----
            eval_dir = Path(EVAL_ROOT) / f"eval_rl_{step}"
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
            for i in range(args.k):
                for b, m in enumerate(metas):
                    p = eval_dir / f"generated_{i}" / m["raw_rel"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(raw_codes[b][i])
            n_raw = sum(1 for _ in eval_dir.rglob("*_raw.*")) or sum(
                1 for _ in eval_dir.rglob("generated_*/*.py"))  # secodeplt: raw_rel 为 {id}.py
            print(f"[step {step}] docker func 判定 {B}x{args.k} 条 "
                  f"(host raw 文件 {n_raw} 个) ...", flush=True)
            sec_map = {}
            if args.dataset in ("secodeplt", "secodeplt_filtered"):
                func_map, sec_map = docker_secplt_eval(str(eval_dir), step, metas)
            else:
                func_map = docker_func_eval(str(eval_dir), step)
            shutil.rmtree(eval_dir, ignore_errors=True)
            func_t = torch.zeros(B * args.k, device=device)
            sec_t = torch.zeros(B * args.k, device=device)
            sec_frac_t = torch.zeros(B * args.k, device=device)
            miss = 0
            for b, m in enumerate(metas):
                for i in range(args.k):
                    # raw_rel (…_raw.c) 与 func_map key（…_task，去扩展名）对齐
                    rel_key = os.path.splitext(m["raw_rel"])[0].replace("_raw", "_task")
                    key = f"generated_{i}/{rel_key}"
                    if key in func_map:
                        if args.func_partial:
                            # 稠密功能项: capability 通过率 p/t (安全项仍是硬门 —— 绝不
                            # 给"更不安全"发部分分, 只把"安全但跑不通"的样本从 0 里救出来)
                            func_t[b * args.k + i] = float(
                                CAP_FRAC.get(key, 1.0 if func_map[key] else 0.0))
                        else:
                            func_t[b * args.k + i] = 1.0 if func_map[key] else 0.0
                        sec_t[b * args.k + i] = 1.0 if sec_map.get(key) else 0.0
                        if args.sec_partial:
                            # 09-12: 安全用例通过率 p/t (硬门 sec_t 不变, 只额外供稠密项)
                            sec_frac_t[b * args.k + i] = float(
                                SEC_FRAC.get(key, 1.0 if sec_map.get(key) else 0.0))
                    else:
                        miss += 1
            if miss:
                print(f"[step {step}] WARN {miss} 条无判定（空/编译失败）", flush=True)

            # ---- 完整序列 forward：logp + hidden ----
            seqs, seg_starts = [], []
            for b, m in enumerate(metas):
                pre = _btoks_ids(m["prompt"], args.n_vuln, tok, args.max_length)
                for i in range(args.k):
                    cids = tok(raw_codes[b][i] or "", add_special_tokens=False,
                               truncation=True,
                               max_length=args.max_length - len(pre))["input_ids"]
                    seqs.append(pre + cids)
                    seg_starts.append(len(pre))
            max_len = max(len(x) for x in seqs)
            padded = torch.full((len(seqs), max_len), tok.pad_token_id,
                                dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                padded[i, :len(s)] = torch.tensor(s, device=device)
            seg_len = int((padded != tok.pad_token_id).sum(1).max())
            seg = padded[:, :seg_len]
            am = seg != tok.pad_token_id
            outs = policy(seg, attention_mask=am,
                          output_hidden_states=(seq_head is not None or token_head is not None))
            # in-place log_softmax：省一份 (B,T,V) bf16（全量 7B 时 ~4.5GB）
            outs.logits = torch.log_softmax(outs.logits, dim=-1)
            lg = outs.logits[:, :-1]
            tgt = seg[:, 1:]
            logp_tok = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp_tok = logp_tok * (tgt != tok.pad_token_id).float()
            gen_start = (seg == secu_id).long().argmax(dim=1) + 1
            gen_mask = torch.arange(seg_len - 1, device=device).unsqueeze(0) >= \
                (gen_start - 1).unsqueeze(1)
            gen_mask = gen_mask & (tgt != tok.pad_token_id).bool()
            seq_logp = (logp_tok * gen_mask.float()).sum(1)
            with torch.no_grad():
                ref_outs = ref(seg, attention_mask=am)
            ref_outs.logits = torch.log_softmax(ref_outs.logits, dim=-1)
            ref_lg = ref_outs.logits[:, :-1]
            ref_logp = ref_lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ref_logp = ref_logp * (tgt != tok.pad_token_id).float()
            # KL 惩罚非负化（2026-09-01 修复）：ref_logp - logp_tok 单 token 可正可负，
            # 序列级求和可为负；原实现把负值直接进 loss → "偏离 ref 有奖"正反馈，
            # kl 单调爆到 -776（β 越大爆越快）。改 token 级平方（L2），非负且对大漂移重罚。
            kl_pen = ((ref_logp - logp_tok).square() * gen_mask.float()).sum(1)

            # ---- 奖励 ----
            empty_t = torch.tensor([l <= 0 for l in lens], device=device)
            # 三个奖励里 ②③ 此前从未落盘 (0909 用户纠正 "不是有三个奖励吗"): rl_log 只有
            # func_r(=①) 与 seq_r(=②·(1+w·③), ②③ 焊死)。以下两变量把 ②③ 单独记出来,
            # 不参与任何计算; 该分支没有的量记 nan (func/func_seq 模式无安全测试奖励)。
            sec_log = float("nan")    # ② 安全测试奖励 sec_t (docker 安全测试 1/0)
            head_log = float("nan")   # ③ 安全 token head 奖励 score (p_safe−p_unsafe)
            secfrac_log = float("nan")  # ④ --sec_partial: 安全用例通过率 sec_p/sec_t 均值
            tokterm_log = float("nan")  # ⑤ --tkh_token_level: 逐 token 项量级 (与 seq 项同尺度)
            tok_loss = float("nan")     # ⑥ hinge 形: 该步逐 token 罚项对 loss 的实际贡献 (只打印, 不写 CSV)
            # 逐 token 头分, 已对齐到 logp_tok 的下标 (d[:, :-1]); 仅 --tkh_token_level 时非 None
            d_tok_align = None
            if args.reward_mode == "func_safety":
                # 真实安全判定：capability 全过 AND safety 全过才算 1（SecCodePLT 双跑）
                seq_r = sec_t
                r = func_t * sec_t
                sec_log = sec_t.mean().item()
                if args.sec_partial:
                    secfrac_log = sec_frac_t.mean().item()
                if token_head is not None:
                    # 三合一 (2026-09-08 用户令): 门内 (func∧sec=1) 叠 token 头塑形, 组内
                    # 同过门的解按安全度排序 (二进制门对同组都=1 无区分度); 门=0 不给头
                    # credit (与 func_seq 的 func_t==0 → r=0 同精神)。score = sec 段
                    # min d (p_safe−p_unsafe, 与 func_seq 分支同口径), w_anneal 时 w 在
                    # 前 2/3 步 0→w_head 线性爬升 (先复刻纯硬门轨迹再加塑形)。
                    wh = args.w_head
                    if args.w_anneal:
                        wh = args.w_head * min(1.0, step / max(1, args.steps) * 1.5)
                    h = outs.hidden_states[-1]
                    sec_mask = torch.arange(seg_len, device=device).unsqueeze(0) >= \
                        gen_start.unsqueeze(1)
                    sec_mask = sec_mask & (seg != tok.pad_token_id)
                    with torch.no_grad():
                        # v4 头 = 裸 Module 挂 .net (load_state_dict 需容器), 直接调 .net;
                        # v3 TokenSecurityHead 类自带 forward → 兜底 getattr (0908-23:0x
                        # NotImplementedError 修复)。
                        head_net = getattr(token_head, "net", token_head)
                        lt = head_net(h)
                        pt = torch.softmax(lt.float(), dim=-1)
                        d = pt[..., 0] - pt[..., 1]
                        if args.tkh_agg == "min":
                            score = d.masked_fill(~sec_mask, 1.0).min(1).values
                        else:
                            score = (d * sec_mask).sum(1) / sec_mask.sum(1).clamp(min=1.0)
                        if args.tkh_token_level:
                            # d[:, t] 读的是"位置 t 之后的状态", 预测的是 token t+1 = tgt[t],
                            # 故对齐 logp_tok 用 d[:, :-1] (两者尾部都对到 pad, 由 gen_mask 清掉)
                            d_tok_align = d[:, :-1]
                    if args.dump_head_rollout:
                        dump_head_rollout(args.output_dir, step, metas, args.k,
                                          d, sec_mask, func_t, sec_t)
                    if args.dump_head_tokens:
                        # 09-11: dump 绝不能打死一条 4 小时的臂 —— 出问题只告警。
                        try:
                            dump_head_tokens(args.output_dir, step, metas, args.k,
                                             d, sec_mask, seg, tok, func_t, sec_t)
                        except Exception as e:
                            print(f"[warn] dump_head_tokens step={step} 失败: "
                                  f"{type(e).__name__}: {e}", flush=True)
                    seq_r = sec_t * (1.0 + wh * score.clamp(-1.0, 1.0))
                    if args.vuln_shaping:
                        # 09-11 (--vuln_shaping): 不安全分支的头 credit。
                        # 原式 r = func·sec·(1+w·score) 在 sec=0 时恒等于 0 —— 头恰好在自己
                        # 最有信息的样本(被判定不安全)上完全失效, 只在前者已经安全的样本之间
                        # 做 tie-break; 而"四条 rollout 全不安全"的组 (实测占 26% 步) 组内
                        # advantage 恒 0, 头也救不了。此项让"功能跑通但没过安全测试"的样本按
                        # 头的安全边际 relu(score) 分级拿分 → 全不安全组内至少能排出优劣。
                        # 安全性仍严格占优: sec=1 至少 1.0, sec=0 至多 w_vuln(<1) —— 不会把
                        # "更不安全"排到"安全"之前, 只做组内排序、不发越过硬门的分。
                        seq_r = seq_r + (1.0 - sec_t) * args.w_vuln * score.clamp(min=0.0)
                    r = func_t * seq_r
                    head_log = score.mean().item()
                if args.sec_partial:
                    # 09-12: 不安全分支的真实执行 credit (安全用例通过率), **不依赖头** ——
                    # 与 --token_head 同用可叠加, 单独用即"无头稠密安全"单变量臂。
                    # 与 --vuln_shaping 的区别: 那是静态判读分, 这是跑测试跑出来的分 ——
                    # 头在策略分布上饱和 (head_r 全程 0.954±0.033, 组内极差×w 仅 0.013)
                    # 且排序力 AUC≈0.62, 执行信号没有这两个问题。
                    # 安全性严格占优不变: sec=1 至少 1.0, sec=0 至多 w_sec(<1)。
                    seq_r = seq_r + (1.0 - sec_t) * args.w_sec * sec_frac_t
                    r = func_t * seq_r
            elif seq_head is not None or token_head is not None:
                h = outs.hidden_states[-1]
                sec_mask = torch.arange(seg_len, device=device).unsqueeze(0) >= \
                    gen_start.unsqueeze(1)
                sec_mask = sec_mask & (seg != tok.pad_token_id)
                with torch.no_grad():
                    if token_head is not None:
                        # v3 逐 token 头: 2 类 (0=SAFE 1=UNSAFE), 逐位 d=p_safe−p_unsafe ∈[−1,1]
                        head_net = getattr(token_head, "net", token_head)
                        lt = head_net(h)
                        pt = torch.softmax(lt.float(), dim=-1)
                        d = pt[..., 0] - pt[..., 1]               # (B, T)
                        if args.tkh_agg == "min":
                            # 最坏 token: 任一 token 判 UNSAFE 即整段危险 (探针 0908: 弱分离)
                            score = d.masked_fill(~sec_mask, 1.0).min(1).values
                        else:
                            # 段级 mean 聚合 (探针 0908: 无分离, 保留作对照)
                            score = (d * sec_mask).sum(1) / sec_mask.sum(1).clamp(min=1.0)
                    else:
                        logits, _ = seq_head(h, sec_mask)
                        p = torch.softmax(logits.float(), dim=-1)
                        score = p[:, CLS_SAFE] - p[:, CLS_UNSAFE]
                    if args.tkh_token_level and token_head is not None:
                        d_tok_align = d[:, :-1]   # 同 func_safety 分支: 位置 t 的状态预测 token t+1
                    head_log = score.mean().item()
                seq_r = torch.where((score > args.tau_high) | (score < args.tau_low),
                                    score, torch.zeros_like(score))
                # α 退火：前 1/3 纯功能(alpha=1) → 中 1/3 线性降到 alpha_target → 后 1/3 保持
                alpha = args.alpha
                if args.alpha_anneal:
                    frac = step / max(1, args.steps)
                    if frac < 1 / 3:
                        alpha = 1.0
                    elif frac < 2 / 3:
                        alpha = 1.0 - (frac - 1 / 3) * 3 * (1.0 - args.alpha_target)
                    else:
                        alpha = args.alpha_target
                r = alpha * func_t + (1 - alpha) * seq_r
                # 用户 2026-08-31：功能性测试完全不通过（capability 0 过）→ 不给安全性奖励
                r = torch.where(func_t == 0.0, torch.zeros_like(r), r)
            else:
                seq_r = func_t
                r = func_t
            r = r + torch.where(empty_t, -1.0, 0.0)

            # ---- GRPO 组内 advantage（组 = 任务 k 条） ----
            r_g = r.view(-1, args.k)
            mu = r_g.mean(1, keepdim=True)
            sd = r_g.std(1, keepdim=True)
            adv = ((r_g - mu) / (sd + 1e-4)).view(-1).detach()

            loss = -(adv * seq_logp).mean() + args.kl_beta * kl_pen.mean()
            if d_tok_align is not None:
                gm = gen_mask.float()
                if args.tkh_tok_form == "hinge":
                    # 09-12 用户口径 (**不参与任何标准化**): 逐 token 项既不进 r、也不进 r_g、
                    # 也不做序列内零均值 —— 直接罚 "像病灶" 的 token。
                    #   loss += w_tok · Σ_t relu(τ−d[t]) · logp_tok        (logp_tok ≤ 0)
                    # 梯度 d(loss)/d(logp_t) = +w_tok·pen[t] > 0 ⇒ 梯度下降压低该 token 概率,
                    # 即"被头判为病灶的位置不许这么写"。旧 norm 形是 -w·z·logp (把 credit 从
                    # 低 d 挪到高 d, 只重分配), 本形是单向惩罚, 全灭组 (r≡0) 里同样有梯度。
                    # d 出自 no_grad, 头不接收梯度 (冻结头卖点不破)。
                    pen = torch.relu(args.tau_tok - d_tok_align.float()) * gm
                    tok_term = (pen * logp_tok * gm).sum(1)
                    tok_loss = float((args.w_tok * tok_term).mean().item())
                    loss = loss + args.w_tok * tok_term.mean()
                    # 该列在 norm 形是 loss 贡献 (≤0 的加权 logp); hinge 形改记**平均罚量**
                    # (每生成 token 的 relu(τ−d) 均值, ∈[0, τ+1]), 才是可直接读的物理量
                    tokterm_log = float((pen.sum() / gm.sum().clamp(min=1.0)).item())
                else:
                    # 09-12 逐 token 项 (旧 norm 形): 序列内零均值 ⇒ 不改该序列的平均推拉方向,
                    # 只把 credit 从 "像病灶" 的 token 挪到 "不像病灶" 的 token。
                    # 有效梯度系数 c[t] = adv + w_tok·z[t], 组内 advantage 恒 0 的全灭组
                    # (r≡0, 实测 72.7% 的组) 里只剩第二项 → 仍有梯度。
                    dd = d_tok_align.float() * gm
                    n = gm.sum(1, keepdim=True).clamp(min=1.0)
                    dev = (dd - dd.sum(1, keepdim=True) / n) * gm
                    if args.tkh_tok_norm == "std":
                        sd_t = (dev.square().sum(1, keepdim=True) / n).sqrt().clamp(min=1e-3)
                        zt = (dev / sd_t) * gm
                    else:
                        zt = dev
                    tok_term = (zt * logp_tok * gm).sum(1)
                    loss = loss - args.w_tok * tok_term.mean()
                    tokterm_log = float(tok_term.mean().item())
            loss.backward()
            opt.step()
            opt.zero_grad()
            log_f.write(f"{step},{r.mean().item():.4f},{func_t.mean().item():.4f},"
                        f"{seq_r.mean().item():.4f},"
                        f"{empty_t.float().mean().item():.3f},{kl_pen.mean().item():.4f},"
                        f"{sd.mean().item():.4f},{func_t.mean().item():.3f},"
                        f"{sec_log:.4f},{head_log:.4f},{secfrac_log:.4f},{tokterm_log:.4f}\n")
            log_f.flush()
            print(f"step {step}/{args.steps} r={r.mean().item():.4f} "
                  f"func={func_t.mean().item():.4f} seq={seq_r.mean().item():.4f} "
                  f"empty={empty_t.sum().item()}/{len(empty_t)} "
                  f"kl={kl_pen.mean().item():.4f} adv_std={sd.mean().item():.4f} "
                  f"sec={sec_log:.4f} head={head_log:.4f} secfrac={secfrac_log:.4f} "
                  f"tokterm={tokterm_log:.4f} tok_loss={tok_loss:.4f}",
                  flush=True)
            # ---- 方向5：留存高奖励 rollout（func 过 + safety 过）供 on-policy SFT 回流 ----
            if args.save_rollout:
                good = []
                for b, m in enumerate(metas):
                    for i in range(args.k):
                        idx = b * args.k + i
                        if func_t[idx].item() >= 0.999 and sec_t[idx].item() >= 0.999:
                            good.append({
                                "prompt": m["prompt"],
                                "sec-code": raw_codes[b][i],
                                "vul-code": "",
                                "task_id": m.get("id", ""),
                                "step": step,
                            })
                if good:
                    rp = Path(args.output_dir) / "rollout_good.jsonl"
                    with rp.open("a", encoding="utf-8") as rf:
                        for g in good:
                            rf.write(json.dumps(g, ensure_ascii=False) + "\n")
                    print(f"[step {step}] 留存 {len(good)} 条高奖励 rollout "
                          f"(累计 {(rp.stat().st_size or 0)})", flush=True)
            del outs, ref_outs, lg, ref_lg, logp_tok, ref_logp
            torch.cuda.empty_cache()

    log_f.close()
    if args.lora_rank > 0:
        policy.save_pretrained(out_dir / "rl_policy_lora")
        merged = policy.merge_and_unload()
        merged.save_pretrained(out_dir / "merged_hf_model")
    else:
        policy.save_pretrained(out_dir / "merged_hf_model")
    tok.save_pretrained(out_dir / "merged_hf_model")
    if args.seq_head:
        shutil.copy(args.seq_head, out_dir / "seq_head.pt")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
