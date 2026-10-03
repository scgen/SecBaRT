#!/usr/bin/env python3
"""PPO-TKH (dualarm + 逐 token 头奖励) 入口 + **ndarray 安全序列化**（路线②：不改共享脚本）。

病根（09-14 17:5x 定位，09-15 10:4x 在 PPO 臂上再次复现）: 819 池里有且只有 **1 题**
（id `bcce7d57`，CWE-77）的 capability 期望值会构造 numpy **object 数组**
（sympy Symbol 装进 ndarray）。共享脚本 `train_7b_rl_docker_func.py:303-304` 把 case
原样 `json.dumps` ⇒ `TypeError: Object of type ndarray is not JSON serializable`
⇒ `docker_secplt_eval` 抛异常 ⇒ 整臂 rc=1。

题序按 `--seed` 洗牌 ⇒ **命中步固定**（seed 42 → step 185，seed 1234 → step 288；
与 D2 五臂的 s777→10 / s42→115 / s1234→193 是同一题、不同洗牌序）。
09-15 01:2x–04:5x 起的 **6 条 PPO 臂 6/6 全灭于此**，且每条都死在 185/288 ——
因为 wrapper 的 5 次重试每次重跑同一 seed，必然撞同一题。

修法: 在 `main()` 之前把模块里的 `_ser_cases` 包一层"先把 ndarray 拆成 list"的前处理，
`__exc__` / `__bytes__` / `__skip__` 三种标记语义**逐字沿用原实现**（本文件与
`sh/data/run_d2_95_ndsafe.py` 的 `_devec` / `_has_skip` / `_ser_cases_safe` 同源）。

为什么能生效: `docker_secplt_eval` 内部对 `_ser_cases` 是**模块全局查找**（`:303`），
所以无论调用方是 `from ... import docker_secplt_eval` 还是 `M.docker_secplt_eval`，
补丁都落在 `train_7b_rl_docker_func` 的模块命名空间上 ⇒ PPO 训练器无需改动一个字。

用法: 与 `-m secbart.train_7b_rl_dualarm_ppo_tkh` 完全相同（argv 原样透传）。
自检: `.venv-cu121/bin/python sh/data/run_rl_ppo_tkh_ndsafe.py --selfcheck`
"""
import os
import sys

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, W)

import numpy as np  # noqa: E402

import secbart.train_7b_rl_docker_func as D  # noqa: E402

_orig_ser_cases = D._ser_cases
_SKIP = {"__skip__": True}


def _devec(x):
    """递归把 ndarray / numpy 标量换成 JSON 基本类型；**跨进程不可还原的对象**
    （sympy Symbol 等）换成既有 `__skip__` 标记（与 code object 同款语义）。"""
    if isinstance(x, np.ndarray):
        return _devec(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_devec(i) for i in x]
    if isinstance(x, dict):
        return {k: _devec(v) for k, v in x.items()}
    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        return x
    if isinstance(x, type) and issubclass(x, BaseException):
        return x                      # 异常类型交回原实现打 __exc__
    if type(x).__name__ == "code":
        return x                      # code object 交回原实现打 __skip__
    return dict(_SKIP)


def _has_skip(x):
    if isinstance(x, dict):
        return bool(x.get("__skip__")) or any(_has_skip(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_skip(i) for i in x)
    return False


def _ser_cases_safe(cases):
    """先拆不可 JSON 化的叶子，再走原 `_ser_cases` 的标记语义。

    最后一步必要: **期望值里含 skip 时，把整条 case 的期望也标成 skip**。
    因为 runner 的 `_run_case` 比较 `out == expected` 发生在 try 之外 —— 若期望是
    「含 dict 的 list」而 out 是 numpy 数组，`==` 会给数组、被真值判断抛
    ValueError，直接炸 runner；标成 `__skip__` 后 runner 按既有语义 `continue`
    （不计数），该题 capability 计 0 ⇒ 与"用例解析失败"的既有处理一致。
    """
    out = _orig_ser_cases(_devec(cases))
    for pair in out:
        if len(pair) > 1 and _has_skip(pair[1]):
            pair[1] = dict(_SKIP)
    return out


D._ser_cases = _ser_cases_safe


def _selfcheck():
    """拿真池子量两个数，**必须分开报**：

      ① `bare`  —— 裸 testcase 对象直接 json.dumps 会抛错的题数。
      ② `real`  —— **训练真路径**（先过原 `_ser_cases`，再 json.dumps）仍会抛错的题数。
         `train_7b_rl_docker_func.py:303-304` 走的就是这条，所以**只有 ② 才是会崩臂的数**。

    ⚠ 09-14 的教训（[[d2-ndarray-json-bug]]）: 只报 ① 会把 398 当成"398 题会崩"，
    而真路径实测只有 1 题（`bcce7d57`）—— 因为原 `_ser_cases` 已把 code object /
    异常类型等换成标记，① 里绝大多数在真路径上根本不抛。
    """
    import json
    import signal
    pool = D.SECPLT_CASES
    items = list(json.load(open(pool)))
    bare = real = 0
    for obj in items:
        env = {}
        ut = obj.get('unittest') or {}
        try:
            signal.alarm(60)
            try:
                exec(ut.get('setup', ''), env)
                exec(ut.get('testcases', 'testcases = {}'), env)
            finally:
                signal.alarm(0)
        except Exception:
            continue
        cases = env.get('testcases', {})
        if not cases:
            continue
        raw = {'capability': cases.get('capability'), 'safety': cases.get('safety')}
        try:
            json.dumps(raw)
            continue                      # ① 就不抛，与本补丁无关
        except TypeError:
            bare += 1
        # ② 真路径: 原 _ser_cases -> json.dumps。抛错 = 会崩臂。
        try:
            json.dumps({'capability': _orig_ser_cases(raw['capability']),
                        'safety': _orig_ser_cases(raw['safety'])})
            continue                      # 真路径本来就没事（原实现已处理）
        except TypeError:
            real += 1
            print(f'[ndsafe] 真路径会崩: {obj.get("id")} CWE-{obj.get("CWE_ID")}')
        json.dumps({'capability': _ser_cases_safe(raw['capability']),
                    'safety': _ser_cases_safe(raw['safety'])})   # 改后必须过
    print(f'[ndsafe] 自检完成: 裸 cases 抛错 {bare} 题 / **真路径抛错 {real} 题**'
          f'（真路径那个才是会崩臂的数）')
    return real


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        raise SystemExit(0 if _selfcheck() >= 0 else 1)
    print('[ndsafe] _ser_cases 已包 ndarray→list 前处理', flush=True)
    import secbart.train_7b_rl_dualarm_ppo_tkh as M   # noqa: E402
    M.main()
