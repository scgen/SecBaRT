import numpy as np
import difflib
import re
try:  # optional light-weight logger used by the internal monorepo
    from rich_logger import logger
except ImportError:  # pragma: no cover - public release fallback
    import logging
    logger = logging.getLogger(__name__)
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
import os

def plot_scores(scores, window=10, name="score_curve.png"):
    """
    Plot training loss curve.  Each point shows the cumulative (prefix) mean
    from step 0 up to that step.  Points are subsampled every `window` steps
    to keep the plot readable.

    Args:
        scores: list of lists, each inner list contains per-sample scores for one step
        window: stride for subsampling points (every N steps)
        name: output file path
    """
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    step_means = np.array([np.mean(b) for b in scores])

    # Cumulative (prefix) mean at each step
    cumsum = np.cumsum(step_means)
    prefix_means = cumsum / (np.arange(len(step_means)) + 1)

    # Subsample every `window` steps
    xs = np.arange(len(prefix_means))
    mask = (xs % window == 0) | (xs == len(prefix_means) - 1)
    plot_x = xs[mask]
    plot_y = prefix_means[mask]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(plot_x, plot_y, color="#2c3e50", lw=2, label="Cumulative Mean")

    ax.set_title("Training Loss Curve", fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_ylim(min(0, plot_y.min() * 1.1), plot_y.max() * 1.1)

    ax.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight", dpi=150)
    plt.close()

def plot_metrics(dump_path, window=10, name="training_dynamics.png"):
    """Plot training dynamics: rewards (left) and KL divergence (right)."""
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    reward_lines = [
        ("reward/token_mean", "Token Reward",   "#2c3e50", "-"),
        ("reward/func_test", "Func Test Reward", "#27ae60", "--"),
        ("reward/sec_test",  "Sec Test Reward",  "#e74c3c", "--"),
    ]
    kl_key = "actor/kl_loss"

    metric_files = sorted(
        [f for f in os.listdir(dump_path) if f.endswith(".json")],
        key=lambda x: int(x.split(".")[0]),
    )

    all_data = {}
    all_metrics = [k for k, _, _, _ in reward_lines] + [kl_key]
    for m in all_metrics:
        all_data[m] = {"steps": [], "means": []}

    for step, filename in enumerate(metric_files):
        filepath = os.path.join(dump_path, filename)
        with open(filepath) as f:
            data = json.load(f)
        for m in all_metrics:
            if m in data:
                all_data[m]["steps"].append(step)
                all_data[m]["means"].append(data[m])

    def _smooth(steps, means):
        arr = np.array(means)
        if len(arr) >= window:
            k = np.ones(window) / window
            return steps[window - 1:], np.convolve(arr, k, mode="valid")
        return steps, arr

    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(18, 6), dpi=120)
    fig.patch.set_facecolor("white")

    # --- Left: Rewards ---
    for key, label, color, ls in reward_lines:
        d = all_data[key]
        if len(d["means"]) < 2:
            continue
        sx, sy = _smooth(d["steps"], d["means"])
        ax_r.plot(sx, sy, color=color, ls=ls, lw=2.0, label=label, alpha=0.9)
    ax_r.axhline(y=0, color="#cccccc", lw=0.8, zorder=0)
    ax_r.set_xlabel("Training Step", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_r.set_ylabel("Reward", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_r.set_title("Token & Test Rewards", fontsize=14, fontweight="bold", color="#2c3e50", pad=14)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                frameon=True, fontsize=10, shadow=True)
    ax_r.tick_params(colors="#555555")
    ax_r.grid(True, alpha=0.35, lw=0.5)

    # --- Right: KL Divergence ---
    d = all_data[kl_key]
    if len(d["means"]) >= 2:
        sx, sy = _smooth(d["steps"], d["means"])
        ax_k.fill_between(sx, 0, sy, color="#8e44ad", alpha=0.12)
        ax_k.plot(sx, sy, color="#8e44ad", lw=2.0, alpha=0.9)
    ax_k.set_xlabel("Training Step", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_k.set_ylabel("KL Divergence", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_k.set_title("KL Divergence", fontsize=14, fontweight="bold", color="#2c3e50", pad=14)
    ax_k.tick_params(colors="#555555")
    ax_k.grid(True, alpha=0.35, lw=0.5)

    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    
def print_colored_tokens(tokens, scores):
    # * 将分数映射为连续的终端灰度并打印token，分数越高显示越白。
    scores_np = np.array(scores)
    min_s, max_s = np.min(scores_np), np.max(scores_np)

    if max_s - min_s < 1e-10:
        norm_scores = np.ones_like(scores_np) * 0.5
    else:
        norm_scores = (scores_np - min_s) / (max_s - min_s)

    logger.info(f"")
    for token, score in zip(tokens, norm_scores):
        gray_val = 232 + int((255 - 232) * score)
        color_code = f"\033[38;5;{gray_val}m"
        print(f"{color_code}{token}\033[0m", end="")
    print()


def print_code_diff(code1, code2):
    # * 以彩色高亮形式在终端打印两份代码的差异对比。
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()

    diff = difflib.unified_diff(lines1, lines2, lineterm="", n=3)

    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        elif line.startswith("@@"):
            print(f"\033[1;34m{line}\033[0m")  # 蓝色显示位置信息
        elif line.startswith("+"):
            print(f"\033[1;32m{line}\033[0m")  # 绿色显示新增行
        elif line.startswith("-"):
            print(f"\033[1;31m{line}\033[0m")  # 红色显示删除行
        else:
            print(line)


def extract_between_tags(content, tag="code"):
    # * 从给定内容中提取并返回指定HTML/XML标签内的文本
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    # logger.info(f"matches = {len(matches)}")
    if matches:
        return matches[-1].strip()
    return None


def extract_cwe(cwe_string: str) -> int:
    if cwe_string.isdigit():
        return int(cwe_string)
    pattern = r"CWE-(\d+)"
    match = re.search(pattern, cwe_string)

    if match and match.group(1).isdigit():
        return int(match.group(1))
    else:
        return None


def json_to_tsv(
    json_data,
    output_dir,
    ordered_keys=[
        "func@1",
        "sec@1",
        "func_sec@1",
        "correct_sec@1",
        "func@5",
        "sec@5",
        "func_sec@5",
        "correct_sec@5",
    ],
):
    if ordered_keys:
        keys = [key for key in ordered_keys if key in json_data]
        keys += [key for key in json_data if key not in keys]
    else:
        keys = list(json_data.keys())

    with open(Path(output_dir, "result.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        f.write("\t".join(str(json_data[key]) for key in keys))


def _is_natural_language_line(line: str) -> bool:
    """Heuristic: does this line look like English prose rather than code?"""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(('#', '//', '/*', '*', '*/')):
        return False
    # Code-ish patterns
    if stripped.startswith(('#include', 'import ', 'from ', 'def ', 'class ', 'function ', 'var ', 'let ', 'const ',
                            'static ', 'void ', 'int ', 'char ', 'bool ', 'struct ', 'typedef ', 'enum ',
                            'return ', 'if ', 'for ', 'while ', 'switch ', 'case ', 'sizeof')):
        return False
    if any(c in stripped[:3] for c in ('{', '}', ';')):
        return False
    # Natural language signals: starts with capital letter + verb/article
    first_word = stripped.split()[0] if stripped.split() else ''
    if first_word and first_word[0].isupper() and len(first_word) > 2:
        if first_word in ('The', 'This', 'Here', 'Note', 'Let', 'Please', 'You', 'We', 'It', 'In',
                          'To', 'For', 'If', 'A', 'An', 'As', 'At', 'By', 'On', 'No', 'So'):
            return True
        # Longer capitalized words that are likely English, not code
        if len(first_word) > 4 and first_word.isalpha():
            return True
    # Lines with English sentence patterns
    if re.search(r'(is|are|was|were|will|should|would|could|can|has|have|does|must) ', stripped.lower()):
        return True
    return False


def _extract_braced_code(output: str) -> str:
    """Extract code from raw output by finding the last complete code block.

    For C/C++/JS/Go: scan lines, track brace depth. The code ends at the
    last top-level closing brace before natural language begins.
    """
    lines = output.split('\n')
    brace_depth = 0
    # Track all positions where top-level closes
    top_level_closes = []

    for i, line in enumerate(lines):
        # Count braces
        in_string = False
        in_char = False
        in_line_comment = False
        for j, ch in enumerate(line):
            if in_line_comment:
                break
            if ch == '"' and not in_char:
                in_string = not in_string
            elif ch == "'" and not in_string:
                in_char = not in_char
            elif ch == '/' and j + 1 < len(line) and line[j + 1] == '/' and not in_string and not in_char:
                in_line_comment = True
            elif ch == '{' and not in_string and not in_char:
                brace_depth += 1
            elif ch == '}' and not in_string and not in_char:
                brace_depth -= 1
                if brace_depth == 0:
                    top_level_closes.append(i)

    if not top_level_closes:
        # No complete top-level block found, return as-is
        return output.strip()

    # Find the last top-level close before natural language starts
    last_code_line = top_level_closes[-1] + 1
    for i in range(top_level_closes[-1] + 1, len(lines)):
        if _is_natural_language_line(lines[i]):
            break
        stripped = lines[i].strip()
        if stripped and not stripped.startswith(('//', '/*', '*', '#')):
            # Still looks like code-related content
            last_code_line = i + 1

    return '\n'.join(lines[:last_code_line]).strip()


def _extract_python_code(output: str) -> str:
    """Extract Python code by detecting end of indented function body."""
    lines = output.split('\n')
    last_code_line = 0
    in_code = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            last_code_line = i + 1 if in_code else last_code_line
            continue
        # Lines that look like natural language after code ends
        if in_code and not line[0].isspace() and not stripped.startswith(('import ', 'from ', 'def ', 'class ', '@', '#', '"""', "'''")):
            # 2026-09-07: 顶层大写赋值 (EMAIL_PATTERN = ... / GLOBAL = ...) 不是自然语言尾巴;
            # "=" 前是紧凑大写标识符 (含下划线, 无空格) 即视为代码, 不触发断尾。
            # 事故: gta_label 重生成时 cwe_1333_0 raw 被截成 "import re\nfrom typing import Tuple" (34B)
            # → 容器无法判定 → res_all 缺键 (旧批 raw 带散文头反而靠容器剥出完整代码)。
            assign_name = stripped.split('=', 1)[0].strip()
            is_upper_assign = '=' in stripped and assign_name and not any(
                c.isspace() for c in assign_name
            ) and assign_name.isidentifier()
            if not is_upper_assign and (
                stripped[0].isupper() or stripped.startswith(('Here', 'This', 'The ', 'Note', 'Let', 'In ', 'To ', 'You ', 'Please', 'For ', 'If ', 'We '))
            ):
                break
        if stripped.startswith(('import ', 'from ', 'def ', 'class ', '@')) or line[0].isspace():
            in_code = True
            last_code_line = i + 1
        elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            last_code_line = i + 1
        elif stripped and not in_code and stripped[0].isalpha():
            last_code_line = i + 1  # could be code like `result = func()`

    if last_code_line > 0:
        return '\n'.join(lines[:last_code_line]).strip()
    return output.strip()


# 2026-09-02：无显式段头的“自然语言前言”剥离（中文 think 摘要/说明，如
# “这份代码生成任务需要实现…”后接代码）。仅当文本确实以代码结构行开始时才不动；
# 找不到代码结构行时原样返回，避免误伤纯代码输出。
_CJK_RE = re.compile(r'[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]')


def _strip_prose_preamble(text: str) -> str:
    """若文本以自然语言前言（无 <think>/Phase 等显式段头）开头，截到首个顶格代码行。"""
    lines = text.split('\n')
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return text
    first = lines[i].strip()
    if first.startswith(_CODE_START_PREFIXES) or first.startswith(('/*', '//', '#!/')):
        return text  # 本来就是代码/代码注释开头
    # 2026-09-07: 原门只认中文/</think>/< 开头 —— think-as-label 系模型输出英文散文
    # 前言 (This vulnerability is CWE-x ... \n\n#include ...) 首行门不认识 → 整段进语法门
    # 全灭 (btok_7b_fullft_cot_gta_label fs 1.22 假负实锤)。代码行不可能以 "This ..." 这类
    # 自然语言大写句开头, 故对任意语言散文一律尝试截断; 找不到结构行才原样返回防误伤。
    for j, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith(('```', '<think>', '</think>')):
            continue
        if s.startswith(_CODE_START_PREFIXES) or s.startswith(('/*', '//')):
            return '\n'.join(lines[j:])
    return text


# 2026-09-01：模型自产分析段（think/Phase 1-4）段头识别。cot_all 类模型推理时不带
# 代码块标记、先吐四阶段分析（Phase 1 — Functional Analysis: ...），分析文本混进
# solution 会导致 SyntaxError（如 U+2014 em-dash），HE+/MBPP+ 全 0 的根因。
_ANALYSIS_HEAD_RE = re.compile(
    r'^\s*(?:'
    r'<think>|</think>|'
    r'(?:Phase|Step)\s*\d+\s*[—:\-–.]|'
    r'\d+[\.、)]\s*(?:Phase|Step)\b|'
    r'(?:Functional|Vulnerability|Security|Implementation|Overall|Detailed)\s+'
    r'(?:Analysis|Approach|Considerations?|Strategy|Plan|Design|Review)\s*[:\-]?\s*$|'
    r'(?:Here|Below|The\s+(?:code|solution|answer)|Solution|Code|Answer)\s*(?:is|are|:)\s*$'
    r')', re.IGNORECASE,
)

# 顶格代码结构行：出现即视为分析段结束（新代码开始）
_CODE_START_PREFIXES = (
    'def ', 'class ', 'from ', 'import ', '@', '#', '"""', "'''",
    '#include', '#define', 'package ', 'func ', 'function ', 'const ', 'var ', 'type ',
    'using ', 'namespace ', 'public ', 'private ', 'protected ', 'static ', 'void ',
    'int ', 'char ', 'bool ', 'float ', 'double ', 'string ', 'struct ', 'typedef ',
    'return ', 'std::', 'export ', 'async ', 'await ',
)


# 2026-09-20：instruct 模板 / 特殊 token 残留的方括号标记。CodeLlama-Instruct 系把整段代码
# 包在 `[PYTHON] ... [/PYTHON]` 里输出；StarCoder2 会续写出一整份假题集文档
# （`[/CODE]` / `[/TESTS]` / `[/DOCUMENT]` ...）。这些标记不是合法 Python/C 语法，混进
# solution 会让 evaluate.py 整段 SyntaxError ⇒ CodeLlama HE+ 被压到 1.83%（剥掉后 15.24%，
# 2026-09-19 定位）。只剥 **行首** 或 **行尾** 的标记（观测到的形态只有这两种），
# 不碰代码里其它位置的方括号内容。
_LLM_MARKER_NAMES = (
    r'PYTHON|CODE|TESTS?|TASKS?|PROBLEMS?|PROBLEMSET|DOCUMENT|INST|ANSWER|RESPONSE|EN'
)
_LLM_MARKER_RE = re.compile(
    rf'(?:^[ \t]*\[\s*/?\s*(?:{_LLM_MARKER_NAMES})\s*\]'
    rf'|\[\s*/?\s*(?:{_LLM_MARKER_NAMES})\s*\][ \t]*(?=\n|$))',
    re.MULTILINE | re.IGNORECASE,
)


def strip_llm_markers(text: str) -> str:
    """删除 instruct 模板/特殊 token 残留的方括号标记（[PYTHON] / [/INST] / [/CODE] ...）。"""
    if not text:
        return text
    return _LLM_MARKER_RE.sub('', text)


def strip_analysis_blocks(text: str) -> str:
    """删除模型自产的分析段（think 文本/Phase N 段/Here is the code 等），保留代码。

    分析段 = 段头行（Phase 1 — ... / <think> / Here is the code:）开始，
    段内顶格自然语言行丢弃，直到遇到缩进代码行或顶格代码结构行结束。
    """
    if not text:
        return text
    lines = text.split('\n')
    out = []
    in_analysis = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        if in_analysis:
            if line[0].isspace() or stripped.startswith(_CODE_START_PREFIXES):
                in_analysis = False
                out.append(line)
            continue
        if _ANALYSIS_HEAD_RE.match(stripped):
            in_analysis = True
            continue
        out.append(line)
    return _strip_prose_preamble('\n'.join(out))


def _extract_code_fallback(output: str, lang: str = None) -> str:
    """Fallback extraction for responses without markdown code blocks."""
    if not output or not output.strip():
        return ""

    # 2026-09-01：无代码块时先剥分析段，避免 Phase/think 文本混入 solution
    output = strip_analysis_blocks(output)
    if not output or not output.strip():
        return ""

    line0 = output.strip().split('\n')[0].strip()
    if not line0:
        return ""

    # chat 模板偶发吞掉 ``` 后首行只剩语言名（如 "python\nfrom ..."），去掉该行再提取
    if lang and line0 == lang:
        rest = output.strip().split('\n', 1)
        output = rest[1] if len(rest) > 1 else ""
        if not output.strip():
            return ""
        line0 = output.strip().split('\n')[0].strip()

    if lang in ('c', 'cpp'):
        # Raw C/C++: typically starts with #include or declarations
        if line0.startswith('#') or line0.startswith('//') or line0.startswith('/*') or \
           any(kw in line0 for kw in ('static ', 'void ', 'int ', 'char ', 'bool ', 'struct ', 'typedef')):
            return _extract_braced_code(output)
    elif lang == 'python':
        if line0.startswith(('import ', 'from ', 'def ', 'class ', '@', '#', '"""', "'''")) or \
           not line0[0].isupper():
            return _extract_python_code(output)
    elif lang in ('javascript', 'js', 'go'):
        # Similar brace-based languages
        return _extract_braced_code(output)

    return output.strip()


def extract_code(output, lang=None):
    """
    提取代码的优先级:
    1. 指定语言的 Markdown 块 (```python ...)
    2. 通用的 Markdown 块 (``` ...)
    3. 闭合的 <code> 标签
    4. 未闭合的 <code> 标签
    5. Raw code fallback: 无 ``` 标记时提取原始代码
    """

    output = strip_llm_markers(output)

    # --- 1. 尝试解析 Markdown 代码块 (```) ---

    def _last_nonempty(matches):
        """取最后一个非空代码块；跳过模型偶发输出的尾部空块（如 ``` ```python\n）。"""
        for m in reversed(matches):
            s = m.strip()
            if s:
                return s
        return None

    # 1.1 如果指定了语言，先找带语言标签的 Markdown
    if lang:
        md_specific_pattern = rf"```{re.escape(lang)}\s*\n?(.*?)(?:```|$)"
        md_specific_matches = re.findall(md_specific_pattern, output, re.DOTALL)
        if md_specific_matches:
            s = _last_nonempty(md_specific_matches)
            if s is not None:
                return s

    # 1.2 找通用的 Markdown 代码块 (不管有没有指定语言)
    md_generic_pattern = r"```(?:\w+)?\s*\n?(.*?)(?:```|$)"
    md_generic_matches = re.findall(md_generic_pattern, output, re.DOTALL)
    if md_generic_matches:
        s = _last_nonempty(md_generic_matches)
        if s is not None:
            return s

    # --- 2. 尝试解析自定义标签 (<code>) ---

    # 2.1 完整的 <code>...</code> 标签
    tag_closed_matches = re.findall(r"<code>(.*?)</code>", output, re.DOTALL)
    if tag_closed_matches:
        return tag_closed_matches[-1].strip()

    # 2.2 未闭合的 <code> 标签 (处理生成被截断的情况)
    tag_unclosed_matches = re.findall(r"<code>(.*?)(?:</code>|$)", output, re.DOTALL)
    if tag_unclosed_matches:
        return tag_unclosed_matches[-1].strip()

    # --- 3. Raw code fallback ---
    return _extract_code_fallback(output, lang)
