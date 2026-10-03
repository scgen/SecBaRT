# In-container helper appended to evaluate.py by cweval_eval_patch.py
import re as _re


def _strip_analysis_preamble(text):
    if not text:
        return text
    _pre = (
        'def ', 'class ', 'from ', 'import ', '@', '#', '"""', "'''",
        '#include', '#define', 'package ', 'func ', 'function ', 'const ', 'var ', 'type ',
        'using ', 'namespace ', 'public ', 'private ', 'protected ', 'static ', 'void ',
        'int ', 'char ', 'bool ', 'float ', 'double ', 'string ', 'struct ', 'typedef ',
        'return ', 'std::', 'export ', 'async ', 'await ',
    )
    lines = text.split('\n')
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return text
    first = lines[i].strip()
    if first.startswith(_pre) or first.startswith(('/*', '//', '#!/')):
        return text
    # 2026-09-07: \u539f\u95e8\u53ea\u8ba4\u4e2d\u6587/</think>/< \u2014\u2014 think-as-label \u82f1\u6587\u6563\u6587\u524d\u8a00 (This vulnerability
    # is CWE-x ... \n\n#include) \u6f0f\u7f51\u6574\u6bb5\u8fdb\u8bed\u6cd5\u95e8\u5168\u706d (gta_label fs 1.22 \u5047\u8d1f)\u3002\u4ee3\u7801\u884c\u4e0d\u53ef\u80fd
    # \u4ee5\u81ea\u7136\u8bed\u8a00\u5927\u5199\u53e5\u5f00\u5934 \u2192 \u4efb\u610f\u8bed\u8a00\u6563\u6587\u4e00\u5f8b\u5c1d\u8bd5\u622a\u5230\u9996\u4e2a\u7ed3\u6784\u884c, \u627e\u4e0d\u5230\u624d\u539f\u6837\u8fd4\u56de\u3002
    for j, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith(('```', '<think>', '</think>')):
            continue
        if s.startswith(_pre) or s.startswith(('/*', '//')):
            return '\n'.join(lines[j:])
    return text
