"""Thin client for the LLM annotator used by the token-label pipeline.

The original internal scripts hard-coded an API key and an endpoint.  This
release keeps the same request convention (OpenAI-compatible chat completions,
``thinking`` disabled when the backend supports it) but reads all credentials
from the environment.

Environment variables
---------------------
ARK_KEY    API key (required for annotation runs)
ARK_BASE   base URL, default ``https://ark.cn-beijing.volces.com/api/plan/v3``
ARK_MODEL  model name, default ``deepseek-v4-flash``
"""
import json
import os
import re

import requests

KEY = os.environ.get("ARK_KEY", "")
BASE = os.environ.get("ARK_BASE", "https://ark.cn-beijing.volces.com/api/plan/v3")
MODEL = os.environ.get("ARK_MODEL", "deepseek-v4-flash")
# Endpoint shape differs across gateways; try the common variants in order.
ENDPOINTS = [f"{BASE}/chat/completions", BASE.rstrip("/") + "/v1/chat/completions", BASE]


def parse_json(txt):
    """Parse a JSON object from a model reply, tolerating surrounding prose."""
    txt = (txt or "").strip()
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None


def post_chat(prompt, system, temperature=0.1, max_tokens=900, timeout=240):
    """POST one chat completion and return ``(status_code, text_or_error)``.

    Returns ``(200, content)`` on success and ``(status, error_message)``
    otherwise so callers can implement their own retry/backoff policy.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Reasoning models on this gateway accept a top-level switch; other
        # OpenAI-compatible servers simply ignore unknown fields.
        "thinking": {"type": "disabled"},
    }
    last = "no endpoint"
    for ep in ENDPOINTS:
        try:
            r = requests.post(ep, headers={"Authorization": f"Bearer {KEY}"},
                              json=payload, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        if r.status_code == 200:
            try:
                return 200, r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = f"bad-response: {type(e).__name__}: {e}"
                continue
        last = f"HTTP {r.status_code}: {r.text[:200]}"
    return -1, last
