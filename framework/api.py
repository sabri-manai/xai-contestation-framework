"""OpenRouter access for both tracks, with an on-disk cache so every run is repeatable and cheap.

- jev_decide(state, questions)  → typed answers (TypeSafe Jev, Decisions API; DATA_RULES §9 pins the model)
- chat(model, messages)         → text (chat completions), JSON mode optional

Cache: benchmark/outputs/cache/<kind>.jsonl, keyed by a hash of the request body. Delete a line to re-run it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import threading

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "benchmark" / "outputs" / "cache"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_MODEL = "typesafe/jev-1.14"
EXPECTED_SNAPSHOT = "typesafe/jev-1.14-20261014"


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY in the environment or in .env at the repo root")
    return key


_KEY = None


def _headers() -> dict:
    global _KEY
    _KEY = _KEY or load_key()
    return {"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json", "X-Title": "XAI contestation framework"}


class Cache:
    def __init__(self, kind: str):
        self.lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{kind}.jsonl"
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    @staticmethod
    def key_of(body: dict, salt: str = "") -> str:
        return hashlib.sha1((salt + json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()

    def get(self, key: str):
        r = self.rows.get(key)
        return r["response"] if r else None

    def put(self, key: str, body: dict, response: dict, meta: dict | None = None) -> None:
        row = {"key": key, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "request": body, "response": response, **(meta or {})}
        with self.lock:
            self.rows[key] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


_CACHES: dict[str, Cache] = {}
_CACHES_LOCK = threading.Lock()


def _cache(kind: str) -> Cache:
    with _CACHES_LOCK:
        if kind not in _CACHES:
            _CACHES[kind] = Cache(kind)
        return _CACHES[kind]


def _post(url: str, body: dict, tries: int = 5, timeout: int = 90) -> dict:
    for attempt in range(tries):
        try:
            r = requests.post(url, headers=_headers(), json=body, timeout=timeout)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1)); continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == tries - 1:
                r.raise_for_status()
            time.sleep(2 ** (attempt + 1)); continue
        if r.status_code != 200:
            raise RuntimeError(f"{url} {r.status_code}: {r.text[:500]}")
        return r.json()
    raise RuntimeError("unreachable")


def jev_decide(state: dict, questions: dict, repeat: int = 0, cache_kind: str = "jev") -> dict:
    """Returns the full Decisions response (answers, model, usage). repeat changes the cache key only."""
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    c = _cache(cache_kind)
    k = c.key_of(body, salt=f"r{repeat}")
    hit = c.get(k)
    if hit is not None:
        return hit
    res = _post(JEV_URL, body)
    if res.get("model") != EXPECTED_SNAPSHOT:
        print(f"WARNING: Jev snapshot {res.get('model')} != {EXPECTED_SNAPSHOT}")
    c.put(k, body, res, {"repeat": repeat})
    return res


def chat(model: str, messages: list[dict], max_tokens: int = 600, temperature: float = 0.0,
         json_mode: bool = False, repeat: int = 0, cache_kind: str = "chat") -> dict:
    """Returns {"text": ..., "usage": ..., "model": ...}."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    c = _cache(cache_kind)
    k = c.key_of(body, salt=f"r{repeat}")
    hit = c.get(k)
    if hit is not None:
        return hit
    res = _post(CHAT_URL, body)
    out = {"text": res["choices"][0]["message"]["content"], "usage": res.get("usage", {}), "model": res.get("model", model)}
    c.put(k, body, out, {"repeat": repeat})
    return out


def parse_json(text: str) -> dict:
    """First balanced JSON object in text; tolerant of code fences."""
    s = text.find("{")
    if s < 0:
        raise ValueError("no JSON object in: " + text[:200])
    depth, in_str, esc = 0, False, False
    for i in range(s, len(text)):
        ch = text[i]
        if in_str:
            esc = (not esc and ch == "\\")
            if not esc and ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[s:i + 1])
    raise ValueError("unbalanced JSON in: " + text[:200])


def total_cost(kind: str) -> float:
    c = _cache(kind)
    tot = 0.0
    for r in c.rows.values():
        u = r["response"].get("usage", {}) if isinstance(r["response"], dict) else {}
        try:
            tot += float(u.get("cost", 0) or 0)
        except (TypeError, ValueError):
            pass
    return tot
