"""The single disk-cached LLM client (llama-server OpenAI API on 127.0.0.1:8301, qwen3:8b GGUF, temperature 0).

Every LLM call of P3 goes through `LLM.chat`. Cache key = sha256(model, prompt, system, max_tokens, temperature);
responses are stored in data/llm_cache.sqlite, so reruns are free. Concurrency: `chat_many` uses a thread pool
matching the server's parallel slots.
"""
from __future__ import annotations

import hashlib, json, os, sqlite3, threading, time
from concurrent.futures import ThreadPoolExecutor

P3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_URL = os.environ.get("P3_LLM_URL", "http://127.0.0.1:8301/v1/chat/completions")
DEFAULT_MODEL = "qwen3:8b"


class LLM:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, cache_path: str | None = None,
                 timeout: float = 300, parallel: int = 8):
        self.url, self.model, self.timeout, self.parallel = url, model, timeout, parallel
        self.cache_path = cache_path or os.path.join(P3, "data", "llm_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._local = threading.local()
        self._db().execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, prompt TEXT, resp TEXT, meta TEXT)")
        self._db().commit()
        self.stats = dict(calls=0, hits=0, seconds=0.0, prompt_tokens=0, completion_tokens=0)
        self._lock = threading.Lock()

    def _db(self):
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(self.cache_path, timeout=120)
            self._local.db.execute("PRAGMA journal_mode=WAL")
        return self._local.db

    def key(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        return hashlib.sha256(json.dumps([self.model, system, prompt, max_tokens, temperature]).encode()).hexdigest()

    def _post(self, payload: dict) -> dict:
        import requests
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def chat(self, prompt: str, system: str = "", max_tokens: int = 256, temperature: float = 0.0, no_think: bool = True,
             cache_prompt: bool = True) -> str:
        if no_think and "qwen3" in self.model and not prompt.rstrip().endswith("/no_think"):
            prompt = prompt.rstrip() + " /no_think"
        k = self.key(prompt, system, max_tokens, temperature)
        row = self._db().execute("SELECT resp FROM c WHERE k=?", (k,)).fetchone()
        if row is not None:
            with self._lock: self.stats["hits"] += 1
            return row[0]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t = time.perf_counter()
        for attempt in range(12):   # the server may be restarted by memguard (60 s+ cooldown): wait up to ~10 min
            try:
                js = self._post(dict(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=temperature,
                                     cache_prompt=cache_prompt))
                break
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(60, 2 ** attempt))
        dt = time.perf_counter() - t
        text = js["choices"][0]["message"]["content"] or ""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()
        u = js.get("usage", {})
        with self._lock:
            self.stats["calls"] += 1; self.stats["seconds"] += dt
            self.stats["prompt_tokens"] += u.get("prompt_tokens", 0); self.stats["completion_tokens"] += u.get("completion_tokens", 0)
        self._db().execute("INSERT OR REPLACE INTO c VALUES (?,?,?,?)", (k, prompt, text, json.dumps(dict(usage=u, seconds=dt))))
        self._db().commit()
        return text

    def chat_many(self, prompts: list[str], **kw) -> list[str]:
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(lambda p: self.chat(p, **kw), prompts))


def llama_server_cmd(port: int = 8301, parallel: int = 8, ctx: int = 32768, model: str = "qwen3/8b") -> list[str]:
    """Command line for our own batched llama-server on the Ollama GGUF blob (AGENT_CONTEXT.md)."""
    man = os.path.expanduser(f"~/.ollama/models/manifests/registry.ollama.ai/library/{model}")
    blob = [l["digest"] for l in json.load(open(man))["layers"] if l["mediaType"].endswith("image.model")][0].replace(":", "-")
    return ["/Applications/Ollama.app/Contents/Resources/llama-server", "--model", os.path.expanduser(f"~/.ollama/models/blobs/{blob}"),
            "--port", str(port), "--host", "127.0.0.1", "-np", str(parallel), "-c", str(ctx), "-cb", "-ngl", "99", "--reasoning-budget", "0"]
