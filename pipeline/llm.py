"""Apertus behind an OpenAI-compatible server (vLLM): JSON-schema constrained answers and a disk cache.

    llm = Apertus()                                   # http://localhost:8000/v1, first served model
    llm.json(messages, schema)                        # -> dict that matches the schema
    llm.text(messages)                                # -> str (thinking stripped)

Every request is cached on disk by the hash of its body, so re-running the pipeline on the
same scenes costs nothing. Delete the cache folder (default .cache/llm) to start fresh.
"""
from __future__ import annotations

import base64, hashlib, json, threading, time, urllib.error, urllib.request
from pathlib import Path

import cv2
import numpy as np

THINK_END = "<|inner_suffix|>"
# Apertus ends a turn with <|assistant_end|> (68) or <|tools_suffix|> (72), both listed as EOS in its
# generation config. The JSON grammar does not mask these special tokens, so they can end a reply in
# the middle of a string. With a schema we bias them away: the grammar then ends with </s>.
SPECIAL_STOPS = {"68": -100, "72": -100}


def image_part(rgb: np.ndarray, max_side: int = 1024, quality: int = 90) -> tuple[dict, float]:
    """An OpenAI image_url content part for an RGB array, resized so its long side is at most
    max_side. Returns (part, scale) with scale = small / full. Same input, same bytes: the server's
    prefix cache then reuses the encoded image across instructions of one scene."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": url}}, scale


def strip_thinking(text: str) -> str:
    return text.rsplit(THINK_END, 1)[-1].strip() if THINK_END in text else text.strip()


class Apertus:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str | None = None,
                 cache_dir: Path | None = Path(".cache/llm"), timeout: float = 900, seed: int = 0):
        self.base_url = base_url.rstrip("/")
        self.timeout, self.seed = timeout, seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or self._served_model()
        self.calls = self.cache_hits = 0
        self._lock = threading.Lock()

    def _served_model(self) -> str:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=30) as r:
                return json.loads(r.read())["data"][0]["id"]
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as e:
            raise SystemExit(f"cannot reach Apertus at {self.base_url} ({e}). Start it first: docker compose up -d")

    # ------------------------------------------------------------------ raw call
    def complete(self, messages: list, *, schema: dict | None = None, max_tokens: int = 512,
                 temperature: float = 0.0, thinking: bool = False, n: int = 1, seed: int | None = None) -> list[str]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "n": n, "seed": self.seed if seed is None else seed,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
            body["logit_bias"] = SPECIAL_STOPS
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            with self._lock:
                self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["replies"]
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
        for attempt in range(3):
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"Apertus server error {e.code}: {e.read().decode()[:500]}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise SystemExit(f"cannot reach Apertus at {self.base_url} ({e})")
                time.sleep(5)
        replies = [c["message"]["content"] or "" for c in out["choices"]]
        with self._lock:
            self.calls += 1
        if path:
            path.write_text(json.dumps({"replies": replies, "seconds": round(time.time() - t0, 2),
                                        "usage": out.get("usage")}, ensure_ascii=False), encoding="utf-8")
        return replies

    # ------------------------------------------------------------------ helpers
    def text(self, messages: list, **kw) -> str:
        return strip_thinking(self.complete(messages, **kw)[0])

    def json(self, messages: list, schema: dict, **kw) -> dict:
        """Constrained decoding: the reply matches the schema. A reply cut off by max_tokens does not
        parse; it is retried with another seed and a little temperature, then raises."""
        for attempt in range(3):
            reply = strip_thinking(self.complete(messages, schema=schema, seed=self.seed + attempt,
                                                 **({**kw, "temperature": 0.3} if attempt else kw))[0])
            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"no valid JSON after 3 attempts, last reply: {reply[:300]!r}")

    def json_samples(self, messages: list, schema: dict, n: int, temperature: float = 0.7, **kw) -> list[dict]:
        return [json.loads(strip_thinking(r)) for r in
                self.complete(messages, schema=schema, n=n, temperature=temperature, **kw)]
