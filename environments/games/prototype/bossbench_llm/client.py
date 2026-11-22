"""Minimal OpenAI-compatible chat client (standard library only): vLLM, SGLang, OpenAI, any /v1/chat/completions.

Notes for Lumine-style models served by vLLM: <|action_start|> / <|action_end|> are special tokens and vLLM drops
special tokens from the returned text by default. Pass --extra-body '{"skip_special_tokens": false}' (the harness
also accepts an action body without markers and counts it as `bare`).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class ChatError(RuntimeError):
    """The endpoint could not produce a reply (network / HTTP / malformed response): an infrastructure failure,
    never scored as a model action."""


class ChatClient:
    def __init__(self, base_url, model, *, api_key=None, api_key_env="OPENAI_API_KEY", temperature=1.0, top_p=1.0,
                 max_tokens=128, timeout=120.0, retries=3, extra_body=None, headers=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env or "", "")
        self.params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        self.extra_body = dict(extra_body or {})
        self.timeout, self.retries = float(timeout), int(retries)
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        if self.api_key:
            self.headers["Authorization"] = "Bearer " + self.api_key

    def describe(self):
        """Public settings for the run log (never the key)."""
        return {"endpoint": self.url, "model": self.model, **self.params, "extra_body": self.extra_body}

    def complete(self, messages):
        """-> (reply text, metadata). Raises ChatError after `retries` failed attempts."""
        body = json.dumps({"model": self.model, "messages": messages, "stream": False, **self.params,
                           **self.extra_body}).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                choice = payload["choices"][0]
                content = choice["message"].get("content")
                if isinstance(content, list):      # some servers return content parts
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                reasoning = choice["message"].get("reasoning_content")
                return (content or ""), {"seconds": time.perf_counter() - start, "attempts": attempt + 1,
                                         "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage"),
                                         "reasoning_content": reasoning}
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last = f"HTTP {e.code}: {detail}"
                if 400 <= e.code < 500 and e.code not in (408, 409, 429):
                    break                            # a request error will not fix itself
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                time.sleep(min(30.0, 2.0 ** attempt))
        raise ChatError(last or "unknown error")
