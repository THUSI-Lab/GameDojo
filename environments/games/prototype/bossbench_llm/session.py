"""One chat session per episode, kept the way Lumine keeps it.

Context = system prompt (+ the latest reasoning, if any) followed by the most recent `window` image-action pairs as
alternating user / assistant turns; the oldest pair is dropped first (FIFO). When the model writes new reasoning
(<|thought_start|>...<|thought_end|>), the pairs are flushed and re-accumulate from that turn, and the reasoning
moves to the head of the context. window=0 sends only the current frame (Lumine's non-history setting).
"""
from __future__ import annotations

from collections import deque

from .lumine import strip_thoughts


class Session:
    def __init__(self, system_prompt, window=20, user_text=None):
        if window < 0:
            raise ValueError("window must be >= 0")
        self.system_prompt, self.window, self.user_text = system_prompt, int(window), user_text
        self.pairs = deque()
        self.thought = None
        self.flushes = 0

    def _user(self, data_url):
        content = []
        if self.user_text:
            content.append({"type": "text", "text": self.user_text})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        return {"role": "user", "content": content}

    def messages(self, data_url):
        """Messages for the current turn: history + the new frame."""
        system = self.system_prompt
        if self.thought:
            system += "\n\n<|thought_start|>" + self.thought + "<|thought_end|>"
        out = [{"role": "system", "content": system}]
        for url, reply in self.pairs:
            out.append(self._user(url))
            out.append({"role": "assistant", "content": reply})
        out.append(self._user(data_url))
        return out

    def record(self, data_url, reply):
        """Store the finished turn. The reasoning (if any) moves to the head; the pair keeps only the action."""
        action_only, thought = strip_thoughts(reply or "")
        if thought:
            self.thought = thought
            self.pairs.clear()
            self.flushes += 1
        if self.window:
            self.pairs.append((data_url, action_only.strip()))
            while len(self.pairs) > self.window:
                self.pairs.popleft()
