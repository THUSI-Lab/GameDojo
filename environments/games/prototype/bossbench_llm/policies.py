"""Policies the harness can drive. Every policy sees only the chat messages (system prompt, frames, its own
replies) plus a small public `turn` record (index, case, format, the game's key names / action vocabulary).

* chat     an OpenAI-compatible endpoint (the real thing)
* python   your own code: --policy python --policy-factory my_module:make  -> object with act(messages, turn)
* replay   a recorded scripted win, converted to the chosen format (pipeline self-test; lumine is lossless)
* noop / random   baselines and smoke tests
"""
from __future__ import annotations

import importlib
import json
import random

from bossbench import keymouse as km

from . import lumine


class Turn(dict):
    """Public per-turn context: index, case_id, game, format, lumine_keys, vocab (semantic only)."""
    __getattr__ = dict.get


class ChatPolicy:
    name = "chat"

    def __init__(self, client):
        self.client = client

    def act(self, messages, turn):
        return self.client.complete(messages)

    def describe(self):
        return {"policy": self.name, **self.client.describe()}


class PythonPolicy:
    name = "python"

    def __init__(self, factory, config=None):
        module, _, attr = factory.partition(":")
        self.factory = factory
        self.impl = getattr(importlib.import_module(module), attr or "make")(dict(config or {}))

    def act(self, messages, turn):
        out = self.impl.act(messages, turn)
        return out if isinstance(out, tuple) else (out, {})

    def describe(self):
        info = getattr(self.impl, "describe", None)
        return {"policy": self.name, "factory": self.factory, **(info() if callable(info) else {})}


class ReplayPolicy:
    """Replays a bossbench/policy-replay@1 recording. Lumine text converts back to exactly the recorded packets, so
    a lumine replay must reproduce the recording's final hash; semantic replays use the nearest compact action
    (compact.CompactSpec.project) and are a smoke test only."""
    name = "replay"

    def __init__(self, path, fmt, mouse_kind="none", spec=None):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.path, self.fmt, self.mouse_kind, self.spec = str(path), fmt, mouse_kind, spec
        self.index = 0

    def act(self, messages, turn):
        packets = self.data["packets"]
        packet = packets[self.index] if self.index < len(packets) else km.NOOP
        self.index += 1
        if self.fmt == "lumine":
            return lumine.from_packet(packet, mouse_kind=self.mouse_kind), {"seconds": 0.0}
        idx = self.spec.project(packet)
        return json.dumps(self.spec.named(idx)), {"seconds": 0.0}

    def describe(self):
        return {"policy": self.name, "replay": self.path, "case_id": self.data.get("case_id"),
                "rules": self.data.get("rules"), "outcome": self.data.get("outcome")}


class NoopPolicy:
    name = "noop"

    def __init__(self, fmt):
        self.fmt = fmt

    def act(self, messages, turn):
        return (lumine.format_action((0, 0, 0), [[]] * km.SUBSTEPS) if self.fmt == "lumine" else "{}"), {"seconds": 0.0}

    def describe(self):
        return {"policy": self.name}


class RandomPolicy:
    """Uniform over this game's own keys (lumine: each chunk 0-2 keys, mouse +-100) or over its action options."""
    name = "random"

    def __init__(self, fmt, seed=0):
        self.fmt, self.rng, self.seed = fmt, random.Random(seed), seed

    def act(self, messages, turn):
        if self.fmt == "lumine":
            keys = list(turn.lumine_keys or [])
            groups = [sorted(self.rng.sample(keys, min(len(keys), self.rng.randint(0, 2)))) for _ in range(km.SUBSTEPS)]
            mouse = (self.rng.randint(-100, 100), self.rng.randint(-100, 100), 0) if turn.mouse_kind != "none" else (0, 0, 0)
            return lumine.format_action(mouse, groups), {"seconds": 0.0}
        vocab = turn.vocab
        return json.dumps({h["name"]: self.rng.choice(h["options"])["label"] for h in vocab.heads}), {"seconds": 0.0}

    def describe(self):
        return {"policy": self.name, "seed": self.seed}
