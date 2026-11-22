"""Semantic (named) actions: one JSON object per 200 ms decision, e.g. {"move": "forward", "turn": "left_10"}.

The vocabulary of every game is read from the game's own declaration, never re-typed here:
* 18 games: web/games/<game>/compact_actions.json (bossbench/compact-actions@1, the 2026-11-10 ViZDoom-style
  action heads). One semantic action = one ordinary 200 ms key/mouse packet (bossbench.make_compact_env).
* terraria_remake (exempt from compact actions): its LLM semantic interface terraria-semantic@1
  (web/games/terraria_remake/semantic_ppo_v1): move / vert / attack, where attack "shoot <deg>" aims relative to
  the player's own body. Its native window is 100 ms, so one 200 ms decision runs two identical windows.

Model output: a JSON object {head: option label}; omitted heads take option 0 ("none"). The text form
"head=label; head=label" is accepted too. Anything else runs a no-op and counts as a format error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re

from .lumine import strip_thoughts

FORMAT = "semantic-action@1"
TERRARIA = "terraria_remake"

_TERRARIA_HEADS = [
    {"name": "move", "en": "horizontal movement", "options": [
        {"label": "none", "en": "stay"}, {"label": "left", "en": "move left"},
        {"label": "right", "en": "move right"}]},
    {"name": "vert", "en": "jump / drop", "options": [
        {"label": "none", "en": "nothing"},
        {"label": "jump", "en": "jump; keep choosing it to fly or glide"},
        {"label": "drop", "en": "drop through the platform below"}]},
    {"name": "attack", "en": "attack", "options": (
        [{"label": "none", "en": "no attack"}]
        + [{"label": f"shoot {i * 22.5:g}",
            "en": f"shoot an arrow toward {i * 22.5:g}° (0° = right, 90° = up)"} for i in range(16)]
        + [{"label": "slash left", "en": "swing the sword to the left"},
           {"label": "slash right", "en": "swing the sword to the right"}])},
]
_SHOOT = re.compile(r"shoot[\s_]*([+-]?\d+(?:\.\d+)?)\s*(?:°|deg|degrees)?", re.I)


def _norm(label):
    return re.sub(r"[\s\-_]+", "_", str(label).strip().lower())


@dataclass
class SemParsed:
    ok: bool
    action: dict | None = None        # canonical {head: label} for every head
    thought: str | None = None
    lenient_fixes: int = 0
    error: str | None = None

    @property
    def text(self):
        return json.dumps(self.action, ensure_ascii=False) if self.ok else None


class Vocabulary:
    """The named action heads of one game."""

    def __init__(self, game, heads, *, window_seconds=0.2, version=None, source=None):
        self.game, self.heads, self.window_seconds = game, heads, window_seconds
        self.version, self.source = version, source
        self.names = [h["name"] for h in heads]

    @classmethod
    def for_game(cls, game):
        if game == TERRARIA:
            return cls(game, _TERRARIA_HEADS, window_seconds=0.1, version="terraria-semantic@1",
                       source="web/games/terraria_remake/semantic_ppo_v1/actions.py")
        from bossbench import compact
        spec = compact.load(game)
        heads = [{"name": h["name"], "zh": h.get("zh"),
                  "options": [{k: v for k, v in o.items() if k in ("label", "zh", "en", "note", "keys", "tap", "subs", "dx", "dy", "dz")}
                              for o in h["options"]]}
                 for h in spec.raw["heads"]]
        return cls(game, heads, version=spec.version,
                   source=compact.declaration_path(game).relative_to(compact.ROOT).as_posix())

    @property
    def size(self):
        return math.prod(len(h["options"]) for h in self.heads)

    def labels(self, head):
        return [o["label"] for o in next(h for h in self.heads if h["name"] == head)["options"]]

    def neutral(self):
        return {h["name"]: h["options"][0]["label"] for h in self.heads}

    def _label(self, head, value, lenient):
        labels = self.labels(head)
        if not isinstance(value, str):
            if isinstance(value, int) and not isinstance(value, bool) and lenient and 0 <= value < len(labels):
                return labels[value], True
            raise ValueError(f"{head}: option must be a label string, got {value!r}")
        if value in labels:
            return value, False
        if self.game == TERRARIA and head == "attack":
            m = _SHOOT.fullmatch(value.strip())
            if m:
                k = math.floor(float(m.group(1)) / 22.5 + 0.5) % 16     # nearest 22.5 deg, ties upward, wraps
                return f"shoot {k * 22.5:g}", True
        by_norm = {_norm(x): x for x in labels}
        if _norm(value) in by_norm:
            return by_norm[_norm(value)], True
        raise ValueError(f"{head}: no option {value!r}; options are {labels}")

    def canonical(self, mapping, lenient=True):
        unknown = [k for k in mapping if k not in self.names]
        if unknown:
            raise ValueError(f"unknown action group(s) {unknown}; groups are {self.names}")
        action, fixes = self.neutral(), 0
        for head, value in mapping.items():
            label, fixed = self._label(head, value, lenient)
            action[head] = label
            fixes += fixed
        return action, fixes

    def parse(self, reply, *, lenient=True):
        """Model reply -> SemParsed. Never raises on model output."""
        text, thought = strip_thoughts("" if reply is None else str(reply))
        for mark in ("<|semantic_start|>", "<|semantic_end|>", "<|compact_start|>", "<|compact_end|>"):
            text = text.replace(mark, " ")
        mapping = None
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    mapping = obj
                    break
        if mapping is None:
            body = text.strip().strip("`").strip()
            if body and ("=" in body or ":" in body):
                mapping = {}
                try:
                    for part in re.split(r"[;\n,]", body):
                        part = part.strip()
                        if not part:
                            continue
                        head, sep, label = part.replace(":", "=", 1).partition("=")
                        if not sep:
                            raise ValueError(f"expected group=option, got {part!r}")
                        head, label = head.strip().strip('"\''), label.strip().strip('"\'')
                        if head in mapping:
                            raise ValueError(f"group {head!r} given twice")
                        mapping[head] = label
                except ValueError as e:
                    return SemParsed(False, thought=thought, error=str(e))
        if mapping is None:
            return SemParsed(False, thought=thought, error="no JSON object in the reply")
        try:
            action, fixes = self.canonical(mapping, lenient)
        except ValueError as e:
            return SemParsed(False, thought=thought, error=str(e))
        return SemParsed(True, action=action, thought=thought, lenient_fixes=fixes)
