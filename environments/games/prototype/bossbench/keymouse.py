"""The key/mouse action packet shared by every BossBench case (proposal 4.4), Python side.

    <|action_start|> dx dy dz ; K1 ; K2 ; K3 ; K4 ; K5 ; K6 <|action_end|>

One packet = 200 ms = 6 sub-steps of 33 ms; K_j = 0-4 keys held during sub-step j. The parser is a
line-for-line port of web/engine/case.js BB.parsePacket, so a
packet means the same thing whether the case runs in Python or in the browser engine.

Both engines tick at 60 Hz (2 ticks per sub-step); `schedule` spreads a packet over any number of
ticks per 200 ms all the same: a key counts as held in a tick if it is held during any
sub-step overlapping that tick, and a press (released -> held between sub-steps) lands in the tick
where that sub-step starts. At 60 Hz this is exactly the web engine's 2 ticks per sub-step.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import gymnasium as gym
import numpy as np

KEYS = ("W", "A", "S", "D", "Space", "Shift", "J", "K", "L", "Q", "E", "R", "LMB", "RMB")
_UP = {k.upper(): k for k in KEYS}
SUBSTEPS = 6
PACKET_SECONDS = 0.2
MAX_KEYS = 4
_PACKET = re.compile(r"\s*<\|action_start\|>([\s\S]*)<\|action_end\|>\s*")
_INT = re.compile(r"-?[0-9]+")


def parse_packet(text):
    """-> {"mouse": (dx, dy, dz), "subs": [frozenset(keys)] * 6}, or None if the packet is malformed."""
    m = _PACKET.fullmatch("" if text is None else str(text))
    if not m:
        return None
    parts = m.group(1).split(";")
    if len(parts) != 1 + SUBSTEPS:
        return None
    mouse = parts[0].split()
    if len(mouse) != 3 or not all(_INT.fullmatch(v) for v in mouse):
        return None
    # Canonicalize leading zeros before int(): Python limits decimal input size,
    # whereas JS rejects enormous values as out of range. Neither may crash.
    normalized = []
    for value in mouse:
        digits = value.lstrip("-").lstrip("0") or "0"
        if len(digits) > 4:
            return None
        normalized.append(("-" if value.startswith("-") else "") + digits)
    dx, dy, dz = (int(v) for v in normalized)
    if abs(dx) > 1000 or abs(dy) > 1000 or abs(dz) > 5:
        return None
    subs = []
    for s in parts[1:]:
        s, held = s.strip(), set()
        if s:
            ks = s.split(",")
            if len(ks) > MAX_KEYS:
                return None
            for k in ks:
                k = _UP.get(k.strip().upper())
                if k is None or k in held:
                    return None
                held.add(k)
        subs.append(frozenset(held))
    return {"mouse": (dx, dy, dz), "subs": subs}


def format_packet(subs, mouse=(0, 0, 0)) -> str:
    """subs: 6 iterables of key names. Does not validate: too many keys gives a packet the env rejects."""
    return "<|action_start|> " + " ".join(str(int(v)) for v in mouse) + " ; " + \
        " ; ".join(",".join(_UP.get(str(k).upper(), str(k)) for k in s) for s in subs) + " <|action_end|>"


NOOP = format_packet([()] * SUBSTEPS)


def to_packet(action) -> str:
    """Anything an agent may hand to env.step -> packet text.

    str                                   the packet itself (LLM agents)
    None                                  NOOP
    {"mouse": (dx,dy,dz), "keys": 6x14}   KeyMouse space sample: 0/1 per sub-step and key, in KEYS order
    {"mouse": ..., "keys": [["D"], ["D", "Space"], ...]}   key names per sub-step
    """
    if action is None:
        return NOOP
    if isinstance(action, str):
        return action
    # Invalid structured output has the same semantics as invalid packet text:
    # the env executes a no-op and counts it, without coercing fractions to ints.
    if not isinstance(action, Mapping) or set(action) - {"mouse", "keys"}:
        return "invalid structured action"
    try:
        mouse = np.asarray(action.get("mouse", (0, 0, 0)))
        if (mouse.shape != (3,) or mouse.dtype.kind not in "iuf" or
                not np.isfinite(mouse).all() or not (mouse == np.trunc(mouse)).all() or
                not (np.abs(mouse) <= (1000, 1000, 5)).all()):
            return "invalid structured action"
        rows = action.get("keys", [()] * SUBSTEPS)
        if isinstance(rows, (str, bytes)) or len(rows) != SUBSTEPS:
            return "invalid structured action"
        if any(isinstance(row, (str, bytes)) for row in rows):
            return "invalid structured action"
        try:
            matrix = np.asarray(rows)
        except ValueError:  # ragged lists of key names
            matrix = None
        if matrix is not None and matrix.shape == (SUBSTEPS, len(KEYS)) and matrix.dtype.kind in "biuf":
            if not ((matrix == 0) | (matrix == 1)).all():
                return "invalid structured action"
            rows = [[KEYS[i] for i in np.flatnonzero(row)] for row in matrix]
        elif not all(all(isinstance(key, str) for key in row) for row in rows):
            return "invalid structured action"
        return format_packet(rows, mouse)
    except (TypeError, ValueError, OverflowError):
        return "invalid structured action"


def schedule(packet, n_ticks, prev=frozenset()):
    """Spread a parsed packet (None = no-op) over n_ticks engine ticks.

    -> ([(held, pressed, (dx, dy, dz)) per tick], held during the last sub-step)
    `prev` is the last sub-step of the previous packet, so a key held across packets is not re-pressed.
    """
    subs = packet["subs"] if packet else [frozenset()] * SUBSTEPS
    mouse = np.asarray(packet["mouse"] if packet else (0, 0, 0), float) / n_ticks
    held = [set() for _ in range(n_ticks)]
    pressed = [set() for _ in range(n_ticks)]
    last = prev
    for j, s in enumerate(subs):
        for k in range(n_ticks):                                 # sub-step [j/6, (j+1)/6) overlaps tick [k/n, (k+1)/n)
            if j * n_ticks < SUBSTEPS * (k + 1) and k * SUBSTEPS < (j + 1) * n_ticks:
                held[k] |= s
        pressed[j * n_ticks // SUBSTEPS] |= s - last
        last = s
    return [(frozenset(h), frozenset(p), tuple(mouse)) for h, p in zip(held, pressed)], last


class KeyMouse(gym.spaces.Dict):
    """Structured form of a packet for RL: mouse (dx, dy, dz) + a 6 x 14 hold matrix (sub-step x KEYS).

    env.step also accepts the packet text directly. Samples respect the 0-4 keys per sub-step rule;
    an action that breaks it is a malformed packet (executed as no-op, counted in format_errors)."""

    def __init__(self, seed=None):
        super().__init__({"mouse": gym.spaces.Box(np.array([-1000, -1000, -5]), np.array([1000, 1000, 5]),
                                                  dtype=np.int64),
                          "keys": gym.spaces.MultiBinary([SUBSTEPS, len(KEYS)])}, seed=seed)

    def sample(self, mask=None, probability=None):
        out = super().sample(mask, probability)
        out["keys"] = _constrained_keys(out["keys"], self.np_random,
                                        None if mask is None else mask["keys"],
                                        None if probability is None else probability["keys"])
        return out

    def contains(self, x) -> bool:
        return super().contains(x) and bool((np.asarray(x["keys"]).sum(1) <= MAX_KEYS).all())


def _constrained_keys(sample, rng, mask=None, probability=None):
    keys = np.asarray(sample).copy()
    rows = keys.reshape(-1, len(KEYS))
    if mask is None and probability is None:
        rows.fill(0)
        for row in rows:
            row[rng.choice(len(KEYS), rng.integers(0, MAX_KEYS + 1), replace=False)] = 1
    else:
        forced = (mask == 1) if mask is not None else (probability == 1)
        for row, fixed in zip(rows, np.asarray(forced).reshape(-1, len(KEYS))):
            if fixed.sum() > MAX_KEYS:
                raise ValueError("key mask/probability forces more than four keys in a substep")
            candidates = np.flatnonzero(row & ~fixed)
            keep = MAX_KEYS - int(fixed.sum())
            if len(candidates) > keep:
                row[rng.choice(candidates, len(candidates) - keep, replace=False)] = 0
    return keys


class BatchedKeyMouse(gym.spaces.Dict):
    """Gym's vector action space preserving KeyMouse's per-substep constraint."""

    def __init__(self, single, n):
        from gymnasium.vector.utils import batch_space
        base = batch_space(gym.spaces.Dict(dict(single.spaces)), n)
        spaces = dict(base.spaces)
        # Gym batches MultiBinary as Box, which loses mask/probability support.
        # Keep binary semantics as well as the leading member dimension.
        spaces["keys"] = gym.spaces.MultiBinary([n, SUBSTEPS, len(KEYS)])
        super().__init__(spaces)

    def sample(self, mask=None, probability=None):
        out = super().sample(mask, probability)
        out["keys"] = _constrained_keys(out["keys"], self.np_random,
                                        None if mask is None else mask["keys"],
                                        None if probability is None else probability["keys"])
        return out

    def contains(self, x):
        return super().contains(x) and bool((np.asarray(x["keys"]).sum(-1) <= MAX_KEYS).all())


# Default Dict batching drops subclass invariants. Register the real batch
# space so SyncVectorEnv/AsyncVectorEnv samplers still emit legal actions.
from gymnasium.vector.utils import batch_space


@batch_space.register(KeyMouse)
def _batch_keymouse(space, n=1):
    return BatchedKeyMouse(space, n)
