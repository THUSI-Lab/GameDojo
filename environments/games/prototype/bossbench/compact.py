"""Compact discrete action spaces for the public BossBench env: bossbench/compact-actions@1.

Most games use fewer than 1,000 action combinations. Terraria has a separate action space.
A game declares a few independent heads (move, turn, attack, ...), each a short list of options. One action
picks one option per head and is held for one 200 ms packet, which is the ViZDoom frame-skip analogue. The
action count is the product of the head sizes and must be below 1000.

Nothing underneath changes. An action is translated into an ordinary key/mouse packet before env.step, so the
simulation, the recordings, the verifier and the raw packet contract (bossbench/policy-gym@2) are all untouched.

Declaration: web/games/<folder>/compact_actions.json

    {"schema": "bossbench/compact-actions@1", "game": "doom_remake", "version": "doom-compact@1",
     "heads": [
       {"name": "move", "zh": "movement", "options": [
          {"label": "none"}, {"label": "forward", "zh": "forward", "keys": ["W", "Shift"]}, ...]},
       {"name": "turn", "zh": "turn", "options": [
          {"label": "none"}, {"label": "left_10", "zh": "turn left 10 degrees", "dx": -60}, ...]}]}

Option fields: label (required, [a-z0-9_]+, unique within its head); zh; note; and at most one of
    keys: [...]       held in all six sub-steps
    tap:  [...]       held in the first sub-step only, so the key is pressed afresh every packet
    subs: [[...] x 6] an explicit per-sub-step pattern (motion inputs, combos)
plus optional relative mouse counts dx, dy, dz for the packet.

Rules:
- Option 0 of every head is neutral (no keys, no mouse), so action [0, 0, ...] is the no-op packet.
- Heads never share a key or a mouse axis.
- Every combination must be a legal packet (at most 4 keys per sub-step).
"""
from __future__ import annotations

from collections.abc import Mapping
import itertools
import json
import math
from pathlib import Path
import re

import gymnasium as gym
import numpy as np

from . import keymouse as km

SCHEMA = 'bossbench/compact-actions@1'
CONTRACT = 'bossbench/policy-compact@1'
MAX_ACTIONS = 1000
EXEMPT_GAMES = frozenset({'terraria_remake'})
ROOT = Path(__file__).resolve().parents[2]
_FOLDER = {'minecraft': 'minecraft_remake'}
_LABEL = re.compile(r'[a-z0-9_]+\Z')
_HEAD = re.compile(r'[a-z][a-z0-9_]*\Z')
_AXES = ('dx', 'dy', 'dz')
_LIMIT = (1000, 1000, 5)
_FIELDS = {'label', 'zh', 'note', 'keys', 'tap', 'subs', 'dx', 'dy', 'dz'}
INVALID = 'invalid compact action'               # executed as a malformed packet: no-op, counted in format_errors


def declaration_path(game):
    return ROOT / 'web' / 'games' / _FOLDER.get(game, game) / 'compact_actions.json'


def _keyset(value, where):
    if not isinstance(value, list) or any(not isinstance(k, str) for k in value):
        raise ValueError(f'{where}: keys must be a list of key names')
    if len(set(value)) != len(value):
        raise ValueError(f'{where}: duplicate key')
    bad = [k for k in value if k not in km.KEYS]
    if bad:
        raise ValueError(f'{where}: unknown key(s) {bad}; allowed {list(km.KEYS)}')
    if len(value) > km.MAX_KEYS:
        raise ValueError(f'{where}: more than {km.MAX_KEYS} keys in one sub-step')
    return frozenset(value)


class CompactSpec:
    """A validated compact action declaration. Indices are per head; flat indices are mixed-radix, first head
    most significant, so flat 0 is the no-op."""

    def __init__(self, data, where='declaration'):
        if not isinstance(data, dict) or data.get('schema') != SCHEMA:
            raise ValueError(f'{where}: schema must be {SCHEMA!r}')
        self.where, self.game, self.version = where, data.get('game'), data.get('version')
        if not isinstance(self.game, str) or not isinstance(self.version, str) or not self.version:
            raise ValueError(f'{where}: game and version must be non-empty strings')
        heads = data.get('heads')
        if not isinstance(heads, list) or not heads:
            raise ValueError(f'{where}: heads must be a non-empty list')
        self.raw, self.heads, keys_owner, axes_owner = data, [], {}, {}
        for h in heads:
            name = h.get('name') if isinstance(h, dict) else None
            if not isinstance(name, str) or not _HEAD.match(name) or any(x['name'] == name for x in self.heads):
                raise ValueError(f'{where}: head names must match [a-z][a-z0-9_]* and be unique: {name!r}')
            opts = h.get('options')
            if not isinstance(opts, list) or len(opts) < 2:
                raise ValueError(f'{where}: head {name!r} needs at least two options')
            parsed, labels = [], set()
            for i, o in enumerate(opts):
                at = f'{where}: {name}[{i}]'
                if not isinstance(o, dict) or set(o) - _FIELDS:
                    raise ValueError(f'{at}: unknown fields {sorted(set(o) - _FIELDS) if isinstance(o, dict) else o}')
                label = o.get('label')
                if not isinstance(label, str) or not _LABEL.match(label) or label in labels:
                    raise ValueError(f'{at}: label must match [a-z0-9_]+ and be unique in the head: {label!r}')
                labels.add(label)
                kinds = [k for k in ('keys', 'tap', 'subs') if k in o]
                if len(kinds) > 1:
                    raise ValueError(f'{at}: use only one of keys / tap / subs')
                subs = [frozenset()] * km.SUBSTEPS
                if 'keys' in o:
                    subs = [_keyset(o['keys'], at)] * km.SUBSTEPS
                elif 'tap' in o:
                    subs = [_keyset(o['tap'], at)] + [frozenset()] * (km.SUBSTEPS - 1)
                elif 'subs' in o:
                    if not isinstance(o['subs'], list) or len(o['subs']) != km.SUBSTEPS:
                        raise ValueError(f'{at}: subs must list {km.SUBSTEPS} sub-steps')
                    subs = [_keyset(s, f'{at} sub {j + 1}') for j, s in enumerate(o['subs'])]
                mouse = []
                for axis, limit in zip(_AXES, _LIMIT):
                    v = o.get(axis, 0)
                    if isinstance(v, bool) or not isinstance(v, int) or abs(v) > limit:
                        raise ValueError(f'{at}: {axis} must be an integer within +-{limit}')
                    mouse.append(v)
                if i == 0 and (any(subs) or any(mouse)):
                    raise ValueError(f'{at}: option 0 must be neutral (no keys, no mouse)')
                parsed.append({'label': label, 'zh': o.get('zh'), 'subs': tuple(subs), 'mouse': tuple(mouse)})
            uni = frozenset().union(*(frozenset().union(*p['subs']) for p in parsed))
            axes = frozenset(a for p in parsed for a, v in zip(range(3), p['mouse']) if v)
            for k in uni:
                if k in keys_owner:
                    raise ValueError(f'{where}: key {k} is used by heads {keys_owner[k]!r} and {name!r}')
                keys_owner[k] = name
            for a in axes:
                if a in axes_owner:
                    raise ValueError(f'{where}: mouse {_AXES[a]} is used by heads {axes_owner[a]!r} and {name!r}')
                axes_owner[a] = name
            self.heads.append({'name': name, 'zh': h.get('zh'), 'options': parsed, 'keys': uni, 'axes': axes})
        self.nvec = [len(h['options']) for h in self.heads]
        self.size = math.prod(self.nvec)
        if self.size >= MAX_ACTIONS and self.game not in EXEMPT_GAMES:
            raise ValueError(f'{where}: {self.size} actions ({" x ".join(map(str, self.nvec))}), must be below {MAX_ACTIONS}')
        bad = None
        for idx in itertools.islice(itertools.product(*(range(n) for n in self.nvec)), 200000):
            if km.parse_packet(self.compose(idx)) is None:
                bad = idx
                break
        if bad is not None:
            raise ValueError(f'{where}: action {self.named(bad)} is not a legal packet (more than 4 keys in a sub-step?)')

    # ---- conversions
    def compose(self, idx):
        subs, mouse = [set() for _ in range(km.SUBSTEPS)], [0, 0, 0]
        for head, i in zip(self.heads, idx):
            option = head['options'][i]
            for j in range(km.SUBSTEPS):
                subs[j] |= option['subs'][j]
            for a in range(3):
                mouse[a] += option['mouse'][a]
        return km.format_packet([sorted(s, key=km.KEYS.index) for s in subs], mouse)

    def unflat(self, i):
        if not 0 <= i < self.size:
            raise ValueError(f'flat action {i} outside [0, {self.size})')
        out = []
        for n in reversed(self.nvec):
            i, r = divmod(i, n)
            out.append(r)
        return tuple(reversed(out))

    def flat(self, idx):
        i = 0
        for n, v in zip(self.nvec, idx):
            i = i * n + v
        return i

    def named(self, idx):
        return {h['name']: h['options'][i]['label'] for h, i in zip(self.heads, idx)}

    def parse(self, action):
        """Action -> per-head indices. Accepts None (no-op), a flat int, a sequence of per-head ints, a mapping
        {head: label or index} (missing heads = option 0), or text "move=forward; turn=left_10" (same rule)."""
        if action is None:
            return (0,) * len(self.heads)
        if isinstance(action, (bool, np.bool_)):
            raise ValueError('a boolean is not an action')
        if isinstance(action, (int, np.integer)):
            return self.unflat(int(action))
        if isinstance(action, str):
            text = action.strip()
            for mark in ('<|compact_start|>', '<|compact_end|>'):
                text = text.replace(mark, '')
            pairs = {}
            for part in re.split(r'[;,\n]', text):
                part = part.strip()
                if not part:
                    continue
                if '=' not in part and ':' not in part:
                    raise ValueError(f'expected head=label, got {part!r}')
                head, _, label = part.replace(':', '=', 1).partition('=')
                head, label = head.strip(), label.strip()
                if head in pairs:
                    raise ValueError(f'head {head!r} given twice')
                pairs[head] = label
            return self.parse(pairs)
        if isinstance(action, Mapping):
            names = [h['name'] for h in self.heads]
            unknown = set(action) - set(names)
            if unknown:
                raise ValueError(f'unknown head(s) {sorted(unknown)}; heads are {names}')
            out = []
            for h in self.heads:
                v = action.get(h['name'], 0)
                if isinstance(v, str):
                    labels = [o['label'] for o in h['options']]
                    if v not in labels:
                        raise ValueError(f'head {h["name"]!r} has no option {v!r}; options are {labels}')
                    out.append(labels.index(v))
                elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and 0 <= v < len(h['options']):
                    out.append(int(v))
                else:
                    raise ValueError(f'head {h["name"]!r}: bad value {v!r}')
            return tuple(out)
        arr = np.asarray(action)
        if arr.ndim != 1 or arr.shape[0] != len(self.heads) or arr.dtype.kind not in 'iu':
            raise ValueError(f'expected {len(self.heads)} integer head indices')
        if any(not 0 <= int(v) < n for v, n in zip(arr, self.nvec)):
            raise ValueError(f'head index out of range for {self.nvec}')
        return tuple(int(v) for v in arr)

    def project(self, packet):
        """Nearest compact action to a packet (packet text or parse_packet() dict), head by head.

        Key heads: Hamming distance over the 6 sub-steps x the head's own keys.
        Mouse: |difference| / 100 per axis the head owns. Ties go to the lowest index, so neutral wins a tie and
        option order matters: list the richer of two equally close options first (cuphead v3 lost jumps this way)."""
        p = km.parse_packet(packet) if isinstance(packet, str) else packet
        if p is None:
            return (0,) * len(self.heads)
        out = []
        for h in self.heads:
            best, best_d = 0, None
            for i, o in enumerate(h['options']):
                d = sum(len((p['subs'][j] & h['keys']) ^ o['subs'][j]) for j in range(km.SUBSTEPS))
                d += sum(abs(p['mouse'][a] - o['mouse'][a]) / 100 for a in h['axes'])
                if best_d is None or d < best_d - 1e-12:
                    best, best_d = i, d
            out.append(best)
        return tuple(out)

    def table(self):
        """What an agent's manual shows: every head and option with its meaning."""
        heads = []
        for h, raw in zip(self.heads, self.raw['heads']):
            heads.append({'name': h['name'], 'zh': h['zh'], 'options': [
                {'index': i, **{k: v for k, v in o.items() if k in ('label', 'zh', 'note', 'keys', 'tap', 'subs', 'dx', 'dy', 'dz')}}
                for i, o in enumerate(raw['options'])]})
        return {'schema': SCHEMA, 'contract': CONTRACT, 'game': self.game, 'version': self.version,
                'size': self.size, 'nvec': list(self.nvec), 'decision_seconds': km.PACKET_SECONDS,
                'text_form': 'head=label; head=label (missing heads = option 0)', 'heads': heads}


def load(game=None, path=None):
    p = Path(path) if path else declaration_path(game)
    if not p.is_file():
        raise FileNotFoundError(f'no compact action declaration for {game!r}: {p}')
    spec = CompactSpec(json.loads(p.read_text(encoding='utf-8')), where=str(p))
    if game and spec.game != game:
        raise ValueError(f'{p}: declares game {spec.game!r}, expected {game!r}')
    return spec


class CompactActionEnv(gym.Wrapper):
    """The public policy env with a small MultiDiscrete action space; each action becomes one ordinary packet."""

    def __init__(self, env, spec):
        super().__init__(env)
        self.compact_spec = spec            # not `spec`: gym.Wrapper.spec is the registry EnvSpec (read-only)
        self.action_space = gym.spaces.MultiDiscrete(spec.nvec)
        self.metadata = {**env.metadata, 'action_contract': CONTRACT, 'compact_version': spec.version,
                         'compact_size': spec.size}

    def manual(self):
        manual = self.env.manual()
        manual['compact_actions'] = self.compact_spec.table()
        return manual

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if 'manual' in info:
            info = {**info, 'manual': self.manual()}
        return obs, info

    def step(self, action, latency=None):
        try:
            idx = self.compact_spec.parse(action)
            packet = self.compact_spec.compose(idx)
        except (ValueError, TypeError, OverflowError):
            idx, packet = None, INVALID
        obs, reward, terminated, truncated, info = self.env.step(packet, latency=latency)
        info = {**info, 'compact_action': None if idx is None else self.compact_spec.named(idx)}
        return obs, reward, terminated, truncated, info


def make_compact_env(case_id, decl_path=None, **kwargs):
    """make_policy_env(case_id) with the game's compact action space (MultiDiscrete, fewer than 1000 actions)."""
    from .policy_env import make_policy_env
    from .scope import POLICY_CASES
    if case_id not in POLICY_CASES:
        raise ValueError(f'case is not registered for the public policy interface: {case_id!r}')
    spec = load(POLICY_CASES[case_id]['game'], decl_path)
    return CompactActionEnv(make_policy_env(case_id, **kwargs), spec)


def compact_gym_id(case_id):
    return 'BossBenchCompact/' + case_id.replace('/', '-') + '-v0'
