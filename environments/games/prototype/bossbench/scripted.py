"""Standard trusted scripted controller + frame capture for every BossBench case (all 12 games).

One API over the three engine routes, used by web/tools/bench_media.py and any trusted collector:

    env = bossbench.make_policy_env(case_id)           # public RGB interface (or bossbench.make(...) symbolic)
    ctrl = scripted.controller(env)                    # reads env.unwrapped; never handed to a policy
    obs, info = env.reset(seed=1, options={"difficulty": 0.5, "engine_seed": 1})
    cap = scripted.FrameCapture(env)                   # optional: frames of this same run at native-tick cadence
    while True:
        obs, reward, terminated, truncated, info = env.step(ctrl())
        frames = cap.drain()                           # [(tick, jpeg bytes), ...] produced during that step
        if terminated or truncated: break

Game hooks (one per engine route):
  * BB.Case web games  BB.GAME.scripted(case)    -> packet   (bridge cmd "scripted", web/engine/case.js scriptedPacket)
  * independent JS     Module.scripted(game)     -> {holds, mouse}  (fallback Module.oracle)
  * Minecraft (Python) bossbench.mc_scripted.scripted(case) -> packet text
Every controller must be deterministic and must not change the simulation; both are checked here or in JS.

HAZARD for any Python-route controller that plans on a deep copy of the live case (found 2026-11-01 in
mc_scripted): FrameCapture below installs an *instance-level* `case.tick` wrapper, and `copy.deepcopy` copies
instance attributes — so a naive clone carries that wrapper with it and the controller's lookahead then drives
the real episode while still reporting a legal packet. Strip bound functions from the clone's own __dict__
before simulating it, e.g.

    clone = copy.deepcopy(case)
    for name, value in list(vars(clone).items()):
        if callable(value) and not isinstance(value, type):
            del vars(clone)[name]          # drop instance-level patches; class methods are unaffected

The state-hash check around each controller call catches this only if the lookahead happens to change the
hash before the call returns, so do not rely on it as the guard.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import keymouse as km

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web'
REPLAY_FORMAT = 'bossbench/policy-replay@1'


def route(env):
    """'web' (BB.Case), 'native' (independent JS backend) or 'py' (Minecraft) for a make()/make_policy_env() env."""
    raw = env.unwrapped
    from .webenv import WebCaseEnv
    from .engine_env import EngineEnv
    from .unified import PyCaseEnv
    if isinstance(raw, WebCaseEnv):
        return 'web'
    if isinstance(raw, EngineEnv):
        if not hasattr(raw.backend, 'page'):
            raise TypeError(f'{raw.case_id}: backend has no scripted controller hook')
        return 'native'
    if isinstance(raw, PyCaseEnv):
        return 'py'
    raise TypeError(f'unsupported env type {type(raw).__name__}')


def _page(env):
    raw = env.unwrapped
    kind = route(env)
    if kind == 'web':
        if raw.transport != 'browser':
            return None
        return raw.t.page
    if kind == 'native':
        return raw.backend.page
    return None


def controller(env):
    """Callable returning the next legal packet text for the env's current (live) state."""
    raw = env.unwrapped
    kind = route(env)
    if kind == 'web':
        def ask():
            return raw._call({'cmd': 'scripted'})['packet']
    elif kind == 'native':
        def ask():
            out = raw.backend.scripted()
            return km.format_packet([tuple(h) for h in out['holds']], tuple(int(round(v)) for v in out['mouse']))
    else:
        from . import mc_scripted

        def ask():
            before = raw.state_hash() if raw.snap is not None else None
            packet = mc_scripted.scripted(raw.case)
            if before is not None and raw.state_hash() != before:
                raise RuntimeError('scripted controller changed the simulation')
            return packet if isinstance(packet, str) else km.to_packet(packet)

    def next_packet():
        packet = ask()
        if km.parse_packet(packet) is None:
            raise RuntimeError(f'{raw.case_id}: scripted controller returned an illegal packet: {packet!r}')
        return packet
    return next_packet


def rules(env):
    """Rule/version string the engine declares for this case (None if the game has none)."""
    raw = env.unwrapped
    kind = route(env)
    if kind == 'web':
        return raw._call({'cmd': 'meta'}).get('rules')
    if kind == 'native':
        return raw.engine_spec.rule_version
    from . import mc4d
    return getattr(mc4d, 'RULES', None)


def state_hash(env):
    """Engine-native final-state hash (BB.hashState / NativeBench.hash / sha1 of the Python state)."""
    raw = env.unwrapped
    kind = route(env)
    if kind == 'web':
        return raw._call({'cmd': 'hash'})['hash']
    if kind == 'native':
        return raw.backend.page.evaluate('() => NativeBench.hash(backendSession.state())')
    return raw.state_hash()


def game_of(env):
    raw = env.unwrapped
    from .scope import POLICY_CASES
    return POLICY_CASES[raw.case_id]['game']


def source_files(env):
    """Files that define this case's rules and pictures (fingerprinted in media provenance)."""
    raw = env.unwrapped
    kind = route(env)
    if kind == 'web':
        from .webenv import game_scripts
        files = [Path(p) for p in game_scripts(raw.game)]
        files += [WEB / 'games' / raw.game / 'index.html', WEB / 'engine' / 'bridge.js']
    elif kind == 'native':
        from .native_games import source_paths
        files = source_paths(raw.case_id)
    else:
        pkg = Path(__file__).resolve().parent
        files = [pkg / name for name in ('mc4d.py', 'mcskin.py', 'rig.py', 'core.py', 'unified.py', 'mc_scripted.py')]
    out = {}
    for path in files:
        if path.is_file():
            out[path.resolve().relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


_WEB_CAPTURE = r'''a => {
  const env = BB.bridgeEnv(), c = env.case;
  if (!BB.Renderer) throw Error('capture needs the browser renderer');
  const canvas = document.createElement('canvas');
  const cap = window.__bbCapture = {R: new BB.Renderer(canvas), every: a.every, quality: a.quality, frames: [], ticks: [], case: c};
  cap.grab = () => {
    const before = BB.hashState(c);
    cap.R.draw(c, c.t / BB.FPS, '');
    if (BB.hashState(c) !== before) throw Error('renderer changed the simulation');
    cap.frames.push(cap.R.canvas.toDataURL('image/jpeg', cap.quality).split(',')[1]); cap.ticks.push(c.t);
  };
  if (!c.__bbCaptureWrapped) {
    const tick = c.tick;
    c.tick = function (keys, mouse) {
      const t0 = this.t; tick.call(this, keys, mouse);
      const k = window.__bbCapture;
      if (k && k.case === this && this.t !== t0 && (this.t % k.every === 0 || this.result)) k.grab();
    };
    c.__bbCaptureWrapped = true;
  }
  cap.grab();
  return {w: cap.R.canvas.width, h: cap.R.canvas.height, hz: BB.FPS};
}'''

_NATIVE_CAPTURE = r'''a => {
  const s = backendSession, M = backendModule;
  const canvas = document.createElement('canvas'); canvas.width = 854; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  const cap = window.__bbCapture = {every: a.every, quality: a.quality, frames: [], ticks: [], session: s, ctx, canvas};
  cap.grab = () => {
    const before = NativeBench.hash(s.state());
    M.draw(s.game, ctx);
    if (NativeBench.hash(s.state()) !== before) throw Error('renderer changed the simulation');
    cap.frames.push(canvas.toDataURL('image/jpeg', cap.quality).split(',')[1]); cap.ticks.push(s.game.t);
  };
  const tick = s.tick.bind(s);
  s.tick = function (keys, mouse) {
    const ok = tick(keys, mouse), k = window.__bbCapture;
    if (ok && k && k.session === s && (s.game.t % k.every === 0 || s.game.outcome)) k.grab();
    return ok;
  };
  cap.grab();
  return {w: 854, h: 480, hz: M.hz};
}'''

_DRAIN = '''() => { const k = window.__bbCapture; if (!k) return {frames: [], ticks: []};
  const out = {frames: k.frames, ticks: k.ticks}; k.frames = []; k.ticks = []; return out; }'''

_CLOSE_NATIVE = '''() => { const k = window.__bbCapture; if (k && k.ctx && backendModule.closeRenderer) backendModule.closeRenderer(k.ctx);
  window.__bbCapture = null; }'''


class FrameCapture:
    """Frames of the live episode every `every` native ticks (default: 30 fps at 60 Hz, every tick below 40 Hz).

    Install right after reset; frames come from a private renderer/canvas reading the same simulation, so the
    policy observation, gameplay RNG and physics are untouched. The terminal tick is always captured.
    """

    def __init__(self, env, every=None, quality=0.92):
        self.env, self.kind, self.raw = env, route(env), env.unwrapped
        fps = int(self.raw.fps)
        self.every = int(every or (2 if fps >= 40 else 1))
        self.fps = fps / self.every
        self.quality = float(quality)
        self._py_frames = []
        if self.kind in ('web', 'native'):
            page = _page(env)
            if page is None:
                raise RuntimeError('frame capture needs the browser transport (obs_mode="rgb")')
            self.page = page
            script = _WEB_CAPTURE if self.kind == 'web' else _NATIVE_CAPTURE
            self.size = page.evaluate(script, {'every': self.every, 'quality': self.quality})
        else:
            case, inner, frames, every = self.raw.case, self.raw.inner, self._py_frames, self.every
            tick = type(case).tick

            def wrapped(action, _case=case):
                t0 = _case.t
                tick(_case, action)
                if _case.t != t0 and (_case.t % every == 0 or _case.outcome()):
                    frames.append((_case.t, inner.render()))
            case.tick = wrapped
            self._unwrap = lambda: case.__dict__.pop('tick', None)
            frames.append((case.t, inner.render()))
            self.size = {'w': case.size[0], 'h': case.size[1], 'hz': fps}

    def drain(self):
        """-> [(tick, jpeg bytes)] captured since the last drain."""
        if self.kind in ('web', 'native'):
            got = self.page.evaluate(_DRAIN)
            return [(t, base64.b64decode(f)) for t, f in zip(got['ticks'], got['frames'])]
        from PIL import Image
        out = []
        for t, rgb in self._py_frames:
            buffer = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, 'JPEG', quality=int(self.quality * 100))
            out.append((t, buffer.getvalue()))
        self._py_frames.clear()
        return out

    def close(self):
        try:
            if self.kind == 'native':
                self.page.evaluate(_CLOSE_NATIVE)
            elif self.kind == 'web':
                self.page.evaluate('() => { window.__bbCapture = null; }')
            else:
                self._unwrap()
        except Exception:
            pass


def play(env, *, seed=1, options=None, max_steps=None, on_step=None, capture=None):
    """Run the scripted controller to the terminal outcome on an existing env; returns a policy-replay dict.

    options are forwarded to reset (difficulty, engine-specific keys); engine_seed defaults to seed so the replay
    and the human play page share the same world. on_step(index, packet, info, frames) is called after each step.
    """
    options = dict(options or {})
    if 'tier' not in options:
        options.setdefault('difficulty', 0.5)
    options.setdefault('engine_seed', int(seed))
    obs, info = env.reset(seed=int(seed), options=options)
    # A replay must pin what was actually played. A `tier` is resolved by the game's declaration at reset
    # time; store the concrete difficulty it produced (plus the tier name as provenance), so the recording
    # still reproduces exactly if that game later recalibrates its tiers.
    if 'tier' in options:
        tier = options.pop('tier')
        options['difficulty'] = float(info['difficulty'])
        options['resolved_from_tier'] = tier
    cap = FrameCapture(env, **capture) if capture is not None else None
    if cap is not None and on_step is not None:
        on_step(-1, None, info, cap.drain())
    ctrl = controller(env)
    raw = env.unwrapped
    packets, terminated, truncated = [], False, False
    limit = max_steps or int(raw.max_seconds / km.PACKET_SECONDS) + 2
    try:
        for index in range(limit):
            packet = ctrl()
            obs, reward, terminated, truncated, info = env.step(packet)
            packets.append(packet)
            if on_step is not None:
                on_step(index, packet, info, cap.drain() if cap is not None else [])
            if terminated or truncated:
                break
    finally:
        if cap is not None:
            cap.close()
    verifier = raw.verifier()
    ticks = int(round(raw.snap['t'] * raw.fps))
    return {
        'format': REPLAY_FORMAT, 'case_id': raw.case_id, 'game': game_of(env), 'contract': 'bossbench/policy-gym@2',
        'rules': rules(env), 'native_hz': raw.fps, 'gym_seed': int(seed), 'engine_seed': int(raw.engine_seed),
        'options': options, 'controller': 'scripted (standard trusted reference controller; privileged state; not a model)',
        'packets': packets, 'steps': len(packets), 'ticks': ticks, 'seconds': round(ticks / raw.fps, 4),
        'outcome': verifier['outcome'], 'terminated': bool(terminated), 'truncated': bool(truncated),
        'verifier': verifier, 'final_hash': state_hash(env), 'source_sha256': source_files(env),
    }


def replay(env, data):
    """Re-run a policy-replay through env; returns (outcome, final_hash, ticks, problems)."""
    problems = []
    raw = env.unwrapped
    if data.get('format') != REPLAY_FORMAT:
        problems.append('not a bossbench/policy-replay@1 file')
    if data.get('case_id') != raw.case_id:
        problems.append(f"case mismatch: {data.get('case_id')} vs {raw.case_id}")
    current = rules(env)
    if data.get('rules') != current:
        problems.append(f"rules mismatch: recording {data.get('rules')!r}, engine {current!r}")
    env.reset(seed=int(data['gym_seed']), options=dict(data['options']))
    done = False
    for index, packet in enumerate(data['packets']):
        if done:
            problems.append(f'packet {index} after terminal outcome')
            break
        _, _, terminated, truncated, _ = env.step(packet)
        done = terminated or truncated
    if not done:
        problems.append('recording ended before a terminal outcome')
    outcome = raw.verifier()['outcome']
    final = state_hash(env)
    ticks = int(round(raw.snap['t'] * raw.fps))
    if outcome != data.get('outcome'):
        problems.append(f"outcome mismatch: {outcome} vs {data.get('outcome')}")
    if final != data.get('final_hash'):
        problems.append(f"final hash mismatch: {final} vs {data.get('final_hash')}")
    if ticks != data.get('ticks'):
        problems.append(f"tick mismatch: {ticks} vs {data.get('ticks')}")
    return outcome, final, ticks, problems


def dumps(data):
    return json.dumps(data, ensure_ascii=False, indent=1) + '\n'
