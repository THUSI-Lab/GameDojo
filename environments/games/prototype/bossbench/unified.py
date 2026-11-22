"""One Gymnasium interface for every BossBench case, whatever engine runs it (proposal 4.4).

    import bossbench
    env = bossbench.make("2d/needle_warden",            # web engine (JS, 60 Hz)
                         obs_mode="symbolic",           # "rgb" | "symbolic" | "vector"
                         time_mode="virtual_rt",        # "paused" | "virtual_rt" | "wallclock_rt"
                         latency=0.3)
    env = bossbench.make("3d/minecraft_wither")          # Python engine (60 Hz) - same API
    obs, info = env.reset(seed=0, options={"difficulty": 0.5})
    obs, reward, terminated, truncated, info = env.step(
        "<|action_start|> 120 0 0 ; W ; W ; W,Space ; W ; LMB ; LMB <|action_end|>")

What is identical across cases (and is all an agent or trainer needs to know):
  step         one 200 ms key/mouse packet (keymouse.py): str, or a dict from env.action_space
  action_space keymouse.KeyMouse()  (mouse dx dy dz + 6 x 14 hold matrix); the key *meaning* is per
               body and is spelled out in info["manual"]["controls"]
  obs          rgb: 480 x 854 x 3 uint8 | symbolic: the case's state dict | vector: float32 Box
  time         paused: the world waits; virtual_rt: ceil(latency / 0.2) no-op packets run first;
               wallclock_rt: floor(wall time since the last return / 0.2) no-op packets run first
  reward       sparse: verifier reward at the end only; dense: + d(boss damage) - d(player damage)
  info         t, state, events, outcome, player_hp, player_max_hp, boss_hp, boss_max_hp, format_ok,
               format_errors, steps, frames, lag_packets, difficulty, engine_seed
               (+ manual at reset, verifier on the last step)
  verifier()   {reward in [-1, 1], win, outcome, boss_damage, seconds, d, player_hp}
  end          win / loss -> terminated; time cap -> truncated; stepping a finished episode raises

info["state"] is always filled so oracles and loggers can read it; an rgb-only agent harness must not
pass it to the agent (that is the perception-gap ablation's job, not the env's).

action_mode="native" (Python engine only) keeps the body's own action space (e.g. MultiDiscrete for
Steve) for RL ablations; one step is still one 200 ms packet with that action held throughout.
"""
from __future__ import annotations

import hashlib
import copy
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
import time

import gymnasium as gym
import numpy as np

from . import keymouse as km
from .core import RES
from .engine_contract import EngineSpec, RenderStatus

OBS_MODES = ("rgb", "symbolic", "vector")
TIME_MODES = ("paused", "virtual_rt", "wallclock_rt")
PROTOCOL = ("one step = one 200 ms packet: <|action_start|> dx dy dz ; K1 ; K2 ; K3 ; K4 ; K5 ; K6 <|action_end|>; "
            "K_j = 0-4 keys from W A S D Space Shift J K L Q E R LMB RMB held during the j-th 33 1/3 ms sub-step (2 ticks at 60 Hz), "
            "comma separated; dx dy in [-1000, 1000] mouse motion, dz in [-5, 5] wheel; "
            "a malformed packet runs as no-op and counts as a format error")


class Json(gym.Space):
    """Observation space of obs_mode='symbolic': a JSON-serialisable dict (the case's state())."""

    def __init__(self, seed=None):
        super().__init__(None, None, seed)

    def contains(self, x) -> bool:
        return isinstance(x, dict)

    def sample(self, mask=None, probability=None):
        return {}

    def __repr__(self):
        return "Json()"

    def __eq__(self, other):
        return isinstance(other, Json)


def event_names(events):
    return [e[0] if isinstance(e, (tuple, list)) else e for e in events]


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _finite_number(value, name, *, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number >= 0" + (f" and <= {upper}" if upper is not None else ""))
    value = float(value)
    if not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ValueError(f"{name} must be a finite number >= 0" + (f" and <= {upper}" if upper is not None else ""))
    return value


def _clean_band(value, where):
    """Validate a game's difficulty_band declaration: a short plain name, or None for 'no segmentation'."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 40 or '\n' in value:
        raise ValueError(f"{where}: difficulty_band must return a short non-empty string or None, "
                         f"got {value!r}")
    return value.strip()


def _reset_options(options):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("reset options must be a mapping")
    return copy.deepcopy(dict(options or {}))


class UnifiedEnv(gym.Env):
    """Engine-independent half: action decoding, time modes, reward, info/verifier schema.

    An engine adapter implements _engine_reset(seed, d, options) and _engine_packet(packet, lag, ok), both returning a
    snapshot dict {state, events, outcome (None | win | loss | timeout), t, frames, player_hp, player_max_hp,
    boss_hp, boss_max_hp}, plus _render / _vector / _manual, and sets fps and max_seconds.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": round(1 / km.PACKET_SECONDS)}

    def __init__(self, case_id, obs_mode="symbolic", action_mode="keymouse", time_mode="paused", latency=0.0,
                 difficulty=0.5, reward_mode="sparse", render_mode=None):
        for name, value, choices in (("obs_mode", obs_mode, OBS_MODES), ("time_mode", time_mode, TIME_MODES),
                                     ("action_mode", action_mode, ("keymouse", "native")),
                                     ("reward_mode", reward_mode, ("sparse", "dense")),
                                     ("render_mode", render_mode, (None, "rgb_array"))):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        self.case_id, self.obs_mode, self.action_mode, self.time_mode = case_id, obs_mode, action_mode, time_mode
        self.latency = self._default_latency = _finite_number(latency, "latency")
        self.difficulty = _finite_number(difficulty, "difficulty", upper=1)
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.action_space = km.KeyMouse()
        self.snap, self.done, self.d, self.engine_seed = None, True, self.difficulty, None
        self._closed = False

    def _obs_space(self, vector_space=None):
        if self.obs_mode == "rgb":
            return gym.spaces.Box(0, 255, (RES[1], RES[0], 3), np.uint8)
        if self.obs_mode == "vector":
            if vector_space is None:
                raise ValueError(f"{self.case_id}: obs_mode='vector' is not available for this engine")
            return vector_space
        return Json()

    @property
    def engine_spec(self):
        """Default for the existing web engine; independent backends override it."""
        return EngineSpec('web-canvas', '2d', 60, self.max_seconds, 'canvas', 'raster')

    def render_status(self):
        return RenderStatus(self.engine_spec.renderer, self.engine_spec.render_kind, True)

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Reset using Gym's seed, or an explicit uint32 engine_seed for imported replays.

        Ordinary seed handling is unchanged. options are forwarded to the web
        game so its initial world settings match browser play.
        """
        self._ensure_open()
        options = _reset_options(options)
        # A public `tier` (easy/normal/hard) resolves to the continuous difficulty the engines understand,
        # using the game's own declared mapping (see tiers.py), so the play page and this interface agree.
        from .tiers import resolve_options
        options = resolve_options(options, self._declared_tiers())
        d = _finite_number(options.get("difficulty", self.difficulty), "difficulty", upper=1)
        latency = _finite_number(options.get("latency", self._default_latency), "latency")
        options["difficulty"] = d
        if "latency" in options:
            options["latency"] = latency
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("seed must be a nonnegative integer or None")
            seed = int(seed)
        if "engine_seed" in options:
            explicit_seed = options["engine_seed"]
            if isinstance(explicit_seed, (bool, np.bool_)) or not isinstance(explicit_seed, Integral) or not 0 <= explicit_seed < 2 ** 32:
                raise ValueError("engine_seed must be an integer in [0, 2**32)")
            options["engine_seed"] = int(explicit_seed)
        self._validate_options(options)
        # Invalid reset conditions must not consume RNG or alter a live episode.
        super().reset(seed=seed)
        self.engine_seed = options.get("engine_seed", None)
        if self.engine_seed is None:
            self.engine_seed = int(self.np_random.integers(2 ** 31 - 1))
        self.latency = latency
        self.d, self.steps, self.format_errors, self.done = d, 0, 0, False
        try:
            self.snap = self._engine_reset(self.engine_seed, d, options)
        except Exception:
            self.snap, self.done = None, True
            raise
        self._prev = self._progress(self.snap)
        info = self._info(True, 0)
        info["manual"] = self.manual()
        obs = self._observe()
        self._t_return = time.perf_counter()
        return obs, info

    def step(self, action, latency=None):
        """latency (s, virtual_rt only) overrides env.latency for this step: the agent's measured think time."""
        self._ensure_reset()
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        latency = self.latency if latency is None else _finite_number(latency, "latency")
        if self.time_mode == "virtual_rt":
            # Bound work at the episode deadline, including huge but finite latency.
            remaining = max(0, math.ceil(self.max_seconds / km.PACKET_SECONDS))
            lag = remaining if latency >= self.max_seconds else math.ceil(latency / km.PACKET_SECONDS - 1e-9)
        elif self.time_mode == "wallclock_rt":
            lag = max(0, int((time.perf_counter() - self._t_return) / km.PACKET_SECONDS))
        else:
            lag = 0
        if self.action_mode == "native":
            packet, ok = action, True
        else:
            packet = km.to_packet(action)
            ok = km.parse_packet(packet) is not None
            if not ok:
                self.format_errors += 1
        self.snap = self._engine_packet(packet, lag, ok)
        self.steps += 1
        out = self.snap["outcome"]
        terminated, truncated = out in ("win", "loss"), out == "timeout"
        prog, reward = self._progress(self.snap), 0.0
        if self.reward_mode == "dense":
            reward += (prog[0] - self._prev[0]) - (prog[1] - self._prev[1])
        self._prev = prog
        info = self._info(ok, lag)
        if terminated or truncated:
            self.done = True
            info["verifier"] = self.verifier()
            reward += info["verifier"]["reward"]
        obs = self._observe()
        self._t_return = time.perf_counter()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        self._ensure_reset()
        img = self._render()
        if not isinstance(img, np.ndarray) or img.shape != (RES[1], RES[0], 3) or img.dtype != np.uint8:
            raise RuntimeError(f"render must be RGB uint8 with shape {(RES[1], RES[0], 3)}")
        return img

    def verifier(self) -> dict:
        self._ensure_reset()
        s = self.snap
        frac = 1 - max(s["boss_hp"], 0) / s["boss_max_hp"]
        win = s["outcome"] == "win"
        return {"reward": 1.0 if win else round(-1.0 + frac, 4), "win": win, "outcome": s["outcome"],
                "boss_damage": round(frac, 4), "seconds": round(s["t"], 4), "d": self.d,
                "player_hp": s["player_hp"]}

    def manual(self) -> dict:
        """Prompt context (proposal 4.4): packet protocol, this body's key meanings, this boss's skill table."""
        self._ensure_open()
        m = self._manual()
        protocol = PROTOCOL
        if getattr(self, 'fps', 60) != 60:
            protocol = PROTOCOL.replace('(2 ticks at 60 Hz)',
                        f'(input sampled on native {self.fps} Hz ticks; actual executed time is reported)')
        return {"protocol": protocol, "controls": list(m["controls"]), "skills": list(m.get("skills", []))}

    def state_hash(self) -> str:
        """sha1 of the symbolic state, for determinism checks across runs and transports."""
        self._ensure_reset()
        return hashlib.sha1(json.dumps(self.snap["state"], sort_keys=True, default=_plain).encode()).hexdigest()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("environment is closed; create a new environment")

    def _ensure_reset(self):
        self._ensure_open()
        if self.snap is None:
            raise RuntimeError("environment has not been reset; call reset()")

    def _validate_options(self, options):
        try:
            json.dumps(options, allow_nan=False, default=_plain)
        except (TypeError, ValueError) as ex:
            raise ValueError("reset options must be finite JSON data") from ex

    def close(self):
        self._closed, self.done = True, True

    # -- shared helpers ----------------------------------------------------------
    def _progress(self, s):
        return (1 - max(s["boss_hp"], 0) / s["boss_max_hp"], 1 - max(s["player_hp"], 0) / s["player_max_hp"])

    def _info(self, ok, lag):
        s = self.snap
        info = {"t": round(s["t"], 4), "state": s["state"], "events": event_names(s["events"]),
                "outcome": s["outcome"], "player_hp": s["player_hp"], "player_max_hp": s["player_max_hp"],
                "boss_hp": s["boss_hp"], "boss_max_hp": s["boss_max_hp"], "format_ok": ok,
                "format_errors": self.format_errors, "steps": self.steps, "frames": s["frames"],
                "lag_packets": lag, "difficulty": self.d, "engine_seed": self.engine_seed}
        band = self._difficulty_band(self.d)
        if band is not None:
            info["difficulty_band"] = band
        return info

    def _observe(self):
        if self.obs_mode == "rgb":
            return self.render()
        if self.obs_mode == "vector":
            return self._vector()
        return self.snap["state"]

    # -- engine adapter --------------------------------------------------------
    def _declared_tiers(self):
        """The game's own {easy, normal, hard} -> difficulty mapping, or None to use tiers.DEFAULT_TIERS."""
        return None

    def _difficulty_band(self, d):
        """The game's own name for the segment `d` falls in, or None if it declares no segmentation.

        2026-11-03, requested by the Cuphead session. Several games map the continuous difficulty onto the
        original's own modes (Cuphead Simple/Regular/Expert at 0.25 and 0.75, DOOM's skill levels, Terraria's
        Classic/Expert gate at 0.75). Behaviour therefore steps at those boundaries: Cuphead measured the
        scripted controller's median duration jumping +8%..+24% across d=0.25, and hits taken rising (Cagney
        0.13 -> 1.13) across d=0.75. Anyone sweeping d to find d* sees a discontinuous curve, and until now
        had to re-derive the boundaries from the game's own source to know why.

        This leaks nothing: the band is a pure function of `difficulty`, which the public info already
        carries. It is a label on a number the policy can already see — not live state.
        """
        return None

    def _engine_reset(self, seed, d, options=None):
        raise NotImplementedError

    def _engine_packet(self, packet, lag, ok):
        raise NotImplementedError

    def _render(self):
        raise NotImplementedError

    def _vector(self):
        raise NotImplementedError

    def _manual(self):
        raise NotImplementedError


class PyCaseEnv(UnifiedEnv):
    """Adapter for the existing Python Case engine: the Case ticks at its own FPS (60 Hz -> 12 ticks per
    packet, 2 per sub-step) and the body's keys() maps each tick's held keys onto its native action."""

    def __init__(self, case_id, factory, case_kw=None, **kw):
        super().__init__(case_id, **kw)
        self.inner = factory(difficulty=self.difficulty, **(case_kw or {}))   # BossBenchEnv: owns case + renderer
        self.case = self.inner.case
        self.fps = self.inner.metadata["render_fps"]
        self.ticks = round(self.fps * km.PACKET_SECONDS)
        self.max_ticks = self.inner.max_ticks
        self.max_seconds = self.max_ticks / self.fps
        if self.action_mode == "native":
            self.action_space = self.case.body.action_space
        self.observation_space = self._obs_space(self.case.observation_space)

    @property
    def engine_spec(self):
        return EngineSpec('minecraft-python', '3d', 60, self.max_seconds, 'moderngl', 'perspective',
                          resolution=tuple(self.case.size))

    def render_status(self):
        error = getattr(self.case, 'render_error', None)
        ready = getattr(self.case, '_gl', None) is not None and error is None
        return RenderStatus('moderngl', 'perspective' if ready else 'raster', ready, error)

    def _validate_options(self, options):
        super()._validate_options(options)
        # Minecraft loadout/camera are public experiment conditions, passed
        # through the same Gym reset contract as the web games' options.
        # Validate before UnifiedEnv changes the live episode or RNG state.
        if self.case_id.startswith("3d/minecraft_"):
            for key, choices in (("weapon", ("sword", "bow")),
                                 ("camera", ("first", "third"))):
                if key in options and options[key] not in choices:
                    raise ValueError(f"Minecraft {key} must be one of {choices}")

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, self._visual_info(info)

    def step(self, action, latency=None):
        obs, reward, terminated, truncated, info = super().step(action, latency)
        return obs, reward, terminated, truncated, self._visual_info(info)

    def _visual_info(self, info):
        # UnifiedEnv builds info before rendering. Inspect the renderer only
        # after _observe returned so a fallback cannot masquerade as 3D RGB.
        if self.case_id.startswith("3d/minecraft_"):
            info = dict(info)
            info.update(camera=self.case.camera,
                        weapon=getattr(self.case.body, "weapon", "sword"))
            if self.obs_mode == "rgb":
                error = getattr(self.case, "render_error", None)
                renderer = getattr(self.case, "_gl", None)
                info["render_error"] = error
                info["renderer"] = "moderngl" if renderer is not None and error is None else "topdown"
                if info["renderer"] == "moderngl":
                    info["render_backend"] = renderer.ctx.info.get("GL_RENDERER")
        return info

    def _snapshot(self, frames):
        c = self.case
        out = c.outcome() or ("timeout" if c.t >= self.max_ticks else None)
        return {"state": c.state(), "events": list(c.events), "outcome": out, "t": c.t / self.fps,
                "player_hp": round(max(float(c.body.hp), 0.0), 4), "player_max_hp": float(c.body.max_hp),
                "boss_hp": round(max(float(c.boss.hp), 0.0), 4), "boss_max_hp": float(c.boss.max_hp), "frames": frames}

    def _engine_reset(self, seed, d, options=None):
        if self.case_id.startswith("3d/minecraft_"):
            self.case.reset(seed, d, options=dict(options or {}))
        else:
            self.case.reset(seed, d)
        self.inner._has_reset, self.inner._done = True, False
        self._held = frozenset()
        return self._snapshot(0)

    def _tick_all(self, actions):
        c, n = self.case, 0
        for a in actions:
            if c.outcome() or c.t >= self.max_ticks:
                break
            c.tick(a() if callable(a) else a)       # keys() runs at tick time: it may read the current state
            n += 1
        return n

    def _key_ticks(self, packet):
        sched, self._held = km.schedule(packet, self.ticks, self._held)
        body, c = self.case.body, self.case
        return [lambda h=h, p=p, m=m: body.keys(c, h, p, m) for h, p, m in sched]

    def _engine_packet(self, packet, lag, ok):
        c, frames = self.case, 0
        c.events.clear()
        native = self.action_mode == "native"
        for _ in range(lag):                                     # the world did not wait
            if c.outcome() or c.t >= self.max_ticks:
                break
            frames += self._tick_all([c.body.noop()] * self.ticks if native else self._key_ticks(None))
        if c.outcome() or c.t >= self.max_ticks:
            return self._snapshot(frames)
        if native:
            a = c.body.parse(packet) if isinstance(packet, str) else packet
            frames += self._tick_all([a] * self.ticks)
        else:
            frames += self._tick_all(self._key_ticks(km.parse_packet(packet) if ok else None))
        return self._snapshot(frames)

    def _render(self):
        return self.inner.render()

    def _vector(self):
        return self.case.observe()

    def _manual(self):
        skills = [{"move": n, "cue": m.get("cue") or m.get("telegraph"), "punish": m.get("punish"),
                   "damage": m.get("damage")} for n, m in self.case.boss.MOVES.items()]
        return {"controls": self.case.body.manual, "skills": skills}

    def _declared_tiers(self):
        # The case's own module declares TIERS (e.g. mc4d.TIERS) next to its RULES constant.
        import sys
        from .tiers import validate_declared
        module = sys.modules.get(type(self.case).__module__)
        return validate_declared(getattr(module, "TIERS", None), where=self.case_id)

    def _difficulty_band(self, d):
        # Optional, same pattern as TIERS: the case's module may declare difficulty_band(d) -> str.
        import sys
        module = sys.modules.get(type(self.case).__module__)
        fn = getattr(module, "difficulty_band", None)
        return _clean_band(fn(d), self.case_id) if callable(fn) else None

    def close(self):
        if not self._closed:
            try:
                self.inner.close()
            finally:
                super().close()
