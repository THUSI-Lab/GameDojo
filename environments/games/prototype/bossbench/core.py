"""BossBench core.

A Case is the composition  Arena (environment) + Body (embodiment) + Boss.
Every component declares a difficulty TABLE; one scalar d in [0, 1] sets all knobs.
BossBenchEnv wraps a Case as a Gymnasium env with three time modes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
import os
import time

import gymnasium as gym
import numpy as np

# fixed simulation tick: 60 Hz = 2 ticks per 33 ms sub-step of the 200 ms packet, as the web engine
# (proposal 4.4). Rules are written per second (or per vanilla 20 Hz Minecraft tick, converted in mc4d), so the
# game plays the same at any rate >= 30; BOSSBENCH_FPS overrides it for equivalence checks only.
FPS = int(os.environ.get("BOSSBENCH_FPS", 60))
RES = (854, 480)  # every case and every renderer outputs 480p, 16:9 (W, H)


class Knobs:
    """d in [0,1] -> named parameters. A row is (easy, hard) or (easy, normal, hard)."""

    def __init__(self, table: dict, d: float):
        self._names = list(table)
        d = float(np.clip(d, 0.0, 1.0))
        for name, row in table.items():
            setattr(self, name, float(np.interp(d, np.linspace(0, 1, len(row)), row)))

    def as_dict(self) -> dict:
        return {n: round(getattr(self, n), 4) for n in self._names}


class Component:
    TABLE: dict = {}

    def reset(self, case: "Case") -> None:
        self.k = Knobs(self.TABLE, case.d)

    def update(self, case: "Case") -> None:
        pass


class Arena(Component):
    """Static geometry + transient entities (projectiles, hazards)."""


class Body(Component):
    """The agent's embodiment: action space, kinematics, health."""
    action_space: gym.Space
    max_hp = 1.0

    def noop(self):
        raise NotImplementedError

    def parse(self, text: str):
        """'up+fire' -> action; lets LLM agents act with words."""
        raise NotImplementedError

    def apply(self, case: "Case", action) -> None:
        raise NotImplementedError

    # key/mouse packets (keymouse.py): what each key means is up to the body (proposal 4.3)
    manual: tuple = ()

    def keys(self, case: "Case", held: frozenset, pressed: frozenset, mouse: tuple):
        """Keys held this tick (+ those newly pressed, + this tick's share of the mouse motion) -> native action."""
        raise NotImplementedError


class Boss(Component):
    """Opponent: a state machine over MOVES (Boss-Spec DSL subset)."""
    MOVES: dict = {}
    SKILLS: tuple = ()
    max_hp = 1.0


class Case:
    """Arena + Body + Boss, stepped at FPS. Subclasses add observe/state/render."""
    name = "case"
    max_seconds = 120.0
    obs_dim = 0

    def __init__(self, arena: Arena, body: Body, boss: Boss):
        self.arena, self.body, self.boss = arena, body, boss
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    def reset(self, seed: int, d: float) -> None:
        self.rng = np.random.default_rng(seed)
        self.d, self.t, self.events = float(d), 0, []
        for c in (self.arena, self.body, self.boss):
            c.reset(self)

    def tick(self, action) -> None:
        self.body.apply(self, action)
        self.boss.update(self)
        self.arena.update(self)
        self.interact()
        self.t += 1

    def interact(self) -> None:
        """Case-specific body/boss contact rules."""

    def outcome(self):
        if self.boss.hp <= 0:
            return "win"
        if self.body.hp <= 0:
            return "loss"
        return None

    def describe(self) -> dict:
        """Difficulty knobs + boss spec: what an LLM generator/verifier sees."""
        return {"case": self.name, "d": self.d,
                "knobs": {type(c).__name__: c.k.as_dict() for c in (self.arena, self.body, self.boss)},
                "moves": self.boss.MOVES, "skills_tested": self.boss.SKILLS}

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError


class BossBenchEnv(gym.Env):
    """Gymnasium wrapper.

    time_mode:
      paused   - the world waits for the agent (turn-based upper bound)
      virtual  - before each action the world runs ceil(latency*FPS) no-op ticks
                 (set env.latency per step to the agent's measured think time)
      realtime - same, but the lag is measured on the wall clock between steps
    reward: terminal  win=+1, loss/timeout=-1+boss_damage_fraction  (+ optional dense shaping)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, case: Case, difficulty=0.5, time_mode="paused", latency=0.0,
                 frame_skip=1, reward_mode="sparse", render_mode=None, max_seconds=None):
        assert time_mode in ("paused", "virtual", "realtime")
        self.case, self.difficulty, self.time_mode, self.latency = case, difficulty, time_mode, latency
        self.frame_skip, self.reward_mode, self.render_mode = frame_skip, reward_mode, render_mode
        seconds = case.max_seconds if max_seconds is None else max_seconds
        if (isinstance(seconds, (bool, np.bool_)) or not isinstance(seconds, Real) or
                not math.isfinite(seconds) or not math.isfinite(seconds * FPS) or seconds < 1 / FPS):
            raise ValueError("max_seconds must be finite and allow at least one simulation tick")
        self.max_ticks = int(seconds * FPS)
        self.action_space = case.body.action_space
        self.observation_space = case.observation_space
        self.renderer = None          # None -> case.render(); or a callable(env) -> frame (e.g. skin.GodotRenderer)
        self.episode = 0
        self._closed = False
        self._has_reset, self._done = False, True

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._ensure_open()
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping")
        d = (options or {}).get("difficulty", self.difficulty)
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, Real) or not math.isfinite(d) or not 0 <= d <= 1:
            raise ValueError("difficulty must be a finite number in [0, 1]")
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
                raise ValueError("seed must be a nonnegative integer or None")
            seed = int(seed)
        super().reset(seed=seed)
        self.case.reset(int(self.np_random.integers(2**31 - 1)), d)
        self._has_reset, self._done = True, False
        self.episode += 1
        self._prev = self._progress()
        self._lag_ticks = 0
        self._t_return = time.perf_counter()
        return self.case.observe(), self._info()

    def step(self, action):
        self._ensure_reset()
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        c = self.case
        if isinstance(action, str):
            action = c.body.parse(action)
        c.events.clear()
        self._lag_ticks = self._lag()
        for a in [c.body.noop()] * self._lag_ticks + [action] * self.frame_skip:
            c.tick(a)
            if c.outcome() or c.t >= self.max_ticks:
                break
        terminated = c.outcome() is not None
        truncated = not terminated and c.t >= self.max_ticks
        prog, reward = self._progress(), 0.0
        if self.reward_mode == "dense":
            reward += (prog[0] - self._prev[0]) - (prog[1] - self._prev[1])
        self._prev = prog
        if terminated or truncated:
            reward += self.verifier()["reward"]
            self._done = True
        obs, info = c.observe(), self._info()
        self._t_return = time.perf_counter()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        self._ensure_reset()
        img = self.renderer(self) if self.renderer else self.case.render()
        assert img.shape == (RES[1], RES[0], 3), f"render must be 480p {RES}, got {img.shape}"
        return img

    def close(self):
        if not self._closed:
            try:
                close = getattr(self.case, "close", None)
                if close is not None:
                    close()
            finally:
                self._closed = True

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("environment is closed; create a new environment")

    def _ensure_reset(self):
        self._ensure_open()
        if not self._has_reset:
            raise RuntimeError("environment has not been reset; call reset()")

    # -- BossBench extras ----------------------------------------------------
    def verifier(self) -> dict:
        self._ensure_reset()
        boss_frac, _ = self._progress()
        win = self.case.outcome() == "win"
        return {"reward": 1.0 if win else -1.0 + boss_frac, "win": win,
                "boss_damage": round(boss_frac, 4), "seconds": self.case.t / FPS, "d": self.case.d}

    def _progress(self):
        b, p = self.case.boss, self.case.body
        return (1 - max(b.hp, 0) / b.max_hp, 1 - max(p.hp, 0) / p.max_hp)

    def _lag(self) -> int:
        if self.time_mode == "virtual":
            return math.ceil(self.latency * FPS - 1e-9)
        if self.time_mode == "realtime":
            return int((time.perf_counter() - self._t_return) * FPS)
        return 0

    def _info(self) -> dict:
        return {"t": self.case.t / FPS, "state": self.case.state(), "events": list(self.case.events),
                "lag_ticks": self._lag_ticks, "difficulty": self.case.d}
