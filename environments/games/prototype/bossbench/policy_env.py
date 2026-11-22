"""Public Gymnasium contract shared by visual models and RL, for 2D and 3D.

The underlying simulations and privileged diagnostics remain accessible through
``unwrapped`` for trusted collectors. This wrapper is an input boundary, not a
security sandbox; policies receive ``rl.rollout.Context``, never the env object.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from . import keymouse as km
from .core import RES
from .scope import POLICY_CASES

CONTRACT_VERSION = "bossbench/policy-gym@2"
PUBLIC_INFO_KEYS = frozenset({
    "case_id", "contract", "t", "steps", "frames", "lag_packets", "format_ok",
    "format_errors", "difficulty", "engine_seed", "outcome", "terminated",
    "truncated", "timing", "renderer",
    # 2026-11-03: the game's own name for the segment `difficulty` falls in, when it declares one.
    # Present only for such games, and safe to expose because it is a pure function of `difficulty`,
    # which this info already carries — a label on a number the policy can already see, not live state.
    # It exists so a d* sweep can explain the steps it measures at mode boundaries instead of
    # re-deriving them from each game's source.
    "difficulty_band",
})
_RAW_PUBLIC_KEYS = ("t", "steps", "frames", "lag_packets", "format_ok", "format_errors",
                    "difficulty", "engine_seed", "outcome")
_RAW_OPTIONAL_KEYS = ("difficulty_band",)


class PolicyEnv(gym.Wrapper):
    """RGB + legal key/mouse packets + sparse reward + public-only info.

    One step holds six key sets over 200 ms, using the declared native engine clock.
    Text and structured actions share the same parser and rejection semantics.
    Actual perspective 3D is mandatory: a fallback image raises an error.
    """

    def __init__(self, env):
        raw = env.unwrapped
        if raw.case_id not in POLICY_CASES:
            raise ValueError(f"case is not registered for the public policy interface: {raw.case_id!r}")
        if (raw.obs_mode, raw.action_mode, raw.reward_mode) != ("rgb", "keymouse", "sparse"):
            raise ValueError("public policy env requires rgb/keymouse/sparse")
        spec = raw.engine_spec
        if raw.fps != spec.native_hz:
            raise ValueError(f"engine clock must match declared {spec.native_hz} Hz; remove runtime FPS overrides")
        if spec.dimension != POLICY_CASES[raw.case_id]['dimension']:
            raise ValueError('engine dimension differs from the registered case')
        if tuple(spec.resolution) != RES:
            raise ValueError("public policy contract requires a renderer size of 854 x 480")
        if spec.dimension == '3d' and spec.render_kind != 'perspective':
            raise ValueError('3D policy observations require perspective rendering')
        super().__init__(env)
        self.metadata = {**env.metadata, "contract": CONTRACT_VERSION, "simulation_fps": raw.fps}
        self._closed = False
        self._elapsed_frames = 0

    def _open(self):
        if self._closed:
            raise RuntimeError("environment is closed; create a new environment")

    def _frame(self, image):
        if not isinstance(image, np.ndarray) or image.shape != (RES[1], RES[0], 3) or image.dtype != np.uint8:
            raise RuntimeError("policy observation must be RGB uint8 (480, 854, 3)")
        if POLICY_CASES[self.unwrapped.case_id]["dimension"] == "3d":
            status = self.unwrapped.render_status()
            if (not status.ready or status.kind != 'perspective' or status.error is not None
                    or status.renderer != self.unwrapped.engine_spec.renderer):
                raise RuntimeError(f"{self.unwrapped.case_id}: real 3D renderer unavailable: {status.error or 'no perspective renderer'}")
        # Neither reused engine buffers nor policy mutation may change history.
        return image.copy()

    def _info(self, info, terminated=False, truncated=False):
        raw = self.unwrapped
        frames, lag = int(info["frames"]), int(info["lag_packets"])
        lag_frames = int(raw.snap.get('lag_frames', min(frames, lag * round(raw.fps * km.PACKET_SECONDS))))
        out = {k: copy.deepcopy(info[k]) for k in _RAW_PUBLIC_KEYS}
        out.update({k: copy.deepcopy(info[k]) for k in _RAW_OPTIONAL_KEYS if k in info})
        out.update(case_id=raw.case_id, contract=CONTRACT_VERSION,
                   terminated=bool(terminated), truncated=bool(truncated),
                   renderer=raw.engine_spec.renderer,
                   timing={"fps": raw.fps, "tick_seconds": 1 / raw.fps,
                           "packet_seconds": km.PACKET_SECONDS, "substeps": km.SUBSTEPS,
                           "ticks_per_substep": raw.fps * km.PACKET_SECONDS / km.SUBSTEPS,
                           "simulation_seconds": frames / raw.fps,
                           "elapsed_seconds": self._elapsed_frames / raw.fps,
                           "lag_frames": lag_frames, "action_frames": frames - lag_frames})
        if "manual" in info:
            out["manual"] = self.manual()
        return out

    def manual(self):
        self._open()
        manual = self.unwrapped.manual()
        return {"protocol": str(manual["protocol"]), "controls": copy.deepcopy(manual["controls"])}

    def reset(self, *, seed=None, options=None):
        self._open()
        obs, info = self.env.reset(seed=seed, options=options)
        self._elapsed_frames = 0
        return self._frame(obs), self._info(info)

    def step(self, action, latency=None):
        self._open()
        obs, reward, terminated, truncated, info = self.env.step(action, latency=latency)
        self._elapsed_frames += int(info["frames"])
        return self._frame(obs), reward, terminated, truncated, self._info(info, terminated, truncated)

    def render(self):
        self._open()
        return self._frame(self.env.render())

    def close(self):
        if not self._closed:
            try:
                self.env.close()
            finally:
                self._closed = True


def make_policy_env(case_id, **kwargs):
    """Create the same public Gym env for model evaluation and RL collection.

    Pass engine-specific experiment conditions in reset(options=...), e.g.
    Minecraft weapon/camera or Terraria dawn_in_seconds; engine_seed imports a
    recording seed exactly. Raw symbolic/native/dense ablations use make().
    """
    if case_id not in POLICY_CASES:
        raise ValueError(f"case is not registered for the public policy interface: {case_id!r}; known: {sorted(POLICY_CASES)}")
    for key, value in (("obs_mode", "rgb"), ("action_mode", "keymouse"), ("reward_mode", "sparse")):
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"public policy contract requires {key}={value!r}")
        kwargs[key] = value
    kwargs.setdefault("time_mode", "paused")
    kwargs.setdefault("render_mode", "rgb_array")
    from . import make
    env = make(case_id, **kwargs)
    try:
        return PolicyEnv(env)
    except Exception:
        env.close()
        raise


def policy_gym_id(case_id):
    return "BossBenchPolicy/" + case_id.replace("/", "-") + "-v0"
