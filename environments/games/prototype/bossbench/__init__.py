"""BossBench: every case behind one Gymnasium interface (unified.py), whatever engine runs it.

    import bossbench
    bossbench.ENVS                                      # {"2d/needle_warden": "web", "3d/minecraft_wither": "py", ...}
    env = bossbench.make("2d/needle_warden", obs_mode="symbolic", time_mode="virtual_rt", latency=0.3)
    obs, info = env.reset(seed=0, options={"difficulty": 0.5})
    obs, r, term, trunc, info = env.step("<|action_start|> 0 0 0 ; D ; D ; D,Space ; D ; J ; J <|action_end|>")
    env.verifier()                                      # {reward in [-1,1], win, outcome, boss_damage, ...}
    gymnasium.make("BossBench/2d-needle_warden-v0")     # same env through the gymnasium registry

Engines: "web" = web/engine + one folder per game under web/games/ (JavaScript, 60 Hz; HK-style 2D cases),
"py" = this package (Minecraft). WEB_GAME maps each web case id to its game folder.
Those engines run at 60 Hz. Independent backends declare their own native clocks.

The retired Python tank demo and its aliases are no longer registered.
"""
import os
import re

from . import mc4d
from .core import FPS, RES, Arena, Body, Boss, BossBenchEnv, Case, Knobs
from .keymouse import NOOP, format_packet, parse_packet
from .unified import PyCaseEnv, UnifiedEnv

PY_CASES = {f"3d/minecraft_{m}": (mc4d.make, {"mob": m}) for m in mc4d.MOBS}


def _web_cases():
    """{case id: game folder} for every BB.CASES['...'] = registration under web/games/<game>/, found without
    starting Node. Folders starting with "_" (the template) or without an index.html, and tests/ shots/, are skipped."""
    games = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web", "games")
    out = {}
    for game in sorted(os.listdir(games)) if os.path.isdir(games) else []:
        if game.startswith(("_", ".")) or not os.path.isfile(os.path.join(games, game, "index.html")):
            continue
        for d, sub, files in os.walk(os.path.join(games, game)):
            sub[:] = [s for s in sub if s not in ("tests", "shots")]
            for f in files:
                if f.endswith(".js"):
                    src = open(os.path.join(d, f), encoding="utf-8").read()
                    for i in re.findall(r"BB\.CASES\['([^']+)'\]\s*=", src):
                        if out.setdefault(i, game) != game:
                            raise RuntimeError(f"web case {i!r} registered by two games: {out[i]}, {game}")
    return out


WEB_GAME = _web_cases()
WEB_CASES = sorted(WEB_GAME)
ENVS = {**{i: "web" for i in WEB_CASES}, **{i: "py" for i in PY_CASES}}
ENGINE_BACKENDS = {}


def register_backend(case_id, factory):
    """Explicitly register an independent GameBackend factory; this does not enroll it in training.

    Call in the integration module on each process, including spawned workers.
    Each factory call must create an independent backend and own its resources.
    """
    if not isinstance(case_id, str) or not re.fullmatch(r'[a-z0-9_]+/[a-z0-9_]+', case_id):
        raise ValueError('case_id must be namespace/name using lowercase letters, digits or underscores')
    if case_id in ENVS:
        raise ValueError(f'case already registered: {case_id}')
    if case_id in ('2d/tank_battle', 'tank_battle_1v1'):
        raise ValueError('retired case cannot be registered again')
    if not callable(factory):
        raise TypeError('backend factory must be callable')
    import gymnasium as gym
    gym.register(gym_id(case_id), entry_point=make, kwargs={'name': case_id})
    ENGINE_BACKENDS[case_id] = factory
    ENVS[case_id] = 'native'


def make(name, **kw):
    """Unified env for any id in ENVS; kw: obs_mode, time_mode, latency, difficulty, reward_mode, action_mode,
    (web) transport, (py) case_kw for the case constructor (e.g. {"camera": "first"})."""
    if name in ENGINE_BACKENDS:
        from .engine_env import EngineEnv
        backend_options = dict(kw.pop('backend_options', {}) or {})
        backend = ENGINE_BACKENDS[name](**backend_options)
        return EngineEnv(name, backend, **kw)
    if name in PY_CASES:
        factory, fixed = PY_CASES[name]
        extra = dict(kw.pop("case_kw", {}) or {})
        if any(key in extra and extra[key] != value for key, value in fixed.items()):
            raise ValueError(f"case_kw cannot override the registered case identity: {fixed}")
        case_kw = {**extra, **fixed}
        return PyCaseEnv(name, lambda **k: factory(**case_kw, **k), **kw)
    if name in WEB_CASES:
        from .webenv import WebCaseEnv
        return WebCaseEnv(name, **kw)
    raise KeyError(f"unknown case {name!r}; known: {sorted(ENVS)}")


def gym_id(name):
    return "BossBench/" + name.replace("/", "-") + "-v0"


from .policy_env import PolicyEnv, make_policy_env, policy_gym_id


try:  # gymnasium.make("BossBench/2d-needle_warden-v0")
    import gymnasium as gym
    for _n in ENVS:
        gym.register(gym_id(_n), entry_point=make, kwargs={"name": _n})
    from .scope import FORMAL_CASES, PUBLIC_WEB_CASES
    for _n in (*FORMAL_CASES, *PUBLIC_WEB_CASES):
        gym.register(policy_gym_id(_n), entry_point=make_policy_env, kwargs={"case_id": _n})
except Exception:
    pass

from .native_games import NATIVE_CASES, backend_factory
for _n in NATIVE_CASES:
    register_backend(_n, backend_factory(_n))
    gym.register(policy_gym_id(_n), entry_point=make_policy_env, kwargs={'case_id': _n})

# 2026-11-10: compact discrete action spaces (bossbench/compact-actions@1, compact.py). Registered for every public
# case whose game ships web/games/<game>/compact_actions.json; the raw key/mouse contract above is unchanged.
from . import compact
from .compact import make_compact_env, compact_gym_id
from .step_reward import StepDamageReward, make_training_env
try:
    from .scope import POLICY_CASES as _PC
    for _n, _v in _PC.items():
        if compact.declaration_path(_v['game']).is_file():
            gym.register(compact_gym_id(_n), entry_point=make_compact_env, kwargs={'case_id': _n})
except Exception:
    pass
