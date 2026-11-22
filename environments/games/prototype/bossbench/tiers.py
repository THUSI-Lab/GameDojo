"""Three public difficulty tiers shared by every BossBench game.

The standard interface takes one of three named tiers. Use continuous difficulty for finer control.

    env.reset(seed=1, options={"tier": "easy"})      # public: one of three tiers
    env.reset(seed=1, options={"difficulty": 0.37})  # still allowed: direct continuous control

Each game declares its own tier -> difficulty mapping next to its rule version, so the simulation, the play
page and this public interface all read one source of truth:

    BB.Case games     BB.GAME.tiers = {easy: 0.15, normal: 0.5, hard: 0.85}     (game.js)
    native JS games   Module.tiers  = {easy: ..., normal: ..., hard: ...}         (game.js)
    Minecraft         mc4d.TIERS    = {"easy": ..., "normal": ..., "hard": ...}

A game that declares nothing uses DEFAULT_TIERS. Passing both `tier` and `difficulty` is rejected, so a
recording can never be ambiguous about which one produced it.

"""
from __future__ import annotations

import math
from numbers import Real
from types import MappingProxyType

TIER_NAMES = ("easy", "normal", "hard")

DEFAULT_TIERS = MappingProxyType({"easy": 0.2, "normal": 0.5, "hard": 0.8})
TIERS = DEFAULT_TIERS          # backwards-compatible name

# Reference-controller completion windows in seconds.
TARGET_SECONDS = MappingProxyType({
    "easy": (20.0, 30.0),
    "normal": (45.0, 90.0),
    "hard": (150.0, 240.0),
})


def validate_declared(declared, where="game"):
    """Return a clean {tier: difficulty} from a game's declaration, or raise with a precise reason.

    A declaration must name exactly easy/normal/hard, each a finite number in [0, 1], strictly increasing
    from easy to hard. A broken declaration fails when the env is created, inside the game that wrote it.
    """
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError(f"{where}: tiers must be an object with keys {TIER_NAMES}")
    if set(declared) != set(TIER_NAMES):
        raise ValueError(f"{where}: tiers must declare exactly {TIER_NAMES}, got {sorted(declared)}")
    clean = {}
    for name in TIER_NAMES:
        value = declared[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{where}: tier {name!r} must be a finite number in [0, 1], got {value!r}")
        clean[name] = float(value)
    if not clean["easy"] < clean["normal"] < clean["hard"]:
        raise ValueError(f"{where}: tiers must increase strictly easy < normal < hard, got {clean}")
    return clean


def difficulty_for(tier, declared=None):
    """Continuous difficulty for a public tier, using the game's own declaration when it has one."""
    if tier not in TIER_NAMES:
        raise ValueError(f"tier must be one of {TIER_NAMES}, got {tier!r}")
    table = declared if declared else DEFAULT_TIERS
    return float(table[tier])


def resolve_options(options, declared=None):
    """Replace a public `tier` with the concrete `difficulty` the engines already understand.

    Returns a new dict; the original is not modified. `tier` and `difficulty` are mutually exclusive so a
    replay's options always say exactly one thing about how hard the fight was.
    """
    if not options or "tier" not in options:
        return dict(options or {})
    resolved = dict(options)
    tier = resolved.pop("tier")
    if "difficulty" in resolved:
        raise ValueError("pass either tier or difficulty, not both: a recording must be unambiguous")
    resolved["difficulty"] = difficulty_for(tier, declared)
    resolved["resolved_from_tier"] = tier      # provenance: which tier the caller asked for
    return resolved


def tier_of(difficulty, declared=None, tolerance=1e-9):
    """The tier that maps to this difficulty, or None when it is a custom continuous value."""
    for tier in TIER_NAMES:
        if abs(difficulty_for(tier, declared) - float(difficulty)) <= tolerance:
            return tier
    return None


def within_target(tier, seconds):
    """True when a scripted run's duration falls inside the reference-controller window for that tier."""
    low, high = TARGET_SECONDS[tier]
    return low <= float(seconds) <= high
