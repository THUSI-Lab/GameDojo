"""Formal BossBench scope, separate from the backwards-compatible registry.

Never discover training/evaluation scope by blindly iterating bossbench.ENVS. This manifest is
explicit so a newly added game cannot silently enter an existing experiment.
"""
from __future__ import annotations

from types import MappingProxyType


FORMAL_GAMES = MappingProxyType({
    "hollowknight_remake": "2d",
    "kirby_remake": "2d",
    "terraria_remake": "2d",
    "minecraft": "3d",
})

FORMAL_CASES = MappingProxyType({
    **{f"2d/{boss}": {"game": "hollowknight_remake", "dimension": "2d", "engine": "web"} for boss in (
        "needle_warden", "iron_bulwark", "hollow_vessel", "sickle_sisters", "pale_vessel")},
    **{f"kirby/{boss}": {"game": "kirby_remake", "dimension": "2d", "engine": "web"} for boss in (
        "whispy_woods", "kracko", "king_dedede")},
    **{f"terraria/{boss}": {"game": "terraria_remake", "dimension": "2d", "engine": "web"} for boss in (
        "king_slime", "eye_of_cthulhu", "skeletron", "eater_of_worlds")},
    **{f"3d/minecraft_{boss}": {"game": "minecraft", "dimension": "3d", "engine": "py"} for boss in (
        "iron_golem", "zombie", "ender_dragon", "wither")},
})

RETIRED_CASES = ("2d/tank_battle", "tank_battle_1v1")
# Public cases are discovered from registrations. Frozen evaluation cases are listed above.
from . import WEB_GAME
PUBLIC_WEB_CASES = MappingProxyType({
    case_id: {"game": game, "dimension": "2d", "engine": "web"}
    for case_id, game in WEB_GAME.items() if case_id not in FORMAL_CASES
})
# Python-route scenes (mc4d.MOBS) are discovered the same way, so Minecraft can add bosses without editing
# this file. Their ids are "3d/<game>_<boss>"; the game name is read from the id rather than assumed, so a
# second Python game would be routed correctly instead of silently labelled Minecraft.
from . import PY_CASES
PUBLIC_PY_CASES = MappingProxyType({
    case_id: {"game": case_id.split("/", 1)[1].split("_", 1)[0], "dimension": "3d", "engine": "py"}
    for case_id in PY_CASES if case_id not in FORMAL_CASES
})
from .native_games import NATIVE_CASES
POLICY_CASES = MappingProxyType({**FORMAL_CASES, **PUBLIC_WEB_CASES, **PUBLIC_PY_CASES, **{
    key: {'game': value['game'], 'dimension': value['dimension'], 'engine': 'native'}
    for key, value in NATIVE_CASES.items()
}})
# Retired cases retain their IDs for replay compatibility and are omitted from the benchmark registry.
BENCH_RETIRED = frozenset({
    "3d/minecraft_zombie",      # Minecraft had 7; its tier calibration is the separate 3d/minecraft_zombie_leader
})
BENCH_SCENES_PER_GAME = 6
BENCH_CASES = MappingProxyType({k: v for k, v in POLICY_CASES.items() if k not in BENCH_RETIRED})
# Every game, including Minecraft (Python), with its dimension; used by the media/scripted harness.
GAMES = MappingProxyType({spec["game"]: spec["dimension"] for spec in POLICY_CASES.values()})
FIRST_CASE = "kirby/whispy_woods"
COLD_START_CASE = "2d/needle_warden"
SCOPE_VERSION = "bossbench/formal-scope@1"


def case_ids(*, dimension: str | None = None, game: str | None = None) -> tuple[str, ...]:
    """Return stable, sorted formal IDs, optionally restricted by game or dimension."""
    if dimension is not None and dimension not in ("2d", "3d"):
        raise ValueError("dimension must be '2d' or '3d'")
    if game is not None and game not in FORMAL_GAMES:
        raise ValueError(f"unknown formal game: {game!r}")
    return tuple(sorted(i for i, spec in FORMAL_CASES.items()
                        if (dimension is None or spec["dimension"] == dimension)
                        and (game is None or spec["game"] == game)))


def validate_registry() -> None:
    """Reject missing/misrouted formal cases without changing the legacy registry."""
    from . import ENVS, WEB_GAME

    for case_id, spec in {**FORMAL_CASES, **PUBLIC_WEB_CASES, **PUBLIC_PY_CASES}.items():
        if ENVS.get(case_id) != spec["engine"]:
            raise RuntimeError(f"formal case {case_id!r} has missing or incorrect engine registration")
        if spec["engine"] == "web" and WEB_GAME.get(case_id) != spec["game"]:
            raise RuntimeError(f"formal case {case_id!r} has incorrect game registration")
