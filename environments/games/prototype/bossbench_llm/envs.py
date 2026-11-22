"""One game episode behind the LLM harness: reset, execute a parsed model action, trusted result, replay record.

Three routes, all on the project's public interfaces (nothing here edits or monkey-patches game code):
* LumineGame        Lumine text -> BossBench packet -> bossbench.make_policy_env (every registered case).
* CompactGame       semantic JSON -> compact action -> bossbench.make_compact_env (18 games).
* TerrariaSemantic  semantic JSON -> terraria-semantic@1 (web/games/terraria_remake/semantic_ppo_v1), 2 x 100 ms.

Only the public RGB frame goes to the model. Results (verifier, final hash) are read on the trusted side for the
log and are never put into the model's context.
"""
from __future__ import annotations

from pathlib import Path
import sys

import bossbench
from bossbench import keymouse as km, scripted
from bossbench.scope import POLICY_CASES

from . import lumine
from .semantic import TERRARIA, Vocabulary

ROOT = Path(bossbench.__file__).resolve().parents[2]
INVALID = "invalid model action"      # any non-packet text: the env runs a no-op and counts a format error


def game_of(case_id):
    if case_id not in POLICY_CASES:
        raise KeyError(f"unknown case {case_id!r}; known: {sorted(POLICY_CASES)}")
    return POLICY_CASES[case_id]["game"]


def _resolve_options(options, info):
    """Pin a tier to the concrete difficulty it produced, as bossbench.scripted.play does."""
    options = dict(options)
    if "tier" in options:
        options["resolved_from_tier"] = options.pop("tier")
        options["difficulty"] = float(info["difficulty"])
    return options


class _PacketGame:
    """Shared by the two routes that end in ordinary 200 ms BossBench packets (replayable by make_policy_env)."""
    fmt = None

    def __init__(self, case_id):
        self.case_id, self.game = case_id, game_of(case_id)
        self.env = None
        self.packets, self.terminated, self.truncated = [], False, False

    def reset(self, seed, options):
        options = dict(options or {})
        options.setdefault("engine_seed", int(seed))      # same world as the play page / scripted replays
        obs, info = self.env.reset(seed=int(seed), options=options)
        self.gym_seed, self.options = int(seed), _resolve_options(options, info)
        self.packets, self.terminated, self.truncated = [], False, False
        return obs, info

    def _run(self, action, packet):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.packets.append(packet)
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        return obs, terminated or truncated, info

    def noop(self):
        return self._step_packet(km.NOOP)

    def result(self):
        raw = self.env.unwrapped
        v = raw.verifier()
        return {"outcome": v["outcome"], "win": bool(v["win"]), "boss_damage": v["boss_damage"],
                "seconds": v["seconds"], "difficulty": v.get("d"), "player_hp": v.get("player_hp"),
                "format_errors": int(getattr(raw, "format_errors", 0))}

    def replay(self):
        raw = self.env.unwrapped
        v = raw.verifier()
        ticks = int(round(raw.snap["t"] * raw.fps))
        return {"format": scripted.REPLAY_FORMAT, "case_id": self.case_id, "game": self.game,
                "contract": "bossbench/policy-gym@2", "rules": scripted.rules(self.env), "native_hz": raw.fps,
                "gym_seed": self.gym_seed, "engine_seed": int(raw.engine_seed), "options": self.options,
                "controller": f"bossbench_llm ({self.fmt} actions, see episode.json)",
                "packets": list(self.packets), "steps": len(self.packets), "ticks": ticks,
                "seconds": round(ticks / raw.fps, 4), "outcome": v["outcome"], "terminated": self.terminated,
                "truncated": self.truncated, "verifier": v, "final_hash": scripted.state_hash(self.env),
                "source_sha256": scripted.source_files(self.env)}

    def provenance(self):
        return {"rules": scripted.rules(self.env), "route": scripted.route(self.env),
                "native_hz": self.env.unwrapped.fps, "max_seconds": getattr(self.env.unwrapped, "max_seconds", None)}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class LumineGame(_PacketGame):
    fmt = "lumine"

    def __init__(self, case_id, mouse_kind):
        super().__init__(case_id)
        self.mouse_kind = mouse_kind
        self.env = bossbench.make_policy_env(case_id)

    def _step_packet(self, packet):
        return self._run(packet, packet)

    def step(self, parsed):
        """parsed: lumine.Parsed. Format errors run a no-op that the env counts as a format error."""
        return self._step_packet(parsed.packet if parsed.ok else INVALID)


class CompactGame(_PacketGame):
    fmt = "semantic"

    def __init__(self, case_id):
        super().__init__(case_id)
        self.vocab = Vocabulary.for_game(self.game)
        self.env = bossbench.make_compact_env(case_id)
        self.spec = self.env.compact_spec

    def _step_packet(self, packet):
        # A packet given to the compact wrapper is not a compact action; run it on the inner policy env.
        obs, _, terminated, truncated, info = self.env.env.step(packet)
        self.packets.append(packet)
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        return obs, terminated or truncated, info

    def step(self, parsed):
        if not parsed.ok:
            return self._run(INVALID, INVALID)
        packet = self.spec.compose(self.spec.parse(parsed.action))
        return self._run(dict(parsed.action), packet)


class TerrariaSemantic:
    fmt = "semantic"

    def __init__(self, case_id):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from web.games.terraria_remake.semantic_ppo_v1 import make_semantic_env
        self.case_id, self.game = case_id, TERRARIA
        self.vocab = Vocabulary.for_game(TERRARIA)
        self.env = make_semantic_env(case_id)
        self.windows = round(0.2 / self.vocab.window_seconds)

    def reset(self, seed, options):
        options = dict(options or {})
        options.setdefault("engine_seed", int(seed))
        obs, info = self.env.reset(seed=int(seed), options=options)
        self.gym_seed, self.options = int(seed), dict(options)
        return obs, info

    def _windows(self, action):
        done, obs, info = False, None, None
        for _ in range(self.windows):
            obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if done:
                break
        return obs, done, info

    def noop(self):
        return self._windows(None)

    def step(self, parsed):
        return self._windows(dict(parsed.action) if parsed.ok else "INVALID")

    def result(self):
        d = self.env.diagnostic()
        v = d.get("verifier", {})
        return {"outcome": v.get("outcome"), "win": bool(v.get("win")), "boss_damage": v.get("boss_damage"),
                "seconds": v.get("seconds"), "difficulty": v.get("d"), "player_hp": v.get("player_hp"),
                "format_errors": None}

    def replay(self, path):
        return self.env.save_replay(path)

    def provenance(self):
        return {"rules": self.env.meta.get("rules"), "route": "terraria-semantic@1",
                "native_hz": 60, "max_seconds": self.env.meta.get("max_seconds")}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def open_game(case_id, fmt, mouse_kind="none"):
    if fmt == "lumine":
        return LumineGame(case_id, mouse_kind)
    if fmt == "semantic":
        return TerrariaSemantic(case_id) if game_of(case_id) == TERRARIA else CompactGame(case_id)
    raise ValueError("format must be 'lumine' or 'semantic'")
