"""Re-run recorded episodes and check they reproduce exactly (outcome, ticks, engine final-state hash).

  python -m bossbench_llm.verify runs/<run>/<episode_dir> [more dirs or replay files ...]
  python -m bossbench_llm.verify replays/*.replay.json

Accepts episode folders written by bossbench_llm.run (replay.json or semantic_replay.json inside) and plain
bossbench/policy-replay@1 files. Also reports source files whose bytes differ from
the ones recorded (a different game version), which is the usual reason a replay stops reproducing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bossbench
from bossbench import scripted

from .envs import ROOT


def _replay(env, data):
    """scripted.replay, but a recording cut before the game ended (collector_cut) is compared where it stopped."""
    if data.get("outcome") is not None:
        return scripted.replay(env, data)
    raw, problems = env.unwrapped, []
    if data.get("rules") != scripted.rules(env):
        problems.append(f"rules mismatch: recording {data.get('rules')!r}, engine {scripted.rules(env)!r}")
    env.reset(seed=int(data["gym_seed"]), options=dict(data["options"]))
    for index, packet in enumerate(data["packets"]):
        _, _, terminated, truncated, _ = env.step(packet)
        if (terminated or truncated) and index != len(data["packets"]) - 1:
            problems.append(f"game ended at packet {index}, recording continues")
            break
    final, ticks = scripted.state_hash(env), int(round(raw.snap["t"] * raw.fps))
    if final != data.get("final_hash"):
        problems.append(f"final hash mismatch: {final} vs {data.get('final_hash')}")
    if ticks != data.get("ticks"):
        problems.append(f"tick mismatch: {ticks} vs {data.get('ticks')}")
    return raw.verifier()["outcome"], final, ticks, problems


def verify_policy_replay(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    env = bossbench.make_policy_env(data["case_id"])
    try:
        current = scripted.source_files(env) if data.get("source_sha256") else {}
        drift = sorted(k for k, v in (data.get("source_sha256") or {}).items() if current.get(k) != v)
        outcome, final, ticks, problems = _replay(env, data)
    finally:
        env.close()
    return {"file": str(path), "case_id": data["case_id"], "outcome": outcome, "ticks": ticks, "final_hash": final,
            "ok": not problems, "problems": problems, "source_drift": drift}


def verify_semantic_replay(path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from web.games.terraria_remake.semantic_ppo_v1.env import verify_replay
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        out = verify_replay(path)
        return {"file": str(path), "case_id": data.get("case_id"), "ok": True, "detail": out}
    except Exception as e:  # noqa: BLE001
        return {"file": str(path), "case_id": data.get("case_id"), "ok": False, "problems": [f"{type(e).__name__}: {e}"]}


def verify(target):
    target = Path(target)
    if target.is_dir():
        if (target / "replay.json").is_file():
            return verify_policy_replay(target / "replay.json")
        if (target / "semantic_replay.json").is_file():
            return verify_semantic_replay(target / "semantic_replay.json")
        return {"file": str(target), "ok": False, "problems": ["no replay.json / semantic_replay.json in this folder"]}
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("format") == scripted.REPLAY_FORMAT:
        return verify_policy_replay(target)
    if str(data.get("schema", "")).startswith("terraria/semantic-replay"):
        return verify_semantic_replay(target)
    return {"file": str(target), "ok": False, "problems": ["unknown replay format"]}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bossbench_llm.verify", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+")
    p.add_argument("--json", help="write the results to this file")
    args = p.parse_args(argv)
    results = []
    for t in args.targets:
        r = verify(t)
        results.append(r)
        state = "OK  " if r["ok"] else "FAIL"
        print(f"{state} {r.get('case_id') or ''} {r['file']}" + ("" if r["ok"] else f"  {r.get('problems')}"))
        if r.get("source_drift"):
            print(f"     source files changed since recording: {r['source_drift']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = sum(not r["ok"] for r in results)
    print(f"{len(results) - bad}/{len(results)} reproduced exactly")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
