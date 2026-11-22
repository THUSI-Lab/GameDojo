"""Run LLM (or baseline) episodes on BossBench games with a Lumine-style loop.

Each 200 ms of game time: one frame -> the model (same chat session, sliding window) -> one action (Lumine text or
semantic JSON) -> executed in the game. Default time mode is `paused`: the game waits for the model, so results do
not depend on inference speed and every run is replayable. `realtime`: every full 200 ms the model takes inserts
one empty 200 ms window before its action (latency under 200 ms costs nothing).

Examples (run from the package root with PYTHONPATH=games/prototype):
  python -m bossbench_llm.run --case terraria/king_slime --tier easy --seeds 1-3 --format lumine --lang en \
      --policy chat --base-url http://127.0.0.1:8000/v1 --model my-model
  python -m bossbench_llm.run --case terraria/queen_bee --tier easy --format semantic --policy random --max-steps 50
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from . import __version__, envs, frames, lumine, prompting
from .client import ChatClient, ChatError
from .policies import ChatPolicy, NoopPolicy, PythonPolicy, RandomPolicy, ReplayPolicy, Turn
from .session import Session

TIERS = ("easy", "normal", "hard")


def parse_seeds(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or any(s < 0 for s in out):
        raise argparse.ArgumentTypeError("seeds must be non-negative integers, e.g. 1-5 or 1,4,9")
    return out


def _stats(values):
    if not values:
        return None
    values = sorted(values)
    return {"mean": round(statistics.fmean(values), 4), "p50": round(values[len(values) // 2], 4),
            "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 4), "max": round(values[-1], 4)}


def _slug(case_id):
    return case_id.replace("/", "__")


def run_episode(case_id, seed, options, args, policy, out_dir):
    game = envs.game_of(case_id)
    kind = prompting.mouse_kind(game)
    system = prompting.system_prompt(game, case_id, args.format, args.lang, thinking=args.thinking,
                                     counters=args.counters)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "system_prompt.txt").write_text(system, encoding="utf-8", newline="\n")
    if args.save_frames:
        (out_dir / "frames").mkdir()
    size = None if args.image_size == "native" else tuple(int(v) for v in args.image_size.split("x"))
    turns_file = (out_dir / "turns.jsonl").open("w", encoding="utf-8")
    counts = {"decisions": 0, "format_errors": 0, "bare": 0, "lenient_fixes": 0, "unbound_keys": 0, "thoughts": 0,
              "lag_packets": 0}
    latencies, status, error = [], "finished", None
    g, provenance, result, final_hash, sources = None, {}, {}, None, None
    try:
        g = envs.open_game(case_id, args.format, kind)
        obs, info = g.reset(seed, options)
        provenance = g.provenance()
        max_steps = args.max_steps or int(math.ceil((provenance.get("max_seconds") or 600) / 0.2)) + 2
        session = Session(system, window=args.history, user_text=args.user_text)
        turn = Turn(case_id=case_id, game=game, format=args.format, mouse_kind=kind,
                    lumine_keys=prompting.lumine_keys(game), vocab=getattr(g, "vocab", None))
        done = False
        for index in range(max_steps):
            start = time.perf_counter()
            jpeg, url = frames.encode(obs, size, args.jpeg_quality)
            if args.save_frames:
                (out_dir / "frames" / f"{index:05d}.jpg").write_bytes(jpeg)
            turn["index"] = index
            try:
                reply, meta = policy.act(session.messages(url), turn)
            except ChatError as e:
                status, error = "infra_error", str(e)
                break
            latency = time.perf_counter() - start
            latencies.append(latency)
            if args.format == "lumine":
                parsed = lumine.parse(reply, mouse_kind=kind, lenient=args.lenient, accept_bare=not args.no_bare)
                extra = {"bare": parsed.bare, "unbound": parsed.unbound, "mouse": list(parsed.mouse)}
                counts["bare"] += parsed.bare
                counts["unbound_keys"] += len(parsed.unbound)
            else:
                parsed = g.vocab.parse(reply, lenient=True)
                extra = {}
            counts["decisions"] += 1
            counts["format_errors"] += not parsed.ok
            counts["lenient_fixes"] += parsed.lenient_fixes
            counts["thoughts"] += bool(parsed.thought)
            game_t = float(info.get("t", 0.0))
            lag = int(latency // 0.2) if args.time == "realtime" else 0
            for _ in range(lag):
                obs, done, info = g.noop()
                counts["lag_packets"] += 1
                if done:
                    break
            if not done:
                obs, done, info = g.step(parsed)
            executed = g.packets[-1] if hasattr(g, "packets") and g.packets else (parsed.action if parsed.ok else None)
            session.record(url, reply)
            record = {"i": index, "t": game_t, "latency_s": round(latency, 4), "lag_packets": lag, "reply": reply,
                      "thought": parsed.thought, "ok": parsed.ok, "error": parsed.error, "action": parsed.text,
                      "executed": executed, "lenient_fixes": parsed.lenient_fixes, **extra,
                      "finish_reason": meta.get("finish_reason"), "usage": meta.get("usage")}
            if meta.get("reasoning_content"):
                record["reasoning_content"] = meta["reasoning_content"]
            turns_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if done:
                break
        else:
            status = "collector_cut"      # max_steps reached before the game ended: not a game outcome
        result = g.result()
        if hasattr(g, "packets"):
            replay = g.replay()
            (out_dir / "replay.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
            final_hash, sources = replay["final_hash"], replay["source_sha256"]
        else:
            replay = g.replay(out_dir / "semantic_replay.json")
            final_hash, sources = None, replay.get("source")
    except Exception as e:  # noqa: BLE001 - record and continue with the next episode
        status, error = "harness_error", f"{type(e).__name__}: {e}"
    finally:
        turns_file.close()
        if g is not None:
            g.close()
    episode = {
        "schema": "bossbench-llm/episode@1", "harness": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case_id, "game": game, **(provenance or {}), "format": args.format, "lang": args.lang,
        "thinking": args.thinking, "counters": args.counters, "history": args.history,
        "image_size": args.image_size, "jpeg_quality": args.jpeg_quality, "time_mode": args.time,
        "lenient": args.lenient if args.format == "lumine" else True, "mouse_kind": kind,
        "seed": seed, "options": options, "policy": policy.describe(),
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "status": status, "error": error, **result, **counts, "latency_s": _stats(latencies),
        "final_hash": final_hash, "source_sha256": sources,
    }
    (out_dir / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
    return episode


def build_policy(args, case_id, fmt):
    if args.policy == "chat":
        if not args.base_url or not args.model:
            raise SystemExit("--policy chat needs --base-url and --model")
        extra = json.loads(args.extra_body) if args.extra_body else None
        client = ChatClient(args.base_url, args.model, api_key_env=args.api_key_env, temperature=args.temperature,
                            top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.timeout, retries=args.retries,
                            extra_body=extra)
        return ChatPolicy(client)
    if args.policy == "python":
        if not args.policy_factory:
            raise SystemExit("--policy python needs --policy-factory module:function")
        return PythonPolicy(args.policy_factory, json.loads(args.policy_config) if args.policy_config else None)
    if args.policy == "replay":
        game = envs.game_of(case_id)
        spec = None
        if fmt == "semantic":
            from bossbench import compact
            spec = compact.load(game)
        return ReplayPolicy(args.replay, fmt, prompting.mouse_kind(game), spec)
    if args.policy == "noop":
        return NoopPolicy(fmt)
    return RandomPolicy(fmt, seed=args.policy_seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bossbench_llm.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", help="case id, e.g. terraria/king_slime (repeatable)")
    p.add_argument("--tier", choices=TIERS, help="difficulty tier (default normal unless --difficulty)")
    p.add_argument("--difficulty", type=float, help="continuous difficulty in [0, 1] instead of a tier")
    p.add_argument("--seeds", type=parse_seeds, default=[1], help="e.g. 1-5 or 1,3,7 (default 1)")
    p.add_argument("--options", help="extra reset options as JSON, e.g. '{\"weapon\": \"bow\"}'")
    p.add_argument("--format", choices=("lumine", "semantic"), default="lumine")
    p.add_argument("--lang", choices=("en",), default="en")
    p.add_argument("--thinking", action="store_true", help="allow <|thought_start|>...<|thought_end|> reasoning")
    p.add_argument("--counters", action="store_true", help="include how to counter each boss move in the prompt")
    p.add_argument("--policy", choices=("chat", "python", "replay", "noop", "random"), default="chat")
    p.add_argument("--base-url"), p.add_argument("--model")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    p.add_argument("--temperature", type=float, default=1.0), p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, help="default 128, or 512 with --thinking")
    p.add_argument("--extra-body", help="JSON merged into each request, e.g. '{\"skip_special_tokens\": false}'")
    p.add_argument("--timeout", type=float, default=120.0), p.add_argument("--retries", type=int, default=3)
    p.add_argument("--policy-factory"), p.add_argument("--policy-config")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--replay", help="bossbench/policy-replay@1 file for --policy replay (sets case/seed/options)")
    p.add_argument("--history", type=int, default=20, help="image-action pairs kept in context (Lumine: 20)")
    p.add_argument("--image-size", default="1280x720", help="WxH sent to the model, or 'native' (854x480)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--user-text", help="optional text placed before each frame (Lumine sends the image only)")
    p.add_argument("--time", choices=("paused", "realtime"), default="paused")
    p.add_argument("--max-steps", type=int, help="decision cap per episode (default: the game's time limit)")
    p.add_argument("--lenient", action="store_true", help="lumine: accept unambiguous key-name variants (counted)")
    p.add_argument("--no-bare", action="store_true", help="lumine: reject actions without <|action_*|> markers")
    p.add_argument("--save-frames", action="store_true", help="keep every JPEG sent to the model")
    p.add_argument("--out", default="runs", help="output directory (a new timestamped folder is created inside)")
    args = p.parse_args(argv)
    if args.max_tokens is None:
        args.max_tokens = 512 if args.thinking else 128
    jobs = []
    if args.policy == "replay":
        if not args.replay:
            raise SystemExit("--policy replay needs --replay FILE")
        data = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        jobs.append((data["case_id"], int(data["gym_seed"]), dict(data["options"])))
    else:
        if not args.case:
            raise SystemExit("give at least one --case")
        base = json.loads(args.options) if args.options else {}
        if args.tier and args.difficulty is not None:
            raise SystemExit("use --tier or --difficulty, not both")
        if args.difficulty is not None:
            base["difficulty"] = args.difficulty
        else:
            base["tier"] = args.tier or "normal"
        for case_id in args.case:
            envs.game_of(case_id)
            for seed in args.seeds:
                jobs.append((case_id, seed, dict(base)))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) / f"{stamp}-{args.format}-{args.policy}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "command.json").write_text(json.dumps({"argv": sys.argv if argv is None else argv,
                                                   "args": {k: v for k, v in vars(args).items()}},
                                                  ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    episodes = []
    for case_id, seed, options in jobs:
        policy = build_policy(args, case_id, args.format)
        tag = options.get("tier") or options.get("resolved_from_tier") or f"d{options.get('difficulty')}"
        ep = run_episode(case_id, seed, options, args, policy, root / f"{_slug(case_id)}__{tag}__s{seed}")
        episodes.append(ep)
        print(f"{case_id:34s} {str(tag):7s} seed {seed:<4d} {ep['status']:13s} outcome={ep.get('outcome')!s:8s} "
              f"boss_damage={ep.get('boss_damage')!s:7s} t={ep.get('seconds')!s:7s} decisions={ep['decisions']:<5d} "
              f"format_errors={ep['format_errors']}", flush=True)
        if ep["status"] == "harness_error":
            print("  harness error:", ep["error"], file=sys.stderr)
    keys = ("case_id", "seed", "status", "outcome", "win", "boss_damage", "seconds", "decisions", "format_errors",
            "bare", "unbound_keys", "lag_packets")
    (root / "summary.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for ep in episodes:
            f.write(",".join(str(ep.get(k, "")) for k in keys) + "\n")
    wins = sum(bool(ep.get("win")) for ep in episodes)
    print(f"{wins}/{len(episodes)} wins -> {root}")
    return 0 if all(ep["status"] in ("finished", "collector_cut") for ep in episodes) else 1


if __name__ == "__main__":
    sys.exit(main())
