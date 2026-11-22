"""Train and evaluate a single-encounter RGB PPO baseline."""
import argparse
import json
from contextlib import closing
from pathlib import Path
from statistics import mean
import sys

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments/games/prototype"))
import bossbench
from bossbench.scope import BENCH_CASES


class FixedTier(gym.Wrapper):
    """Apply the requested difficulty to every episode reset."""

    def __init__(self, env, tier):
        super().__init__(env)
        self.tier = tier

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "tier" not in options and "difficulty" not in options:
            options["tier"] = self.tier
        return super().reset(seed=seed, options=options)


def make_env(case, tier):
    env = bossbench.make_training_env(case, action_format="compact", time_mode="paused")
    width, height = 160, 96

    def resize(frame):
        return np.array(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))

    space = gym.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
    return gym.wrappers.TransformObservation(FixedTier(env, tier), resize, space)


def evaluate(model, env, episodes, seed):
    results = []
    for episode_seed in range(seed, seed + episodes):
        obs, _ = env.reset(seed=episode_seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        result = env.unwrapped.verifier()
        results.append({
            "seed": episode_seed,
            "win": result["win"],
            "progress": result["boss_damage"],
            "outcome": result["outcome"],
        })
    return {"win_rate": mean(r["win"] for r in results),
            "mean_progress": mean(r["progress"] for r in results), "episodes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(BENCH_CASES), default="doom/cyberdemon")
    parser.add_argument("--tier", choices=("easy", "normal", "hard"), default="easy")
    parser.add_argument("--device", default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--rollout-steps", type=int, default=128)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--output", type=Path, default=Path("runs/ppo.zip"))
    test = commands.add_parser("eval")
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--episodes", type=int, default=3)
    test.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.command == "train" and (args.steps < 2 or args.rollout_steps < 2):
        parser.error("--steps and --rollout-steps must be at least 2")
    if args.command == "eval" and args.episodes < 1:
        parser.error("--episodes must be at least 1")

    with closing(make_env(args.case, args.tier)) as env:
        if args.command == "train":
            batch = min(args.rollout_steps, 64)
            model = PPO("CnnPolicy", env, n_steps=args.rollout_steps, batch_size=batch,
                        seed=args.seed, device=args.device, verbose=1)
            model.learn(total_timesteps=args.steps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
        else:
            model = PPO.load(args.checkpoint, env=env, device=args.device)
            results = evaluate(model, env, args.episodes, args.seed)
            print(json.dumps({"case": args.case, "tier": args.tier, **results}, indent=2))


if __name__ == "__main__":
    main()
