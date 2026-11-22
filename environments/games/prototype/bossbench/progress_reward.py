"""Versioned high-water progress reward; trusted scoring only, never policy input."""
from __future__ import annotations

import math
import gymnasium as gym

REWARD_CONTRACT = 'bossbench/step-progress-highwater@1'


def unit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError('progress must be numeric')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError('progress must be finite and in [0,1]')
    return value


class HighWaterProgress:
    def __init__(self, initial):
        self.initial = self.peak = self.final = unit(initial)
        self.steps = 0

    def step(self, current):
        current = unit(current)
        peak = max(self.peak, current)
        reward = peak - self.peak
        self.peak, self.final = peak, current
        self.steps += 1
        return reward

    def diagnostics(self):
        return dict(contract=REWARD_CONTRACT, initial_progress=self.initial,
                    peak_progress=self.peak, final_progress=self.final,
                    episode_return=self.peak - self.initial, steps=self.steps)


def objective_progress(snapshot, case_id):
    """Read scoring fields while preserving the original game's HP and outcome."""
    from .step_reward import objective_progress as hp_progress
    if case_id == 'cuphead/root_pack':
        boss = snapshot['state']['boss']
        score = boss['reward_objective']
        if score['version'] != 'root-pack-fixed-phase@1':
            raise RuntimeError('Root Pack fixed phase scoring unavailable')
        return unit(score['progress'])
    if case_id == 'nfsmw/speedtrap':
        game = snapshot['state']['game']
        score, target = game['you'], game['reward_target_score']
        if game['reward_target_version'] != 'speedtrap-reachable-profile@1':
            raise RuntimeError('Speedtrap calibration unavailable')
        if not math.isfinite(score) or score < 0 or not math.isfinite(target) or target <= 0:
            raise RuntimeError('Invalid speedtrap score/target')
        return unit(min(1.0, score / target))
    return unit(hp_progress(snapshot))


class StepProgressReward(gym.Wrapper):
    """Reward new peaks at env.step boundaries, including each step's lag window.

    No reward rescaling by batch, terminal bonus, time/death/format penalty, or
    repeated credit after healing. Imported initial progress is not rewarded.
    """
    def __init__(self, env, case_id):
        super().__init__(env)
        self.case_id = case_id
        self.metadata = dict(env.metadata, reward_contract=REWARD_CONTRACT)
        self.scorer = None
        self._closed, self._done = False, True

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError('environment is closed')
        obs, info = self.env.reset(seed=seed, options=options)
        self._done = True
        self.scorer = HighWaterProgress(objective_progress(self.unwrapped.snap, self.case_id))
        self._done = False
        return obs, dict(info, reward_contract=REWARD_CONTRACT)

    def step(self, action, latency=None):
        if self._closed or self._done or self.scorer is None:
            raise RuntimeError('reset required before step')
        try:
            obs, _, terminated, truncated, info = self.env.step(action, latency=latency)
            reward = self.scorer.step(objective_progress(self.unwrapped.snap, self.case_id))
        except BaseException:
            self._done = True
            raise
        self._done = bool(terminated or truncated)
        return obs, reward, terminated, truncated, dict(info, reward_contract=REWARD_CONTRACT)

    def reward_diagnostics(self):
        if self.scorer is None:
            raise RuntimeError('reset required')
        return dict(self.scorer.diagnostics(), done=self._done)

    def close(self):
        if not self._closed:
            try:
                self.env.close()
            finally:
                self._closed = self._done = True
