"""Damage/progress-only rewards per Gym step, shared by every registered game.

The engine's aggregate objective HP/progress is trusted scoring data. It is never
added to the image observation or the model prompt.
"""
from __future__ import annotations

import math
import gymnasium as gym

REWARD_CONTRACT = 'bossbench/step-damage@1'


def objective_progress(snapshot):
    """Use the existing aggregate objective, including all phases/required parts."""
    if not isinstance(snapshot, dict):
        raise RuntimeError('reward requires a valid current engine snapshot')
    hp, maximum = float(snapshot['boss_hp']), float(snapshot['boss_max_hp'])
    if not math.isfinite(hp) or not math.isfinite(maximum) or maximum <= 0 or not 0 <= hp <= maximum:
        raise RuntimeError('invalid aggregate objective HP/progress')
    return 1.0 - hp / maximum


class StepDamageReward(gym.Wrapper):
    """Replace sparse/dense return with the signed increment of objective progress.

    A hit gives positive reward; a heal or objective regression gives negative
    reward. No clipping, player-health penalty, step cost or terminal bonus is
    added. Delay windows belong to the same step. Across a complete episode,
    sum(reward) = final progress - reset progress.
    """
    def __init__(self, env):
        super().__init__(env)
        self.metadata = dict(env.metadata, reward_contract=REWARD_CONTRACT)
        self._reward_previous = None
        self._reward_total = 0.0
        self._reward_steps = 0
        self._reward_closed = False
        self._reward_done = True

    def reset(self, *, seed=None, options=None):
        if self._reward_closed:
            raise RuntimeError('environment is closed')
        obs, info = self.env.reset(seed=seed, options=options)
        self._reward_previous = objective_progress(self.unwrapped.snap)
        self._reward_total = 0.0
        self._reward_steps = 0
        self._reward_done = False
        return obs, dict(info, reward_contract=REWARD_CONTRACT)

    def step(self, action, latency=None):
        if self._reward_closed or self._reward_done or self._reward_previous is None:
            raise RuntimeError('reset required before step')
        try:
            obs, _, terminated, truncated, info = self.env.step(action, latency=latency)
            current = objective_progress(self.unwrapped.snap)
        except BaseException:
            self._reward_done = True
            raise
        reward = current - self._reward_previous
        self._reward_previous = current
        self._reward_total += reward
        self._reward_steps += 1
        self._reward_done = bool(terminated or truncated)
        return obs, float(reward), terminated, truncated, dict(info, reward_contract=REWARD_CONTRACT)

    def reward_diagnostics(self):
        """Collector-only summary, not part of the visual observation."""
        return dict(contract=REWARD_CONTRACT, episode_return=self._reward_total,
                    steps=self._reward_steps, done=self._reward_done)

    def close(self):
        if not self._reward_closed:
            try:
                self.env.close()
            finally:
                self._reward_closed = self._reward_done = True


def make_training_env(case_id, *, action_format='keymouse', **kwargs):
    """Public RGB environment; new calls default to versioned high-water progress.

    action_format='compact' selects the game's declared MultiDiscrete interface.
    All reset options, engine clocks, terminal flags and replay rules are kept.
    The raw engine remains sparse internally; this wrapper replaces that scalar.
    Explicit 'boss_damage' or 'bossbench/step-damage@1' retains legacy signed
    delta behavior. New releases use 'bossbench/step-progress-highwater@1'.
    """
    from .progress_reward import StepProgressReward, REWARD_CONTRACT as HIGHWATER
    mode = kwargs.pop('reward_mode', HIGHWATER)
    if mode not in ('boss_damage', REWARD_CONTRACT, HIGHWATER):
        raise ValueError('unknown versioned training reward')
    if action_format == 'keymouse':
        from .policy_env import make_policy_env
        env = make_policy_env(case_id, **kwargs)
    elif action_format == 'compact':
        from .compact import make_compact_env
        env = make_compact_env(case_id, **kwargs)
    else:
        raise ValueError('action_format must be keymouse or compact')
    try:
        return StepDamageReward(env) if mode in ('boss_damage', REWARD_CONTRACT) else StepProgressReward(env, case_id)
    except BaseException:
        env.close()
        raise
