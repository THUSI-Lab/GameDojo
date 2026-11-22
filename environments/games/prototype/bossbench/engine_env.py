"""Gym adapter for independent game backends (engine_contract.GameBackend)."""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math

from . import keymouse as km
from .engine_contract import EngineSnapshot, InputPacket
from .unified import UnifiedEnv


class EngineEnv(UnifiedEnv):
    """No access to a game world, Body, Boss or a specific rendering library."""
    def __init__(self, case_id, backend, **kwargs):
        self.backend = backend
        try:
            super().__init__(case_id, **kwargs)
            if self.action_mode != 'keymouse':
                raise ValueError('independent engine adapter accepts keymouse; native actions stay inside the backend')
            self.fps = self.engine_spec.native_hz
            self.max_seconds = self.engine_spec.max_seconds
            self.max_ticks = math.ceil(self.max_seconds * self.fps)
            self.observation_space = self._obs_space()
        except BaseException:
            backend.close()
            raise

    @property
    def engine_spec(self):
        return self.backend.spec

    def render_status(self):
        return self.backend.render_status()

    def _declared_tiers(self):
        declared = getattr(self.backend, 'declared_tiers', None)
        return declared() if callable(declared) else None

    def _difficulty_band(self, d):
        band = getattr(self.backend, 'difficulty_band', None)
        return band(d) if callable(band) else None

    def _validate_options(self, options):
        super()._validate_options(options)
        self.backend.validate_options(copy.deepcopy(options))

    def _snapshot(self, value, *, previous=None, reset=False):
        if not isinstance(value, EngineSnapshot):
            raise TypeError('backend must return EngineSnapshot')
        if type(value.ticks) is not int or type(value.frames) is not int or min(value.ticks, value.frames) < 0:
            raise ValueError('backend ticks/frames must be nonnegative integers')
        if reset and (value.ticks != 0 or value.frames != 0 or value.outcome is not None):
            raise ValueError('reset must return a fresh live episode at tick zero')
        if previous is not None:
            expected = round(previous['t'] * self.fps) + value.frames
            if value.ticks != expected or value.frames > math.ceil(self.fps * km.PACKET_SECONDS):
                raise ValueError('backend native time does not match the executed packet')
            if not value.frames and value.outcome is None:
                raise ValueError('live packet did not advance simulation')
        if value.outcome not in (None, 'win', 'loss', 'timeout'):
            raise ValueError('unknown backend outcome')
        if value.ticks >= self.max_ticks and value.outcome is None:
            raise ValueError('backend exceeded its declared episode deadline')
        for hp, maximum in ((value.player_hp,value.player_max_hp),(value.boss_hp,value.boss_max_hp)):
            if not math.isfinite(hp) or not math.isfinite(maximum) or maximum <= 0 or not 0 <= hp <= maximum:
                raise ValueError('invalid normalized backend health/progress')
        if not isinstance(value.state, dict):
            raise ValueError('backend diagnostics state must be a JSON object')
        result = asdict(value)
        json.dumps(result, allow_nan=False)
        result.pop('ticks')
        result['t'] = value.ticks / self.fps
        return copy.deepcopy(result)

    def _engine_reset(self, seed, d, options=None):
        return self._snapshot(self.backend.reset(seed, d, copy.deepcopy(options or {})), reset=True)

    def _engine_packet(self, packet, lag, ok):
        try:
            return self._advance_packets(packet, lag, ok)
        except BaseException:
            # A failed backend may have advanced privately; never expose the stale snapshot as live.
            self.snap, self.done = None, True
            raise

    def _advance_packets(self, packet, lag, ok):
        parsed = km.parse_packet(packet) if ok else None
        action = (InputPacket(tuple(parsed['mouse']),tuple(frozenset(h) for h in parsed['subs']))
                  if parsed is not None else InputPacket.noop())
        snapshot = self.snap
        frames = lag_frames = 0
        events = []
        for _ in range(lag):
            if snapshot['outcome']:
                break
            snapshot = self._snapshot(self.backend.advance(InputPacket.noop()),previous=snapshot)
            frames += snapshot['frames']
            lag_frames += snapshot['frames']
            events.extend(snapshot['events'])
        if not snapshot['outcome']:
            snapshot = self._snapshot(self.backend.advance(action),previous=snapshot)
            frames += snapshot['frames']
            events.extend(snapshot['events'])
        return {**snapshot,'frames':frames,'events':events,'lag_frames':lag_frames}

    def _manual(self):
        return copy.deepcopy(self.backend.manual())

    def _render(self):
        return self.backend.render()

    def close(self):
        if not self._closed:
            try:
                self.backend.close()
            finally:
                super().close()
