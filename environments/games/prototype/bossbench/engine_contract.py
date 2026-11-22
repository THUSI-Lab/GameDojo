"""External engine boundary. No Arena/Body/Boss/Case or renderer library required."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

API_VERSION = 'bossbench/engine@1'


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    dimension: str
    native_hz: int
    max_seconds: float
    renderer: str
    render_kind: str  # raster | perspective
    rule_version: str = '1'
    resolution: tuple[int, int] = (854, 480)

    def __post_init__(self):
        if self.dimension not in ('2d', '3d') or self.render_kind not in ('raster', 'perspective'):
            raise ValueError('invalid engine dimension or render_kind')
        if isinstance(self.native_hz, bool) or not isinstance(self.native_hz, int) or not 5 <= self.native_hz <= 1000:
            raise ValueError('native_hz must be an integer in [5, 1000]')
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('max_seconds must be positive and finite')
        if not self.engine_id or not self.renderer or not self.rule_version:
            raise ValueError('engine identity, renderer and rules must be declared')
        if len(self.resolution) != 2 or any(type(v) is not int or v <= 0 for v in self.resolution):
            raise ValueError('resolution must contain positive integer width/height')


@dataclass(frozen=True)
class RenderStatus:
    renderer: str
    kind: str
    ready: bool
    error: str | None = None


@dataclass(frozen=True)
class InputPacket:
    """200ms of input; six equally spaced held-key samples and total relative mouse delta."""
    mouse: tuple[float, float, float]
    holds: tuple[frozenset[str], ...]

    @classmethod
    def noop(cls):
        return cls((0, 0, 0), (frozenset(),) * 6)


@dataclass(frozen=True)
class TickInput:
    held: frozenset[str]
    pressed: frozenset[str]
    mouse: tuple[float, float, float]


class NativeClock:
    """Optional integer scheduler: sample holds at each native tick, without clock drift.

    The right endpoint belongs to the preceding input interval. Thus 35Hz uses
    7 ticks per packet while 24Hz alternates 4/5, executing exactly 24 per second.
    Engines with their own native input queue may implement this boundary themselves.
    """
    def __init__(self, hz):
        self.hz = hz
        self.reset()

    def reset(self):
        self.packets = 0
        self.ticks = 0
        self.held = frozenset()

    def schedule(self, packet: InputPacket):
        end = (self.packets + 1) * self.hz // 5
        count = end - self.ticks
        result = []
        for tick in range(self.ticks + 1, end + 1):
            sub = (tick * 30 + self.hz - 1) // self.hz - 1 - self.packets * 6
            held = packet.holds[max(0, min(5, sub))]
            pressed = held - self.held
            mouse = tuple(v / count for v in packet.mouse)
            result.append(TickInput(held, pressed, mouse))
            self.held = held
        self.packets += 1
        self.ticks = end
        return result


@dataclass
class EngineSnapshot:
    """Normalized diagnostics, owned by the trusted adapter; never a model observation.

    ticks is absolute native time, frames is the number executed by this call.
    Multipart bosses report aggregate remaining progress as hp/max_hp here.
    state must include all replay-relevant game state/RNG/input state.
    """
    ticks: int
    frames: int
    outcome: str | None
    player_hp: float
    player_max_hp: float
    boss_hp: float
    boss_max_hp: float
    state: dict
    events: tuple = ()


class GameBackend(Protocol):
    """A game may be an in-process engine, subprocess, RPC client or browser client.

    validate_options must be pure. reset must establish a fresh independent
    episode. advance executes one packet and stops on the exact terminal tick.
    render is read-only and returns a perspective RGB for a 3D game. close must
    release resources and be idempotent. The adapter owns process isolation.
    """
    spec: EngineSpec

    def validate_options(self, options: dict) -> None: ...
    def reset(self, seed: int, difficulty: float, options: dict) -> EngineSnapshot: ...
    def advance(self, packet: InputPacket) -> EngineSnapshot: ...
    def render(self) -> np.ndarray: ...
    def render_status(self) -> RenderStatus: ...
    def manual(self) -> dict: ...
    def close(self) -> None: ...
