"""Lumine-format action text <-> BossBench key/mouse packets.

Lumine (arXiv 2511.08892; internal CPT preprocessing `get_lumine_token`) emits one action per 200 ms:

    <|action_start|>X Y Z ; K1 ; K2 ; K3 ; K4 ; K5 ; K6<|action_end|>

* X, Y: relative mouse displacement, integers in [-1000, 1000] (X > 0 right, Y > 0 down), in units of 1/1000 of
  the screen width / height. Z: scroll steps in [-5, 5] (Z > 0 up).
* K1..K6: six 33 ms chunks; each holds 0-4 key names separated by spaces. A key listed in consecutive chunks
  stays held (not pressed again); an unlisted key is released.
* Key names: capital letters A-Z; Space Shift Ctrl Alt Tab Esc Caps; number keys as words zero..nine;
  F1-F12 as One..Twelve; mouse buttons LB RB MB.
* Optional reasoning before the action: <|thought_start|> ... <|thought_end|>.

BossBench executes "<|action_start|> dx dy dz ; K1 ; ... ; K6 <|action_end|>" with comma-separated keys from
W A S D Space Shift J K L Q E R LMB RMB (prototype/bossbench/keymouse.py). This module is the only translation
between the two; it never changes timing (both are 6 x 33 ms per 200 ms) and never reads game state.

Mouse units: on "pointer" games (an on-screen crosshair follows the mouse) X/Y are screen-normalised exactly as in
Lumine, so X = 1000 moves the crosshair one full screen width (854 game pixels). On "camera" and "none" games X/Y
are passed through unchanged (1 unit = 1 game mouse count), which keeps every turn rate the game allows reachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import re

from bossbench import keymouse as km

FORMAT = "lumine-action@1"
SCREEN = (854, 480)  # Game coordinate size; model image preprocessing is configured separately.

LETTERS = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
FKEY_WORDS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve")
SPECIAL = ("Space", "Shift", "Ctrl", "Alt", "Tab", "Esc", "Caps")
MOUSE_BUTTONS = ("LB", "RB", "MB")
LUMINE_KEYS = frozenset(LETTERS + NUMBER_WORDS + FKEY_WORDS + SPECIAL + MOUSE_BUTTONS)

# Lumine key name -> BossBench key name. Valid Lumine keys missing here have no binding in any BossBench game.
TO_BOSSBENCH = {k: k for k in ("W", "A", "S", "D", "J", "K", "L", "Q", "E", "R", "Space", "Shift")}
TO_BOSSBENCH.update({"LB": "LMB", "RB": "RMB"})
FROM_BOSSBENCH = {v: k for k, v in TO_BOSSBENCH.items()}

# Lenient mode only: unambiguous case variants and the BossBench spellings of the mouse buttons.
_LENIENT = {k.upper(): k for k in LETTERS + SPECIAL + MOUSE_BUTTONS}
_LENIENT.update({"LMB": "LB", "RMB": "RB", "MMB": "MB", "SPACEBAR": "Space", "CONTROL": "Ctrl", "ESCAPE": "Esc"})

_THOUGHT = re.compile(r"<\|thought_start\|>([\s\S]*?)<\|thought_end\|>")
_THINK = re.compile(r"<think>([\s\S]*?)</think>")
_ACTION = re.compile(r"<\|action_start\|>([\s\S]*?)<\|action_end\|>")
_INT = re.compile(r"[+-]?[0-9]{1,6}")
_TRAILERS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>")


def mouse_scale(kind):
    """(sx, sy): game mouse counts per Lumine unit for this game's mouse kind."""
    if kind == "pointer":
        return Fraction(SCREEN[0], 1000), Fraction(SCREEN[1], 1000)
    return Fraction(1), Fraction(1)


def _round_scaled(value, scale):
    """round(value * scale), halves away from zero, in exact integer arithmetic."""
    scale = Fraction(scale)
    num, den = scale.numerator, scale.denominator
    q = (2 * abs(value) * num + den) // (2 * den)
    return -q if value < 0 else q


@dataclass
class Parsed:
    ok: bool                                   # a format-valid Lumine action was found
    packet: str | None = None                  # BossBench packet to execute; None -> run a no-op (format error)
    mouse: tuple = (0, 0, 0)                   # Lumine X Y Z as written
    groups: list = field(default_factory=list)  # Lumine key tokens per chunk (canonical names)
    thought: str | None = None
    unbound: list = field(default_factory=list)  # valid Lumine keys with no BossBench binding (dropped)
    bare: bool = False                         # the <|action_*|> markers were missing (e.g. stripped by the server)
    lenient_fixes: int = 0                     # tokens accepted only by lenient normalisation
    error: str | None = None

    @property
    def text(self):
        """Canonical Lumine text of what was parsed (None when not ok)."""
        return format_action(self.mouse, self.groups) if self.ok else None


def format_action(mouse, groups):
    """Canonical Lumine text: 'X Y Z ; g1 ; ... ; g6' wrapped in the action markers."""
    x, y, z = (int(v) for v in mouse)
    if len(groups) != km.SUBSTEPS:
        raise ValueError("Lumine actions have exactly 6 chunks")
    return "<|action_start|>%d %d %d ; " % (x, y, z) + " ; ".join(" ".join(g) for g in groups) + "<|action_end|>"


def strip_thoughts(text):
    """-> (text without reasoning blocks, last reasoning text or None). Accepts Lumine and <think> blocks."""
    thought = None
    for pattern in (_THOUGHT, _THINK):
        found = pattern.findall(text)
        if found:
            thought = found[-1].strip() or thought
        text = pattern.sub(" ", text)
    return text, thought


def _canonical_token(token, lenient):
    if token in LUMINE_KEYS:
        return token, False
    if lenient:
        fixed = _LENIENT.get(token.upper())
        if fixed is not None:
            return fixed, True
    return None, False


def parse(reply, *, mouse_kind="none", lenient=False, accept_bare=True):
    """Model reply text -> Parsed (with the BossBench packet to run). Never raises on model output."""
    text = "" if reply is None else str(reply)
    text, thought = strip_thoughts(text)
    found = _ACTION.findall(text)
    bare = False
    if len(found) > 1:
        return Parsed(False, thought=thought, error=f"{len(found)} action blocks in one reply")
    if found:
        body = found[0]
    else:
        if "<|action_start|>" in text:
            return Parsed(False, thought=thought, error="unterminated action (no <|action_end|>)")
        body = text.strip()
        for trailer in _TRAILERS:
            body = body.replace(trailer, "")
        body = body.strip()
        if not accept_bare or not body:
            return Parsed(False, thought=thought, error="no <|action_start|>...<|action_end|> block")
        bare = True
    parts = body.split(";")
    if len(parts) != 1 + km.SUBSTEPS:
        return Parsed(False, thought=thought, bare=bare,
                      error=f"expected mouse + 6 chunks separated by ';', got {len(parts)} fields")
    mouse_tokens = parts[0].split()
    if len(mouse_tokens) != 3 or not all(_INT.fullmatch(v) for v in mouse_tokens):
        return Parsed(False, thought=thought, bare=bare, error=f"mouse field must be 3 integers, got {parts[0].strip()!r}")
    x, y, z = (int(v) for v in mouse_tokens)
    if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 5:
        return Parsed(False, thought=thought, bare=bare, error=f"mouse out of range: {x} {y} {z}")
    groups, unbound, fixes = [], [], 0
    for index, part in enumerate(parts[1:], 1):
        raw = part.replace(",", " ").split() if lenient else part.split()
        if len(raw) > km.MAX_KEYS:
            return Parsed(False, thought=thought, bare=bare, error=f"chunk {index} has {len(raw)} keys (max 4)")
        group = []
        for token in raw:
            name, fixed = _canonical_token(token, lenient)
            if name is None:
                return Parsed(False, thought=thought, bare=bare, error=f"chunk {index}: unknown key name {token!r}")
            if name in group:
                return Parsed(False, thought=thought, bare=bare, error=f"chunk {index}: key {name} repeated")
            fixes += fixed
            group.append(name)
            if name not in TO_BOSSBENCH:
                unbound.append(name)
        groups.append(group)
    sx, sy = mouse_scale(mouse_kind)
    subs = [[TO_BOSSBENCH[k] for k in g if k in TO_BOSSBENCH] for g in groups]
    packet = km.format_packet(subs, (_round_scaled(x, sx), _round_scaled(y, sy), z))
    if km.parse_packet(packet) is None:   # cannot happen for valid input; keep the contract explicit
        return Parsed(False, thought=thought, bare=bare, error="internal: converted packet rejected")
    return Parsed(True, packet=packet, mouse=(x, y, z), groups=groups, thought=thought, unbound=unbound,
                  bare=bare, lenient_fixes=fixes)


def from_packet(packet, *, mouse_kind="none"):
    """BossBench packet text -> canonical Lumine text that converts back to exactly the same packet.

    Raises ValueError if the packet is invalid or its mouse cannot be expressed in Lumine units for this game."""
    p = km.parse_packet(packet if packet is not None else km.NOOP)
    if p is None:
        raise ValueError(f"not a valid BossBench packet: {packet!r}")
    sx, sy = mouse_scale(mouse_kind)
    dx, dy, dz = p["mouse"]
    x, y = _round_scaled(dx, 1 / sx), _round_scaled(dy, 1 / sy)
    if abs(x) > 1000 or abs(y) > 1000 or _round_scaled(x, sx) != dx or _round_scaled(y, sy) != dy:
        raise ValueError(f"mouse {dx} {dy} is not expressible in Lumine units for a {mouse_kind!r} game")
    groups = [sorted(FROM_BOSSBENCH[k] for k in sub) for sub in p["subs"]]
    return format_action((x, y, dz), groups)


def same_packet(a, b):
    """True when two BossBench packet texts parse to the same action."""
    pa, pb = km.parse_packet(a), km.parse_packet(b)
    return pa is not None and pb is not None and pa["mouse"] == pb["mouse"] and pa["subs"] == pb["subs"]
