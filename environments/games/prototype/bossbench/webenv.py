"""Gymnasium binding for the web engine (web/engine + web/games/<game>/, 60 Hz): the Hollow-Knight-style 2D cases.

The simulation stays in JavaScript (one source of truth for the browser game, the Node tests and
training). Each case belongs to one game folder (bossbench.WEB_GAME), whose index.html fixes the script order.
This module drives it through web/engine/bridge.js over one of two transports:

  node     `node web/tools/gym_server.js <game>`, JSON lines on stdin/stdout. symbolic only, fast, no browser.
  browser  a headless Chromium page (Playwright) with the game's scripts + renderer loaded; needed for obs_mode="rgb".
           BB_CHANNEL=msedge|chrome picks an installed browser instead of Playwright's own build.

transport="auto" picks browser for rgb and node otherwise. Both transports load the same game rules.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import subprocess

import numpy as np

from .unified import UnifiedEnv, _finite_number, _plain

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")
NO_PAGE = ("engine/input.js", "engine/app.js")   # browser-app UI: needs the game's DOM, not part of the env

DEFAULT_GAME = "hollowknight_remake"


def game_of(case_id):
    from . import WEB_GAME
    if case_id not in WEB_GAME:
        raise KeyError(f"no web game registers {case_id!r}; known: {sorted(WEB_GAME)}")
    return WEB_GAME[case_id]


def game_scripts(game):
    """Absolute paths of web/games/<game>/index.html's scripts in page order, minus the browser-app layer."""
    gdir = os.path.join(WEB, "games", game)
    with open(os.path.join(gdir, "index.html"), encoding="utf-8") as source:
        html = source.read()
    out = [os.path.normpath(os.path.join(gdir, s)) for s in re.findall(r'<script\s+src="([^"]+)"', html)]
    return [p for p in out if os.path.relpath(p, WEB).replace(os.sep, "/") not in NO_PAGE]


class NodeTransport:
    def __init__(self, game=DEFAULT_GAME):
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("web cases need Node.js on PATH (node web/tools/gym_server.js <game>)")
        self.p = subprocess.Popen([node, os.path.join(WEB, "tools", "gym_server.js"), game], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)

    def __call__(self, msg):
        self.p.stdin.write(json.dumps(msg, allow_nan=False, default=_plain) + "\n")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"web gym server exited (code {self.p.poll()})")
        return json.loads(line)

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait(timeout=5)
        elif not self.p.stdin.closed:
            self.p.stdin.close()
        if not self.p.stdout.closed:
            self.p.stdout.close()


class BrowserTransport:
    """One shared headless browser per process, one page per env."""
    _pw = _browser = None
    _owner_pid = None

    @classmethod
    def _launch(cls):
        if cls._browser is not None and cls._owner_pid != os.getpid():
            raise RuntimeError("browser runtime belongs to another process; use AsyncVectorEnv(context='spawn')")
        if cls._browser is None:
            from playwright.sync_api import sync_playwright
            cls._pw = sync_playwright().start()
            cls._owner_pid = os.getpid()
            want = os.environ.get("BB_CHANNEL")
            errors = []
            for ch in ([want] if want else [None, "chrome", "msedge"]):
                try:
                    cls._browser = cls._pw.chromium.launch(channel=ch, args=["--no-sandbox"])
                    break
                except Exception as ex:      # this channel is not installed; try the next one
                    errors.append(f"{ch or 'playwright chromium'}: {str(ex).splitlines()[0]}")
            if cls._browser is None:
                cls._pw.stop()
                cls._pw = None
                raise RuntimeError("no headless browser for rgb web cases (run `playwright install chromium` "
                                   "or set BB_CHANNEL):\n  " + "\n  ".join(errors))
            atexit.register(cls._shutdown)
        return cls._browser

    @classmethod
    def _shutdown(cls):
        if cls._owner_pid is not None and cls._owner_pid != os.getpid():
            # A forked child's atexit must not close the parent's browser/pipes.
            cls._browser = cls._pw = cls._owner_pid = None
            return
        try:
            if cls._browser is not None:
                cls._browser.close()
        except Exception:
            pass
        try:
            if cls._pw is not None:
                cls._pw.stop()
        except Exception:
            pass
        cls._browser = cls._pw = cls._owner_pid = None

    def __init__(self, game=DEFAULT_GAME):
        self.page = self._launch().new_page()
        self._pid = os.getpid()
        try:
            for path in game_scripts(game):
                self.page.add_script_tag(path=path)
            self.page.add_script_tag(path=os.path.join(WEB, "engine", "bridge.js"))
        except Exception:
            self.close()
            raise

    def __call__(self, msg):
        if self._pid != os.getpid():
            raise RuntimeError("browser page belongs to another process; create env in a spawn worker")
        # JSON.stringify on the page, as the Node transport does (Playwright's own serializer keeps -0 apart from 0)
        return json.loads(self.page.evaluate(
            "m => { try { return JSON.stringify(BB.bridge(m)); } catch (e) { return JSON.stringify({ error: String(e.stack || e) }); } }",
            msg))

    def close(self):
        if self._pid != os.getpid():
            return
        try:
            self.page.close()
        except Exception:
            pass


def list_cases(game=DEFAULT_GAME):
    t = NodeTransport(game)
    try:
        return t({"cmd": "cases"})["cases"]
    finally:
        t.close()


class WebCaseEnv(UnifiedEnv):
    fps = 60

    def __init__(self, case_id, transport="auto", game=None, **kw):
        super().__init__(case_id, **kw)
        if self.action_mode != "keymouse":
            raise ValueError("web cases only take key/mouse packets (action_mode='keymouse')")
        self.observation_space = self._obs_space(None)
        if transport not in ("auto", "browser", "node"):
            raise ValueError("transport must be 'auto', 'browser' or 'node'")
        if transport == "auto":
            transport = "browser" if self.obs_mode == "rgb" else "node"
        if self.obs_mode == "rgb" and transport != "browser":
            raise ValueError("RGB observations require browser transport")
        self.transport, self.game = transport, game or game_of(case_id)
        self.t = BrowserTransport(self.game) if transport == "browser" else NodeTransport(self.game)
        try:
            meta = self._call({"cmd": "make", "id": case_id, "rgb": transport == "browser"})
            from .tiers import validate_declared
            self._tiers = validate_declared(self._call({"cmd": "meta"}).get("tiers"),
                                            where=f"{self.game} BB.GAME.tiers")
        except Exception:
            self.close()
            raise
        self.fps, self.max_seconds, self._man = meta["fps"], meta["max_seconds"], meta["manual"]
        self._frame = None

    def _declared_tiers(self):
        return self._tiers

    def _difficulty_band(self, d):
        """The game's own BB.GAME.difficultyBand(d), read through the bridge like `tiers`.

        Cached per difficulty: the band is a pure function of d, but _info() runs on every step, so querying
        the page each time cost ~25% of a symbolic step (0.095 ms of 0.39 ms) and 2.3% of an RGB step. Found
        and measured by the Cuphead session, 2026-11-03. A dict keyed by d keeps `reset(difficulty=...)`
        correct across difficulties within one episode-sequence without re-querying.
        """
        key = float(d)
        cache = getattr(self, '_band_cache', None)
        if cache is None:
            cache = self._band_cache = {}
        if key not in cache:
            from .unified import _clean_band
            try:
                value = self._call({'cmd': 'band', 'd': key}).get('band')
            except Exception:
                value = None
            cache[key] = _clean_band(value, f'{self.case_id}.difficultyBand')
        return cache[key]

    def _validate_options(self, options):
        super()._validate_options(options)
        if self.case_id.startswith("terraria/") and "dawn_in_seconds" in options:
            options["dawn_in_seconds"] = _finite_number(options["dawn_in_seconds"], "dawn_in_seconds", upper=540)

    def _call(self, msg):
        res = self.t(msg)
        if "error" in res:
            raise RuntimeError(f"{self.case_id}: {res['error']}")
        return res

    def _snapshot(self, res):
        if "frame" in res:
            f = res["frame"]
            self._frame = np.frombuffer(base64.b64decode(f["b64"]), np.uint8).reshape(f["shape"])
        i = res["info"]
        return {"state": res["state"], "events": i["events"], "outcome": i["outcome"], "t": i["t"],
                "frames": i["frames"], "player_hp": i["player_hp"], "player_max_hp": i["player_max_hp"],
                "boss_hp": i["boss_hp"], "boss_max_hp": i["boss_max_hp"]}

    def _engine_reset(self, seed, d, options=None):
        # Game options (e.g. Terraria's dawn_in_seconds) must reach the same reset
        # as human play. Keep the top-level difficulty for older bridge clients.
        return self._snapshot(self._call({"cmd": "reset", "seed": seed, "difficulty": d,
                                         "options": dict(options or {})}))

    def _engine_packet(self, packet, lag, ok):
        return self._snapshot(self._call({"cmd": "step", "packet": packet, "lag": lag}))

    def _render(self):
        if self.transport != "browser":
            raise RuntimeError(f"{self.case_id}: rgb frames need transport='browser' (obs_mode='rgb' picks it)")
        return self._frame

    def _manual(self):
        return self._man

    def close(self):
        if not self._closed:
            try:
                if getattr(self, "t", None) is not None:
                    self.t.close()
            finally:
                self.t = None
                super().close()
