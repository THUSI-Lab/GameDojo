"""Optional browser-backed integrations; no dependency on a game's internal world classes.

The protocol also permits other implementations (native executable, RPC, Python).
Only this integration knows JavaScript and Canvas. All instances have isolated pages.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .engine_contract import EngineSpec, EngineSnapshot, RenderStatus

WEB = Path(__file__).resolve().parents[2] / 'web'


def _discover_native_games():
    """Every web/games/<game>/manifest.json with engine "independent" is a native game.

    New games register themselves by adding their own manifest; no shared file has to change.
    """
    games = []
    for manifest in sorted((WEB / 'games').glob('*/manifest.json')):
        if manifest.parent.name.startswith(('_', '.')):
            continue
        try:
            data = json.loads(manifest.read_text(encoding='utf8'))
        except (OSError, ValueError):
            continue
        if data.get('engine') == 'independent':
            games.append(manifest.parent.name)
    return tuple(games)


NATIVE_GAMES = _discover_native_games()


def _load_manifests():
    """Each independent game declares itself in web/games/<game>/manifest.json (bossbench/game-manifest@1).

    The JS module's own declaration (hz, version, renderer) is checked against it when a backend starts,
    so a game folder can bump its rule version or add script files without editing this module.
    """
    cases = {}
    for game in NATIVE_GAMES:
        manifest = json.loads((WEB / 'games' / game / 'manifest.json').read_text(encoding='utf8'))
        if manifest.get('schema') != 'bossbench/game-manifest@1' or manifest.get('game') != game:
            raise RuntimeError(f'{game}: invalid manifest.json')
        for case_id in manifest['cases']:
            if case_id in cases:
                raise RuntimeError(f'native case declared twice: {case_id}')
            cases[case_id] = {'game': game, 'dimension': manifest['dimension'], 'hz': manifest['native_hz'],
                              'module': manifest['module'], 'renderer': manifest['renderer'],
                              'seconds': manifest['max_seconds'], 'version': manifest['rules'],
                              'scripts': tuple(manifest['scripts'])}
    return cases


NATIVE_CASES = _load_manifests()


def source_paths(case_id):
    spec = NATIVE_CASES[case_id]
    return [Path(__file__), *(WEB / 'games' / path for path in spec['scripts'])]


class JavaScriptBackend:
    def __init__(self, case_id):
        from .webenv import BrowserTransport
        self.case_id = case_id
        entry = NATIVE_CASES[case_id]
        self.spec = EngineSpec(entry['game'], entry['dimension'], entry['hz'], entry['seconds'],
                               entry['renderer'], 'perspective' if entry['dimension'] == '3d' else 'raster',
                               rule_version=entry['version'])
        self.page = None
        self._pid = os.getpid()
        self._ready = False
        self._error = None
        self._reset = False
        try:
            self.page = BrowserTransport._launch().new_page(viewport={'width': 854, 'height': 480})
            self.page.set_content('<canvas id="view" width="854" height="480"></canvas>')
            for script in source_paths(case_id)[1:]:
                self.page.add_script_tag(path=str(script))
            self.page.evaluate('name => { window.backendModule=window[name]; window.backendCanvas=document.querySelector("canvas"); }', entry['module'])
            declaration = self.page.evaluate('() => [backendModule.hz,backendModule.version,backendModule.renderer]')
            if declaration != [self.spec.native_hz, self.spec.rule_version, self.spec.renderer]:
                raise RuntimeError('Python/JavaScript engine declaration mismatch')
            from .tiers import validate_declared
            self._tiers = validate_declared(self.page.evaluate('() => backendModule.tiers || null'),
                                            where=f'{entry["module"]}.tiers')
            # A game that declares validateOptions opts in to game-specific reset options (loadouts etc.);
            # one that does not keeps refusing anything beyond the harness's own keys.
            self._accepts_options = bool(self.page.evaluate(
                '() => typeof backendModule.validateOptions === "function"'))
            self._has_band = bool(self.page.evaluate(
                '() => typeof backendModule.difficultyBand === "function"'))
        except BaseException:
            self.close()
            raise

    def _open(self):
        if self.page is None:
            raise RuntimeError('backend is closed')
        if self._pid != os.getpid():
            raise RuntimeError('backend belongs to another process; use spawned workers')

    # Reset options the shared harness itself understands. Anything else is a *game* option: it is forwarded
    # only when the game declares a validator (Module.validateOptions) and is rejected by that validator's own
    # rules, so a typo still fails loudly instead of being silently ignored. 2026-11-03, requested by
    # isaac_remake (item loadouts); games without a validator keep the old behaviour of refusing extras.
    HARNESS_OPTIONS = frozenset({'difficulty', 'latency', 'engine_seed', 'resolved_from_tier'})

    def game_options(self, options):
        """Split reset options into the harness's own and the game's, validating the latter in the game."""
        extra = {k: v for k, v in (options or {}).items() if k not in self.HARNESS_OPTIONS}
        if not extra:
            return None
        if not self._accepts_options:
            raise ValueError(f'unsupported {self.spec.engine_id} reset options: {sorted(extra)}')
        try:
            self.page.evaluate('a => backendModule.validateOptions(a)', extra)
        except Exception as error:                      # the game's own message is the useful one
            raise ValueError(f'{self.spec.engine_id} rejected reset options {sorted(extra)}: '
                             f'{str(error).splitlines()[0][:160]}') from None
        return extra

    def validate_options(self, options):
        self.game_options(options)

    def _snapshot(self, expression, argument=None):
        # JSON canonicalization matches the Node validation path, including -0.
        result = json.loads(self.page.evaluate(f'a => JSON.stringify({expression})', argument))
        result['events'] = tuple(result['events'])
        return EngineSnapshot(**result)

    def reset(self, seed, difficulty, options):
        self._open()
        extra = self.game_options(options)
        self.page.evaluate('a => { window.backendSession=new NativeBench.Session(backendModule,a.id,a.seed,a.d,a.options);'
                           ' backendSession.recording=false; }',
                           {'id': self.case_id, 'seed': seed, 'd': difficulty, 'options': extra})
        self._reset = True
        self._ready, self._error = False, None
        return self._snapshot('backendSession.snapshot()')

    def advance(self, packet):
        self._open()
        if not self._reset:
            raise RuntimeError('reset before advancing')
        return self._snapshot('backendSession.packet(a)',
                              {'holds': [sorted(h) for h in packet.holds], 'mouse': list(packet.mouse)})

    def render(self):
        self._open()
        if not self._reset:
            raise RuntimeError('reset before rendering')
        try:
            result = self.page.evaluate('''() => {
                const context=backendCanvas.getContext("2d");
                backendModule.draw(backendSession.game,context);
                return {png:backendCanvas.toDataURL("image/png").split(",")[1],
                        status:backendModule.renderStatus ? backendModule.renderStatus(context) : null};
            }''')
            status = result['status']
            if status is not None and (not status.get('ready') or status.get('error')
                                       or status.get('kind') != self.spec.render_kind
                                       or status.get('renderer') != self.spec.renderer):
                raise RuntimeError(f"game renderer not ready: {status}")
            png = result['png']
            with Image.open(io.BytesIO(base64.b64decode(png))) as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
            if rgb.shape != (480, 854, 3):
                raise RuntimeError('renderer returned an incorrect resolution')
            self._ready, self._error = True, None
            return rgb
        except Exception as error:
            self._ready, self._error = False, str(error)
            raise

    def render_status(self):
        return RenderStatus(self.spec.renderer, self.spec.render_kind, self._ready, self._error)

    def declared_tiers(self):
        """The module's own Module.tiers = {easy, normal, hard} -> difficulty, validated at startup."""
        return getattr(self, '_tiers', None)

    def difficulty_band(self, d):
        """Module.difficultyBand(d) -> the game's own name for the segment d falls in, or None.

        Optional. Pure function of d, so it exposes nothing the public info does not already carry, and so
        it is cached per difficulty — _info() runs every step, and querying the page each time wasted a
        measurable share of a step (measured by the Cuphead session on the web route, 2026-11-03).
        """
        if not getattr(self, '_has_band', False):
            return None
        key = float(d)
        cache = getattr(self, '_band_cache', None)
        if cache is None:
            cache = self._band_cache = {}
        if key not in cache:
            from .unified import _clean_band
            cache[key] = _clean_band(
                self.page.evaluate('d => backendModule.difficultyBand(d) ?? null', key),
                f'{self.case_id}.difficultyBand')
        return cache[key]

    def scripted(self):
        """Trusted reference controller (Module.scripted, else Module.oracle) for the current state.

        Returns {holds: 6 key arrays, mouse: [dx, dy, dz]}; it must not change the simulation.
        """
        self._open()
        if not self._reset:
            raise RuntimeError('reset before asking the scripted controller')
        return self.page.evaluate('''() => {
            const M=backendModule, f=M.scripted||M.oracle;
            if(typeof f!=='function') throw Error(M.id+': no scripted controller (Module.scripted)');
            const before=NativeBench.hash(backendSession.state());
            const out=f.call(M, backendSession.game);
            if(NativeBench.hash(backendSession.state())!==before) throw Error('scripted controller changed the simulation');
            return {holds:out.holds.map(h=>[...h]), mouse:[...(out.mouse||[0,0,0])]};
        }''')

    def manual(self):
        self._open()
        return self.page.evaluate('''() => ({controls:Array.isArray(backendModule.controls)
            ? backendModule.controls : Object.entries(backendModule.controls).map(([key,value])=>key+": "+value)})''')

    def close(self):
        page, self.page = self.page, None
        self._ready = False
        if page is not None and self._pid == os.getpid() and not page.is_closed():
            try:
                page.evaluate('''() => {
                    if(window.backendModule && backendModule.closeRenderer && window.backendCanvas)
                        backendModule.closeRenderer(backendCanvas.getContext("2d"));
                }''')
            finally:
                page.close()


def backend_factory(case_id):
    return lambda: JavaScriptBackend(case_id)
