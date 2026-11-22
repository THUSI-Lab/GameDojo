"""System prompts for the LLM harness: Lumine format and semantic format, English only.

The Lumine-format prompt copies the Lumine / CPT system prompt (6 x 33 ms chunks, space-separated key names,
LB/RB mouse buttons) and appends what this game needs: the keys that do something here, what the mouse does, and
the boss's moves. Health bars and other status are not described: the model reads them from the screen.

Per-game text lives in prompts/games/<game>.json (schema bossbench-llm/game-prompt@1); the semantic action groups
are read live from the game's own declaration (semantic.Vocabulary). A prompt is a fixed concatenation of parts
(intro, format, game, boss, footer); the HTML page joins the same parts with the same separator.
"""
from __future__ import annotations
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from . import lumine
from .semantic import TERRARIA, Vocabulary
DATA = Path(os.environ.get('BOSSBENCH_LLM_PROMPTS') or Path(__file__).resolve().parent / 'prompts' / 'games')
SCHEMA = 'bossbench-llm/game-prompt@1'
SEP = '\n\n'
LANGS = ('en',)
FORMATS = ('lumine', 'semantic')

@lru_cache(maxsize=None)
def load_game(game):
    path = DATA / f'{game}.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or data.get('game') != game:
        raise ValueError(f'{path}: not a {SCHEMA} file for {game}')
    return data

def available_games():
    return sorted((p.stem for p in DATA.glob('*.json')))

def scene(game_data, case_id):
    for s in game_data['scenes']:
        if s['case_id'] == case_id:
            return s
    raise KeyError(f"{case_id} not in {game_data['game']} prompt data")

def mouse_kind(game):
    return load_game(game).get('mouse', {}).get('kind', 'none')

def _t(pair, lang):
    if pair is None:
        return ''
    if isinstance(pair, str):
        return pair
    return (pair.get(lang) or pair.get('en') or pair.get('zh') or '').strip()

def _sentence(text, lang):
    # Prompts are English only; lang is kept so existing callers do not change.
    text = text.strip()
    if text and text[-1] == ')' and text[-2:-1] not in '.!?':
        text += '.'
    elif text and text[-1] not in '.!?;:)"':
        text += '.'
    return text
_INTRO = {('lumine', 'zh'): 'You are an experienced PC game player, proficient in keyboard and mouse operations. You are playing "{title}". Based on the current game screen and the corresponding historical trajectory, plan the next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms apart. Every step lasts 33ms from its start time until the next step begins.', ('lumine', 'en'): 'You are an experienced PC game player, proficient in keyboard and mouse operations. You are playing "{title}". Based on the current game screen and the corresponding historical trajectory, plan the next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms apart. Every step lasts 33ms from its start time until the next step begins.', ('semantic', 'zh'): 'You are an experienced PC game player. You are playing "{title}". Based on the current game screen and the corresponding historical trajectory, choose the action for the next 200ms.', ('semantic', 'en'): 'You are an experienced PC game player. You are playing "{title}". Based on the current game screen and the corresponding historical trajectory, choose the action for the next 200ms.'}
_LUMINE_FORMAT = {'zh': '**Output Format**\n<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>\n**Explanation**\n1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means move right, Y>0 means move down) and scroll amount Z (Z>0 means scroll up). X and Y are integers in [-1000, 1000]; Z is an integer in [-5, 5].\n2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are separated by spaces, and groups are separated by semicolons.\n- Each group can contain up to 4 keys.\n- If a group has no keys, leave it empty but keep the `;`.\n- A key listed in consecutive groups stays held down and is not pressed again; a key that is not listed is released.\n3. Only output a plain string that conforms to the above format — no line breaks and no quotation marks.\n**Key Naming Rules**\n- Letter keys: the capital letter, e.g., `W`, `A`.\n- Other keys (Space, Shift, etc.): the real key name with an initial capital letter, e.g., `Space`, `Shift`.\n- Mouse buttons: `LB` (left button), `RB` (right button).', 'en': '**Output Format**\n<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>\n**Explanation**\n1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means move right, Y>0 means move down) and scroll amount Z (Z>0 means scroll up). X and Y are integers in [-1000, 1000]; Z is an integer in [-5, 5].\n2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are separated by spaces, and groups are separated by semicolons.\n- Each group can contain up to 4 keys.\n- If a group has no keys, leave it empty but keep the `;`.\n- A key listed in consecutive groups stays held down and is not pressed again; a key that is not listed is released.\n3. Only output a plain string that conforms to the above format — no line breaks and no quotation marks.\n**Key Naming Rules**\n- Letter keys: the capital letter, e.g., `W`, `A`.\n- Other keys (Space, Shift, etc.): the real key name with an initial capital letter, e.g., `Space`, `Shift`.\n- Mouse buttons: `LB` (left button), `RB` (right button).'}
_THINKING = {('lumine', 'zh'): '**Reasoning (optional)**: when the situation changes (a new attack starts, your plan fails), you may first write a short thought as <|thought_start|>your reasoning<|thought_end|> and then the action; otherwise output only the action.', ('lumine', 'en'): '**Reasoning (optional)**: when the situation changes (a new attack starts, your plan fails), you may first write a short thought as <|thought_start|>your reasoning<|thought_end|> and then the action; otherwise output only the action.', ('semantic', 'zh'): '**Reasoning (optional)**: when the situation changes, you may write a short thought as <|thought_start|>your reasoning<|thought_end|> before the JSON; otherwise output only the JSON.', ('semantic', 'en'): '**Reasoning (optional)**: when the situation changes, you may write a short thought as <|thought_start|>your reasoning<|thought_end|> before the JSON; otherwise output only the JSON.'}

def _semantic_example(vocab):
    if vocab.game == TERRARIA:
        return '{"move": "left", "vert": "jump", "attack": "shoot 135"}'
    pick = {}
    for h in vocab.heads[:2]:
        pick[h['name']] = h['options'][min(1, len(h['options']) - 1)]['label']
    return json.dumps(pick, ensure_ascii=False)

def _option_text(game_data, vocab, head, option, lang):
    if lang == 'zh':
        return option.get('zh') or option['label']
    glossary = game_data.get('compact_glossary', {}).get(head['name'], {}).get('options', {})
    return glossary.get(option['label']) or option.get('en') or option['label'].replace('_', ' ')

def _head_text(game_data, head, lang):
    if lang == 'zh':
        return head.get('zh') or head['name']
    return game_data.get('compact_glossary', {}).get(head['name'], {}).get('en') or head.get('en') or head['name']

def _semantic_format(game_data, vocab, lang):
    zh = lang == 'zh'
    lines = ['**Output Format**', 'Output one JSON object on a single line that picks one option for each action group, for example:', _semantic_example(vocab)]
    lines += ['- All groups act at the same time, for the next 200ms.', '- A group you leave out takes its first option (none = do nothing).', '- Use the option names exactly as listed; do not output option indices or any other text.']
    lines.append('**Action groups**')
    for head in vocab.heads:
        if vocab.game == TERRARIA and head['name'] == 'attack':
            opts = [o for o in head['options'] if not o['label'].startswith('shoot ')]
            shoot = '`shoot <angle>` (shoot an arrow toward that angle; integer degrees 0-359, 0° = right, 90° = up, rounded to multiples of 22.5°)'
            items = [f"`{o['label']}` ({_option_text(game_data, vocab, head, o, lang)})" for o in opts[:1]]
            items.append(shoot)
            items += [f"`{o['label']}` ({_option_text(game_data, vocab, head, o, lang)})" for o in opts[1:]]
        else:
            items = [f"`{o['label']}` ({_option_text(game_data, vocab, head, o, lang)})" for o in head['options']]
        sep = '; '
        lines.append(f"- `{head['name']}` ({_head_text(game_data, head, lang)}): " + sep.join(items))
    return '\n'.join(lines)
_DEG = re.compile('(\\d+(?:\\.\\d+)?)\\s*°')

def _camera_sign(vocab):
    """+1 when X > 0 turns the view right (normal), -1 when this game turns it left; None if unknown."""
    if vocab is None:
        return None
    for h in vocab.heads:
        for o in h['options']:
            dx = o.get('dx', 0)
            if dx:
                label = (o['label'] + ' ' + (o.get('zh') or '')).lower()
                left = 'left' in label or 'left' in label
                right = 'right' in label or 'right' in label
                if left != right:
                    return 1 if (dx < 0) == left else -1
    return None

def _deg_per_count(vocab):
    """Degrees per mouse count from the game's own semantic options ("turn left 10 degrees" with dx -152), or None."""
    if vocab is None:
        return None
    best = None
    for h in vocab.heads:
        for o in h['options']:
            dx, text = (o.get('dx', 0), o.get('zh') or '')
            m = _DEG.search(text)
            if dx and m and ('lock' not in text) and ('look down' not in text) and ('look up' not in text):
                rate = float(m.group(1)) / abs(dx)
                if best is None or abs(dx) > best[0]:
                    best = (abs(dx), rate)
    return best[1] if best else None

def _mouse_line(game_data, vocab, lang):
    m = game_data.get('mouse', {})
    kind, zh = (m.get('kind', 'none'), lang == 'zh')
    extra = _t(m, lang)
    if kind == 'none':
        return '- Mouse: not used in this game; write X Y Z as 0 0 0.'
    if kind == 'pointer':
        base = '- Mouse: X/Y move the crosshair; X = 1000 moves it one full screen width to the right, Y = 1000 one full screen height down.'
        return base + (' ' + _sentence(extra, lang) if extra else '')
    deg = _deg_per_count(vocab) or m.get('deg_per_count_x')
    sign = _camera_sign(vocab)
    if deg:
        x = round(10 / float(deg))
        right = (sign or 1) > 0
        base = f"- Mouse: X/Y turn the view; X = {x} turns about 10° to the {('right' if right else 'left')}" + (' (note: in this game X > 0 turns LEFT)' if not right else '') + '.'
    else:
        base = '- Mouse: X/Y turn the view.'
    return base + (' ' + _sentence(extra, lang) if extra else '')

def game_block(game, fmt, lang):
    data, zh = (load_game(game), lang == 'zh')
    vocab = None
    if fmt == 'semantic' or data.get('mouse', {}).get('kind') == 'camera':
        try:
            vocab = Vocabulary.for_game(game)
        except (FileNotFoundError, ValueError):
            if fmt == 'semantic':
                raise
    lines = ['**Game**']
    for key in ('view', 'goal'):
        if data.get(key):
            lines.append(_sentence(_t(data[key], lang), lang))
    if fmt == 'lumine':
        lines.append('**Controls** (only these keys do anything in this game)')
        for k in data['keys']:
            name = lumine.FROM_BOSSBENCH.get(k['key'], k['key'])
            lines.append(f'- `{name}`: {_t(k, lang)}')
        lines.append(_mouse_line(data, vocab, lang))
    notes = [n for n in data.get('notes', []) if _t(n, lang)]
    if notes:
        lines.append('**Mechanics**')
        lines += [f'- {_sentence(_t(n, lang), lang)}' for n in notes]
    return '\n'.join(lines)

def boss_block(game, case_id, lang, counters=False):
    s, zh = (scene(load_game(game), case_id), lang == 'zh')
    lines = [f"**Boss: {_t(s['boss'], lang)}**"]
    if s.get('summary'):
        lines.append(_sentence(_t(s['summary'], lang), lang))
    lines.append('Moves:')
    for i, mv in enumerate(s['moves'], 1):
        body = ' '.join((x for x in (_sentence(_t(mv.get('tell'), lang), lang), _sentence(_t(mv.get('effect'), lang), lang)) if x))
        tag = _t(mv.get('only'), lang)
        name = f"{_t(mv['name'], lang)} [{tag}]" if tag else _t(mv['name'], lang)
        line = f'{i}. {name}: {body}'
        if counters and _t(mv.get('counter'), lang):
            line += f" How to deal with it: {_sentence(_t(mv['counter'], lang), lang)}"
        lines.append(line)
    phases = [p for p in s.get('phases', []) if _t(p, lang)]
    if phases:
        lines.append('Phases:')
        lines += [f'- {_sentence(_t(p, lang), lang)}' for p in phases]
    return '\n'.join(lines)
FOOTER = {'zh': 'Health bars and other status are shown on the screen; read them yourself.', 'en': 'Health bars and other status are shown on the screen; read them yourself.'}

def parts(game, case_id, fmt='lumine', lang='en', *, thinking=False, counters=False):
    """The prompt as ordered parts; system_prompt() joins them with SEP. The HTML page uses the same parts."""
    if fmt not in FORMATS or lang not in LANGS:
        raise ValueError(f'fmt must be one of {FORMATS}, lang one of {LANGS}')
    data = load_game(game)
    intro = _INTRO[fmt, lang].format(title=_t(data['title'], lang))
    if fmt == 'lumine':
        fmt_text = _LUMINE_FORMAT[lang]
    else:
        fmt_text = _semantic_format(data, Vocabulary.for_game(game), lang)
    if thinking:
        fmt_text += '\n' + _THINKING[fmt, lang]
    return [intro + '\n' + fmt_text, game_block(game, fmt, lang), boss_block(game, case_id, lang, counters), FOOTER[lang]]

def system_prompt(game, case_id, fmt='lumine', lang='en', *, thinking=False, counters=False):
    return SEP.join(parts(game, case_id, fmt, lang, thinking=thinking, counters=counters))

def lumine_keys(game):
    """Lumine key names this game binds (for the random baseline and the docs)."""
    return [lumine.FROM_BOSSBENCH[k['key']] for k in load_game(game)['keys'] if k['key'] in lumine.FROM_BOSSBENCH]

def main(argv=None):
    """python -m bossbench_llm.prompting --case terraria/king_slime --format lumine --lang zh [--thinking] [--counters]"""
    import argparse
    import sys
    from bossbench.scope import POLICY_CASES
    p = argparse.ArgumentParser(prog='python -m bossbench_llm.prompting', description='print a system prompt')
    p.add_argument('--case', required=True)
    p.add_argument('--format', choices=FORMATS, default='lumine')
    p.add_argument('--lang', choices=LANGS, default='en')
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--counters', action='store_true')
    args = p.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    game = POLICY_CASES[args.case]['game']
    print(system_prompt(game, args.case, args.format, args.lang, thinking=args.thinking, counters=args.counters))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
