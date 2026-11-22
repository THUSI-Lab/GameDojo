"""Run from the delivered package in a clean Python environment. No model calls."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--game', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'games/prototype'))
import bossbench
from bossbench.scope import BENCH_CASES
from bossbench_llm import lumine, prompting
cases = sorted(k for k,v in BENCH_CASES.items() if v['game']==args.game)
report = {'game':args.game,'python':sys.version,'checks':[], 'started':time.time(), 'test':'reset + RGB + one valid 200ms action; not a win/performance evaluation'}
for case in cases:
    for tier in ('easy','normal','hard'):
        env = None
        start = time.time()
        try:
            env = bossbench.make_policy_env(case)
            obs, info = env.reset(seed=1, options={'tier':tier})
            assert obs.shape == (480,854,3), obs.shape
            assert str(obs.dtype) == 'uint8', obs.dtype
            kind = prompting.mouse_kind(args.game)
            action = lumine.from_packet(bossbench.NOOP, mouse_kind=kind)
            parsed = lumine.parse('<think>\n\n</think>\n\n'+action, mouse_kind=kind)
            assert parsed.ok, parsed
            nxt, reward, terminated, truncated, step_info = env.step(parsed.packet)
            assert nxt.shape == (480,854,3)
            assert prompting.system_prompt(args.game, case, 'lumine', 'en')
            report['checks'].append({'case':case,'tier':tier,'passed':True,'rgb_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),'reward':float(reward),'seconds':time.time()-start})
        except Exception:
            report['checks'].append({'case':case,'tier':tier,'passed':False,'error':traceback.format_exc(),'seconds':time.time()-start})
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass
        Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
report['passed'] = len(report['checks']) == 18 and all(c['passed'] for c in report['checks'])
report['finished'] = time.time()
Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'game':args.game,'passed':report['passed'],'checks':len(report['checks'])}), flush=True)
sys.exit(0 if report['passed'] else 1)
