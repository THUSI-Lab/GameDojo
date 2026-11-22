import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'games/prototype'))
from bossbench_llm.run import main
defaults=['--policy','chat','--format','lumine','--lang','en','--base-url','http://127.0.0.1:8000/v1','--model','gamedojo-4b','--history','2','--image-size','854x480','--seeds','1','--policy-seed','1','--temperature','1','--top-p','1','--max-tokens','256','--extra-body',json.dumps({'skip_special_tokens':False,'chat_template_kwargs':{'enable_thinking':False}})]
sys.exit(main(defaults+sys.argv[1:]))
