<p align="center">
  <img src="https://gamedojo-benchmark.github.io/assets/img/logo.webp" alt="GameDojo" width="520">
</p>

<h1 align="center">GameDojo: Benchmarking Real-Time Interactive Agents in Video Games</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2611.30771"><img src="https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper (arXiv)"></a>
  <a href="https://gamedojo-benchmark.github.io/"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/THUSI-Lab/GameDojo"><img src="https://img.shields.io/badge/Dataset%20%26%20Benchmark-ffcc4d?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Dataset &amp; Benchmark"></a>
  <a href="https://huggingface.co/THUSI-Lab/GameDojo-4B"><img src="https://img.shields.io/badge/Model-ffcc4d?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Model"></a>
  <a href="https://huggingface.co/papers/2611.30771"><img src="https://img.shields.io/badge/Daily%20Paper-232737?style=for-the-badge&amp;logo=huggingface&amp;logoColor=ffcc4d" alt="Daily Paper"></a>
</p>

GameDojo evaluates interactive agents on 120 boss encounters across 20 game environments. It provides computer-use and semantic action interfaces, as well as real-time and time-paused interaction, to study how action abstraction and time constraints affect performance.

## To-do

- [x] Release the benchmark environments, training interfaces and PPO example, and evaluation harness.
- [ ] Release the 600-hour GameDojo-Data trajectory dataset.

## Download and set up

Use Python 3.12. Install Node.js to use symbolic observations in shared web games. Minecraft RGB rendering needs an OpenGL context.

```sh
git clone https://github.com/THUSI-Lab/GameDojo.git
cd GameDojo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-games.txt 'huggingface_hub==1.30.0'
python download_environments.py
python -m playwright install chromium
```

The download script retrieves the matching game runtimes at a pinned revision from the [Hugging Face environment bundle](https://huggingface.co/datasets/THUSI-Lab/GameDojo). This GitHub checkout provides the shared Python interfaces and harness. To get the complete environment bundle directly, clone the Hugging Face dataset repository.

## Launch the games

```sh
python -m http.server 8766 --bind 127.0.0.1 --directory environments/games/web
```

Open [the local launcher](http://127.0.0.1:8766/). For Minecraft, start its Python server in a second terminal with the same virtual environment active, then open [the Minecraft page](http://127.0.0.1:8765/).

```sh
python environments/games/prototype/bossbench/play_server.py
```

The following runs one reset, RGB observation, and 200 ms action for each Terraria encounter and difficulty.

```sh
mkdir -p runs
python environments/check_games.py --game terraria_remake --out runs/terraria-check.json
```

## Train a visual policy

The example trains a single-encounter PPO policy with downsampled RGB observations, compact actions, and GameDojo's progress reward. It uses time-paused interaction for a reproducible local baseline. It provides a starting point for custom visual-policy training.

```sh
python -m pip install 'stable-baselines3==2.9.2'
python examples/ppo.py --case doom/cyberdemon --tier easy train --steps 10000 --output runs/doom-ppo.zip
python examples/ppo.py --case doom/cyberdemon --tier easy eval --checkpoint runs/doom-ppo.zip --episodes 5
```

Evaluation uses separate seeds and reports win rate and mean progress as fractions. For a custom trainer, set `PYTHONPATH=environments/games/prototype` and use `bossbench.make_training_env(case_id, action_format="compact")`. Pass `options={"tier": "easy"}` to `reset` for a fixed difficulty.

## Evaluate agents

Check the harness locally with an empty action before connecting a model.

```sh
python environments/run_model.py --case doom/cyberdemon --tier normal --policy noop --max-steps 1
```

Serve [GameDojo-4B](https://huggingface.co/THUSI-Lab/GameDojo-4B) or another vision-language model on an OpenAI-compatible chat endpoint. Use the served model name for `--model`. If the endpoint requires a key, set `OPENAI_API_KEY` in your environment.

```sh
python environments/run_model.py --case doom/cyberdemon --tier normal \
  --policy chat --base-url http://127.0.0.1:8000/v1 --model gamedojo-4b \
  --format lumine --time paused --seeds 1-3 --out runs/eval
```

Use `--format semantic` for semantic actions and `--time realtime` for inference-latency-aware interaction. Repeat `--case` to evaluate several encounters. `normal` is the benchmark's medium difficulty. Results under `--out` include per-episode outcomes and boss health reduced (`boss_damage`), plus JSON and CSV summaries. Use the [official case registry](environments/games/prototype/bossbench/scope.py) for encounter IDs.

## License

GameDojo code is available for non-commercial scientific research under the [GameDojo Non-Commercial Research License](LICENSE). The original games and their intellectual property belong to the respective original game companies and rights holders. See the [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{zhang2026gamedojo,
  title   = {GameDojo: Benchmarking Real-Time Interactive Agents in Video Games},
  author  = {Zhang, Kuan and Yang, Zhihao and Chen, Yukun and Yin, Xingyiliang and Ouyang, Mingyu and Su, Yue and Wu, Xiangnan and Li, Jiaming and Chen, Longze and Luo, Run and Chen, Zirong and Wang, Lu and Li, Yiming},
  journal = {arXiv preprint arXiv:2611.30771},
  year    = {2026}
}
```

We also invite you to explore and cite [Gamer](https://mihoyo-gamer.github.io/), [GameScaling](https://gamescaling.github.io/), and [Survey: Towards Generalist Game Agents](https://github.com/THUSI-Lab/Awesome-LFMs-Play-Games).
