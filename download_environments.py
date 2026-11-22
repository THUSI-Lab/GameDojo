"""Download the matching game runtimes into a GameDojo code checkout."""
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download

ENVIRONMENT_REPO = "THUSI-Lab/GameDojo"
ENVIRONMENT_REVISION = "f6e5d7408cd9352fcf8bd43afc2a638b58212e16"
BACKEND = "environments/games/prototype/bossbench"
ENVIRONMENT_FILES = [
    "environments/games/web/**",
    "environments/THIRD_PARTY_NOTICES.md",
    *(f"{BACKEND}/{file}" for file in (
        "mc4d.py", "mc_scripted.py", "mcskin.py", "play_server.py", "rig.py",
    )),
]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=ENVIRONMENT_REPO, help="Hugging Face dataset repo ID")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=ENVIRONMENT_REVISION,
        allow_patterns=ENVIRONMENT_FILES,
        local_dir=root,
    )
    print(f"Installed GameDojo environments in {root / 'environments'}")


if __name__ == "__main__":
    main()
