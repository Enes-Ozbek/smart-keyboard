"""Keep model weights out of git (R7): store them on the Hugging Face Hub.

    huggingface-cli login                                  # once, with a write token
    python -m keyboard.cli upload --models-repo USER/smart-keyboard-models
    python -m keyboard.cli download --models-repo USER/smart-keyboard-models
"""
from __future__ import annotations

from .config import Config


def _repo(cfg: Config) -> str:
    if not cfg.models_repo:
        raise SystemExit("set --models-repo USER/NAME (a Hugging Face model repo)")
    return cfg.models_repo


def upload(cfg: Config) -> None:
    from huggingface_hub import HfApi

    repo = _repo(cfg)
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(cfg.artifacts), repo_type="model",
                      commit_message="smart keyboard model weights")
    cfg.save()
    print(f"uploaded {cfg.artifacts} -> https://huggingface.co/{repo}")


def download(cfg: Config) -> None:
    from huggingface_hub import snapshot_download

    repo = _repo(cfg)
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(cfg.artifacts))
    print(f"downloaded https://huggingface.co/{repo} -> {cfg.artifacts}")


def ensure_artifacts(cfg: Config) -> None:
    """Used by the app: fetch the weights if they are not on disk yet."""
    if (cfg.artifacts / "gpt2-int8" / "model_int8.pt").exists():
        return
    if not cfg.models_repo:
        raise SystemExit("No trained models found in artifacts/. Train them with "
                         "`python -m keyboard.cli all`, or download them with "
                         "`python app.py --models-repo USER/NAME`.")
    download(cfg)
