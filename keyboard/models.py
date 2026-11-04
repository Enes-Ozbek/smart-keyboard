"""Load the trained models of the ladder in a fixed order."""
from __future__ import annotations

from .config import Config
from .predict import LanguageModel

# names used on the command line (`--models`) and in the results table
LADDER = ["ngram-addone", "ngram-kn", "ffnn", "gpt2-fp32", "gpt2-int8"]


def load_model(cfg: Config, key: str, device: str = "cpu") -> LanguageModel:
    if key in ("ngram-addone", "ngram-kn"):
        from .bpe import BPE, bpe_path
        from .ngram import NGramCounts, NGramLM, ngram_path

        counts = NGramCounts.load(ngram_path(cfg))
        mode = "addone" if key == "ngram-addone" else "kn"
        return NGramLM(counts, BPE(bpe_path(cfg)), mode, ngram_path(cfg), cfg.language)
    if key == "ffnn":
        from .ffnn import FFNNLM

        return FFNNLM(cfg, device)
    if key == "gpt2-fp32":
        from .gpt2 import load_finetuned

        return load_finetuned(cfg, device)
    if key == "gpt2-int8":
        from .quantize import load_int8

        return load_int8(cfg)                 # dynamic INT8 runs on the CPU only
    raise ValueError(f"unknown model {key!r}; choose from {LADDER}")


def load_models(cfg: Config, keys: list[str] | None = None, device: str = "cpu"
                ) -> dict[str, LanguageModel]:
    models = {}
    for key in keys or LADDER:
        models[key] = load_model(cfg, key, device)
        print(f"loaded {key}")
    return models
