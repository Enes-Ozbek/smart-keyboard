"""All settings of the project in one place.

Every step (data prep, training, evaluation, app) reads its settings from a
`Config` object. The defaults below are the ones used for the homework runs;
the command line (see `keyboard/cli.py`) can override any of them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---------------- data ----------------
    language: str = "en"            # "en" or "tr" (only changes lower-casing rules)
    source: str = "wikitext2"       # "wikitext2" | "wikipedia" | "textfile"
    textfile: str = ""              # path to a .txt corpus when source == "textfile"
    max_words: int = 1_500_000      # how many words to take when source == "wikipedia"
    data_dir: str = str(ROOT / "data")
    artifacts_dir: str = str(ROOT / "artifacts")
    results_dir: str = str(ROOT / "results")

    # ---------------- B: BPE tokenizer ----------------
    bpe_vocab_size: int = 8000

    # ---------------- B: n-gram language models ----------------
    ngram_order: int = 4            # 4-gram over BPE tokens

    # ---------------- A: feedforward neural language model ----------------
    ffnn_context: int = 5           # previous tokens the network sees
    ffnn_embed_dim: int = 128
    ffnn_hidden_dim: int = 512
    ffnn_dropout: float = 0.3
    ffnn_epochs: int = 10
    ffnn_batch_size: int = 512
    ffnn_lr: float = 0.1            # plain SGD with momentum
    ffnn_momentum: float = 0.9
    ffnn_max_train_tokens: int = 0  # 0 = use the whole training split

    # ---------------- C: GPT-2 fine-tuning ----------------
    gpt2_base_model: str = "gpt2"   # Hugging Face id (GPT-2 small, 124M parameters)
    gpt2_block_size: int = 256
    gpt2_batch_size: int = 8
    gpt2_epochs: int = 1
    gpt2_lr: float = 5e-5
    gpt2_max_train_tokens: int = 0  # 0 = use the whole training split

    # ---------------- next-word prediction ----------------
    context_tokens: int = 48        # GPT-2 context length used for suggestions
    beam_size: int = 8              # candidate words kept while assembling sub-word pieces
    max_word_pieces: int = 4        # longest word, in sub-word pieces
    suggestions: int = 3            # top-3 next words

    # ---------------- evaluation ----------------
    eval_positions: int = 1000      # test positions for top-3 accuracy; each call is also
                                    # timed, so latency is averaged over these runs (R4: >= 100)
    ppl_max_words: int = 0          # 0 = perplexity on the whole test split
    ppl_window: int = 512           # GPT-2 sliding-window perplexity
    ppl_stride: int = 256
    seed: int = 42

    # ---------------- weights hosting (R7: no weights in git) ----------------
    models_repo: str = ""           # e.g. "your-hf-user/smart-keyboard-models"

    # ------------------------------------------------------------------
    @property
    def data(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def results(self) -> Path:
        return Path(self.results_dir)

    def settings_path(self) -> Path:
        return self.artifacts / "settings.json"

    def save(self) -> None:
        """Remember the settings so later steps (and the app) use the same ones."""
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.settings_path().write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, artifacts_dir: str | None = None, **overrides) -> "Config":
        """Defaults, then saved settings (if any), then explicit overrides."""
        cfg = cls()
        if artifacts_dir:
            cfg.artifacts_dir = artifacts_dir
        path = cfg.settings_path()
        if path.exists():
            saved = json.loads(path.read_text())
            known = {f.name for f in fields(cls)}
            for key, value in saved.items():
                # folder paths stay local: settings may come from another machine
                if key in known and not key.endswith("_dir"):
                    setattr(cfg, key, value)
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg
