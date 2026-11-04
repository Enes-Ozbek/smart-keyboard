"""Step 1: download the corpus, normalize it and write train/valid/test files.

Output: data/train.txt, data/valid.txt, data/test.txt (one paragraph per line)
and data/stats.json with the size of each split.

Sources
- wikitext2 (default): WikiText-2 raw (Merity et al., 2016), English Wikipedia
  articles with official train/validation/test splits, via Hugging Face datasets.
- wikipedia: the first `max_words` words of a Wikipedia dump (any language),
  split 90/5/5 by article.
- textfile: your own .txt file, split 90/5/5 by chunks of 20 paragraphs.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .config import Config
from .text import detokenize_wikitext, normalize, words

SPLITS = ("train", "valid", "test")


def _split_documents(docs: list[list[str]], seed: int) -> dict[str, list[str]]:
    """Shuffle documents with a fixed seed and split 90/5/5 (no document is shared)."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    n_valid = n_test = max(1, len(docs) // 20)
    parts = {
        "test": order[:n_test],
        "valid": order[n_test:n_test + n_valid],
        "train": order[n_test + n_valid:],
    }
    return {split: [p for i in sorted(idx) for p in docs[i]] for split, idx in parts.items()}


def _load_wikitext2() -> dict[str, list[str]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    out = {}
    for split, hf_name in (("train", "train"), ("valid", "validation"), ("test", "test")):
        paragraphs = []
        for line in ds[hf_name]["text"]:
            line = line.strip()
            if not line or line.startswith("="):      # skip blank lines and " = Heading = "
                continue
            paragraphs.append(detokenize_wikitext(line))
        out[split] = paragraphs
    return out


def _load_wikipedia(cfg: Config) -> dict[str, list[str]]:
    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", f"20231101.{cfg.language}",
                          split="train", streaming=True)
    docs, n_words = [], 0
    for article in stream:
        paragraphs = [p.strip() for p in article["text"].split("\n") if len(p.split()) >= 5]
        if paragraphs:
            docs.append(paragraphs)
            n_words += sum(len(p.split()) for p in paragraphs)
        if n_words >= cfg.max_words:
            break
    return _split_documents(docs, cfg.seed)


def _load_textfile(cfg: Config) -> dict[str, list[str]]:
    text = Path(cfg.textfile).read_text(encoding="utf-8", errors="ignore")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 3]
    docs = [paragraphs[i:i + 20] for i in range(0, len(paragraphs), 20)]
    return _split_documents(docs, cfg.seed)


def prepare(cfg: Config) -> dict:
    if cfg.source == "wikitext2":
        raw = _load_wikitext2()
    elif cfg.source == "wikipedia":
        raw = _load_wikipedia(cfg)
    elif cfg.source == "textfile":
        raw = _load_textfile(cfg)
    else:
        raise ValueError(f"unknown source {cfg.source!r}")

    cfg.data.mkdir(parents=True, exist_ok=True)
    stats = {"source": cfg.source, "language": cfg.language}
    for split in SPLITS:
        lines = [normalize(p, cfg.language) for p in raw[split]]
        lines = [line for line in lines if line]
        (cfg.data / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[split] = {"paragraphs": len(lines),
                        "words": sum(len(words(line)) for line in lines)}
    (cfg.data / "stats.json").write_text(json.dumps(stats, indent=2))
    cfg.save()
    print(json.dumps(stats, indent=2))
    return stats


def read_split(cfg: Config, split: str) -> list[str]:
    """Paragraphs of one split, as written by `prepare`."""
    path = cfg.data / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python -m keyboard.cli prepare` first")
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]
