"""Step 2 (area B): a BPE tokenizer trained on our own training split.

The n-gram models and the feedforward network work on these BPE pieces, so
they have an open vocabulary (no unknown words).  A word is split into a
first piece and continuation pieces marked with "##", e.g.
"keyboards" -> ["key", "##board", "##s"].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import Config
from .text import is_word

UNK, BOS = "[UNK]", "<s>"      # BOS marks the start of every paragraph


def bpe_path(cfg: Config) -> Path:
    return cfg.artifacts / "bpe" / "tokenizer.json"


def train_bpe(cfg: Config) -> Path:
    tokenizer = Tokenizer(models.BPE(unk_token=UNK, continuing_subword_prefix="##"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()   # \w+|[^\w\s]+, like text.words
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.bpe_vocab_size,
        min_frequency=2,
        special_tokens=[UNK, BOS],
        continuing_subword_prefix="##",
    )
    tokenizer.train([str(cfg.data / "train.txt")], trainer)
    path = bpe_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    print(f"BPE: {tokenizer.get_vocab_size()} pieces -> {path}")
    return path


class BPE:
    """Thin wrapper used by the n-gram and feedforward models."""

    def __init__(self, path: Path):
        self.tok = Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()
        if self.vocab_size >= 65536:
            raise ValueError("vocab must fit in 16 bits for the n-gram tables")
        self.bos_id = self.tok.token_to_id(BOS)
        pieces = [self.tok.id_to_token(i) for i in range(self.vocab_size)]
        cont = [p.startswith("##") for p in pieces]
        # masks used by the shared next-word decoder (keyboard/predict.py)
        self.is_cont = np.array(cont)                                        # attaches to previous piece
        self.cont_word = np.array([c and is_word(p[2:]) for p, c in zip(pieces, cont)])
        self.word_start = np.array([(not c) and is_word(p) for p, c in zip(pieces, cont)])
        self.piece_text = [p[2:] if c else p for p, c in zip(pieces, cont)]

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_paragraphs(self, paragraphs: list[str]) -> np.ndarray:
        """One token stream: <s> paragraph <s> paragraph ..."""
        ids: list[int] = []
        for enc in self.tok.encode_batch(paragraphs):
            ids.append(self.bos_id)
            ids.extend(enc.ids)
        return np.array(ids, dtype=np.int64)
