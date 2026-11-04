"""Step 4 (area A): a feedforward neural language model (Bengio et al., 2003).

    previous C pieces -> embedding layer -> concatenate -> hidden layer (tanh)
                      -> dropout -> softmax over the BPE vocabulary

Trained from scratch with plain SGD (momentum) and backpropagation on the
cross-entropy of the next piece.  Uses the same BPE pieces as the n-gram models.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bpe import BPE, bpe_path
from .config import Config
from .predict import LanguageModel


class FeedForwardLM(nn.Module):
    def __init__(self, vocab_size: int, context: int, embed_dim: int, hidden_dim: int,
                 dropout: float):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(context * embed_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:   # [B, C] -> logits [B, V]
        x = self.embed(ctx).flatten(1)
        h = torch.tanh(self.hidden(x))
        return self.out(self.drop(h))


def _windows(stream: np.ndarray, positions: np.ndarray, context: int) -> np.ndarray:
    """Context windows for target positions in a stream padded with `context` start symbols."""
    return stream[positions[:, None] - context + np.arange(context)[None, :]]


def _pad(ids: np.ndarray, bos_id: int, context: int) -> np.ndarray:
    return np.concatenate([np.full(context, bos_id), ids]).astype(np.int64)


@torch.no_grad()
def stream_nll(model: FeedForwardLM, ids: np.ndarray, bos_id: int, device: str,
               batch_size: int = 4096) -> tuple[float, int]:
    """Total NLL of every token of `ids` except the first (which is <s>)."""
    model.eval()
    C = model.context
    stream = _pad(ids, bos_id, C)
    targets = np.arange(C + 1, len(stream))
    total = 0.0
    for i in range(0, len(targets), batch_size):
        pos = targets[i:i + batch_size]
        x = torch.from_numpy(_windows(stream, pos, C)).to(device)
        y = torch.from_numpy(stream[pos]).to(device)
        total += F.cross_entropy(model(x), y, reduction="sum").item()
    return total, len(targets)


def ffnn_dir(cfg: Config) -> Path:
    return cfg.artifacts / "ffnn"


def train_ffnn(cfg: Config) -> Path:
    from .data import read_split

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe = BPE(bpe_path(cfg))
    train_ids = bpe.encode_paragraphs(read_split(cfg, "train"))
    if cfg.ffnn_max_train_tokens:
        train_ids = train_ids[:cfg.ffnn_max_train_tokens]
    valid_ids = bpe.encode_paragraphs(read_split(cfg, "valid"))

    hparams = dict(vocab_size=bpe.vocab_size, context=cfg.ffnn_context,
                   embed_dim=cfg.ffnn_embed_dim, hidden_dim=cfg.ffnn_hidden_dim,
                   dropout=cfg.ffnn_dropout)
    model = FeedForwardLM(**hparams).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.ffnn_lr, momentum=cfg.ffnn_momentum)

    C = cfg.ffnn_context
    stream = _pad(train_ids, bpe.bos_id, C)
    targets = np.arange(C + 1, len(stream))
    out_dir = ffnn_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ppl, lr = math.inf, cfg.ffnn_lr
    history = []
    print(f"FFNN: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(targets):,} training windows, device={device}")

    for epoch in range(1, cfg.ffnn_epochs + 1):
        model.train()
        rng.shuffle(targets)
        t0, running, seen = time.time(), 0.0, 0
        for i in range(0, len(targets), cfg.ffnn_batch_size):
            pos = targets[i:i + cfg.ffnn_batch_size]
            x = torch.from_numpy(_windows(stream, pos, C)).to(device)
            y = torch.from_numpy(stream[pos]).to(device)
            loss = F.cross_entropy(model(x), y)       # forward pass
            optimizer.zero_grad()
            loss.backward()                            # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                           # SGD update
            running += loss.item() * len(pos)
            seen += len(pos)
            step = i // cfg.ffnn_batch_size + 1
            if step % 500 == 0:
                print(f"    epoch {epoch}: {seen:,}/{len(targets):,} windows, "
                      f"train ppl {math.exp(running / seen):.1f} ({time.time() - t0:.0f}s)", flush=True)
        nll, n = stream_nll(model, valid_ids, bpe.bos_id, device)
        valid_ppl = math.exp(nll / n)
        train_ppl = math.exp(running / seen)
        history.append({"epoch": epoch, "lr": lr, "train_ppl": train_ppl, "valid_ppl": valid_ppl,
                        "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {epoch}: train ppl {train_ppl:.1f}  valid ppl {valid_ppl:.1f}  "
              f"lr {lr:g}  ({time.time() - t0:.0f}s)")
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            torch.save(model.state_dict(), out_dir / "ffnn.pt")
        else:                                          # no improvement: halve the learning rate
            lr /= 2
            for group in optimizer.param_groups:
                group["lr"] = lr

    (out_dir / "ffnn_config.json").write_text(json.dumps(hparams, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"FFNN: best valid ppl (per BPE piece) {best_ppl:.1f} -> {out_dir}")
    return out_dir


class FFNNLM(LanguageModel):
    def __init__(self, cfg: Config, device: str = "cpu"):
        self.bpe = BPE(bpe_path(cfg))
        self.dir = ffnn_dir(cfg)
        hparams = json.loads((self.dir / "ffnn_config.json").read_text())
        self.model = FeedForwardLM(**hparams)
        self.model.load_state_dict(torch.load(self.dir / "ffnn.pt", map_location="cpu"))
        self.model.to(device).eval()
        self.device, self.language = device, cfg.language
        self.name, self.precision = "Feedforward NN", "FP32"
        self.word_start, self.is_cont = self.bpe.word_start, self.bpe.is_cont
        self.cont_word, self.piece_text = self.bpe.cont_word, self.bpe.piece_text

    def context_ids(self, text: str) -> list[int]:
        return [self.bpe.bos_id] * self.model.context + self.bpe.encode(text)

    @torch.no_grad()
    def logprobs(self, batch: list[list[int]]) -> np.ndarray:
        C = self.model.context
        x = torch.tensor([seq[-C:] for seq in batch], device=self.device)
        return F.log_softmax(self.model(x), dim=-1).cpu().numpy()

    def test_nll(self, paragraphs: list[str]) -> tuple[float, int]:
        ids = self.bpe.encode_paragraphs(paragraphs)
        return stream_nll(self.model, ids, self.bpe.bos_id, self.device)

    def size_mb(self) -> float:
        return (self.dir / "ffnn.pt").stat().st_size / 1e6
