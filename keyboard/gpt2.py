"""Step 5 (area C): fine-tune GPT-2 small on the training split.

GPT-2 is a decoder-only Transformer (masked self-attention) with its own
byte-level BPE tokenizer.  We continue training it as a causal language model
on our corpus: chunks of `gpt2_block_size` tokens, AdamW, one epoch.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .predict import LanguageModel
from .text import is_word


def finetuned_dir(cfg: Config) -> Path:
    return cfg.artifacts / "gpt2-finetuned"


def _token_blocks(tokenizer, paragraphs: list[str], block: int) -> torch.Tensor:
    ids: list[int] = []
    for enc in tokenizer(paragraphs, add_special_tokens=False)["input_ids"]:
        ids.extend(enc + tokenizer.encode("\n"))
    n = len(ids) // block
    return torch.tensor(ids[:n * block]).view(n, block)


@torch.no_grad()
def _blocks_ppl(model, blocks: torch.Tensor, device: str, batch: int = 8) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(blocks), batch):
        x = blocks[i:i + batch].to(device)
        logits = model(x).logits[:, :-1]
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                 x[:, 1:].reshape(-1), reduction="sum").item()
        count += x[:, 1:].numel()
    return math.exp(total / count)


def finetune_gpt2(cfg: Config) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    from .data import read_split

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.gpt2_base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.gpt2_base_model).to(device)

    train = _token_blocks(tokenizer, read_split(cfg, "train"), cfg.gpt2_block_size)
    if cfg.gpt2_max_train_tokens:
        train = train[: max(1, cfg.gpt2_max_train_tokens // cfg.gpt2_block_size)]
    valid = _token_blocks(tokenizer, read_split(cfg, "valid"), cfg.gpt2_block_size)
    print(f"GPT-2: base {cfg.gpt2_base_model}, {train.numel():,} training tokens, device={device}")
    before = _blocks_ppl(model, valid, device)
    print(f"  valid ppl (per GPT-2 token) before fine-tuning: {before:.1f}")

    steps = cfg.gpt2_epochs * math.ceil(len(train) / cfg.gpt2_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.gpt2_lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0, step = time.time(), 0
    for epoch in range(cfg.gpt2_epochs):
        model.train()
        order = torch.randperm(len(train))
        for i in range(0, len(train), cfg.gpt2_batch_size):
            x = train[order[i:i + cfg.gpt2_batch_size]].to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(x, labels=x).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 100 == 0 or step == steps:
                print(f"  step {step}/{steps}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    after = _blocks_ppl(model, valid, device)
    print(f"  valid ppl (per GPT-2 token) after fine-tuning: {after:.1f}")

    out = finetuned_dir(cfg)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "finetune_log.json").write_text(json.dumps(
        {"base_model": cfg.gpt2_base_model, "train_tokens": int(train.numel()),
         "epochs": cfg.gpt2_epochs, "valid_ppl_before": before, "valid_ppl_after": after,
         "seconds": round(time.time() - t0, 1), "device": device}, indent=2))
    print(f"GPT-2: saved -> {out}")
    return out


class GPT2LM(LanguageModel):
    """Keyboard wrapper for GPT-2 (FP32 or INT8 model object)."""

    def __init__(self, model, tokenizer, name: str, precision: str, files: list[Path],
                 cfg: Config, device: str = "cpu"):
        self.model, self.tok = model.eval(), tokenizer
        self.name, self.precision, self.files = name, precision, files
        self.device, self.language = device, cfg.language
        self.context_len, self.cfg = cfg.context_tokens, cfg
        V = model.config.vocab_size              # size of the logits vector
        text = [tokenizer.decode([i]) if i < len(tokenizer) else "" for i in range(V)]
        # a piece starting with a space starts a new word; letters/digits glue to the word
        self.word_start = np.array([t.startswith(" ") and is_word(t[1:]) for t in text])
        self.is_cont = np.array([bool(t) and t[0].isalnum() for t in text])
        self.cont_word = np.array([is_word(t) for t in text])
        self.piece_text = [t[1:] if t.startswith(" ") else t for t in text]
        self.eos = tokenizer.eos_token_id
        # key/value cache of the last context, reused while the beam search extends it
        self._ctx: list[int] | None = None
        self._past = None
        self.use_cache = True

    def context_ids(self, text: str) -> list[int]:
        ids = self.tok.encode(text.rstrip()) if text.strip() else []
        if len(ids) < self.context_len:
            return [self.eos] + ids            # the text starts here, like a new document
        return ids[-self.context_len:]

    @torch.no_grad()
    def logprobs(self, batch: list[list[int]]) -> np.ndarray:
        """Next-piece log-probabilities.

        The beam search first asks for the context alone, then for the context plus a few
        pieces.  We keep the attention key/value cache of the context, so the later calls
        only run the new pieces through the network instead of the whole context again.
        """
        ctx = self._ctx
        if (self.use_cache and ctx is not None and self._past is not None
                and all(len(seq) > len(ctx) and seq[:len(ctx)] == ctx for seq in batch)):
            try:
                past = copy.deepcopy(self._past)
                past.batch_repeat_interleave(len(batch))      # one copy of the context per beam
                new = torch.tensor([seq[len(ctx):] for seq in batch], device=self.device)
                logits = self.model(new, past_key_values=past, use_cache=True).logits[:, -1, :]
                return F.log_softmax(logits.float(), dim=-1).cpu().numpy()
            except (AttributeError, TypeError, RuntimeError):
                self.use_cache = False                        # older library: recompute instead
        x = torch.tensor(batch, device=self.device)
        single = len(batch) == 1 and self.use_cache
        out = self.model(x, use_cache=single)
        if single:
            self._ctx, self._past = list(batch[0]), out.past_key_values
        return F.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def test_nll(self, paragraphs: list[str]) -> tuple[float, int]:
        """Sliding-window NLL: each token is scored once, with up to `ppl_window` tokens of context."""
        ids: list[int] = [self.eos]
        for enc in self.tok(paragraphs, add_special_tokens=False)["input_ids"]:
            ids.extend(enc + self.tok.encode("\n"))
        ids_t = torch.tensor(ids)
        W, S = self.cfg.ppl_window, self.cfg.ppl_stride
        total, scored, prev_end = 0.0, 0, 1
        for start in range(0, len(ids), S):
            end = min(start + W, len(ids))
            x = ids_t[start:end].unsqueeze(0).to(self.device)
            logits = self.model(x).logits[0, :-1].float()
            targets = x[0, 1:]
            new_from = prev_end - start - 1            # only score tokens not scored before
            total += F.cross_entropy(logits[new_from:], targets[new_from:], reduction="sum").item()
            scored += len(targets) - new_from
            prev_end = end
            if end == len(ids):
                break
        return total, scored

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.files) / 1e6


def load_finetuned(cfg: Config, device: str = "cpu") -> GPT2LM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = finetuned_dir(cfg)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
    return GPT2LM(model, tokenizer, "GPT-2 fine-tuned", "FP32", weights, cfg, device)
