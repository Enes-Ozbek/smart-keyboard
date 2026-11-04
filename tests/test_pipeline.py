"""Fast offline tests: run with `pytest -q`.

They use a small synthetic corpus and a tiny GPT-2 built from scratch, so no
download is needed.  The real runs use WikiText-2 and GPT-2 small.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest
import torch

from keyboard.config import Config
from keyboard.text import normalize, split_partial, words

SUBJECTS = ["the cat", "a dog", "the old man", "my sister", "the teacher", "a small bird",
            "the city council", "our team", "the river", "his brother"]
VERBS = ["saw", "liked", "followed", "found", "painted", "helped", "visited", "described"]
OBJECTS = ["the house", "a red car", "the green hill", "the station", "a quiet garden",
           "the history of the town", "the music hall", "a long letter", "the north bridge"]
ENDINGS = ["yesterday", "in the morning", "after the war", "with great care", "every summer",
           "near the market", "for many years"]


def synthetic_corpus(n_paragraphs: int = 600, seed: int = 0) -> str:
    rng = random.Random(seed)
    paragraphs = []
    for _ in range(n_paragraphs):
        sentences = []
        for _ in range(rng.randint(2, 5)):
            s = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(ENDINGS)}"
            sentences.append(s.capitalize() + ".")
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def tiny_gpt2(path, train_file):
    """A 2-layer GPT-2 with its own byte-level BPE, saved like a Hugging Face model."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train([str(train_file)], trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>", unk_token="<|endoftext|>")
    config = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=64, n_layer=2, n_head=2,
                        bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    GPT2LMHeadModel(config).save_pretrained(path)
    tok.save_pretrained(path)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("kb")
    corpus = root / "corpus.txt"
    corpus.write_text(synthetic_corpus())
    c = Config(source="textfile", textfile=str(corpus), data_dir=str(root / "data"),
               artifacts_dir=str(root / "artifacts"), results_dir=str(root / "results"),
               bpe_vocab_size=300, ngram_order=3, ffnn_context=3, ffnn_embed_dim=16,
               ffnn_hidden_dim=32, ffnn_epochs=2, ffnn_batch_size=256,
               gpt2_block_size=32, gpt2_batch_size=8, gpt2_lr=1e-3,
               context_tokens=16, ppl_window=64, ppl_stride=32, eval_positions=30)
    from keyboard.data import prepare

    prepare(c)
    c.gpt2_base_model = str(root / "tiny-gpt2")
    tiny_gpt2(root / "tiny-gpt2", c.data / "train.txt")
    return c


def test_text_helpers():
    assert normalize("  The  Cat\nSAT ") == "the cat sat"
    assert normalize("Iİ", "tr") == "ıi"
    assert words("don't stop, now!") == ["don", "'", "t", "stop", ",", "now", "!"]
    assert split_partial("the quick bro") == ("the quick ", "bro")
    assert split_partial("the quick ") == ("the quick ", "")
    assert split_partial("end.") == ("end.", "")


def test_insert_word():
    from app import insert_word

    assert insert_word("the quick bro", "brown") == "the quick brown "
    assert insert_word("the quick ", "brown") == "the quick brown "
    assert insert_word("end.", "the") == "end. the "


def test_prepare_splits(cfg):
    stats = json.loads((cfg.data / "stats.json").read_text())
    assert all(stats[s]["words"] > 0 for s in ("train", "valid", "test"))
    train = set((cfg.data / "train.txt").read_text().splitlines())
    test = set((cfg.data / "test.txt").read_text().splitlines())
    assert len(train & test) < len(test)          # mostly different paragraphs


def test_ngram_is_a_distribution(cfg):
    from keyboard.bpe import BPE, bpe_path, train_bpe
    from keyboard.data import read_split
    from keyboard.ngram import NGramCounts, ngram_path, train_ngram

    train_bpe(cfg)
    train_ngram(cfg)
    bpe = BPE(bpe_path(cfg))
    counts = NGramCounts.load(ngram_path(cfg))
    ids = bpe.encode_paragraphs(read_split(cfg, "test")[:20])
    rng = np.random.default_rng(0)
    for mode in ("kn", "addone"):
        lp = counts.sequence_logprobs(ids, bpe.bos_id, mode)
        dist = counts.kn_distribution if mode == "kn" else counts.addone_distribution
        padded = [bpe.bos_id] * (counts.order - 1) + ids.tolist()
        for i in rng.integers(1, len(ids), 10):
            p = dist(padded[:i + counts.order - 1])
            assert abs(p.sum() - 1) < 1e-6
            assert math.isclose(np.log(p[ids[i]]), lp[i], rel_tol=1e-6, abs_tol=1e-9)


def test_conv1d_to_linear_keeps_outputs():
    from transformers import GPT2Config, GPT2LMHeadModel

    from keyboard.quantize import conv1d_to_linear

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=16, n_embd=32, n_layer=1,
                                       n_head=2)).eval()
    x = torch.randint(0, 50, (2, 8))
    before = model(x).logits
    after = conv1d_to_linear(model)(x).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_full_pipeline(cfg):
    from keyboard.cli import run_step
    from keyboard.models import LADDER, load_model
    from keyboard.predict import predict_next_words

    for step in ("bpe", "ngram", "ffnn", "gpt2", "quantize", "evaluate"):
        run_step(step, cfg)

    results = json.loads((cfg.results / "results.json").read_text())
    assert set(results["models"]) == set(LADDER)
    for r in results["models"].values():
        assert math.isfinite(r["perplexity_per_word"]) and r["size_mb"] > 0
        assert r["latency_runs"] == cfg.eval_positions
    m = results["models"]
    assert m["ngram-kn"]["perplexity_per_word"] < m["ngram-addone"]["perplexity_per_word"]
    assert m["gpt2-int8"]["size_mb"] < m["gpt2-fp32"]["size_mb"]
    assert (cfg.results / "results.md").read_text().startswith("| Model |")

    # the key/value cache used during the beam search must not change the result
    lm = load_model(cfg, "gpt2-fp32")
    ctx = lm.context_ids("the old man visited the")
    lm.logprobs([ctx])
    beams = [ctx + [t] for t in range(5)]
    cached = lm.logprobs(beams)
    lm.use_cache = False
    assert np.allclose(cached, lm.logprobs(beams), atol=1e-4)

    for key in LADDER:
        lm = load_model(cfg, key)
        pred = predict_next_words(lm, "The old man visited the ", 3)
        assert 1 <= len(pred) <= 3 and all(w.isalpha() for w, _ in pred)
        completions = predict_next_words(lm, "the old man visited the sta", 3)
        assert all(w.startswith("sta") for w, _ in completions)
