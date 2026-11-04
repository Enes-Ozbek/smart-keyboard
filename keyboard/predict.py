"""Shared next-word prediction for every model.

All four models predict sub-word pieces, not whole words.  `predict_next_words`
turns any of them into a keyboard: it runs a small beam search that glues
pieces into words, and scores each word as

    P(word) = P(piece_1) * P(piece_2 | piece_1) * ... * P(next piece starts a new word)

Every model implements the small `LanguageModel` interface below, so the
n-gram models, the feedforward network and GPT-2 (FP32 and INT8) are compared
with exactly the same decoding and the same test positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .text import normalize, split_partial


class LanguageModel:
    """What a model must provide to be used by the keyboard and the evaluation."""

    name: str = "model"
    precision: str = "FP32"
    language: str = "en"
    word_start: np.ndarray   # bool [V]: piece can start a word (letters only)
    is_cont: np.ndarray      # bool [V]: piece attaches to the previous piece
    cont_word: np.ndarray    # bool [V]: continuation piece made of letters
    piece_text: list[str]    # piece -> the letters it adds to the word

    def context_ids(self, text: str) -> list[int]:
        """Token ids of the (normalized) text before the next word."""
        raise NotImplementedError

    def logprobs(self, batch: list[list[int]]) -> np.ndarray:
        """Natural-log next-piece distributions, shape [len(batch), V].
        All sequences in a batch have the same length."""
        raise NotImplementedError

    def test_nll(self, paragraphs: list[str]) -> tuple[float, int]:
        """Total negative log-likelihood (nats) of the paragraphs and token count."""
        raise NotImplementedError

    def size_mb(self) -> float:
        raise NotImplementedError


@dataclass
class _Beam:
    pieces: list[int]
    word: str
    score: float


def _compatible(word: str, prefix: str) -> bool:
    return word.startswith(prefix) or prefix.startswith(word)


def predict_next_words(lm: LanguageModel, text: str, k: int = 3, beam: int = 8,
                       max_pieces: int = 4) -> list[tuple[str, float]]:
    """Top-k next words (with probabilities) after `text`.

    If the text ends in the middle of a word ("the quick bro"), only words
    starting with that prefix are returned, so the keyboard completes it.
    """
    context, partial = split_partial(text)
    partial = normalize(partial, lm.language)
    ctx = lm.context_ids(normalize(context, lm.language))

    first = lm.logprobs([ctx])[0]
    candidates = np.flatnonzero(lm.word_start)
    if partial:
        candidates = np.array([i for i in candidates if _compatible(lm.piece_text[i], partial)],
                              dtype=np.int64)
        if candidates.size == 0:
            return []
    best = candidates[np.argsort(-first[candidates])[:beam]]
    beams = [_Beam([int(t)], lm.piece_text[t], float(first[t])) for t in best]

    finished: dict[str, float] = {}
    cont_ids = np.flatnonzero(lm.cont_word)
    for depth in range(max_pieces):
        dists = lm.logprobs([ctx + b.pieces for b in beams])
        extended: list[_Beam] = []
        for b, dist in zip(beams, dists):
            # probability that the word ends here = next piece is NOT a continuation
            p_end = np.logaddexp.reduce(dist[~lm.is_cont])
            if b.word.startswith(partial) and b.word != partial:   # typed word itself is no help
                finished[b.word] = np.logaddexp(finished.get(b.word, -np.inf), b.score + p_end)
            if depth + 1 < max_pieces and cont_ids.size:
                top = cont_ids[np.argsort(-dist[cont_ids])[:beam]]
                for t in top:
                    word = b.word + lm.piece_text[t]
                    if not partial or _compatible(word, partial):
                        extended.append(_Beam(b.pieces + [int(t)], word, b.score + float(dist[t])))
        if not extended:
            break
        extended.sort(key=lambda x: -x.score)
        beams = extended[:beam]
        # longer words can only get less likely: stop when no beam can enter the top-k
        if len(finished) >= k:
            kth = sorted(finished.values(), reverse=True)[k - 1]
            beams = [b for b in beams if b.score > kth]
            if not beams:
                break

    ranked = sorted(finished.items(), key=lambda kv: -kv[1])[:k]
    return [(word, float(np.exp(score))) for word, score in ranked]
