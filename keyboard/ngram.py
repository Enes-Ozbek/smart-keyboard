"""Step 3 (area B, classical baseline): n-gram language models over BPE pieces.

Two smoothing methods on the same counts (Jurafsky & Martin, ch. 3):

* add-one (Laplace):  P(w | h) = (c(h w) + 1) / (c(h) + V)
* interpolated Kneser-Ney:
      P(w | h) = max(c(h w) - d, 0) / c(h)  +  lambda(h) * P_KN(w | h')
      lambda(h) = d * |{w : c(h w) > 0}| / c(h)
  where h' drops the oldest token of h, and every lower order uses
  *continuation counts* (in how many different contexts a sequence appears)
  instead of raw counts.

Storage: every n-gram of token ids is packed into one 64-bit integer
(16 bits per token) and kept in a sorted numpy array, so a lookup is a
binary search (`np.searchsorted`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .bpe import BPE, bpe_path
from .config import Config
from .predict import LanguageModel

BITS = 16
LOW = np.uint64((1 << BITS) - 1)


def pack(grams: np.ndarray) -> np.ndarray:
    """[M, n] token ids -> [M] uint64 codes (first token in the highest bits)."""
    grams = np.asarray(grams, dtype=np.uint64)
    if grams.ndim == 1:
        grams = grams[None, :]
    code = np.zeros(grams.shape[0], dtype=np.uint64)
    for j in range(grams.shape[1]):
        code = (code << np.uint64(BITS)) | grams[:, j]
    return code


def _discount(values: np.ndarray) -> float:
    """Kneser-Ney discount d = n1 / (n1 + 2 n2) (Chen & Goodman), n_k = #counts equal to k."""
    n1, n2 = np.sum(values == 1), np.sum(values == 2)
    if n1 == 0 or n2 == 0:
        return 0.75
    return float(np.clip(n1 / (n1 + 2 * n2), 0.1, 0.95))


class NGramCounts:
    """Count tables shared by the add-one and Kneser-Ney models."""

    def __init__(self, order: int, vocab_size: int):
        self.order, self.V = order, vocab_size
        self.codes: dict[int, np.ndarray] = {}    # n -> sorted unique n-gram codes
        self.values: dict[int, np.ndarray] = {}   # n -> raw counts (n = order) or continuation counts
        self.raw: dict[int, np.ndarray] = {}      # n -> raw counts (used by add-one)
        self.ctx: dict[int, tuple] = {}           # n -> (context codes, first row, total, distinct)
        self.discount: dict[int, float] = {}
        self.p_unigram: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, ids: np.ndarray, bos_id: int) -> "NGramCounts":
        ids = np.concatenate([np.full(self.order - 1, bos_id), ids]).astype(np.uint64)
        for n in range(1, self.order + 1):
            codes, counts = np.unique(pack(sliding_window_view(ids, n)), return_counts=True)
            self.codes[n], self.raw[n] = codes, counts.astype(np.float64)

        for n in range(1, self.order + 1):
            if n == self.order:
                self.values[n] = self.raw[n]
            else:
                # continuation count of g = number of distinct (n+1)-grams that end with g
                suffix = self.codes[n + 1] & np.uint64((1 << (BITS * n)) - 1)
                uniq, cc = np.unique(suffix, return_counts=True)
                cont = np.zeros(len(self.codes[n]))
                cont[np.searchsorted(self.codes[n], uniq)] = cc
                self.values[n] = cont
            self.discount[n] = _discount(self.values[n])

        for n in range(2, self.order + 1):
            prefix = self.codes[n] >> np.uint64(BITS)
            uniq, start = np.unique(prefix, return_index=True)
            total = np.add.reduceat(self.values[n], start)
            distinct = np.add.reduceat((self.values[n] > 0).astype(np.float64), start)
            raw_total = np.add.reduceat(self.raw[n], start)
            self.ctx[n] = (uniq, start, total, distinct, raw_total)

        # lowest order: continuation unigram, interpolated with the uniform distribution
        cont1 = np.zeros(self.V)
        cont1[self.codes[1].astype(np.int64)] = self.values[1]
        d1, total1 = self.discount[1], cont1.sum()
        self.p_unigram = (np.maximum(cont1 - d1, 0) / total1
                          + d1 * np.count_nonzero(cont1) / total1 / self.V)
        return self

    # ------------------------------------------------------------- helpers
    def _rows(self, n: int, history: list[int]):
        """Rows of the order-n table whose first n-1 tokens equal history[-(n-1):]."""
        if len(history) < n - 1:
            return None
        uniq, start, total, distinct, raw_total = self.ctx[n]
        code = pack(np.array(history[-(n - 1):]))[0]
        i = int(np.searchsorted(uniq, code))
        if i == len(uniq) or uniq[i] != code:
            return None
        lo = start[i]
        hi = start[i + 1] if i + 1 < len(start) else len(self.codes[n])
        words = (self.codes[n][lo:hi] & LOW).astype(np.int64)
        return words, lo, hi, total[i], distinct[i], raw_total[i]

    def kn_distribution(self, history: list[int]) -> np.ndarray:
        p = self.p_unigram.copy()
        for n in range(2, self.order + 1):
            rows = self._rows(n, history)
            if rows is None:            # unseen context: keep the lower-order estimate
                continue
            words, lo, hi, total, distinct, _ = rows
            d = self.discount[n]
            p *= d * distinct / total
            p[words] += np.maximum(self.values[n][lo:hi] - d, 0) / total
        return p

    def addone_distribution(self, history: list[int]) -> np.ndarray:
        rows = self._rows(self.order, history)
        if rows is None:
            return np.full(self.V, 1.0 / self.V)
        words, lo, hi, _, _, raw_total = rows
        p = np.full(self.V, 1.0 / (raw_total + self.V))
        p[words] += self.raw[self.order][lo:hi] / (raw_total + self.V)
        return p

    # ------------------------------------------- vectorized test likelihood
    def _lookup(self, table: np.ndarray, codes: np.ndarray):
        idx = np.searchsorted(table, codes)
        idx_c = np.minimum(idx, len(table) - 1)
        return idx_c, table[idx_c] == codes

    def sequence_logprobs(self, ids: np.ndarray, bos_id: int, mode: str) -> np.ndarray:
        """log P(ids[i] | previous order-1 ids) for every position i (all at once)."""
        N = self.order
        padded = np.concatenate([np.full(N - 1, bos_id), ids]).astype(np.uint64)
        windows = sliding_window_view(padded, N)          # [len(ids), N], last column = target
        target = windows[:, -1].astype(np.int64)
        if mode == "addone":
            uniq, start, total, distinct, raw_total = self.ctx[N]
            ci, cseen = self._lookup(uniq, pack(windows[:, :-1]))
            gi, gseen = self._lookup(self.codes[N], pack(windows))
            c_hw = np.where(gseen, self.raw[N][gi], 0.0)
            c_h = np.where(cseen, raw_total[ci], 0.0)
            return np.log((c_hw + 1) / (c_h + self.V))

        p = self.p_unigram[target]
        for n in range(2, N + 1):
            uniq, start, total, distinct, _ = self.ctx[n]
            ci, cseen = self._lookup(uniq, pack(windows[:, N - n:-1]))
            gi, gseen = self._lookup(self.codes[n], pack(windows[:, N - n:]))
            d = self.discount[n]
            tot = np.where(cseen, total[ci], 1.0)
            val = np.where(gseen, self.values[n][gi], 0.0)
            p_n = np.maximum(val - d, 0) / tot + d * distinct[ci] / tot * p
            p = np.where(cseen, p_n, p)
        return np.log(p)

    # ---------------------------------------------------------- save / load
    def save(self, path: Path) -> None:
        arrays = {"meta": np.array([self.order, self.V])}
        for n in range(1, self.order + 1):
            arrays[f"codes{n}"] = self.codes[n]
            arrays[f"raw{n}"] = self.raw[n].astype(np.uint32)
            arrays[f"values{n}"] = self.values[n].astype(np.uint32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "NGramCounts":
        z = np.load(path)
        order, V = (int(x) for x in z["meta"])
        self = cls(order, V)
        for n in range(1, order + 1):
            self.codes[n] = z[f"codes{n}"]
            self.raw[n] = z[f"raw{n}"].astype(np.float64)
            self.values[n] = z[f"values{n}"].astype(np.float64)
            self.discount[n] = _discount(self.values[n])
        for n in range(2, order + 1):
            prefix = self.codes[n] >> np.uint64(BITS)
            uniq, start = np.unique(prefix, return_index=True)
            self.ctx[n] = (uniq, start,
                           np.add.reduceat(self.values[n], start),
                           np.add.reduceat((self.values[n] > 0).astype(np.float64), start),
                           np.add.reduceat(self.raw[n], start))
        cont1 = np.zeros(V)
        cont1[self.codes[1].astype(np.int64)] = self.values[1]
        d1, total1 = self.discount[1], cont1.sum()
        self.p_unigram = (np.maximum(cont1 - d1, 0) / total1
                          + d1 * np.count_nonzero(cont1) / total1 / V)
        return self


def ngram_path(cfg: Config) -> Path:
    return cfg.artifacts / "ngram" / f"ngram{cfg.ngram_order}.npz"


def train_ngram(cfg: Config) -> Path:
    from .data import read_split

    bpe = BPE(bpe_path(cfg))
    ids = bpe.encode_paragraphs(read_split(cfg, "train"))
    counts = NGramCounts(cfg.ngram_order, bpe.vocab_size).fit(ids, bpe.bos_id)
    path = ngram_path(cfg)
    counts.save(path)
    sizes = {n: len(counts.codes[n]) for n in counts.codes}
    print(f"n-gram: {len(ids):,} training tokens, distinct n-grams {sizes} -> {path}")
    return path


class NGramLM(LanguageModel):
    """Keyboard wrapper around the count tables (mode = 'kn' or 'addone')."""

    def __init__(self, counts: NGramCounts, bpe: BPE, mode: str, path: Path, language: str):
        self.counts, self.bpe, self.mode, self.path = counts, bpe, mode, path
        self.language = language
        self.name = f"{counts.order}-gram " + ("Kneser-Ney" if mode == "kn" else "add-one")
        self.precision = "counts"
        self.word_start, self.is_cont = bpe.word_start, bpe.is_cont
        self.cont_word, self.piece_text = bpe.cont_word, bpe.piece_text

    def context_ids(self, text: str) -> list[int]:
        # same padding as in training: order-1 start symbols before the text
        return [self.bpe.bos_id] * (self.counts.order - 1) + self.bpe.encode(text)

    def logprobs(self, batch: list[list[int]]) -> np.ndarray:
        dist = self.counts.kn_distribution if self.mode == "kn" else self.counts.addone_distribution
        return np.log(np.stack([dist(h) for h in batch]))

    def test_nll(self, paragraphs: list[str]) -> tuple[float, int]:
        ids = self.bpe.encode_paragraphs(paragraphs)
        lp = self.counts.sequence_logprobs(ids, self.bpe.bos_id, self.mode)
        return float(-lp[1:].sum()), len(ids) - 1     # the very first <s> is given

    def size_mb(self) -> float:
        return self.path.stat().st_size / 1e6
