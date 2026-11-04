"""Step 7: evaluate every model on the same held-out test split (R3, R4).

Metrics
- Perplexity per word: exp(total NLL / number of words).  Each model uses its
  own tokenizer, so we divide the NLL of the whole test text by the same word
  count (regex words + one end-of-paragraph symbol per paragraph) for all of
  them.  Per-token perplexity is also saved, but it is only comparable between
  models that share a tokenizer (the n-grams and the feedforward network).
- Top-3 next-word accuracy: at `eval_positions` random word positions of the
  test text (same positions for all models), is the true next word among the
  3 suggestions?
- Latency per suggestion: wall-clock time of one `predict_next_words` call on
  the CPU, averaged over all those calls (at least 100).
- Size: bytes on disk of the saved model.

Writes results/results.json and results/results.md (the slide table).
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import time

import torch

from .config import Config
from .data import read_split
from .models import LADDER, load_model
from .predict import predict_next_words
from .text import WORD_RE, is_word, words

ROLES = {
    "ngram-addone": "Classical baseline (add-one smoothing)",
    "ngram-kn": "Classical baseline (Kneser-Ney smoothing)",
    "ffnn": "Neural network trained from scratch",
    "gpt2-fp32": "Transformer (fine-tuned GPT-2 small)",
    "gpt2-int8": "Quantized Transformer (dynamic INT8)",
}


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = {"cpu": cpu, "cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__}
    if torch.cuda.is_available():
        info["gpu_used_for_perplexity"] = torch.cuda.get_device_name(0)
    return info


def test_paragraphs(cfg: Config) -> list[str]:
    paragraphs = read_split(cfg, "test")
    if not cfg.ppl_max_words:
        return paragraphs
    out, n = [], 0
    for p in paragraphs:
        out.append(p)
        n += len(words(p))
        if n >= cfg.ppl_max_words:
            break
    return out


def sample_positions(paragraphs: list[str], n: int, seed: int) -> list[dict]:
    """Word positions with at least 3 words of context, right after a space."""
    candidates = []
    for pi, p in enumerate(paragraphs):
        for wi, m in enumerate(WORD_RE.finditer(p)):
            if wi >= 3 and m.start() > 0 and p[m.start() - 1] == " " and is_word(m.group()):
                candidates.append((pi, m.start(), m.group()))
    picked = random.Random(seed).sample(candidates, min(n, len(candidates)))
    return [{"context": paragraphs[pi][:start][-300:], "target": word} for pi, start, word in picked]


def evaluate(cfg: Config, keys: list[str] | None = None) -> dict:
    keys = keys or LADDER
    cfg.results.mkdir(parents=True, exist_ok=True)
    paragraphs = test_paragraphs(cfg)
    n_words = sum(len(words(p)) for p in paragraphs) + len(paragraphs)
    positions = sample_positions(paragraphs, cfg.eval_positions, cfg.seed)
    (cfg.results / "eval_positions.json").write_text(json.dumps(positions, indent=1))
    if len(positions) < 100:
        print(f"warning: only {len(positions)} timed runs (R4 asks for at least 100)")

    out_path = cfg.results / "results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {"models": {}}
    results.update({"hardware": hardware(), "test_paragraphs": len(paragraphs),
                    "test_words": n_words, "eval_positions": len(positions)})
    gpu = torch.cuda.is_available()

    for key in keys:
        lm = load_model(cfg, key)
        print(f"\n== {key}: {lm.name} ({lm.precision})")

        # perplexity (the GPU only speeds this up; INT8 stays on the CPU)
        on_gpu = gpu and lm.precision == "FP32" and hasattr(lm, "model")
        if on_gpu:
            lm.model.to("cuda"); lm.device = "cuda"
        t0 = time.time()
        nll, n_tokens = lm.test_nll(paragraphs)
        if on_gpu:
            lm.model.to("cpu"); lm.device = "cpu"
        ppl_word, ppl_token = math.exp(nll / n_words), math.exp(nll / n_tokens)
        print(f"   perplexity per word {ppl_word:.1f} (per token {ppl_token:.1f}, {time.time() - t0:.0f}s)")

        # top-3 accuracy and latency, on the CPU, same positions for every model
        for p in positions[:3]:                                  # warm-up, not timed
            predict_next_words(lm, p["context"], cfg.suggestions, cfg.beam_size, cfg.max_word_pieces)
        hits, times, examples = 0, [], []
        for i, p in enumerate(positions):
            t = time.perf_counter()
            pred = predict_next_words(lm, p["context"], cfg.suggestions, cfg.beam_size,
                                      cfg.max_word_pieces)
            times.append((time.perf_counter() - t) * 1000)
            guess = [w for w, _ in pred]
            hits += p["target"] in guess
            if i < 5:
                examples.append({"context": p["context"][-80:], "target": p["target"], "top3": guess})
        acc = hits / len(positions)
        mean_ms = statistics.mean(times)
        print(f"   top-3 accuracy {acc:.3f}, latency {mean_ms:.1f} ms (mean of {len(times)} runs)")

        results["models"][key] = {
            "name": lm.name, "role": ROLES[key], "precision": lm.precision,
            "perplexity_per_word": ppl_word, "perplexity_per_token": ppl_token,
            "test_tokens": n_tokens, "top3_accuracy": acc,
            "size_mb": lm.size_mb(), "latency_ms_mean": mean_ms,
            "latency_ms_std": statistics.pstdev(times),
            "latency_ms_p95": sorted(times)[int(0.95 * (len(times) - 1))],
            "latency_runs": len(times), "examples": examples,
        }
        del lm

    out_path.write_text(json.dumps(results, indent=2))
    (cfg.results / "results.md").write_text(results_table(results), encoding="utf-8")
    print("\n" + results_table(results))
    return results


def results_table(results: dict) -> str:
    hw = results["hardware"]
    lines = [
        "| Model | Role | Precision | Perplexity per word (lower is better) | Top-3 next-word accuracy | Size (MB) | Latency per suggestion (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in LADDER:
        r = results["models"].get(key)
        if not r:
            continue
        lines.append(f"| {r['name']} | {r['role']} | {r['precision']} | {r['perplexity_per_word']:.1f} | "
                     f"{100 * r['top3_accuracy']:.1f}% | {r['size_mb']:.1f} | "
                     f"{r['latency_ms_mean']:.1f} ± {r['latency_ms_std']:.1f} |")
    lines += [
        "",
        f"Test set: {results['test_paragraphs']} paragraphs, {results['test_words']:,} words; "
        f"top-3 accuracy and latency over {results['eval_positions']} test positions.",
        f"Latency: mean ± std of one suggestion call on the CPU ({hw['cpu']}, {hw['cores']} cores, "
        f"{hw['torch_threads']} PyTorch threads, {hw['os']}).",
    ]
    if "gpu_used_for_perplexity" in hw:
        lines.append(f"Perplexity of FP32 models computed on {hw['gpu_used_for_perplexity']}.")
    return "\n".join(lines) + "\n"
