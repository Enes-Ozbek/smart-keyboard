"""Command line for the whole pipeline.

    python -m keyboard.cli all             # every step below, in order
    python -m keyboard.cli prepare         # 1. corpus -> data/{train,valid,test}.txt
    python -m keyboard.cli bpe             # 2. train the BPE tokenizer
    python -m keyboard.cli ngram           # 3. n-gram counts (add-one + Kneser-Ney)
    python -m keyboard.cli ffnn            # 4. feedforward neural LM (SGD)
    python -m keyboard.cli gpt2            # 5. fine-tune GPT-2 small
    python -m keyboard.cli quantize        # 6. dynamic INT8 GPT-2
    python -m keyboard.cli evaluate        # 7. results/results.md
    python -m keyboard.cli upload --models-repo USER/NAME     # weights -> Hugging Face Hub
    python -m keyboard.cli download --models-repo USER/NAME   # weights <- Hugging Face Hub

Any field of `keyboard.config.Config` can be overridden, e.g. `--ffnn-epochs 3`.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields

from .config import Config

STEPS = ["prepare", "bpe", "ngram", "ffnn", "gpt2", "quantize", "evaluate"]


def run_step(step: str, cfg: Config, models: list[str] | None = None) -> None:
    t0 = time.time()
    print(f"\n######## {step} ########")
    if step == "prepare":
        from .data import prepare
        prepare(cfg)
    elif step == "bpe":
        from .bpe import train_bpe
        train_bpe(cfg)
    elif step == "ngram":
        from .ngram import train_ngram
        train_ngram(cfg)
    elif step == "ffnn":
        from .ffnn import train_ffnn
        train_ffnn(cfg)
    elif step == "gpt2":
        from .gpt2 import finetune_gpt2
        finetune_gpt2(cfg)
    elif step == "quantize":
        from .quantize import quantize_gpt2
        quantize_gpt2(cfg)
    elif step == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, models)
    print(f"######## {step} done in {time.time() - t0:.0f}s")
    cfg.save()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m keyboard.cli")
    parser.add_argument("step", choices=STEPS + ["all", "upload", "download"])
    parser.add_argument("--models", nargs="+", help="evaluate only these (see keyboard/models.py)")
    for f in fields(Config):
        parser.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=None)
    args = parser.parse_args(argv)
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)}
    cfg = Config.load(overrides.pop("artifacts_dir"), **overrides)

    if args.step == "all":
        for step in STEPS:
            run_step(step, cfg, args.models)
    elif args.step == "upload":
        from .hub import upload
        upload(cfg)
    elif args.step == "download":
        from .hub import download
        download(cfg)
    else:
        run_step(args.step, cfg, args.models)


if __name__ == "__main__":
    main()
