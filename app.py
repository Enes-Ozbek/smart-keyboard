"""Smart Keyboard: next-word suggestions from every model of the ladder, side by side.

    python app.py                     # open http://127.0.0.1:7860
    python app.py --share             # public link (e.g. from Google Colab)
    python app.py --models-repo USER/NAME   # download trained weights first

The first column is the model the keyboard runs: the INT8-quantized GPT-2 (R5).
As you type, each model shows its top-3 next words and how long it took;
clicking a word inserts it.
"""
from __future__ import annotations

import argparse
import time

import gradio as gr

from keyboard.config import Config
from keyboard.hub import ensure_artifacts
from keyboard.models import load_models
from keyboard.predict import predict_next_words
from keyboard.text import split_partial

# the quantized Transformer first: it is the keyboard's main model
ORDER = ["gpt2-int8", "gpt2-fp32", "ffnn", "ngram-kn", "ngram-addone"]
TITLES = {
    "gpt2-int8": "GPT-2 INT8 (keyboard model)",
    "gpt2-fp32": "GPT-2 FP32",
    "ffnn": "Feedforward NN",
    "ngram-kn": "4-gram Kneser-Ney",
    "ngram-addone": "4-gram add-one",
}


def insert_word(text: str, word: str) -> str:
    """Replace the word being typed (if any) with the clicked suggestion."""
    if not word:
        return text
    context, _ = split_partial(text)
    if context and not context[-1].isspace():
        context += " "
    return context + word + " "


def build(cfg: Config, keys: list[str]) -> gr.Blocks:
    models = load_models(cfg, keys)

    def suggest(text: str):
        updates = []
        for key, lm in models.items():
            t0 = time.perf_counter()
            pred = predict_next_words(lm, text or "", cfg.suggestions, cfg.beam_size,
                                      cfg.max_word_pieces)
            ms = (time.perf_counter() - t0) * 1000
            for i in range(cfg.suggestions):
                if i < len(pred):
                    updates.append(gr.update(value=pred[i][0], visible=True))
                else:
                    updates.append(gr.update(value="", visible=False))
            probs = ", ".join(f"{w} {p:.1%}" for w, p in pred)
            updates.append(f"**{ms:.0f} ms** · {probs}")
        return updates

    with gr.Blocks(title="Smart Keyboard") as demo:
        gr.Markdown("# Smart Keyboard: next-word prediction\n"
                    "Type in the box. Every model suggests the next 3 words; click one to insert it. "
                    "The keyboard itself runs the **INT8-quantized GPT-2** (first column).")
        box = gr.Textbox(label="Type here", lines=4, autofocus=True,
                         placeholder="the history of the city ")
        outputs, buttons = [], []
        with gr.Row():
            for key in models:
                with gr.Column(min_width=180):
                    gr.Markdown(f"### {TITLES.get(key, key)}")
                    for _ in range(cfg.suggestions):
                        b = gr.Button("", size="sm", visible=False,
                                      variant="primary" if key == "gpt2-int8" else "secondary")
                        buttons.append(b)
                        outputs.append(b)
                    info = gr.Markdown("")
                    outputs.append(info)
        box.input(suggest, box, outputs, trigger_mode="always_last", show_progress="hidden")
        for b in buttons:
            b.click(insert_word, [box, b], box, show_progress="hidden").then(
                suggest, box, outputs, show_progress="hidden")
        demo.load(suggest, box, outputs, show_progress="hidden")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--models", nargs="+", default=None, help=f"subset of {ORDER}")
    parser.add_argument("--models-repo", default=None, help="Hugging Face repo with the weights")
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.artifacts_dir, models_repo=args.models_repo)
    ensure_artifacts(cfg)
    keys = args.models or ORDER
    build(cfg, keys).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
