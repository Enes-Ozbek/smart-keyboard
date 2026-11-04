"""Step 6 (area D, mandatory): INT8 quantization of the fine-tuned GPT-2.

PyTorch dynamic quantization stores the weights of the linear layers as 8-bit
integers and quantizes each layer's input on the fly, with ONE scale for the
whole input tensor (and only 7 bits on x86 CPUs).  GPT-2 has a few hidden
dimensions whose values are far larger than all the others ("outlier
features"), so with plain dynamic quantization that one scale wipes out the
small values.  Our first run did exactly that: test perplexity per word went
from 40 (FP32) to 27,844 (INT8).

So the quantizer is built in three standard steps, each checked on validation
text (never on the test split):

1. Per-channel weights: one scale per output row of a weight matrix instead
   of one per matrix.
2. SmoothQuant (Xiao et al., 2023): for every input channel j of a layer,
   s_j = max|X_j|^a / max|W_j|^(1-a).  The input channel is divided by s_j
   and the matching weight column multiplied by s_j, so the FP32 model
   computes exactly the same thing, but the outliers move from the
   activations (one scale per tensor, hard to quantize) into the weights
   (one scale per row, easy).  The division is folded into the LayerNorm in
   front of the layer, or into the value projection for the attention output
   layer.  The MLP output layer comes after a GELU, so it cannot be smoothed.
3. Mixed precision: if the INT8 model is still more than 2% worse than FP32
   on validation text, the layers with the largest INT8 error are kept in
   FP32, a few at a time, until it is within 2%.

The choices and every validation perplexity are written to
artifacts/gpt2-int8/quantization.json (useful for the slides).

GPT-2 in Hugging Face implements its linear layers as `Conv1D` (a Linear with
a transposed weight), which PyTorch's quantizer does not recognise, so we first
convert each `Conv1D` into an equivalent `nn.Linear`.
"""
from __future__ import annotations

import copy
import json
import platform
import shutil
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .gpt2 import GPT2LM, _blocks_ppl, _token_blocks, finetuned_dir

CHECK_LEN = 256          # validation blocks of 256 GPT-2 tokens ...
CHECK_BLOCKS = 8         # ... 8 for the activation statistics, 8 others for the perplexity checks
TOLERANCE = 0.02         # INT8 may be at most 2% worse than FP32 on the validation blocks
ALPHAS = (0.5, 0.75)     # SmoothQuant migration strengths that are tried
KEEP_STEPS = (1, 2, 3, 4, 6, 8, 12, 16)   # how many layers may stay in FP32


def int8_dir(cfg: Config) -> Path:
    return cfg.artifacts / "gpt2-int8"


def _set_engine() -> None:
    engines = torch.backends.quantized.supported_engines
    if platform.machine().lower() in ("arm64", "aarch64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"      # Apple Silicon / ARM CPUs
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"       # x86 CPUs
    elif "x86" in engines:
        torch.backends.quantized.engine = "x86"


def conv1d_to_linear(model: nn.Module) -> nn.Module:
    """Replace every transformers Conv1D with an nn.Linear that computes the same thing."""
    from transformers.pytorch_utils import Conv1D

    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Conv1D):
                n_in, n_out = child.weight.shape            # Conv1D weight is [in, out]
                linear = nn.Linear(n_in, n_out)
                linear.weight.data = child.weight.data.t().contiguous()
                linear.bias.data = child.bias.data.clone()
                setattr(module, name, linear)
    return model


def linear_names(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _qconfig(per_channel: bool):
    from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig

    return per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig


def _quantize(model: nn.Module, layers: list[str], per_channel: bool) -> nn.Module:
    """Dynamic INT8 quantization of the named linear layers (in place)."""
    _set_engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # torch.ao.quantization prints a deprecation notice
        return torch.ao.quantization.quantize_dynamic(
            model, {name: _qconfig(per_channel) for name in layers}, dtype=torch.qint8, inplace=True)


# ---------------------------------------------------------------- SmoothQuant
@torch.no_grad()
def input_absmax(model: nn.Module, blocks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Largest |value| of every input channel of every linear layer, over the blocks."""
    stats: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def hook(_, args, name=name):
                x = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(0).float()
                stats[name] = torch.maximum(stats[name], x) if name in stats else x
            hooks.append(module.register_forward_pre_hook(hook))
    model.eval()
    for block in blocks:
        model(block.unsqueeze(0))
    for h in hooks:
        h.remove()
    return stats


def _scales(act_max: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    w_max = weight.abs().amax(0).float().clamp(min=1e-5)         # per input channel
    return (act_max.clamp(min=1e-5).pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)


@torch.no_grad()
def smooth(model: nn.Module, stats: dict[str, torch.Tensor], alpha: float, layers: set[str]) -> None:
    """SmoothQuant, in place.  The FP32 outputs of the model do not change."""
    for i, block in enumerate(model.transformer.h):
        p = f"transformer.h.{i}."
        attn, mlp = block.attn, block.mlp
        E = attn.c_proj.in_features
        if p + "attn.c_proj" in layers:            # input = attention-weighted values
            s = _scales(stats[p + "attn.c_proj"], attn.c_proj.weight, alpha)
            attn.c_attn.weight[2 * E:] /= s[:, None]            # value rows of the q|k|v projection
            attn.c_attn.bias[2 * E:] /= s
            attn.c_proj.weight *= s
        for ln, lin, name in ((block.ln_1, attn.c_attn, "attn.c_attn"),
                              (block.ln_2, mlp.c_fc, "mlp.c_fc")):
            if p + name in layers:                 # input = LayerNorm output
                s = _scales(stats[p + name], lin.weight, alpha)
                ln.weight /= s
                ln.bias /= s
                lin.weight *= s
    if "lm_head" in layers:
        ln_f = model.transformer.ln_f
        s = _scales(stats["lm_head"], model.lm_head.weight, alpha)
        ln_f.weight /= s
        ln_f.bias /= s
        # the LM head shares its weight with the token embeddings: give it its own copy
        model.lm_head.weight = nn.Parameter(model.lm_head.weight * s)
        model.config.tie_word_embeddings = False


def build_int8(base: nn.Module, stats: dict | None, alpha: float | None, fp32_layers: set[str],
               per_channel: bool = True) -> nn.Module:
    """A quantized copy of `base` (a GPT-2 whose Conv1D layers are already nn.Linear)."""
    model = copy.deepcopy(base)
    layers = [n for n in linear_names(model) if n not in fp32_layers]
    if alpha is not None:
        smooth(model, stats, alpha, set(layers))
    return _quantize(model, layers, per_channel)


@torch.no_grad()
def layer_errors(model: nn.Module, blocks: torch.Tensor, per_channel: bool) -> dict[str, float]:
    """Relative INT8 error of each linear layer's output, averaged over tokens."""
    errors: dict[str, list[float]] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            q = _quantize(nn.Sequential(copy.deepcopy(module)), ["0"], per_channel)[0]

            def hook(_, args, out, name=name, q=q):
                y, yq = out.float(), q(args[0]).float()
                rel = ((yq - y) ** 2).sum(-1) / (y ** 2).sum(-1).clamp(min=1e-12)
                errors.setdefault(name, []).append(rel.mean().item())
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    for block in blocks:
        model(block.unsqueeze(0))
    for h in hooks:
        h.remove()
    return {n: sum(v) / len(v) for n, v in errors.items()}


# ---------------------------------------------------------------- the step
def _validation_blocks(cfg: Config, tokenizer, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .data import read_split

    ids = _token_blocks(tokenizer, read_split(cfg, "valid"), 1).view(-1)
    length = max(2, min(CHECK_LEN, max_len, len(ids) // (2 * CHECK_BLOCKS)))  # small corpora
    blocks = ids[:len(ids) // length * length].view(-1, length)
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(cfg.seed))
    stats_blocks = blocks[order[:CHECK_BLOCKS]]
    check_blocks = blocks[order[CHECK_BLOCKS:2 * CHECK_BLOCKS]]
    return stats_blocks, check_blocks


def quantize_gpt2(cfg: Config) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    src, out = finetuned_dir(cfg), int8_dir(cfg)
    tokenizer = AutoTokenizer.from_pretrained(src)
    base = conv1d_to_linear(AutoModelForCausalLM.from_pretrained(src).cpu().eval())
    stats_blocks, check_blocks = _validation_blocks(cfg, tokenizer, base.config.n_positions)
    n_check = check_blocks.numel()
    t0 = time.time()

    def ppl(model: nn.Module) -> float:
        return _blocks_ppl(model, check_blocks, "cpu", batch=1)

    tried = []

    def report(label: str, value: float) -> float:
        tried.append({"setting": label, "valid_ppl": round(value, 2)})
        print(f"  {label:<58} {value:10.2f}   ({time.time() - t0:.0f}s)", flush=True)
        return value

    print(f"Quantization check: perplexity per GPT-2 token on {n_check:,} validation tokens")
    fp32 = report("FP32 (reference)", ppl(base))
    report("INT8, plain dynamic quantization (per-tensor weights)",
           ppl(build_int8(base, None, None, set(), per_channel=False)))
    best_ppl = report("INT8, per-channel weights", ppl(build_int8(base, None, None, set())))
    alpha = None
    stats = input_absmax(base, stats_blocks)
    for a in ALPHAS:
        value = report(f"INT8, per-channel + SmoothQuant (alpha {a})",
                       ppl(build_int8(base, stats, a, set())))
        if value < best_ppl:
            best_ppl, alpha = value, a

    target = fp32 * (1 + TOLERANCE)
    fp32_layers: list[str] = []
    if best_ppl > target:
        smoothed = copy.deepcopy(base)
        if alpha is not None:
            smooth(smoothed, stats, alpha, set(linear_names(smoothed)))
        errors = layer_errors(smoothed, stats_blocks, per_channel=True)
        del smoothed
        ranked = sorted(errors, key=errors.get, reverse=True)
        print("  layers with the largest INT8 error: "
              + ", ".join(f"{n} ({errors[n]:.2g})" for n in ranked[:5]))
        for k in KEEP_STEPS:
            keep = ranked[:k]
            value = report(f"  + {k} most sensitive layer(s) kept in FP32",
                           ppl(build_int8(base, stats, alpha, set(keep))))
            if value < best_ppl:
                best_ppl, fp32_layers = value, keep
            if value <= target:
                break
    if best_ppl > target:
        print(f"warning: INT8 is still {100 * (best_ppl / fp32 - 1):.0f}% worse than FP32 "
              "on the validation blocks; using the best setting found")

    qmodel = build_int8(base, stats, alpha, set(fp32_layers))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out / "model_int8.pt")
    # config + tokenizer files so the INT8 model loads without the FP32 weights
    for f in src.iterdir():
        if f.suffix in (".json", ".txt", ".model") and f.name != "finetune_log.json":
            shutil.copy(f, out / f.name)
    qmodel.config.to_json_file(out / "config.json")      # tie_word_embeddings may have changed
    n_linear = len(linear_names(base))
    info = {
        "method": "PyTorch dynamic quantization (INT8 weights, activations quantized on the fly)",
        "dtype": "qint8", "weights": "per-channel", "smoothquant_alpha": alpha,
        "fp32_layers": fp32_layers,
        "int8_layers": f"{n_linear - len(fp32_layers)} of {n_linear} linear layers",
        "engine": torch.backends.quantized.engine,
        "validation_tokens": n_check, "valid_ppl_fp32": round(fp32, 2),
        "valid_ppl_int8": round(best_ppl, 2), "tried": tried,
    }
    (out / "quantization.json").write_text(json.dumps(info, indent=2))
    fp32_mb = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e6
    int8_mb = (out / "model_int8.pt").stat().st_size / 1e6
    print(f"INT8: SmoothQuant alpha {alpha}, {info['int8_layers']} in INT8, "
          f"valid ppl {fp32:.2f} (FP32) -> {best_ppl:.2f} (INT8)")
    print(f"INT8: {fp32_mb:.0f} MB -> {int8_mb:.0f} MB  ({out})")
    return out


def load_int8(cfg: Config) -> GPT2LM:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = int8_dir(cfg)
    info_path = path / "quantization.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path))
    model = conv1d_to_linear(model.cpu().eval())
    fp32_layers = set(info.get("fp32_layers", []))
    layers = [n for n in linear_names(model) if n not in fp32_layers]
    qmodel = _quantize(model, layers, per_channel=info.get("weights") == "per-channel")
    state = torch.load(path / "model_int8.pt", map_location="cpu", weights_only=False)
    qmodel.load_state_dict(state)                 # INT8 weights (and the smoothed LayerNorms)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return GPT2LM(qmodel, tokenizer, "GPT-2 fine-tuned", "INT8", [path / "model_int8.pt"], cfg)
