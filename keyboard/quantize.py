"""Step 6 (area D, mandatory): dynamic INT8 quantization of the fine-tuned GPT-2.

Dynamic quantization stores the weights of every linear layer as 8-bit
integers (with one scale per layer) and quantizes the activations on the fly,
so matrix multiplications run in INT8 on the CPU.  Embeddings and layer norms
stay in FP32.

GPT-2 in Hugging Face implements its linear layers as `Conv1D` (a Linear with
a transposed weight), which PyTorch's quantizer does not recognise, so we first
convert each `Conv1D` into an equivalent `nn.Linear`.
"""
from __future__ import annotations

import json
import platform
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .gpt2 import GPT2LM, finetuned_dir


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


def quantize_int8(model: nn.Module) -> nn.Module:
    _set_engine()
    model = conv1d_to_linear(model.cpu().eval())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # torch.ao.quantization prints a deprecation notice
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def quantize_gpt2(cfg: Config) -> Path:
    from transformers import AutoModelForCausalLM

    src, out = finetuned_dir(cfg), int8_dir(cfg)
    model = AutoModelForCausalLM.from_pretrained(src)
    qmodel = quantize_int8(model)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out / "model_int8.pt")
    # config + tokenizer files so the INT8 model loads without the FP32 weights
    for f in src.iterdir():
        if f.suffix in (".json", ".txt", ".model") and f.name != "finetune_log.json":
            shutil.copy(f, out / f.name)
    (out / "quantization.json").write_text(json.dumps(
        {"method": "PyTorch dynamic quantization", "dtype": "qint8",
         "layers": "all nn.Linear (attention, MLP, LM head)",
         "engine": torch.backends.quantized.engine}, indent=2))
    fp32 = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e6
    int8 = (out / "model_int8.pt").stat().st_size / 1e6
    print(f"INT8: {fp32:.0f} MB -> {int8:.0f} MB  ({out})")
    return out


def load_int8(cfg: Config) -> GPT2LM:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = int8_dir(cfg)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path))
    qmodel = quantize_int8(model)                 # same structure as the saved model ...
    state = torch.load(path / "model_int8.pt", map_location="cpu", weights_only=False)
    qmodel.load_state_dict(state)                 # ... then load the INT8 weights
    tokenizer = AutoTokenizer.from_pretrained(path)
    return GPT2LM(qmodel, tokenizer, "GPT-2 fine-tuned", "INT8", [path / "model_int8.pt"], cfg)
