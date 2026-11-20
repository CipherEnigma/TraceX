"""Model backends (Member 3).

    from tracex_llm.backends import load_backend
    llm = load_backend("qwen7b")                 # or "qwen3b", "mock"
    outs = llm.generate([prompt.messages(), ...])
    outs[0].text, outs[0].seconds
    llm.info()                                   # model, quantisation, GPU, versions -> stored with results

`hf`   : Hugging Face transformers + bitsandbytes 4-bit (Kaggle GPU). torch/transformers are
         imported lazily so the rest of the package works without them.
`mock` : deterministic stand-in for tests and the dashboard. It ranks the POI ids that appear in
         the prompt by how often they appear. It is NOT a model and is never used for reported numbers.
"""
from __future__ import annotations

import json
import platform
import re
import time
from collections import Counter
from dataclasses import dataclass

from . import config as C


@dataclass
class GenOutput:
    text: str
    prompt_tokens: int | None = None
    new_tokens: int | None = None
    seconds: float | None = None   # wall time of the batch divided by its size


class Backend:
    key: str

    def generate(self, batch: list[list[dict]]) -> list[GenOutput]:
        raise NotImplementedError

    def info(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hugging Face (Qwen2.5, 4-bit)
# ---------------------------------------------------------------------------
class HFBackend(Backend):
    def __init__(self, cfg: C.ModelConfig, hf_id: str | None = None, batch_size: int = C.BATCH_SIZE):
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.key = cfg.key
        self.hf_id = hf_id or cfg.hf_id
        self.batch_size = batch_size
        if not torch.cuda.is_available():
            raise RuntimeError("HF backend needs a CUDA GPU (run on Kaggle); use model 'mock' to test locally")

        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.tok.padding_side = "left"          # decoder-only batching
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = {"device_map": "auto"}
        # transformers renamed torch_dtype -> dtype in 4.56
        dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        kw[dtype_key] = torch.float16
        if cfg.load_in_4bit:
            q = dict(C.QUANTIZATION)
            q["bnb_4bit_compute_dtype"] = getattr(torch, q["bnb_4bit_compute_dtype"])
            kw["quantization_config"] = BitsAndBytesConfig(**q)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kw)
        self.model.eval()

        gc = self.model.generation_config
        gc.do_sample = C.GENERATION["do_sample"]
        gc.repetition_penalty = C.GENERATION["repetition_penalty"]
        gc.temperature, gc.top_p, gc.top_k = None, None, None   # unused by greedy; silences warnings
        gc.pad_token_id = self.tok.pad_token_id

    def _render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run(self, texts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=C.GENERATION["max_new_tokens"],
                                      do_sample=C.GENERATION["do_sample"],
                                      repetition_penalty=C.GENERATION["repetition_penalty"])
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / len(texts)
        n_in = enc["input_ids"].shape[1]
        res = []
        for i, row in enumerate(out):
            gen = row[n_in:]
            n_new = int((gen != self.tok.pad_token_id).sum())
            res.append(GenOutput(self.tok.decode(gen, skip_special_tokens=True).strip(),
                                 prompt_tokens=int(enc["attention_mask"][i].sum()),
                                 new_tokens=n_new, seconds=sec))
        return res

    def _run_safe(self, texts):
        """Run a batch; on CUDA OOM split it in half and retry."""
        try:
            return self._run(texts)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(texts) == 1:
                raise
            mid = len(texts) // 2
            return self._run_safe(texts[:mid]) + self._run_safe(texts[mid:])

    def generate(self, batch):
        texts = [self._render(m) for m in batch]
        out = []
        for i in range(0, len(texts), self.batch_size):
            out += self._run_safe(texts[i:i + self.batch_size])
        return out

    def info(self):
        import transformers
        torch = self.torch
        info = {
            "model": self.key, "hf_id": self.hf_id, "backend": "hf",
            "quantization": "4bit-nf4" if getattr(self.model, "is_loaded_in_4bit", False) else "none",
            "generation": C.GENERATION, "batch_size": self.batch_size,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
        }
        try:
            import bitsandbytes
            info["bitsandbytes"] = bitsandbytes.__version__
        except ImportError:
            pass
        return info


# ---------------------------------------------------------------------------
# Mock (tests / dashboard only)
# ---------------------------------------------------------------------------
_POI = re.compile(r"POI (\d+)")


class MockBackend(Backend):
    key = "mock"

    def generate(self, batch):
        out = []
        for messages in batch:
            user = messages[-1]["content"]
            counts = Counter(int(x) for x in _POI.findall(user))
            ranked = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            text = json.dumps({"prediction": ranked[:C.N_PREDICTIONS],
                               "reason": "mock: most frequent POI ids in the prompt"})
            out.append(GenOutput(text, prompt_tokens=len(user) // 4, new_tokens=len(text) // 4, seconds=0.0))
        return out

    def info(self):
        return {"model": "mock", "backend": "mock", "note": "not a language model; tests only",
                "python": platform.python_version()}


def load_backend(model: str, **kw) -> Backend:
    if model not in C.MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {tuple(C.MODELS)}")
    cfg = C.MODELS[model]
    if cfg.backend == "mock":
        return MockBackend()
    return HFBackend(cfg, **kw)
