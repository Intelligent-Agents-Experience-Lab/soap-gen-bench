"""Single-model manager: load, generate, unload. One model in memory at a time.

Falls back to AirLLM (layer-by-layer loading) automatically when a model
is too large to fit in VRAM via standard HuggingFace loading.
Install: pip install airllm
"""
import gc
from typing import Optional

import mlflow
import torch
from mlflow.entities import SpanType
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'Instruct: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Output: ' + message['content'] + '\\n' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ 'Output: ' }}{% endif %}"
    "{% endfor %}"
)


def _is_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(e).lower()


class ModelManager:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._current_key: Optional[str] = None
        self._use_airllm: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_standard(self, model_name: str, adapter_path: Optional[str]) -> None:
        """Standard HuggingFace fp16 load. Raises on CUDA OOM."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

        trust_order = [True, False] if "Phi-3" in model_name else [False, True]
        base_model = None
        last_err: Optional[Exception] = None
        for trust in trust_order:
            try:
                kwargs: dict = dict(
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=trust,
                )
                if "Phi-3" in model_name:
                    kwargs["attn_implementation"] = "eager"
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name, **kwargs
                ).to(self.device)
                break
            except Exception as e:
                last_err = e
                if _is_oom(e):
                    raise
                continue

        if base_model is None:
            raise RuntimeError(f"Could not load {model_name}: {last_err}") from last_err

        if adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
        else:
            self._model = base_model

    def _load_airllm(self, model_name: str, adapter_path: Optional[str]) -> None:
        """AirLLM layer-by-layer load — uses a fraction of VRAM regardless of model size."""
        if adapter_path:
            raise RuntimeError(
                f"AirLLM fallback does not support LoRA adapters. "
                f"Either increase VRAM or run without an adapter for {model_name}."
            )
        try:
            from airllm import AutoModel as AirAutoModel
        except ImportError as e:
            raise ImportError(
                "AirLLM is not installed. Run: pip install airllm"
            ) from e

        print(f"  VRAM too small — falling back to AirLLM for {model_name}…", flush=True)
        self._model = AirAutoModel.from_pretrained(model_name)
        self._tokenizer = self._model.tokenizer

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

    def load(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        key = f"{model_name}:{adapter_path}"
        if self._current_key == key and self._model is not None:
            return
        self.unload()

        print(f"Loading {model_name} (adapter={adapter_path})…", flush=True)
        self._use_airllm = False
        try:
            self._load_standard(model_name, adapter_path)
        except Exception as e:
            if _is_oom(e):
                # Clear VRAM before handing off to AirLLM
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._model = None
                self._tokenizer = None
                self._load_airllm(model_name, adapter_path)
                self._use_airllm = True
            else:
                self.unload()
                raise

        self._current_key = key
        loader = "AirLLM" if self._use_airllm else "standard HuggingFace"
        print(f"Loaded {model_name} via {loader}.", flush=True)

    def unload(self) -> None:
        if self._model is not None:
            print(f"Unloading {self._current_key}…", flush=True)
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._current_key = None
            self._use_airllm = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @mlflow.trace(span_type=SpanType.LLM)
    def generate(self, messages: list, max_new_tokens: int = 512) -> str:
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load() first.")
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._use_airllm:
            return self._generate_airllm(prompt, max_new_tokens)
        return self._generate_standard(prompt, max_new_tokens)

    def _generate_standard(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def _generate_airllm(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=2048,
        )
        output = self._model.generate(
            inputs["input_ids"].cuda(),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        # AirLLM returns memory-mapped token arrays via sequences_ptr
        full_ids = output.sequences_ptr[0]
        new_ids = full_ids[inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def generate_stream(self, messages: list, max_new_tokens: int = 512) -> None:
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load() first.")
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._use_airllm:
            # AirLLM does not support streaming; print buffered output
            print(self._generate_airllm(prompt, max_new_tokens))
            return
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                streamer=streamer,
            )


# Module-level singleton shared by all sub-packages
manager = ModelManager()
