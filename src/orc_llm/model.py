from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .settings import ModelSettings, load_settings


Message = dict[str, str]


@dataclass
class CompletionResult:
    text: str
    raw: dict[str, Any]


class LocalLiquidModel:
    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.settings = settings or load_settings()
        self._llm: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.settings.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.settings.model_path}. "
                "Run scripts/download_model.py or set ORC_LLM_MODEL_PATH."
            )

        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": str(self.settings.model_path),
            "n_ctx": self.settings.n_ctx,
            "n_gpu_layers": self.settings.n_gpu_layers,
            "n_batch": self.settings.n_batch,
            "n_ubatch": self.settings.n_ubatch,
            "use_mmap": self.settings.use_mmap,
            "use_mlock": self.settings.use_mlock,
            "offload_kqv": self.settings.offload_kqv,
            "verbose": self.settings.verbose,
        }
        if self.settings.n_threads is not None:
            kwargs["n_threads"] = self.settings.n_threads
        if self.settings.n_threads_batch is not None:
            kwargs["n_threads_batch"] = self.settings.n_threads_batch

        self._llm = Llama(**kwargs)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        self.load()
        assert self._llm is not None

        raw = self._llm.create_chat_completion(
            messages=list(messages),
            temperature=self.settings.temperature if temperature is None else temperature,
            max_tokens=self.settings.max_tokens if max_tokens is None else max_tokens,
        )
        text = raw["choices"][0]["message"]["content"]
        return CompletionResult(text=text, raw=raw)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a precise, concise assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )


model = LocalLiquidModel()
