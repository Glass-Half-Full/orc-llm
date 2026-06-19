from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSettings:
    model_path: Path
    n_ctx: int = 4096
    n_gpu_layers: int = 999
    n_threads: int | None = None
    temperature: float = 0.1
    max_tokens: int = 256
    verbose: bool = False


def load_settings() -> ModelSettings:
    model_path = Path(
        os.getenv(
            "ORC_LLM_MODEL_PATH",
            str(repo_root() / "models" / DEFAULT_MODEL_FILE),
        )
    ).expanduser()

    n_threads_raw = os.getenv("ORC_LLM_N_THREADS")
    return ModelSettings(
        model_path=model_path,
        n_ctx=int(os.getenv("ORC_LLM_N_CTX", "4096")),
        n_gpu_layers=int(os.getenv("ORC_LLM_N_GPU_LAYERS", "999")),
        n_threads=int(n_threads_raw) if n_threads_raw else None,
        temperature=float(os.getenv("ORC_LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("ORC_LLM_MAX_TOKENS", "256")),
        verbose=os.getenv("ORC_LLM_VERBOSE", "").lower() in {"1", "true", "yes"},
    )

