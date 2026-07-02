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
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    use_mmap: bool = True
    use_mlock: bool = False
    offload_kqv: bool = True
    temperature: float = 0.1
    max_tokens: int = 256
    verbose: bool = False


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_settings() -> ModelSettings:
    model_path = Path(
        os.getenv(
            "ORC_LLM_MODEL_PATH",
            str(repo_root() / "models" / DEFAULT_MODEL_FILE),
        )
    ).expanduser()

    return ModelSettings(
        model_path=model_path,
        n_ctx=int(os.getenv("ORC_LLM_N_CTX", "4096")),
        n_gpu_layers=int(os.getenv("ORC_LLM_N_GPU_LAYERS", "-1")),
        n_batch=int(os.getenv("ORC_LLM_N_BATCH", "512")),
        n_ubatch=int(os.getenv("ORC_LLM_N_UBATCH", "512")),
        n_threads=_optional_int("ORC_LLM_N_THREADS"),
        n_threads_batch=_optional_int("ORC_LLM_N_THREADS_BATCH"),
        use_mmap=_env_bool("ORC_LLM_USE_MMAP", True),
        use_mlock=_env_bool("ORC_LLM_USE_MLOCK", False),
        offload_kqv=_env_bool("ORC_LLM_OFFLOAD_KQV", True),
        temperature=float(os.getenv("ORC_LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("ORC_LLM_MAX_TOKENS", "256")),
        verbose=_env_bool("ORC_LLM_VERBOSE", False),
    )
