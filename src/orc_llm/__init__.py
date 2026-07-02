"""Local LiquidAI LFM2.5 service helpers."""

from .notebook import apply_prompt_to_column, benchmark_tokens_per_second, load_model

__all__ = [
    "__version__",
    "apply_prompt_to_column",
    "benchmark_tokens_per_second",
    "load_model",
]

__version__ = "0.1.0"
