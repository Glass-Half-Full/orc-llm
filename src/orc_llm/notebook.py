from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from .dataframe import insert_column_after
from .model import LocalLiquidModel
from .settings import load_settings


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, concise assistant. Return only the requested output."
)


def load_model(
    *,
    model_path: str | Path | None = None,
    n_ctx: int | None = None,
    n_gpu_layers: int | None = None,
    n_batch: int | None = None,
    n_ubatch: int | None = None,
    n_threads: int | None = None,
    n_threads_batch: int | None = None,
    use_mmap: bool | None = None,
    use_mlock: bool | None = None,
    offload_kqv: bool | None = None,
    verbose: bool | None = None,
    preload: bool = True,
) -> LocalLiquidModel:
    """Create a local model instance for notebook reuse.

    Defaults come from the ORC_LLM_* environment variables. The repository
    defaults are tuned for the 1.2B Q4 GGUF on a small GPU/CPU machine.
    """

    settings = load_settings()
    overrides: dict[str, Any] = {}
    if model_path is not None:
        overrides["model_path"] = Path(model_path).expanduser()
    for name, value in {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "n_threads": n_threads,
        "n_threads_batch": n_threads_batch,
        "use_mmap": use_mmap,
        "use_mlock": use_mlock,
        "offload_kqv": offload_kqv,
        "verbose": verbose,
    }.items():
        if value is not None:
            overrides[name] = value

    model = LocalLiquidModel(replace(settings, **overrides))
    if preload:
        model.load()
    return model


def apply_prompt_to_column(
    dataframe: Any,
    *,
    text_column: str,
    prompt: str,
    output_column: str | None = None,
    model: LocalLiquidModel | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0,
    max_tokens: int = 128,
    progress: bool | Callable[[int, int], None] = False,
) -> Any:
    """Apply a local prompt to each value in a pandas DataFrame column.

    If `prompt` contains `{text}`, `{value}`, or a column placeholder like
    `{notes}`, those placeholders are formatted from the current row. Otherwise
    the selected column value is appended to the prompt.
    """

    pd = _require_pandas()
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if text_column not in dataframe.columns:
        raise ValueError(f"text_column not found: {text_column}")

    output_column = output_column or f"{text_column}_output"
    llm = model or load_model(preload=True)
    result = dataframe.copy()
    records = result.to_dict(orient="records")
    outputs: list[str] = []
    total = len(records)

    for index, row in enumerate(records, start=1):
        text = _stringify(row.get(text_column, ""))
        rendered_prompt = render_prompt(prompt, text=text, row=row, text_column=text_column)
        completion = llm.complete(
            rendered_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs.append(completion.text.strip())
        _report_progress(progress, index, total)

    result[output_column] = outputs
    ordered_columns = insert_column_after(list(result.columns), text_column, output_column)
    return result.loc[:, ordered_columns]


def render_prompt(
    prompt: str,
    *,
    text: str,
    row: Mapping[str, Any],
    text_column: str,
) -> str:
    fields = {key: _stringify(value) for key, value in row.items()}
    fields["text"] = text
    fields["value"] = text

    known_placeholders = {"{text}", "{value}", f"{{{text_column}}}"}
    known_placeholders.update(f"{{{column}}}" for column in row)
    if any(placeholder in prompt for placeholder in known_placeholders):
        rendered = prompt
        for name, value in fields.items():
            rendered = rendered.replace(f"{{{name}}}", value)
        return rendered

    return f"{prompt.rstrip()}\n\n{text_column}:\n{text}\n\nOutput:"


def benchmark_tokens_per_second(
    *,
    model: LocalLiquidModel | None = None,
    prompt: str = "Write one short sentence about local dataframe analysis.",
    max_tokens: int = 32,
) -> dict[str, float | int | str]:
    """Measure local generation speed for the currently loaded model."""

    llm = model or load_model(preload=True)
    started = perf_counter()
    result = llm.complete(prompt, temperature=0, max_tokens=max_tokens)
    elapsed = perf_counter() - started
    usage = result.raw.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or len(result.text.split()))
    tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "completion_tokens": completion_tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "sample": result.text.strip(),
    }


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for notebook dataframe helpers. "
            'Install with: python -m pip install -e ".[notebook]"'
        ) from exc
    return pd


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _report_progress(
    progress: bool | Callable[[int, int], None],
    index: int,
    total: int,
) -> None:
    if callable(progress):
        progress(index, total)
    elif progress:
        print(f"{index}/{total}", end="\r" if index < total else "\n")
