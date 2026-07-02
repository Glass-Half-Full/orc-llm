# orc-llm

Local notebook and FastAPI wrapper for the LiquidAI `LFM2.5-1.2B-Instruct` GGUF model.

This repo is set up for a small local machine: CPU-only works, and an NVIDIA RTX 2050 can offload layers through `llama-cpp-python` when installed with CUDA support. The default model is the practical quantized file:

`models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

The model is downloaded from a GitHub Release asset, not Git LFS and not `codeload.github.com`.

After that one-time download, inference is fully local and does not require API keys or external model/API calls.

## Setup

```bash
git clone git@github.com:Glass-Half-Full/orc-llm.git
cd orc-llm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python scripts/download_model.py
```

For Jupyter/DataFrame usage, install the notebook extra instead:

```bash
python -m pip install -e ".[notebook]"
python scripts/download_model.py
```

For NVIDIA/CUDA acceleration, reinstall `llama-cpp-python` with CUDA support after activating the venv:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
```

On Windows PowerShell:

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE="1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## Run

```bash
bash scripts/run_api.sh
```

The API starts on `http://127.0.0.1:8000` by default. If the model file is missing, `scripts/run_api.sh` downloads it from the release asset automatically.

Useful environment variables:

```bash
export ORC_LLM_MODEL_PATH="models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
export ORC_LLM_N_CTX=4096
export ORC_LLM_N_GPU_LAYERS=-1
export ORC_LLM_N_BATCH=512
export ORC_LLM_N_UBATCH=512
export ORC_LLM_N_THREADS=8
```

For RTX 2050-class hardware, keep context moderate (`4096` first, `8192` only if you have memory headroom). Use `ORC_LLM_N_GPU_LAYERS=-1` with a CUDA-enabled `llama-cpp-python` build to offload all layers. If you hit GPU memory pressure, reduce `ORC_LLM_N_CTX` to `2048` or set fewer GPU layers. For CPU-only runs, set `ORC_LLM_N_GPU_LAYERS=0` and tune `ORC_LLM_N_THREADS` to your physical CPU cores.

## Jupyter DataFrame Usage

The notebook helper runs the GGUF model in the Python process. It does not call an external model API.

```python
import pandas as pd
from orc_llm import apply_prompt_to_column, benchmark_tokens_per_second, load_model

model = load_model(
    n_ctx=4096,
    n_gpu_layers=-1,
    n_batch=512,
    n_ubatch=512,
)

df = pd.DataFrame(
    {
        "id": [1, 2],
        "notes": [
            "Customer asked for a refund after the warranty expired.",
            "Customer shared payment card details in plain text.",
        ],
    }
)

prompt = """Classify this note as one of:
- meets_policy
- violates_policy
- needs_review

Note:
{text}

Return only the label."""

result = apply_prompt_to_column(
    df,
    text_column="notes",
    output_column="policy_result",
    prompt=prompt,
    model=model,
    max_tokens=32,
)

result
```

To measure your actual local speed:

```python
benchmark_tokens_per_second(model=model, max_tokens=32)
```

The full notebook-style example is in `examples/notebook_dataframe_usage.py`.

## Chat

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d @examples/chat_request.json
```

## Dataframe Policy Check API

```bash
curl -s http://127.0.0.1:8000/v1/evaluate-dataframe \
  -H 'content-type: application/json' \
  -d @examples/evaluate_dataframe_request.json
```

The dataframe endpoint accepts record-style rows and inserts the output column immediately after the assessed source column.

## Model Source And License

The GGUF is redistributed from:

`LiquidAI/LFM2.5-1.2B-Instruct-GGUF`

Default file:

`LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

Direct GitHub download:

`https://github.com/Glass-Half-Full/orc-llm/releases/download/liquidai-lfm2.5-q4-k-m/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

The model is under the Liquid AI LFM Open License v1.0. See `licenses/LFM_OPEN_LICENSE_v1.0.txt`.
