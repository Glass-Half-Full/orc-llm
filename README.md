# orc-llm

FastAPI wrapper for the LiquidAI `LFM2.5-1.2B-Instruct` GGUF model.

This repo is set up for a small local machine: CPU-only works, and an NVIDIA RTX 2050 can offload layers through `llama-cpp-python` when installed with CUDA support. The default model is the practical quantized file:

`models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

The model is downloaded from a GitHub Release asset, not Git LFS and not `codeload.github.com`.

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
export ORC_LLM_N_GPU_LAYERS=999
export ORC_LLM_N_THREADS=8
```

For RTX 2050-class hardware, keep context moderate (`4096` or `8192`) unless you have a specific reason to spend memory on a longer prompt.

## Chat

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d @examples/chat_request.json
```

## Dataframe Policy Check

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
