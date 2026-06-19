from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
FILENAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"


def main() -> None:
    repo_id = os.getenv("ORC_LLM_HF_REPO", REPO_ID)
    filename = os.getenv("ORC_LLM_MODEL_FILE", FILENAME)
    local_dir = Path(os.getenv("ORC_LLM_MODEL_DIR", "models"))
    local_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()

