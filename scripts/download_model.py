from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://github.com/Glass-Half-Full/orc-llm/releases/download/"
    "liquidai-lfm2.5-q4-k-m/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
)
MODEL_SHA256 = "b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"
MODEL_FILENAME = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    url = os.getenv("ORC_LLM_MODEL_URL", MODEL_URL)
    filename = os.getenv("ORC_LLM_MODEL_FILE", MODEL_FILENAME)
    local_dir = Path(os.getenv("ORC_LLM_MODEL_DIR", "models"))
    expected_sha256 = os.getenv("ORC_LLM_MODEL_SHA256", MODEL_SHA256)
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / filename

    if destination.exists() and sha256(destination) == expected_sha256:
        print(destination)
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    urlretrieve(url, partial)
    actual_sha256 = sha256(partial)
    if actual_sha256 != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded model checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
