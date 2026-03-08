"""
## Diffusion Model
wget -O models/diffusion_models/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/resolve/main/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors

## Text Encoder
wget -O models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

## VAE
wget -O models/vae/qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TypedDict

import requests


class ModelInfo(TypedDict):
    url: str
    filename: str
    subdir: str


MODELS: Dict[str, ModelInfo] = {
    "diffusion_models": {
        "url": (
            "https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/"
            "resolve/main/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors"
        ),
        "filename": "svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors",
        "subdir": "models/diffusion_models",
    },
    "text_encoders": {
        "url": (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/"
            "resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
        ),
        "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "subdir": "models/text_encoders",
    },
    "vae": {
        "url": (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/"
            "resolve/main/split_files/vae/qwen_image_vae.safetensors"
        ),
        "filename": "qwen_image_vae.safetensors",
        "subdir": "models/vae",
    },
}


import threading
import requests
from pathlib import Path


def _download_file(url: str, target: Path, workers: int = 8) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    # Check if server supports range requests
    head = requests.head(url, timeout=30)
    content_length = int(head.headers.get("Content-Length", 0))
    accept_ranges = head.headers.get("Accept-Ranges", "none")

    if not content_length or accept_ranges == "none":
        # Fallback: single-threaded stream
        _download_single(url, target)
        return

    # Split file into chunks and download in parallel
    chunk_size = content_length // workers
    ranges = [
        (i * chunk_size, (i + 1) * chunk_size - 1 if i < workers - 1 else content_length - 1)
        for i in range(workers)
    ]

    parts: dict[int, bytes] = {}
    errors: list[Exception] = []
    lock = threading.Lock()

    def fetch(idx: int, start: int, end: int) -> None:
        try:
            resp = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=600,
            )
            resp.raise_for_status()
            with lock:
                parts[idx] = resp.content
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [
        threading.Thread(target=fetch, args=(i, s, e))
        for i, (s, e) in enumerate(ranges)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]

    with target.open("wb") as f:
        for i in range(workers):
            f.write(parts[i])


def _download_single(url: str, target: Path) -> None:
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=65536):  # 64KB vs 8KB
                if chunk:
                    f.write(chunk)


def ensure_models(base_dir: str | Path = ".") -> None:
    """
    Check if required model files exist under ``base_dir`` and
    download any that are missing.
    """

    base = Path(base_dir)

    for name, info in MODELS.items():
        target = base / info["subdir"] / info["filename"]
        if target.exists():
            continue

        print(f"Downloading {name} model to {target}...")
        _download_file(info["url"], target)
        print(f"Finished downloading {name}.")


if __name__ == "__main__":
    ensure_models()