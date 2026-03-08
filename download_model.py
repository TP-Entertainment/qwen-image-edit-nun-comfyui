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


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


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