from __future__ import annotations

import queue
import threading
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, TypedDict


class ModelInfo(TypedDict):
    url: str
    filename: str
    subdir: str
    checksum: str



def file_checksum(path, chunk_size=1 << 23):
    q = queue.Queue(maxsize=4)  # buffer tối đa 4 chunk (~32MB RAM)

    def reader():
        with open(path, "rb", buffering=0) as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                q.put(chunk)
        q.put(None)  # sentinel

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    h = hashlib.sha256(usedforsecurity=False)
    while (chunk := q.get()) is not None:
        h.update(chunk)

    t.join()
    return h.hexdigest()

MODELS: Dict[str, ModelInfo] = {
    "diffusion_models": {
        "url": (
            "https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/"
            "resolve/main/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors"
        ),
        "filename": "svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-4steps.safetensors",
        "subdir": "models/diffusion_models",
        "checksum": "ec63cbacb29d80264a786f21f7b4b26f19147b3bdb25895966906d78bde0473b",
    },
    "text_encoders": {
        "url": (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/"
            "resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
        ),
        "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "subdir": "models/text_encoders",
        "checksum": "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"
    },
    "vae": {
        "url": (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/"
            "resolve/main/split_files/vae/qwen_image_vae.safetensors"
        ),
        "filename": "qwen_image_vae.safetensors",
        "subdir": "models/vae",
        "checksum": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"
    },
}


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "--tries=3", "--compression=auto",
        "--no-http-keep-alive", "-O", str(target), url],
        check=True,
    )


def ensure_models(base_dir: str | Path = ".") -> None:
    """
    Check if required model files exist under ``base_dir`` and
    download any that are missing.
    """

    base = Path(base_dir)

    for name, info in MODELS.items():
        target = base / info["subdir"] / info["filename"]
        if target.exists():
            if file_checksum(target).lower() == info["checksum"].lower():
                logging.info(f"Checksum matches for {name} at {target}, skipping download.")
                continue
            logging.info(f"Checksum mismatch for {name} at {target}, removing and re-downloading.")
        tmp = target.parent / f"{target.name}.tmp"
        logging.info(f"Downloading {name} model to {tmp}...")
        _download_file(info["url"], tmp)
        if file_checksum(tmp).lower() != info["checksum"].lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {name} after download: "
                f"expected {info['checksum']}"
            )
        tmp.replace(target)
        logging.info(f"Finished downloading {name}.")


if __name__ == "__main__":
    ensure_models()