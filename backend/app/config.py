from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Text-to-Image-to-Video Generator"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TEXT_TO_VIDEO_DATA_DIR", BASE_DIR / "data")).resolve()

OUTPUTS_DIR = DATA_DIR / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = DATA_DIR / "models"

DB_PATH = DATA_DIR / "text-to-video.sqlite3"
DEFAULT_Z_IMAGE_MODEL = os.environ.get("Z_IMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEFAULT_Z_IMAGE_GGUF_URL = os.environ.get(
    "Z_IMAGE_GGUF_URL",
    "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q4_K_M.gguf",
)
LOCAL_Z_IMAGE_DIR = Path(os.environ.get("Z_IMAGE_MODEL_PATH", "/content/models/Z-Image-Turbo"))
LOCAL_Z_IMAGE_GGUF = Path(os.environ.get("Z_IMAGE_GGUF_PATH", "/content/models/z-image-turbo-Q4_K_M.gguf"))


def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, TEMP_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

