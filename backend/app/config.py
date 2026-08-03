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
SD_MODEL_DIR = MODELS_DIR / "stable-diffusion"

DB_PATH = DATA_DIR / "text-to-video.sqlite3"
DEFAULT_SD_MODEL = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")


def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, TEMP_DIR, MODELS_DIR, SD_MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
