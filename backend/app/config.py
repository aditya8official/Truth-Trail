"""Runtime configuration for the Truth Trails API."""

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BACKEND_ROOT / "models" / "mobilenetv2_finetuned_best.keras"
MODEL_PATH = Path(os.environ.get("TRUTH_TRAILS_MODEL_PATH", DEFAULT_MODEL_PATH))

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 30
MAX_SAMPLED_VIDEO_FRAMES = 12
MAX_FACES_PER_FRAME = 3
MAX_VIDEO_FRAME_DIMENSION = 4096

IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
}

DEFAULT_CORS_ORIGINS = {
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://aditya8official.github.io",
}

configured_origins = os.environ.get("TRUTH_TRAILS_CORS_ORIGINS")
if configured_origins:
    CORS_ORIGINS = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
else:
    CORS_ORIGINS = sorted(DEFAULT_CORS_ORIGINS)
