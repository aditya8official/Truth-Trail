"""Model loading and image inference for the experimental baseline."""

from hashlib import sha256
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf

from app.config import MODEL_PATH


IMAGE_SIZE = (128, 128)
MODEL_NAME = "MobileNetV2 Fine-Tuned"
FAKE_THRESHOLD = 0.5
SCORE_SEMANTICS = "Uncalibrated sigmoid model score; not a probability."
PREDICTION_WARNING = (
    "Experimental image classifier. This prediction may be incorrect and "
    "must not be treated as proof of authenticity."
)

_model = None
_model_version = None
_model_lock = Lock()


class ModelUnavailableError(RuntimeError):
    """Raised when the saved model is missing or cannot be loaded."""


def _model_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_model():
    """Load the trained model once and return the shared instance."""
    global _model, _model_version

    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise ModelUnavailableError("The trained model file is unavailable.")

                try:
                    loaded_model = tf.keras.models.load_model(
                        MODEL_PATH,
                        compile=False,
                    )
                    _model_version = _model_digest(MODEL_PATH)[:12]
                    _model = loaded_model
                except Exception as exc:
                    raise ModelUnavailableError(
                        "The trained model could not be loaded."
                    ) from exc

    return _model


def model_status() -> dict:
    """Return health information without exposing local filesystem paths."""
    return {
        "model_loaded": _model is not None,
        "model_available": MODEL_PATH.is_file(),
        "model": MODEL_NAME,
        "model_version": _model_version,
    }


def _preprocess_rgb_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("The uploaded file is not a valid image.")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    image = image.astype(np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def predict_rgb_images(images: list[np.ndarray]) -> np.ndarray:
    """Predict scores for already decoded RGB images, such as video crops."""
    if not images:
        raise ValueError("No face images were provided for analysis.")

    batch = np.stack([_preprocess_rgb_image(image) for image in images], axis=0)
    model = get_model()

    with _model_lock:
        raw_scores = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

    if raw_scores.size != len(images):
        raise RuntimeError("The model returned an unexpected number of scores.")
    if not np.all(np.isfinite(raw_scores)) or np.any((raw_scores < 0) | (raw_scores > 1)):
        raise RuntimeError("The model returned an invalid score.")

    return raw_scores.astype(float)


def _classification(fake_score: float) -> dict:
    return {
        "predicted_label": "fake" if fake_score >= FAKE_THRESHOLD else "real",
        "fake_score": round(fake_score, 6),
        "real_score": round(1.0 - fake_score, 6),
        "threshold": FAKE_THRESHOLD,
        "model": MODEL_NAME,
        "model_version": _model_version,
        "experimental": True,
        "score_semantics": SCORE_SEMANTICS,
        "warning": PREDICTION_WARNING,
    }


def analyze_image(image_bytes: bytes) -> dict:
    """Run one image through the baseline model."""
    scores = predict_rgb_images([_decode_image_rgb(image_bytes)])
    return _classification(float(scores[0]))


def _decode_image_rgb(image_bytes: bytes) -> np.ndarray:
    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The uploaded file is not a valid image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode and preprocess one uploaded image into a model batch."""
    return np.expand_dims(_preprocess_rgb_image(_decode_image_rgb(image_bytes)), axis=0)


def classify_video_score(fake_score: float) -> dict:
    """Format an aggregate video score using the same fixed threshold."""
    return _classification(fake_score)
