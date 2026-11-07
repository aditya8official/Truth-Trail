"""Bounded, experimental frame sampling for video uploads.

This reuses the image baseline on Haar-cascade face crops. The model was not
trained or validated on video frame crops, so video scores are exploratory.
"""

from pathlib import Path

import cv2
import numpy as np

from app.config import (
    MAX_FACES_PER_FRAME,
    MAX_SAMPLED_VIDEO_FRAMES,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FRAME_DIMENSION,
)
from app.inference import classify_video_score, predict_rgb_images


class NoFacesDetectedError(ValueError):
    """Raised when no usable face is found in sampled frames."""


class VideoTooLongError(ValueError):
    """Raised when video metadata exceeds the configured duration limit."""


VIDEO_WARNING = (
    "Video analysis is experimental. The image model was trained on still "
    "face images, not video frame crops; frame aggregation is unvalidated. "
    "Audio is not analyzed. This result is not proof of authenticity."
)


def _crop_with_margin(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(frame_bgr.shape[1], x + width + margin_x)
    y1 = min(frame_bgr.shape[0], y + height + margin_y)
    crop_bgr = frame_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _sample_frame_indexes(frame_count: int, duration: float) -> list[int]:
    sample_count = min(
        MAX_SAMPLED_VIDEO_FRAMES,
        frame_count,
        max(1, int(np.ceil(duration)) + 1),
    )
    return np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()


def analyze_video(video_path: Path) -> dict:
    """Sample video frames, detect faces, and return a caveated summary."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded file could not be decoded as a supported video.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
            raise ValueError("The video has invalid or unavailable metadata.")
        if max(frame_width, frame_height) > MAX_VIDEO_FRAME_DIMENSION:
            raise ValueError("Video dimensions exceed the supported limit.")

        duration = frame_count / fps
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise VideoTooLongError(
                f"Video exceeds the {MAX_VIDEO_DURATION_SECONDS}-second limit."
            )

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise RuntimeError("The bundled face detector could not be loaded.")

        sample_indexes = _sample_frame_indexes(frame_count, duration)
        sampled_frames = 0
        face_records: list[dict] = []
        face_crops: list[np.ndarray] = []
        frame_records: list[dict] = []

        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue

            sampled_frames += 1
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )
            boxes = sorted(boxes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)
            boxes = boxes[:MAX_FACES_PER_FRAME]
            timestamp = round(frame_index / fps, 2)

            frame_record = {
                "timestamp_seconds": timestamp,
                "faces": [],
            }
            frame_records.append(frame_record)

            for x, y, width, height in boxes:
                crop = _crop_with_margin(
                    frame_bgr,
                    (int(x), int(y), int(width), int(height)),
                )
                face_crops.append(crop)
                face_records.append({
                    "frame_index": len(frame_records) - 1,
                    "box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                })

        if not face_crops:
            raise NoFacesDetectedError(
                "No faces were detected in the sampled video frames."
            )

        face_scores = predict_rgb_images(face_crops)
        per_frame_scores: list[float] = []

        for face_record, score in zip(face_records, face_scores):
            fake_score = float(score)
            face_result = {
                **face_record["box"],
                "predicted_label": "fake" if fake_score >= 0.5 else "real",
                "fake_score": round(fake_score, 6),
                "real_score": round(1.0 - fake_score, 6),
            }
            frame_records[face_record["frame_index"]]["faces"].append(face_result)

        for frame in frame_records:
            scores = [face["fake_score"] for face in frame["faces"]]
            if scores:
                frame_fake_score = float(np.mean(scores))
                frame["fake_score"] = round(frame_fake_score, 6)
                frame["real_score"] = round(1.0 - frame_fake_score, 6)
                frame["predicted_label"] = "fake" if frame_fake_score >= 0.5 else "real"
                per_frame_scores.append(frame_fake_score)

        aggregate_fake_score = float(np.median(per_frame_scores))
        summary = classify_video_score(aggregate_fake_score)
        summary.update({
            "analysis_status": "completed",
            "analysis_type": "video",
            "duration_seconds": round(duration, 2),
            "frames_sampled": sampled_frames,
            "frames_analyzed": len(per_frame_scores),
            "faces_detected": len(face_crops),
            "aggregation_method": "Median of sampled frame mean fake scores; not validated.",
            "frame_results": [frame for frame in frame_records if frame["faces"]],
            "audio_analyzed": False,
            "warning": VIDEO_WARNING,
        })
        return summary
    finally:
        capture.release()
