"""Focused route tests that do not require a running server or model weights."""

import asyncio
from hashlib import sha256
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app import main
from app.video_analysis import _sample_frame_indexes


def make_upload(content: bytes, content_type: str, filename: str = "sample") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class ImageRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_image_returns_transparent_report_metadata(self):
        content = b"fake encoded image bytes"
        upload = make_upload(content, "image/png", "face.png")
        prediction = {
            "predicted_label": "real",
            "fake_score": 0.2,
            "real_score": 0.8,
            "threshold": 0.5,
            "model": "test model",
            "model_version": "abc123",
            "experimental": True,
            "score_semantics": "uncalibrated",
            "warning": "test warning",
        }

        with patch.object(main, "classify_image", return_value=prediction):
            result = await main.analyze_image_endpoint(
                upload,
                source_reference="https://example.test/source",
            )

        self.assertEqual(result["analysis_type"], "image")
        self.assertEqual(result["analysis_status"], "completed")
        self.assertEqual(result["file"]["sha256"], sha256(content).hexdigest())
        self.assertFalse(result["source_record"]["verified"])
        self.assertTrue(upload.file.closed)

    async def test_rejects_unsupported_content_type(self):
        with self.assertRaises(HTTPException) as caught:
            await main.analyze_image_endpoint(make_upload(b"x", "text/plain"))
        self.assertEqual(caught.exception.status_code, 415)

    async def test_rejects_empty_image(self):
        with self.assertRaises(HTTPException) as caught:
            await main.analyze_image_endpoint(make_upload(b"", "image/png"))
        self.assertEqual(caught.exception.status_code, 400)

    async def test_rejects_image_over_size_limit(self):
        with patch.object(main, "MAX_IMAGE_SIZE_BYTES", 4):
            with self.assertRaises(HTTPException) as caught:
                await main.analyze_image_endpoint(
                    make_upload(b"12345", "image/png"),
                )
        self.assertEqual(caught.exception.status_code, 413)

    async def test_maps_invalid_image_to_bad_request(self):
        with patch.object(main, "classify_image", side_effect=ValueError("bad image")):
            with self.assertRaises(HTTPException) as caught:
                await main.analyze_image_endpoint(make_upload(b"bytes", "image/png"))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, "bad image")

    async def test_maps_missing_model_to_service_unavailable(self):
        with patch.object(main, "classify_image", side_effect=main.ModelUnavailableError()):
            with self.assertRaises(HTTPException) as caught:
                await main.analyze_image_endpoint(make_upload(b"bytes", "image/png"))
        self.assertEqual(caught.exception.status_code, 503)

    async def test_health_reports_model_state_not_hardcoded_success(self):
        with patch.object(main, "model_status", return_value={
            "model_loaded": False,
            "model_available": False,
            "model": "MobileNetV2 Fine-Tuned",
            "model_version": None,
        }):
            response = main.health()
        self.assertEqual(response.status_code, 503)


class VideoRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_video_sampling_spans_full_clip(self):
        indexes = _sample_frame_indexes(frame_count=900, duration=30.0)
        self.assertEqual(len(indexes), 12)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 899)
        self.assertEqual(len(indexes), len(set(indexes)))

    def test_video_sampling_does_not_duplicate_short_clip_frames(self):
        indexes = _sample_frame_indexes(frame_count=5, duration=5.0)
        self.assertEqual(indexes, [0, 1, 2, 3, 4])

    async def test_rejects_unsupported_video_type(self):
        with self.assertRaises(HTTPException) as caught:
            await main.analyze_video_endpoint(make_upload(b"x", "text/plain", "x.txt"))
        self.assertEqual(caught.exception.status_code, 415)

    async def test_rejects_mismatched_video_extension(self):
        with self.assertRaises(HTTPException) as caught:
            await main.analyze_video_endpoint(
                make_upload(b"x", "video/mp4", "x.mov"),
            )
        self.assertEqual(caught.exception.status_code, 415)

    async def test_rejects_video_over_size_limit(self):
        with patch.object(main, "MAX_VIDEO_SIZE_BYTES", 4):
            with self.assertRaises(HTTPException) as caught:
                await main.analyze_video_endpoint(
                    make_upload(b"12345", "video/mp4", "clip.mp4"),
                )
        self.assertEqual(caught.exception.status_code, 413)

    async def test_maps_video_without_faces_to_unprocessable(self):
        with patch.object(
            main,
            "classify_video",
            side_effect=main.NoFacesDetectedError("No faces found."),
        ):
            with self.assertRaises(HTTPException) as caught:
                await main.analyze_video_endpoint(
                    make_upload(b"video bytes", "video/mp4", "clip.mp4"),
                )
        self.assertEqual(caught.exception.status_code, 422)

    async def test_video_file_is_temporary_and_report_has_source_record(self):
        upload = make_upload(b"video bytes", "video/mp4", "clip.mp4")
        result_stub = {
            "analysis_type": "video",
            "predicted_label": "fake",
            "fake_score": 0.7,
            "real_score": 0.3,
            "warning": "experimental",
        }

        with patch.object(main, "classify_video", return_value=result_stub) as classifier:
            result = await main.analyze_video_endpoint(
                upload,
                source_reference="Source supplied by user",
            )

        temp_path = classifier.call_args.args[0]
        self.assertFalse(temp_path.exists())
        self.assertEqual(result["file"]["size_bytes"], len(b"video bytes"))
        self.assertFalse(result["source_record"]["verified"])
        self.assertTrue(upload.file.closed)


if __name__ == "__main__":
    unittest.main()
