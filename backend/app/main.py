"""FastAPI application for the Truth Trails local analysis prototype."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
import logging
from pathlib import Path
import tempfile
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    CORS_ORIGINS,
    IMAGE_CONTENT_TYPES,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    VIDEO_CONTENT_TYPES,
)
from app.inference import (
    ModelUnavailableError,
    analyze_image as classify_image,
    get_model,
    model_status,
)
from app.video_analysis import (
    NoFacesDetectedError,
    VideoTooLongError,
    analyze_video as classify_video,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthtrails.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(get_model)
    except ModelUnavailableError as exc:
        app.state.model_error = str(exc)
        logger.error("Model is unavailable at startup: %s", exc)
    else:
        app.state.model_error = None
    yield


app = FastAPI(
    title="Truth Trails API",
    description=(
        "Experimental real-versus-fake face image analysis API. "
        "Video analysis reuses the still-image baseline and is not validated."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _clean_filename(filename: str | None) -> str:
    raw_name = (filename or "upload").replace("\\", "/").split("/")[-1]
    cleaned = "".join(char for char in raw_name if char.isprintable()).strip()
    return cleaned[:180] or "upload"


def _source_record(source_reference: str | None) -> dict:
    value = (source_reference or "").strip()[:500] or None
    return {
        "reference": value,
        "verified": False,
        "note": "User-supplied reference; Truth Trails has not verified its source.",
    }


def _analysis_metadata(file: UploadFile, size_bytes: int, content: bytes) -> dict:
    return {
        "analysis_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file": {
            "filename": _clean_filename(file.filename),
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "sha256": sha256(content).hexdigest(),
        },
    }


async def _save_upload_limited(upload: UploadFile, destination: Path, max_bytes: int) -> tuple[int, str]:
    total = 0
    digest = sha256()
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Video exceeds the 100 MB upload limit.",
                )
            digest.update(chunk)
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return total, digest.hexdigest()


def _handle_inference_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelUnavailableError):
        return HTTPException(
            status_code=503,
            detail="The analysis model is unavailable. Check the backend setup.",
        )
    if isinstance(exc, NoFacesDetectedError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, VideoTooLongError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Analysis failed")
    return HTTPException(
        status_code=500,
        detail="Analysis failed. Check the backend logs.",
    )


@app.get("/")
def root():
    return {
        "name": "Truth Trails API",
        "status": "running",
        "docs": "/docs",
        "image_endpoint": "/analyze/image",
        "video_endpoint": "/analyze/video",
    }


@app.get("/health")
def health():
    state = model_status()
    if not state["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                **state,
                "detail": "The model is not loaded.",
            },
        )
    return {"status": "healthy", **state}


@app.post("/analyze/image")
@app.post("/analyze", include_in_schema=False)
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    source_reference: str | None = Form(default=None, max_length=500),
):
    try:
        if file.content_type not in IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Upload a JPEG, PNG, or WebP image.",
            )

        content = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        try:
            result = await run_in_threadpool(classify_image, content)
        except Exception as exc:
            raise _handle_inference_error(exc) from exc

        result.update(_analysis_metadata(file, len(content), content))
        result["analysis_type"] = "image"
        result["analysis_status"] = "completed"
        result["source_record"] = _source_record(source_reference)
        return result
    finally:
        await file.close()


@app.post("/analyze/video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    source_reference: str | None = Form(default=None, max_length=500),
):
    try:
        if file.content_type not in VIDEO_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Upload an MP4, MOV, AVI, WebM, or MKV video.",
            )

        suffix = Path(_clean_filename(file.filename)).suffix.lower()
        expected_suffix = VIDEO_CONTENT_TYPES[file.content_type]
        if suffix and suffix != expected_suffix:
            raise HTTPException(
                status_code=415,
                detail="The file extension does not match its video content type.",
            )

        with tempfile.TemporaryDirectory(prefix="truth-trails-video-") as temp_dir:
            temp_path = Path(temp_dir) / f"upload{expected_suffix}"
            size_bytes, digest = await _save_upload_limited(
                file,
                temp_path,
                MAX_VIDEO_SIZE_BYTES,
            )

            try:
                result = await run_in_threadpool(classify_video, temp_path)
            except Exception as exc:
                raise _handle_inference_error(exc) from exc

        result.update({
            "analysis_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_status": "completed",
            "file": {
                "filename": _clean_filename(file.filename),
                "content_type": file.content_type,
                "size_bytes": size_bytes,
                "sha256": digest,
            },
            "source_record": _source_record(source_reference),
        })
        return result
    finally:
        await file.close()
