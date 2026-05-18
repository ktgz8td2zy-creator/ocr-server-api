from __future__ import annotations

import base64
import io
import logging
import re
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings
from app.models import (
    HealthResponse,
    OCRByBase64Request,
    OCRByUrlRequest,
    OCRItem,
    OCRResponse,
)
from app.ocr import OCREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="OCR Server API",
    description="OCR API powered by PP-OCRv5 — supports PaddleOCR (quality) and ppocr-lite (fast) engines",
    version="2.0.0",
)

# ── helpers ──────────────────────────────────────────────────────────────────

_MAX_BYTES = settings.max_image_size_mb * 1024 * 1024


def _run_ocr(image_input) -> OCRResponse:
    engine = OCREngine.get()
    raw_results = engine.run(image_input)
    items = []
    for r in raw_results:
        x, y, w, h = r.box
        items.append(
            OCRItem(
                text=r.text,
                score=round(r.score, 4),
                box={"x": x, "y": y, "width": w, "height": h},
            )
        )
    return OCRResponse(
        success=True,
        count=len(items),
        results=items,
        text="\n".join(i.text for i in items),
    )


async def _fetch_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=settings.url_fetch_timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.content
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, f"Image exceeds {settings.max_image_size_mb}MB limit")
    return data


def _validate_image(data: bytes) -> None:
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, f"Image exceeds {settings.max_image_size_mb}MB limit")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "Invalid or unsupported image format")


def _decode_base64(data_uri: str) -> bytes:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_uri, re.DOTALL)
    b64 = match.group(1) if match else data_uri
    try:
        return base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 encoding")


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    engine = OCREngine.get()
    return HealthResponse(
        status="ok" if engine.ready else "not_ready",
        engine_ready=engine.ready,
        engine_name=engine.engine_name,
    )


@app.post(
    "/ocr/file",
    response_model=OCRResponse,
    tags=["OCR"],
    summary="OCR from uploaded image file",
)
async def ocr_from_file(file: UploadFile = File(...)):
    data = await file.read()
    _validate_image(data)

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or ".png").suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return _run_ocr(tmp.name)


@app.post(
    "/ocr/url",
    response_model=OCRResponse,
    tags=["OCR"],
    summary="OCR from image URL",
)
async def ocr_from_url(body: OCRByUrlRequest):
    data = await _fetch_image(body.url)
    _validate_image(data)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return _run_ocr(tmp.name)


@app.post(
    "/ocr/base64",
    response_model=OCRResponse,
    tags=["OCR"],
    summary="OCR from base64-encoded image",
)
async def ocr_from_base64(body: OCRByBase64Request):
    data = _decode_base64(body.image)
    _validate_image(data)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return _run_ocr(tmp.name)


# ── error handler ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
