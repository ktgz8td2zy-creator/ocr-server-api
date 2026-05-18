from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- Response schemas ----------

class BBoxOut(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OCRItem(BaseModel):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    box: BBoxOut


class OCRResponse(BaseModel):
    success: bool = True
    count: int
    results: list[OCRItem]
    text: str = Field(default="", description="All text joined by newlines for convenience")


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    engine_name: str = ""
    model_cache_dir: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


# ---------- Request schemas ----------

class OCRByUrlRequest(BaseModel):
    url: str = Field(description="Image URL to fetch and run OCR on")


class OCRByBase64Request(BaseModel):
    image: str = Field(description="Base64-encoded image data (with or without data:image/... prefix)")
