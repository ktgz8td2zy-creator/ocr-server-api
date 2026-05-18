import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("OCR_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", os.getenv("OCR_PORT", "9418"))))
    workers: int = field(default_factory=lambda: int(os.getenv("OCR_WORKERS", "1")))

    # Engine selection: "auto", "paddleocr", "ppocr_lite"
    engine: str = field(default_factory=lambda: os.getenv("OCR_ENGINE", "auto"))

    # PaddleOCR device: "gpu:0", "cpu", "npu:0", etc.
    device: str | None = field(default_factory=lambda: os.getenv("OCR_DEVICE") or None)

    # ppocr_lite params
    det_thresh: float = field(default_factory=lambda: float(os.getenv("OCR_DET_THRESH", "0.3")))
    det_box_thresh: float = field(default_factory=lambda: float(os.getenv("OCR_DET_BOX_THRESH", "0.5")))
    det_unclip_ratio: float = field(default_factory=lambda: float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.6")))
    rec_batch_size: int = field(default_factory=lambda: int(os.getenv("OCR_REC_BATCH_SIZE", "24")))
    use_cls: bool = field(default_factory=lambda: os.getenv("OCR_USE_CLS", "false").lower() == "true")

    # ONNX providers (ppocr_lite only)
    providers: list[str] = field(default_factory=lambda: _parse_providers())

    # Image limits
    max_image_size_mb: int = field(default_factory=lambda: int(os.getenv("OCR_MAX_IMAGE_MB", "20")))

    # URL fetch timeout
    url_fetch_timeout: float = field(default_factory=lambda: float(os.getenv("OCR_URL_TIMEOUT", "30")))

    # Model cache dir
    model_cache_dir: str | None = field(default_factory=lambda: os.getenv("OCR_MODEL_CACHE_DIR"))


def _parse_providers() -> list[str]:
    val = os.getenv("OCR_ONNX_PROVIDERS", "")
    if val:
        return [p.strip() for p in val.split(",") if p.strip()]
    return []


settings = Settings()
