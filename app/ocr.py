from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    score: float
    box: tuple[int, int, int, int]  # x, y, width, height


class BaseOCREngine(ABC):
    @abstractmethod
    def run(self, image_input) -> list[OCRResult]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def ready(self) -> bool:
        return True


# ── RapidOCR backend (high quality, PP-OCRv5 ONNX) ───────────────────────────


class RapidOCREngine(BaseOCREngine):
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        logger.info("RapidOCR engine initialized")

    @property
    def name(self) -> str:
        return "rapidocr"

    def run(self, image_input) -> list[OCRResult]:
        result, _ = self._engine(image_input)
        if not result:
            return []
        items = []
        for item in result:
            box_pts, text, score = item[0], item[1], item[2]
            xs = [p[0] for p in box_pts]
            ys = [p[1] for p in box_pts]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            items.append(OCRResult(text=text, score=float(score), box=(x1, y1, x2 - x1, y2 - y1)))
        return items


# ── PaddleOCR backend (high quality, requires PaddlePaddle) ──────────────────


class PaddleOCREngine(BaseOCREngine):
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        kwargs: dict = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if settings.device:
            kwargs["device"] = settings.device

        self._engine = PaddleOCR(**kwargs)
        logger.info("PaddleOCR engine initialized (PP-OCRv5 server)")

    @property
    def name(self) -> str:
        return "paddleocr"

    def run(self, image_input) -> list[OCRResult]:
        results = self._engine.predict(image_input)
        items = []
        for res in results:
            data = res.res if hasattr(res, "res") else res
            if isinstance(data, dict):
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                boxes = data.get("rec_boxes", [])
                for i in range(len(texts)):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
                    if hasattr(box, "shape") and len(box.shape) == 2 and box.shape[0] == 4:
                        x1, y1 = int(box[0][0]), int(box[0][1])
                        x2, y2 = int(box[2][0]), int(box[2][1])
                    elif hasattr(box, "__len__") and len(box) >= 4:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        x1 = y1 = x2 = y2 = 0
                    items.append(OCRResult(text=texts[i], score=score, box=(x1, y1, x2 - x1, y2 - y1)))
        return items


# ── ppocr_lite backend (fast, mobile models) ─────────────────────────────────


class PPOCRLiteEngine(BaseOCREngine):
    def __init__(self) -> None:
        from ppocr_lite import PPOCRLite as _PPOCRLite

        kwargs: dict = {
            "det_thresh": settings.det_thresh,
            "det_box_thresh": settings.det_box_thresh,
            "det_unclip_ratio": settings.det_unclip_ratio,
            "rec_batch_size": settings.rec_batch_size,
            "use_cls": settings.use_cls,
        }
        if settings.providers:
            kwargs["providers"] = settings.providers

        self._engine = _PPOCRLite(**kwargs)
        logger.info("ppocr_lite engine initialized (PP-OCRv5 mobile)")

    @property
    def name(self) -> str:
        return "ppocr_lite"

    def run(self, image_input) -> list[OCRResult]:
        raw = self._engine.run(image_input)
        return [
            OCRResult(
                text=r.text,
                score=float(r.score),
                box=(int(r.box.x), int(r.box.y), int(r.box.width), int(r.box.height)),
            )
            for r in raw
        ]


# ── Engine factory ────────────────────────────────────────────────────────────

_ENGINES: dict[str, type[BaseOCREngine]] = {}


def _register(name: str, cls: type[BaseOCREngine]) -> None:
    _ENGINES[name] = cls


def _discover_engines() -> None:
    if _ENGINES:
        return
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401

        _register("rapidocr", RapidOCREngine)
    except ImportError:
        pass
    try:
        from ppocr_lite import PPOCRLite  # noqa: F401

        _register("ppocr_lite", PPOCRLiteEngine)
    except ImportError:
        pass
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        _register("paddleocr", PaddleOCREngine)
    except ImportError:
        pass


class OCREngine:
    """Thread-safe singleton that delegates to the configured backend."""

    _instance: OCREngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        _discover_engines()
        engine_name = settings.engine
        if engine_name == "auto":
            for pref in ("rapidocr", "paddleocr", "ppocr_lite"):
                if pref in _ENGINES:
                    engine_name = pref
                    break
            else:
                engine_name = next(iter(_ENGINES)) if _ENGINES else ""

        if engine_name not in _ENGINES:
            available = ", ".join(_ENGINES.keys()) or "none"
            raise RuntimeError(f"OCR engine '{engine_name}' not available. Available: {available}")

        logger.info("Using OCR engine: %s", engine_name)
        self._backend: BaseOCREngine = _ENGINES[engine_name]()
        self._engine_name = engine_name

    @classmethod
    def get(cls) -> OCREngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def ready(self) -> bool:
        return self._backend.ready

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def run(self, image_input) -> list[OCRResult]:
        return self._backend.run(image_input)
