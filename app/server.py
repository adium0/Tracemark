"""FastAPI backend for the Integrity Theory demo.

GET  /             -> the frontend (static/index.html)
POST /api/analyze  -> multipart image upload, returns the verdict and explanations as JSON
GET  /api/health   -> used by the launcher to know when the server is ready
"""
import io
import json
import re
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import explain
from .model import CLASS_NAMES, load_model, preprocess_upload

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "integrity_model_v5.1.pth"
STATIC = ROOT / "static"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024        # 20 MB
MAX_PIXELS = 40_000_000                    # 40 megapixels
# Formats Pillow is allowed to try. The extension and the browser's content type are
# ignored: the bytes decide. ("JPEG" also covers MPO, the multi-picture JPEG many phones write.)
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"]
FRIENDLY_FORMATS = "JPEG, PNG, WebP, BMP, GIF or TIFF"

# Decompression-bomb protection: anything over 40 MP is refused. Pillow warns between
# 1x and 2x the limit and raises above 2x; we turn the warning into an error too.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

STATE = {}
MODEL_LOCK = threading.Lock()   # one analysis at a time: Grad-CAM attaches hooks to the shared model


@asynccontextmanager
async def lifespan(_app):
    started = time.perf_counter()
    model = load_model(WEIGHTS)
    with torch.no_grad():   # warm-up pass so the first real request is not slower
        model(torch.zeros(1, 3, 224, 224), torch.zeros(1, 1024), torch.zeros(1, 2))
    STATE["model"] = model
    print(f"[integrity-demo] Model loaded from {WEIGHTS.name} in {(time.perf_counter() - started):.1f}s (CPU)")
    yield
    STATE.clear()


class UploadSizeLimit:
    """Refuse oversized uploads from the Content-Length header, before they are parsed.

    The body is still read and discarded (up to a sane cap) so the client receives the
    JSON error instead of a dropped connection.
    """

    def __init__(self, app, limit, drain_cap=256 * 1024 * 1024):
        self.app = app
        self.limit = limit
        self.drain_cap = drain_cap

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/api/analyze":
            length = None
            for name, value in scope.get("headers", []):
                if name == b"content-length":
                    try:
                        length = int(value)
                    except ValueError:
                        length = None
            # Allow some room for the multipart wrapper around the file.
            if length is not None and length > self.limit + 1024 * 1024:
                if length <= self.drain_cap:
                    more = True
                    while more:
                        message = await receive()
                        more = message.get("more_body", False) and message["type"] == "http.request"
                body = json.dumps(_error_body("too_large", _too_large_message(length, approx=True))).encode()
                await send({"type": "http.response.start", "status": 413, "headers": [
                    (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


app = FastAPI(title="Integrity Theory demo", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(UploadSizeLimit, limit=MAX_UPLOAD_BYTES)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(RequestValidationError)
async def _no_file(_request, _exc):
    return _error(400, "no_file", "No image was uploaded. Send it as a multipart form field named 'file'.")


# --- errors ------------------------------------------------------------------------------

def _error_body(code, message):
    return {"error": {"code": code, "message": message}}


def _error(status, code, message):
    return JSONResponse(status_code=status, content=_error_body(code, message))


def _mb(n):
    return n / (1024 * 1024)


def _too_large_message(size, approx=False):
    about = "about " if approx else ""
    return (f"This file is {about}{_mb(size):.1f} MB. The limit is 20 MB. "
            "Try a smaller or more compressed copy of the image.")


def _not_an_image_message(filename, data):
    head = data[:32]
    if head[4:8] == b"ftyp" and any(t in head[8:12] for t in (b"heic", b"heix", b"mif1", b"msf1", b"hevc", b"avif")):
        return ("This looks like a HEIC/HEIF or AVIF photo, which this demo can't read. "
                "Export it as JPEG or PNG and try again.")
    parts = ["This file isn't a supported image."]
    if head.startswith(b"%PDF"):
        parts.append("It looks like a PDF document.")
    else:
        sample = data[:4096]
        try:
            sample.decode("utf-8")
            if b"\x00" not in sample:
                parts.append("It looks like a text file.")
        except UnicodeDecodeError:
            pass
    ext = Path(filename or "").suffix.lower()
    if ext:
        parts.append(f"Its name ends in “{ext}”, but the contents are what count, "
                     f"and they are not a {FRIENDLY_FORMATS} image.")
    else:
        parts.append(f"Its contents are not a {FRIENDLY_FORMATS} image.")
    return " ".join(parts)


def _bomb_message(exc):
    m = re.search(r"\((\d+) pixels\)", str(exc))
    if m:
        mp = int(m.group(1)) / 1_000_000
        return (f"This image declares {mp:.1f} megapixels. The limit is 40 megapixels, to protect "
                "your computer from images that expand to a huge size in memory. Resize it and try again.")
    return "This image declares more than 40 megapixels, which is over the limit. Resize it and try again."


# --- routes ------------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok" if "model" in STATE else "loading", "app": "integrity-demo",
            "model": "IntegrityFusionModelV4", "weights": WEIGHTS.name}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    t_start = time.perf_counter()

    # 1. Size
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        return _error(400, "empty", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        file.file.seek(0, io.SEEK_END)
        return _error(413, "too_large", _too_large_message(file.file.tell()))

    # 2. Real format (from the bytes), pixel limit and full decode
    try:
        probe = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        return _error(413, "too_many_pixels", _bomb_message(exc))
    except UnidentifiedImageError:
        return _error(415, "not_an_image", _not_an_image_message(file.filename, data))
    except Exception as exc:   # header present but unreadable
        return _error(422, "corrupt", f"The file could not be read as an image: {exc}.")

    fmt = probe.format
    try:
        probe.load()                     # decode every pixel now, so damage shows up here
        img = probe.convert("RGB")       # exactly what predict_v4.preprocess_image() does
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        return _error(413, "too_many_pixels", _bomb_message(exc))
    except Exception as exc:
        return _error(422, "corrupt", (
            f"The file is a {fmt} image, but its data is damaged or incomplete, so it can't be decoded "
            f"({exc}). Try re-saving or re-downloading it."))

    width, height = img.size
    exif = img.getexif()
    file_info = {
        "name": file.filename, "bytes": len(data), "format": fmt, "mode": probe.mode,
        "width": width, "height": height, "megapixels": width * height / 1_000_000,
    }
    readable = {271: "Make", 272: "Model", 305: "Software", 306: "DateTime"}
    exif_tags = {}
    for tag, name in readable.items():
        value = exif.get(tag)
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        if value:
            exif_tags[name] = str(value).strip("\x00 ").strip()[:80]
    markers = [m for m in ("c2pa", "JUMBF") if m.encode() in data]

    model = STATE["model"]
    with MODEL_LOCK:
        # 3. Model inputs and verdict
        t_inf = time.perf_counter()
        prep = preprocess_upload(img, data)
        verdict = explain.predict(model, prep)
        inference_ms = (time.perf_counter() - t_inf) * 1000

        # 4. Explanations
        t_exp = time.perf_counter()
        abl = explain.ablation(model, prep)
        cam = explain.grad_cam(model, prep)

    cam_stats = explain.gradcam_stats(cam["cam"])
    display = explain.fit_for_display(img)
    overlay, heatmap = explain.gradcam_images(display, cam["cam"])
    freq = explain.frequency_images(prep)
    ela = explain.ela_image(img)

    provenance = {
        "exif": prep["has_exif"] == 1.0,
        "c2pa": prep["has_c2pa"] == 1.0,
        "c2pa_markers": markers,
        "c2pa_offset": min((data.find(m.encode()) for m in markers), default=None),
        "exif_count": len(exif),
        "exif_tags": exif_tags,
        "exif_orientation": exif.get(0x0112),
    }
    summary = explain.build_summary(verdict, provenance, abl, cam_stats, file_info)
    explain_ms = (time.perf_counter() - t_exp) * 1000

    result = {
        "file": file_info,
        "model": {"name": "IntegrityFusionModelV4", "weights": WEIGHTS.name, "device": "cpu"},
        "verdict": {
            "label": CLASS_NAMES[verdict["class_index"]],
            "is_fake": verdict["class_index"] == 1,
            **verdict,
        },
        "provenance": provenance,
        "ablation": abl,
        "gradcam": {
            "overlay_png": explain.png_b64(overlay),
            "heatmap_png": explain.png_b64(heatmap),
            "image_png": explain.png_b64(display),
            "layer": "spatial_branch.stage3",
            "grid": cam["grid"],
            "fake_logit": cam["fake_logit"],
            "raw_max": cam["raw_max"],
            **cam_stats,
        },
        "frequency": freq,
        "ela": ela,
        "summary": summary,
    }
    result["timing_ms"] = {
        "inference": inference_ms,
        "explanations": explain_ms,
        "total": (time.perf_counter() - t_start) * 1000,
    }
    return result
