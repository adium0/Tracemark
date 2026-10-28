"""Acceptance checks for the Integrity Theory demo.

Run from the integrity-demo folder, after run.sh / run.bat has installed everything once:

    Mac/Linux:  .venv/bin/python checks/acceptance.py [your_image.jpg ...]
    Windows:    .venv\\Scripts\\python checks\\acceptance.py [your_image.jpg ...]

It starts its own server on port 8765, builds test files in a temporary folder, and checks:
  1. P(fake) from /api/analyze matches the repo's original predict_v4.predict() with
     integrity_model_v5.1.pth to 4 decimal places (and the input tensors are bit-identical).
  2. A PNG with the bytes "c2pa" appended reports C2PA found.
  3. A 21 MB file, a .txt renamed to .jpg, an image declaring 50 megapixels and a
     truncated JPEG are all rejected with clear messages.
"""
import importlib.util
import io
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "integrity_model_v5.1.pth"
REFERENCE = ROOT / "checks" / "reference" / "predict_v4.py"
PORT = int(os.environ.get("CHECK_PORT", "8765"))
BASE = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, str(ROOT))

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


# --- HTTP helpers ------------------------------------------------------------------------

def upload(path, filename=None):
    data = Path(path).read_bytes()
    boundary = uuid.uuid4().hex
    name = filename or Path(path).name
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{BASE}/api/analyze", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for_server(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# --- reference: the repo's own predict_v4.predict() --------------------------------------

def reference_p_fake(img_path):
    """Run predict_v4.predict() unchanged and capture the softmax it computes."""
    import torch
    import torch.nn.functional as F

    spec = importlib.util.spec_from_file_location("predict_v4_reference", REFERENCE)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    captured = {}
    original_softmax = F.softmax

    def capturing_softmax(*args, **kwargs):
        out = original_softmax(*args, **kwargs)
        captured["probs"] = out.detach().clone()
        return out

    F.softmax = capturing_softmax
    try:
        devnull = io.StringIO()
        stdout, sys.stdout = sys.stdout, devnull
        try:
            ref.predict(str(img_path), model_path=str(WEIGHTS))
        finally:
            sys.stdout = stdout
    finally:
        F.softmax = original_softmax
    return float(captured["probs"][0, 1]), ref


def tensors_identical(img_path, ref):
    """preprocess_upload() (server) vs predict_v4.preprocess_image() (reference)."""
    import torch
    from PIL import Image
    from app.model import preprocess_upload

    a_img, a_freq, a_meta, _ = ref.preprocess_image(str(img_path))
    data = Path(img_path).read_bytes()
    b = preprocess_upload(Image.open(io.BytesIO(data)).convert("RGB"), data)
    return (torch.equal(a_img, b["image"]) and torch.equal(a_freq, b["freq"]) and torch.equal(a_meta, b["meta"]))


# --- test files --------------------------------------------------------------------------

def make_photo_like(path, seed=7, exif=True, size=(640, 480)):
    """A synthetic test picture (gradients, shapes, noise), saved as JPEG with EXIF."""
    from PIL import Image, ImageDraw, ImageFilter
    rnd = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(255 * x / w), int(255 * y / h), int(128 + 100 * ((x * y) % 97) / 97))
    draw = ImageDraw.Draw(img)
    for _ in range(25):
        x0, y0 = rnd.randrange(w), rnd.randrange(h)
        x1, y1 = x0 + rnd.randrange(20, 200), y0 + rnd.randrange(20, 200)
        colour = tuple(rnd.randrange(256) for _ in range(3))
        (draw.ellipse if rnd.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=colour)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    noise = Image.effect_noise(size, 18).convert("RGB")
    img = Image.blend(img, noise, 0.12)
    kwargs = {"quality": 92}
    if exif:
        ex = Image.Exif()
        ex[271] = "AcceptanceCheck"
        ex[272] = "Synthetic Camera"
        ex[305] = "checks/acceptance.py"
        kwargs["exif"] = ex.tobytes()
    img.save(path, "JPEG", **kwargs)


def make_c2pa_png(path, src_jpeg):
    from PIL import Image
    Image.open(src_jpeg).convert("RGB").save(path, "PNG")
    with open(path, "ab") as f:
        f.write(b"c2pa")


def make_21mb(path):
    """A real PNG of random noise, just over 21 MB (and well under 40 MP)."""
    from PIL import Image
    side = 2720
    Image.frombytes("RGB", (side, side), os.urandom(side * side * 3)).save(path, "PNG", compress_level=0)


def make_fake_jpg_text(path):
    Path(path).write_text("These are my lecture notes, not a picture.\n" * 20, encoding="utf-8")


def make_50mp_png(path, width=10000, height=5000):
    """A tiny PNG whose header declares width x height pixels (50 MP)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 64)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def make_truncated_jpeg(path, src_jpeg):
    data = Path(src_jpeg).read_bytes()
    Path(path).write_bytes(data[: len(data) // 2])


# --- main --------------------------------------------------------------------------------

def main():
    user_images = [Path(p) for p in sys.argv[1:]]
    tmp = Path(tempfile.mkdtemp(prefix="integrity-checks-"))
    photo = tmp / "test_photo.jpg"
    make_photo_like(photo)
    c2pa = tmp / "with_c2pa.png"
    make_c2pa_png(c2pa, photo)
    big = tmp / "big_21mb.png"
    make_21mb(big)
    txt = tmp / "notes.jpg"
    make_fake_jpg_text(txt)
    bomb = tmp / "declares_50mp.png"
    make_50mp_png(bomb)
    trunc = tmp / "truncated.jpg"
    make_truncated_jpeg(trunc, photo)

    env = dict(os.environ, PORT=str(PORT), NO_BROWSER="1")
    server = subprocess.Popen([sys.executable, "-m", "app.launch"], cwd=str(ROOT), env=env)
    try:
        print(f"Starting the server on {BASE} ...")
        if not wait_for_server():
            print("  Server did not start.")
            return 1

        print("\n1. Same answer as predict_v4.predict()")
        for img in [photo, c2pa] + user_images:
            status, res = upload(img)
            if status != 200:
                report(f"{img.name}: analyze", False, f"HTTP {status}: {res}")
                continue
            api = res["verdict"]["p_fake"]
            ref_p, ref = reference_p_fake(img)
            same = round(api, 4) == round(ref_p, 4) and abs(api - ref_p) < 5e-5
            report(f"{img.name}: P(fake) API {api:.6f} vs predict_v4 {ref_p:.6f}", same)
            report(f"{img.name}: model inputs bit-identical to preprocess_image()", tensors_identical(img, ref))

        print("\n2. C2PA marker")
        status, res = upload(c2pa)
        ok = status == 200 and res["provenance"]["c2pa"] is True
        report("PNG with 'c2pa' appended reports C2PA found", ok,
               f"c2pa={res.get('provenance', {}).get('c2pa')}, markers={res.get('provenance', {}).get('c2pa_markers')}")
        status, res = upload(photo)
        report("Plain JPEG reports no C2PA and EXIF found", status == 200 and not res["provenance"]["c2pa"] and res["provenance"]["exif"])

        print("\n3. Rejected uploads")
        cases = [
            (big, None, 413, "too_large", f"21 MB file ({big.stat().st_size / 1048576:.1f} MB)"),
            (txt, None, 415, "not_an_image", ".txt renamed to .jpg"),
            (bomb, None, 413, "too_many_pixels", "PNG declaring 10000 x 5000 = 50 MP"),
            (trunc, None, 422, "corrupt", "Truncated JPEG"),
        ]
        for path, name, want_status, want_code, label in cases:
            status, res = upload(path, name)
            err = res.get("error", {})
            ok = status == want_status and err.get("code") == want_code
            report(f"{label}: HTTP {status}, {err.get('code')}", ok, f"message: {err.get('message')}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed. Test files are in {tmp}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
