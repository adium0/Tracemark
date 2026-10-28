# Integrity Theory: local demo

A small web app for the Integrity Theory deepfake detector. You upload an image and get the
model's verdict (real or AI-generated) plus an explanation of how it got there. Everything
runs on your own computer, on the CPU. Images are never sent anywhere.

> Student prototype. Quick-scan model (v5.1). Not reliable for spliced or edited images.

---

## Quick start

**Windows:** double-click `run.bat`.
If Windows shows "Windows protected your PC", click **More info → Run anyway** (this happens
with any script downloaded from the internet).

**macOS / Linux:** open a terminal in this folder and run:

```bash
./run.sh
```

If you get "permission denied", run `bash run.sh` instead.

The script installs what it needs, starts the server and opens
**http://127.0.0.1:8000** in your browser. Keep the terminal window open while you use the demo.
To stop it, press **Ctrl+C** in that window (or close it).

### What the first run does

1. **Finds a 64-bit Python 3.10, 3.11 or 3.12.** If none is installed (or the one you have can't make
   virtual environments), it downloads [uv](https://github.com/astral-sh/uv), a small standalone
   tool, and uses it to fetch a private Python 3.12. Both go into the `.tools` folder here and
   nothing else on your computer is changed.
2. **Creates a virtual environment** in `.venv`.
3. **Installs CPU-only PyTorch** (`torch`, `torchvision`) from `https://download.pytorch.org/whl/cpu`.
   This is the slow part: a few hundred MB. If that index can't be reached, it falls back to
   the standard package index (PyPI).
4. **Installs the rest** from `requirements.txt` (FastAPI, uvicorn, python-multipart, Pillow, numpy).
5. **Starts the server** and opens your browser once the model has loaded.

Later runs skip straight to step 5 and start in a few seconds. To reinstall from scratch,
delete the `.venv` folder. To remove the demo completely, delete this folder.

You need an internet connection for the first run only.

### Options

| Setting | Mac/Linux | Windows |
| --- | --- | --- |
| Use another port | `PORT=8001 ./run.sh` | `set PORT=8001` then `run.bat` |
| Don't open a browser | `NO_BROWSER=1 ./run.sh` | `set NO_BROWSER=1` then `run.bat` |
| Another PyTorch index | `TORCH_INDEX_URL=... ./run.sh` | `set TORCH_INDEX_URL=...` then `run.bat` |

---

## Using the demo

Drag an image onto the upload card, click **Choose an image**, or paste an image with Ctrl+V / Cmd+V.
The analysis starts straight away and usually takes under a second or two.

Accepted: JPEG, PNG, WebP, BMP, GIF and TIFF, up to **20 MB** and **40 megapixels**.
The server checks the file's actual contents with Pillow; the file name and extension are ignored.
These uploads are refused with a message that says why:

- files over 20 MB,
- files that aren't images (for example a `.txt` renamed to `.jpg`), and HEIC/AVIF photos,
- images that declare more than 40 megapixels (protection against "decompression bombs"),
- damaged or cut-off image files.

**Calm mode** (top right) turns off all animation. It switches on by itself if your device asks
for reduced motion. **Theme** switches between your device setting, light and dark.

---

## What each result section means

Every number and picture is computed from the real model and your file. Nothing is estimated
or made up, and the plain-English summary is built from fixed rules (no language model).

**02 Verdict.** The model's answer, with the two probabilities from its softmax output:
P(real) for class 0 and P(fake) for class 1. Confidence is the larger of the two. The bar
shows both, with a line at 50% (the decision point). Below it is the server's processing time in ms.

**03 How the model decided.** A short plain-English summary written by fixed rules over the
numbers below. For example: how close the score is to 50%, which input group mattered most
in the ablation, whether a C2PA or EXIF marker was found, where Grad-CAM is strongest, and
notes such as "the image was squeezed to a square".

**04 Evidence by branch.** The model has three groups of inputs that meet in its final layers
(the "fusion head"):

- **Spatial features**: 128 values from a small ResNet-style CNN that looks at the image (224 × 224).
- **Frequency features**: 64 values from a small network that reads part of the image's frequency spectrum.
- **Metadata flags**: 2 values, "has EXIF" and "has C2PA".

The fusion head is run three more times, each time with one group replaced by zeros. The bar
shows how much P(fake) changes:

- bar to the right (**pushed toward fake**): without this group P(fake) is lower, so the group raised it.
- bar to the left (**pushed toward real**): without this group P(fake) is higher, so the group lowered it.

The three effects don't have to add up to P(fake); groups can overlap or depend on each other.
If both metadata flags are 0, removing them changes nothing by definition.
"Show as a table" gives the exact numbers.

**05 Provenance.** The two metadata flags exactly as the model sees them:

- **EXIF**: does the image carry an EXIF block? If so, a few readable fields (camera make and model, software, date) are shown.
- **C2PA / JUMBF**: do the bytes `c2pa` or `JUMBF` appear anywhere in the file? C2PA "content
  credentials" are attached by several AI image generators, and also by some cameras and editors.
  This is a raw byte search, exactly as in `predict_v4.py`, not a full C2PA validation.

This model leans heavily on the C2PA flag (the original repo describes it as detecting
AI images "primarily using C2PA metadata"), so a C2PA marker usually pushes the verdict to AI-generated.

**06 Grad-CAM.** Which parts of the image the spatial branch used. The heatmap is Grad-CAM at
the output of `spatial_branch.stage3` (a 14 × 14 grid of 128 channels), made by back-propagating
the "fake" logit. Brighter areas raised the fake score more. If the verdict is "real", these
are the areas that pushed hardest toward fake. "Overlay" is smoothed; "Heatmap only" shows the
true 14 × 14 cells. The model squeezes every image into a 224 × 224 square and, like `predict_v4.py`,
ignores EXIF rotation, so this view shows the pixels as stored in the file.

**07 Frequency branch.** On the left, the full 256 × 256 log-magnitude spectrum,
log(1 + |FFT|), of the grayscale image. The centre holds the lowest frequencies. The outlined
32 × 32 square is the only part the frequency branch reads. On the right, that 32 × 32 crop
enlarged: these 1,024 values are the branch's input.

**08 ELA (extra forensic view, not an input to this model).** Error level analysis: the image is
re-saved as JPEG at quality 90 and the difference is amplified. Areas that re-compress
differently can stand out after editing. The model never sees this view; it is only there for
comparison. It is most useful for JPEG files; images that were never JPEG-compressed light up everywhere.

**09 Details.** File name, the format detected from the file contents, size, raw logits, exact
probabilities and the processing time split into model and explanation steps.

---

## The model

Only two files from https://github.com/kausikragavs/Integrity-theory are used:

- `predict_v4.py`: the classes `SEBlock`, `ResidualBlock`, `IntegrityResNet`, `FrequencyNetwork`
  and `IntegrityFusionModelV4`, and `preprocess_image()`, are copied unchanged into
  `app/model.py` (between the `VERBATIM` markers).
- `integrity_model_v5.1.pth`: the trained weights, in `weights/`. They are loaded with
  `model.load_state_dict(torch.load(path, map_location="cpu"))` followed by `model.eval()`,
  once, when the server starts.

Preprocessing is the same as in `predict_v4.py`:

- **Spatial:** resize to 224 × 224, convert to a tensor, normalise with the ImageNet mean and standard deviation.
- **Metadata:** `has_exif` = 1.0 if `img.getexif()` is not empty; `has_c2pa` = 1.0 if `b'JUMBF'` or `b'c2pa'` is in the raw file bytes.
- **Frequency:** resize to 256 × 256 → grayscale → tensor → `fft2` → `fftshift` → absolute value →
  crop `[112:144, 112:144]` → `log(1 + x)` → flatten to 1,024 values.
- **Output:** softmax over 2 logits; class 0 = Real, class 1 = AI-generated (Fake).

The server's `preprocess_upload()` repeats these steps on the uploaded file in memory.
`checks/acceptance.py` confirms it produces bit-identical tensors to the original `preprocess_image()`.

Not used, because they don't work from the repo: the ONNX models (their `.onnx.data` weight
files are missing), `integrity_v9_tri_stream.pth` (no model class for it exists), and
`predict_v2.py` / `train_v2.py` (no v2 weights exist).

---

## Acceptance checks

After the first run has installed everything, run:

```bash
.venv/bin/python checks/acceptance.py            # macOS / Linux
.venv\Scripts\python checks\acceptance.py        # Windows
```

You can add your own images as arguments. The script starts its own server on port 8765 and checks that:

1. P(fake) from `/api/analyze` matches the repo's original `predict_v4.predict()` (an unchanged
   copy is in `checks/reference/`) with `integrity_model_v5.1.pth` to 4 decimal places, and the
   model inputs are bit-identical;
2. a PNG with the bytes `c2pa` appended reports C2PA found;
3. a 21 MB file, a `.txt` renamed to `.jpg`, a PNG declaring 50 megapixels and a cut-off JPEG are
   all rejected with clear messages.

---

## API

`POST /api/analyze`: multipart form with one field, `file`. Returns JSON with `verdict`,
`provenance`, `ablation`, `gradcam`, `frequency`, `ela`, `summary`, `file` and `timing_ms`.
All pictures are base64-encoded PNGs. Errors return
`{"error": {"code": "...", "message": "..."}}` with HTTP status 400, 413, 415 or 422.

`GET /` serves the page. `GET /api/health` reports whether the model has loaded.

---

## Troubleshooting

- **"Port 8000 is already used by another program."** Close that program, or use another port (see Options).
  If the demo is already running, the launcher just opens it again.
- **PyTorch fails to install.** The lines just above the error say why: usually the internet connection,
  or a full disk (allow about 2 GB free). Fix that and run the script again; finished downloads are kept.
- **Windows: "No such file or directory" while installing PyTorch.** The folder path is too
  long for Windows. Move the folder somewhere short, such as `C:\integrity-demo`, delete `.venv`
  and run `run.bat` again.
- **Windows: PyTorch can't load (`fbgemm.dll` or `WinError 126`).** The Microsoft Visual C++ runtime
  is missing. `run.bat` downloads and installs it automatically (Windows asks for permission);
  you can also install it from https://aka.ms/vs/17/release/vc_redist.x64.exe.
- **Intel Macs** get PyTorch 2.2 (the last version built for them); the demo works the same.
- **Anything else odd:** delete `.venv` and run the script again.

---

## Folder layout

```
integrity-demo/
  app/
    model.py        model classes and preprocessing copied from predict_v4.py, plus the in-memory version
    explain.py      verdict, ablation, Grad-CAM, frequency view, ELA and the summary rules
    server.py       FastAPI app: upload checks, /api/analyze, serves the page
    launch.py       starts uvicorn and opens the browser when the model is ready
  static/           index.html, styles.css, app.js, fonts/ (no build step)
  weights/          integrity_model_v5.1.pth
  checks/           acceptance.py and an unchanged copy of predict_v4.py for comparison
  requirements.txt
  run.bat, run.sh
```

Fonts (Lexend, Atkinson Hyperlegible, JetBrains Mono) are bundled so the page works offline.
They are licensed under the SIL Open Font License; see `static/fonts/LICENSES-OFL.txt`.
