"""How the model reached its verdict.

Every number and picture produced here comes from the loaded model or from the
uploaded file. Nothing is estimated, sampled or invented. The plain-English summary is
built from fixed rules over those numbers (no language model is involved).
"""
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageStat

DISPLAY_MAX_SIDE = 768   # longest side of the pictures sent back to the browser
ELA_QUALITY = 90

# Sequential colour map for heatmaps (anchor points of matplotlib's "inferno").
_INFERNO = [
    (0.000, (0, 0, 4)), (0.125, (31, 12, 72)), (0.250, (85, 15, 109)),
    (0.375, (136, 34, 106)), (0.500, (186, 54, 85)), (0.625, (227, 89, 51)),
    (0.750, (249, 140, 10)), (0.875, (249, 201, 50)), (1.000, (252, 255, 164)),
]
_STOPS = [s for s, _ in _INFERNO]
HEAT_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), _STOPS, [c[i] for _, c in _INFERNO]) for i in range(3)],
    axis=1,
).round().astype(np.uint8)

CORAL = (255, 89, 73)
INK = (20, 20, 20)

BRANCHES = [
    {
        "key": "spatial",
        "label": "Spatial features",
        "short": "spatial features",
        "noun": "the image content seen by the CNN",
        "be": "was",
        "detail": "128 values from the image CNN (spatial_branch)",
        "dims": 128,
    },
    {
        "key": "frequency",
        "label": "Frequency features",
        "short": "frequency features",
        "noun": "the frequency spectrum",
        "be": "was",
        "detail": "64 values from the FFT network (freq_branch)",
        "dims": 64,
    },
    {
        "key": "metadata",
        "label": "Metadata flags",
        "short": "metadata flags",
        "noun": "the metadata flags",
        "be": "were",
        "detail": "2 values: has EXIF, has C2PA",
        "dims": 2,
    },
]

REGION_NAMES = [
    ["top-left", "top-centre", "top-right"],
    ["centre-left", "centre", "centre-right"],
    ["bottom-left", "bottom-centre", "bottom-right"],
]


# --- small helpers --------------------------------------------------------------------

def png_b64(img):
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fit_for_display(img, max_side=DISPLAY_MAX_SIDE):
    """Downscale (never upscale) so the longest side is at most max_side."""
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return out


def _to_uint8(arr):
    """Min-max scale an array to 0..255."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


# --- 1. verdict -------------------------------------------------------------------------

@torch.no_grad()
def predict(model, prep):
    """The same forward pass and softmax as predict_v4.predict()."""
    outputs = model(prep["image"], prep["freq"], prep["meta"])
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    return {
        "logits": [float(v) for v in outputs[0]],
        "p_real": float(probabilities[0, 0]),
        "p_fake": float(probabilities[0, 1]),
        "class_index": int(predicted_class.item()),
        "confidence": float(confidence.item()),
    }


# --- 3. evidence by branch (ablation) ---------------------------------------------------

def _fusion_head(model, spatial_features, freq_features, metadata_flags):
    # The same lines as IntegrityFusionModelV4.forward() after the two branches.
    fused_vector = torch.cat((spatial_features, freq_features, metadata_flags), dim=1)
    x = F.relu(model.fusion_fc1(fused_vector))
    x = model.dropout(x)
    return model.fusion_fc2(x)


@torch.no_grad()
def ablation(model, prep):
    """Run the fusion head with each input group replaced by zeros.

    delta = P(fake) with everything - P(fake) without that group.
    delta > 0: the group pushed toward fake.  delta < 0: it pushed toward real.
    """
    spatial = model.spatial_branch(prep["image"])
    freq = model.freq_branch(prep["freq"])
    meta = prep["meta"]

    def p_fake(s, f, m):
        return float(F.softmax(_fusion_head(model, s, f, m), dim=1)[0, 1])

    full = p_fake(spatial, freq, meta)
    variants = {
        "spatial": (torch.zeros_like(spatial), freq, meta),
        "frequency": (spatial, torch.zeros_like(freq), meta),
        "metadata": (spatial, freq, torch.zeros_like(meta)),
    }
    items = []
    for b in BRANCHES:
        without = p_fake(*variants[b["key"]])
        delta = full - without
        items.append({
            "key": b["key"],
            "label": b["label"],
            "detail": b["detail"],
            "dims": b["dims"],
            "p_fake_without": without,
            "delta": delta,
            "direction": "fake" if delta > 0 else ("real" if delta < 0 else "none"),
        })
    return {"p_fake_full": full, "items": items}


# --- 4. Grad-CAM on spatial_branch.stage3 -----------------------------------------------

def grad_cam(model, prep):
    """Grad-CAM for the 'fake' logit at the output of spatial_branch.stage3 (128 x 14 x 14)."""
    store = {}

    def keep_output(_module, _inputs, output):
        store["act"] = output

    handle = model.spatial_branch.stage3.register_forward_hook(keep_output)
    try:
        with torch.enable_grad():
            logits = model(prep["image"], prep["freq"], prep["meta"])
            fake_logit = logits[0, 1]
            act = store["act"]
            (grads,) = torch.autograd.grad(fake_logit, act)
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)            # one weight per channel
    cam = F.relu((weights * act).sum(dim=1))[0].detach()       # [14, 14]
    raw_max = float(cam.max())
    cam_norm = cam / raw_max if raw_max > 0 else torch.zeros_like(cam)
    return {
        "cam": cam_norm,                    # 0..1, shape [14, 14]
        "raw_max": raw_max,
        "fake_logit": float(fake_logit.detach()),
        "grid": list(cam.shape),
    }


def gradcam_stats(cam_norm):
    """Where the Grad-CAM response sits (3 x 3 regions) and how concentrated it is."""
    cam = cam_norm.numpy().astype(np.float64)
    total = cam.sum()
    if total <= 0:
        return {"has_signal": False, "peak_region": None, "peak_region_share": 0.0, "top10_share": 0.0}
    rows = np.array_split(np.arange(cam.shape[0]), 3)
    cols = np.array_split(np.arange(cam.shape[1]), 3)
    region_mass = np.array([[cam[np.ix_(r, c)].sum() for c in cols] for r in rows])
    region_mean = np.array([[cam[np.ix_(r, c)].mean() for c in cols] for r in rows])
    ri, ci = np.unravel_index(np.argmax(region_mean), region_mean.shape)
    n_top = max(1, int(round(cam.size * 0.10)))                 # top 10% of grid cells
    top_share = float(np.sort(cam.ravel())[::-1][:n_top].sum() / total)
    return {
        "has_signal": True,
        "peak_region": REGION_NAMES[ri][ci],
        "peak_region_share": float(region_mass[ri, ci] / total),
        "top10_share": top_share,
    }


def gradcam_images(display_img, cam_norm):
    """Overlay (smooth) and heatmap-only (true 14 x 14 cells) pictures."""
    w, h = display_img.size
    up = F.interpolate(cam_norm[None, None], size=(h, w), mode="bilinear", align_corners=False)
    up = up[0, 0].clamp(0, 1).numpy()

    base = np.asarray(display_img, dtype=np.float32)
    gray = base.mean(axis=2, keepdims=True)
    base = (0.45 * base + 0.55 * gray) * 0.85                  # muted so the heat stands out
    heat = HEAT_LUT[(up * 255).round().astype(np.uint8)].astype(np.float32)
    alpha = (0.80 * up)[..., None]
    overlay = (base * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

    cells = HEAT_LUT[(cam_norm.numpy() * 255).round().astype(np.uint8)]
    heatmap = Image.fromarray(cells).resize((w, h), Image.Resampling.NEAREST)
    return Image.fromarray(overlay), heatmap


# --- 5. frequency view ------------------------------------------------------------------

def frequency_images(prep):
    """Full 256 x 256 log-magnitude spectrum with the model's 32 x 32 crop outlined,
    plus the crop itself (exactly the 1024 values fed to the frequency branch)."""
    log_full = torch.log(1 + prep["magnitude_spectrum"]).numpy()
    scale = 2
    full = Image.fromarray(_to_uint8(log_full)).resize((256 * scale, 256 * scale), Image.Resampling.NEAREST)
    full = full.convert("RGB")
    draw = ImageDraw.Draw(full)
    x0, x1 = 112 * scale, 144 * scale - 1                      # the crop [112:144, 112:144]
    for pad, colour, width in ((5, INK, 1), (4, CORAL, 4)):
        draw.rectangle([x0 - pad, x0 - pad, x1 + pad, x1 + pad], outline=colour, width=width)

    crop_vals = prep["freq"][0].reshape(32, 32).numpy()
    crop = Image.fromarray(_to_uint8(crop_vals)).resize((256, 256), Image.Resampling.NEAREST)
    return {
        "spectrum_png": png_b64(full),
        "crop_png": png_b64(crop),
        "spectrum_min": float(log_full.min()),
        "spectrum_max": float(log_full.max()),
        "crop_min": float(crop_vals.min()),
        "crop_max": float(crop_vals.max()),
        "crop_mean": float(crop_vals.mean()),
        "crop_box": [112, 144],
    }


# --- 6. ELA (extra view, not a model input) ---------------------------------------------

def ela_image(img_rgb, quality=ELA_QUALITY):
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img_rgb, resaved)
    max_diff = max(hi for _lo, hi in diff.getextrema())
    scale = 255.0 / max_diff if max_diff else 1.0
    mean_diff = float(sum(ImageStat.Stat(diff).mean) / 3.0)
    # Scaling is linear, so downscaling first and amplifying second gives the same picture, faster.
    amplified = ImageEnhance.Brightness(fit_for_display(diff)).enhance(scale)
    return {
        "png": png_b64(amplified),
        "quality": quality,
        "max_diff": int(max_diff),
        "mean_diff": mean_diff,
        "amplification": scale,
    }


# --- 7. plain-English summary (fixed rules) ---------------------------------------------

def build_summary(verdict, provenance, abl, cam_stats, file_info):
    out = []
    p_fake, conf = verdict["p_fake"], verdict["confidence"]
    is_fake = verdict["class_index"] == 1
    label = "AI-generated" if is_fake else "real"

    # Verdict and margin
    if p_fake >= 0.995:
        p_text = "P(fake) > 0.99"
    elif p_fake < 0.005:
        p_text = "P(fake) < 0.01"
    else:
        p_text = f"P(fake) = {p_fake:.2f}"
    conf_text = "above 99.9%" if conf >= 0.9995 else f"{conf * 100:.1f}%"
    text = f"The model rates this image as {label} ({p_text}, confidence {conf_text})."
    if conf < 0.70:
        text += " That is close to the 50% decision line, so treat the verdict as uncertain."
    elif conf < 0.90:
        text += " That is a moderate margin."
    else:
        text += " That is a clear margin, but a confident score is not proof."
    out.append({"kind": "verdict", "text": text})

    # Strongest branch, by how much P(fake) moves when it is removed
    full = abl["p_fake_full"]
    by_key = {b["key"]: b for b in BRANCHES}
    items = sorted(abl["items"], key=lambda i: abs(i["delta"]), reverse=True)
    top = items[0]
    if abs(top["delta"]) >= 0.01:
        b = by_key[top["key"]]
        verb = "lowers" if top["delta"] > 0 else "raises"
        out.append({"kind": "evidence", "text": (
            f"Removing the {b['short']} {verb} P(fake) from {full:.2f} to {top['p_fake_without']:.2f}, "
            f"so {b['noun']} {b['be']} the strongest evidence toward '{top['direction']}'.")})
        for other in items[1:]:
            if abs(other["delta"]) < 0.05:
                continue
            ob = by_key[other["key"]]
            if other["direction"] == top["direction"]:
                out.append({"kind": "evidence", "text": (
                    f"The {ob['short']} also pushed toward '{other['direction']}' "
                    f"(removing them changes P(fake) by {-other['delta']:+.2f}).")})
            else:
                out.append({"kind": "evidence", "text": (
                    f"The {ob['short']} pushed the other way, toward '{other['direction']}' "
                    f"(removing them changes P(fake) by {-other['delta']:+.2f}).")})
    else:
        out.append({"kind": "evidence", "text": (
            "No single input group changes P(fake) by more than 0.01 when removed: "
            "the remaining groups reach the same score on their own.")})

    # Provenance
    if provenance["c2pa"]:
        markers = " and ".join(f"'{m}'" for m in provenance["c2pa_markers"])
        out.append({"kind": "provenance", "text": (
            f"A C2PA content-credential marker ({markers}) was found in the file. Several generative AI "
            "tools attach C2PA credentials to their output, though some cameras and editing apps add them too.")})
    else:
        out.append({"kind": "provenance", "text": (
            "No C2PA/JUMBF marker was found in the file. Most images carry none, "
            "so its absence says little either way.")})
    if provenance["exif"]:
        extra = ""
        tags = provenance.get("exif_tags") or {}
        if tags.get("Software"):
            extra = f", including Software = \"{tags['Software']}\""
        elif tags.get("Make") or tags.get("Model"):
            extra = f", including camera = \"{' '.join(v for v in (tags.get('Make'), tags.get('Model')) if v)}\""
        n = provenance["exif_count"]
        out.append({"kind": "provenance", "text": (
            f"The file has EXIF metadata ({n} tag{'' if n == 1 else 's'}{extra}).")})
    else:
        out.append({"kind": "provenance", "text": (
            "The file has no EXIF metadata. Screenshots, images saved from the web and "
            "most AI generators have none.")})
    if not provenance["exif"] and not provenance["c2pa"]:
        out.append({"kind": "evidence", "text": (
            "Both metadata flags are 0, so removing them leaves P(fake) unchanged: "
            "this decision rests on the image content.")})

    # Grad-CAM
    if not cam_stats["has_signal"]:
        out.append({"kind": "gradcam", "text": "Grad-CAM found no image regions that raise the 'fake' score."})
    else:
        spread = ""
        if cam_stats["top10_share"] >= 0.5:
            spread = ", concentrated in a small area"
        elif cam_stats["top10_share"] <= 0.25:
            spread = ", spread across much of the image"
        text = (f"In the spatial branch, Grad-CAM for the 'fake' score is strongest in the "
                f"{cam_stats['peak_region']} of the image{spread}.")
        if not is_fake:
            text += " These are the regions that pushed hardest toward 'fake', even though the overall verdict is 'real'."
        out.append({"kind": "gradcam", "text": text})

    # Notes about what the model actually saw
    w, h = file_info["width"], file_info["height"]
    if max(w, h) / max(1, min(w, h)) >= 1.5:
        out.append({"kind": "note", "text": (
            f"The model squeezes the {w} × {h} image into a 224 × 224 square without keeping its proportions, "
            "so shapes are distorted in what the spatial branch sees.")})
    if provenance.get("exif_orientation") not in (None, 1):
        out.append({"kind": "note", "text": (
            "The EXIF data says this photo should be displayed rotated or flipped. The model (like "
            "predict_v4.py) reads the stored pixels without applying that, so the views below may look rotated.")})
    return out
