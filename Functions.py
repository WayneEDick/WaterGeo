from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import os

from dataclasses import replace

import numpy as np
import cv2

from geo_types import BBox, CC, CCKind


# ============================================================
# CC Debug Rendering Constants (Grayscale)
# ============================================================

GRAY_BACKGROUND = 255   # white
GRAY_LIGHT      = 210   # low density
GRAY_MEDIUM     = 150   # medium density
GRAY_DARK       = 90    # high density
GRAY_OVERLAP    = 30    # bbox overlap (>= 2)

DENSITY_LIGHT_MAX  = 0.20
DENSITY_MEDIUM_MAX = 0.45


# -----------------------------
# Types
# -----------------------------

Context = Dict[str, Any]


# -----------------------------
# Passes (match Script.yaml)
# -----------------------------

def load_normalize(ctx: Context, cfg: Dict[str, Any]) -> None:
    png_path = cfg.get("inputs", {}).get("png_path")
    if not png_path or not isinstance(png_path, str):
        raise ValueError('load_normalize: cfg["inputs"]["png_path"] must be a non-empty string')

    p = os.path.abspath(png_path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"load_normalize: cannot read PNG: {p}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]
    ctx["imageGray"] = img
    ctx["page"] = {
        "H": int(H),
        "W": int(W),
        "dpi": int(cfg.get("inputs", {}).get("dpi", 300) or 300),
        "path": p,
    }


def binarize(ctx: Context, cfg: Dict[str, Any]) -> None:
    g = ctx.get("imageGray")
    if not isinstance(g, np.ndarray) or g.ndim != 2 or g.dtype != np.uint8:
        raise ValueError('binarize: requires ctx["imageGray"] as 2D uint8 (run load_normalize first)')

    expected = (cfg.get("inputs", {}).get("expected_polarity") or "auto").lower()

    _t1, bw_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _t2, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def ink_frac(mask255: np.ndarray) -> float:
        return float((mask255 > 0).mean())

    if expected == "normal":
        mask255 = bw_inv
    elif expected == "inverted":
        mask255 = bw
    else:
        mask255 = bw_inv if ink_frac(bw_inv) <= ink_frac(bw) else bw

    ctx["mask255"] = mask255
    ctx["mask01"] = (mask255 > 0).astype(np.uint8)
    ctx["imageBin"] = (255 - mask255).astype(np.uint8)


def connected_components(ctx: Context, cfg: Dict[str, Any]) -> None:
    bin_img = ctx.get("mask255", None)
    if bin_img is None:
        raise ValueError('connected_components: ctx["mask255"] missing (run binarize first)')

    if bin_img.dtype != np.uint8:
        bin_img = bin_img.astype(np.uint8)

    connectivity = int(cfg.get("params", {}).get("cc", {}).get("connectivity", 4))


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=connectivity
    )

    ctx["ccCount"] = int(num_labels)
    ctx["ccLabels"] = labels
    ctx["ccStats"] = stats
    ctx["ccCentroids"] = centroids


def build_ccs(ctx: Context, cfg: Dict[str, Any]) -> None:
    stats = ctx.get("ccStats", None)
    cents = ctx.get("ccCentroids", None)
    if stats is None or cents is None:
        raise ValueError("build_ccs: requires ccStats and ccCentroids (run connected_components first)")

    min_area = int(cfg.get("params", {}).get("cc", {}).get("min_area_px", 1) or 1)

    print("min_area:", min_area)

    page = ctx.get("page", None)
    if page is None:
        raise ValueError("build_ccs: ctx['page'] missing (run load_normalize first)")
    H = int(page["H"])
    W = int(page["W"])

    ccs: List[CC] = []
    for label in range(1, int(stats.shape[0])):  # skip background label 0
        x = int(stats[label, 0])
        y = int(stats[label, 1])
        w = int(stats[label, 2])
        h = int(stats[label, 3])
        area = int(stats[label, 4])

        if area < min_area or w <= 0 or h <= 0:
            continue

        box_cx = float(x) + float(w) / 2.0
        box_cy = float(y) + float(h) / 2.0
        ink_cx = float(cents[label, 0])
        ink_cy = float(cents[label, 1])

        box_cx = min(max(box_cx, 0.0), float(W - 1))
        box_cy = min(max(box_cy, 0.0), float(H - 1))
        ink_cx = min(max(ink_cx, 0.0), float(W - 1))
        ink_cy = min(max(ink_cy, 0.0), float(H - 1))

        ccs.append(CC(
            cc_id=int(label),
            bbox=(y, x, h, w),
            ink_px=area,
            box_center_yx=(box_cy, box_cx),
            ink_center_yx=(ink_cy, ink_cx),
        ))

    # --- CC interface lock-in (DEBUG) ---
    for cc in ccs[:50]:
        t, l, h, w = cc.bbox
        assert h > 0 and w > 0
        assert isinstance(cc.ink_px, int) and cc.ink_px > 0
    # --- end CC lock-in ---

    ctx["ccs"] = ccs


def debug_render_cc_boxes(ctx: Context, cfg: Dict[str, Any]) -> None:
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_cc_boxes: ctx['page'] missing (run load_normalize first)")

    H = int(page["H"])
    W = int(page["W"])
    stem = Path(page["path"]).stem

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_centers = bool(cfg.get("outputs", {}).get("debug_draw_cc_box_centers", True))
    out_png = out_dir / f"{stem}_cc_boxes.png"

    ccs: List[CC] = ctx.get("ccs", [])
    G = np.full((H, W), GRAY_BACKGROUND, dtype=np.uint8)
    K = np.zeros((H, W), dtype=np.uint8)

    # --- BOX PAINT LOOP (RESTORED) ---
    for cc in ccs:
        t, l, h, w = cc.bbox
        if h <= 0 or w <= 0:
            continue

        density = float(cc.ink_px) / float(h * w)
        if density < DENSITY_LIGHT_MAX:
            gray = GRAY_LIGHT
        elif density < DENSITY_MEDIUM_MAX:
            gray = GRAY_MEDIUM
        else:
            gray = GRAY_DARK

        y0 = max(0, t)
        y1 = min(H, t + h)
        x0 = max(0, l)
        x1 = min(W, l + w)
        if y1 <= y0 or x1 <= x0:
            continue

        G[y0:y1, x0:x1] = gray
        K[y0:y1, x0:x1] += 1

    G[K >= 2] = GRAY_OVERLAP

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale != 1:
        G = np.repeat(np.repeat(G, upscale, axis=0), upscale, axis=1)

    if draw_centers:
        rgb = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
        r = max(1, int(upscale))
        for cc in ccs:
            by, bx = cc.box_center_yx
            px = int(round(bx * upscale))
            py = int(round(by * upscale))
            if 0 <= py < rgb.shape[0] and 0 <= px < rgb.shape[1]:
                cv2.circle(rgb, (px, py), r, (0, 0, 255), thickness=-1)
        cv2.imwrite(str(out_png), rgb)
    else:
        cv2.imwrite(str(out_png), G)

    ctx["debug_cc_boxes_png"] = str(out_png)


# ============================================================
# G3: CC Box Classification (B.1–B.7) + Color Debug Render
# ============================================================

# Empirical defaults from G3_4.docx
G3_DEFAULTS = {
    "MaxChW": 72,
    "MaxChH": 57,
    "minChW": 3,
    "minChH": 11,
}

# Debug colors (RGB tuples).
# NOTE: OpenCV uses BGR; we convert at use-time.
G3_DEBUG_RGB = {
    # From G3_4.docx (Wayne-tuned for visibility):
    # - char_like: muted olive/brown
    # - image_like: neutral gray
    # - speck: black
    CCKind.CHAR_LIKE: (140, 130, 85),
    CCKind.IMAGE_LIKE: (180, 180, 180),
    CCKind.SPECK: (0, 0, 0),
    CCKind.UNKNOWN: (210, 210, 210),

    # Legacy kinds kept for compatibility (should not be produced in the new G3).
    CCKind.H_INK_BAND: (95, 160, 140),
    CCKind.V_INK_BAND: (95, 160, 140),
    CCKind.VERT_MARK: (95, 160, 140),
    CCKind.DASH_MARK: (95, 160, 140),
}


def _g3_params(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Return pixel thresholds for G3 (all ints)."""
    p = cfg.get("params", {}).get("g3", {})
    out = {}
    for k, dv in G3_DEFAULTS.items():
        try:
            out[k] = int(p.get(k, dv))
        except Exception:
            out[k] = int(dv)
    return out


def _cc_w(cc: CC) -> int:
    _t, _l, _h, w = cc.bbox
    return int(w)


def _cc_h(cc: CC) -> int:
    _t, _l, h, _w = cc.bbox
    return int(h)


def classify_ccs_g3(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G3: classify each CC bbox using the (pixel-domain) predicates B.1–B.7.

    We enforce a stable priority order to ensure a single label per CC.

    Saved to ctx:
      - ctx['ccs_g3']: List[CC] (same as ccs but with cc.kind updated)
      - ctx['g3_sets']: Dict[str, List[int]] mapping kind -> cc_id list
      - ctx['g3_params']: the thresholds used
    """
    ccs = ctx.get("ccs", None)
    if not isinstance(ccs, list):
        raise ValueError("classify_ccs_g3: ctx['ccs'] missing (run build_ccs first)")

    th = _g3_params(cfg)
    MaxChW = th["MaxChW"]
    MaxChH = th["MaxChH"]
    minChW = th["minChW"]
    minChH = th["minChH"]

    # ------------------------------------------------------------
    # G3 predicates (Wayne / G3_4.docx)
    #
    # IMPORTANT CHANGE (Jan 2026):
    # - CharLike uses OR at the minima:
    #       (w >= minChW) OR (h >= minChH)
    #   while staying below the maxima.
    # - Speck uses AND with strict < at the minima:
    #       (w < minChW) AND (h < minChH)
    # - We no longer produce hInkBand, vInkBand, dash_mark, vert_mark.
    #   Everything non-speck and non-image-like that meets B.1 is CHAR_LIKE;
    #   otherwise UNKNOWN.
    # ------------------------------------------------------------

    def is_char_like(cc: CC) -> bool:
        w = _cc_w(cc)
        h = _cc_h(cc)
        return (w <= MaxChW) and (h <= MaxChH) and ((w >= minChW) or (h >= minChH))

    def is_image_like(cc: CC) -> bool:
        w = _cc_w(cc)
        h = _cc_h(cc)
        return (w > MaxChW) or (h > MaxChH)

    def is_speck(cc: CC) -> bool:
        w = _cc_w(cc)
        h = _cc_h(cc)
        return (w < minChW) and (h < minChH)

    # Priority order (ensures total, single label)
    priority = [
        (CCKind.IMAGE_LIKE, is_image_like),
        (CCKind.SPECK, is_speck),
        (CCKind.CHAR_LIKE, is_char_like),
    ]

    g3_sets: Dict[str, List[int]] = {k.value: [] for k, _ in priority}
    g3_sets[CCKind.UNKNOWN.value] = []

    ccs_g3: List[CC] = []
    for cc in ccs:
        kind = CCKind.UNKNOWN
        for k, pred in priority:
            if pred(cc):
                kind = k
                break
        ccs_g3.append(replace(cc, kind=kind))
        g3_sets[kind.value].append(int(cc.cc_id))

    ctx["ccs_g3"] = ccs_g3
    ctx["g3_sets"] = g3_sets
    ctx["g3_params"] = th


def debug_render_g3_boxes(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Render a color-coded bbox debug PNG based on G3 kind classification.

    IMPORTANT (LV): We draw *thick rectangle outlines* (and an optional light fill)
    so the boxes are visible even at high zoom. This is a G3-level view: *only boxes*.
    """
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g3_boxes: ctx['page'] missing (run load_normalize first)")

    # Prefer the original grayscale background if present (better contrast than imageBin)
    base = ctx.get("imageGray", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        # Fallback: binarized image
        base = ctx.get("imageBin", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        raise ValueError("debug_render_g3_boxes: ctx['imageGray' or 'imageBin'] missing (run load_normalize/binarize first)")

    ccs: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs, list):
        raise ValueError("debug_render_g3_boxes: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")

    H = int(page["H"])
    W = int(page["W"])
    stem = Path(page["path"]).stem

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    # Convert base to color and upscale for easier viewing
    rgb = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if upscale != 1:
        rgb = cv2.resize(rgb, (W * upscale, H * upscale), interpolation=cv2.INTER_NEAREST)

    # Make outlines thick enough to see in LV viewing
    thickness = max(3, 2 * int(upscale))

    # Optional: light fill for each box (helps when outlines sit on dark ink)
    fill_alpha = float(cfg.get("outputs", {}).get("g3_debug_fill_alpha", 0.15) or 0.15)
    if fill_alpha < 0.0:
        fill_alpha = 0.0
    if fill_alpha > 0.8:
        fill_alpha = 0.8

    overlay = rgb.copy()

    for cc in ccs:
        t, l, h, w = cc.bbox
        if h <= 0 or w <= 0:
            continue

        rgb_col = G3_DEBUG_RGB.get(cc.kind, G3_DEBUG_RGB[CCKind.UNKNOWN])
        # Convert RGB -> BGR for OpenCV
        bgr = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))

        x0 = int(l * upscale)
        y0 = int(t * upscale)
        x1 = int((l + w) * upscale)
        y1 = int((t + h) * upscale)

        # clip
        x0 = max(0, min(x0, rgb.shape[1] - 1))
        y0 = max(0, min(y0, rgb.shape[0] - 1))
        x1 = max(0, min(x1, rgb.shape[1] - 1))
        y1 = max(0, min(y1, rgb.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        # Fill (on overlay) then outline (on rgb)
        if fill_alpha > 0.0:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), bgr, thickness=thickness)

    if fill_alpha > 0.0:
        rgb = cv2.addWeighted(overlay, fill_alpha, rgb, 1.0 - fill_alpha, 0)

    out_png = out_dir / f"{stem}_g3_boxes.png"
    cv2.imwrite(str(out_png), rgb)
    ctx["debug_g3_boxes_png"] = str(out_png)


def debug_render_g3_boxes_only(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Render a *boxes-only* G3 debug PNG: colored CC rectangles on a blank page.

    Purpose (LV): Wayne wants to visually verify the *geometry* of CC boxes without
    the cognitive interference of the glyph ink. This draws ONLY rectangles.

    Output: <stem>_g3_boxes_only.png in outputs.out_dir
    """
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g3_boxes_only: ctx['page'] missing (run load_normalize first)")

    ccs: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs, list):
        raise ValueError("debug_render_g3_boxes_only: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")

    H = int(page["H"])
    W = int(page["W"])
    stem = Path(page["path"]).stem

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    # Blank white canvas (no glyph ink)
    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)

    # Thick outlines for visibility
    thickness = max(3, 2 * int(upscale))

    # Fill defaults to solid (1.0). You can dial this down if you prefer outlines.
    #  - 1.0: solid colored boxes
    #  - 0.0: outline only
    fill_alpha = float(cfg.get("outputs", {}).get("g3_boxes_only_fill_alpha", 1.0) or 1.0)
    if fill_alpha < 0.0:
        fill_alpha = 0.0
    if fill_alpha > 1.0:
        fill_alpha = 1.0

    overlay = rgb.copy()

    for cc in ccs:
        t, l, h, w = cc.bbox
        if h <= 0 or w <= 0:
            continue

        rgb_col = G3_DEBUG_RGB.get(cc.kind, G3_DEBUG_RGB[CCKind.UNKNOWN])
        bgr = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))

        x0 = int(l * upscale)
        y0 = int(t * upscale)
        x1 = int((l + w) * upscale)
        y1 = int((t + h) * upscale)

        # clip
        x0 = max(0, min(x0, rgb.shape[1] - 1))
        y0 = max(0, min(y0, rgb.shape[0] - 1))
        x1 = max(0, min(x1, rgb.shape[1] - 1))
        y1 = max(0, min(y1, rgb.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        if fill_alpha > 0.0:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), bgr, thickness=thickness)

    if fill_alpha > 0.0:
        rgb = cv2.addWeighted(overlay, fill_alpha, rgb, 1.0 - fill_alpha, 0)

    out_png = out_dir / f"{stem}_g3_boxes_only.png"
    cv2.imwrite(str(out_png), rgb)
    ctx["debug_g3_boxes_only_png"] = str(out_png)
