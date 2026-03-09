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


# ============================================================
# G4.0: Page Constants for Glyphify (baseline-free)
# ============================================================

# G4 debug colors for char_like subclasses (RGB tuples).
# NOTE: OpenCV uses BGR; we convert at use-time.
G4_DEBUG_RGB = {
    "Md": (160, 70, 60),    # red-brown (main_like)
    "Mc": (150, 140, 110),  # tan (capt_like)
    "Mm": (70, 125, 180),   # blue-accent (math_like)
    "MNot": (130, 90, 165), # purple-accent (not_glfy)
}

# ============================================================
# G4 Band Debug Colors (LV-friendly, 9 bands)
# band 0 = smallest, band 8 = largest
# ============================================================

G4_BAND_RGB = [
    (60, 60, 60),        # 0 dark gray
    (166, 124, 82),     # 1 light brown
    (255, 178, 102),    # 2 light orange
    (102, 204, 204),    # 3 light teal
    (178, 102, 255),    # 4 light purple (dominant band)
    (255, 153, 153),    # 5 light red
    (153, 255, 153),    # 6 light green
    (153, 204, 255),    # 7 light blue
    (180, 180, 180),    # 8 gray (largest)
]


G4_DEFAULTS = {
    "near_square_min": 0.9,
    "near_square_max": 1.1,
    "tau": 0.6,
    "md_lo": 0.9,
    "md_hi": 1.35,
    "mc_lo": 0.35,
    "mc_hi": 0.7,
    "mm_lo": 1.35,
    "mm_hi": 2.2,
}


def _g4_params(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return numeric parameters for G4.0 (all floats)."""
    p = cfg.get("params", {}).get("g4", {})
    out: Dict[str, float] = {}
    for k, dv in G4_DEFAULTS.items():
        try:
            out[k] = float(p.get(k, dv))
        except Exception:
            out[k] = float(dv)
    return out


def g4_page_constants(ctx: Context, cfg: Dict[str, Any]) -> None:
    """G4.0: Compute page constants for Glyphify.

    Inputs:
      - ctx['ccs_g3'] : List[CC] with cc.kind in {char_like, image_like, speck, unknown}

    Outputs (ctx['g4']):
      - xhd, xhc, xhm (floats)
      - h_hist (List[int]) bincount histogram over heights of near-square char_like boxes
      - sd_bins: (bin_lo, bin_hi) inclusive range of dominant band
      - classes_by_id: Dict[int, str] mapping cc_id -> {'Md','Mc','Mm','MNot'} for char_like only
      - sets: Dict[str, List[int]] mapping class -> cc_id list (char_like only)
      - params: used G4 parameters
    """
    ccs: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs, list):
        raise ValueError("g4_page_constants: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")

    th = _g4_params(cfg)
    ns_min = th["near_square_min"]
    ns_max = th["near_square_max"]
    tau = th["tau"]

    # --- Step 1: Q = near-square char_like ---
    q_heights: List[int] = []
    q_cc_ids: List[int] = []

    for cc in ccs:
        if cc.kind != CCKind.CHAR_LIKE:
            continue
        h = _cc_h(cc)
        w = _cc_w(cc)
        if h <= 0 or w <= 0:
            continue
        r = float(h) / float(w)
        if (r > ns_min) and (r < ns_max):
            q_heights.append(int(h))
            q_cc_ids.append(int(cc.cc_id))

    if not q_heights:
        raise ValueError("g4_page_constants: no near-square char_like boxes found (cannot build h_Hist)")

    # --- Step 2.1: height histogram ---
    max_h = int(max(q_heights))
    counts = np.bincount(np.array(q_heights, dtype=np.int32), minlength=max_h + 1)
    h_hist = counts.tolist()

    # --- Step 2.2: dominant band Sd (deterministic) ---
    p_bin = int(np.argmax(counts))
    peak = int(counts[p_bin])
    if peak <= 0:
        raise ValueError("g4_page_constants: dominant peak is empty (unexpected)")

    thr = float(tau) * float(peak)

    lo = p_bin
    while lo - 1 >= 0 and float(counts[lo - 1]) >= thr:
        lo -= 1
    hi = p_bin
    while hi + 1 < counts.shape[0] and float(counts[hi + 1]) >= thr:
        hi += 1

    # Boxes B in Sd are those in Q with h(B) in [lo, hi]
    xhd_candidates = [h for h in q_heights if lo <= h <= hi]
    if not xhd_candidates:
        raise ValueError("g4_page_constants: Sd band contains no boxes (unexpected)")

    # --- Step 2.3: xhd = min height in Sd ---
    xhd = float(min(xhd_candidates))
    xhc = 0.35 * xhd
    xhm = 1.35 * xhd

    # --- Step 3: class assignment for ALL char_like (not only near-square) ---
    md_lo = th["md_lo"] * xhd
    md_hi = th["md_hi"] * xhd
    mc_lo = th["mc_lo"] * xhd
    mc_hi = th["mc_hi"] * xhd
    mm_lo = th["mm_lo"] * xhd
    mm_hi = th["mm_hi"] * xhd

    classes_by_id: Dict[int, str] = {}
    sets: Dict[str, List[int]] = {"Md": [], "Mc": [], "Mm": [], "MNot": []}

    for cc in ccs:
        if cc.kind != CCKind.CHAR_LIKE:
            continue
        h = float(_cc_h(cc))
        cid = int(cc.cc_id)

        # Boundary tie-break priority: Mm, then Md, then Mc
        if (h >= mm_lo) and (h <= mm_hi):
            lab = "Mm"
        elif (h > md_lo) and (h < md_hi):
            lab = "Md"
        elif (h >= mc_lo) and (h <= mc_hi):
            lab = "Mc"
        else:
            lab = "MNot"

        classes_by_id[cid] = lab
        sets[lab].append(cid)

    ctx["g4"] = {
        "xhd": xhd,
        "xhc": xhc,
        "xhm": xhm,
        "h_hist": h_hist,
        "sd_bins": (int(lo), int(hi)),
        "peak_bin": int(p_bin),
        "peak_count": int(peak),
        "tau": float(tau),
        "classes_by_id": classes_by_id,
        "sets": sets,
        "params": th,
    }

def g4_height_band(h: float, peak_h: float) -> int:
    """
    Map height h to one of 9 logarithmic bands around peak_h.
    Band 4 is the dominant band.
    """
    if h <= 0 or peak_h <= 0:
        return 4

    import math

    # log2 scale, 3 bands per octave
    band = int(math.floor(math.log2(h / peak_h) * 3.0)) + 4

    if band < 0:
        return 0
    if band > 8:
        return 8
    return band

def debug_render_g4_boxes(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Render a debug PNG like G3, but char_like boxes are colored by G4 class (Md/Mc/Mm/MNot)."""
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g4_boxes: ctx['page'] missing (run load_normalize first)")

    base = ctx.get("imageGray", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        base = ctx.get("imageBin", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        raise ValueError("debug_render_g4_boxes: ctx['imageGray' or 'imageBin'] missing")

    ccs: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs, list):
        raise ValueError("debug_render_g4_boxes: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")

    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "classes_by_id" not in g4:
        raise ValueError("debug_render_g4_boxes: ctx['g4'] missing (run g4_page_constants first)")

    classes_by_id: Dict[int, str] = g4["classes_by_id"]

    H = int(page["H"])
    W = int(page["W"])
    stem = Path(page["path"]).stem

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    rgb = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if upscale != 1:
        rgb = cv2.resize(rgb, (W * upscale, H * upscale), interpolation=cv2.INTER_NEAREST)

    thickness = max(3, 2 * int(upscale))
    fill_alpha = float(cfg.get("outputs", {}).get("g4_debug_fill_alpha", 0.15) or 0.15)
    fill_alpha = min(max(fill_alpha, 0.0), 0.8)

    overlay = rgb.copy()

    for cc in ccs:
        t, l, h, w = cc.bbox
        if h <= 0 or w <= 0:
            continue

        if cc.kind == CCKind.CHAR_LIKE:
            g4 = ctx.get("g4", {})
            peak_h = g4.get("xhd", None)

            if peak_h is None:
                rgb_col = (120, 120, 120)
            else:
                band = g4_height_band(h, float(peak_h))
                rgb_col = G4_BAND_RGB[band]
        else:
            # keep existing G3 colors for non-char_like
            rgb_col = G3_DEBUG_RGB.get(cc.kind, G3_DEBUG_RGB[CCKind.UNKNOWN])

        bgr = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))

        x0 = int(l * upscale)
        y0 = int(t * upscale)
        x1 = int((l + w) * upscale)
        y1 = int((t + h) * upscale)

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

    out_png = out_dir / f"{stem}_g4_boxes.png"
    cv2.imwrite(str(out_png), rgb)
    ctx["debug_g4_boxes_png"] = str(out_png)


# -----------------------------
# G4 Debug: Histogram Rendering
# -----------------------------

def debug_render_g4_histogram(ctx: Context, cfg: Dict[str, Any]) -> None:
    """G4 debug: render the height histogram to a PNG.

    Reads:
      - ctx['g4']['h_hist'] : List[int]
      - ctx['g4']['sd_bins'] : [lo, hi] or (lo, hi) (optional)
      - ctx['page']['path'] : used only to derive output stem
      - cfg['outputs']['out_dir'] : output directory (required)

    Writes:
      - <out_dir>/<stem>_g4_hist.png
      - ctx['debug_g4_hist_png'] = path string

    Note:
      Runner will still save JSON snapshots; this function adds the PNG.
    """
    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict):
        raise ValueError("debug_render_g4_histogram: ctx['g4'] missing")

    peak_h = g4.get("xhd", None)

    if not bool(cfg.get("outputs", {}).get("save_debug_images", True)):
        # Debug images disabled by config; do nothing.
        return

    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict):
        raise ValueError("debug_render_g4_histogram: ctx['g4'] missing (run g4_page_constants first)")

    h_hist = g4.get("h_hist", None)
    if not isinstance(h_hist, list) or len(h_hist) == 0:
        raise ValueError("debug_render_g4_histogram: g4['h_hist'] missing/empty")

    sd_bins = g4.get("sd_bins", None)

    # LV-safe layout parameters (override-able via cfg['outputs'])
    out_cfg = cfg.get("outputs", {})
    bar_w  = int(out_cfg.get("g4_hist_bar_w", 8) or 8)
    height = int(out_cfg.get("g4_hist_height", 600) or 600)
    margin = int(out_cfg.get("g4_hist_margin", 60) or 60)

    bar_w = max(1, bar_w)
    height = max(200, height)
    margin = max(20, margin)

    # Scale
    max_val = max(int(v) for v in h_hist) if h_hist else 1
    max_val = max(1, max_val)
    scale_y = float(height - 2 * margin) / float(max_val)

    width = margin * 2 + bar_w * len(h_hist)
    bg = int(GRAY_BACKGROUND) if "GRAY_BACKGROUND" in globals() else 255
    img = np.full((height, width, 3), (bg, bg, bg), dtype=np.uint8)  # BGR canvas

    # Dominant band range
    lo = hi = None
    if isinstance(sd_bins, (list, tuple)) and len(sd_bins) == 2:
        lo = int(sd_bins[0])
        hi = int(sd_bins[1])

    # Draw bars (color by height band)
    for i, v in enumerate(h_hist):
        v = int(v)
        if v <= 0:
            continue

        x0 = margin + i * bar_w
        x1 = x0 + bar_w - 1
        y1 = height - margin
        y0 = int(round(y1 - v * scale_y))

        # Map histogram bin index to band color
        if peak_h is not None:
            band = g4_height_band(i, peak_h)
            rgb = G4_BAND_RGB[band]
        else:
            rgb = (120, 120, 120)

        # OpenCV uses BGR
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr, thickness=-1)
        #bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        #cv2.rectangle(img, (x0, y0), (x1, y1), bgr, thickness=-1)

    # Axis line
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)

    # Output path
    out_dir = Path(cfg.get("outputs", {}).get("out_dir", "out_geo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(ctx.get("page", {}).get("path", "page")).stem
    out_png = out_dir / f"{stem}_g4_hist.png"

    ok = cv2.imwrite(str(out_png), img)
    if not ok:
        raise IOError(f"debug_render_g4_histogram: cv2.imwrite failed for {out_png}")

    ctx["debug_g4_hist_png"] = str(out_png)
