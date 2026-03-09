from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import os

from dataclasses import replace

import numpy as np
import cv2

from geo_types_updated import BBox, CC, CCKind


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


# ============================================================
# Phase 2: V_Graph + Floater + H_Graph (Wayne spec Feb 2026)
# ============================================================

from dataclasses import dataclass
from math import ceil, floor
from typing import Tuple, Optional

# Extend debug colors for Phase-2 kinds (safe if kinds not used)
G3_DEBUG_RGB.update({
    getattr(CCKind, "BIG_SYM", CCKind.UNKNOWN): (110, 110, 200),        # bluish
    getattr(CCKind, "V_CHAR_STACK", CCKind.UNKNOWN): (120, 90, 40),     # darker olive/brown
    getattr(CCKind, "FLOATER_BLOB", CCKind.UNKNOWN): (200, 120, 120),   # muted red/pink
})


@dataclass(slots=True)
class VBox:
    id: int
    l: float
    r: float
    t: float
    b: float
    w: float
    h: float
    cx: float
    cy: float
    xh: float
    kind: CCKind
    members: List[int]
    overcap_above: int = 0
    overcap_below: int = 0
    grid_cell: Optional[Tuple[int, int]] = None


@dataclass(slots=True)
class Token:
    kind: str
    l: float
    r: float
    t: float
    b: float
    w: float
    h: float
    cx: float
    cy: float
    xh: float
    members: List[int]
    src_id: int = 0

    # ---- G5a annotations (filled by g5a_classify_tokens) ----
    is_text_like: bool = False
    length: int = 0
    singleton: bool = False
    shape: Optional[str] = None
    token_id: int = 0

    role: str = "OTHER"
    inline: bool = False
# -----------------------------
# Phase-2 params from Script.yaml
# -----------------------------

def _p2_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg.get("params", {}).get("phase2", {}) or {}
    # Defaults match pseudocode
    return {
        "grid_edge_mult": float(p.get("grid_edge_mult", 0.60)),
        "cand_margin_mult": float(p.get("cand_margin_mult", 2.0)),
        "eps_mult": float(p.get("eps_mult", 0.10)),
        "delta_mult": float(p.get("delta_mult", 0.85)),
        "a_mult": float(p.get("a_mult", 1.00)),
        "d_mult": float(p.get("d_mult", 0.70)),
        "max_above": int(p.get("max_above", 3)),
        "max_below": int(p.get("max_below", 2)),
        "allow_overcap": bool(p.get("allow_overcap", True)),
        "speck_h_mult": float(p.get("speck_h_mult", 0.35)),
        "speck_w_mult": float(p.get("speck_w_mult", 0.35)),
        "tall_mult": float(p.get("tall_mult", 2.80)),
        "thin_mult": float(p.get("thin_mult", 0.35)),
        "wide_mult": float(p.get("wide_mult", 3.00)),
        "flat_mult": float(p.get("flat_mult", 0.25)),
        "ov_mult": float(p.get("ov_mult", 0.50)),
        "support_gap_mult": float(p.get("support_gap_mult", 1.20)),
        "tau_tight_mult": float(p.get("tau_tight_mult", 0.25)),
        "wd_mult": float(p.get("wd_mult", 0.75)),
        "wide_factor": float(p.get("wide_factor", 2.0)),
        "bin_height_mult": float(p.get("bin_height_mult", 1.5)),
        "sat_h_mult": float(p.get("sat_h_mult", 1.0)),
    }


# -----------------------------
# Box helpers
# -----------------------------

def _cc_edges(cc: CC) -> Tuple[int, int, int, int]:
    t, l, h, w = cc.bbox
    return t, l, t + h, l + w  # top, left, bottom, right


def _cc_center_xy(cc: CC) -> Tuple[float, float]:
    cy, cx = cc.box_center_yx
    return float(cx), float(cy)


def _union_bbox_from_cc_ids(cc_ids: List[int], cc_by_id: Dict[int, CC]) -> Tuple[float, float, float, float]:
    t_min = 1e18
    l_min = 1e18
    b_max = -1e18
    r_max = -1e18
    for cid in cc_ids:
        cc = cc_by_id[cid]
        t, l, b, r = _cc_edges(cc)
        t_min = min(t_min, t)
        l_min = min(l_min, l)
        b_max = max(b_max, b)
        r_max = max(r_max, r)
    return float(l_min), float(r_max), float(t_min), float(b_max)  # l,r,t,b


def _make_vbox_from_cc_members(new_id: int, member_ids: List[int], cc_by_id: Dict[int, CC], kind: CCKind) -> VBox:
    l, r, t, b = _union_bbox_from_cc_ids(member_ids, cc_by_id)
    w = r - l
    h = b - t
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    # Use median xh over members if present in ctx, else fallback to bbox height proxy
    xhs = []
    for cid in member_ids:
        cc = cc_by_id[cid]
        # CC does not carry xh; we store per-cc xh in ctx['g4']['xhd'] only, so use xhd proxy here.
        # This is OK because Phase 2 geometry is scaled by xh_d.
        xhs.append(h)
    xh = float(np.median(xhs)) if xhs else float(h)
    return VBox(id=new_id, l=l, r=r, t=t, b=b, w=w, h=h, cx=cx, cy=cy, xh=xh, kind=kind, members=list(member_ids))


def _ov_y_box(a: VBox, b: VBox) -> float:
    return max(0.0, min(a.b, b.b) - max(a.t, b.t))


def _gap_y_cc(P: CC, S: CC) -> float:
    Pt, Pl, Pb, Pr = _cc_edges(P)
    St, Sl, Sb, Sr = _cc_edges(S)
    return float(max(0, Pt - Sb, St - Pb))


def _gap_y_box(a: VBox, b: VBox) -> float:
    return float(max(0.0, a.t - b.b, b.t - a.b))


def _gap_x_box(a: VBox, b: VBox) -> float:
    return float(b.l - a.r)


# -----------------------------
# Grid for ids (hash)
# -----------------------------

def _build_grid_for_ccs(ccs: List[CC], g: float) -> Dict[Tuple[int, int], List[int]]:
    grid: Dict[Tuple[int, int], List[int]] = {}
    for cc in ccs:
        cx, cy = _cc_center_xy(cc)
        i = int(floor(cx / g))
        j = int(floor(cy / g))
        grid.setdefault((i, j), []).append(cc.cc_id)
    return grid


def _cand_cc_ids(P: CC, grid: Dict[Tuple[int, int], List[int]], g: float, margin_mult: float, xh_d: float) -> List[int]:
    # conservative radii: half extents + margin in xh_d units
    t, l, b, r = _cc_edges(P)
    cx, cy = _cc_center_xy(P)
    hx = max(r - cx, cx - l)
    hy = max(b - cy, cy - t)
    Rx = int(ceil((hx + margin_mult * xh_d) / g))
    Ry = int(ceil((hy + margin_mult * xh_d) / g))
    i0 = int(floor(cx / g))
    j0 = int(floor(cy / g))

    out: List[int] = []
    for i in range(i0 - Rx, i0 + Rx + 1):
        for j in range(j0 - Ry, j0 + Ry + 1):
            for qid in grid.get((i, j), []):
                if qid != P.cc_id:
                    out.append(qid)
    return out


# -----------------------------
# Pass 2A: V_Graph
# -----------------------------

def satellite(P: CC, S: CC, th: Dict[str, Any], xh_d: float) -> bool:
    """
    Return True iff S is a vertical satellite of planet P.

    Policy (CSUN aggressive):
      - P must be CHAR_LIKE
      - S may be SPECK or CHAR_LIKE

    Geometry:
      - Height gate is planet-relative: S.h <= sat_h_mult * P.h
      - Horizontal containment is the main safety lock
      - Vertical proximity uses planet-relative gap:
            delta = delta_mult * P.h
        (this is the key fix)
      - Envelope uses xh_d scaling (kept as-is)
    """
    # --- kind gates ---
    if P.kind != CCKind.CHAR_LIKE:
        return False
    if S.kind not in {CCKind.SPECK, CCKind.CHAR_LIKE}:
        return False
    if P.cc_id == S.cc_id:
        return False

    # --- unpack bboxes ---
    Pt, Pl, Pb, Pr = _cc_edges(P)
    St, Sl, Sb, Sr = _cc_edges(S)
    P_h = float(Pb - Pt)
    S_h = float(Sb - St)

    # --- size gate (planet-relative) ---
    sat_h_mult = float(th.get("sat_h_mult", 1.0))
    if S_h > sat_h_mult * P_h:
        return False

    # --- tolerances ---
    eps   = float(th.get("eps_mult", 0.10)) * float(xh_d)     # horizontal slack
    delta = float(th.get("delta_mult", 0.85)) * P_h           # <-- planet-relative gap (new)
    a     = float(th.get("a_mult", 1.00)) * float(xh_d)       # envelope above (keep)
    d     = float(th.get("d_mult", 0.70)) * float(xh_d)       # envelope below (keep)

    # --- containment / centering (primary safety lock) ---
    Scx, _Scy = _cc_center_xy(S)
    if Sl < Pl - eps:
        return False
    if Sr > Pr + eps:
        return False
    if not (Pl < Scx < Pr):
        return False

    # --- vertical proximity ---
    ov = max(0, min(Pb, Sb) - max(Pt, St))      # vertical overlap length
    gy = _gap_y_cc(P, S)                        # vertical gap (0 if overlap)
    if ov <= 0 and gy > delta:
        return False

    # --- vertical envelope (keep existing behavior) ---
    if St < Pt - a:
        return False
    if Sb > Pb + d:
        return False

    return True

def _vertical_companion_speck(P: CC, S: CC, th: Dict[str, Any], xh_d: float) -> bool:
    # Backward-compatible wrapper; prefer satellite(P,S,...)
    return satellite(P, S, th, xh_d)

def v_graph(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Phase 2A: V_Graph (vertical stacks): attaches SPECK and CHAR_LIKE satellites to CHAR_LIKE planets."""
    ccs: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs, list):
        raise ValueError("v_graph: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")
    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "xhd" not in g4:
        raise ValueError("v_graph: ctx['g4']['xhd'] missing (run g4_page_constants first)")

    th = _p2_params(cfg)
    xh_d = float(g4["xhd"])
    g = th["grid_edge_mult"] * xh_d

    cc_by_id: Dict[int, CC] = {cc.cc_id: cc for cc in ccs}

    # Grid built on union Planet ∪ Debris (all ccs)
    grid = _build_grid_for_ccs(ccs, g)

    planets = [cc for cc in ccs if cc.kind == CCKind.CHAR_LIKE]
    planets.sort(key=lambda c: (_cc_edges(c)[0], _cc_edges(c)[1], c.cc_id))

    assigned_to_planet: Dict[int, int] = {}
    used: set[int] = set()

    vboxes: List[VBox] = []
    vbox_by_id: Dict[int, VBox] = {}

    next_vid = 1_000_000  # large id space to avoid collisions with cc ids

    def fresh_id() -> int:
        nonlocal next_vid
        next_vid += 1
        return next_vid

    for P in planets:
        if P.cc_id in used:
            continue

        cand_ids = _cand_cc_ids(P, grid, g, th["cand_margin_mult"], xh_d)

        eligible_above: List[Tuple[float, float, int]] = []
        eligible_below: List[Tuple[float, float, int]] = []

        Pcx, Pcy = _cc_center_xy(P)

        for sid in cand_ids:
            if sid in used:
                continue
            if sid in assigned_to_planet:
                continue
            S = cc_by_id.get(sid)
            if S is None:
                continue

            if satellite(P, S, th, xh_d):
                gy = _gap_y_cc(P, S)
                Scx, Scy = _cc_center_xy(S)
                dx = abs(Scx - Pcx)
                if Scy < Pcy:
                    eligible_above.append((gy, dx, sid))
                else:
                    eligible_below.append((gy, dx, sid))

        eligible_above.sort()
        eligible_below.sort()

        companions: List[int] = []
        overcap_above = 0
        overcap_below = 0

        # take above
        for k, (_gy, _dx, sid) in enumerate(eligible_above):
            if k >= th["max_above"]:
                overcap_above += 1
                if th["allow_overcap"]:
                    continue
                break
            companions.append(sid)
            assigned_to_planet[sid] = P.cc_id

        # take below
        for k, (_gy, _dx, sid) in enumerate(eligible_below):
            if k >= th["max_below"]:
                overcap_below += 1
                if th["allow_overcap"]:
                    continue
                break
            companions.append(sid)
            assigned_to_planet[sid] = P.cc_id

        member_ids = [P.cc_id] + companions
        for mid in member_ids:
            used.add(mid)

        vid = fresh_id()
        kind = CCKind.V_CHAR_STACK if companions else P.kind
        vb = _make_vbox_from_cc_members(vid, member_ids, cc_by_id, kind)
        vb.overcap_above = overcap_above
        vb.overcap_below = overcap_below

        vboxes.append(vb)
        vbox_by_id[vid] = vb

    # Survivors as singleton VBoxes (including unassigned specks)
    for cc in ccs:
        if cc.cc_id not in used:
            vid = fresh_id()
            vb = _make_vbox_from_cc_members(vid, [cc.cc_id], cc_by_id, cc.kind)
            vboxes.append(vb)
            vbox_by_id[vid] = vb

    ctx["vboxes"] = vboxes
    ctx["vbox_by_id"] = vbox_by_id
    ctx["phase2"] = {"g": g, "params": th}


# -----------------------------
# Debug: V_Graph boxes-only (like g3_boxes_only)
# -----------------------------

def debug_render_vg_boxes_only(ctx: Context, cfg: Dict[str, Any]) -> None:
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_vg_boxes_only: ctx['page'] missing")

    vboxes: List[VBox] = ctx.get("vboxes", None)
    if not isinstance(vboxes, list):
        raise ValueError("debug_render_vg_boxes_only: ctx['vboxes'] missing (run v_graph first)")

    H = int(page["H"]); W = int(page["W"])
    stem = Path(page["path"]).stem
    out_dir = Path(cfg["outputs"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1: upscale = 1

    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)
    thickness = max(3, 2 * int(upscale))
    fill_alpha = float(cfg.get("outputs", {}).get("vg_boxes_only_fill_alpha", 1.0) or 1.0)
    fill_alpha = max(0.0, min(1.0, fill_alpha))
    overlay = rgb.copy()

    # draw VBoxes
    for vb in vboxes:
        # bbox in pixel coords
        x0 = int(vb.l * upscale); y0 = int(vb.t * upscale)
        x1 = int(vb.r * upscale); y1 = int(vb.b * upscale)
        x0 = max(0, min(x0, rgb.shape[1] - 1))
        x1 = max(0, min(x1, rgb.shape[1] - 1))
        y0 = max(0, min(y0, rgb.shape[0] - 1))
        y1 = max(0, min(y1, rgb.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        rgb_col = G3_DEBUG_RGB.get(vb.kind, G3_DEBUG_RGB[CCKind.UNKNOWN])
        bgr = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))

        if fill_alpha > 0.0:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=thickness)

        # member center markers (small squares)
        if cfg.get("outputs", {}).get("debug_draw_cc_box_centers", True):
            cx = int(vb.cx * upscale); cy = int(vb.cy * upscale)
            s = max(3, int(3 * upscale))
            cv2.rectangle(overlay, (cx - s, cy - s), (cx + s, cy + s), (0, 0, 0), thickness=thickness)

    if fill_alpha > 0.0:
        rgb = cv2.addWeighted(overlay, fill_alpha, rgb, 1.0 - fill_alpha, 0)

    out_path = out_dir / f"{stem}_vg_boxes_only.png"
    cv2.imwrite(str(out_path), rgb)


# -----------------------------
# Pass 2A.5: MarkFloaters on VBoxes
# -----------------------------

def mark_floaters(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    CSUN tokenization policy (Wayne):
      - V_Graph already decides which CCs are character glyphs (planet CHAR_LIKE + satellites).
      - At this phase we must NOT do shape/support semantics.
      - A VBox is text-like iff it contains at least one CHAR_LIKE CC member.
      - Otherwise it is a non-text blob (floater) that will be tokenized separately by H_Graph.

    Implementation:
      - Build cc_kind_by_id from ctx['ccs_g3'].
      - For each VBox:
          * keep BIG_SYM as-is (reserved)
          * if has any CHAR_LIKE member -> set kind=CHAR_LIKE (even if originally V_CHAR_STACK)
          * else -> set kind=FLOATER_BLOB
    """
    vboxes: List[VBox] = ctx.get("vboxes", None)
    vbox_by_id: Dict[int, VBox] = ctx.get("vbox_by_id", None)
    if not isinstance(vboxes, list) or not isinstance(vbox_by_id, dict):
        raise ValueError("mark_floaters: requires ctx['vboxes'] and ctx['vbox_by_id'] (run v_graph first)")

    ccs_g3: List[CC] = ctx.get("ccs_g3", None)
    if not isinstance(ccs_g3, list):
        raise ValueError("mark_floaters: ctx['ccs_g3'] missing (run classify_ccs_g3 first)")

    cc_kind_by_id: Dict[int, CCKind] = {int(cc.cc_id): cc.kind for cc in ccs_g3}

    def has_char_like_member(B: VBox) -> bool:
        return any(cc_kind_by_id.get(int(cid)) == CCKind.CHAR_LIKE for cid in B.members)

    for vb in vboxes:
        if vb.kind == getattr(CCKind, "BIG_SYM", CCKind.UNKNOWN):
            continue
        if has_char_like_member(vb):
            vb.kind = CCKind.CHAR_LIKE
        else:
            vb.kind = CCKind.FLOATER_BLOB

    ctx["vboxes"] = vboxes

# -----------------------------
# Pass 2B: H_Graph (tokens)
# -----------------------------

def h_graph(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Phase 2B: H_Graph (tokenization)

    CSUN policy (Wayne, "sloppy Fortran tokenization"):
      1) Build text runs by merging CHAR_LIKE VBoxes until a non-text VBox is met.
      2) Build non-text blob runs by merging FLOATER_BLOB VBoxes until a space or CHAR_LIKE is met.
      3) Spaces are emitted when the x-gap exceeds tau_tight; WD_SPACE vs WIDE_SPACE uses wd_mult and wide_factor.
      4) BIG_SYM is emitted as an atomic token (reserved for later parsing work).
    """
    vboxes: List[VBox] = ctx.get("vboxes", None)
    vbox_by_id: Dict[int, VBox] = ctx.get("vbox_by_id", None)
    if not isinstance(vboxes, list) or not isinstance(vbox_by_id, dict):
        raise ValueError("h_graph: requires ctx['vboxes'] and ctx['vbox_by_id'] (run v_graph first)")

    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "xhd" not in g4:
        raise ValueError("h_graph: ctx['g4']['xhd'] missing")

    th = _p2_params(cfg)
    xh_d = float(g4["xhd"])
    wd_space = th["wd_mult"] * xh_d
    bin_h = th["bin_height_mult"] * xh_d
    tau_tight = th["tau_tight_mult"] * xh_d

    bins: Dict[int, List[int]] = {}
    for vb in vboxes:
        k = int(floor(vb.cy / bin_h))
        bins.setdefault(k, []).append(vb.id)

    tokens: List[Token] = []

    def emit_h_run(member_ids: List[int]) -> None:
        l = min(vbox_by_id[i].l for i in member_ids)
        r = max(vbox_by_id[i].r for i in member_ids)
        t = min(vbox_by_id[i].t for i in member_ids)
        b = max(vbox_by_id[i].b for i in member_ids)
        w = r - l; h = b - t
        cx = (l + r) / 2.0; cy = (t + b) / 2.0
        src_id = int(floor(cy / bin_h))
        tokens.append(Token(kind="H_RUN", l=l, r=r, t=t, b=b, w=w, h=h, cx=cx, cy=cy, xh=float(xh_d), members=list(member_ids), src_id=src_id))

    def emit_blob_run(member_ids: List[int]) -> None:
        l = min(vbox_by_id[i].l for i in member_ids)
        r = max(vbox_by_id[i].r for i in member_ids)
        t = min(vbox_by_id[i].t for i in member_ids)
        b = max(vbox_by_id[i].b for i in member_ids)
        w = r - l; h = b - t
        cx = (l + r) / 2.0; cy = (t + b) / 2.0
        src_id = int(floor(cy / bin_h))
        tokens.append(Token(kind="FLOATER_BLOB_TOKEN", l=l, r=r, t=t, b=b, w=w, h=h, cx=cx, cy=cy, xh=float(xh_d), members=list(member_ids), src_id=src_id))

    def emit_atomic(kind: str, vid: int) -> None:
        B = vbox_by_id[vid]
        src_id = int(floor(B.cy / bin_h))
        tokens.append(Token(kind=kind, l=B.l, r=B.r, t=B.t, b=B.b, w=B.w, h=B.h, cx=B.cx, cy=B.cy, xh=float(xh_d), members=[vid], src_id=src_id))

    def emit_space(kind: str, A: VBox, B: VBox) -> None:
        l = A.r; r = B.l
        t = min(A.t, B.t); b = max(A.b, B.b)
        w = r - l; h = b - t
        cx = (l + r) / 2.0; cy = (t + b) / 2.0
        src_id = int(floor(cy / bin_h))
        tokens.append(Token(kind=kind, l=l, r=r, t=t, b=b, w=w, h=h, cx=cx, cy=cy, xh=float(xh_d), members=[], src_id=src_id))

    def flush(run_kind: str, run_ids: List[int]) -> None:
        if not run_ids:
            return
        if run_kind == "TEXT":
            emit_h_run(run_ids)
        elif run_kind == "BLOB":
            emit_blob_run(run_ids)
        else:
            emit_h_run(run_ids)

    for k in sorted(bins.keys()):
        ids = bins[k]
        ids.sort(key=lambda vid: (vbox_by_id[vid].l, vbox_by_id[vid].t, vid))

        run_kind: str = ""
        run: List[int] = []
        prev: Optional[int] = None

        for cur in ids:
            Cur = vbox_by_id[cur]
            cur_is_big = (Cur.kind == getattr(CCKind, "BIG_SYM", CCKind.UNKNOWN))
            cur_is_blob = (Cur.kind == CCKind.FLOATER_BLOB)
            cur_is_text = (Cur.kind == CCKind.CHAR_LIKE)

            if cur_is_big:
                flush(run_kind, run)
                run_kind, run, prev = "", [], None
                emit_atomic("BIG_SYM_TOKEN", cur)
                continue

            if prev is not None:
                Prev = vbox_by_id[prev]
                gx = Cur.l - Prev.r
                if gx >= 0 and gx > tau_tight:
                    flush(run_kind, run)
                    run_kind, run = "", []
                    if gx > th["wide_factor"] * wd_space:
                        emit_space("WIDE_SPACE", Prev, Cur)
                    else:
                        emit_space("WD_SPACE", Prev, Cur)
                    prev = None

            if cur_is_text:
                if run_kind != "TEXT":
                    flush(run_kind, run)
                    run_kind, run = "TEXT", []
                run.append(cur); prev = cur
                continue

            # treat everything else as blob (including Cur.kind == FLOATER_BLOB)
            if run_kind != "BLOB":
                flush(run_kind, run)
                run_kind, run = "BLOB", []
            run.append(cur); prev = cur

        flush(run_kind, run)

    ctx["tokens_h"] = tokens

# -----------------------------
# Debug: H_Graph token boxes-only
# -----------------------------

# ============================================================
# CSUN Token Debug Colors (LV-optimized)
# ============================================================

# RGB tuples (we convert to BGR at draw-time).
CSUN_TOKEN_RGB = {
    "H_RUN": (47, 107, 255),               # Blue
    "BIG_SYM_TOKEN": (122, 61, 184),       # Purple
    "FLOATER_BLOB_TOKEN": (255, 92, 168),  # Pink
    "WD_SPACE": (217, 217, 217),           # Light grey
    "WIDE_SPACE": (184, 184, 184),         # Darker light grey

    # Optional shape colors (not used unless you decide to override by shape later)
    "FracBarLike": (255, 179, 107),        # Pale orange
    "FenceLike": (120, 200, 255),          # Light blue
    "Other": (216, 199, 166),              # Beige
}


# -----------------------------
# Debug: H_Graph token boxes-only (CSUN colors + key)
# -----------------------------

def debug_render_hg_boxes_only(ctx: Context, cfg: Dict[str, Any]) -> None:
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_hg_boxes_only: ctx['page'] missing")

    tokens: List[Token] = ctx.get("tokens_h", None)
    if not isinstance(tokens, list):
        raise ValueError("debug_render_hg_boxes_only: ctx['tokens_h'] missing (run h_graph first)")

    H = int(page["H"]); W = int(page["W"])
    stem = Path(page["path"]).stem
    out_dir = Path(cfg["outputs"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    # blank white canvas (no glyph ink)
    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)

    thickness = max(3, 2 * int(upscale))
    overlay = rgb.copy()

    # Draw token rectangles (solid fill + thick outline)
    for i, tok in enumerate(tokens, start=1):
        tok.token_id = tok.token_id or i
        col = CSUN_TOKEN_RGB.get(tok.kind, CSUN_TOKEN_RGB["Other"])
        bgr = (int(col[2]), int(col[1]), int(col[0]))

        x0 = int(tok.l * upscale); y0 = int(tok.t * upscale)
        x1 = int(tok.r * upscale); y1 = int(tok.b * upscale)
        x0 = max(0, min(x0, overlay.shape[1] - 1))
        x1 = max(0, min(x1, overlay.shape[1] - 1))
        y0 = max(0, min(y0, overlay.shape[0] - 1))
        y1 = max(0, min(y1, overlay.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=thickness)

    rgb = overlay

    # --------------------------------------------------
    # Color key (top-left)
    # --------------------------------------------------
    key_items = [
        ("H_RUN", "Blue"),
        ("BIG_SYM_TOKEN", "Purple"),
        ("FLOATER_BLOB_TOKEN", "Pink"),
        ("WD_SPACE", "Light Grey"),
        ("WIDE_SPACE", "Darker Grey"),
    ]

    key_x = 20
    key_y = 40
    box_size = 30
    for name, label in key_items:
        col = CSUN_TOKEN_RGB.get(name, CSUN_TOKEN_RGB["Other"])
        bgr = (int(col[2]), int(col[1]), int(col[0]))

        cv2.rectangle(rgb, (key_x, key_y - box_size), (key_x + box_size, key_y), bgr, thickness=-1)
        cv2.putText(
            rgb,
            f"{name} ({label})",
            (key_x + box_size + 15, key_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        key_y += 50

    out_path = out_dir / f"{stem}_hg_boxes_only_CSUN.png"
    cv2.imwrite(str(out_path), rgb)
    ctx["debug_hg_boxes_only_png"] = str(out_path)


# ============================================================
# G5a: Token classification + LV-friendly text report
# ============================================================

def g5a_classify_tokens(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G5a: Token *behavioral* classification in post-G4 context.

    Inputs:
      - ctx["tokens_h"] : List[Token] from P2c_H_Graph
      - ctx["g4"]["xhd"] : xh_d

    Writes:
      - ctx["tokens_g5a"] : same Token objects with added attributes:
            tok.role   : str
            tok.inline : bool (meaningful for FLOATER_BLOB_TOKEN)
            tok.shape  : str (heuristic shape label for non-text tokens)
    """
    tokens: List[Token] = ctx.get("tokens_h", None)
    if not isinstance(tokens, list):
        raise ValueError("g5a_classify_tokens: ctx['tokens_h'] missing (run h_graph first)")

    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "xhd" not in g4:
        raise ValueError("g5a_classify_tokens: ctx['g4']['xhd'] missing (run g4_page_constants first)")
    xh_d = float(g4["xhd"])

    # Phase-2 knobs (already in Script.yaml)
    p2 = cfg.get("params", {}).get("phase2", {}) or {}
    bin_height_mult   = float(p2.get("bin_height_mult", 1.5))
    ov_mult           = float(p2.get("ov_mult", 0.5))
    support_gap_mult  = float(p2.get("support_gap_mult", 1.2))

    # Shape heuristics (for blobs/large symbols)
    tall_mult = float(p2.get("tall_mult", 2.8))
    thin_mult = float(p2.get("thin_mult", 0.35))
    wide_mult = float(p2.get("wide_mult", 3.0))
    flat_mult = float(p2.get("flat_mult", 0.25))

    # BigSym band (relative to xh_d). Keep as Phase2 params; do not rename existing keys.
    big_sym_h_min_mult = float(p2.get("big_sym_h_min_mult", 1.60))
    big_sym_h_max_mult = float(p2.get("big_sym_h_max_mult", 3.50))

    # Near-square band (optional)
    g4p = cfg.get("params", {}).get("g4", {}) or {}
    near_sq_min = float(g4p.get("near_square_min", 0.9))
    near_sq_max = float(g4p.get("near_square_max", 1.1))

    bin_h = max(1.0, bin_height_mult * xh_d)
    ov_y_min = ov_mult * xh_d
    gap_x_max = support_gap_mult * xh_d

    # Bin tokens by cy (same binning style as Phase2)
    bins: Dict[int, List[Token]] = {}
    for tok in tokens:
        k = int(floor(float(tok.cy) / bin_h))
        bins.setdefault(k, []).append(tok)

    def ov_y(a: Token, b: Token) -> float:
        return max(0.0, min(float(a.b), float(b.b)) - max(float(a.t), float(b.t)))

    def gap_left(L: Token, R: Token) -> float:
        # positive if disjoint in x
        return float(R.l) - float(L.r)

    def classify_shape(tok: Token) -> str:
        """
        Heuristic shape label for non-text tokens.
        NOTE: This is *not* the same as tok.role.
        """
        w = float(tok.w)
        h = float(tok.h)

        if (w >= wide_mult * xh_d) and (h <= flat_mult * xh_d):
            return "FracBarLike"
        if (h >= tall_mult * xh_d) and (w <= thin_mult * xh_d):
            return "FenceLike"

        ratio = w / max(h, 1e-6)
        if near_sq_min <= ratio <= near_sq_max:
            return "BigSymLike"

        return "Other"

    def floater_role(tok: Token) -> str:
        """
        Geometric floater role (LV-friendly, no ALL CAPS):
          Fence | Bar | BigSym | Unknown
        Uses xh_d-scaled thresholds (Phase2 multipliers).
        """
        w = float(tok.w)
        h = float(tok.h)

        # Fence: tall + thin
        if (h >= tall_mult * xh_d) and (w <= thin_mult * xh_d):
            return "Fence"

        # Bar: wide + flat
        if (w >= wide_mult * xh_d) and (h <= flat_mult * xh_d):
            return "Bar"

        # BigSym: size band between text and true images (height-based)
        if (h >= big_sym_h_min_mult * xh_d) and (h <= big_sym_h_max_mult * xh_d):
            return "BigSym"

        return "Unknown"

    # Default annotations
    for tok in tokens:
        tok.role = "Other"
        tok.inline = False

        # Preserve old fields
        tok.is_text_like = (tok.kind == "H_RUN")
        tok.length = int(len(tok.members) if tok.kind == "H_RUN" and tok.members else (1 if tok.kind in {"FLOATER_BLOB_TOKEN","BIG_SYM_TOKEN"} else 0))
        tok.singleton = bool(tok.kind == "H_RUN" and tok.length == 1)

        if tok.kind == "H_RUN":
            tok.role = "TextRun"
            tok.inline = True
            tok.shape = "-"
        elif tok.kind in {"WD_SPACE", "WIDE_SPACE"}:
            tok.role = "Space"
            tok.inline = True
            tok.shape = "-"
        elif tok.kind == "BIG_SYM_TOKEN":
            tok.role = "BigSym"
            tok.inline = False
            tok.shape = "BigSymLike"
        elif tok.kind == "FLOATER_BLOB_TOKEN":
            tok.role = floater_role(tok)   # Fence | Bar | BigSym | Unknown
            tok.shape = classify_shape(tok)
            # inline will be decided below
        else:
            tok.shape = "-"

    # Decide floater inline flag using same-bin text support.
    # IMPORTANT: role is geometric (Fence/Bar/BigSym/Unknown). Inline is a separate attribute.
    for k, toks in bins.items():
        # Sort left-to-right
        toks.sort(key=lambda t: (float(t.l), float(t.t), t.kind))

        # Pre-extract text runs in this bin
        text_runs = [t for t in toks if t.kind == "H_RUN"]
        if not text_runs:
            continue

        for tok in toks:
            if tok.kind != "FLOATER_BLOB_TOKEN":
                continue

            # Find nearest supporting text to left and right with sufficient y-overlap
            best = float("inf")

            # left
            for L in reversed(text_runs):
                if float(L.r) <= float(tok.l) and ov_y(tok, L) >= ov_y_min:
                    g = gap_left(L, tok)
                    if g >= 0:
                        best = min(best, g)
                    break

            # right
            for R in text_runs:
                if float(R.l) >= float(tok.r) and ov_y(tok, R) >= ov_y_min:
                    g = gap_left(tok, R)
                    if g >= 0:
                        best = min(best, g)
                    break

            tok.inline = (best <= gap_x_max)

    ctx["tokens_g5a"] = tokens
    ctx["g5a"] = {
        "xh_d": xh_d,
        "bin_height_mult": bin_height_mult,
        "ov_mult": ov_mult,
        "support_gap_mult": support_gap_mult,
        "tall_mult": tall_mult,
        "thin_mult": thin_mult,
        "wide_mult": wide_mult,
        "flat_mult": flat_mult,
        "big_sym_h_min_mult": big_sym_h_min_mult,
        "big_sym_h_max_mult": big_sym_h_max_mult,
        "near_sq_min": near_sq_min,
        "near_sq_max": near_sq_max,
    }


# -----------------------------
# Debug: G5a token boxes-only (post-classification)
# -----------------------------

def debug_render_g5a_tokens(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Render a *boxes-only* token debug PNG after G5a classification.

    This is the missing view Wayne needs: same geometry as H_Graph tokens,
    but colored by *tok.role* (Fence/Bar/BigSym/Unknown/TextRun/Space) when present,
    falling back to tok.kind for tokens without a role.

    Output: <stem>_g5a_boxes_only.png in outputs.out_dir
    """

    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g5a_tokens: ctx['page'] missing")

    # Prefer post-G5a tokens if present; fall back to raw H_Graph tokens.
    tokens: List[Token] = ctx.get("tokens_g5a", None)
    if not isinstance(tokens, list):
        tokens = ctx.get("tokens_h", None)
    if not isinstance(tokens, list):
        raise ValueError("debug_render_g5a_tokens: ctx['tokens_g5a'] or ctx['tokens_h'] missing")

    H = int(page["H"]); W = int(page["W"])
    stem = Path(page["path"]).stem
    out_dir = Path(cfg["outputs"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    # Blank white canvas (no glyph ink)
    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)
    thickness = max(3, 2 * int(upscale))
    overlay = rgb.copy()

    # Role palette (RGB; convert to BGR for OpenCV)
    ROLE_RGB = {
        # Text / spacing
        "TextRun": (47, 107, 255),     # blue (match CSUN H_RUN)
        "Space": (217, 217, 217),      # light gray

        # Floater sub-roles
        "Fence": (120, 200, 255),      # light blue
        "Bar": (255, 179, 107),        # pale orange
        "BigSym": (122, 61, 184),      # purple
        "Unknown": (255, 92, 168),     # pink (unknown blob)
        "Other": (216, 199, 166),      # beige
    }

    # Kind fallback palette (RGB)
    KIND_RGB = {
        "H_RUN": (47, 107, 255),
        "BIG_SYM_TOKEN": (122, 61, 184),
        "FLOATER_BLOB_TOKEN": (255, 92, 168),
        "WD_SPACE": (217, 217, 217),
        "WIDE_SPACE": (184, 184, 184),
    }

    # Draw token rectangles (solid fill + thick outline)
    for i, tok in enumerate(tokens, start=1):
        tok.token_id = getattr(tok, "token_id", 0) or i

        role = getattr(tok, "role", None)
        kind = getattr(tok, "kind", None)

        if role in ROLE_RGB:
            col = ROLE_RGB[role]
        elif kind in KIND_RGB:
            col = KIND_RGB[kind]
        else:
            col = ROLE_RGB["Other"]

        bgr = (int(col[2]), int(col[1]), int(col[0]))

        x0 = int(float(tok.l) * upscale); y0 = int(float(tok.t) * upscale)
        x1 = int(float(tok.r) * upscale); y1 = int(float(tok.b) * upscale)
        x0 = max(0, min(x0, overlay.shape[1] - 1))
        x1 = max(0, min(x1, overlay.shape[1] - 1))
        y0 = max(0, min(y0, overlay.shape[0] - 1))
        y1 = max(0, min(y1, overlay.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=thickness)

    rgb = overlay

    # --------------------------------------------------
    # Key (top-left)
    # --------------------------------------------------
    key_items = [
        ("TextRun", "Blue"),
        ("BigSym", "Purple"),
        ("Unknown", "Pink"),
        ("Space", "Light Grey"),
        ("Fence", "Light Blue"),
        ("Bar", "Pale Orange"),
    ]

    key_x = 20
    key_y = 40
    box_size = 30
    for name, label in key_items:
        col = ROLE_RGB.get(name, ROLE_RGB["Other"])
        bgr = (int(col[2]), int(col[1]), int(col[0]))

        cv2.rectangle(rgb, (key_x, key_y - box_size), (key_x + box_size, key_y), bgr, thickness=-1)
        cv2.putText(
            rgb,
            f"{name} ({label})",
            (key_x + box_size + 15, key_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        key_y += 50

    out_path = out_dir / f"{stem}_g5a_boxes_only.png"
    cv2.imwrite(str(out_path), rgb)
    ctx["debug_g5a_boxes_only_png"] = str(out_path)


def debug_render_g5a_gap_stream(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    Render a *boxes-only* debug PNG that explicitly shows the alternating

        TOKEN  GAP  TOKEN  GAP  TOKEN ...

    structure on each detected source line.

    Notes:
      - Gaps are *geometric* (no WD/WIDE semantics here).
      - A GAP box is defined from adjacent tokens B0 (pred) and B1 (succ):
            G.l = B0.r, G.r = B1.l
            G.t = min(B0.t, B1.t), G.b = max(B0.b, B1.b)
        (matching Wayne's cleaned GAP definition)

    Output: <stem>_g5a_gap_stream.png in outputs.out_dir
    """

    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g5a_gap_stream: ctx['page'] missing")

    # Prefer post-G5a tokens; fall back to raw H_Graph tokens.
    tokens: List[Token] = ctx.get("tokens_g5a", None)
    if not isinstance(tokens, list):
        tokens = ctx.get("tokens_h", None)
    if not isinstance(tokens, list):
        raise ValueError("debug_render_g5a_gap_stream: ctx['tokens_g5a'] or ctx['tokens_h'] missing")

    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "xhd" not in g4:
        raise ValueError("debug_render_g5a_gap_stream: ctx['g4']['xhd'] missing (run g4_page_constants first)")
    xh_d = float(g4["xhd"])

    H = int(page["H"]); W = int(page["W"])
    stem = Path(page["path"]).stem
    out_dir = Path(cfg["outputs"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1:
        upscale = 1

    # Blank white canvas (no glyph ink)
    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)
    overlay = rgb.copy()
    thickness = max(3, 2 * int(upscale))

    # Token role palette (RGB; convert to BGR for OpenCV)
    ROLE_RGB = {
        "TextRun": (47, 107, 255),     # blue
        "Fence": (120, 200, 255),      # light blue
        "Bar": (255, 179, 107),        # pale orange
        "BigSym": (122, 61, 184),      # purple
        "Unknown": (255, 92, 168),     # pink
        "Other": (216, 199, 166),      # beige
    }

    # GAP palette (RGB)
    GAP_RGB = {
        "GAP": (217, 217, 217),        # light gray fill
        "GAP_OUTLINE": (0, 0, 0),      # black outline
    }

    # Line binning (same as G5a)
    p2 = cfg.get("params", {}).get("phase2", {}) or {}
    bin_height_mult = float(p2.get("bin_height_mult", 1.5))
    bin_h = max(1.0, bin_height_mult * xh_d)

    bins: Dict[int, List[Token]] = {}
    for tok in tokens:
        k = int(floor(float(tok.cy) / bin_h))
        bins.setdefault(k, []).append(tok)

    def _clip_rect(x0: int, y0: int, x1: int, y1: int):
        x0 = max(0, min(x0, overlay.shape[1] - 1))
        x1 = max(0, min(x1, overlay.shape[1] - 1))
        y0 = max(0, min(y0, overlay.shape[0] - 1))
        y1 = max(0, min(y1, overlay.shape[0] - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    # Draw per-line TOKEN / GAP stream
    for _, line_toks in sorted(bins.items(), key=lambda kv: kv[0]):
        line = sorted(line_toks, key=lambda t: (float(t.l), float(t.t)))

        # Draw tokens first
        for tok in line:
            role = getattr(tok, "role", None)
            kind = getattr(tok, "kind", None)

            if role in ROLE_RGB:
                col = ROLE_RGB[role]
            else:
                # fall back to kind → a role-like guess
                if kind == "H_RUN":
                    col = ROLE_RGB["TextRun"]
                elif kind == "BIG_SYM_TOKEN":
                    col = ROLE_RGB["BigSym"]
                elif kind == "FLOATER_BLOB_TOKEN":
                    col = ROLE_RGB["Unknown"]
                else:
                    col = ROLE_RGB["Other"]

            bgr = (int(col[2]), int(col[1]), int(col[0]))
            x0 = int(float(tok.l) * upscale); y0 = int(float(tok.t) * upscale)
            x1 = int(float(tok.r) * upscale); y1 = int(float(tok.b) * upscale)
            clipped = _clip_rect(x0, y0, x1, y1)
            if clipped is None:
                continue
            x0, y0, x1, y1 = clipped
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=-1)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, thickness=thickness)

        # Now draw gaps between adjacent tokens on this line
        for a, b in zip(line, line[1:]):
            gl = float(a.r)
            gr = float(b.l)
            if gr <= gl:
                continue  # overlap/touch => no gap token

            gt = min(float(a.t), float(b.t))
            gb = max(float(a.b), float(b.b))

            x0 = int(gl * upscale); x1 = int(gr * upscale)
            y0 = int(gt * upscale); y1 = int(gb * upscale)

            clipped = _clip_rect(x0, y0, x1, y1)
            if clipped is None:
                continue
            x0, y0, x1, y1 = clipped

            fill = GAP_RGB["GAP"]
            bgr_fill = (int(fill[2]), int(fill[1]), int(fill[0]))
            out = GAP_RGB["GAP_OUTLINE"]
            bgr_out = (int(out[2]), int(out[1]), int(out[0]))

            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr_fill, thickness=-1)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr_out, thickness=max(1, thickness // 2))

    rgb = overlay

    # Key (top-left)
    key_items = [
        ("Token (TextRun)", ROLE_RGB["TextRun"]),
        ("Token (BigSym)", ROLE_RGB["BigSym"]),
        ("Token (Fence)", ROLE_RGB["Fence"]),
        ("Token (Bar)", ROLE_RGB["Bar"]),
        ("Token (Unknown)", ROLE_RGB["Unknown"]),
        ("GAP (geometry)", GAP_RGB["GAP"]),
    ]
    key_x = 20
    key_y = 40
    box_size = 30
    for label, col in key_items:
        bgr = (int(col[2]), int(col[1]), int(col[0]))
        cv2.rectangle(rgb, (key_x, key_y - box_size), (key_x + box_size, key_y), bgr, thickness=-1)
        cv2.putText(
            rgb,
            label,
            (key_x + box_size + 15, key_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        key_y += 50

    out_path = out_dir / f"{stem}_g5a_gap_stream.png"
    cv2.imwrite(str(out_path), rgb)
    ctx["debug_g5a_gap_stream_png"] = str(out_path)


def g5a_write_token_report(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    Write an LV-friendly YAML token report: <stem>_g5a_tokens.yml
    No external YAML dependency; we write simple YAML by hand.
    """
    from pathlib import Path

    page = ctx.get("page", None)
    if page is None:
        raise ValueError("g5a_write_token_report: ctx['page'] missing")

    tokens: List[Token] = ctx.get("tokens_g5a", None)
    if not isinstance(tokens, list):
        raise ValueError("g5a_write_token_report: ctx['tokens_g5a'] missing (run g5a_classify_tokens first)")

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page["path"]).stem

    # --- counts ---
    kind_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    inline_floaters = 0
    block_floaters = 0

    for tok in tokens:
        kind_counts[tok.kind] = kind_counts.get(tok.kind, 0) + 1
        role = getattr(tok, "role", "OTHER") or "OTHER"
        role_counts[role] = role_counts.get(role, 0) + 1
        if tok.kind == "FLOATER_BLOB_TOKEN":
            if bool(getattr(tok, "inline", False)):
                inline_floaters += 1
            else:
                block_floaters += 1

    g5a = ctx.get("g5a", {}) or {}
    xh_d = g5a.get("xh_d", None)

    # Stable readable order: top->bottom, left->right
    toks_sorted = sorted(tokens, key=lambda t: (float(t.t), float(t.l), t.kind))

    # --- YAML helpers ---
    def y_str(s: str) -> str:
        # Safe single-line YAML string quoting
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{s}\""

    def y_bool(b: bool) -> str:
        return "true" if b else "false"

    def y_num(x) -> str:
        # keep ints as ints, floats as compact floats
        if isinstance(x, bool):
            return y_bool(x)
        if isinstance(x, int):
            return str(x)
        try:
            fx = float(x)
            if abs(fx - int(round(fx))) < 1e-9:
                return str(int(round(fx)))
            return f"{fx:.3f}".rstrip("0").rstrip(".")
        except Exception:
            return y_str(str(x))

    lines: List[str] = []
    lines.append("g5a_report:")
    lines.append(f"  stem: {y_str(stem)}")
    if xh_d is not None:
        lines.append(f"  xh_d: {y_num(xh_d)}")

    # knobs (optional but handy)
    lines.append("  params:")
    for k in ["tall_mult", "thin_mult", "wide_mult", "flat_mult", "near_sq_min", "near_sq_max"]:
        if k in g5a:
            lines.append(f"    {k}: {y_num(g5a[k])}")

    lines.append("  counts:")
    lines.append("    by_kind:")
    for k in sorted(kind_counts.keys()):
        lines.append(f"      {k}: {kind_counts[k]}")
    lines.append("    by_role:")
    for k in sorted(role_counts.keys()):
        lines.append(f"      {k}: {role_counts[k]}")
    lines.append("    floaters:")
    lines.append(f"      inline: {inline_floaters}")
    lines.append(f"      block: {block_floaters}")

    lines.append("  tokens:")
    for i, tok in enumerate(toks_sorted, 1):
        role = getattr(tok, "role", "OTHER") or "OTHER"
        inline = bool(getattr(tok, "inline", False))
        shape = getattr(tok, "shape", None) or "-"
        is_txt = bool(getattr(tok, "is_text_like", False))
        length = int(getattr(tok, "length", 0) or 0)
        mem = int(len(tok.members) if getattr(tok, "members", None) else 0)

        l = int(round(float(tok.l))); t = int(round(float(tok.t)))
        r = int(round(float(tok.r))); b = int(round(float(tok.b)))
        w = int(round(float(tok.w))); h = int(round(float(tok.h)))

        lines.append("    - idx: " + str(i))
        lines.append("      kind: " + y_str(tok.kind))
        lines.append("      role: " + y_str(role))
        lines.append("      inline: " + y_bool(inline))
        lines.append("      shape: " + y_str(shape))
        lines.append("      is_text_like: " + y_bool(is_txt))
        lines.append("      length: " + str(length))
        lines.append("      bbox:")
        lines.append(f"        l: {l}")
        lines.append(f"        t: {t}")
        lines.append(f"        r: {r}")
        lines.append(f"        b: {b}")
        lines.append(f"        w: {w}")
        lines.append(f"        h: {h}")
        lines.append("      members_count: " + str(mem))

    out_path = out_dir / f"{stem}_g5a_tokens.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    ctx["g5a_report_yml"] = str(out_path)
    """Write a compact, LV-friendly text report of the G5a-classified tokens."""
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("g5a_write_token_report: ctx['page'] missing")

    tokens: List[Token] = ctx.get("tokens_g5a", None)
    if not isinstance(tokens, list):
        raise ValueError("g5a_write_token_report: ctx['tokens_g5a'] missing (run g5a_classify_tokens first)")

    out_dir = Path(cfg.get("outputs", {}).get("out_dir", "out_geo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page["path"]).stem
    out_path = out_dir / f"{stem}_g5a_tokens.txt"

    # counts
    kind_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    shape_counts: Dict[str, int] = {}
    inline_float = 0
    block_float = 0

    for tok in tokens:
        kind_counts[tok.kind] = kind_counts.get(tok.kind, 0) + 1
        role = getattr(tok, "role", "-") or "-"
        role_counts[role] = role_counts.get(role, 0) + 1
        shp = getattr(tok, "shape", None) or "-"
        shape_counts[shp] = shape_counts.get(shp, 0) + 1

        if tok.kind == "FLOATER_BLOB_TOKEN":
            if getattr(tok, "inline", False):
                inline_float += 1
            else:
                block_float += 1

    g5a = ctx.get("g5a", {})
    xh_d = g5a.get("xh_d", None)

    # stable sort (top->bottom, left->right)
    toks_sorted = sorted(tokens, key=lambda t: (float(t.t), float(t.l), t.kind))

    lines: List[str] = []
    lines.append(f"G5a Token Report: {stem}")
    lines.append("")
    if xh_d is not None:
        lines.append(f"xh_d = {float(xh_d):.3f}")
    lines.append(f"phase2 multipliers: ov={g5a.get('ov_mult')}, support_gap={g5a.get('support_gap_mult')}, bin_h_mult={g5a.get('bin_height_mult')}")
    lines.append(f"shape mults: tall={g5a.get('tall_mult')}, thin={g5a.get('thin_mult')}, wide={g5a.get('wide_mult')}, flat={g5a.get('flat_mult')}")
    lines.append(f"near-square band: [{g5a.get('near_sq_min')}, {g5a.get('near_sq_max')}]")
    lines.append("")
    lines.append(f"Floaters: inline={inline_float}, block={block_float}")
    lines.append("")

    lines.append("Counts by kind:")
    for k in sorted(kind_counts.keys()):
        lines.append(f"  {k}: {kind_counts[k]}")
    lines.append("")
    lines.append("Counts by role:")
    for k in sorted(role_counts.keys()):
        lines.append(f"  {k}: {role_counts[k]}")
    lines.append("")
    lines.append("Counts by shape:")
    for s in sorted(shape_counts.keys()):
        lines.append(f"  {s}: {shape_counts[s]}")

    lines.append("")
    lines.append("Token list (sorted by top,left):")
    lines.append("  idx  kind               role             inl shape            txt len  l   t   r   b   w   h   members")
    lines.append("  ---  -----------------  ---------------  --- ---------------  --- ---  --- --- --- --- --- --- --------")

    for i, tok in enumerate(toks_sorted, 1):
        kind = tok.kind
        role = (getattr(tok, "role", None) or "-")
        inline = "Y" if getattr(tok, "inline", False) else "N"
        shape = (getattr(tok, "shape", None) or "-")

        is_txt = "Y" if getattr(tok, "is_text_like", False) else "N"
        length = int(getattr(tok, "length", 0) or 0)
        mem = len(tok.members) if getattr(tok, "members", None) else 0

        l = int(round(float(tok.l))); t = int(round(float(tok.t)))
        r = int(round(float(tok.r))); b = int(round(float(tok.b)))
        w = int(round(float(tok.w))); h = int(round(float(tok.h)))

        lines.append(f"  {i:3d}  {kind:<17}  {role:<15}  {inline:^3} {shape:<15}  {is_txt:^3} {length:3d}  {l:3d} {t:3d} {r:3d} {b:3d} {w:3d} {h:3d} {mem:8d}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    ctx["g5a_report_txt"] = str(out_path)


# ============================================================
# G6a: Fraction blocks
# ============================================================

def g6a_fraction_blocks(ctx: Context, cfg: Dict[str, Any]) -> None:
    """Detect first-pass fraction blocks from G5 tokens.

    Output in ctx['g6a_frac_blocks'] as list of dicts with:
      id, bar_token_id, numerator_token_ids, denominator_token_ids, child_token_ids, bbox, src_ids, confidence, warnings
    """
    tokens: List[Token] = ctx.get("tokens_g5a") or ctx.get("tokens_h")
    if not isinstance(tokens, list):
        raise ValueError("g6a_fraction_blocks: requires ctx['tokens_g5a'] or ctx['tokens_h']")
    g4 = ctx.get("g4", None)
    if not isinstance(g4, dict) or "xhd" not in g4:
        raise ValueError("g6a_fraction_blocks: ctx['g4']['xhd'] missing")
    xh_d = float(g4["xhd"])

    p2 = cfg.get("params", {}).get("phase2", {}) or {}
    frac_bar_min_w = float(p2.get("frac_bar_min_w_mult", 0.8)) * xh_d
    frac_bar_max_h = float(p2.get("frac_bar_max_h_mult", 0.60)) * xh_d
    frac_aspect_min = float(p2.get("frac_bar_aspect_min", 3.5))
    frac_margin_x = float(p2.get("frac_margin_x_mult", 0.3)) * xh_d
    frac_support_y = float(p2.get("frac_support_y_mult", 2.5)) * xh_d
    frac_overlap_min = float(p2.get("frac_overlap_min", 0.25))

    # Assign stable token ids if missing
    for i, tok in enumerate(tokens, 1):
        if not getattr(tok, 'token_id', 0):
            tok.token_id = i

    non_space = [t for t in tokens if t.kind not in {"WD_SPACE", "WIDE_SPACE"}]

    def overlap_w(a: Token, b: Token) -> float:
        return max(0.0, min(float(a.r), float(b.r)) - max(float(a.l), float(b.l)))

    def is_bar(tok: Token) -> bool:
        return (float(tok.w) >= frac_bar_min_w and float(tok.h) <= max(1.0, frac_bar_max_h) and
                float(tok.w) / max(float(tok.h), 1.0) >= frac_aspect_min)

    def in_x_band(tok: Token, bar: Token) -> bool:
        ext_l = float(bar.l) - frac_margin_x
        ext_r = float(bar.r) + frac_margin_x
        ov = max(0.0, min(float(tok.r), ext_r) - max(float(tok.l), ext_l))
        return ov > 0.0

    frac_blocks: List[Dict[str, Any]] = []
    consumed: set[int] = set()

    for bar in sorted(non_space, key=lambda t: (float(t.t), float(t.l))):
        if bar.token_id in consumed or not is_bar(bar):
            continue

        above = []
        below = []
        for tok in non_space:
            if tok.token_id == bar.token_id or tok.token_id in consumed:
                continue
            if not in_x_band(tok, bar):
                continue
            ov = overlap_w(tok, bar) / max(1.0, min(float(tok.w), float(bar.w)))
            if ov < frac_overlap_min:
                continue
            if float(tok.cy) < float(bar.cy):
                vgap = float(bar.t) - float(tok.b)
                if -0.25 * xh_d <= vgap <= frac_support_y:
                    above.append((vgap, tok))
            elif float(tok.cy) > float(bar.cy):
                vgap = float(tok.t) - float(bar.b)
                if -0.25 * xh_d <= vgap <= frac_support_y:
                    below.append((vgap, tok))

        if not above or not below:
            continue

        # choose nearest source rows above and below, then collect all tokens on those rows in x band
        nearest_above_src = min(above, key=lambda x: max(0.0, x[0]))[1].src_id
        nearest_below_src = min(below, key=lambda x: max(0.0, x[0]))[1].src_id

        num = [tok for _, tok in above
           if tok.src_id <= nearest_above_src
           and tok.src_id >= nearest_above_src - 2
           and in_x_band(tok, bar)]
        den = [tok for _, tok in below
           if tok.src_id >= nearest_below_src
           and tok.src_id <= nearest_below_src + 2
           and in_x_band(tok, bar)]
        if not num or not den:
            continue

        child_ids = [t.token_id for t in num] + [bar.token_id] + [t.token_id for t in den]
        if any(cid in consumed for cid in child_ids):
            continue

        l = min([bar.l] + [t.l for t in num] + [t.l for t in den])
        r = max([bar.r] + [t.r for t in num] + [t.r for t in den])
        t0 = min([bar.t] + [t.t for t in num] + [t.t for t in den])
        b0 = max([bar.b] + [t.b for t in num] + [t.b for t in den])

        frac_blocks.append({
            "id": len(frac_blocks) + 1,
            "bar_token_id": int(bar.token_id),
            "numerator_token_ids": [int(t.token_id) for t in sorted(num, key=lambda z: (z.l, z.t))],
            "denominator_token_ids": [int(t.token_id) for t in sorted(den, key=lambda z: (z.l, z.t))],
            "child_token_ids": [int(x) for x in child_ids],
            "bbox": {"l": int(round(l)), "t": int(round(t0)), "r": int(round(r)), "b": int(round(b0)),
                     "w": int(round(r-l)), "h": int(round(b0-t0))},
            "src_ids": sorted(set([int(bar.src_id)] + [int(t.src_id) for t in num] + [int(t.src_id) for t in den])),
            "confidence": round(min(1.0, 0.5 + 0.1 * len(num) + 0.1 * len(den)), 3),
            "warnings": []
        })
        consumed.update(child_ids)

    ctx["g6a_frac_blocks"] = frac_blocks
    ctx["g6a"] = {
        "frac_bar_min_w": frac_bar_min_w,
        "frac_bar_max_h": frac_bar_max_h,
        "frac_aspect_min": frac_aspect_min,
        "frac_margin_x": frac_margin_x,
        "frac_support_y": frac_support_y,
        "frac_overlap_min": frac_overlap_min,
    }


def g6a_write_frac_report(ctx: Context, cfg: Dict[str, Any]) -> None:
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("g6a_write_frac_report: ctx['page'] missing")
    fracs = ctx.get("g6a_frac_blocks", None)
    if not isinstance(fracs, list):
        raise ValueError("g6a_write_frac_report: ctx['g6a_frac_blocks'] missing (run g6a_fraction_blocks first)")
    out_dir = Path(cfg.get("outputs", {}).get("out_dir", "out_geo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page["path"]).stem
    out_path = out_dir / f"{stem}_g6a_frac_blocks.yaml"
    lines = ["g6a_frac_blocks:"]
    for fb in fracs:
        lines.append(f"  - id: {fb['id']}")
        lines.append(f"    bar_token_id: {fb['bar_token_id']}")
        lines.append(f"    numerator_token_ids: {fb['numerator_token_ids']}")
        lines.append(f"    denominator_token_ids: {fb['denominator_token_ids']}")
        lines.append(f"    child_token_ids: {fb['child_token_ids']}")
        lines.append("    bbox:")
        for k in ["l","t","r","b","w","h"]:
            lines.append(f"      {k}: {fb['bbox'][k]}")
        lines.append(f"    src_ids: {fb['src_ids']}")
        lines.append(f"    confidence: {fb['confidence']}")
        lines.append(f"    warnings: {fb['warnings']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx["g6a_frac_report_yml"] = str(out_path)


def debug_render_g6a_frac_boxes(ctx: Context, cfg: Dict[str, Any]) -> None:
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_g6a_frac_boxes: ctx['page'] missing")
    fracs = ctx.get("g6a_frac_blocks", None)
    if not isinstance(fracs, list):
        raise ValueError("debug_render_g6a_frac_boxes: ctx['g6a_frac_blocks'] missing")
    tokens: List[Token] = ctx.get("tokens_g5a") or ctx.get("tokens_h")
    tok_by_id = {int(getattr(t, 'token_id', 0) or i+1): t for i, t in enumerate(tokens)}
    H = int(page["H"]); W = int(page["W"])
    upscale = int(cfg.get("outputs", {}).get("debug_upscale", 2) or 2)
    if upscale < 1: upscale = 1
    rgb = np.full((H * upscale, W * upscale, 3), 255, dtype=np.uint8)
    # draw children lightly
    for fb in fracs:
        for tid in fb["child_token_ids"]:
            t = tok_by_id.get(int(tid))
            if t is None: continue
            x0,y0,x1,y1 = [int(v*upscale) for v in (t.l,t.t,t.r,t.b)]
            cv2.rectangle(rgb,(x0,y0),(x1,y1),(220,220,220),thickness=-1)
            cv2.rectangle(rgb,(x0,y0),(x1,y1),(180,180,180),thickness=max(1,upscale))
    # draw frac bboxes bold in red, bar token in blue
    for fb in fracs:
        b=fb['bbox']
        x0,y0,x1,y1 = [int(b[k]*upscale) for k in ('l','t','r','b')]
        cv2.rectangle(rgb,(x0,y0),(x1,y1),(0,0,255),thickness=max(2,2*upscale))
        bar = tok_by_id.get(int(fb['bar_token_id']))
        if bar is not None:
            bx0,by0,bx1,by1 = [int(v*upscale) for v in (bar.l,bar.t,bar.r,bar.b)]
            cv2.rectangle(rgb,(bx0,by0),(bx1,by1),(255,0,0),thickness=max(2,2*upscale))
        cv2.putText(rgb, f"frac {fb['id']}", (x0, max(20,y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*upscale, (0,0,0), 2, cv2.LINE_AA)
    out_dir = Path(cfg.get("outputs", {}).get("out_dir", "out_geo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page["path"]).stem
    out_path = out_dir / f"{stem}_g6a_frac_boxes.png"
    cv2.imwrite(str(out_path), rgb)
    ctx["debug_g6a_frac_boxes_png"] = str(out_path)
