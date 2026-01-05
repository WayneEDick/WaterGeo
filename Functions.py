"""
Functions.py — Water Geo (Rails-first)

This file implements the *geometry-only* passes that recover:
- CCs (connected components)
- Rails (aligned CC edges)
- Line bands (validated horizontal strips)
- Gutters/column boundaries (validated vertical corridors)
- Columns (derived regions between boundaries)

Key design principles:
- Born-digital PNG ⇒ axis-aligned rails (no skew/orientation model required).
- Column boundaries are detected; columns are what remain.
- Bad justification is handled by *acceptance predicates* (persistence + crossings + two-rail + width stability).

See:
- WATER GEO v2 design: column boundaries/columns and SPACE-vs-MER semantics.
- Rails.docx: explicit gutter acceptance/rejection predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np


import os
import cv2
# ============================================================
# CC Debug Rendering Constants (Grayscale)
# ============================================================

# Grayscale levels (0 = black, 255 = white)
GRAY_BACKGROUND = 255   # white
GRAY_LIGHT      = 210   # low density
GRAY_MEDIUM     = 150   # medium density
GRAY_DARK       = 90    # high density
GRAY_OVERLAP    = 30    # bbox overlap (>= 2)

# Density thresholds
DENSITY_LIGHT_MAX  = 0.20
DENSITY_MEDIUM_MAX = 0.45

# -----------------------------
# Types
# -----------------------------

BBox = Tuple[int, int, int, int]  # (top, left, height, width)


def bbox_right(b: BBox) -> int:
    t, l, h, w = b
    return l + w


def bbox_bottom(b: BBox) -> int:
    t, l, h, w = b
    return t + h


@dataclass(frozen=True)
class CC:
    id: int
    bbox: BBox
    area: int
    box_center: Tuple[float, float]  # (y, x)


@dataclass(frozen=True)
class Rail:
    center: int
    lo: int
    hi: int
    support: int
    members: Optional[List[int]] = None


@dataclass(frozen=True)
class LineBand:
    y_top: int
    y_bot: int
    mode_id: int
    support_cc_ids: List[int]
    score: float


@dataclass(frozen=True)
class GutterBand:
    x0: int
    x1: int
    support_ratio: float
    cross_rate: float
    two_rail_ok: bool
    width_iqr: float


@dataclass(frozen=True)
class Column:
    x0: int
    x1: int
    confidence: float


# Context is a dict that accumulates outputs of passes.
Context = Dict[str, Any]


# -----------------------------
# Utility: 1D rail clustering
# -----------------------------

def cluster_1d(values: List[int], eps: int) -> List[Rail]:
    """
    Cluster sorted 1D coordinates into rails using a simple sweep:
    - values within eps are grouped
    - rail center is the median of group
    """
    if not values:
        return []

    vals = sorted(values)
    rails: List[Rail] = []
    group: List[int] = [vals[0]]

    def flush(g: List[int]) -> None:
        g_sorted = sorted(g)
        center = g_sorted[len(g_sorted) // 2]
        rails.append(Rail(center=center, lo=g_sorted[0], hi=g_sorted[-1], support=len(g_sorted)))

    for v in vals[1:]:
        if abs(v - group[-1]) <= eps:
            group.append(v)
        else:
            flush(group)
            group = [v]
    flush(group)
    return rails


def prune_rails(rails: List[Rail], total: int, min_abs: int, min_frac: float) -> List[Rail]:
    thr = max(min_abs, int(round(min_frac * total)))
    return [r for r in rails if r.support >= thr]


# -----------------------------
# Passes
# -----------------------------

def to_gray_handle_alpha(img: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV-loaded image to grayscale.

    Handles:
      - Grayscale images (H, W)
      - BGR images (H, W, 3)
      - BGRA images (H, W, 4) by compositing alpha against white

    Assumes OpenCV channel order (BGR[A]).
    """
    NDIM_GRAY = 2
    NDIM_COLOR = 3

    CHANNELS_BGR = 3
    CHANNELS_BGRA = 4

    ALPHA_CH = 3
    COLOR_CH_SLICE = slice(0, 3)

    WHITE_U8 = 255

    if img.ndim == NDIM_GRAY:
        return img

    if img.ndim != NDIM_COLOR:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")

    if img.shape[2] == CHANNELS_BGR:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.shape[2] == CHANNELS_BGRA:
        bgr = img[:, :, COLOR_CH_SLICE]
        alpha = img[:, :, ALPHA_CH].astype(np.float32) / float(WHITE_U8)

        gray_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gray = gray_rgb * alpha + float(WHITE_U8) * (1.0 - alpha)
        return gray.astype(np.uint8)

    raise ValueError(f"Unsupported channel count: {img.shape[2]}")


def load_normalize(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G0: Load & Normalize

    Reads the PNG specified by cfg["inputs"]["png_path"] and stores:
      ctx["imageGray"] : 2D uint8 grayscale (H,W)
      ctx["page"]      : dict with basic page metadata (H,W,dpi,path)

    Notes:
      - Uses OpenCV (cv2) for robust Windows PNG IO.
      - No polarity assumptions are made here.
    """
    png_path = cfg.get("inputs", {}).get("png_path")
    if not png_path or not isinstance(png_path, str):
        raise ValueError('load_normalize: cfg["inputs"]["png_path"] must be a non-empty string')

    p = os.path.abspath(png_path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"load_normalize: cannot read PNG: {p}")

    # Ensure grayscale uint8
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
    """
    G1: Binarize

    Produces a stable ink mask from ctx["imageGray"].

    Outputs:
      ctx["mask255"]  : uint8 mask in {0,255} with ink=255
      ctx["mask01"]   : uint8 mask in {0,1}   with ink=1
      ctx["imageBin"] : uint8 image in {0,255} with ink=0, background=255 (legacy convention)

    Polarity:
      - If cfg["inputs"]["expected_polarity"] is "auto" (default), we choose the polarity
        that yields *less* ink (background should dominate for born-digital pages).
      - If "normal": treat dark pixels as ink (THRESH_BINARY_INV).
      - If "inverted": treat light pixels as ink (THRESH_BINARY).
    """
    g = ctx.get("imageGray")
    if not isinstance(g, np.ndarray):
        raise ValueError('binarize: requires ctx["imageGray"] (run G0_load_normalize first)')
    if g.ndim != 2 or g.dtype != np.uint8:
        raise ValueError('binarize: ctx["imageGray"] must be 2D uint8')

    expected = (cfg.get("inputs", {}).get("expected_polarity") or "auto").lower()

    # Otsu both ways
    _t1, bw_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # dark->ink=255
    _t2, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)          # light->ink=255

    def ink_frac(mask255: np.ndarray) -> float:
        return float((mask255 > 0).mean())

    if expected == "normal":
        mask255 = bw_inv
    elif expected == "inverted":
        mask255 = bw
    else:
        # auto: pick the one with less ink
        mask255 = bw_inv if ink_frac(bw_inv) <= ink_frac(bw) else bw

    mask01 = (mask255 > 0).astype(np.uint8)

    # Legacy convention used elsewhere in this repo: ink=0, background=255
    imageBin = (255 - mask255).astype(np.uint8)

    ctx["mask255"] = mask255
    ctx["mask01"] = mask01
    ctx["imageBin"] = imageBin

def connected_components(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G2: Connected Components
    Requires ctx["bin"] (uint8 image, 0 background, 255 foreground) from G1_binarize.
    Outputs:
      ctx["ccLabels"] : int32 label image
      ctx["ccCount"]  : number of labels (including background label 0)
      ctx["ccStats"]  : stats array [label, x, y, w, h, area]
      ctx["ccCentroids"] : centroids array
    """
    bin_img = ctx.get("mask255", None)
    if bin_img is None:
        raise ValueError('connected_components: ctx["bin"] is missing. Did G1_binarize run?')

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise ImportError(
            "connected_components: OpenCV (cv2) not available in this venv. "
            "Install with: pip install opencv-python"
        ) from e

    # Ensure foreground is 255 and background is 0
    # (OpenCV expects non-zero as foreground)
    if bin_img.dtype != "uint8":
        bin_img = bin_img.astype("uint8")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    ctx["ccCount"] = num_labels
    ctx["ccLabels"] = labels
    ctx["ccStats"] = stats
    ctx["ccCentroids"] = centroids
    ctx["ccs"] = []


def build_ccs(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G2b: Build CC objects from OpenCV stats/centroids.

    Requires:
      ctx["ccStats"], ctx["ccCentroids"] from connected_components()

    Produces:
      ctx["ccs"] : List[CC]
    """
    stats = ctx.get("ccStats", None)
    cents = ctx.get("ccCentroids", None)
    if stats is None or cents is None:
        raise ValueError("build_ccs: requires ccStats and ccCentroids in ctx (run connected_components first)")

    ccs: List[CC] = []

    # OpenCV stats rows: [x, y, w, h, area]; background label is 0 at index 0
    for label in range(1, int(stats.shape[0])):  # skip background
        x = int(stats[label, 0])
        y = int(stats[label, 1])
        w = int(stats[label, 2])
        h = int(stats[label, 3])
        area = int(stats[label, 4])

        min_area = int(cfg.get("params", {}).get("cc", {}).get("min_area_px", 1))
        if area < min_area:
            continue
        if w <= 0 or h <= 0:
            continue

        cx = float(cents[label, 0])
        cy = float(cents[label, 1])

        # CC.box_center is (y, x) by convention
        ccs.append(CC(
            id=int(label),
            bbox=(y, x, h, w),
            area=area,
            box_center=(cy, cx),
        ))

    ctx["ccs"] = ccs


def debug_render_cc_boxes(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G2c: Debug Render CC Boxes (rectangle-only)

    Creates a grayscale page artifact where each CC bounding box is filled with a
    discrete gray level determined by density = area / (h*w). Overlaps (bbox overlap)
    are forced to GRAY_OVERLAP.

    Requires:
      ctx["page"] : contains H, W, path
      ctx["ccs"]  : built by build_ccs()

    Writes:
      <out_dir>/<stem>_cc_boxes.png
    """
    page = ctx.get("page", None)
    if page is None:
        raise ValueError("debug_render_cc_boxes: ctx['page'] missing (run load_normalize first)")

    H = int(page["H"])
    W = int(page["W"])
    in_path = str(page.get("path", "page"))
    stem = Path(in_path).stem

    out_dir = Path(cfg["outputs"]["out_dir"])
    out_png = out_dir / f"{stem}_cc_boxes.png"

    ccs: List[CC] = ctx.get("ccs", [])
    # Always produce an image so you can tell the step ran.
    G = np.full((H, W), GRAY_BACKGROUND, dtype=np.uint8)
    K = np.zeros((H, W), dtype=np.uint8)

    for cc in ccs:
        t, l, h, w = cc.bbox
        if h <= 0 or w <= 0:
            continue

        density = float(cc.area) / float(h * w)

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

    cv2.imwrite(str(out_png), G)
    ctx["debug_cc_boxes_png"] = str(out_png)


def find_rails_and_lines(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R0: FindRailsAndLines (Water-Geo)

    Produces:
      ctx["modes"]      : list of dicts per size-mode (h_star, eps_rail, cc_ids)
      ctx["rails"]      : rails per mode (T,B,L,R)
      ctx["lineBands"]  : validated line bands (y_top,y_bot,mode,support_cc_ids)
      ctx["lineRails"]  : per line band: baseline/top + secondary rails (optional)
    """
    # NOTE: This is a structured skeleton. Fill the "TODO" bodies with your existing primitives.
    ccs: List[CC] = ctx["ccs"]
    if not ccs:
        ctx["modes"] = []
        ctx["rails"] = []
        ctx["lineBands"] = []
        ctx["lineRails"] = []
        return

    # --- TODO(1): select glyph-like CCs (exclude large blocks/rules/dust) ---
    glyph_ccs = ccs  # placeholder

    # --- TODO(2): estimate size modes from heights ---
    heights = np.array([cc.bbox[2] for cc in glyph_ccs], dtype=np.int32)
    # Placeholder: single mode = all
    modes = [{
        "mode_id": 0,
        "cc_ids": [cc.id for cc in glyph_ccs],
        "h_star": int(np.median(heights)) if len(heights) else 0,
    }]
    # eps_rail derived from h_star
    eps_min = cfg["params"]["rails"]["eps_frac_min"]
    eps_max = cfg["params"]["rails"]["eps_frac_max"]
    for m in modes:
        h_star = max(1, m["h_star"])
        eps = int(round(((eps_min + eps_max) / 2.0) * h_star))
        m["eps_rail"] = max(1, eps)

    ctx["modes"] = modes

    # --- rails per mode ---
    rails_out = []
    for m in modes:
        mode_ccs = [cc for cc in glyph_ccs if cc.id in set(m["cc_ids"])]
        total = len(mode_ccs)
        eps = m["eps_rail"]

        Tvals = [cc.bbox[0] for cc in mode_ccs]
        Bvals = [bbox_bottom(cc.bbox) for cc in mode_ccs]
        Lvals = [cc.bbox[1] for cc in mode_ccs]
        Rvals = [bbox_right(cc.bbox) for cc in mode_ccs]

        T = prune_rails(cluster_1d(Tvals, eps), total,
                        cfg["params"]["rails"]["min_support_abs"],
                        cfg["params"]["rails"]["min_support_frac"])
        B = prune_rails(cluster_1d(Bvals, eps), total,
                        cfg["params"]["rails"]["min_support_abs"],
                        cfg["params"]["rails"]["min_support_frac"])
        L = prune_rails(cluster_1d(Lvals, eps), total,
                        cfg["params"]["rails"]["min_support_abs"],
                        cfg["params"]["rails"]["min_support_frac"])
        R = prune_rails(cluster_1d(Rvals, eps), total,
                        cfg["params"]["rails"]["min_support_abs"],
                        cfg["params"]["rails"]["min_support_frac"])

        rails_out.append({"mode_id": m["mode_id"], "T": T, "B": B, "L": L, "R": R})

    ctx["rails"] = rails_out

    # --- TODO(3): propose & validate line bands from (T,B) rail pairs ---
    ctx["lineBands"] = []
    ctx["lineRails"] = []


def find_columns_from_lrails(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R1: FindColumnsFromLRails (Water-Geo)

    This is the place we bake in the *anti-bad-justification* logic:
    - Candidate gaps (Breuel-style thresholding)
    - Promote to global gutter bands by persistence across strips
    - Accept only if all acceptance predicates pass:
        C1 persistence, C2 low crossings, C3 two-rail boundary, C4 width stability

    Produces:
      ctx["gutters"] : List[GutterBand]
      ctx["columns"] : List[Column]
    """
    # This pass assumes you already have line strips/bands.
    line_bands: List[LineBand] = ctx.get("lineBands", [])
    ccs: List[CC] = ctx["ccs"]

    # If you haven't built line bands yet, you can use coarse "bands" as strips.
    # For now, treat as placeholder.
    strips = line_bands

    # --- TODO(1): per-strip candidate gaps (Stage A) ---
    # --- TODO(2): vote to global x-bands (Stage B) ---
    # --- TODO(3): apply acceptance predicates (Stage C) ---

    ctx["gutters"] = []
    ctx["columns"] = []


def assign_lines_to_columns(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R2: AssignLinesToColumns

    Assign each validated line band to a column by horizontal overlap / proximity to L/R rails.
    This tightens both:
      - column confidence
      - line-local column membership (used by downstream passes)
    """
    # Placeholder: depends on ctx["columns"] + ctx["lineBands"]
    ctx["lineColumnIds"] = []


# -----------------------------
# Placeholders for later phases
# -----------------------------

def glyph_clustering(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()


def gap_tokens_and_islands(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()


def mer_detection(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()


def tmi_detection(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()


def block_detection(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()


def build_geo_cil(ctx: Context, cfg: Dict[str, Any]) -> None:
    raise NotImplementedError()
