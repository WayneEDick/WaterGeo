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

def assign_lines_to_columns(ctx, cfg):
    """
    R2 placeholder.
    After R1, ctx should have columns and (optionally) lineCandidates / rails.
    For now: no-op so the pipeline can proceed.
    """
    # If you already have line candidates, keep them untouched.
    # Later: attach each line to a column index using its x-center and the column zones.
    return


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


# -----------------------------
# R1 helpers: gutters/columns (G3) using existing LineBands as strips
# -----------------------------

def _cc_intersects_yband(cc: CC, y0: int, y1: int) -> bool:
    # half-open y-intervals
    t = cc.bbox[0]
    b = bbox_bottom(cc.bbox)
    return not (b <= y0 or t >= y1)


def _cc_intersects_xband(cc: CC, x0: int, x1: int) -> bool:
    l = cc.bbox[1]
    r = bbox_right(cc.bbox)
    return not (r <= x0 or l >= x1)


def _strip_ccs(ccs: List[CC], y0: int, y1: int) -> List[CC]:
    return [cc for cc in ccs if _cc_intersects_yband(cc, y0, y1)]


def _iqr(vals: List[float]) -> float:
    if not vals:
        return float("inf")
    a = np.asarray(vals, dtype=np.float32)
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    return q3 - q1


def _percentile_pos(vals: List[int], p: float, default: float) -> float:
    if not vals:
        return default
    a = np.asarray([v for v in vals if v > 0], dtype=np.float32)
    if a.size == 0:
        return default
    return float(np.percentile(a, p))


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), k, mode="same")


def _runs_from_bool(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i in range(mask.size):
        if bool(mask[i]) and not in_run:
            in_run = True
            start = i
        elif (not bool(mask[i])) and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, int(mask.size)))
    return runs


def _expand_run_to_local_threshold(acc_s: np.ndarray, a0: int, a1: int, *, frac: float) -> Tuple[int, int]:
    seg = acc_s[a0:a1]
    if seg.size == 0:
        return (a0, a1)
    p = int(a0 + int(np.argmax(seg)))
    peak = float(acc_s[p])
    if peak <= 0:
        return (a0, a1)
    thr = frac * peak
    L = p
    R = p + 1
    while L > 0 and float(acc_s[L - 1]) >= thr:
        L -= 1
    while R < acc_s.size and float(acc_s[R]) >= thr:
        R += 1
    return (int(L), int(R))


def _derive_columns_from_gutters(W: int, gutters: List[GutterBand]) -> List[Column]:
    intervals = sorted([(max(0, g.x0), min(W, g.x1)) for g in gutters], key=lambda t: (t[0], t[1]))
    merged: List[List[int]] = []
    for (a0, a1) in intervals:
        if a1 <= a0:
            continue
        if not merged:
            merged.append([a0, a1])
        else:
            b0, b1 = merged[-1]
            if a0 <= b1:
                merged[-1][1] = max(b1, a1)
            else:
                merged.append([a0, a1])

    cols: List[Column] = []
    cur = 0
    for (g0, g1) in merged:
        if g0 > cur:
            cols.append(Column(x0=int(cur), x1=int(g0), confidence=1.0))
        cur = max(cur, g1)
    if cur < W:
        cols.append(Column(x0=int(cur), x1=int(W), confidence=1.0))
    return cols


def draw_debug_rails_png(
    png_in_path: str,
    png_out_path: str,
    *,
    ccs: Optional[List[CC]] = None,
    gutters: Optional[List[GutterBand]] = None,
    columns: Optional[List[Column]] = None,
    draw_cc_boxes: bool = True,
    cc_alpha: float = 0.18,
    rail_alpha: float = 0.90,
    rail_thickness: int = 2,
    cc_thickness: int = 1,
) -> str:
    """
    Draw only:
      - gutter boundary rails (x0 and x1 for each gutter)
      - optional column boundary rails (x0/x1 per column)
      - optional faint CC rectangles
    Does NOT draw inline gaps.
    """
    img = cv2.imread(png_in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {png_in_path}")
    H, W = img.shape[:2]

    base = img.copy()
    overlay_cc = img.copy()
    overlay_rails = img.copy()

    if draw_cc_boxes and ccs:
        for cc in ccs:
            t, l, h, w = cc.bbox
            x0 = max(0, min(W - 1, int(l)))
            y0 = max(0, min(H - 1, int(t)))
            x1 = max(0, min(W - 1, int(l + w)))
            y1 = max(0, min(H - 1, int(t + h)))
            cv2.rectangle(overlay_cc, (x0, y0), (x1, y1), (255, 255, 255), cc_thickness)

    if gutters:
        for g in gutters:
            x0 = max(0, min(W - 1, int(g.x0)))
            x1 = max(0, min(W - 1, int(g.x1)))
            cv2.line(overlay_rails, (x0, 0), (x0, H - 1), (0, 0, 255), rail_thickness)
            cv2.line(overlay_rails, (x1, 0), (x1, H - 1), (0, 0, 255), rail_thickness)

    if columns:
        for c in columns:
            x0 = max(0, min(W - 1, int(c.x0)))
            x1 = max(0, min(W - 1, int(c.x1)))
            cv2.line(overlay_rails, (x0, 0), (x0, H - 1), (0, 0, 255), 1)
            cv2.line(overlay_rails, (x1, 0), (x1, H - 1), (0, 0, 255), 1)

    out = cv2.addWeighted(overlay_cc, float(cc_alpha), base, 1.0 - float(cc_alpha), 0.0)
    out = cv2.addWeighted(overlay_rails, float(rail_alpha), out, 1.0 - float(rail_alpha), 0.0)
    cv2.imwrite(png_out_path, out)
    return png_out_path

def find_columns_from_lrails(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R1: FindColumnsFromLRails (Water-Geo)

    This pass consumes the validated horizontal strips already produced in R0 (ctx["lineBands"]).
    It promotes long-gap evidence into *global* vertical gutter bands and accepts only if the
    anti-bad-justification predicates pass:

      C1 persistence        (support ratio across strips)
      C2 low crossings      (few CCs intersect the candidate corridor)
      C3 two-rail boundary  (both boundary rails are stable; reject single-rail ragged-right)
      C4 width stability    (IQR of corridor width is small)

    Produces:
      ctx["gutters"] : List[GutterBand]
      ctx["columns"] : List[Column]
      ctx["debugRailsPng"] (optional)
    """
    ccs: List[CC] = ctx.get("ccs", [])
    line_bands: List[LineBand] = ctx.get("lineBands", [])

    page = ctx.get("page", {})
    W = int(page.get("W", 0))
    H = int(page.get("H", 0))

    if not line_bands:
        # Fallback: allow R1 to run before R0 produces validated line bands.
        # We create coarse horizontal strips across the page so gutter detection can proceed
        # and emit a rails debug PNG.
        step = int(cfg.get("params", {}).get("gutters", {}).get("fallback_band_step_px", 32))
        if step <= 0:
            step = 32
        line_bands = []
        y = 0
        mode_id = 0
        while y < H:
            y2 = min(H, y + step)
            line_bands.append(LineBand(y_top=y, y_bot=y2, mode_id=mode_id, support_cc_ids=[], score=0.0))
            y = y2

    if W <= 0 or H <= 0:
        raise ValueError("find_columns_from_lrails: missing page W/H in ctx['page']")
    if not ccs:
        ctx["gutters"] = []
        ctx["columns"] = [Column(0, W, 0.0)] if W > 0 else []
        return

    gcfg = cfg["params"]["gutters"]
    breuel_multiplier = float(gcfg.get("breuel_multiplier", 1.5))
    gap_long_percentile = 100.0 * float(gcfg.get("gap_long_percentile", 0.85))
    support_ratio_min = float(gcfg.get("support_ratio_min", 0.35))
    cross_rate_max = float(gcfg.get("cross_rate_max", 0.20))
    width_iqr_max_px = float(gcfg.get("width_iqr_max_px", 12))
    edge_peak_neighborhood_px = int(gcfg.get("edge_peak_neighborhood_px", 6))

    # Rail-stability threshold: reuse rails eps range as a conservative bound
    # (later you may parameterize this directly in gutters block).
    rcfg = cfg["params"]["rails"]
    rail_std_max = float(0.5 * (rcfg.get("eps_frac_max", 0.12) + rcfg.get("eps_frac_min", 0.06)) * W)
    rail_std_max = max(6.0, rail_std_max)  # absolute floor for stability
    rail_support_min = max(4, int(0.15 * len(line_bands)))

    # Outer margin gating (reduces accumulator pollution on ragged-right pages)
    margin_px = int(0.04 * W)

    # ------------------------------------------------------------
    # 1) Accumulate long-gap corridor votes into a global x score
    # ------------------------------------------------------------
    acc = np.zeros((W,), dtype=np.float32)
    strip_meta: List[Dict[str, Any]] = []

    for band in line_bands:
        y0 = int(band.y_top)
        y1 = int(band.y_bot)
        strip_ccs = _strip_ccs(ccs, y0, y1)
        if len(strip_ccs) < 2:
            continue

        strip_ccs_sorted = sorted(strip_ccs, key=lambda c: (c.bbox[1], bbox_right(c.bbox)))

        gaps: List[int] = []
        spans: List[Tuple[int, int, int, int]] = []  # (xL, xR, left_cc_id, right_cc_id)
        for i in range(len(strip_ccs_sorted) - 1):
            L = strip_ccs_sorted[i]
            R = strip_ccs_sorted[i + 1]
            gap = int(R.bbox[1] - bbox_right(L.bbox))
            if gap > 0:
                gaps.append(gap)
                spans.append((bbox_right(L.bbox), int(R.bbox[1]), int(L.id), int(R.id)))

        g_long = _percentile_pos(gaps, gap_long_percentile, default=float(max(12, int(0.01 * W))))

        for (gap, (xL, xR, left_id, right_id)) in zip(gaps, spans):
            if gap < breuel_multiplier * g_long:
                continue
            x0 = max(0, min(W - 1, int(xL)))
            x1 = max(0, min(W, int(xR)))
            if x1 <= x0:
                continue
            # early reject: outer margin zones
            if x0 <= margin_px or x1 >= (W - margin_px):
                continue
            acc[x0:x1] += 1.0
            strip_meta.append({
                "y0": y0, "y1": y1,
                "x0": x0, "x1": x1,
                "gap": float(gap),
                "g_long": float(g_long),
                "left_id": left_id,
                "right_id": right_id,
            })

    acc_s = _smooth_1d(acc, max(1, edge_peak_neighborhood_px))
    peak = float(acc_s.max()) if acc_s.size else 0.0
    if peak <= 0.0:
        ctx["gutters"] = []
        ctx["columns"] = [Column(0, W, 0.0)]
        ctx["debugGutters"] = {"acc": acc.tolist(), "acc_s": acc_s.tolist(), "note": "no peaks"}
        return

    # ------------------------------------------------------------
    # 2) Candidate bands = runs above fraction-of-peak, then expand
    # ------------------------------------------------------------
    peak_frac = 0.30
    expand_frac = 0.55
    above = acc_s >= (peak_frac * peak)
    runs = _runs_from_bool(above)

    candidate_bands: List[Tuple[int, int]] = []
    for (a0, a1) in runs:
        L, R = _expand_run_to_local_threshold(acc_s, a0, a1, frac=expand_frac)
        if (R - L) >= 2:
            candidate_bands.append((L, R))

    # ------------------------------------------------------------
    # 3) Accept/reject each candidate band using C1..C4 (+ single-rail reject)
    # ------------------------------------------------------------
    gutters: List[GutterBand] = []
    reject_log: List[Dict[str, Any]] = []

    for (x0, x1) in candidate_bands:
        support = 0
        strip_count = 0
        cross_total = 0

        widths: List[float] = []
        left_rail_samples: List[float] = []
        right_rail_samples: List[float] = []

        for band in line_bands:
            y0 = int(band.y_top)
            y1 = int(band.y_bot)
            strip_count += 1
            strip_ccs = _strip_ccs(ccs, y0, y1)
            if len(strip_ccs) < 2:
                continue
            sc = sorted(strip_ccs, key=lambda c: (c.bbox[1], bbox_right(c.bbox)))

            # crossings: number of CCs intersecting the corridor in this strip
            cross_here = sum(1 for cc in sc if _cc_intersects_xband(cc, x0, x1))
            cross_total += cross_here

            # find best local overlap gap to sample rails and width
            best = None
            for i in range(len(sc) - 1):
                L = sc[i]
                R = sc[i + 1]
                span0 = bbox_right(L.bbox)
                span1 = int(R.bbox[1])
                if span1 <= span0:
                    continue
                ov0 = max(span0, x0)
                ov1 = min(span1, x1)
                if ov1 <= ov0:
                    continue
                overlap_w = ov1 - ov0
                if (best is None) or (overlap_w > best["overlap_w"]):
                    best = {
                        "overlap_w": overlap_w,
                        "gap_w": float(span1 - span0),
                        "left_edge": float(span0),
                        "right_edge": float(span1),
                    }

            if best is not None:
                support += 1
                widths.append(best["gap_w"])
                left_rail_samples.append(best["left_edge"])
                right_rail_samples.append(best["right_edge"])

        if strip_count == 0:
            continue

        support_ratio = support / float(strip_count)
        cross_rate = cross_total / float(strip_count)

        # C1 persistence
        if support_ratio < support_ratio_min:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C1_persistence", "support_ratio": support_ratio})
            continue

        # C2 low crossings
        if cross_rate > cross_rate_max:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C2_cross_rate", "cross_rate": cross_rate})
            continue

        # C3 two-rail boundary: both sides stable, reject single-stable-rail (ragged-right giveaway)
        if len(left_rail_samples) < rail_support_min or len(right_rail_samples) < rail_support_min:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C3_rail_support", "rail_support": len(left_rail_samples)})
            continue

        left_std = float(np.std(np.asarray(left_rail_samples, dtype=np.float32)))
        right_std = float(np.std(np.asarray(right_rail_samples, dtype=np.float32)))

        left_stable = left_std <= rail_std_max
        right_stable = right_std <= rail_std_max
        if left_stable ^ right_stable:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C3_single_stable_rail", "left_std": left_std, "right_std": right_std})
            continue
        if not (left_stable and right_stable):
            reject_log.append({"x0": x0, "x1": x1, "reject": "C3_rail_unstable", "left_std": left_std, "right_std": right_std})
            continue

        # C4 width stability (IQR)
        width_iqr = _iqr(widths)
        if width_iqr > width_iqr_max_px:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C4_width_iqr", "width_iqr": width_iqr})
            continue

        gutters.append(GutterBand(
            x0=int(x0),
            x1=int(x1),
            support_ratio=float(support_ratio),
            cross_rate=float(cross_rate),
            two_rail_ok=True,
            width_iqr=float(width_iqr),
        ))

    # sort gutters left-to-right, and derive columns as complements
    gutters = sorted(gutters, key=lambda g: (g.x0, g.x1))
    columns = _derive_columns_from_gutters(W, gutters)

    ctx["gutters"] = gutters
    ctx["columns"] = columns
    ctx["debugGutters"] = {
        "acc_peak": peak,
        "candidate_bands": candidate_bands,
        "rejections": reject_log,
        "rail_std_max": rail_std_max,
        "margin_px": margin_px,
    }

    # ------------------------------------------------------------
    # Optional: rail debug image (rails + faint CC boxes; no inline gaps)
    # ------------------------------------------------------------
    if cfg.get("outputs", {}).get("save_debug_images", False):
        in_png = str(page.get("path", ""))
        if in_png:
            out_dir = Path(cfg["outputs"]["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(in_png).stem
            out_png = str(out_dir / f"{stem}_R1_rails.png")
            draw_debug_rails_png(
                in_png,
                out_png,
                ccs=ccs,
                gutters=gutters,
                columns=columns,
                draw_cc_boxes=True,
            )
            ctx["debugRailsPng"] = out_png

    # R2 consumes columns + lineBands; leave placeholder output intact for now
    ctx.setdefault("lineColumnIds", [])

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
