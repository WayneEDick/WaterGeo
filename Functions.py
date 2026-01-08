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
            cv2.line(overlay_rails, (x0, 0), (x0, H - 1), (255, 255, 255), rail_thickness)
            cv2.line(overlay_rails, (x1, 0), (x1, H - 1), (255, 255, 255), rail_thickness)

    if columns:
        for c in columns:
            x0 = max(0, min(W - 1, int(c.x0)))
            x1 = max(0, min(W - 1, int(c.x1)))
            cv2.line(overlay_rails, (x0, 0), (x0, H - 1), (255, 255, 255), 1)
            cv2.line(overlay_rails, (x1, 0), (x1, H - 1), (255, 255, 255), 1)

    out = cv2.addWeighted(overlay_cc, float(cc_alpha), base, 1.0 - float(cc_alpha), 0.0)
    out = cv2.addWeighted(overlay_rails, float(rail_alpha), out, 1.0 - float(rail_alpha), 0.0)
    cv2.imwrite(png_out_path, out)
    return png_out_path

def find_columns_from_lrails(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R1: FindColumnsFromLRails (Water-Geo) — LEFT-RAIL ANCHORED

    Core invariant:
      • Real vertical separators (margins, gutters) have a straight LEFT rail (anchor).
      • The opposing boundary may be ragged; we represent it by an envelope (quantile).
      • Rivers (justification artifacts) do not create a stable anchor rail.

    Produces:
      ctx["gutters"] : List[GutterBand] (corridors between envelope and anchor)
      ctx["columns"] : List[Column]     (complements of gutters)
      ctx["debugRailsPng"] (optional)   (thin black rails + bbox-only CCs)
    """
    ccs: List[CC] = ctx.get("ccs", [])
    line_bands: List[LineBand] = ctx.get("lineBands", [])

    page = ctx.get("page", {})
    W = int(page.get("W", 0))
    H = int(page.get("H", 0))

    if W <= 0 or H <= 0:
        raise ValueError("find_columns_from_lrails: missing page W/H in ctx['page']")

    if not ccs:
        ctx["gutters"] = []
        ctx["columns"] = [Column(0, W, 0.0)]
        return

    # Fallback: allow R1 to run before R0 produces validated line bands.
    if not line_bands:
        step = int(cfg.get("params", {}).get("gutters", {}).get("fallback_band_step_px", 32))
        if step <= 0:
            step = 32
        y = 0
        line_bands = []
        while y < H:
            y2 = min(H, y + step)
            line_bands.append(LineBand(y_top=y, y_bot=y2, mode_id=0, support_cc_ids=[], score=0.0))
            y = y2

    gcfg = cfg["params"]["gutters"]

    # Minimum corridor width (your "1.5–2em" intuition belongs here)
    min_width_px = float(gcfg.get("min_width_px", 18))

    # Voting + candidate generation (keep Breuel-ish to find “big gaps”, but acceptance is rail-based)
    breuel_multiplier = float(gcfg.get("breuel_multiplier", 1.5))
    gap_long_percentile = 100.0 * float(gcfg.get("gap_long_percentile", 0.85))
    edge_peak_neighborhood_px = int(gcfg.get("edge_peak_neighborhood_px", 7))

    # Acceptance
    support_ratio_min = float(gcfg.get("support_ratio_min", 0.35))
    cross_rate_max = float(gcfg.get("cross_rate_max", 0.20))

    # Anchor straightness threshold (in px). Keep conservative and simple.
    # You can later tie this to h_star, but this works now.
    anchor_std_max = float(gcfg.get("anchor_std_max_px", 8.0))

    # Envelope quantile: high quantile of ragged boundary (right edges) gives a stable frontier
    env_q = float(gcfg.get("envelope_quantile", 90.0))

    # Reduce noise at extreme edges (still useful), but do NOT use this to reject real margins.
    margin_px = int(0.02 * W)

    # ------------------------------------------------------------
    # 1) Accumulate corridor votes in X (evidence only)
    # ------------------------------------------------------------
    acc = np.zeros((W,), dtype=np.float32)

    # We also keep per-strip gap stats so we can compute g_long per strip.
    for band in line_bands:
        y0 = int(band.y_top)
        y1 = int(band.y_bot)
        sc = _strip_ccs(ccs, y0, y1)
        if len(sc) < 2:
            continue
        sc = sorted(sc, key=lambda c: (c.bbox[1], bbox_right(c.bbox)))

        gaps: List[int] = []
        spans: List[Tuple[int, int]] = []
        for i in range(len(sc) - 1):
            L = sc[i]
            R = sc[i + 1]
            gap = int(R.bbox[1] - bbox_right(L.bbox))
            if gap > 0:
                gaps.append(gap)
                spans.append((int(bbox_right(L.bbox)), int(R.bbox[1])))

        g_long = _percentile_pos(gaps, gap_long_percentile, default=float(max(int(min_width_px), 12)))

        for (gap, (xL, xR)) in zip(gaps, spans):
            # Candidate gating: only unusually large gaps become evidence.
            if gap < min_width_px and gap < breuel_multiplier * g_long:
                continue
            x0 = max(0, min(W, xL))
            x1 = max(0, min(W, xR))
            if x1 <= x0:
                continue
            # Ignore tiny slivers touching extreme edges in the accumulator (not a margin reject!)
            if x0 < margin_px or x1 > (W - margin_px):
                continue
            acc[x0:x1] += 1.0

    acc_s = _smooth_1d(acc, max(1, edge_peak_neighborhood_px))
    peak = float(acc_s.max()) if acc_s.size else 0.0

    if peak <= 0.0:
        # No candidate evidence -> no gutters
        ctx["gutters"] = []
        ctx["columns"] = [Column(0, W, 0.0)]
        ctx["debugGutters"] = {"note": "no peaks", "acc_peak": 0.0}
        # Still emit debug rails (margins only) if requested
        if cfg.get("outputs", {}).get("save_debug_images", False):
            in_png = str(page.get("path", ""))
            if in_png:
                out_dir = Path(cfg["outputs"]["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(in_png).stem
                out_png = str(out_dir / f"{stem}_R1_rails.png")
                _draw_debug_rails_png_black_bbox(
                    in_png, out_png, ccs=ccs, gutters=[], columns=[Column(0, W, 1.0)]
                )
                ctx["debugRailsPng"] = out_png
        return

    # ------------------------------------------------------------
    # 2) Candidate bands = runs above fraction-of-peak (evidence)
    # ------------------------------------------------------------
    peak_frac = float(gcfg.get("peak_frac", 0.30))
    above = acc_s >= (peak_frac * peak)
    runs = _runs_from_bool(above)

    # Expand around local maxima (keeps candidates fat enough)
    expand_frac = float(gcfg.get("expand_frac", 0.55))
    candidate_bands: List[Tuple[int, int]] = []
    for (a0, a1) in runs:
        L, R = _expand_run_to_local_threshold(acc_s, a0, a1, frac=expand_frac)
        if (R - L) >= 2:
            candidate_bands.append((int(L), int(R)))

    # ------------------------------------------------------------
    # 3) Accept candidates using LEFT-RAIL ANCHOR + ENVELOPE
    # ------------------------------------------------------------
    gutters_out: List[GutterBand] = []
    reject_log: List[Dict[str, Any]] = []

    # Helper to compute crossings for a corridor
    def corridor_cross_rate(x0: int, x1: int) -> float:
        cross_total = 0
        strip_count = 0
        for band in line_bands:
            y0 = int(band.y_top)
            y1 = int(band.y_bot)
            strip_count += 1
            sc = _strip_ccs(ccs, y0, y1)
            if not sc:
                continue
            cross_total += sum(1 for cc in sc if _cc_intersects_xband(cc, x0, x1))
        return (cross_total / float(strip_count)) if strip_count else 1e9

    # Main accept loop
    for (x0, x1) in candidate_bands:
        anchor_samples: List[float] = []     # straight LEFT rail samples (x of right-CC left edge)
        env_samples: List[float] = []        # ragged boundary samples (x of left-CC right edge)
        strip_count = 0

        for band in line_bands:
            y0 = int(band.y_top)
            y1 = int(band.y_bot)
            strip_count += 1
            sc = _strip_ccs(ccs, y0, y1)
            if len(sc) < 2:
                continue
            sc = sorted(sc, key=lambda c: (c.bbox[1], bbox_right(c.bbox)))

            # Choose the best local gap that overlaps the candidate evidence band,
            # then treat the RIGHT CC left edge as the anchor candidate.
            best = None
            for i in range(len(sc) - 1):
                L = sc[i]
                R = sc[i + 1]
                span0 = int(bbox_right(L.bbox))
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
                        "anchor": float(span1),      # straight left rail sample (right CC left edge)
                        "env": float(span0),         # opposing boundary sample (left CC right edge)
                        "gap_w": float(span1 - span0)
                    }

            if best is not None and best["gap_w"] >= min_width_px:
                anchor_samples.append(best["anchor"])
                env_samples.append(best["env"])

        if strip_count == 0:
            continue

        support_ratio = len(anchor_samples) / float(strip_count)

        # C1 persistence (enough strips vote)
        if support_ratio < support_ratio_min:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C1_persistence", "support_ratio": support_ratio})
            continue

        # Anchor straightness: rail must be straight (low std)
        a_std = float(np.std(np.asarray(anchor_samples, dtype=np.float32))) if anchor_samples else 1e9
        if a_std > anchor_std_max:
            reject_log.append({"x0": x0, "x1": x1, "reject": "A_anchor_not_straight", "anchor_std": a_std})
            continue

        # Compute final anchor and envelope rails
        anchor = float(np.median(np.asarray(anchor_samples, dtype=np.float32)))
        env = float(np.percentile(np.asarray(env_samples, dtype=np.float32), env_q)) if env_samples else (anchor - min_width_px)

        width = anchor - env

        if width < min_width_px:
            reject_log.append({"x0": x0, "x1": x1, "reject": "W_width_too_small", "width": width})
            continue

        # C2 low crossings: measure crossings on the FINAL corridor [env, anchor)
        xL = int(max(0, min(W, round(env))))
        xR = int(max(0, min(W, round(anchor))))
        if xR <= xL:
            reject_log.append({"x0": x0, "x1": x1, "reject": "W_bad_corridor_order"})
            continue

        cross_rate = corridor_cross_rate(xL, xR)
        if cross_rate > cross_rate_max:
            reject_log.append({"x0": x0, "x1": x1, "reject": "C2_cross_rate", "cross_rate": cross_rate})
            continue

        # Accept: store as GutterBand using the corridor boundaries (envelope, anchor)
        gutters_out.append(GutterBand(
            x0=int(xL),
            x1=int(xR),
            support_ratio=float(support_ratio),
            cross_rate=float(cross_rate),
            two_rail_ok=True,                 # in the *new* model, this is always true once accepted
            width_iqr=float(iqr([width])),    # keep field but store something deterministic
        ))

    # Sort and derive columns
    gutters_out = sorted(gutters_out, key=lambda g: (g.x0, g.x1))
    columns = _derive_columns_from_gutters(W, gutters_out)

    ctx["gutters"] = gutters_out
    ctx["columns"] = columns
    ctx["debugGutters"] = {
        "acc_peak": peak,
        "candidate_bands": candidate_bands,
        "rejections": reject_log,
        "support_ratio_min": support_ratio_min,
        "cross_rate_max": cross_rate_max,
        "min_width_px": min_width_px,
        "anchor_std_max": anchor_std_max,
        "envelope_quantile": env_q,
        "peak_frac": peak_frac,
        "expand_frac": expand_frac,
    }

    # ------------------------------------------------------------
    # Debug image: bbox-only CCs + THIN BLACK rails
    # ------------------------------------------------------------
    if cfg.get("outputs", {}).get("save_debug_images", False):
        in_png = str(page.get("path", ""))
        if in_png:
            out_dir = Path(cfg["outputs"]["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(in_png).stem
            out_png = str(out_dir / f"{stem}_R1_rails.png")
            _draw_debug_rails_png_black_bbox(
                in_png, out_png, ccs=ccs, gutters=gutters_out, columns=columns
            )
            ctx["debugRailsPng"] = out_png


def _draw_debug_rails_png_black_bbox(
    png_in_path: str,
    png_out_path: str,
    *,
    ccs: List[CC],
    gutters: List[GutterBand],
    columns: List[Column],
) -> str:
    """
    Debug artifact rules (Wayne request):
      - CCs shown as bounding boxes only (no glyph pixels, no fills)
      - rails thin BLACK lines
      - no inline gaps
    """
    img = cv2.imread(png_in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {png_in_path}")
    H, W = img.shape[:2]

    # Draw CC bounding boxes (thin gray)
    for cc in ccs:
        t, l, h, w = cc.bbox
        x0 = max(0, min(W - 1, int(l)))
        y0 = max(0, min(H - 1, int(t)))
        x1 = max(0, min(W - 1, int(l + w)))
        y1 = max(0, min(H - 1, int(t + h)))
        cv2.rectangle(img, (x0, y0), (x1, y1), (160, 160, 160), 1)

    # Draw gutter rails (thin black)
    for g in gutters:
        x0 = max(0, min(W - 1, int(g.x0)))
        x1 = max(0, min(W - 1, int(g.x1)))
        cv2.line(img, (x0, 0), (x0, H - 1), (0, 0, 0), 1)
        cv2.line(img, (x1, 0), (x1, H - 1), (0, 0, 0), 1)

    # Optional: column boundaries (thin black). Helpful to verify complement logic.
    for c in columns:
        x0 = max(0, min(W - 1, int(c.x0)))
        x1 = max(0, min(W - 1, int(c.x1)))
        cv2.line(img, (x0, 0), (x0, H - 1), (0, 0, 0), 1)
        cv2.line(img, (x1, 0), (x1, H - 1), (0, 0, 0), 1)

    cv2.imwrite(png_out_path, img)
    return png_out_path

# -----------------------------
# Placeholders for later phases
# -----------------------------

def assign_lines_to_columns(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    R2: AssignLinesToColumns
    Placeholder so Runner can proceed.
    Real implementation will attach each line band / line candidate to a column.
    """
    return


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
