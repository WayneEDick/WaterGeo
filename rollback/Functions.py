# Functions_02_000.py — Water Geo (PNG-first) with CC-first column-aware structure
# New early stages:
#   extract_ccs -> estimate_text_scale_from_ccs -> detect_graphics_blocks_from_ccs
#   -> segment_columns_from_ccs -> assign_ccs_to_columns -> segment_lines_per_column_from_ccs
#
# Downstream stages reuse your glyph merge + space inference + prose mask + islandify.

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Set

import os
import numpy as np
import cv2

REGISTRY: Dict[str, Any] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def _ctx_map(ctx: Any) -> Dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return ctx
    cache = getattr(ctx, "cache", None)
    if isinstance(cache, dict):
        return cache
    return {}


def _ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _bbox_from_cc(cc: Tuple[int, int, int, int, int]) -> Tuple[float, float, float, float]:
    x, y, w, h, _a = cc
    return (float(x), float(y), float(x + w), float(y + h))


def _bbox(tok: Dict[str, Any]) -> Tuple[float, float, float, float]:
    b = tok.get("bbox")
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError("token missing bbox")
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _h_bbox(b: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(b[3]) - float(b[1]))


def _w_bbox(b: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(b[2]) - float(b[0]))


def _touching_bbox(a: Tuple[float, float, float, float],
                   b: Tuple[float, float, float, float],
                   pad: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def _center(cc_or_bbox) -> Tuple[float, float]:
    if isinstance(cc_or_bbox, tuple) and len(cc_or_bbox) == 5:
        x, y, w, h, _a = cc_or_bbox
        return (float(x) + 0.5 * float(w), float(y) + 0.5 * float(h))
    if isinstance(cc_or_bbox, tuple) and len(cc_or_bbox) == 4:
        x0, y0, x1, y1 = cc_or_bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    raise ValueError("bad bbox/cc")


# -----------------------
# Existing core steps
# -----------------------

@register("load_image")
def load_image(path: str, __context: Any = None, **kwargs) -> np.ndarray:
    if not path:
        raise ValueError("load_image: missing path")
    p = os.path.abspath(path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"load_image: cannot read: {p}")
    return _ensure_gray_u8(img)


@register("binarize_png")
def binarize_png(__context: Any = None,
                 method: str = "otsu",
                 invert: bool = False,
                 blur_ksize: int = 0,
                 threshold: int = 200,
                 **kwargs) -> np.ndarray:
    ctx = _ctx_map(__context)
    img = ctx.get("load_image")
    if not isinstance(img, np.ndarray):
        raise ValueError("binarize_png: requires context['load_image']")
    g = _ensure_gray_u8(img)

    if blur_ksize and blur_ksize > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        g = cv2.GaussianBlur(g, (k, k), 0)

    if method.lower() == "otsu":
        _t, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method.lower() == "fixed":
        t = int(threshold)
        _t, bw = cv2.threshold(g, t, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"binarize_png: unknown method '{method}'")

    if invert:
        bw = 255 - bw

    return (bw > 0).astype(np.uint8) * 255


@register("extract_ccs")
def extract_ccs(__context: Any = None,
                binary: Optional[np.ndarray] = None,
                min_area: int = 5,
                connectivity: int = 8,
                **kwargs) -> List[Tuple[int, int, int, int, int]]:
    ctx = _ctx_map(__context)
    mask = binary if isinstance(binary, np.ndarray) else ctx.get("binarize_png")
    if not isinstance(mask, np.ndarray):
        raise ValueError("extract_ccs: need binary mask (arg 'binary' or context['binarize_png'])")

    m = (mask > 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=int(connectivity))
    out: List[Tuple[int, int, int, int, int]] = []
    for k in range(1, num):
        x, y, w, h, area = map(int, stats[k])
        if area >= int(min_area):
            out.append((x, y, w, h, area))
    return out


# -----------------------
# New: robust text scale from CCs
# -----------------------

@register("estimate_text_scale_from_ccs")
def estimate_text_scale_from_ccs(__context: Any = None,
                                 trim_top_frac: float = 0.10,
                                 min_ccs: int = 50,
                                 fallback_xheight: float = 16.0,
                                 **kwargs) -> Dict[str, float]:
    ctx = _ctx_map(__context)
    ccs = ctx.get("extract_ccs", [])
    if not isinstance(ccs, list) or not ccs:
        return {"xheight": float(fallback_xheight), "med_h": float(fallback_xheight), "med_area": float(fallback_xheight * fallback_xheight)}

    hs = np.array([float(h) for (_x, _y, _w, h, _a) in ccs], dtype=np.float32)
    areas = np.array([float(a) for (_x, _y, _w, _h, a) in ccs], dtype=np.float32)

    if len(hs) < int(min_ccs):
        med_h = float(np.median(hs))
        med_area = float(np.median(areas))
        xh = max(8.0, min(64.0, med_h))
        return {"xheight": float(xh), "med_h": float(med_h), "med_area": float(med_area)}

    # Trim the top fraction to reduce influence of big blocks.
    q = float(np.quantile(hs, 1.0 - float(trim_top_frac)))
    hs2 = hs[hs <= q]
    areas2 = areas[hs <= q]
    if len(hs2) < 10:
        hs2 = hs
        areas2 = areas

    med_h = float(np.median(hs2))
    med_area = float(np.median(areas2))
    xh = float(max(8.0, min(64.0, med_h)))
    return {"xheight": xh, "med_h": float(med_h), "med_area": float(med_area)}


# -----------------------
# New: detect "graphics-like" blocks (big obstacles)
# -----------------------

@register("detect_graphics_blocks_from_ccs")
def detect_graphics_blocks_from_ccs(__context: Any = None,
                                    h_factor: float = 6.0,
                                    w_factor: float = 20.0,
                                    area_factor: float = 200.0,
                                    pad_px: float = 2.0,
                                    **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    ccs = ctx.get("extract_ccs", [])
    if not isinstance(ccs, list) or not ccs:
        return {"graphics_blocks": []}

    scale = ctx.get("estimate_text_scale_from_ccs", {})
    med_h = float(scale.get("med_h", 16.0))
    med_area = float(scale.get("med_area", 256.0))

    big_idxs = []
    for i, (x, y, w, h, area) in enumerate(ccs):
        if float(h) >= float(h_factor) * med_h:
            big_idxs.append(i); continue
        if float(w) >= float(w_factor) * med_h:
            big_idxs.append(i); continue
        if float(area) >= float(area_factor) * med_area:
            big_idxs.append(i); continue

    if not big_idxs:
        return {"graphics_blocks": []}

    # Union-find merge big CCs that touch (to form blocks)
    parent = {i: i for i in big_idxs}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bboxes = {i: _bbox_from_cc(ccs[i]) for i in big_idxs}

    for ii in range(len(big_idxs)):
        i = big_idxs[ii]
        bi = bboxes[i]
        for jj in range(ii + 1, len(big_idxs)):
            j = big_idxs[jj]
            if _touching_bbox(bi, bboxes[j], pad=float(pad_px)):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in big_idxs:
        r = find(i)
        groups.setdefault(r, []).append(i)

    blocks = []
    for g in groups.values():
        xs0, ys0, xs1, ys1 = [], [], [], []
        for idx in g:
            x0, y0, x1, y1 = bboxes[idx]
            xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
        blocks.append({
            "bbox": (float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))),
            "members": int(len(g)),
        })

    # Sort blocks top-to-bottom then left-to-right
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return {"graphics_blocks": blocks}


# -----------------------
# New: columns from CCs (gutter search)
# -----------------------

@register("segment_columns_from_ccs")
def segment_columns_from_ccs(__context: Any = None,
                             bins: int = 512,
                             gutter_max_frac: float = 0.20,
                             gutter_min_px: int = 12,
                             prefer_mid_frac: float = 0.35,
                             ignore_graphics: bool = True,
                             **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    img = ctx.get("load_image")
    ccs = ctx.get("extract_ccs", [])
    if not isinstance(img, np.ndarray) or not isinstance(ccs, list):
        raise ValueError("segment_columns_from_ccs: need context['load_image'] and context['extract_ccs']")

    H, W = img.shape[:2]

    graphics = ctx.get("detect_graphics_blocks_from_ccs", {}).get("graphics_blocks", [])
    gboxes = [tuple(b["bbox"]) for b in graphics] if (ignore_graphics and isinstance(graphics, list)) else []

    # Density histogram across x
    hist = np.zeros(int(bins), dtype=np.float32)
    bw = float(W) / float(bins)

    def is_in_graphics(ccbbox) -> bool:
        if not gboxes:
            return False
        for gb in gboxes:
            if _touching_bbox(ccbbox, gb, pad=0.0):
                return True
        return False

    for cc in ccs:
        bb = _bbox_from_cc(cc)
        if ignore_graphics and is_in_graphics(bb):
            continue
        x0, y0, x1, y1 = bb
        area = max(1.0, (x1 - x0) * (y1 - y0))
        b0 = int(max(0, min(bins - 1, np.floor(x0 / bw))))
        b1 = int(max(0, min(bins - 1, np.floor((x1 - 1e-6) / bw))))
        if b1 < b0:
            b1 = b0
        hist[b0:b1 + 1] += float(area)

    # Normalize
    if hist.max() > 0:
        hist = hist / hist.max()

    # Candidate gutter = long run of low density near middle
    thr = 0.03  # conservative: "almost empty"
    low = hist <= thr

    # Find runs
    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(low.tolist()):
        if v and not in_run:
            in_run = True
            s = i
        elif not v and in_run:
            runs.append((s, i - 1))
            in_run = False
    if in_run:
        runs.append((s, len(low) - 1))

    # Convert runs to pixel gutters and score them
    max_w_px = float(gutter_max_frac) * float(W)
    mid = 0.5 * float(W)
    best = None
    best_score = None

    for (a, b) in runs:
        x0 = a * bw
        x1 = (b + 1) * bw
        wpx = x1 - x0
        if wpx < float(gutter_min_px) or wpx > max_w_px:
            continue
        # Prefer gutters whose center lies near the middle
        c = 0.5 * (x0 + x1)
        dist = abs(c - mid)
        score = dist / max(1.0, (prefer_mid_frac * W))
        # smaller score is better
        if best is None or score < best_score:
            best = (float(x0), float(x1))
            best_score = score

    if best is None:
        return {
            "page": {"W": int(W), "H": int(H)},
            "columns": [{"col_id": 0, "x0": 0.0, "x1": float(W)}],
            "gutters": [],
        }

    gx0, gx1 = best
    columns = [
        {"col_id": 0, "x0": 0.0, "x1": float(gx0)},
        {"col_id": 1, "x0": float(gx1), "x1": float(W)},
    ]
    return {
        "page": {"W": int(W), "H": int(H)},
        "columns": columns,
        "gutters": [{"x0": float(gx0), "x1": float(gx1)}],
    }


@register("assign_ccs_to_columns")
def assign_ccs_to_columns(__context: Any = None,
                          keep_spanning: bool = True,
                          spanning_gutter_overlap_frac: float = 0.35,
                          **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    ccs = ctx.get("extract_ccs", [])
    cols = ctx.get("segment_columns_from_ccs", {}).get("columns", [])
    gutters = ctx.get("segment_columns_from_ccs", {}).get("gutters", [])
    if not isinstance(ccs, list) or not isinstance(cols, list):
        raise ValueError("assign_ccs_to_columns: need extract_ccs and segment_columns_from_ccs")

    gutter = None
    if gutters and isinstance(gutters, list) and len(gutters) >= 1:
        g = gutters[0]
        gutter = (float(g["x0"]), float(g["x1"]))

    out = []
    for cc in ccs:
        bb = _bbox_from_cc(cc)
        cx, cy = _center(bb)
        col_id = 0
        # choose column by center x
        for c in cols:
            if float(c["x0"]) <= cx < float(c["x1"]):
                col_id = int(c["col_id"])
                break

        spanning = False
        if gutter is not None and keep_spanning:
            gx0, gx1 = gutter
            x0, y0, x1, y1 = bb
            overlap = max(0.0, min(x1, gx1) - max(x0, gx0))
            gw = max(1.0, gx1 - gx0)
            if overlap >= float(spanning_gutter_overlap_frac) * gw:
                spanning = True

        out.append({
            "cc": cc,
            "bbox": bb,
            "col_id": int(col_id),
            "spanning": bool(spanning),
        })

    return {"ccs": out}


# -----------------------
# New: line segmentation per column from CCs
# -----------------------

@register("segment_lines_per_column_from_ccs")
def segment_lines_per_column_from_ccs(__context: Any = None,
                                      y_gap_factor: float = 0.80,
                                      min_line_ccs: int = 8,
                                      ignore_graphics: bool = True,
                                      **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    img = ctx.get("load_image")
    W = int(img.shape[1]) if isinstance(img, np.ndarray) else 0

    ccinfo = ctx.get("assign_ccs_to_columns", {}).get("ccs", [])
    cols = ctx.get("segment_columns_from_ccs", {}).get("columns", [])
    if not isinstance(ccinfo, list) or not isinstance(cols, list):
        raise ValueError("segment_lines_per_column_from_ccs: need assign_ccs_to_columns and segment_columns_from_ccs")

    scale = ctx.get("estimate_text_scale_from_ccs", {})
    med_h = float(scale.get("med_h", 16.0))
    gap = max(1.0, float(y_gap_factor) * med_h)

    graphics = ctx.get("detect_graphics_blocks_from_ccs", {}).get("graphics_blocks", [])
    gboxes = [tuple(b["bbox"]) for b in graphics] if (ignore_graphics and isinstance(graphics, list)) else []

    def is_in_graphics(bbx) -> bool:
        if not gboxes:
            return False
        for gb in gboxes:
            if _touching_bbox(bbx, gb, pad=0.0):
                return True
        return False

    # group CCs by column
    by_col: Dict[int, List[Dict[str, Any]]] = {}
    for r in ccinfo:
        bb = tuple(r["bbox"])
        if ignore_graphics and is_in_graphics(bb):
            continue
        if r.get("spanning"):
            # spanning handled separately (not used for per-column line bands)
            continue
        by_col.setdefault(int(r.get("col_id", 0)), []).append(r)

    per_col_lines = {}
    all_lines = []
    global_line_id = 0

    for c in cols:
        cid = int(c["col_id"])
        items = by_col.get(cid, [])
        if len(items) < int(min_line_ccs):
            per_col_lines[str(cid)] = []
            continue

        # cluster by cy
        centers = []
        for r in items:
            cx, cy = _center(tuple(r["bbox"]))
            centers.append((cy, r))
        centers.sort(key=lambda t: t[0])

        clusters: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        last = None
        for cy, r in centers:
            if last is None or abs(float(cy) - float(last)) <= gap:
                cur.append(r)
            else:
                clusters.append(cur)
                cur = [r]
            last = cy
        if cur:
            clusters.append(cur)

        lines = []
        for cl in clusters:
            ys0, ys1 = [], []
            for r in cl:
                x0, y0, x1, y1 = tuple(r["bbox"])
                ys0.append(float(y0))
                ys1.append(float(y1))
            y0b = float(min(ys0))
            y1b = float(max(ys1))
            line = {
                "line_id": int(global_line_id),
                "col_id": int(cid),
                "y0": y0b,
                "y1": y1b,
            }
            global_line_id += 1
            lines.append(line)
            all_lines.append(line)

        per_col_lines[str(cid)] = lines

    # Sort reading order: column 0 top->bottom then column 1 top->bottom
    all_lines.sort(key=lambda d: (int(d["col_id"]), float(d["y0"])))
    return {"per_col": per_col_lines, "lines": all_lines}


# -----------------------
# Your existing glyph merge + spacing + prose + islandify
# Modified only where it needs line_id support
# -----------------------

def _assign_line_id_from_lines(lines: List[Dict[str, Any]], col_id: int, y: float) -> int:
    """Assign a global line_id by nearest y-band within a column."""
    cand = [ln for ln in lines if int(ln.get("col_id", -1)) == int(col_id)]
    if not cand:
        return 0
    for ln in cand:
        if float(ln["y0"]) <= y <= float(ln["y1"]):
            return int(ln["line_id"])
    mids = [0.5 * (float(ln["y0"]) + float(ln["y1"])) for ln in cand]
    return int(np.argmin([abs(y - m) for m in mids]))


@register("merge_ccs_to_glyphs")
def merge_ccs_to_glyphs(__context: Any = None,
                        ccs: Optional[List[Tuple[int, int, int, int, int]]] = None,
                        pad_px: float = 1.0,
                        **kwargs) -> List[Dict[str, Any]]:
    ctx = _ctx_map(__context)

    # Prefer column-assigned CCs if available
    ccinfo = ctx.get("assign_ccs_to_columns", {}).get("ccs", None)
    lines_all = ctx.get("segment_lines_per_column_from_ccs", {}).get("lines", [])

    scale = ctx.get("estimate_text_scale_from_ccs", {})
    xh = float(scale.get("xheight", 16.0))

    recs = []
    if isinstance(ccinfo, list) and ccinfo:
        for r in ccinfo:
            cc = tuple(r["cc"])
            bb = tuple(r["bbox"])
            col_id = int(r.get("col_id", 0))
            x0, y0, x1, y1 = bb
            cy = 0.5 * (y0 + y1)
            lid = _assign_line_id_from_lines(lines_all, col_id, cy)
            recs.append({
                "cc": cc,
                "bbox": bb,
                "line_id": int(lid),
                "col_id": int(col_id),
                "area": int(cc[4]),
            })
    else:
        ccs_in = ccs if isinstance(ccs, list) else ctx.get("extract_ccs", [])
        if not isinstance(ccs_in, list) or not ccs_in:
            return []
        # fallback: single column/line id=0
        for cc in ccs_in:
            bb = _bbox_from_cc(cc)
            recs.append({"cc": cc, "bbox": bb, "line_id": 0, "col_id": 0, "area": int(cc[4])})

    # group by line_id
    by_line: Dict[int, List[int]] = {}
    for i, r in enumerate(recs):
        by_line.setdefault(int(r["line_id"]), []).append(i)

    glyphs: List[Dict[str, Any]] = []

    for lid, idxs in by_line.items():
        n = len(idxs)
        parent = {i: i for i in idxs}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        v_close = 0.60 * xh
        h_close = 0.35 * xh
        short_h = 0.35 * xh

        for ii in range(n):
            i = idxs[ii]
            bi = recs[i]["bbox"]
            ix0, iy0, ix1, iy1 = bi
            wi = max(1.0, ix1 - ix0)
            hi = max(1.0, iy1 - iy0)
            for jj in range(ii + 1, n):
                j = idxs[jj]
                bj = recs[j]["bbox"]
                jx0, jy0, jx1, jy1 = bj
                wj = max(1.0, jx1 - jx0)
                hj = max(1.0, jy1 - jy0)

                if _touching_bbox(bi, bj, pad=float(pad_px)):
                    union(i, j)
                    continue

                ox = max(0.0, min(ix1, jx1) - max(ix0, jx0))
                oy = max(0.0, min(iy1, jy1) - max(iy0, jy0))

                # dot-on-stem
                x_overlap_ratio = ox / min(wi, wj)
                if x_overlap_ratio >= 0.45:
                    vgap = 0.0
                    if iy1 < jy0:
                        vgap = jy0 - iy1
                    elif jy1 < iy0:
                        vgap = iy0 - jy1
                    if vgap <= v_close:
                        union(i, j)
                        continue

                # equals bars
                y_overlap_ratio = oy / min(hi, hj)
                if y_overlap_ratio >= 0.35 and (hi <= short_h and hj <= short_h):
                    hgap = 0.0
                    if ix1 < jx0:
                        hgap = jx0 - ix1
                    elif jx1 < ix0:
                        hgap = ix0 - jx1
                    if hgap <= h_close:
                        union(i, j)
                        continue

        groups: Dict[int, List[int]] = {}
        for i in idxs:
            r = find(i)
            groups.setdefault(r, []).append(i)

        for gidxs in groups.values():
            xs0 = []; ys0 = []; xs1 = []; ys1 = []; area = 0
            col_id = int(recs[gidxs[0]].get("col_id", 0))
            for i in gidxs:
                x0, y0, x1, y1 = recs[i]["bbox"]
                xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
                area += int(recs[i]["area"])
            glyphs.append({
                "kind": "GLYPH",
                "bbox": (float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))),
                "line_id": int(lid),
                "col_id": int(col_id),
                "area": int(area),
                "members": int(len(gidxs)),
            })

    glyphs.sort(key=lambda t: (int(t.get("col_id", 0)), int(t.get("line_id", 0)), float(_bbox(t)[0])))
    return glyphs


def _kmeans2_1d(values: np.ndarray, iters: int = 25) -> Tuple[float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return (vmin, vmax)
    c0, c1 = vmin, vmax
    for _ in range(int(iters)):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        a = values[d0 <= d1]
        b = values[d0 > d1]
        if len(a) > 0:
            c0 = float(a.mean())
        if len(b) > 0:
            c1 = float(b.mean())
    return (c0, c1) if c0 <= c1 else (c1, c0)


@register("infer_space_threshold_per_line")
def infer_space_threshold_per_line(__context: Any = None,
                                   glyphs: Optional[List[Dict[str, Any]]] = None,
                                   min_gaps_for_kmeans: int = 4,
                                   fallback_cutoff_factor: float = 0.60,
                                   **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    glyphs_in = glyphs if isinstance(glyphs, list) else ctx.get("merge_ccs_to_glyphs", [])
    scale = ctx.get("estimate_text_scale_from_ccs", {})
    xh = float(scale.get("xheight", 16.0))

    if not isinstance(glyphs_in, list) or not glyphs_in:
        return {"per_line": {}}

    by_line: Dict[int, List[Dict[str, Any]]] = {}
    for g in glyphs_in:
        if g.get("kind") != "GLYPH":
            continue
        by_line.setdefault(int(g.get("line_id", 0)), []).append(g)
    for lid in by_line:
        by_line[lid].sort(key=lambda t: _bbox(t)[0])

    out: Dict[str, Any] = {"per_line": {}}

    for lid, gl in by_line.items():
        gaps: List[float] = []
        for i in range(len(gl) - 1):
            x0, y0, x1, y1 = _bbox(gl[i])
            nx0, ny0, nx1, ny1 = _bbox(gl[i + 1])
            gaps.append(float(max(0.0, nx0 - x1)))

        if len(gaps) >= int(min_gaps_for_kmeans) and (max(gaps) > min(gaps)):
            arr = np.array(gaps, dtype=np.float32)
            c0, c1 = _kmeans2_1d(arr)
            cutoff = 0.5 * (c0 + c1)
        else:
            cutoff = float(fallback_cutoff_factor) * xh

        labeled = []
        for i, gap_px in enumerate(gaps):
            label = "SPACE" if float(gap_px) >= float(cutoff) else "INTRA"
            labeled.append({"i": int(i), "gap_px": float(gap_px), "label": label})

        out["per_line"][str(int(lid))] = {"cutoff_px": float(cutoff), "gaps": labeled}

    return out


@register("insert_space_tokens")
def insert_space_tokens(__context: Any = None, **kwargs) -> List[Dict[str, Any]]:
    ctx = _ctx_map(__context)
    glyphs = ctx.get("merge_ccs_to_glyphs", [])
    gaps_info = ctx.get("infer_space_threshold_per_line", {})
    lines = ctx.get("segment_lines_per_column_from_ccs", {}).get("lines", [])

    if not isinstance(glyphs, list):
        raise ValueError("insert_space_tokens: missing glyph list")
    if not isinstance(gaps_info, dict):
        raise ValueError("insert_space_tokens: missing gaps info")

    # map line_id -> y-band
    yband = {}
    if isinstance(lines, list):
        for ln in lines:
            yband[int(ln["line_id"])] = (float(ln["y0"]), float(ln["y1"]))

    by_line: Dict[int, List[Dict[str, Any]]] = {}
    for g in glyphs:
        if g.get("kind") != "GLYPH":
            continue
        by_line.setdefault(int(g.get("line_id", 0)), []).append(g)
    for lid in by_line:
        by_line[lid].sort(key=lambda t: _bbox(t)[0])

    out_tokens: List[Dict[str, Any]] = []

    for lid, gl in by_line.items():
        y0b, y1b = yband.get(int(lid), (0.0, 0.0))

        per = gaps_info.get("per_line", {}).get(str(int(lid)), {})
        gaps = per.get("gaps", [])
        label_by_i = {int(g["i"]): str(g["label"]) for g in gaps if isinstance(g, dict) and "i" in g and "label" in g}
        gap_by_i = {int(g["i"]): float(g["gap_px"]) for g in gaps if isinstance(g, dict) and "i" in g and "gap_px" in g}

        for i in range(len(gl)):
            out_tokens.append(gl[i])
            if i < len(gl) - 1 and label_by_i.get(int(i)) == "SPACE":
                x0, y0, x1, y1 = _bbox(gl[i])
                nx0, ny0, nx1, ny1 = _bbox(gl[i + 1])
                sx0 = float(x1)
                sx1 = float(nx0)
                if sx1 < sx0:
                    sx1 = sx0
                out_tokens.append({
                    "kind": "SPACE",
                    "bbox": (sx0, y0b, sx1, y1b),
                    "line_id": int(lid),
                    "col_id": int(gl[i].get("col_id", 0)),
                    "gap_px": float(gap_by_i.get(int(i), max(0.0, sx1 - sx0))),
                })

    out_tokens.sort(key=lambda t: (int(t.get("col_id", 0)), int(t.get("line_id", 0)), float(_bbox(t)[0]), 0 if t.get("kind") == "GLYPH" else 1))
    return out_tokens


def _line_tokens(tokens: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        out.setdefault(int(t.get("line_id", 0)), []).append(t)
    for lid in out:
        out[lid].sort(key=lambda t: _bbox(t)[0])
    return out


@register("identify_prose_runs_and_mask")
def identify_prose_runs_and_mask(__context: Any = None,
                                 tokens: Optional[List[Dict[str, Any]]] = None,
                                 min_run_tokens: int = 5,
                                 max_token_height_factor: float = 2.5,
                                 raised_marker_min_per_line: int = 2,
                                 raised_center_factor: float = 0.45,
                                 **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    toks = tokens if isinstance(tokens, list) else ctx.get("insert_space_tokens", [])
    scale = ctx.get("estimate_text_scale_from_ccs", {})
    xh = float(scale.get("xheight", 16.0))

    if not isinstance(toks, list):
        raise ValueError("identify_prose_runs_and_mask: need token list")

    toks2 = list(toks)
    toks2.sort(key=lambda t: (int(t.get("col_id", 0)), int(t.get("line_id", 0)), _bbox(t)[0], 0 if t.get("kind") == "GLYPH" else 1))

    by_line = _line_tokens(toks2)

    prose_ids: Set[int] = set()
    idx = {id(t): i for i, t in enumerate(toks2)}

    lines = ctx.get("segment_lines_per_column_from_ccs", {}).get("lines", [])
    yband = {int(ln["line_id"]): (float(ln["y0"]), float(ln["y1"])) for ln in lines} if isinstance(lines, list) else {}

    def _h(tok: Dict[str, Any]) -> float:
        x0, y0, x1, y1 = _bbox(tok)
        return max(1.0, y1 - y0)

    def _center_y(tok: Dict[str, Any]) -> float:
        x0, y0, x1, y1 = _bbox(tok)
        return 0.5 * (y0 + y1)

    for lid, line in by_line.items():
        glyphs = [t for t in line if t.get("kind") == "GLYPH"]
        if len(glyphs) < int(min_run_tokens):
            continue

        hs = sorted([_h(t) for t in glyphs])
        med_h = hs[len(hs) // 2]
        normal = [t for t in glyphs if _h(t) <= float(max_token_height_factor) * med_h]
        if len(normal) < int(min_run_tokens):
            continue

        y0b, y1b = yband.get(int(lid), (None, None))
        if y0b is not None:
            line_mid = 0.5 * (float(y0b) + float(y1b))
        else:
            cs = sorted([_center_y(t) for t in glyphs])
            line_mid = cs[len(cs) // 2]

        raised = 0
        for t in glyphs:
            if _h(t) <= 0.65 * xh and (_center_y(t) < line_mid - float(raised_center_factor) * xh):
                raised += 1

        if raised >= int(raised_marker_min_per_line):
            continue

        for t in normal:
            prose_ids.add(idx[id(t)])

    return {"prose_token_ids": sorted(list(prose_ids)), "tokens_ordered": toks2}


@register("pick_residual")
def pick_residual(__context: Any = None, **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    mask = ctx.get("identify_prose_runs_and_mask")
    if not isinstance(mask, dict):
        raise ValueError("pick_residual: need context['identify_prose_runs_and_mask']")

    toks = mask.get("tokens_ordered", [])
    prose_ids = set(mask.get("prose_token_ids", []))

    prose = []
    resid = []
    for i, t in enumerate(toks):
        if t.get("kind") != "GLYPH":
            continue
        (prose if i in prose_ids else resid).append(t)

    return {"prose_tokens": prose, "residual_tokens": resid}


@register("islandify")
def islandify(__context: Any = None,
              pad: float = 1.0,
              **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    pr = ctx.get("pick_residual", {})
    resid = pr.get("residual_tokens", [])
    if not isinstance(resid, list):
        raise ValueError("islandify: need residual token list")

    n = len(resid)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        bi = _bbox(resid[i])
        for j in range(i + 1, n):
            if _touching_bbox(bi, _bbox(resid[j]), pad=float(pad)):
                union(i, j)

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(resid[i])

    islands = list(groups.values())

    def island_key(isle):
        x0s = []; y0s = []
        for t in isle:
            x0, y0, x1, y1 = _bbox(t)
            x0s.append(x0); y0s.append(y0)
        return (min(y0s) if y0s else 0.0, min(x0s) if x0s else 0.0)

    islands.sort(key=island_key)
    return {"islands": islands}


@register("count_tokens")
def count_tokens(__context: Any = None, **kwargs) -> Dict[str, int]:
    ctx = _ctx_map(__context)
    glyphs = ctx.get("merge_ccs_to_glyphs", [])
    n = len(glyphs) if isinstance(glyphs, list) else 0
    print(f"count_glyphs: {n}")
    return {"count_glyphs": int(n)}
