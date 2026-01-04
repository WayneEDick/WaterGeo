# Functions_02_001.py — Water Geo clean functions (single-file registry)
# PNG-first pipeline core:
#   load_image -> binarize_png -> segment_lines -> extract_ccs -> tokens -> prose mask -> residual -> islandify

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


# -----------------------------
# Load + Binarize
# -----------------------------

@register("load_image")
def load_image(path: str, __context: Any = None, **kwargs) -> np.ndarray:
    """Load image as grayscale uint8."""
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
    """
    Return ink mask uint8 in {0,255}, with ink=255 (mask>0 means ink).
    """
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


# -----------------------------
# Segment Lines (HARD zero-row separators + fallbacks)
# Output: list[(y0,y1)] for each line
# -----------------------------

@register("segment_lines")
def segment_lines(__context: Any = None,
                  binary: Optional[np.ndarray] = None,
                  eps_ink: int = 0,
                  min_valley_rows: int = 1,
                  cc_min_area: int = 5,
                  **kwargs) -> List[Tuple[int, int]]:
    """
    Segment text lines using hard zero rows in horizontal projection.
    Fallbacks:
      (1) near-zero valleys (≤ eps_ink) with thickness ≥ min_valley_rows
      (2) CC y-centroid clustering if projection yields 0 or 1 line
    """
    ctx = _ctx_map(__context)
    mask = binary if isinstance(binary, np.ndarray) else ctx.get("binarize_png")
    if not isinstance(mask, np.ndarray):
        raise ValueError("segment_lines: need binary mask (arg 'binary' or context['binarize_png'])")

    ink = (mask > 0).astype(np.uint8)
    H, W = ink.shape[:2]

    proj = ink.sum(axis=1)  # ink pixels per row

    # Hard separators: rows where proj == 0
    seps = (proj == 0)

    def _runs_of_true(b: np.ndarray) -> List[Tuple[int, int]]:
        runs = []
        in_run = False
        s = 0
        for i, v in enumerate(b.tolist()):
            if v and not in_run:
                in_run = True
                s = i
            elif not v and in_run:
                runs.append((s, i - 1))
                in_run = False
        if in_run:
            runs.append((s, len(b) - 1))
        return runs

    # First pass: lines are runs of NON-sep rows
    nonsep = ~seps
    raw_line_runs = _runs_of_true(nonsep)

    # If everything is one run (no zero rows), try near-zero valleys
    if len(raw_line_runs) <= 1:
        near = (proj <= int(eps_ink))
        valley_runs = _runs_of_true(near)
        # Keep only valleys thick enough
        valley_runs = [(a, b) for (a, b) in valley_runs if (b - a + 1) >= int(min_valley_rows)]
        if valley_runs:
            # Build separators from valley rows
            sep2 = np.zeros_like(seps, dtype=bool)
            for a, b in valley_runs:
                sep2[a:b+1] = True
            nonsep2 = ~sep2
            raw_line_runs = _runs_of_true(nonsep2)

    # If still 0/1 lines, fallback to CC clustering
    if len(raw_line_runs) <= 1:
        # CC-based clustering on y-centroids
        num, labels, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
        ccs = []
        for k in range(1, num):
            x, y, w, h, area = map(int, stats[k])
            if area >= int(cc_min_area):
                cy = float(cents[k][1])
                ccs.append((cy, y, y + h - 1, h))
        if not ccs:
            return [(0, H - 1)]

        heights = sorted([h for (_cy, _y0, _y1, h) in ccs])
        med_h = heights[len(heights)//2]
        gap = max(1.0, 0.7 * float(med_h))

        ccs.sort(key=lambda t: t[0])
        clusters: List[List[Tuple[float,int,int,int]]] = []
        cur: List[Tuple[float,int,int,int]] = []
        last = None
        for item in ccs:
            cy = item[0]
            if last is None or abs(cy - last) <= gap:
                cur.append(item)
            else:
                clusters.append(cur)
                cur = [item]
            last = cy
        if cur:
            clusters.append(cur)

        lines = []
        for cl in clusters:
            y0 = min(t[1] for t in cl)
            y1 = max(t[2] for t in cl)
            lines.append((int(y0), int(y1)))
        print(f"segment_lines: {len(lines)} lines (CC fallback)")
        return lines

    lines = [(int(a), int(b)) for (a, b) in raw_line_runs]
    print(f"segment_lines: {len(lines)} lines")
    return lines


# -----------------------------
# Connected components
# -----------------------------

@register("extract_ccs")
def extract_ccs(__context: Any = None,
                binary: Optional[np.ndarray] = None,
                min_area: int = 5,
                connectivity: int = 8,
                **kwargs) -> List[Tuple[int, int, int, int, int]]:
    """
    Returns list of (x, y, w, h, area) excluding background.
    """
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


@register("estimate_baseline_and_xheight")
def estimate_baseline_and_xheight(__context: Any = None, **kwargs) -> Dict[str, float]:
    """
    Placeholder global estimates:
      xheight := median CC height (clamped)
      baseline := median CC bottom y (rough)
    """
    ctx = _ctx_map(__context)
    ccs = ctx.get("extract_ccs", [])
    if not isinstance(ccs, list) or not ccs:
        return {"baseline": 0.0, "xheight": 16.0}

    hs = sorted(float(h) for (_x, _y, _w, h, _a) in ccs)
    bottoms = sorted(float(y + h) for (_x, y, _w, h, _a) in ccs)

    n = len(hs)
    med_h = hs[n//2] if n % 2 else 0.5*(hs[n//2 - 1] + hs[n//2])
    med_b = bottoms[n//2] if n % 2 else 0.5*(bottoms[n//2 - 1] + bottoms[n//2])

    xh = float(max(8.0, min(64.0, med_h)))
    return {"baseline": float(med_b), "xheight": xh}


# -----------------------------
# Tokenization (minimal, CC-as-token)
# -----------------------------

def _assign_line_id(lines: List[Tuple[int,int]], y: float) -> int:
    if not lines:
        return 0
    for lid, (y0,y1) in enumerate(lines):
        if y0 <= y <= y1:
            return lid
    mids = [0.5*(a+b) for (a,b) in lines]
    return int(np.argmin([abs(y-m) for m in mids]))

@register("merge_tokens_by_gap_baseline")
def merge_tokens_by_gap_baseline(__context: Any = None,
                                 ccs: Optional[List[Tuple[int,int,int,int,int]]] = None,
                                 lines: Optional[List[Tuple[int,int]]] = None,
                                 **kwargs) -> List[Dict[str, Any]]:
    """
    Minimal token list: each CC -> token dict with bbox + line_id.
    bbox is (x0,y0,x1,y1) float.
    """
    ctx = _ctx_map(__context)
    ccs_in = ccs if isinstance(ccs, list) else ctx.get("extract_ccs", [])
    lines_in = lines if isinstance(lines, list) else ctx.get("segment_lines", [])

    toks: List[Dict[str, Any]] = []
    for (x,y,w,h,area) in ccs_in:
        cy = y + 0.5*h
        lid = _assign_line_id(lines_in, cy)
        toks.append({
            "bbox": (float(x), float(y), float(x+w), float(y+h)),
            "line_id": int(lid),
            "area": int(area),
        })

    toks.sort(key=lambda t: (t["line_id"], t["bbox"][0]))
    return toks


# -----------------------------
# Island rules (ported in-place, no external deps)
# -----------------------------

def _bbox(tok: Dict[str, Any]) -> Tuple[float,float,float,float]:
    b = tok.get("bbox")
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError("token missing bbox")
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])

def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _touching(a: Dict[str, Any], b: Dict[str, Any], pad: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)

def _center_y(tok: Dict[str, Any]) -> float:
    x0,y0,x1,y1 = _bbox(tok)
    return 0.5*(y0+y1)

def _h(tok: Dict[str, Any]) -> float:
    x0,y0,x1,y1 = _bbox(tok)
    return max(1.0, y1-y0)

def _w(tok: Dict[str, Any]) -> float:
    x0,y0,x1,y1 = _bbox(tok)
    return max(1.0, x1-x0)

def _line_tokens(tokens: List[Dict[str,Any]]) -> Dict[int, List[Dict[str,Any]]]:
    out: Dict[int, List[Dict[str,Any]]] = {}
    for t in tokens:
        out.setdefault(int(t.get("line_id", 0)), []).append(t)
    for lid in out:
        out[lid].sort(key=lambda t: _bbox(t)[0])
    return out

@register("identify_prose_runs_and_mask")
def identify_prose_runs_and_mask(__context: Any = None,
                                 tokens: Optional[List[Dict[str,Any]]] = None,
                                 min_run_tokens: int = 5,
                                 max_token_height_factor: float = 2.5,
                                 **kwargs) -> Dict[str, Any]:
    """
    Identify prose runs per line and return:
      - prose_token_ids: set of indices (in returned 'tokens_ordered') that are prose
      - tokens_ordered: stable token list (sorted by line_id, x)
    This is a conservative heuristic: "prose" = long-ish runs of similar-height tokens on same line.
    """
    ctx = _ctx_map(__context)
    toks = tokens if isinstance(tokens, list) else ctx.get("merge_tokens_by_gap_baseline", [])
    if not isinstance(toks, list):
        raise ValueError("identify_prose_runs_and_mask: need token list")

    # Stable order
    toks2 = list(toks)
    toks2.sort(key=lambda t: (int(t.get("line_id",0)), _bbox(t)[0]))

    by_line = _line_tokens(toks2)

    prose_ids: Set[int] = set()
    # Build a mapping token->index in toks2
    idx = {id(t): i for i, t in enumerate(toks2)}

    for lid, line in by_line.items():
        if len(line) < int(min_run_tokens):
            continue
        hs = [_h(t) for t in line]
        med_h = sorted(hs)[len(hs)//2]
        good = [t for t in line if _h(t) <= float(max_token_height_factor)*med_h]
        if len(good) >= int(min_run_tokens):
            for t in good:
                prose_ids.add(idx[id(t)])

    return {"prose_token_ids": sorted(list(prose_ids)), "tokens_ordered": toks2}

@register("pick_residual")
def pick_residual(__context: Any = None, **kwargs) -> Dict[str, Any]:
    """
    Produce:
      - residual_tokens: tokens not in prose
      - prose_tokens: prose tokens
    """
    ctx = _ctx_map(__context)
    mask = ctx.get("identify_prose_runs_and_mask")
    if not isinstance(mask, dict):
        raise ValueError("pick_residual: need context['identify_prose_runs_and_mask']")

    toks = mask.get("tokens_ordered", [])
    prose_ids = set(mask.get("prose_token_ids", []))

    prose = []
    resid = []
    for i, t in enumerate(toks):
        (prose if i in prose_ids else resid).append(t)

    return {"prose_tokens": prose, "residual_tokens": resid}

@register("islandify")
def islandify(__context: Any = None,
              pad: float = 1.0,
              **kwargs) -> Dict[str, Any]:
    """
    Islandify residual tokens via simple union-find on bbox touching (with pad).
    Returns:
      - islands: list[list[token]]
    """
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
        for j in range(i+1, n):
            if _touching(resid[i], resid[j], pad=float(pad)):
                union(i, j)

    groups: Dict[int, List[Dict[str,Any]]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(resid[i])

    islands = list(groups.values())
    # Sort islands by top-left
    def island_key(isle):
        x0s = []; y0s = []
        for t in isle:
            x0,y0,x1,y1 = _bbox(t)
            x0s.append(x0); y0s.append(y0)
        return (min(y0s) if y0s else 0.0, min(x0s) if x0s else 0.0)

    islands.sort(key=island_key)
    return {"islands": islands}

@register("count_tokens")
def count_tokens(__context: Any = None, **kwargs) -> Dict[str, int]:
    ctx = _ctx_map(__context)
    toks = ctx.get("merge_tokens_by_gap_baseline", [])
    n = len(toks) if isinstance(toks, list) else 0
    print(f"count_tokens: {n}")
    return {"count": int(n)}
