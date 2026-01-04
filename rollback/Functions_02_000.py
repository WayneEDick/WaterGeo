# Functions_02_001.py — Water Geo (PNG-first) with CC→Glyph merge + per-line space inference
# Core pipeline:
#   load_image -> binarize_png -> segment_lines -> extract_ccs ->
#   merge_ccs_to_glyphs -> infer_space_threshold_per_line -> insert_space_tokens ->
#   identify_prose_runs_and_mask -> pick_residual -> islandify

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

def _bbox_from_cc(cc: Tuple[int,int,int,int,int]) -> Tuple[float,float,float,float]:
    x,y,w,h,_a = cc
    return (float(x), float(y), float(x+w), float(y+h))

def _bbox(tok: Dict[str, Any]) -> Tuple[float,float,float,float]:
    b = tok.get("bbox")
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError("token missing bbox")
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])

def _h(tok: Dict[str, Any]) -> float:
    x0,y0,x1,y1 = _bbox(tok)
    return max(1.0, y1-y0)

def _center_y(tok: Dict[str, Any]) -> float:
    x0,y0,x1,y1 = _bbox(tok)
    return 0.5*(y0+y1)

def _touching_bbox(a: Tuple[float,float,float,float],
                   b: Tuple[float,float,float,float],
                   pad: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)

def _assign_line_id(lines: List[Tuple[int,int]], y: float) -> int:
    if not lines:
        return 0
    for lid, (y0,y1) in enumerate(lines):
        if y0 <= y <= y1:
            return lid
    mids = [0.5*(a+b) for (a,b) in lines]
    return int(np.argmin([abs(y-m) for m in mids]))


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


@register("segment_lines")
def segment_lines(__context: Any = None,
                  binary: Optional[np.ndarray] = None,
                  eps_ink: int = 0,
                  min_valley_rows: int = 1,
                  cc_min_area: int = 5,
                  **kwargs) -> List[Tuple[int,int]]:
    ctx = _ctx_map(__context)
    mask = binary if isinstance(binary, np.ndarray) else ctx.get("binarize_png")
    if not isinstance(mask, np.ndarray):
        raise ValueError("segment_lines: need binary mask (arg 'binary' or context['binarize_png'])")

    ink = (mask > 0).astype(np.uint8)
    H, W = ink.shape[:2]
    proj = ink.sum(axis=1)
    seps = (proj == 0)

    def _runs_of_true(b: np.ndarray) -> List[Tuple[int,int]]:
        runs = []
        in_run = False
        s = 0
        for i, v in enumerate(b.tolist()):
            if v and not in_run:
                in_run = True
                s = i
            elif not v and in_run:
                runs.append((s, i-1))
                in_run = False
        if in_run:
            runs.append((s, len(b)-1))
        return runs

    nonsep = ~seps
    raw_line_runs = _runs_of_true(nonsep)

    if len(raw_line_runs) <= 1:
        near = (proj <= int(eps_ink))
        valley_runs = _runs_of_true(near)
        valley_runs = [(a,b) for (a,b) in valley_runs if (b-a+1) >= int(min_valley_rows)]
        if valley_runs:
            sep2 = np.zeros_like(seps, dtype=bool)
            for a,b in valley_runs:
                sep2[a:b+1] = True
            raw_line_runs = _runs_of_true(~sep2)

    if len(raw_line_runs) <= 1:
        num, labels, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
        ccs = []
        for k in range(1, num):
            x,y,w,h,area = map(int, stats[k])
            if area >= int(cc_min_area):
                cy = float(cents[k][1])
                ccs.append((cy, y, y+h-1, h))
        if not ccs:
            return [(0, H-1)]
        heights = sorted([h for (_cy,_y0,_y1,h) in ccs])
        med_h = heights[len(heights)//2]
        gap = max(1.0, 0.7*float(med_h))

        ccs.sort(key=lambda t: t[0])
        clusters: List[List[Tuple[float,int,int,int]]] = []
        cur: List[Tuple[float,int,int,int]] = []
        last = None
        for item in ccs:
            cy = item[0]
            if last is None or abs(cy-last) <= gap:
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

    lines = [(int(a), int(b)) for (a,b) in raw_line_runs]
    print(f"segment_lines: {len(lines)} lines")
    return lines


@register("extract_ccs")
def extract_ccs(__context: Any = None,
                binary: Optional[np.ndarray] = None,
                min_area: int = 5,
                connectivity: int = 8,
                **kwargs) -> List[Tuple[int,int,int,int,int]]:
    ctx = _ctx_map(__context)
    mask = binary if isinstance(binary, np.ndarray) else ctx.get("binarize_png")
    if not isinstance(mask, np.ndarray):
        raise ValueError("extract_ccs: need binary mask (arg 'binary' or context['binarize_png'])")

    m = (mask > 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=int(connectivity))
    out: List[Tuple[int,int,int,int,int]] = []
    for k in range(1, num):
        x,y,w,h,area = map(int, stats[k])
        if area >= int(min_area):
            out.append((x,y,w,h,area))
    return out


@register("estimate_baseline_and_xheight")
def estimate_baseline_and_xheight(__context: Any = None, **kwargs) -> Dict[str, float]:
    ctx = _ctx_map(__context)
    ccs = ctx.get("extract_ccs", [])
    if not isinstance(ccs, list) or not ccs:
        return {"baseline": 0.0, "xheight": 16.0}

    hs = sorted(float(h) for (_x,_y,_w,h,_a) in ccs)
    bottoms = sorted(float(y + h) for (_x,y,_w,h,_a) in ccs)

    n = len(hs)
    med_h = hs[n//2] if n % 2 else 0.5*(hs[n//2 - 1] + hs[n//2])
    med_b = bottoms[n//2] if n % 2 else 0.5*(bottoms[n//2 - 1] + bottoms[n//2])

    xh = float(max(8.0, min(64.0, med_h)))
    return {"baseline": float(med_b), "xheight": xh}


@register("merge_ccs_to_glyphs")
def merge_ccs_to_glyphs(__context: Any = None,
                        ccs: Optional[List[Tuple[int,int,int,int,int]]] = None,
                        pad_px: float = 1.0,
                        **kwargs) -> List[Dict[str, Any]]:
    ctx = _ctx_map(__context)
    ccs_in = ccs if isinstance(ccs, list) else ctx.get("extract_ccs", [])
    lines = ctx.get("segment_lines", [])
    metrics = ctx.get("estimate_baseline_and_xheight", {})
    xh = float(metrics.get("xheight", 16.0))

    if not isinstance(ccs_in, list) or not ccs_in:
        return []

    recs = []
    for cc in ccs_in:
        x,y,w,h,area = cc
        cy = y + 0.5*h
        lid = _assign_line_id(lines, cy)
        recs.append({
            "cc": cc,
            "bbox": _bbox_from_cc(cc),
            "line_id": int(lid),
            "area": int(area),
        })

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
            wi = max(1.0, ix1-ix0)
            hi = max(1.0, iy1-iy0)
            for jj in range(ii+1, n):
                j = idxs[jj]
                bj = recs[j]["bbox"]
                jx0, jy0, jx1, jy1 = bj
                wj = max(1.0, jx1-jx0)
                hj = max(1.0, jy1-jy0)

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

                # equals bars (short + y-overlap + close)
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
            for i in gidxs:
                x0,y0,x1,y1 = recs[i]["bbox"]
                xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
                area += int(recs[i]["area"])
            glyphs.append({
                "kind": "GLYPH",
                "bbox": (float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))),
                "line_id": int(lid),
                "area": int(area),
                "members": int(len(gidxs)),
            })

    glyphs.sort(key=lambda t: (int(t["line_id"]), float(_bbox(t)[0])))
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
    if c0 <= c1:
        return (c0, c1)
    return (c1, c0)

@register("infer_space_threshold_per_line")
def infer_space_threshold_per_line(__context: Any = None,
                                   glyphs: Optional[List[Dict[str,Any]]] = None,
                                   min_gaps_for_kmeans: int = 4,
                                   fallback_cutoff_factor: float = 0.60,
                                   **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    glyphs_in = glyphs if isinstance(glyphs, list) else ctx.get("merge_ccs_to_glyphs", [])
    metrics = ctx.get("estimate_baseline_and_xheight", {})
    xh = float(metrics.get("xheight", 16.0))

    if not isinstance(glyphs_in, list) or not glyphs_in:
        return {"per_line": {}}

    by_line: Dict[int, List[Dict[str,Any]]] = {}
    for g in glyphs_in:
        if g.get("kind") != "GLYPH":
            continue
        by_line.setdefault(int(g.get("line_id", 0)), []).append(g)
    for lid in by_line:
        by_line[lid].sort(key=lambda t: _bbox(t)[0])

    out: Dict[str, Any] = {"per_line": {}}

    for lid, gl in by_line.items():
        gaps: List[float] = []
        for i in range(len(gl)-1):
            x0,y0,x1,y1 = _bbox(gl[i])
            nx0,ny0,nx1,ny1 = _bbox(gl[i+1])
            gaps.append(float(max(0.0, nx0 - x1)))

        if len(gaps) >= int(min_gaps_for_kmeans) and (max(gaps) > min(gaps)):
            arr = np.array(gaps, dtype=np.float32)
            c0, c1 = _kmeans2_1d(arr)
            cutoff = 0.5*(c0 + c1)
        else:
            cutoff = float(fallback_cutoff_factor) * xh

        labeled = []
        for i, gap in enumerate(gaps):
            label = "SPACE" if float(gap) >= float(cutoff) else "INTRA"
            labeled.append({"i": int(i), "gap_px": float(gap), "label": label})

        out["per_line"][str(int(lid))] = {"cutoff_px": float(cutoff), "gaps": labeled}

    return out


@register("insert_space_tokens")
def insert_space_tokens(__context: Any = None, **kwargs) -> List[Dict[str, Any]]:
    ctx = _ctx_map(__context)
    glyphs = ctx.get("merge_ccs_to_glyphs", [])
    gaps_info = ctx.get("infer_space_threshold_per_line", {})
    lines = ctx.get("segment_lines", [])

    if not isinstance(glyphs, list):
        raise ValueError("insert_space_tokens: missing glyph list")
    if not isinstance(gaps_info, dict):
        raise ValueError("insert_space_tokens: missing gaps info")

    by_line: Dict[int, List[Dict[str,Any]]] = {}
    for g in glyphs:
        if g.get("kind") != "GLYPH":
            continue
        by_line.setdefault(int(g.get("line_id",0)), []).append(g)
    for lid in by_line:
        by_line[lid].sort(key=lambda t: _bbox(t)[0])

    out_tokens: List[Dict[str, Any]] = []

    for lid, gl in by_line.items():
        if isinstance(lines, list) and 0 <= int(lid) < len(lines):
            y0b, y1b = lines[int(lid)]
        else:
            y0b, y1b = (0, 0)
        y0b, y1b = float(y0b), float(y1b)

        per = gaps_info.get("per_line", {}).get(str(int(lid)), {})
        gaps = per.get("gaps", [])
        label_by_i = {int(g["i"]): str(g["label"]) for g in gaps if isinstance(g, dict) and "i" in g and "label" in g}
        gap_by_i = {int(g["i"]): float(g["gap_px"]) for g in gaps if isinstance(g, dict) and "i" in g and "gap_px" in g}

        for i in range(len(gl)):
            out_tokens.append(gl[i])
            if i < len(gl)-1 and label_by_i.get(int(i)) == "SPACE":
                x0,y0,x1,y1 = _bbox(gl[i])
                nx0,ny0,nx1,ny1 = _bbox(gl[i+1])
                sx0 = float(x1)
                sx1 = float(nx0)
                if sx1 < sx0:
                    sx1 = sx0
                out_tokens.append({
                    "kind": "SPACE",
                    "bbox": (sx0, y0b, sx1, y1b),
                    "line_id": int(lid),
                    "gap_px": float(gap_by_i.get(int(i), max(0.0, sx1-sx0))),
                })

    out_tokens.sort(key=lambda t: (int(t.get("line_id",0)), float(_bbox(t)[0]), 0 if t.get("kind")=="GLYPH" else 1))
    return out_tokens


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
                                 raised_marker_min_per_line: int = 2,
                                 raised_center_factor: float = 0.45,
                                 **kwargs) -> Dict[str, Any]:
    ctx = _ctx_map(__context)
    toks = tokens if isinstance(tokens, list) else ctx.get("insert_space_tokens", [])
    metrics = ctx.get("estimate_baseline_and_xheight", {})
    xh = float(metrics.get("xheight", 16.0))

    if not isinstance(toks, list):
        raise ValueError("identify_prose_runs_and_mask: need token list")

    toks2 = list(toks)
    toks2.sort(key=lambda t: (int(t.get("line_id",0)), _bbox(t)[0], 0 if t.get("kind")=="GLYPH" else 1))

    by_line = _line_tokens(toks2)

    prose_ids: Set[int] = set()
    idx = {id(t): i for i, t in enumerate(toks2)}

    lines = ctx.get("segment_lines", [])

    for lid, line in by_line.items():
        glyphs = [t for t in line if t.get("kind") == "GLYPH"]
        if len(glyphs) < int(min_run_tokens):
            continue

        hs = sorted([_h(t) for t in glyphs])
        med_h = hs[len(hs)//2]
        normal = [t for t in glyphs if _h(t) <= float(max_token_height_factor)*med_h]
        if len(normal) < int(min_run_tokens):
            continue

        if isinstance(lines, list) and 0 <= int(lid) < len(lines):
            y0b, y1b = lines[int(lid)]
            line_mid = 0.5*(float(y0b)+float(y1b))
        else:
            cs = sorted([_center_y(t) for t in glyphs])
            line_mid = cs[len(cs)//2]

        raised = 0
        for t in glyphs:
            if _h(t) <= 0.65*xh and (_center_y(t) < line_mid - float(raised_center_factor)*xh):
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
        for j in range(i+1, n):
            if _touching_bbox(bi, _bbox(resid[j]), pad=float(pad)):
                union(i, j)

    groups: Dict[int, List[Dict[str,Any]]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(resid[i])

    islands = list(groups.values())

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
    glyphs = ctx.get("merge_ccs_to_glyphs", [])
    n = len(glyphs) if isinstance(glyphs, list) else 0
    print(f"count_glyphs: {n}")
    return {"count_glyphs": int(n)}
