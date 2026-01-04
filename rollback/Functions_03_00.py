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
import numpy as np


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

def load_normalize(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G0: Load & Normalize (placeholder)
    Expected outputs:
      ctx["imageGray"], ctx["page"]
    """
    raise NotImplementedError("Hook this to your existing PNG/page loader.")


def binarize(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G1: Binarize (placeholder)
    Expected outputs:
      ctx["imageBin"] with convention ink=0, background=255 (per WATER GEO v2).
    """
    raise NotImplementedError("Hook this to your existing Otsu/auto-polarity binarizer.")


def connected_components(ctx: Context, cfg: Dict[str, Any]) -> None:
    """
    G2: Connected Components (placeholder)
    Expected outputs:
      ctx["ccs"] = List[CC]
      ctx["ccLabels"] (optional)
    """
    raise NotImplementedError("Hook this to your existing CC extractor.")


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
