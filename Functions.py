from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import os

import numpy as np
import cv2

from geo_types import BBox, CC


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

    connectivity = int(cfg.get("params", {}).get("cc", {}).get("connectivity", 8) or 8)
    connectivity = 8 if connectivity != 4 else 4

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
