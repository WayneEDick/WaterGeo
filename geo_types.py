# geo_types.py
#
# Water Geo "constitution":
# - defines core types (BBox, Mask, CC)
# - defines invariants (assert_bbox, assert_mask01)
# - defines small, pure geometry helpers
#
# NO algorithmic logic belongs here (no CC detection, no columns, no MER).

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TypeAlias

import numpy as np


# ----------------------------
# Core Types
# ----------------------------

# BBox = (top, left, height, width)
BBox: TypeAlias = Tuple[int, int, int, int]

# A mask is a 2D uint8 array with values in {0,1}.
Mask: TypeAlias = np.ndarray


# ----------------------------
# BBox helpers + invariants
# ----------------------------

def bbox_top(b: BBox) -> int:
    return b[0]


def bbox_left(b: BBox) -> int:
    return b[1]


def bbox_height(b: BBox) -> int:
    return b[2]


def bbox_width(b: BBox) -> int:
    return b[3]


def bbox_right(b: BBox) -> int:
    t, l, h, w = b
    return l + w


def bbox_bottom(b: BBox) -> int:
    t, l, h, w = b
    return t + h


def bbox_area(b: BBox) -> int:
    t, l, h, w = b
    return h * w


def assert_bbox(b: BBox, *, bounds_hw: Optional[Tuple[int, int]] = None) -> None:
    """
    Invariants:
      - BBox is 4 ints
      - height > 0, width > 0
      - top >= 0, left >= 0
      - if bounds_hw provided (H,W): bbox is inside image bounds
    """
    t, l, h, w = b

    assert isinstance(t, int) and isinstance(l, int) and isinstance(h, int) and isinstance(w, int), "BBox must be 4 ints"
    assert h > 0 and w > 0, "BBox height/width must be > 0"
    assert t >= 0 and l >= 0, "BBox top/left must be >= 0"

    if bounds_hw is not None:
        H, W = bounds_hw
        assert isinstance(H, int) and isinstance(W, int), "bounds_hw must be (int,int)"
        assert H > 0 and W > 0, "bounds_hw must be positive"
        assert bbox_bottom(b) <= H, f"BBox bottom {bbox_bottom(b)} exceeds image height {H}"
        assert bbox_right(b) <= W, f"BBox right {bbox_right(b)} exceeds image width {W}"


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """
    Half-open rectangles:
      a covers rows [top, bottom) and cols [left, right)
    """
    assert_bbox(a)
    assert_bbox(b)
    return not (
        bbox_right(a) <= bbox_left(b)
        or bbox_right(b) <= bbox_left(a)
        or bbox_bottom(a) <= bbox_top(b)
        or bbox_bottom(b) <= bbox_top(a)
    )


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    assert_bbox(outer)
    assert_bbox(inner)
    return (
        bbox_top(outer) <= bbox_top(inner)
        and bbox_left(outer) <= bbox_left(inner)
        and bbox_bottom(inner) <= bbox_bottom(outer)
        and bbox_right(inner) <= bbox_right(outer)
    )


# ----------------------------
# Mask helpers + invariants
# ----------------------------

def assert_mask01(mask: Mask, *, bounds_hw: Optional[Tuple[int, int]] = None) -> None:
    """
    Invariants:
      - mask is numpy ndarray
      - 2D
      - dtype uint8
      - values in {0,1}
      - if bounds_hw provided (H,W): shape equals (H,W)
    """
    assert isinstance(mask, np.ndarray), "mask must be a numpy ndarray"
    assert mask.ndim == 2, "mask must be 2D"
    assert mask.dtype == np.uint8, "mask dtype must be uint8"

    if mask.size > 0:
        mn = int(mask.min())
        mx = int(mask.max())
        assert mn >= 0 and mx <= 1, f"mask values must be in {{0,1}}; got min={mn}, max={mx}"

    if bounds_hw is not None:
        H, W = bounds_hw
        assert isinstance(H, int) and isinstance(W, int), "bounds_hw must be (int,int)"
        assert mask.shape == (H, W), f"mask shape {mask.shape} must match bounds {(H, W)}"


def normalize_to_mask01(arr: np.ndarray) -> Mask:
    """
    Convert any array with "ink" as nonzero into a {0,1} uint8 mask.
    Useful at IO boundaries (e.g., OpenCV uses {0,255}; we normalize to {0,1}.)
    """
    assert isinstance(arr, np.ndarray), "arr must be a numpy ndarray"
    m = (arr != 0).astype(np.uint8)
    assert_mask01(m)
    return m


# ----------------------------
# Connected Components (CC)
# ----------------------------

class CCKind(str, Enum):
    UNKNOWN = "unknown"
    TEXT_LIKE = "text_like"
    IMAGE_LIKE = "image_like"
    RULE_LIKE = "rule_like"   # thin strokes (rules), not letters


@dataclass(frozen=True, slots=True)
class CC:
    """
    Geometry-only connected component record.
    No recognition. No grouping logic.
    """
    cc_id: int
    bbox: BBox
    ink_px: int
    kind: CCKind = CCKind.UNKNOWN
    parent: Optional[int] = None  # optional grouping later (blocks/lines)

    def __post_init__(self) -> None:
        assert isinstance(self.cc_id, int) and self.cc_id >= 0, "cc_id must be a nonnegative int"
        assert_bbox(self.bbox)
        assert isinstance(self.ink_px, int) and self.ink_px >= 0, "ink_px must be a nonnegative int"
        # kind validated by Enum
