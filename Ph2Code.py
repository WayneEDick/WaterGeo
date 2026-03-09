## Code-ready pseudocode for Phase 2
```
Assumes **Phase 1** already produced CC Boxes with frozen fields:

* `l,r,t,b,w,h,cx,cy,xh,kind,id`
  and you have constants `xh_d` (dominant x-height) and a choice of `g` (grid cell size).

I’ll give:

* `BuildGrid`
* `Cand(P)` (conservative neighborhood based on half-extents + margin)
* `V_Graph` (vertical eye boxes)
* `H_Graph` (horizontal runs with hard breaks)

I’ll keep it language-agnostic but close to Python/C/Fortran style.

---

# Data types

```text

script.yaml constants will  be flagged where i  find them

```
Box:
  id: int
  l,r,t,b: float
  w,h: float
  cx,cy: float
  xh: float
  kind: enum {CHAR_LIKE, BIG_SYM, OTHER, ...} 
  members: list[int]      // empty for Phase 1, nonempty for phase outputs
  grid_cell: (int,int)    // optional debug

Token:
  kind: enum {H_RUN, WD_SPACE, WIDE_SPACE, BIG_SYM_TOKEN, FLOATER_BLOB_TOKEN}
  l,r,t,b,cx,cy,w,h: float
  xh: float
  members: list[int]      // for H_RUN, member VBox ids; for atomic, singleton
```

---

# Utility geometry

```text
function union_bbox(box_ids, BoxMap):
  l = min(BoxMap[i].l for i in box_ids)
  r = max(BoxMap[i].r for i in box_ids)
  t = min(BoxMap[i].t for i in box_ids)
  b = max(BoxMap[i].b for i in box_ids)
  return (l,r,t,b)

function make_box_from_members(new_id, member_ids, BoxMap, kind_tag):
  (l,r,t,b) = union_bbox(member_ids, BoxMap)
  w = r-l; h = b-t
  cx = (l+r)/2; cy = (t+b)/2
  xh = median(BoxMap[i].xh for i in member_ids)
  return Box(new_id,l,r,t,b,w,h,cx,cy,xh,kind_tag, members=member_ids)
```

Edge overlap/gap (all edge-based):

```text
function ov_y(A,B):
  return max(0, min(A.b,B.b) - max(A.t,B.t))

function gap_y(A,B):
  // nonnegative vertical separation
  return max(0, A.t - B.b, B.t - A.b)

function gap_x(A,B):
  // assume B is to the right of A in ordering contexts
  return B.l - A.r
```

---

# Grid (hash only)

## BuildGrid(BoxSet)

```text
function BuildGrid(BoxSet, g):
  grid = dict mapping (i,j) -> list[box_id]
  for B in BoxSet:
    i = floor(B.cx / g)
    j = floor(B.cy / g)
    grid[(i,j)].append(B.id)
    B.grid_cell = (i,j)  // optional
  return grid
```

## Conservative neighborhood radii (half-extents + margin)

This implements the “don’t miss neighbors even if centers mislead” rule.

```text
function Cand(P, grid, g, margin_mult):
  // margin_mult is "m" in xh units; start m=2.0 for safety
  m = margin_mult
  hx = max(P.r - P.cx, P.cx - P.l)     // half-extent in x
  hy = max(P.b - P.cy, P.cy - P.t)     // half-extent in y

  Rx = ceil((hx + m * P.xh) / g)
  Ry = ceil((hy + m * P.xh) / g)

  (i0,j0) = P.grid_cell  // or recompute from cx,cy

  C = empty list
  for i in [i0-Rx .. i0+Rx]:
    for j in [j0-Ry .. j0+Ry]:
      for qid in grid.get((i,j), empty):
        if qid != P.id:
          C.append(qid)
  return C
```

Notes:

* This returns *candidates only*. Every real decision is edge geometry.
* `m=2.0` is safe; can tighten later. Put in script.yam.

---

# Pass 2A: V_Graph (vertical eye boxes)

### Constants (initial)

```text

// In script.yaml
eps_mult = 0.10   // ε = 0.10*xh(P)
delta_mult = 0.60 // δ = 0.60*xh(P)
a_mult = 1.00     // above window
d_mult = 0.70     // below window
k_max = 3
cand_margin_mult = 2.0
```

## VerticalCompanion(P,S)

```text
function VerticalCompanion(P, S):
  if P.kind != CHAR_LIKE:
    return false

  xh = P.xh
  eps = eps_mult * xh
  delta = delta_mult * xh
  a = a_mult * xh
  d = d_mult * xh

  // Horizontal containment / centering
  if S.l < P.l - eps: return false
  if S.r > P.r + eps: return false
  if not (P.l < S.cx and S.cx < P.r): return false

  // Vertical relation: overlap OR close
  if ov_y(P,S) <= 0 and gap_y(P,S) > delta:
    return false

  // Stack envelope constraint
  if S.t < P.t - a: return false
  if S.b > P.b + d: return false

  return true
```

## V_Graph main

```text
function V_Graph(CCBoxes, xh_d):
  // 1) build grid for CCBoxes
  g = 0.60 * xh_d // grid_edge=0.60 in script.yaml
  grid = BuildGrid(CCBoxes, g)

  assigned_to_planet = dict mapping sid -> pid (initially empty)
  VBoxMap = empty map new_id -> VBox
  new_boxes = empty list

  // stable planet order: (t, l, id)
  planets = [B for B in CCBoxes if B.kind == CHAR_LIKE]
  sort(planets by (B.t, B.l, B.id))

  used = set()  // CC ids already absorbed into some VBox

  for P in planets:
    if P.id in used:
      continue

    cands = Cand(P, grid, g, cand_margin_mult)

// Keep one Box type.

// Add a small “role” field (or derived property):

// role = PLANET_CANDIDATE (char_like, big_sym)

// role = DEBRIS_CANDIDATE (speck)

// Then enforce rules in the attachment pass:

// Anchors: P.role == PLANET_CANDIDATE

// Attachable: s.role == DEBRIS_CANDIDATE

// Important consequence of your statement:

// A planet can be attached (e.g., base x is attached to a fraction bar? or to an overline group?) — so attachment is not just “specks onto planets.”

// So generalize the machinery: “host” vs “guest” is not tied to the box type, it’s tied to the attachment relation.

//Practical policy set:

//Debris: may be guest only (can’t host anything).

//Planet: may be host, and may also be guest in a higher-level composite.

//Once a box becomes a host (has satellites), it becomes a compound glyph object; the host remains the representative ID.
//Next?

//So your earlier loop becomes consistent:

//Grid contains all boxes.

//Loop over “host-eligible” boxes (planets).

//Candidate query returns neighbors; filter to “guest-eligible” (debris) for dot/diacritic attachment.

// Later, you may add a second attachment phase where planets can attach to other planets (bars, radicals, limits) using a different predicate set.


    eligible = empty list
    for sid in cands:
      S = CCBoxesById[sid]
      if sid in used:
        continue
      if sid in assigned_to_planet:
        continue
      if VerticalCompanion(P, S):
        // score tuple for sorting companions
        gy = gap_y(P,S)
        dx = abs(S.cx - P.cx)
        eligible.append((gy, dx, S.id))

    sort(eligible by (gy, dx, id))

    companions = []
    for (gy, dx, sid) in eligible:
      companions.append(sid)
      assigned_to_planet[sid] = P.id
      if length(companions) == k_max:
        break

    member_ids = [P.id] + companions
    for mid in member_ids:
      used.add(mid)

    // Emit VBox
    vid = fresh_id()
    VBox = make_box_from_members(vid, member_ids, CCBoxesById, kind_tag=V_CHAR_STACK if companions else P.kind)
    VBoxMap[vid] = VBox
    new_boxes.append(VBox)

  // Add any CC boxes not used into VBox list as singletons
  for B in CCBoxes:
    if B.id not in used and B.id not in assigned_to_planet:
      vid = fresh_id()
      VBox = make_box_from_members(vid, [B.id], CCBoxesById, kind_tag=B.kind)
      VBoxMap[vid] = VBox
      new_boxes.append(VBox)

  // 2) build new grid for VBox level
  gridV = BuildGrid(new_boxes, g)  // g may remain tied to xh_d; ok for v1

  return (new_boxes, VBoxMap, gridV, g)
```

---

# Pass 2B: H_Graph (horizontal runs between hard breaks)

We assume we are scanning within a **provisional horizontal context**. For v1, we can do a simple line-like grouping by y-banding; or you can just do a global left-to-right sweep within coarse y bins.

Here’s a clean v1 that works well in practice:

* Build y-bins of height `bin_h = 1.5*xh_d` (approx line height)
* Process bins top-to-bottom; within each bin sort by `l`

### Constants (initial)

```text
// script.yaml
tau_tight_mult = 0.25     // τ_tight = 0.25*xh_ref
wd_mult = 0.75            // wd_space = 0.75*xh_d   (default)
wide_factor = 2.0         // WIDE_SPACE if gap > 2*wd_space
bin_height_mult = 1.5     // y-bin height
```

## H_Graph

```text
function H_Graph(VBoxes, xh_d):
  wd_space = wd_mult * xh_d
  bin_h = bin_height_mult * xh_d

  // 1) place VBoxes into y-bins by cy
  bins = dict mapping bin_index -> list[VBox_id]
  for B in VBoxes:
    k = floor(B.cy / bin_h)
    bins[k].append(B.id)

  tokens = empty list

  // process bins in order
  for k in sorted keys(bins):
    ids = bins[k]
    // sort left-to-right; tie by (t,id)
    sort(ids by (VBoxMap[id].l, VBoxMap[id].t, id))

    // scan building runs
    run_members = empty list
    run_bbox_accumulator = empty list  // store ids, then union at emit
    prev_id = None

    for cur_id in ids:
      Cur = VBoxMap[cur_id]

      // atomic tokens: BIG_SYM breaks runs on both sides
      if Cur.kind == BIG_SYM:
        // flush current run if any
        if length(run_members) > 0:
          tokens.append(emit_H_RUN(run_members))
          run_members = []

        // emit BIG_SYM token
        tokens.append(emit_atomic_token(BIG_SYM_TOKEN, [cur_id], Cur))
        prev_id = None
        continue

      if prev_id is None:
        run_members.append(cur_id)
        prev_id = cur_id
        continue

      Prev = VBoxMap[prev_id]
      gx = Cur.l - Prev.r
      xh_ref = median(Prev.xh, Cur.xh)
      tau_tight = tau_tight_mult * xh_ref

      // If negative gap (overlap), treat as tight adjacency
      if gx < 0:
        run_members.append(cur_id)
        prev_id = cur_id
        continue

      // Hard space break if not tight
      if gx > tau_tight:
        // flush run
        if length(run_members) > 0:
          tokens.append(emit_H_RUN(run_members))
          run_members = []

        // emit space token
        if gx > wide_factor * wd_space:
          tokens.append(emit_space_token(WIDE_SPACE, Prev, Cur))
        else:
          tokens.append(emit_space_token(WD_SPACE, Prev, Cur))

        // start new run
        run_members.append(cur_id)
        prev_id = cur_id
      else:
        // tight adjacency => no space token
        run_members.append(cur_id)
        prev_id = cur_id

    // flush end-of-bin run
    if length(run_members) > 0:
      tokens.append(emit_H_RUN(run_members))

  return tokens
```

### Emit helpers

```text
function emit_H_RUN(member_ids):
  tid = fresh_id()
  (l,r,t,b) = union_bbox(member_ids, VBoxMap)
  w=r-l; h=b-t; cx=(l+r)/2; cy=(t+b)/2
  xh = median(VBoxMap[i].xh for i in member_ids)
  return Token(H_RUN, l,r,t,b,cx,cy,w,h,xh, members=member_ids)

function emit_space_token(space_kind, A, B):
  // space bbox spans the gap between A and B, with a small vertical extent
  l = A.r
  r = B.l
  // conservative vertical span: union, or a tight band around their overlap
  t = min(A.t, B.t)
  b = max(A.b, B.b)
  w = r-l; h=b-t; cx=(l+r)/2; cy=(t+b)/2
  xh = median(A.xh, B.xh)
  return Token(space_kind, l,r,t,b,cx,cy,w,h,xh, members=[])

function emit_atomic_token(kind_token, member_ids, B):
  return Token(kind_token, B.l,B.r,B.t,B.b,B.cx,B.cy,B.w,B.h,B.xh, members=member_ids)
```

---

# What to expect (sanity checks)

* V_Graph should reduce “vertical floaters” near letters without eating large constructs.
* H_Graph should output:

  * alternating `H_RUN` and `WD_SPACE/WIDE_SPACE` tokens
  * `BIG_SYM_TOKEN` standing alone
* `WIDE_SPACE` should correlate with real layout breaks.

---

If you want next, I can replace the coarse y-bin scan with a **more correct “scan context”** rule based on vertical overlap (still cheap), so H_Graph doesn’t accidentally mix two adjacent lines in the same bin.
