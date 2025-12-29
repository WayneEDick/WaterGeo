# Runner_02_001.py — Water Geo clean runner (PNG-first)
# Usage:
#   python Runner_02_001.py --script Script_02_001.yaml
#
# Runner loads Script_02_001.yaml which defines a list of steps.
# Each step:
#   - name: registry function name
#   - args: dict of args (supports ${step_name} references)
#   - save: optional output filename under --outdir
#
# Context:
#   - After each step, context[step_name] = returned value
#   - ${step_name} in args is replaced with that value

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import yaml  # pip install pyyaml

# The Functions_02_001.py file must define REGISTRY (dict name -> callable)
from Functions_02_001 import REGISTRY


_REF_RE = re.compile(r"^\$\{([^}]+)\}$")


def _resolve_refs(obj: Any, context: Dict[str, Any]) -> Any:
    """Resolve ${step_name} references recursively."""
    if isinstance(obj, str):
        m = _REF_RE.match(obj.strip())
        if m:
            key = m.group(1)
            if key not in context:
                raise KeyError(f"Unresolved reference: {obj} (missing '{key}' in context)")
            return context[key]
        return obj
    if isinstance(obj, list):
        return [_resolve_refs(x, context) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, context) for k, v in obj.items()}
    return obj


def _is_ndarray(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, np.ndarray)
    except Exception:
        return False


def _save_output(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if _is_ndarray(obj):
        import cv2
        import numpy as np
        arr = obj
        if arr.ndim == 2:
            cv2.imwrite(path, arr)
            return
        if arr.ndim == 3:
            # assume BGR or RGB-ish; write as-is
            cv2.imwrite(path, arr)
            return
        raise ValueError(f"Cannot save ndarray with shape {arr.shape} to {path}")

    # JSON for everything else
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="YAML script with steps")
    ap.add_argument("--outdir", default="out", help="Output directory for saved artifacts")
    args = ap.parse_args()

    with open(args.script, "r", encoding="utf-8") as f:
        script = yaml.safe_load(f)

    steps: List[Dict[str, Any]] = script.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("Script must contain a non-empty 'steps' list.")

    context: Dict[str, Any] = {}
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    for i, step in enumerate(steps, start=1):
        name = step.get("name")
        if not name or name not in REGISTRY:
            raise KeyError(f"Step {i}: unknown function '{name}'. Known: {sorted(REGISTRY.keys())}")

        fn = REGISTRY[name]
        raw_args = step.get("args", {}) or {}
        if not isinstance(raw_args, dict):
            raise ValueError(f"Step {i} '{name}': args must be a dict.")

        call_args = _resolve_refs(raw_args, context)

        # Inject context (so functions can read previous outputs)
        call_args["__context"] = context

        print(f"[{i:02d}] {name}")
        out = fn(**call_args)
        context[name] = out

        # Optional save
        save = step.get("save")
        if isinstance(save, str) and save.strip():
            save_path = os.path.join(outdir, save.strip())
            _save_output(out, save_path)
            print(f"     saved -> {save_path}")

        # Small, helpful type info
        tname = type(out).__name__
        if _is_ndarray(out):
            print(f"     type  -> ndarray shape={out.shape} dtype={out.dtype}")
        elif isinstance(out, (list, dict)):
            print(f"     type  -> {tname} len={len(out)}")
        else:
            print(f"     type  -> {tname}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
