"""
Runner.py — Water Geo (Rails-first)

Loads Script.yaml and executes each enabled step by calling Functions.py by name.

Usage:
  python Runner.py Script.yaml
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import dataclasses

import yaml

import Functions  # local file


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python Runner.py Script.yaml")
        return 2

    script_path = sys.argv[1]
    cfg = load_yaml(script_path)

    out_dir = Path(cfg["outputs"]["out_dir"])
    ensure_dir(out_dir)

    ctx: Dict[str, Any] = {"cfg": cfg}

    for step in cfg.get("steps", []):
        if not step.get("enabled", True):
            continue

        name = step["name"]
        fn_name = step["fn"]
        fn = getattr(Functions, fn_name, None)
        if fn is None:
            raise RuntimeError(f"Step {name}: function not found: {fn_name}")

        print(f"\n=== {name} ({fn_name}) ===")
        fn(ctx, cfg)

        # lightweight checkpoint after each step (safe to delete later)
        if cfg["outputs"].get("save_json", True):
            chk = out_dir / f"{name}.json"
            with open(chk, "w", encoding="utf-8") as f:
                json.dump(_jsonable(ctx), f, indent=2)

    print("\nDone.")
    return 0


def _jsonable(x: Any) -> Any:
    """
    Make ctx JSON-safe. This is intentionally simple; refine later.
    """
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if dataclasses.is_dataclass(x):
        return _jsonable(dataclasses.asdict(x))
    if hasattr(x, "__dict__"):
        return _jsonable(vars(x))
    return str(x)


if __name__ == "__main__":
    raise SystemExit(main())



