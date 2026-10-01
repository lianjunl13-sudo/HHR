#!/usr/bin/env python3
"""Check the minimum HHR build and runtime environment."""

from __future__ import annotations

import importlib
import shutil
import sys


MODULES = ("torch", "transformers", "triton", "numpy")
TOOLS = ("cmake", "ninja", "nvcc")


def main() -> None:
    failures = []
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or later is required")

    loaded = {}
    for name in MODULES:
        try:
            loaded[name] = importlib.import_module(name)
        except ImportError:
            failures.append(f"missing Python module: {name}")

    for name in TOOLS:
        if shutil.which(name) is None:
            failures.append(f"missing build tool: {name}")

    torch = loaded.get("torch")
    if torch is not None and not torch.cuda.is_available():
        failures.append("CUDA is not available through PyTorch")

    if failures:
        raise SystemExit("\n".join(failures))

    versions = [
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ]
    for name, module in loaded.items():
        versions.append(f"{name}={getattr(module, '__version__', 'unknown')}")
    print("HHR environment check passed: " + ", ".join(versions))


if __name__ == "__main__":
    main()
