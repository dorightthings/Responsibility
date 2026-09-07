#!/usr/bin/env python3
"""Check dependencies and one synthetic grouped-LR model update; write no files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        import torch
        import qlib
        from qlib.data._libs import rolling  # noqa: F401 -- verify binary import
        from responsibility.training.config import load_experiment_config
        from responsibility.training.engine import (
            _model, build_optimizer, optimizer_metadata, set_global_seed,
        )

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check the NVIDIA driver and cu128 PyTorch "
                "installation; --device cpu is only a CPU diagnostic."
            )
        torch.set_num_threads(1)
        device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
        set_global_seed(0)
        config = load_experiment_config(ROOT / "configs/csi300.yaml")
        model = _model(config, seed=0, device=device)
        optimizer, counts = build_optimizer(model, config)
        prediction = model(
            torch.randn(5, 8, 221, device=device),
            torch.randn(5, 14, device=device),
            torch.randn(5, 8, device=device),
            torch.ones(4, device=device),
        )
        if tuple(prediction.shape) != (5,) or not torch.isfinite(prediction).all():
            raise RuntimeError("model prediction has an invalid shape or nonfinite values")
        prediction.square().mean().backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise RuntimeError("model gradient contains nonfinite values")
        torch.nn.utils.clip_grad_value_(model.parameters(), config.gradient_clip_value)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("model update contains nonfinite values")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {
            "status": "PASS",
            "scope": "synthetic model update only; no dataset or performance evaluation",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "qlib": qlib.__version__,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML")
            },
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda" else None
            ),
            "parameter_counts": counts,
            "optimizer": optimizer_metadata(optimizer),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
