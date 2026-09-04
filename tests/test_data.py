"""Minimal stdlib tests for portable data-manifest path resolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from responsibility.training.data import resolve_data_paths


class ResolveDataPathsTest(unittest.TestCase):
    def test_mapping_paths_and_provider_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for segment in ("train", "valid", "test"):
                path = root / "samplers/short_csi300" / (segment + ".pkl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            feature_root = root / "responsibility/short_csi300"
            feature_root.mkdir(parents=True)
            (feature_root / "manifest.json").write_text("{}\n", encoding="utf-8")
            scale = root / "direction_scales/short_csi300.json"
            scale.parent.mkdir(parents=True)
            scale.write_text("{}\n", encoding="utf-8")
            provider = root / "qlib/cn_data"
            provider.mkdir(parents=True)
            manifest = {
                "datasets": {
                    "short_csi300": {
                        "samplers": {
                            name: {"path": "samplers/short_csi300/%s.pkl" % name}
                            for name in ("train", "valid", "test")
                        },
                        "responsibility_root": "responsibility/short_csi300",
                        "direction_scale": "direction_scales/short_csi300.json",
                    }
                },
                "qlib_provider": {
                    "path": "qlib/cn_data",
                    "manifest": "qlib/PROVIDER_MANIFEST.json",
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            resolved = resolve_data_paths(root, "short_csi300")
            self.assertEqual(resolved.provider, provider.resolve())
            self.assertEqual(
                resolved.samplers["train"],
                (root / "samplers/short_csi300/train.pkl").resolve(),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
