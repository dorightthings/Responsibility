"""Small integration tests for manifests, private archives and progress export."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = REPOSITORY_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location("test_" + path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortableDataToolsTest(unittest.TestCase):
    def _make_data_tree(self, root: Path) -> Path:
        for dataset in ("short_csi300", "short_csi800_direct"):
            sampler = root / "samplers" / dataset
            sampler.mkdir(parents=True)
            for segment in ("train", "valid", "test"):
                (sampler / (segment + ".pkl")).write_bytes(b"trusted-test-pickle")
            feature = root / "responsibility" / dataset
            feature.mkdir(parents=True)
            (feature / "manifest.json").write_text("{}\n", encoding="utf-8")
            for segment in ("train", "valid", "test"):
                directory = feature / segment
                directory.mkdir()
                np.save(str(directory / "stock_state.npy"), np.zeros((2, 14), np.float32))
                np.save(
                    str(directory / "rule_components.npy"),
                    np.zeros((2, 8), np.float32),
                )
            scale = root / "direction_scales" / (dataset + ".json")
            scale.parent.mkdir(parents=True, exist_ok=True)
            scale.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "source_segment": "train",
                        "direction_scale": [1.0, 1.0, 1.0, 1.0],
                    }
                ),
                encoding="utf-8",
            )
        provider = root / "source/qlib_bin"
        anchors = [
            provider / "calendars/day.txt",
            provider / "instruments/csi300.txt",
            provider / "instruments/csi800.txt",
            provider / "features/sh000300/close.day.bin",
            provider / "features/sh000906/close.day.bin",
        ]
        for path in anchors:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"anchor")
        return provider

    def test_manifest_and_archive_are_relative_and_directly_extractable(self) -> None:
        manifest_module = load_script("build_data_manifest.py")
        package_module = load_script("package_data.py")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root = base / "data"
            data_root.mkdir()
            provider = self._make_data_tree(data_root)
            args = argparse.Namespace(
                data_root=data_root,
                provider=provider,
                datasets=["short_csi300", "short_csi800_direct"],
                data_version="unit-test-v1",
                archive_url="",
                archive_sha256="",
                source_url="https://example.invalid/provider.tar.gz",
                source_archive_sha256="0" * 64,
                skip_file_sha256=True,
            )
            payload = manifest_module.build_manifest(args)
            manifest_module.write_json_new(data_root / "manifest.json", payload)
            encoded = json.dumps(payload)
            self.assertNotIn(str(base), encoded)
            self.assertEqual(payload["qlib_provider"]["path"], "source/qlib_bin")

            archive = base / "private-data.tar.gz"
            self.assertEqual(
                package_module.main(
                    [
                        "--data-root",
                        str(data_root),
                        "--output",
                        str(archive),
                    ]
                ),
                0,
            )
            with tarfile.open(str(archive), "r:gz") as handle:
                names = set(handle.getnames())
            self.assertIn("manifest.json", names)
            self.assertIn("samplers/short_csi300/train.pkl", names)
            self.assertIn("source/qlib_bin/calendars/day.txt", names)
            self.assertFalse(any(name.startswith("data/") for name in names))
            self.assertTrue(Path(str(archive) + ".sha256").is_file())


class ProgressExportTest(unittest.TestCase):
    def test_export_removes_absolute_run_directory(self) -> None:
        module = load_script("export_progress.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary"
            summary.mkdir()
            with (summary / "per_seed.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["dataset", "seed", "run_dir", "IC"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "short_csi300",
                        "seed": "0",
                        "run_dir": "/private/machine/path",
                        "IC": "0.1",
                    }
                )
            (summary / "summary.csv").write_text(
                "dataset,IC_mean\nshort_csi300,0.1\n", encoding="utf-8"
            )
            (summary / "summary.md").write_text("# Result\n", encoding="utf-8")
            self.assertEqual(
                module.main(
                    [
                        "--summary-dir",
                        str(summary),
                        "--experiment-id",
                        "unit_test",
                        "--machine",
                        "machine_a",
                        "--results-root",
                        str(root / "results"),
                    ]
                ),
                0,
            )
            output = root / "results/unit_test"
            self.assertNotIn("run_dir", (output / "per_seed.csv").read_text())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["machine"], "machine_a")
            self.assertNotIn(str(root), json.dumps(metadata))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
