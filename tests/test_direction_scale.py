"""CPU tests for train-only responsibility direction scaling."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from responsibility.preprocessing.direction_scale import (
    build_direction_scale,
    estimate_direction_scale,
    initial_directions,
)
from responsibility.preprocessing.responsibility_features import (
    SCHEMA_VERSION,
    multiindex_sha256,
)


class DirectionSamplerStub:
    def __init__(self, index: pd.MultiIndex) -> None:
        self._index = index

    def get_index(self) -> pd.MultiIndex:
        return self._index


class DirectionScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3)
        instruments = ["SH600000", "SH600001", "SH600002"] * 2
        self.index = pd.MultiIndex.from_arrays(
            [dates, instruments], names=["datetime", "instrument"]
        )
        self.components = np.asarray(
            [
                [1, 2, 4, 8, 2, 3, 5, 9],
                [2, 4, 5, 7, 3, 5, 8, 10],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [2, 5, 7, 10, 1, 4, 7, 11],
                [4, 7, 8, 13, 3, 6, 10, 14],
                [6, 8, 11, 15, 5, 9, 12, 18],
            ],
            dtype=np.float32,
        )

    def test_estimate_matches_manual_daily_centering(self) -> None:
        result = estimate_direction_scale(self.index, self.components)
        square_sum = np.zeros(4, dtype=np.float64)
        count = 0
        for start, stop in ((0, 3), (3, 6)):
            values = self.components[start:stop]
            valid = np.any(values != 0.0, axis=1)
            directions = initial_directions(values[valid])
            centered = directions - directions.mean(axis=0, keepdims=True)
            square_sum += np.square(centered).sum(axis=0)
            count += int(valid.sum())
        expected = np.sqrt(square_sum / count)
        self.assertTrue(np.allclose(result["direction_scale"], expected, atol=0.0))
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["days"], 2)

    def test_build_uses_train_index_and_writes_no_machine_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampler_dir = root / "samplers"
            sampler_dir.mkdir()
            with (sampler_dir / "train.pkl").open("wb") as handle:
                pickle.dump(DirectionSamplerStub(self.index), handle)

            feature_root = root / "features"
            (feature_root / "train").mkdir(parents=True)
            np.save(
                str(feature_root / "train/stock_state.npy"),
                np.zeros((len(self.index), 14), dtype=np.float32),
                allow_pickle=False,
            )
            np.save(
                str(feature_root / "train/rule_components.npy"),
                self.components,
                allow_pickle=False,
            )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": "short_csi300",
                "segments": {
                    "train": {
                        "rows": len(self.index),
                        "index_sha256": multiindex_sha256(self.index),
                        "stock_state": {"path": "train/stock_state.npy"},
                        "rule_components": {"path": "train/rule_components.npy"},
                    }
                },
            }
            (feature_root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = root / "scales/short_csi300.json"
            payload = build_direction_scale(
                dataset="short_csi300",
                sampler_dir=sampler_dir,
                feature_root=feature_root,
                output=output,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(payload["source_segment"], "train")
            self.assertEqual(payload["count"], 5)
            self.assertNotIn(str(root), output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                build_direction_scale(
                    dataset="short_csi300",
                    sampler_dir=sampler_dir,
                    feature_root=feature_root,
                    output=output,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
