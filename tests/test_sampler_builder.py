"""Contract tests for the portable frozen sampler builder."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from responsibility.preprocessing.samplers import (
    DATASET_SPECS,
    DATA_KEYS,
    LABEL,
    SEGMENTS,
    dump_pickle_new,
    infer_processor_config,
    learn_processor_config,
    market_feature_config,
    reorder_sampler_by_datetime,
    validate_provider,
)


class FrozenConfigurationTest(unittest.TestCase):
    def test_market63_order_matches_frozen_schema(self) -> None:
        fields, names = market_feature_config()
        payload = json.dumps(
            {"fields": fields, "names": names}, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(len(fields), 63)
        self.assertEqual(len(names), 63)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "73fb015824a83023b8132ede72026d98c5e80175e14a45e1bdc1607c45ab3fa0",
        )

    def test_segments_data_keys_label_and_processors_are_frozen(self) -> None:
        self.assertEqual(
            dict(SEGMENTS),
            {
                "train": ("2008-01-01", "2020-03-31"),
                "valid": ("2020-04-01", "2020-06-30"),
                "test": ("2020-07-01", "2022-12-31"),
            },
        )
        self.assertEqual(dict(DATA_KEYS), {"train": "learn", "valid": "infer", "test": "infer"})
        self.assertEqual(LABEL, "Ref($close, -5) / Ref($close, -1) - 1")
        self.assertEqual(
            infer_processor_config(),
            [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
        )
        self.assertEqual(learn_processor_config(), [{"class": "DropnaLabel"}])
        self.assertEqual(DATASET_SPECS["short_csi300"].instruments, "csi300")
        self.assertEqual(
            DATASET_SPECS["short_csi800_direct"].instruments, "csi800"
        )


class _FakeSampler:
    def __init__(self) -> None:
        self.data_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-01"), "B"),
                (pd.Timestamp("2020-01-02"), "B"),
                (pd.Timestamp("2020-01-01"), "A"),
                (pd.Timestamp("2020-01-02"), "A"),
            ],
            names=["datetime", "instrument"],
        )
        self.idx_map = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])

    def get_index(self) -> pd.MultiIndex:
        return self.data_index


class SamplerUtilitiesTest(unittest.TestCase):
    def test_reorder_sampler_groups_dates_then_instruments(self) -> None:
        sampler = reorder_sampler_by_datetime(_FakeSampler())
        self.assertEqual(
            list(sampler.get_index()),
            [
                (pd.Timestamp("2020-01-01"), "A"),
                (pd.Timestamp("2020-01-01"), "B"),
                (pd.Timestamp("2020-01-02"), "A"),
                (pd.Timestamp("2020-01-02"), "B"),
            ],
        )
        np.testing.assert_array_equal(
            sampler.idx_map,
            np.asarray([[0, 1], [0, 0], [1, 1], [1, 0]]),
        )

    def test_provider_validation_is_path_driven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provider = Path(temporary) / "portable_provider"
            required = [
                provider / "calendars/day.txt",
                provider / "instruments/csi300.txt",
                provider / "features/sh000300/close.day.bin",
                provider / "features/sh000300/amount.day.bin",
                provider / "features/sh000905/close.day.bin",
                provider / "features/sh000905/amount.day.bin",
                provider / "features/sh000906/close.day.bin",
                provider / "features/sh000906/amount.day.bin",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            self.assertEqual(
                validate_provider(provider, ["short_csi300"]), provider.resolve()
            )
            with self.assertRaises(FileNotFoundError):
                validate_provider(provider, ["short_csi800_direct"])

    def test_pickle_writer_is_exclusive_and_hashes_written_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.pkl"
            value = {"array": np.arange(5, dtype=np.float32)}
            record = dump_pickle_new(value, path)
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with path.open("rb") as handle:
                loaded = pickle.load(handle)
            np.testing.assert_array_equal(loaded["array"], value["array"])
            with self.assertRaises(FileExistsError):
                dump_pickle_new(value, path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
