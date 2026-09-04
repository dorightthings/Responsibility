"""CPU tests for portable responsibility-feature construction."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from responsibility.preprocessing import responsibility_features as features


class SamplerStub:
    """Small pickle-compatible stand-in exposing the Qlib sampler index API."""

    def __init__(self, index: pd.MultiIndex) -> None:
        self._index = index

    def get_index(self) -> pd.MultiIndex:
        return self._index


class ResponsibilityFeatureTest(unittest.TestCase):
    def test_dataset_benchmark_cannot_be_silently_changed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be SH000300"):
            features.build_responsibility_feature_store(
                dataset="short_csi300",
                sampler_dir=Path("unused"),
                qlib_provider=Path("unused"),
                output_root=Path("unused"),
                benchmark="SH000906",
            )

    def test_sampler_metadata_dataset_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sampler_dir = Path(temporary)
            (sampler_dir / "build_metadata.json").write_text(
                json.dumps({"status": "DONE", "dataset": "short_csi800_direct"}),
                encoding="utf-8",
            )
            for segment in features.SEGMENTS:
                (sampler_dir / (segment + ".pkl")).touch()
            with self.assertRaisesRegex(
                features.ResponsibilityFeatureError, "sampler dataset mismatch"
            ):
                features._sampler_paths(
                    sampler_dir, expected_dataset="short_csi300"
                )

    def test_daily_formula_and_invalid_volume(self) -> None:
        size = 180
        positions = np.arange(size, dtype=np.float64)
        market_log_returns = np.zeros(size, dtype=np.float64)
        market_log_returns[1:] = 0.001 + 0.0006 * np.sin(positions[1:] / 4.0)
        market_close = 100.0 * np.exp(np.cumsum(market_log_returns))
        beta = 1.7
        stock_log_returns = np.zeros(size, dtype=np.float64)
        stock_log_returns[1:] = 0.0002 + beta * market_log_returns[1:]
        stock_close = 20.0 * np.exp(np.cumsum(stock_log_returns))
        volume = np.full(size, 1000.0, dtype=np.float64)
        market_returns, market_valid = features._log_returns(market_close)

        state, components = features.compute_daily_features(
            stock_close, volume, market_returns, market_valid
        )
        probe = 90
        self.assertAlmostEqual(float(state[probe, 8]), beta, places=9)
        self.assertTrue(np.allclose(state[probe, :8], components[probe], atol=1e-12))
        expected_c5 = np.sum(beta * market_returns[probe - 4 : probe + 1])
        self.assertAlmostEqual(float(components[probe, 1]), float(expected_c5), places=10)

        halted = volume.copy()
        halted[120] = 0.0
        halted_state, halted_components = features.compute_daily_features(
            stock_close, halted, market_returns, market_valid
        )
        self.assertTrue(np.all(halted_components[120:122] == 0.0))
        self.assertTrue(np.all(np.isnan(halted_state[120:122, :8])))
        self.assertEqual(float(halted_state[120, 13]), 0.0)
        self.assertEqual(float(halted_state[121, 13]), 1.0)

    def test_build_store_binds_exact_sampler_indexes(self) -> None:
        calendar = pd.bdate_range("2019-01-01", periods=230)
        stocks = ("SH600000", "SH600001")
        segment_dates = {
            "train": calendar[125:190],
            "valid": calendar[190:205],
            "test": calendar[205:220],
        }

        market_returns = np.zeros(len(calendar), dtype=np.float64)
        positions = np.arange(len(calendar), dtype=np.float64)
        market_returns[1:] = 0.0004 + 0.006 * np.sin(positions[1:] / 7.0)
        market_close = 100.0 * np.exp(np.cumsum(market_returns))
        frames = []
        for order, instrument in enumerate(("SH000300",) + stocks):
            if instrument == "SH000300":
                close = market_close
            else:
                residual = 0.002 * np.cos(positions / (5.0 + order))
                returns = 0.0002 + (0.8 + 0.3 * order) * market_returns + residual
                returns[0] = 0.0
                close = (20.0 + order) * np.exp(np.cumsum(returns))
            volume = 1000.0 + 2.0 * positions + 20.0 * np.sin(positions / (3.0 + order))
            index = pd.MultiIndex.from_product(
                [[instrument], calendar], names=["instrument", "datetime"]
            )
            frames.append(
                pd.DataFrame({"$close": close, "$volume": volume}, index=index)
            )
        raw_quotes = pd.concat(frames).sort_index()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampler_dir = root / "samplers"
            sampler_dir.mkdir()
            expected_indexes = {}
            for segment, dates in segment_dates.items():
                index = pd.MultiIndex.from_product(
                    [dates, stocks], names=["datetime", "instrument"]
                )
                expected_indexes[segment] = index
                with (sampler_dir / (segment + ".pkl")).open("wb") as handle:
                    pickle.dump(SamplerStub(index), handle)
            provider = root / "provider"
            provider.mkdir()
            output = root / "features"
            with mock.patch.object(
                features,
                "_query_raw_quotes",
                return_value=(calendar, raw_quotes),
            ):
                manifest = features.build_responsibility_feature_store(
                    dataset="short_csi300",
                    sampler_dir=sampler_dir,
                    qlib_provider=provider,
                    output_root=output,
                    chunk_rows=17,
                )

            self.assertEqual(manifest["benchmark"], "SH000300")
            self.assertFalse((output / "train/.raw_state.tmp.npy").exists())
            serialized = json.dumps(manifest)
            self.assertNotIn(str(root), serialized)
            for segment in features.SEGMENTS:
                record = manifest["segments"][segment]
                self.assertEqual(
                    record["index_sha256"],
                    features.multiindex_sha256(expected_indexes[segment]),
                )
                loaded = features.load_feature_segment(
                    output, segment, expected_index=expected_indexes[segment]
                )
                self.assertEqual(loaded.stock_state.dtype, np.float32)
                self.assertEqual(loaded.stock_state.shape[1], 14)
                self.assertEqual(loaded.rule_components.shape[1], 8)
                self.assertTrue(np.isfinite(loaded.stock_state).all())
                self.assertTrue(np.isfinite(loaded.rule_components).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
