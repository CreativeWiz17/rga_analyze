import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rga_analyze.pipeline import (
    _isolated_pressure_spike_mask,
    _read_helium_file,
    plot_location,
    read_helium_file,
)


class ReadHeliumFileTests(unittest.TestCase):
    def test_rejects_malformed_and_filename_mismatched_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location = root / "08-2.small_top_stm_window"
            baseline = location / "Baseline"
            baseline.mkdir(parents=True)
            source = baseline / "MassSpecData-02115-20261008-145005.csv"
            source.write_text(
                "\n".join(
                    [
                        'PressureUnits="Torr"',
                        '2026/10/07 20:30:48.975, 4.000, 8.7165000"',
                        "2026/10/07 20:30:49.183, 4.000, 8.0e-11,",
                        "2026/10/08 14:50:05.642, 4.000, 2.0e-11,",
                        "2026/10/08 14:50:06.000, 3.000, 1.0e-10,",
                    ]
                ),
                encoding="utf-8",
            )

            rows, unit, malformed_rows, timestamp_mismatches = _read_helium_file(
                source, root
            )
            public_rows, public_unit = read_helium_file(source, root)

        self.assertEqual(unit, "Torr")
        self.assertEqual(public_unit, unit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(public_rows, rows)
        self.assertEqual(rows[0]["pressure"], 2.0e-11)
        self.assertEqual(malformed_rows, 1)
        self.assertEqual(timestamp_mismatches, 1)


class IsolatedPressureSpikeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = pd.date_range("2026-10-08", periods=5, freq="s")

    def test_detects_single_sample_spike_between_similar_readings(self) -> None:
        data = pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "pressure": [1e-11, 1.1e-11, 4e-9, 0.9e-11, 1e-11],
                "phase": ["baseline"] * 5,
            }
        )

        np.testing.assert_array_equal(
            _isolated_pressure_spike_mask(data),
            [False, False, True, False, False],
        )

    def test_detects_two_sample_spike_run_bounded_by_ordinary_readings(self) -> None:
        data = pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "pressure": [1e-11, 1.04e-11, 2.19e-9, 4.16e-9, 1.88e-12],
                "phase": ["baseline"] * 5,
            }
        )

        np.testing.assert_array_equal(
            _isolated_pressure_spike_mask(data),
            [False, False, True, True, False],
        )

    def test_keeps_sustained_high_readings(self) -> None:
        data = pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "pressure": [1e-11, 4e-9, 4e-9, 4e-9, 1e-11],
                "phase": ["spray"] * 5,
            }
        )

        self.assertFalse(_isolated_pressure_spike_mask(data).any())

    def test_does_not_detect_across_phase_changes_or_time_gaps(self) -> None:
        data = pd.DataFrame(
            {
                "timestamp": [
                    self.timestamps[0],
                    self.timestamps[1],
                    self.timestamps[2] + pd.Timedelta(seconds=3),
                    self.timestamps[3] + pd.Timedelta(seconds=3),
                    self.timestamps[4] + pd.Timedelta(seconds=3),
                ],
                "pressure": [1e-11, 1e-11, 4e-9, 1e-11, 1e-11],
                "phase": ["baseline", "baseline", "spray", "spray", "spray"],
            }
        )

        self.assertFalse(_isolated_pressure_spike_mask(data).any())


class LinearPhaseTrendTests(unittest.TestCase):
    def test_plots_and_reports_separate_baseline_and_spray_slopes(self) -> None:
        spray_start = pd.Timestamp("2026-10-08 15:00:00")
        timestamps = [
            spray_start - pd.Timedelta(minutes=9),
            spray_start - pd.Timedelta(minutes=8),
            spray_start - pd.Timedelta(minutes=7),
            spray_start,
            spray_start + pd.Timedelta(minutes=1),
            spray_start + pd.Timedelta(minutes=2),
        ]
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "pressure": [
                    1.0e-10,
                    1.1e-10,
                    1.2e-10,
                    2.0e-10,
                    2.2e-10,
                    2.4e-10,
                ],
                "pressure_unit": ["Torr"] * 6,
                "phase": ["baseline"] * 3 + ["spray"] * 3,
                "spray_start": [spray_start] * 6,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "fit.png"
            try:
                with patch("rga_analyze.pipeline.plt.close"):
                    plot_location(data, "test_location", output, fit_degree=1)
                axes = plt.gcf().axes[0]
                trend_lines = {
                    line.get_label(): line
                    for line in axes.lines
                    if line.get_label() in {"Baseline trend", "Spray trend"}
                }

                self.assertTrue(output.is_file())
                self.assertEqual(set(trend_lines), {"Baseline trend", "Spray trend"})
                baseline_slope = np.polyfit(
                    trend_lines["Baseline trend"].get_xdata() / 60,
                    trend_lines["Baseline trend"].get_ydata(),
                    1,
                )[0]
                spray_slope = np.polyfit(
                    trend_lines["Spray trend"].get_xdata() / 60,
                    trend_lines["Spray trend"].get_ydata(),
                    1,
                )[0]
                self.assertAlmostEqual(baseline_slope, 1.0e-11, places=22)
                self.assertAlmostEqual(spray_slope, 2.0e-11, places=22)
                summary = "\n".join(text.get_text() for text in axes.texts)
                self.assertIn("Baseline: slope = 1.000e-11 Torr/min (n=3", summary)
                self.assertIn("Spray: slope = 2.000e-11 Torr/min (n=3", summary)
            finally:
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
