import tempfile
import unittest
from pathlib import Path

import numpy as np

from rga_analyze.pipeline import (
    _pressure_symlog_linthresh,
    _read_helium_file,
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


class PressureScaleTests(unittest.TestCase):
    def test_uses_symlog_for_isolated_large_pressure_peak(self) -> None:
        pressures = np.concatenate((np.full(1000, 1e-11), [4e-9]))

        self.assertEqual(_pressure_symlog_linthresh(pressures), 1e-11)

    def test_keeps_linear_scale_for_narrow_pressure_range(self) -> None:
        pressures = np.array([-2e-11, -1e-11, 1e-11, 2e-11])

        self.assertIsNone(_pressure_symlog_linthresh(pressures))


if __name__ == "__main__":
    unittest.main()
