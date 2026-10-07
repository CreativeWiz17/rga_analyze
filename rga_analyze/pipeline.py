from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd


DATA_ROW = re.compile(
    r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
ELAPSED_ROW = re.compile(
    r"^(\d+):(\d{2}(?:\.\d+)?)\s*,\s*4\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
PRESSURE_UNIT = re.compile(r'PressureUnits="{1,2}([^"]+)"{1,2}')
FILE_TIMESTAMP = re.compile(r"(\d{8})-(\d{6})")


def read_helium_file(path: Path, root: Path) -> tuple[list[dict], str | None]:
    """Read AMU-4 rows and the pressure unit from one RGA export."""
    text = path.read_text(encoding="utf-8", errors="replace")
    unit_match = PRESSURE_UNIT.search(text)
    unit = unit_match.group(1) if unit_match else None
    file_match = FILE_TIMESTAMP.search(path.name)
    file_timestamp = (
        pd.to_datetime("".join(file_match.groups()), format="%Y%m%d%H%M%S")
        if file_match
        else None
    )
    rows = []

    for line in text.splitlines():
        line = line.strip()
        match = DATA_ROW.match(line)
        elapsed_match = ELAPSED_ROW.match(line)
        if match and float(match.group(2)) == 4:
            rows.append(
                {
                    "timestamp": pd.to_datetime(match.group(1), format="mixed"),
                    "pressure": float(match.group(3)),
                    "pressure_unit": unit,
                    "location": path.parent.relative_to(root).as_posix(),
                    "source_file": path.relative_to(root).as_posix(),
                }
            )
        elif elapsed_match:
            if file_timestamp is None:
                raise ValueError(f"Clock-time data has no timestamp in its filename: {path}")
            minute = int(elapsed_match.group(1))
            second = float(elapsed_match.group(2))
            same_hour = file_timestamp.replace(minute=minute, second=0) + pd.to_timedelta(second, unit="s")
            candidates = [same_hour + pd.Timedelta(hours=offset) for offset in (-1, 0, 1)]
            timestamp = min(candidates, key=lambda value: abs(value - file_timestamp))
            rows.append(
                {
                    "timestamp": timestamp,
                    "pressure": float(elapsed_match.group(3)),
                    "pressure_unit": unit,
                    "location": path.parent.relative_to(root).as_posix(),
                    "source_file": path.relative_to(root).as_posix(),
                }
            )

    return rows, unit


def plot_location(
    data: pd.DataFrame,
    location: str,
    output: Path,
    fit_degree: int | None = None,
) -> None:
    if fit_degree not in (None, 1, 2):
        raise ValueError("fit_degree must be None, 1 (linear), or 2 (quadratic)")

    data = data.sort_values("timestamp").copy()
    units = data["pressure_unit"].dropna().unique()
    ylabel = f"Helium pressure ({units[0]})" if len(units) == 1 else "Helium pressure (unit unavailable)"
    location_label = re.sub(r"^\d+\.", "", location.rsplit("/", 1)[-1]).replace("_", " ")

    start = data["timestamp"].iloc[0]
    seconds = (data["timestamp"] - start).dt.total_seconds()
    segments = seconds.diff().gt(2).cumsum()
    duration = float(seconds.iloc[-1])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for segment_id in segments.unique():
        mask = segments == segment_id
        ax.plot(
            seconds[mask],
            data.loc[mask, "pressure"],
            color="#1769aa",
            linewidth=0.75,
        )

    if fit_degree is not None:
        fit = np.polynomial.Polynomial.fit(
            seconds.to_numpy(dtype=float),
            data["pressure"].to_numpy(dtype=float),
            fit_degree,
        )
        fit_seconds = np.linspace(0, duration, 500)
        fit_name = "linear" if fit_degree == 1 else "quadratic"
        ax.plot(
            fit_seconds,
            fit(fit_seconds),
            color="#e67e22",
            linewidth=2.2,
            zorder=4,
        )

    gaps = seconds.diff()
    for position in range(1, len(data)):
        gap_seconds = float(gaps.iloc[position])
        if gap_seconds <= 2:
            continue
        left = float(seconds.iloc[position - 1])
        right = float(seconds.iloc[position])
        ax.axvspan(left, right, color="#c62828", alpha=0.14, zorder=0)
        ax.axvline(left, color="#c62828", linewidth=0.8, alpha=0.8, linestyle="--")
        ax.axvline(right, color="#c62828", linewidth=0.8, alpha=0.8, linestyle="--")

    peak_position = int(data["pressure"].to_numpy().argmax())
    peak_time = float(seconds.iloc[peak_position])
    peak_pressure = float(data["pressure"].iloc[peak_position])
    ax.scatter(
        [peak_time],
        [peak_pressure],
        color="#c62828",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=5,
    )

    major_tick = 300 if duration > 720 else 120 if duration > 360 else 60
    minor_tick = 30 if duration > 720 else 10
    ax.set_xlim(0, max(duration, 1))
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value // 60)}:{int(value % 60):02d}")
    )
    ax.set_xlabel("Elapsed time (min:s)")
    ax.set_ylabel(ylabel)
    title = f"Helium pressure — {location_label}"
    if fit_degree is not None:
        title += f" ({fit_name} fit)"
    ax.set_title(title, fontsize=14, pad=12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.grid(axis="both", which="major", color="#9aa0a6", alpha=0.5, linewidth=0.6)
    ax.grid(axis="x", which="minor", color="#c5c9cc", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(root: Path, output: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")

    folders: dict[str, list[dict]] = {}
    skipped_files = []
    for path in sorted(root.rglob("*.csv")):
        rows, _ = read_helium_file(path, root)
        if rows:
            folders.setdefault(rows[0]["location"], []).extend(rows)
        else:
            skipped_files.append(path.relative_to(root).as_posix())

    plot_dir = output / "plots" / "helium_by_location"
    fit_dir = output / "plots" / "helium_by_location_fits"
    csv_dir = output / "processed_data" / "amu4_by_location"
    for location, rows in sorted(folders.items()):
        data = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        data["elapsed_minutes"] = (
            data["timestamp"] - data["timestamp"].iloc[0]
        ).dt.total_seconds() / 60
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", location)
        csv_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_dir / f"{filename}.csv", index=False)
        plot_location(data, location, plot_dir / f"{filename}.png")
        plot_location(data, location, fit_dir / "linear" / f"{filename}.png", fit_degree=1)
        plot_location(data, location, fit_dir / "quadratic" / f"{filename}.png", fit_degree=2)

    return {
        "locations": len(folders),
        "samples": sum(map(len, folders.values())),
        "files_without_amu4_time_series": skipped_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AMU-4 pressure by RGA location folder.")
    parser.add_argument("--root", type=Path, default=Path("10.16_data"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), indent=2))
