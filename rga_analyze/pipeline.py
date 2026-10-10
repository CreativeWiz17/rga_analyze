from __future__ import annotations

import argparse
import hashlib
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
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,?\s*$"
)
DATA_ROW_PREFIX = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}")
ELAPSED_ROW = re.compile(
    r"^(\d+):(\d{2}(?:\.\d+)?)\s*,\s*4\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,?\s*$"
)
PRESSURE_UNIT = re.compile(r'PressureUnits="{1,2}([^"]+)"{1,2}')
FILE_TIMESTAMP = re.compile(r"(\d{8})-(\d{6})")
DATE_DATA_FOLDER = re.compile(r"^\d{1,2}\.\d{1,2}(?:_|$)")
FILE_TIMESTAMP_TOLERANCE = pd.Timedelta(minutes=15)
ISOLATED_SPIKE_FACTOR = 10
ISOLATED_SPIKE_NEIGHBOR_RATIO = 10
MAX_ISOLATED_SPIKE_RUN_LENGTH = 2
MAX_CONNECTED_SAMPLE_GAP_SECONDS = 2


def _location_and_phase(path: Path, root: Path) -> tuple[str, str]:
    parts = path.relative_to(root).parts
    baseline_index = next(
        (
            index
            for index, part in enumerate(parts[:-1])
            if "baseline" in part.casefold()
        ),
        None,
    )
    if baseline_index is None:
        return path.parent.relative_to(root).as_posix(), "spray"

    location = Path(*parts[:baseline_index]).as_posix() or "."
    return location, "baseline"


def read_helium_file(path: Path, root: Path) -> tuple[list[dict], str | None]:
    """Read AMU-4 rows and the pressure unit from one RGA export."""
    rows, unit, _, _ = _read_helium_file(path, root)
    return rows, unit


def _read_helium_file(
    path: Path, root: Path
) -> tuple[list[dict], str | None, int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    unit_match = PRESSURE_UNIT.search(text)
    unit = unit_match.group(1) if unit_match else None
    location, phase = _location_and_phase(path, root)
    file_match = FILE_TIMESTAMP.search(path.name)
    file_timestamp = (
        pd.to_datetime("".join(file_match.groups()), format="%Y%m%d%H%M%S")
        if file_match
        else None
    )
    rows = []
    malformed_rows = 0
    timestamp_mismatches = 0

    for line in text.splitlines():
        line = line.strip()
        match = DATA_ROW.fullmatch(line)
        elapsed_match = ELAPSED_ROW.fullmatch(line)
        if match:
            if float(match.group(2)) == 4:
                timestamp = pd.to_datetime(match.group(1), format="mixed")
                if (
                    file_timestamp is not None
                    and abs(timestamp - file_timestamp) > FILE_TIMESTAMP_TOLERANCE
                ):
                    timestamp_mismatches += 1
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "pressure": float(match.group(3)),
                        "pressure_unit": unit,
                        "location": location,
                        "phase": phase,
                        "source_file": path.relative_to(root).as_posix(),
                    }
                )
        elif DATA_ROW_PREFIX.match(line):
            malformed_rows += 1
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
                    "location": location,
                    "phase": phase,
                    "source_file": path.relative_to(root).as_posix(),
                }
            )

    return rows, unit, malformed_rows, timestamp_mismatches


def plot_location(
    data: pd.DataFrame,
    location: str,
    output: Path,
    fit_degree: int | None = None,
) -> None:
    if fit_degree not in (None, 1, 2):
        raise ValueError("fit_degree must be None, 1 (linear), or 2 (quadratic)")

    data = data.sort_values("timestamp").copy()
    spike_mask = _isolated_pressure_spike_mask(data)
    data.loc[spike_mask, "pressure"] = np.nan
    units = data["pressure_unit"].dropna().unique()
    ylabel = f"Helium pressure ({units[0]})" if len(units) == 1 else "Helium pressure (unit unavailable)"
    location_label = re.sub(r"^\d+\.", "", location.rsplit("/", 1)[-1]).replace("_", " ")

    spray_start = (
        data["spray_start"].dropna().iloc[0]
        if "spray_start" in data and data["spray_start"].notna().any()
        else data["timestamp"].iloc[0]
    )
    start = pd.Timestamp(spray_start)
    seconds = (data["timestamp"] - start).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    phases = data["phase"] if "phase" in data else pd.Series("spray", index=data.index)
    has_baseline = (phases == "baseline").any()
    phase_colors = {"baseline": "#73777b", "spray": "#1769aa"}
    for phase in ("baseline", "spray"):
        phase_mask = phases == phase
        if not phase_mask.any():
            continue
        phase_seconds = seconds[phase_mask]
        phase_data = data.loc[phase_mask]
        phase_gaps = phase_seconds.diff()
        segment_ids = phase_gaps.gt(MAX_CONNECTED_SAMPLE_GAP_SECONDS).cumsum()
        for _, segment_indices in segment_ids.groupby(segment_ids).groups.items():
            ax.plot(
                phase_seconds.loc[segment_indices],
                phase_data.loc[segment_indices, "pressure"],
                color=phase_colors[phase],
                linewidth=0.75,
            )

    for position in range(1, len(data)):
        gap_seconds = float(seconds.iloc[position] - seconds.iloc[position - 1])
        if gap_seconds <= MAX_CONNECTED_SAMPLE_GAP_SECONDS:
            continue
        left = float(seconds.iloc[position - 1])
        right = float(seconds.iloc[position])
        ax.axvspan(left, right, color="#c62828", alpha=0.14, zorder=0)
        ax.axvline(left, color="#c62828", linewidth=0.8, alpha=0.8, linestyle="--")
        ax.axvline(right, color="#c62828", linewidth=0.8, alpha=0.8, linestyle="--")

    if fit_degree is not None:
        fit_data = (
            data.loc[(data["phase"] == "spray") & data["pressure"].notna()]
            if "phase" in data
            else data.loc[data["pressure"].notna()]
        )
        if len(fit_data) < fit_degree + 1:
            raise ValueError("Not enough spray samples to calculate the requested fit")
        fit_minutes_data = (fit_data["timestamp"] - start).dt.total_seconds() / 60
        fit = np.polynomial.Polynomial.fit(
            fit_minutes_data.to_numpy(dtype=float),
            fit_data["pressure"].to_numpy(dtype=float),
            fit_degree,
        )
        fit_minutes = np.linspace(
            float(fit_minutes_data.min()),
            float(fit_minutes_data.max()),
            500,
        )
        fit_name = "linear" if fit_degree == 1 else "quadratic"
        ax.plot(
            fit_minutes * 60,
            fit(fit_minutes),
            color="#e67e22",
            linewidth=2.2,
            zorder=4,
        )
        coefficients = fit.convert().coef
        equation = f"P(t) = {coefficients[0]:.3e}"
        for power, coefficient in enumerate(coefficients[1:], start=1):
            sign = "+" if coefficient >= 0 else "-"
            term = f" {sign} {abs(coefficient):.3e} t"
            if power > 1:
                term += f"^{power}"
            equation += term
        pressures = fit_data["pressure"].to_numpy(dtype=float)
        residual_sum_squares = float(np.sum((pressures - fit(fit_minutes_data)) ** 2))
        total_sum_squares = float(np.sum((pressures - pressures.mean()) ** 2))
        r_squared = (
            1 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else None
        )
        fit_summary = f"{equation}\nR^2 = {r_squared:.3f}" if r_squared is not None else f"{equation}\nR^2 = undefined"
        ax.text(
            0.02,
            0.98,
            fit_summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#9aa0a6",
                "alpha": 0.9,
            },
            zorder=6,
        )

    peak_data = (
        data.loc[(data["phase"] == "spray") & data["pressure"].notna()]
        if "phase" in data and (data["phase"] == "spray").any()
        else data.loc[data["pressure"].notna()]
    )
    if not peak_data.empty:
        peak_position = int(peak_data["pressure"].to_numpy().argmax())
        peak_time = float(
            (peak_data["timestamp"].iloc[peak_position] - start).total_seconds()
        )
        peak_pressure = float(peak_data["pressure"].iloc[peak_position])
        ax.scatter(
            [peak_time],
            [peak_pressure],
            color="#c62828",
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=5,
        )

    minimum_time = float(seconds.min())
    maximum_time = float(seconds.max())
    duration = maximum_time - minimum_time
    major_tick = next(
        (
            tick
            for tick in (
                60,
                120,
                300,
                600,
                900,
                1800,
                3600,
                7200,
                10800,
                21600,
                43200,
                86400,
            )
            if duration / tick <= 8
        ),
        86400,
    )
    minor_tick = max(30, major_tick // 5)
    ax.set_xlim(min(minimum_time, 0), max(maximum_time, 1))
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_elapsed))
    if has_baseline:
        ax.set_xlabel("Time relative to spray start (min:s; baseline is negative)")
    else:
        ax.set_xlabel("Elapsed time from first sample (min:s; no baseline data)")
    ax.set_ylabel(ylabel)
    title = f"Helium pressure - {location_label}"
    if fit_degree is not None:
        title += f" ({fit_name} fit)"
    if spike_mask.any():
        title += f" ({int(spike_mask.sum())} isolated spike sample(s) omitted)"
    ax.set_title(title, fontsize=14, pad=12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.grid(axis="both", which="major", color="#9aa0a6", alpha=0.5, linewidth=0.6)
    ax.grid(axis="x", which="minor", color="#c5c9cc", alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_elapsed(value: float, _: object) -> str:
    sign = "-" if value < 0 else ""
    elapsed = int(round(abs(value)))
    return f"{sign}{elapsed // 60}:{elapsed % 60:02d}"


def _isolated_pressure_spike_mask(data: pd.DataFrame) -> np.ndarray:
    pressures = data["pressure"].to_numpy(dtype=float)
    timestamps = data["timestamp"]
    phases = (
        data["phase"].to_numpy()
        if "phase" in data
        else np.full(len(data), "spray")
    )
    spike_mask = np.zeros(len(data), dtype=bool)

    for run_length in range(MAX_ISOLATED_SPIKE_RUN_LENGTH, 0, -1):
        for start in range(1, len(data) - run_length):
            end = start + run_length - 1
            if spike_mask[start : end + 1].any():
                continue
            if not np.all(phases[start - 1 : end + 2] == phases[start]):
                continue

            neighborhood_indices = (start - 1, end + 1)
            neighborhood = np.abs(pressures[list(neighborhood_indices)])
            spike_run = np.abs(pressures[start : end + 1])
            if not np.isfinite(np.concatenate((neighborhood, spike_run))).all():
                continue

            gaps = [
                (
                    timestamps.iloc[index + 1] - timestamps.iloc[index]
                ).total_seconds()
                for index in range(start - 1, end + 1)
            ]
            if any(gap > MAX_CONNECTED_SAMPLE_GAP_SECONDS for gap in gaps):
                continue

            neighbor_low = float(neighborhood.min())
            neighbor_high = float(neighborhood.max())
            neighbors_are_similar = (
                neighbor_low == 0 and neighbor_high == 0
            ) or neighbor_high <= ISOLATED_SPIKE_NEIGHBOR_RATIO * neighbor_low
            if (
                neighbors_are_similar
                and np.all(spike_run > ISOLATED_SPIKE_FACTOR * neighbor_high)
            ):
                spike_mask[start : end + 1] = True

    return spike_mask


def analyze(root: Path, output: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")

    locations: dict[str, list[dict]] = {}
    skipped_files = []
    malformed_rows_excluded = 0
    timestamp_mismatches_excluded = 0
    paths = sorted(root.rglob("*.csv"))
    baseline_digests: dict[tuple[str, str], set[bytes]] = {}
    for path in paths:
        location, phase = _location_and_phase(path, root)
        if phase == "baseline":
            key = (location, path.name)
            digest = hashlib.sha256(path.read_bytes()).digest()
            baseline_digests.setdefault(key, set()).add(digest)

    duplicate_baseline_copies = 0
    for path in paths:
        location, phase = _location_and_phase(path, root)
        if phase == "spray":
            key = (location, path.name)
            digest = hashlib.sha256(path.read_bytes()).digest()
            if digest in baseline_digests.get(key, set()):
                duplicate_baseline_copies += 1
                continue

        rows, _, malformed_rows, timestamp_mismatches = _read_helium_file(path, root)
        malformed_rows_excluded += malformed_rows
        timestamp_mismatches_excluded += timestamp_mismatches
        if rows:
            for row in rows:
                locations.setdefault(row["location"], []).append(row)
        else:
            skipped_files.append(path.relative_to(root).as_posix())

    folder_match = DATE_DATA_FOLDER.match(root.name)
    output_date = folder_match.group().rstrip("_") if folder_match else root.name
    date_output = output / output_date
    for location, rows in sorted(locations.items()):
        data = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        spray_rows = data.loc[data["phase"] == "spray"]
        if spray_rows.empty:
            spray_start = data["timestamp"].min()
        else:
            first_spray_file = min(
                spray_rows["source_file"].unique(),
                key=lambda source_file: Path(source_file).name,
            )
            spray_start = spray_rows.loc[
                spray_rows["source_file"] == first_spray_file,
                "timestamp",
            ].min()
        data["spray_start"] = spray_start
        data["elapsed_minutes"] = (
            data["timestamp"] - spray_start
        ).dt.total_seconds() / 60
        isolated_spikes = _isolated_pressure_spike_mask(data)
        usable_spray_samples = int(
            ((data["phase"] == "spray") & ~isolated_spikes).sum()
        )
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", location)
        plot_dir = date_output / "plots" / "helium_by_location"
        fit_dir = date_output / "plots" / "helium_by_location_fits"
        csv_dir = date_output / "processed_data" / "amu4_by_location"
        csv_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_dir / f"{filename}.csv", index=False)
        plot_location(data, location, plot_dir / f"{filename}.png")
        for degree, name in ((1, "linear"), (2, "quadratic")):
            if usable_spray_samples >= degree + 1:
                plot_location(
                    data,
                    location,
                    fit_dir / name / f"{filename}.png",
                    fit_degree=degree,
                )

    return {
        "output_folder": output_date,
        "locations": len(locations),
        "samples": sum(map(len, locations.values())),
        "duplicate_baseline_copies_excluded": duplicate_baseline_copies,
        "malformed_rows_excluded": malformed_rows_excluded,
        "rows_outside_file_time_window_excluded": timestamp_mismatches_excluded,
        "files_without_amu4_time_series": skipped_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AMU-4 pressure by RGA location folder.")
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        help="Data folder to analyze; repeat to include multiple folders. "
        "Defaults to sibling folders named with a month.day prefix.",
    )
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()
    roots = args.root
    if roots is None:
        roots = sorted(
            path
            for path in Path.cwd().iterdir()
            if path.is_dir() and DATE_DATA_FOLDER.match(path.name)
        )
        if not roots:
            parser.error(
                "No data folders found; provide one or more --root paths."
            )
    for root in roots:
        print(f"Input: {root}")
        print(json.dumps(analyze(root, args.output), indent=2))
