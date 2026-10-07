from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot_location(data: pd.DataFrame, location: str, output: Path) -> None:
    data = data.sort_values("timestamp").copy()
    sessions = data["timestamp"].diff().dt.total_seconds().gt(120).cumsum()
    groups = [part for _, part in data.groupby(sessions)]
    units = data["pressure_unit"].dropna().unique()
    ylabel = f"AMU 4 partial pressure ({units[0]})" if len(units) == 1 else "AMU 4 signal (pressure unit unavailable)"

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(11, max(4.5, len(groups) * 3.8)), squeeze=False, sharey=True
    )
    previous_end = None
    for number, (ax, session) in enumerate(zip(axes[:, 0], groups), start=1):
        session = session.sort_values("timestamp").copy()
        start = session["timestamp"].iloc[0]
        end = session["timestamp"].iloc[-1]
        seconds = (session["timestamp"] - start).dt.total_seconds()
        pressure = session.set_index("timestamp")["pressure"]
        trend = pressure.rolling("15s", center=True, min_periods=1).median()
        session["trend"] = trend.to_numpy()
        segments = seconds.diff().gt(2).cumsum()
        for segment_id in segments.unique():
            mask = segments == segment_id
            ax.plot(
                seconds[mask],
                session.loc[mask, "pressure"],
                color="#87919b",
                linewidth=0.65,
                alpha=0.45,
                label="Raw samples" if segment_id == segments.iloc[0] else None,
            )
            ax.plot(
                seconds[mask],
                session.loc[mask, "trend"],
                color="#1769aa",
                linewidth=1.8,
                label="15-second rolling median" if segment_id == segments.iloc[0] else None,
            )
        for position in range(1, len(session)):
            gap = float(seconds.iloc[position] - seconds.iloc[position - 1])
            if gap > 10:
                left = float(seconds.iloc[position - 1])
                right = float(seconds.iloc[position])
                ax.axvspan(left, right, color="#d1495b", alpha=0.08)
                ax.text(
                    (left + right) / 2,
                    0.97,
                    f"No samples for {gap:.0f} sec",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#7f1d2d",
                )

        duration = float(seconds.iloc[-1])
        major_tick = 300 if duration > 720 else 60
        peak_position = int(session["trend"].to_numpy().argmax())
        peak_time = float(seconds.iloc[peak_position])
        peak_pressure = float(session["trend"].iloc[peak_position])
        ax.scatter([peak_time], [peak_pressure], color="#d1495b", s=25, zorder=4)
        peak_near_right = peak_time > duration * 0.7
        ax.annotate(
            f"Highest 15-s median: {peak_pressure:.2g} {units[0] if len(units) == 1 else ''}\n"
            f"at {int(peak_time // 60)}:{int(peak_time % 60):02d}",
            xy=(peak_time, peak_pressure),
            xytext=(-8 if peak_near_right else 8, 8),
            textcoords="offset points",
            ha="right" if peak_near_right else "left",
            fontsize=8,
            color="#7f1d2d",
        )
        title = f"Data segment {number} — starts {start:%Y-%m-%d %H:%M:%S}"
        if previous_end is not None:
            missing_seconds = (start - previous_end).total_seconds()
            title += f" ({missing_seconds / 60:.1f} min with no samples before this)"
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, max(duration, 1))
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{int(value // 60)}:{int(value % 60):02d}")
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.grid(axis="both", which="major", color="#9aa0a6", alpha=0.55, linewidth=0.6)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        previous_end = end

    fig.supylabel(ylabel)
    axes[-1, 0].set_xlabel("Elapsed time (min:sec; minor ticks every 10 sec)")
    fig.suptitle(f"AMU 4 helium pressure — {location}", fontsize=13)
    fig.tight_layout(rect=(0.04, 0.02, 1, 0.97))
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
