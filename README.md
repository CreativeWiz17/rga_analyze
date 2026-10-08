# RGA helium plots

Plots AMU-4 pressure by location. Each input folder is analyzed independently and named from its month.day prefix, for example `10.06_data` writes to `analysis_outputs/10.06/`.

## Run

Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

Then, from this project folder, run:

```bash
python -m rga_analyze
```

By default, scans sibling data folders whose names start with a month and day, such as `10.06_data` and `10.7_deep_data`. Each input folder is processed independently and its output is grouped under `analysis_outputs/<month.day>/`.

To process specific folders instead, pass `--root` once per folder. Each one still runs separately:

```bash
python -m rga_analyze --root path/to/data --root path/to/another_data --output path/to/results
```

## Results

- Results are grouped under `analysis_outputs/<month.day>/`, based on the input folder name.
- `plots/helium_by_location/`: raw, unfitted time-series graphs, one per location.
- `plots/helium_by_location_fits/linear/`: raw readings with a straight-line fit to the spray phase, its equation, and R².
- `plots/helium_by_location_fits/quadratic/`: raw readings with a second-degree (quadratic) fit to the spray phase, its equation, and R².
- `processed_data/amu4_by_location/`: the AMU-4 samples used for each graph, with timestamps, phase, source-file names, spray start, and elapsed time.

The run summary lists any CSV that did not contain usable chronological AMU-4 samples. Those files are left out rather than guessed into the plot.

## How to read the plots

- Each plot shows raw helium pressure. There is no rolling median or smoothing.
- CSVs in a nested folder whose name contains `Baseline` are shown as the null baseline. If an identical CSV is also copied directly into the location folder, that duplicate is counted as baseline only; the first remaining CSV marks spray start.
- Baseline samples are gray and spray samples are blue. Both use their actual sample timestamps; the baseline-colored trace meets the first spray sample at zero, where the blue spray trace begins.
- The x-axis is time relative to spray start (`min:s`): baseline readings are negative and the first spray sample is at zero. No baseline subtraction is applied to the pressure values.
- The red point marks the highest measured sample in the spray phase; it is a visual reference, not an automatic leak verdict.
- Gaps longer than two seconds, including the baseline-to-spray transition, are left unconnected and marked in red.
- Fit graphs add an orange curve and show its equation (with time from spray start in minutes) and R². The linear fit summarizes the average direction; the quadratic fit allows broad curvature. Both use only the spray phase, so neither is designed to isolate a short-lived peak. R² describes how much of the variation in the recorded pressures is captured by that fit; it does not establish a leak or its cause.

These fits are exploratory visual comparisons, not a leak-rate calculation or a leak verdict. Use the raw plots to identify a time range before interpreting a short rise or peak; a fit over the full spray phase can hide local behavior. Source files are never changed.
