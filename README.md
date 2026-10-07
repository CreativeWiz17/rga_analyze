# RGA helium plots

Plots AMU-4 pressure separately for each location folder in `10.16_data`.

## Run

Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

Then, from this project folder, run:

```bash
python -m rga_analyze
```

Defaults: reads `10.16_data` and writes to `analysis_outputs`. For different folders:

```bash
python -m rga_analyze --root path/to/data --output path/to/results
```

## Results

- `analysis_outputs/plots/helium_by_location/`: raw, unfitted time-series graphs, one per location folder.
- `analysis_outputs/plots/helium_by_location_fits/linear/`: raw readings with a straight-line fit across the full recording, its equation, and R².
- `analysis_outputs/plots/helium_by_location_fits/quadratic/`: raw readings with a second-degree (quadratic) fit across the full recording, its equation, and R².
- `analysis_outputs/processed_data/amu4_by_location/`: the AMU-4 samples used for each graph, with timestamps and source-file names.

The run summary lists any CSV that did not contain usable chronological AMU-4 samples. Those files are left out rather than guessed into the plot.

## How to read the plots

- Each plot shows raw helium pressure. There is no rolling median or smoothing.
- The x-axis is elapsed time (`min:s`), starting from the first sample in that location folder. The y-axis uses the pressure unit in the RGA metadata.
- The red point marks the highest measured sample; it is a visual reference, not an automatic leak verdict.
- Red shading and dashed red edges mark time gaps longer than two seconds; the trace is not drawn across them.
- Fit graphs add an orange curve and show its equation (with elapsed time in minutes) and R². The linear fit summarizes the average direction; the quadratic fit allows broad curvature. Both use the entire recording, so neither is designed to isolate a short-lived peak. R² describes how much of the variation in the recorded pressures is captured by that fit; it does not establish a leak or its cause.

These full-recording fits are exploratory visual comparisons, not a leak-rate calculation or a leak verdict. Use the raw plots to identify a time range before interpreting a short rise or peak; a fit over the entire test can hide local behavior. Source files are never changed.
