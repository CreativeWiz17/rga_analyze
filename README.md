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

- `analysis_outputs/plots/helium_by_location/`: one time-series graph per location folder.
- `analysis_outputs/processed_data/amu4_by_location/`: the AMU-4 samples used for each graph, with timestamps and source-file names.

The run summary lists any CSV that did not contain usable chronological AMU-4 samples. Those files are left out rather than guessed into the plot.

## How to read the plots

- Gray is the measured AMU-4 pressure; blue is its 15-second rolling median, which makes the slower trend easier to see.
- The red dot marks the highest smoothed value in that segment. It is a visual guide, not an automatic leak verdict.
- Time is elapsed minutes and seconds. Pressure units come from the RGA metadata.
- Shaded, labeled blanks mean no samples were recorded; lines are not drawn across them. Large gaps get a separate panel.

The pipeline does not yet calculate a line equation or R². A single straight-line fit across an entire test could hide a brief helium peak. Those fits should be added for time ranges selected from the plots; their equation and R² describe that range's linear trend, not leak probability or leak rate. Raw files are never changed.
