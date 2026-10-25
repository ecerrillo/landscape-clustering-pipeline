# Landscape analysis pipeline — standalone package

Self-contained copy of the pipeline, including a small sample dataset (one
DEM tile, 225 points) so it can be verified with no external data.

## What it does

This pipeline evaluates whether the apparent scarcity of settlement evidence
around a site is a genuine behavioural signal or an artefact of detectability,
by testing the spatial coherence of surface artefact scatters against terrain
structure. From a LiDAR-derived DEM it computes slope and topographic position
rasters, then classifies the terrain into erosion–transport–deposition zones.
Artefacts are clustered with
HDBSCAN over a terrain-cost distance matrix that models gravity-driven,
downslope-biased displacement rather than plain planimetric proximity, so
clusters reflect plausible terrain-position pathways instead of arbitrary
spatial grouping. Resulting clusters are characterised with spatial
diagnostics (nearest-neighbour distances, kernel density) and cross-tabulated
against landscape position to distinguish primary occupation signal from
downslope-redistributed or terrain-position-affected material.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --config config.yml
```

`run_all.py` runs the three analysis steps in order — `main.py` (pipeline),
`sensitivity_analysis.py` (robustness of the clustering) and
`cell_density_model.py` (cell-scale density model) — stopping at the first
failure and listing the reports written. Use `--quick` for a reduced
sensitivity sweep (smoke test only) and `--skip` to leave out steps, e.g.
`python run_all.py --config config.yml --skip sensitivity`. Each step can
also be run on its own, as described below.

GDAL's Python bindings (`osgeo.gdal`) are required and are not always
installable via pip alone depending on platform — if `pip install -r
requirements.txt` fails on `GDAL`, install the matching GDAL system package
first (e.g. `brew install gdal` / `apt install gdal-bin libgdal-dev`) or use
conda (`conda install -c conda-forge gdal`) before retrying.

Outputs are written to `outputs/`: an enriched GeoPackage/CSV/Parquet,
per-cluster summary and spatial diagnostic tables, figures, and a markdown
run report under `outputs/reports/`.

### Sensitivity analysis

After `main.py` has produced `outputs/points_enriched.gpkg`, check how robust
its clustering is to positional error, cost-model settings and HDBSCAN
parameters:

```bash
python sensitivity_analysis.py --config config.yml
```

This reuses `hdbscan_clustering.py` directly (no reimplementation), so it can
never drift from the pipeline it is checking. It writes a markdown report to
`outputs/reports/sensitivity_report.md` and three tables to `outputs/tables/`
— no figures. Pass `--quick` for a fast smoke test with reduced replicate and
grid counts.

### Cell-scale density model

Tests whether prehistoric finds are denser where historic finds occur. After
`main.py` has produced `outputs/points_enriched.gpkg`:

```bash
python cell_density_model.py --config config.yml
```

The surveyed area is divided into square cells (20 m and 40 m by default, to
assess sensitivity to the modifiable areal unit problem) and only fully
surveyed cells are kept. In each cell, prehistoric finds (all, clustered and
unclustered/noise) and historic finds are counted. The number of prehistoric
finds is modelled with a negative binomial (NB2) regression with cell area as
exposure (log offset), optional survey-zone fixed effects, and two historic
terms:

- **presence** (`hpos`): ratio of prehistoric density between a cell with
  historic finds at a reference density and a cell without them;
- **intensity** (`lp`): change in prehistoric density when historic density
  doubles, among cells with historic finds.

Standard errors are clustered by 200 m spatial blocks, with t intervals on
G − 1 degrees of freedom. Settings are under `cell_density_model` in
`config.yml`; historic finds are, by default, the values excluded by
`point_filter`, and the survey area defaults to the DEM tile footprint. It
writes `outputs/tables/cell_density_model.csv`, the cell counts per size and
`outputs/reports/cell_density_report.md`. A model that cannot be estimated
(e.g. too few cells with both kinds of finds, or fewer than ten blocks) is
reported as failed rather than dropped; on the 225-point sample this happens
for clustered finds at 20 m.

### Tests

```bash
python -m pytest tests/
```

The tests check the grid (only fully surveyed cells, conservation of
counts, zone assignment), recover a simulated presence effect, and run the
model on the bundled sample.

## Using your own data

Replace the contents of `data/` and edit `config.yml` (`paths.dem_index`,
`paths.points_gpkg`, `crs.analysis_crs`) — or better, start from
`config.example.yml` in the parent project, which documents every option and
which ones need recalibrating per site (in particular
`terrain_position.critical_slope_deg`, and `clustering.min_cluster_size`/
`min_samples`, which were scaled down here for this 225-point sample and
should be scaled back up for a full-size dataset).

## Files

- `run_all.py` — runs every step in order (`python run_all.py --config config.yml`)
- `main.py` — pipeline entry point (`python main.py --config config.yml`)
- `raster_processing.py`, `hdbscan_clustering.py`, `generate_reports.py` — pipeline modules
- `sensitivity_analysis.py` — post-hoc robustness check (`python sensitivity_analysis.py --config config.yml`)
- `cell_density_model.py` — cell-scale prehistoric/historic density model (`python cell_density_model.py --config config.yml`)
- `tests/` — pytest suite (`python -m pytest tests/`)
- `CITATION.cff` — citation metadata (also used by Zenodo)
- `data/dem/sample_tile.tif` — sample DEM (2 m resolution, ~1 km²)
- `data/index.csv` — DEM tile index (one row, points at the sample tile)
- `data/points.gpkg` — 225-point sample (subset of the full project dataset falling within the sample tile)

## Citation

If you use this software, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button shows
it in APA and BibTeX). Each release is archived on Zenodo with its own DOI;
cite the DOI of the version you used.

## Data attribution

The sample DEM tile (`data/dem/sample_tile.tif`) is derived from the digital
terrain models of the Plan Nacional de Ortofotografía Aérea (PNOA-LiDAR),
© Instituto Geográfico Nacional de España (IGN/CNIG), distributed under a
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license via the
[Centro de Descargas del CNIG](https://centrodedescargas.cnig.es/).
