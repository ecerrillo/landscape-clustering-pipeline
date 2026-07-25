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
python main.py --config config.yml
```

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

## Using your own data

Replace the contents of `data/` and edit `config.yml` (`paths.dem_index`,
`paths.points_gpkg`, `crs.analysis_crs`) — or better, start from
`config.example.yml` in the parent project, which documents every option and
which ones need recalibrating per site (in particular
`terrain_position.critical_slope_deg`, and `clustering.min_cluster_size`/
`min_samples`, which were scaled down here for this 225-point sample and
should be scaled back up for a full-size dataset).

## Files

- `main.py` — entry point (`python main.py --config config.yml`)
- `raster_processing.py`, `hdbscan_clustering.py`, `generate_reports.py` — pipeline modules
- `sensitivity_analysis.py` — post-hoc robustness check (`python sensitivity_analysis.py --config config.yml`)
- `data/dem/sample_tile.tif` — sample DEM (2 m resolution, ~1 km²)
- `data/index.csv` — DEM tile index (one row, points at the sample tile)
- `data/points.gpkg` — 225-point sample (subset of the full project dataset falling within the sample tile)

## Data attribution

The sample DEM tile (`data/dem/sample_tile.tif`) is derived from the digital
terrain models of the Plan Nacional de Ortofotografía Aérea (PNOA-LiDAR),
© Instituto Geográfico Nacional de España (IGN/CNIG), distributed under a
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license via the
[Centro de Descargas del CNIG](https://centrodedescargas.cnig.es/).
