#!/usr/bin/env python3
"""
Cell-scale association between prehistoric and historic surface finds.

Tests whether prehistoric finds (P) are denser where historic finds (H) occur,
using only fully surveyed square cells, at one or more cell sizes (default
20 m and 40 m) so that sensitivity to the modifiable areal unit problem
(MAUP) can be assessed.

For each cell i, the number of prehistoric finds is modelled with a negative
binomial (NB2) regression:

    P_i ~ NB2(mu_i, alpha)
    log(mu_i) = log(area_i in ha) + b0 [+ zone_i] + bH * hpos_i + bD * lp_i
    Var(P_i)  = mu_i + alpha * mu_i^2

    hpos_i = 1 if the cell contains any historic find, 0 otherwise.
    lp_i   = log2(H density_i / REF) if hpos_i = 1; 0 if hpos_i = 0.

    exp(bH) = ratio of P density between a cell with H at the reference
              density REF and a cell without H (presence effect).
    exp(bD) = change in P density when H density doubles, among cells
              containing H (intensity effect).

Uncertainty: robust standard errors clustered by square spatial blocks
(sandwich estimator with finite-sample correction) and t intervals with
G - 1 degrees of freedom (G = number of blocks).

Three responses are fitted: all prehistoric finds (`pre`), clustered
prehistoric finds (`pre_cl`, cluster >= 0) and unclustered prehistoric finds
(`pre_no`, HDBSCAN noise, cluster == -1).

Inputs (all taken from config.yml):
* prehistoric points with cluster labels: the GeoPackage written by main.py
  (`output.gpkg_path`), so run `python main.py --config config.yml` first;
* historic points: the rows of `paths.points_gpkg` whose
  `point_filter.field` matches `cell_density_model.historic_values`
  (by default, the values excluded by `point_filter`);
* survey area: `cell_density_model.survey_area` (polygon layer) or, if null,
  the footprint of the DEM tiles listed in `paths.dem_index`.

Output: one CSV table with the model estimates, one CSV per cell size with
the cell counts, and a short markdown report, under `outputs/`.

Usage
-----
    python cell_density_model.py --config config.yml
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import statsmodels.api as sm
from pyproj import CRS
from scipy.stats import t as student_t
from shapely.geometry import box
from shapely.ops import unary_union
from statsmodels.stats.sandwich_covariance import cov_cluster

from main import load_config, normalize_text

LOGGER = logging.getLogger("cell_density_model")

RESPONSES = ("pre", "pre_cl", "pre_no")
DEFAULTS: Dict[str, Any] = {
    "cell_sizes_m": [20, 40],
    "block_size_m": 200.0,
    "reference_h_ha": None,
    "survey_area": None,
    "survey_layer": None,
    "zone_field": None,
    "historic_values": None,
    "exclude_historic_ids": [],
    "id_field": "fid",
}


# --------------------------------------------------------------------------
# Data input
# --------------------------------------------------------------------------
def model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the cell_density_model section merged with defaults."""
    cfg = dict(DEFAULTS)
    cfg.update(config.get("cell_density_model") or {})
    if cfg["historic_values"] is None:
        cfg["historic_values"] = list(config.get("point_filter", {}).get("exclude_values", []))
    if not cfg["historic_values"]:
        raise ValueError("cell_density_model.historic_values is empty: cannot identify historic finds.")
    return cfg


def read_layer(path: Path, layer: Optional[str], crs: CRS) -> gpd.GeoDataFrame:
    """Read a vector layer and reproject it to the analysis CRS."""
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path}: layer CRS is undefined.")
    return gdf.to_crs(crs)


def historic_points(config: Dict[str, Any], cfg: Dict[str, Any], crs: CRS) -> gpd.GeoDataFrame:
    """Rows of the raw point layer classified as historic, minus excluded ids."""
    raw = read_layer(Path(config["paths"]["points_gpkg"]), config["paths"].get("points_layer"), crs)
    pf = config.get("point_filter", {})
    field = pf.get("field", "material")
    if field not in raw.columns:
        raise ValueError(f"Field '{field}' (point_filter.field) not found in points_gpkg.")
    opts = dict(case_sensitive=bool(pf.get("case_sensitive", False)),
                strip_whitespace=bool(pf.get("strip_whitespace", True)),
                normalize_accents=bool(pf.get("normalize_accents", True)))
    wanted = {normalize_text(v, **opts) for v in cfg["historic_values"]}
    hist = raw[raw[field].map(lambda v: normalize_text(v, **opts)).isin(wanted)]
    if cfg["exclude_historic_ids"]:
        id_field = cfg["id_field"]
        ids = hist[id_field] if id_field in hist.columns else pd.Series(hist.index, index=hist.index)
        hist = hist[~ids.isin(cfg["exclude_historic_ids"])]
    return hist


def prehistoric_points(config: Dict[str, Any], crs: CRS) -> gpd.GeoDataFrame:
    """Enriched prehistoric points written by main.py (must contain `cluster`)."""
    out = config.get("output", {})
    path = Path(out.get("gpkg_path", "outputs/points_enriched.gpkg"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python main.py --config ...` first.")
    pts = read_layer(path, out.get("gpkg_layer"), crs)
    if "cluster" not in pts.columns:
        raise ValueError(f"{path} has no 'cluster' column.")
    return pts


def survey_polygons(config: Dict[str, Any], cfg: Dict[str, Any], crs: CRS) -> List[Tuple[str, Any]]:
    """List of (zone label, polygon). Falls back to the DEM tile footprints."""
    if cfg["survey_area"]:
        gdf = read_layer(Path(cfg["survey_area"]), cfg["survey_layer"], crs)
        zone = cfg["zone_field"]
        labels = gdf[zone].astype(str) if zone and zone in gdf.columns else pd.Series(["all"] * len(gdf))
        return [(lab, geom) for lab, geom in zip(labels, gdf.geometry)]
    index = pd.read_csv(config["paths"]["dem_index"])
    tiles = gpd.GeoDataFrame(geometry=[box(r.minx, r.miny, r.maxx, r.maxy) for r in index.itertuples()],
                             crs=index["crs"].iloc[0]).to_crs(crs)
    LOGGER.info("No survey_area given: using the footprint of %d DEM tile(s).", len(tiles))
    return [("all", unary_union(list(tiles.geometry)))]


# --------------------------------------------------------------------------
# Grid of fully surveyed cells
# --------------------------------------------------------------------------
def full_cells(areas: Sequence[Tuple[str, Any]], cs: float) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Square cells of side cs anchored at the minimum of the survey extent,
    keeping only those lying entirely within the surveyed area."""
    survey = unary_union([g for _, g in areas])
    x0, y0, x1, y1 = survey.bounds
    ii, jj = np.meshgrid(np.arange(int((x1 - x0) // cs) + 1),
                         np.arange(int((y1 - y0) // cs) + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    cells = shapely.box(x0 + ii * cs, y0 + jj * cs, x0 + (ii + 1) * cs, y0 + (jj + 1) * cs)
    keep = shapely.covers(survey, cells)                 # full cell: covered by the survey
    ii, jj, cells = ii[keep], jj[keep], cells[keep]
    # Zone = surveyed polygon occupying the largest share of the cell
    shares = np.column_stack([shapely.area(shapely.intersection(cells, g)) for _, g in areas])
    zone = np.array([z for z, _ in areas])[shares.argmax(axis=1)]
    cent = shapely.centroid(cells)
    df = pd.DataFrame({"i": ii, "j": jj, "x": shapely.get_x(cent), "y": shapely.get_y(cent),
                       "area_ha": cs * cs / 1e4, "zone": zone})
    return df, (x0, y0)


def count_points(cells: pd.DataFrame, xy: np.ndarray, origin: Tuple[float, float], cs: float) -> np.ndarray:
    """Number of points (n x 2 array of coordinates) falling in each cell."""
    x0, y0 = origin
    if len(xy) == 0:
        return np.zeros(len(cells), dtype=int)
    idx = pd.MultiIndex.from_arrays([((xy[:, 0] - x0) // cs).astype(int), ((xy[:, 1] - y0) // cs).astype(int)])
    counts = pd.Series(1, index=idx).groupby(level=[0, 1]).sum()
    return counts.reindex(pd.MultiIndex.from_frame(cells[["i", "j"]])).fillna(0).astype(int).to_numpy()


def build_cells(areas, P_xy, P_cluster, H_xy, cs) -> pd.DataFrame:
    """Full cells with counts of all, clustered and unclustered P, and of H."""
    cells, origin = full_cells(areas, cs)
    P_cluster = np.asarray(P_cluster)
    cells["pre"] = count_points(cells, P_xy, origin, cs)
    cells["pre_cl"] = count_points(cells, P_xy[P_cluster >= 0], origin, cs)
    cells["pre_no"] = count_points(cells, P_xy[P_cluster == -1], origin, cs)
    cells["his"] = count_points(cells, H_xy, origin, cs)
    return cells


def reference_density(cells: pd.DataFrame) -> float:
    """Geometric mean of historic density (H/ha) over cells containing H."""
    pos = cells[cells.his > 0]
    if pos.empty:
        raise ValueError("No cell contains historic finds: the model cannot be fitted.")
    return float(np.exp(np.log(pos.his / pos.area_ha).mean()))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def fit_nb(cells: pd.DataFrame, response: str, ref_h_ha: float, block_size: float) -> Dict[str, Any]:
    """Fit NB2 by maximum likelihood and return effects with block-robust CIs."""
    d = cells.copy()
    # Explanatory variables
    h_ha = d.his / d.area_ha
    d["hpos"] = (d.his > 0).astype(float)
    d["lp"] = np.where(d.his > 0, np.log2(h_ha.where(h_ha > 0, ref_h_ha) / ref_h_ha), 0.0)
    d["block"] = (np.floor(d.x / block_size).astype(int).astype(str) + "_" +
                  np.floor(d.y / block_size).astype(int).astype(str))
    # Zones without any find of the response are omitted (intercept not estimable)
    with_finds = d.groupby("zone")[response].sum()
    omitted = sorted(with_finds.index[with_finds == 0])
    d = d[~d.zone.isin(omitted)].reset_index(drop=True)

    X = pd.get_dummies(d.zone, prefix="zone", drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    X["hpos"], X["lp"] = d.hpos, d.lp
    offset = np.log(d.area_ha)

    # Starting values from a Poisson fit to stabilise convergence
    start = sm.GLM(d[response], X, family=sm.families.Poisson(), offset=offset).fit().params
    model = sm.NegativeBinomial(d[response], X, offset=offset, loglike_method="nb2")
    fit = None
    for method in ("bfgs", "newton", "nm"):         # fall back to other optimisers if needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trial = model.fit(start_params=np.r_[start, 1.0], method=method, maxiter=5000, disp=False)
        if (trial.mle_retvals.get("converged", False) and np.isfinite(trial.llf)
                and np.all(np.isfinite(trial.params))):
            fit = trial
            break
    if fit is None:
        raise RuntimeError("NB model did not converge")

    if max(abs(fit.params["hpos"]), abs(fit.params["lp"])) > 20:
        raise ValueError("effects not identifiable (too few cells with both P and H; separation)")

    groups = pd.factorize(d.block)[0]
    n_blocks = int(groups.max() + 1)
    if n_blocks < 10:
        raise ValueError(f"Only {n_blocks} spatial blocks: too few for block-robust errors.")
    cov = cov_cluster(fit, groups)                      # block-clustered sandwich
    q = student_t.ppf(0.975, n_blocks - 1)
    names = list(X.columns) + ["alpha"]
    result: Dict[str, Any] = {"cells": len(d), "blocks": n_blocks, "omitted_zones": ",".join(omitted),
                              "alpha": float(fit.params.iloc[-1]), "aic": float(fit.aic)}
    for term in ("hpos", "lp"):
        j = names.index(term)
        b, se = float(fit.params.iloc[j]), float(np.sqrt(cov[j, j]))
        result[f"{term}_ratio"] = float(np.exp(b))
        result[f"{term}_lo"], result[f"{term}_hi"] = float(np.exp(b - q * se)), float(np.exp(b + q * se))
    return result


def run_models(areas, P_xy, P_cluster, H_xy, cell_sizes, block_size, ref_h_ha=None,
               responses: Sequence[str] = RESPONSES) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame], float]:
    """Build the grids, fit every response at every cell size, return results."""
    grids = {cs: build_cells(areas, P_xy, P_cluster, H_xy, cs) for cs in cell_sizes}
    if ref_h_ha is None:                       # computed at the finest scale, then kept fixed
        ref_h_ha = reference_density(grids[min(cell_sizes)])
    survey_ha = unary_union([g for _, g in areas]).area / 1e4
    rows = []
    for cs, cells in grids.items():
        base = {"cell_m": cs, "full_cells_ha": cells.area_ha.sum(), "survey_ha": survey_ha,
                "P_in_cells": int(cells.pre.sum()), "H_in_cells": int(cells.his.sum()),
                "reference_h_ha": ref_h_ha}
        for response in responses:
            try:
                rows.append({**base, "response": response, "status": "ok",
                             **fit_nb(cells, response, ref_h_ha, block_size)})
            except Exception as exc:                # recorded, never silently dropped
                LOGGER.warning("Cell %s m, %s: model failed (%s)", cs, response, exc)
                rows.append({**base, "response": response, "status": f"failed: {exc}"})
    return pd.DataFrame(rows), grids, ref_h_ha


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(table: pd.DataFrame, path: Path) -> None:
    """Short markdown summary of the model estimates."""
    lines = ["# Cell-scale density model", "",
             "Negative binomial (NB2) regression of prehistoric finds per fully surveyed cell on the "
             "presence of historic finds (`hpos`) and on a doubling of historic density (`lp`). "
             "Ratios with 95% intervals from block-clustered robust errors.", "",
             "| Cell (m) | Response | Full cells (ha) | Cells | Blocks | Presence ratio (95% CI) | Doubling ratio (95% CI) |",
             "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        if r.status == "ok":
            lines.append(f"| {r.cell_m:g} | {r.response} | {r.full_cells_ha:.2f} | {int(r.cells)} | {int(r.blocks)} | "
                         f"{r.hpos_ratio:.2f} ({r.hpos_lo:.2f}–{r.hpos_hi:.2f}) | "
                         f"{r.lp_ratio:.2f} ({r.lp_lo:.2f}–{r.lp_hi:.2f}) |")
        else:
            lines.append(f"| {r.cell_m:g} | {r.response} | {r.full_cells_ha:.2f} | – | – | {r.status} | – |")
    lines += ["", f"Reference historic density: {table.reference_h_ha.iloc[0]:.2f} finds/ha."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config.yml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    cfg = model_config(config)
    crs = CRS.from_user_input(config["crs"]["analysis_crs"])
    areas = survey_polygons(config, cfg, crs)
    P = prehistoric_points(config, crs)
    H = historic_points(config, cfg, crs)
    LOGGER.info("Prehistoric finds: %d (%d unclustered); historic finds: %d.",
                len(P), int((P.cluster == -1).sum()), len(H))

    xy = lambda g: np.column_stack([g.geometry.x, g.geometry.y])
    table, grids, ref = run_models(areas, xy(P), P.cluster.to_numpy(), xy(H),
                                   cfg["cell_sizes_m"], float(cfg["block_size_m"]), cfg["reference_h_ha"])

    out = Path(config.get("paths", {}).get("output_dir", "outputs"))
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tables" / "cell_density_model.csv", index=False)
    for cs, cells in grids.items():
        cells.to_csv(out / "tables" / f"cell_counts_{cs:g}m.csv", index=False)
    write_report(table, out / "reports" / "cell_density_report.md")
    LOGGER.info("Reference historic density: %.2f finds/ha.", ref)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
