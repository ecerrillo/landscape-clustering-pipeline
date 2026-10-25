"""Tests for cell_density_model.py.

Run from the repository root:  python -m pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cell_density_model as cdm  # noqa: E402


def points_in_cells(cells: pd.DataFrame, counts: np.ndarray, cs: float, origin, rng) -> np.ndarray:
    """Place counts[k] uniform random points inside cell k."""
    x0, y0 = origin
    xs, ys = [], []
    for (i, j), n in zip(cells[["i", "j"]].to_numpy(), counts):
        xs.append(x0 + (i + rng.uniform(0.05, 0.95, n)) * cs)
        ys.append(y0 + (j + rng.uniform(0.05, 0.95, n)) * cs)
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])


def test_full_cells_exclude_partial_cells():
    # L-shaped survey area: 200 x 200 m square minus its upper-right 100 x 100 m quarter,
    # plus a 10 m strip that only produces partial 20 m cells.
    survey = Polygon([(0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)]).union(box(200, 0, 210, 100))
    cells, origin = cdm.full_cells([("all", survey)], 20)
    assert origin == (0.0, 0.0)
    # 100 cells in the square, minus 25 in the removed quarter; the strip adds none
    assert len(cells) == 75
    assert np.isclose(cells.area_ha.sum(), 75 * 0.04)
    assert all(survey.covers(box(x - 10, y - 10, x + 10, y + 10)) for x, y in cells[["x", "y"]].to_numpy())


def test_counts_are_conserved():
    rng = np.random.default_rng(1)
    areas = [("all", box(0, 0, 400, 400))]
    cells, origin = cdm.full_cells(areas, 40)
    n = rng.poisson(3, len(cells))
    xy = points_in_cells(cells, n, 40, origin, rng)
    assert cdm.count_points(cells, xy, origin, 40).sum() == len(xy) == n.sum()
    np.testing.assert_array_equal(cdm.count_points(cells, xy, origin, 40), n)


def test_zones_assigned_by_largest_share():
    areas = [("A", box(0, 0, 100, 100)), ("B", box(100, 0, 200, 100))]
    cells, _ = cdm.full_cells(areas, 50)
    assert len(cells) == 8
    assert (cells.loc[cells.x < 100, "zone"] == "A").all() and (cells.loc[cells.x > 100, "zone"] == "B").all()


def test_recovers_simulated_presence_effect():
    """P is simulated with a known 5-fold density increase where H is present."""
    rng = np.random.default_rng(20261025)
    cs, true_ratio = 20.0, 5.0
    areas = [("all", box(0, 0, 1400, 1400))]                  # 49 blocks of 200 m
    cells, origin = cdm.full_cells(areas, cs)
    has_h = rng.random(len(cells)) < 0.15
    h = np.where(has_h, rng.integers(1, 6, len(cells)), 0)
    mu = 10.0 * cells.area_ha.to_numpy() * np.where(has_h, true_ratio, 1.0)   # 10 P/ha baseline
    alpha = 1.0
    p = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    P_xy = points_in_cells(cells, p, cs, origin, rng)
    H_xy = points_in_cells(cells, h, cs, origin, rng)
    clusters = np.zeros(len(P_xy), dtype=int)

    table, grids, ref = cdm.run_models(areas, P_xy, clusters, H_xy, [cs], 200.0, responses=("pre",))
    row = table.iloc[0]
    assert row.status == "ok"
    assert grids[cs].pre.sum() == len(P_xy) and grids[cs].his.sum() == len(H_xy)
    assert row.hpos_lo < true_ratio < row.hpos_hi             # CI covers the true effect
    assert row.hpos_lo > 1                                    # and excludes "no effect"
    assert row.lp_lo < 1 < row.lp_hi                          # no intensity effect was simulated


def test_too_few_blocks_is_reported_not_raised():
    rng = np.random.default_rng(3)
    areas = [("all", box(0, 0, 200, 200))]                    # a single 200 m block
    cells, origin = cdm.full_cells(areas, 20)
    xy = points_in_cells(cells, rng.poisson(2, len(cells)), 20, origin, rng)
    H = points_in_cells(cells, rng.poisson(0.3, len(cells)), 20, origin, rng)
    table, _, _ = cdm.run_models(areas, xy, np.zeros(len(xy), int), H, [20], 200.0, responses=("pre",))
    assert table.status.iloc[0].startswith("failed")


@pytest.mark.skipif(not (ROOT / "data" / "points.gpkg").exists(), reason="sample data not available")
def test_runs_on_bundled_sample():
    """Smoke test on the bundled 225-point sample, over the DEM tile footprint."""
    import geopandas as gpd

    pts = gpd.read_file(ROOT / "data" / "points.gpkg").to_crs("EPSG:25829")
    historic = pts.material.isin(["Cerámica a torno", "Constructivo"])
    xy = lambda g: np.column_stack([g.geometry.x, g.geometry.y])
    P, H = pts[~historic], pts[historic]
    index = pd.read_csv(ROOT / "data" / "index.csv")
    areas = [("all", box(*index.loc[0, ["minx", "miny", "maxx", "maxy"]]))]
    table, grids, _ = cdm.run_models(areas, xy(P), np.zeros(len(P), int), xy(H), [20, 40], 200.0,
                                     responses=("pre",))
    assert (table.status == "ok").all()
    assert (table.P_in_cells == len(P)).all() and (table.H_in_cells == len(H)).all()
    assert (table.hpos_ratio > 1).all()
