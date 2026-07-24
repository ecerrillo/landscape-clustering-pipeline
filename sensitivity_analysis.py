#!/usr/bin/env python3
"""
Sensitivity and robustness analysis for the landscape analysis pipeline.

Runs the same three checks reported in the manuscript's supplementary
material, on whichever dataset `config.yml` points at (the bundled 190-point
demo sample by default):

* Positional stability: Gaussian coordinate perturbations, full cost-matrix
  reconstruction per replicate, HDBSCAN re-fit at the reference parameters.
* Dissimilarity-matrix sensitivity: one-at-a-time variation of the
  topographic cost model (symmetrisation choice is not exposed by this
  pipeline's cost model — it always uses min — so this varies profile
  resolution, the uphill/downhill penalty ratio, and a Euclidean control).
* HDBSCAN parameter sensitivity: a min_cluster_size x min_samples grid on the
  fixed reference distance matrix.

Reuses `hdbscan_clustering.build_directional_path_cost_matrix` and
`enforce_max_cluster_diameter` directly rather than re-deriving the cost
model, so this script can never drift from the pipeline it is checking.

Output: one markdown report and three CSV tables under
`outputs/reports/` and `outputs/tables/`. No figures.

Usage
-----
    python sensitivity_analysis.py --config config.yml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import hdbscan as hdbscan_lib
from hdbscan_clustering import build_directional_path_cost_matrix, enforce_max_cluster_diameter
from main import load_config
from raster_processing import build_dem_mosaic_vrt

NOISE = -1

# ---------------------------------------------------------------------------
# Sweep definitions. Scaled down from the manuscript's full-dataset sweep to
# match this package's small demo sample; reduce further with --quick if a
# faster smoke test is all that's needed.
# ---------------------------------------------------------------------------
SIGMAS_M = [1.0, 2.0, 3.0]
N_REPLICATES = 50
MASTER_SEED = 20260719

PROFILE_SAMPLES = [10, 40]
UPHILL_PENALTIES = [2.0, 4.0, 10.0]  # ratios to the fixed downhill penalty

GRID_MIN_CLUSTER_SIZE = [10, 15, 20, 25]
GRID_MIN_SAMPLES = [5, 8, 12, 15]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce replicate/grid counts for a fast smoke test.",
    )
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    """Console-only logger (this script does not touch outputs/logs/run.log)."""
    logger = logging.getLogger("sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def load_points_and_reference(
    config: Dict[str, Any], logger: logging.Logger
) -> Tuple[pd.DataFrame, np.ndarray, Path]:
    """
    Load the enriched output of `main.py` (x/y columns plus the reference
    cluster labels it computed) and resolve the DEM mosaic used to build it.
    """
    gpkg_path = Path(config["output"]["gpkg_path"])
    if not gpkg_path.exists():
        raise FileNotFoundError(
            f"{gpkg_path} not found. Run `python main.py --config {config.get('_path', 'config.yml')}` "
            "first so this script has a reference partition to check."
        )
    gdf = gpd.read_file(gpkg_path)
    for col in ("x", "y", "cluster"):
        if col not in gdf.columns:
            raise ValueError(f"'{col}' column missing from {gpkg_path}; re-run main.py.")
    points = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    reference_labels = points["cluster"].to_numpy(dtype=int)
    logger.info("Loaded %d points from %s.", len(points), gpkg_path)

    mosaic_path = build_dem_mosaic_vrt(Path(config["paths"]["rasters_dir"]), logger)
    return points, reference_labels, mosaic_path


def run_hdbscan(
    matrix: np.ndarray,
    points: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    min_cluster_size: int,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit HDBSCAN on a precomputed matrix and apply the diameter postprocess."""
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="precomputed",
        core_dist_n_jobs=1,
        prediction_data=False,
    )
    clusterer.fit(matrix)
    labels, probs = enforce_max_cluster_diameter(
        np.asarray(clusterer.labels_, dtype=int),
        np.asarray(clusterer.probabilities_, dtype=float),
        points,
        config,
        logger,
    )
    probs = np.where(labels == NOISE, 0.0, probs)
    return labels, probs


def summarise(ref: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Partition description plus agreement with the reference partition."""
    noise = labels == NOISE
    sizes = np.array(
        [int(np.sum(labels == c)) for c in sorted({int(v) for v in labels if v != NOISE})],
        dtype=float,
    )
    both = (ref != NOISE) & (labels != NOISE)
    out = {
        "n_clusters": int(len(sizes)),
        "pct_noise": float(100.0 * noise.mean()) if len(labels) else np.nan,
        "size_min": float(sizes.min()) if sizes.size else np.nan,
        "size_max": float(sizes.max()) if sizes.size else np.nan,
        "prob_mean_clustered": float(probs[~noise].mean()) if (~noise).any() else np.nan,
        "ari_global": float(adjusted_rand_score(ref, labels)),
    }
    out["ari_clustered"] = (
        float(adjusted_rand_score(ref[both], labels[both])) if both.sum() >= 2 else np.nan
    )
    return out


def silhouette_clustered(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette over clustered points only; NaN when undefined."""
    mask = labels != NOISE
    if mask.sum() < 3 or len(set(labels[mask].tolist())) < 2:
        return float("nan")
    idx = np.where(mask)[0]
    return float(
        silhouette_score(matrix[np.ix_(idx, idx)], labels[mask], metric="precomputed")
    )


def derive_seed(*tokens: object) -> int:
    """Hash-derived child seed, reproducible from MASTER_SEED and the tokens."""
    import hashlib

    payload = "|".join(str(t) for t in (MASTER_SEED, *tokens))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


# ---------------------------------------------------------------------------
# The three sweeps
# ---------------------------------------------------------------------------
def table1_positional_stability(
    points: pd.DataFrame,
    ref: np.ndarray,
    config: Dict[str, Any],
    mosaic_path: Path,
    logger: logging.Logger,
    sigmas: list,
    n_replicates: int,
) -> pd.DataFrame:
    """Monte Carlo coordinate perturbation, full matrix rebuilt per replicate."""
    mcs = int(config["clustering"]["min_cluster_size"])
    ms = int(config["clustering"]["min_samples"])
    rows = []
    for sigma in sigmas:
        reps = []
        for rep in range(n_replicates):
            rng = np.random.default_rng(derive_seed("perturbation", sigma, rep))
            perturbed = points.copy()
            perturbed["x"] = points["x"] + rng.normal(0.0, sigma, size=len(points))
            perturbed["y"] = points["y"] + rng.normal(0.0, sigma, size=len(points))
            matrix = build_directional_path_cost_matrix(perturbed, config, logger, mosaic_path)
            labels, probs = run_hdbscan(matrix, perturbed, config, logger, mcs, ms)
            reps.append(summarise(ref, labels, probs))
        df = pd.DataFrame(reps)
        rows.append(
            {
                "sigma_m": sigma,
                "n_replicates": n_replicates,
                "n_clusters_mean": df["n_clusters"].mean(),
                "n_clusters_sd": df["n_clusters"].std(ddof=1),
                "pct_noise_mean": df["pct_noise"].mean(),
                "pct_noise_sd": df["pct_noise"].std(ddof=1),
                "ari_global_mean": df["ari_global"].mean(),
                "ari_global_p05": df["ari_global"].quantile(0.05),
                "ari_global_p95": df["ari_global"].quantile(0.95),
                "ari_clustered_mean": df["ari_clustered"].mean(),
                "ari_clustered_p05": df["ari_clustered"].quantile(0.05),
                "ari_clustered_p95": df["ari_clustered"].quantile(0.95),
            }
        )
        logger.info("sigma=%.1fm done (%d replicates).", sigma, n_replicates)
    return pd.DataFrame(rows)


def table2_distance_model(
    points: pd.DataFrame,
    ref: np.ndarray,
    config: Dict[str, Any],
    mosaic_path: Path,
    logger: logging.Logger,
    profile_samples: list,
    uphill_penalties: list,
) -> pd.DataFrame:
    """One-at-a-time variation of the topographic cost-model settings."""
    mcs = int(config["clustering"]["min_cluster_size"])
    ms = int(config["clustering"]["min_samples"])
    path_cfg = config["clustering"]["directional_path_cost"]
    downhill = float(path_cfg["downhill_penalty"])

    def run_scenario(component: str, name: str, cfg_override: Dict[str, Any]) -> Dict[str, Any]:
        scenario_cfg = {**config, "clustering": {**config["clustering"]}}
        scenario_cfg["clustering"]["directional_path_cost"] = {**path_cfg, **cfg_override}
        matrix = build_directional_path_cost_matrix(points, scenario_cfg, logger, mosaic_path)
        labels, probs = run_hdbscan(matrix, points, scenario_cfg, logger, mcs, ms)
        row = {"component": component, "scenario": name, **summarise(ref, labels, probs)}
        row["silhouette"] = silhouette_clustered(matrix, labels)
        return row

    rows = [
        run_scenario(
            "Reference partition",
            f"n_samples={path_cfg['n_samples']}; ratio={path_cfg['uphill_penalty']/downhill:g}",
            {},
        )
    ]
    for ns in profile_samples:
        rows.append(run_scenario("Profile samples", str(ns), {"n_samples": ns}))
    for up in uphill_penalties:
        rows.append(
            run_scenario(
                "Uphill/downhill penalty", f"ratio_{up/downhill:g}", {"uphill_penalty": up}
            )
        )

    # Euclidean control: no topographic cost model, plain planimetric distance.
    diff = points[["x", "y"]].to_numpy()[:, None, :] - points[["x", "y"]].to_numpy()[None, :, :]
    euclidean = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(euclidean, 0.0)
    labels, probs = run_hdbscan(euclidean, points, config, logger, mcs, ms)
    rows.append(
        {
            "component": "Distance model",
            "scenario": "Euclidean",
            **summarise(ref, labels, probs),
            "silhouette": silhouette_clustered(euclidean, labels),
        }
    )
    return pd.DataFrame(rows)


def table3_hdbscan_grid(
    points: pd.DataFrame,
    ref: np.ndarray,
    config: Dict[str, Any],
    mosaic_path: Path,
    logger: logging.Logger,
    grid_mcs: list,
    grid_ms: list,
) -> pd.DataFrame:
    """min_cluster_size x min_samples grid on the fixed reference matrix."""
    matrix = build_directional_path_cost_matrix(points, config, logger, mosaic_path)
    rows = []
    for mcs in grid_mcs:
        for ms in grid_ms:
            labels, probs = run_hdbscan(matrix, points, config, logger, mcs, ms)
            rows.append(
                {"min_cluster_size": mcs, "min_samples": ms, **summarise(ref, labels, probs)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    reference: Dict[str, Any],
    config: Dict[str, Any],
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render the three tables as one markdown report."""
    mcs = int(config["clustering"]["min_cluster_size"])
    ms = int(config["clustering"]["min_samples"])
    lines = [
        "# Sensitivity and robustness assessment",
        "",
        f"The reference partition used the directional path-cost distance matrix "
        f"defined in `config.yml` together with HDBSCAN (min_cluster_size = {mcs}, "
        f"min_samples = {ms}), identifying {reference['n_clusters']} clusters and "
        f"{reference['pct_noise']:.2f}% noise.",
        "",
        "## Table 1. Positional stability under Gaussian coordinate perturbations",
        "",
        "| σ (m) | Replicates | Clusters, μ±σ | Noise %, μ±σ | Global ARI [P5–P95] | Clustered ARI [P5–P95] |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in t1.iterrows():
        lines.append(
            f"| {r.sigma_m:g} | {int(r.n_replicates)} "
            f"| {r.n_clusters_mean:.2f} ± {r.n_clusters_sd:.2f} "
            f"| {r.pct_noise_mean:.2f} ± {r.pct_noise_sd:.2f} "
            f"| {r.ari_global_mean:.3f} [{r.ari_global_p05:.3f}–{r.ari_global_p95:.3f}] "
            f"| {r.ari_clustered_mean:.3f} [{r.ari_clustered_p05:.3f}–{r.ari_clustered_p95:.3f}] |"
        )
    lines += [
        "",
        "## Table 2. Sensitivity to the topographic dissimilarity settings",
        "",
        "| Component | Scenario | Clusters | Noise (%) | Global ARI | Clustered ARI | Silhouette |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in t2.iterrows():
        lines.append(
            f"| {r.component} | {r.scenario} | {r.n_clusters} | {r.pct_noise:.2f} "
            f"| {r.ari_global:.3f} | {r.ari_clustered:.3f} | {r.silhouette:.3f} |"
        )
    lines += [
        "",
        "## Table 3. Sensitivity to min_cluster_size (MCS) and min_samples (MS)",
        "",
        "| MCS | MS | Clusters | Noise (%) | Size (min–max) | Mean prob. (clustered) | Global ARI | Clustered ARI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t3.iterrows():
        b = "**" if (r.min_cluster_size == mcs and r.min_samples == ms) else ""
        lines.append(
            f"| {b}{int(r.min_cluster_size)}{b} | {b}{int(r.min_samples)}{b} "
            f"| {b}{int(r.n_clusters)}{b} | {b}{r.pct_noise:.2f}{b} "
            f"| {b}{r.size_min:g}–{r.size_max:g}{b} | {b}{r.prob_mean_clustered:.3f}{b} "
            f"| {b}{r.ari_global:.3f}{b} | {b}{r.ari_clustered:.3f}{b} |"
        )
    lines += ["", "The bold row is the reference configuration.", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the sensitivity analysis and write the report and tables."""
    args = parse_args()
    logger = setup_logging()
    config = load_config(Path(args.config))
    config["_path"] = args.config

    sigmas = SIGMAS_M[:2] if args.quick else SIGMAS_M
    n_replicates = 10 if args.quick else N_REPLICATES
    grid_mcs = GRID_MIN_CLUSTER_SIZE[:2] if args.quick else GRID_MIN_CLUSTER_SIZE
    grid_ms = GRID_MIN_SAMPLES[:2] if args.quick else GRID_MIN_SAMPLES

    points, ref_labels, mosaic_path = load_points_and_reference(config, logger)

    logger.info("Table 3: HDBSCAN parameter grid...")
    t3 = table3_hdbscan_grid(points, ref_labels, config, mosaic_path, logger, grid_mcs, grid_ms)

    logger.info("Table 2: dissimilarity-matrix sensitivity...")
    t2 = table2_distance_model(
        points, ref_labels, config, mosaic_path, logger, PROFILE_SAMPLES, UPHILL_PENALTIES
    )

    logger.info("Table 1: coordinate perturbation (%d replicates x %d sigmas)...",
                n_replicates, len(sigmas))
    t1 = table1_positional_stability(
        points, ref_labels, config, mosaic_path, logger, sigmas, n_replicates
    )

    matrix = build_directional_path_cost_matrix(points, config, logger, mosaic_path)
    mcs = int(config["clustering"]["min_cluster_size"])
    ms = int(config["clustering"]["min_samples"])
    labels, probs = run_hdbscan(matrix, points, config, logger, mcs, ms)
    reference = summarise(ref_labels, labels, probs)

    tables_dir = Path(config["paths"]["output_dir"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    t1.to_csv(tables_dir / "sensitivity_table1_positional_stability.csv", index=False)
    t2.to_csv(tables_dir / "sensitivity_table2_dissimilarity_settings.csv", index=False)
    t3.to_csv(tables_dir / "sensitivity_table3_hdbscan_parameters.csv", index=False)

    report_path = Path(config["paths"]["output_dir"]) / "reports" / "sensitivity_report.md"
    write_report(reference, config, t1, t2, t3, report_path)
    logger.info("Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
