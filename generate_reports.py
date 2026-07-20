#!/usr/bin/env python3
"""
Output generation for the landscape analysis pipeline.

Covers post-clustering spatial diagnostics (nearest-neighbour, Ripley's K,
KDE), writing the enriched GeoPackage/CSV/Parquet tables and cluster summary,
the markdown run report, and all figures (histograms, boxplots, pairplot).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sns = None


def get_material_field(config: Dict[str, Any]) -> str:
    """Return the material field name used for categorical encoding."""
    if "features_sub" in config:
        return config["features_sub"].get("material_layer_field", "layer")
    return config.get("features", {}).get("material_layer_field", "layer")


def run_cluster_diagnostics(
    points: gpd.GeoDataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Compute cluster diagnostics: NN analysis, Ripley's K and KDE summaries."""
    diag_cfg = config.get("cluster_diagnostics", {})
    if not diag_cfg.get("enable", False):
        return None

    nn_cfg = diag_cfg.get("nn", {})
    compute_nn = nn_cfg.get("compute", False)
    max_dist = nn_cfg.get("max_distance")
    rip_cfg = diag_cfg.get("ripley", {})
    compute_ripley = rip_cfg.get("compute", False)
    radii = sorted({float(r) for r in rip_cfg.get("radii", []) if float(r) > 0})
    den_cfg = diag_cfg.get("density", {})
    compute_density = den_cfg.get("compute", False)
    bandwidths = sorted({float(b) for b in den_cfg.get("bandwidths", []) if float(b) > 0})

    cluster_rows: List[Dict[str, Any]] = []
    ripley_rows: List[Dict[str, Any]] = []
    density_rows: List[Dict[str, Any]] = []
    for cluster_id, subset in points.groupby("cluster"):
        if cluster_id == -1:
            continue
        if len(subset) < 2:
            continue
        geoms = subset.geometry.values
        hull = MultiPoint(geoms).convex_hull
        minx, miny, maxx, maxy = hull.bounds
        diameter = float(np.hypot(maxx - minx, maxy - miny))
        area_m2 = float(hull.area)
        n_points = int(len(subset))
        cluster_index = subset.index.to_numpy(dtype=np.int64)
        x = subset.geometry.x.to_numpy(dtype="float32")
        y = subset.geometry.y.to_numpy(dtype="float32")
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_matrix = np.hypot(dx, dy).astype("float32")
        offdiag_mask = ~np.eye(n_points, dtype=bool)

        nn_mean = None
        nn_median = None
        nn_std = None
        nn_cv = None
        nn_expected_csr = None
        nn_clark_evans_r = None
        nn_zscore = None
        if compute_nn:
            nn_candidates = np.where(offdiag_mask, dist_matrix, np.inf)
            arr = np.min(nn_candidates, axis=1)
            arr = arr[np.isfinite(arr)]
            if max_dist is not None:
                arr = arr[arr <= float(max_dist)]
            if arr.size > 0:
                nn_mean = float(arr.mean())
                nn_median = float(np.median(arr))
                nn_std = float(arr.std())
                nn_cv = float(arr.std() / nn_mean) if nn_mean > 0 else None
                if area_m2 > 0:
                    intensity = n_points / area_m2
                    if intensity > 0:
                        nn_expected_csr = float(0.5 / np.sqrt(intensity))
                        if nn_expected_csr > 0:
                            nn_clark_evans_r = float(nn_mean / nn_expected_csr)
                        se = float(0.26136 / np.sqrt(n_points * intensity))
                        if se > 0:
                            nn_zscore = float((nn_mean - nn_expected_csr) / se)

        cluster_rows.append(
            {
                "cluster": int(cluster_id),
                "n_points": n_points,
                "area_m2": area_m2,
                "diameter_m": diameter,
                "nn_mean": nn_mean,
                "nn_median": nn_median,
                "nn_std": nn_std,
                "nn_cv": nn_cv,
                "nn_expected_csr": nn_expected_csr,
                "nn_clark_evans_r": nn_clark_evans_r,
                "nn_zscore": nn_zscore,
            }
        )

        if compute_ripley and radii and area_m2 > 0:
            for radius in radii:
                pair_count = float(np.sum((dist_matrix <= radius) & offdiag_mask))
                k_obs = float(area_m2 * pair_count / (n_points * (n_points - 1)))
                k_theory = float(np.pi * radius * radius)
                l_minus_r = float(np.sqrt(k_obs / np.pi) - radius) if k_obs >= 0 else np.nan
                ripley_rows.append(
                    {
                        "cluster": int(cluster_id),
                        "n_points": n_points,
                        "area_m2": area_m2,
                        "radius_m": float(radius),
                        "pair_count": pair_count,
                        "k_obs": k_obs,
                        "k_theory": k_theory,
                        "l_minus_r": l_minus_r,
                    }
                )

        if compute_density and bandwidths:
            for bandwidth in bandwidths:
                kernel = np.exp(-0.5 * (dist_matrix**2) / (bandwidth**2))
                density_vals = kernel.sum(axis=1) / (2.0 * np.pi * bandwidth * bandwidth * n_points)
                peak_pos = int(np.argmax(density_vals))
                density_rows.append(
                    {
                        "cluster": int(cluster_id),
                        "n_points": n_points,
                        "bandwidth_m": float(bandwidth),
                        "density_mean": float(np.mean(density_vals)),
                        "density_median": float(np.median(density_vals)),
                        "density_p90": float(np.percentile(density_vals, 90)),
                        "density_max": float(np.max(density_vals)),
                        "density_std": float(np.std(density_vals)),
                        "peak_point_index": int(cluster_index[peak_pos]),
                        "peak_x": float(x[peak_pos]),
                        "peak_y": float(y[peak_pos]),
                    }
                )

    cluster_df = pd.DataFrame(cluster_rows)
    ripley_df = pd.DataFrame(ripley_rows)
    density_df = pd.DataFrame(density_rows)

    if cluster_df.empty:
        logger.info("Cluster diagnostics enabled but no clusters eligible for diagnostics.")
        return None

    logger.info(
        "Computed diagnostics: clusters=%d ripley_rows=%d density_rows=%d",
        len(cluster_df),
        len(ripley_df),
        len(density_df),
    )
    return {
        "cluster": cluster_df,
        "ripley": ripley_df,
        "density": density_df,
    }


def safe_write_gpkg(
    gdf: gpd.GeoDataFrame, path: Path, layer: str, logger: logging.Logger
) -> None:
    """Write a GeoPackage layer with best-effort overwrite."""
    if path.exists():
        try:
            import fiona  # type: ignore

            if layer in fiona.listlayers(path):
                fiona.remove(path, layer=layer)
        except Exception as exc:  # pragma: no cover - depends on Fiona
            logger.warning("Could not remove existing layer; overwriting file: %s", exc)
            path.unlink()
    gdf.to_file(path, layer=layer, driver="GPKG")


def write_parquet_isolated(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet in a fresh subprocess that never imports GDAL.

    GDAL's built-in Arrow/Parquet drivers and pyarrow both register a 'file'
    scheme with the process-wide Arrow C++ filesystem registry; whichever loads
    first wins and the other raises ArrowKeyError. Since this pipeline may load
    GDAL in the current process (directional_path_cost mode builds an elevation
    mosaic via osgeo.gdal) before ever writing Parquet, the write is isolated in
    a subprocess where GDAL was never imported, so no conflict can occur.
    """
    pkl_path = path.with_suffix(".tmp.pkl")
    df.to_pickle(pkl_path)
    script = (
        "import pandas as pd, sys\n"
        "df = pd.read_pickle(sys.argv[1])\n"
        "df.to_parquet(sys.argv[2], index=False)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(pkl_path), str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
    finally:
        pkl_path.unlink(missing_ok=True)


def save_tables(
    df: pd.DataFrame, config: Dict[str, Any], run_id: str, logger: logging.Logger
) -> Dict[str, Path]:
    """Save tabular outputs (CSV/Parquet)."""
    output_dir = Path(config["paths"]["output_dir"]) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    if config["output"].get("save_tables_csv"):
        csv_path = output_dir / f"points_enriched_{run_id}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV table: %s", csv_path)
        saved["points_csv"] = csv_path
    if config["output"].get("save_tables_parquet"):
        pq_path = output_dir / f"points_enriched_{run_id}.parquet"
        try:
            write_parquet_isolated(df, pq_path)
            logger.info("Saved Parquet table: %s", pq_path)
            saved["points_parquet"] = pq_path
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to write Parquet table: %s", exc)
    return saved


def save_cluster_summaries(
    gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    run_id: str,
    logger: logging.Logger,
    diagnostics: Optional[pd.DataFrame] = None,
) -> Path:
    """Save summary tables for clusters."""
    output_dir = Path(config["paths"]["output_dir"]) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    material_field = get_material_field(config)

    def summarize(group_field: str) -> pd.DataFrame:
        subset = gdf[gdf[group_field] != -1]
        if subset.empty:
            return pd.DataFrame(columns=[group_field])
        summary = subset.groupby(group_field).size().to_frame("count")
        for col in [
            "elev_mean",
            "slope_mean",
            "curvature_mean",
            "flowacc_mean",
            "drain_dist_mean",
        ]:
            if col in subset.columns:
                summary[col] = subset.groupby(group_field)[col].mean()
        if "cluster_prob" in subset.columns:
            summary["cluster_prob_mean"] = subset.groupby(group_field)["cluster_prob"].mean()
            summary["cluster_prob_std"] = subset.groupby(group_field)["cluster_prob"].std()
        if material_field in subset.columns:
            dist = pd.crosstab(subset[group_field], subset[material_field], normalize="index")
            dist.columns = [f"material_{c}" for c in dist.columns]
            summary = summary.join(dist)
        if "taph_context" in subset.columns:
            taph = pd.crosstab(subset[group_field], subset["taph_context"], normalize="index")
            taph.columns = [f"taph_{c}" for c in taph.columns]
            summary = summary.join(taph)
        return summary.reset_index()

    cluster_noise = int((gdf["cluster"] == -1).sum()) if "cluster" in gdf else 0
    if cluster_noise:
        logger.info("Excluded %d noise points from summaries.", cluster_noise)
    summary_df = summarize("cluster")
    if diagnostics is not None and not diagnostics.empty:
        summary_df = summary_df.merge(diagnostics, on="cluster", how="left")
    summary_path = output_dir / f"cluster_summary_{run_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved cluster summary: %s", summary_path)
    return summary_path


def write_run_report(
    config: Dict[str, Any],
    run_id: str,
    logger: logging.Logger,
    summary: Dict[str, Any],
    output_paths: Dict[str, Path],
) -> Path:
    """Write a markdown report with run configuration, metrics and output files."""
    report_dir = Path(config["paths"]["output_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"analysis_report_{run_id}.md"

    cluster_metrics = summary.get("cluster_metrics", {})
    filter_stats = summary.get("point_filter", {})
    spatial_diag = summary.get("spatial_diagnostics", {})

    lines = [
        f"# Landscape Analysis Report ({run_id})",
        "",
        f"- Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- Analysis CRS: {config['crs'].get('analysis_crs')}",
        "",
        "## Input and Filtering",
        f"- Loaded points: {summary.get('points_loaded', 0)}",
        f"- Points after filter: {summary.get('points_after_filter', 0)}",
        f"- Points assigned to tiles: {summary.get('points_assigned_tiles', 0)}",
        "",
    ]

    if filter_stats.get("enabled", False):
        lines.extend(
            [
                f"- Filter field: {filter_stats.get('field')}",
                f"- Excluded values: {', '.join(str(v) for v in filter_stats.get('exclude_values', []))}",
                f"- Removed by filter: {filter_stats.get('removed_points', 0)}",
            ]
        )
        if filter_stats.get("missing_field", False):
            lines.append("- Filter warning: field not found in GPKG; no exclusion applied.")
    else:
        lines.append("- Point filter: disabled")

    lines.extend(
        [
            "",
            "## Clustering",
            f"- Total clusters (excluding noise): {summary.get('n_clusters', 0)}",
            f"- Noise points: {summary.get('n_noise', 0)}",
            f"- Noise ratio: {summary.get('noise_ratio', 0.0):.4f}",
            f"- Mean cluster probability: {summary.get('cluster_prob_mean', 0.0):.4f}",
            f"- Figures generated: {summary.get('figure_count', 0)}",
            "",
            "### Clustering quality metrics",
        ]
    )
    if cluster_metrics:
        for metric_name, metric_value in cluster_metrics.items():
            lines.append(f"- {metric_name}: {metric_value:.4f}")
    else:
        lines.append("- Not available (insufficient valid clusters).")

    lines.extend(
        [
            "",
            "## Spatial Diagnostics",
            f"- NN analysis enabled: {spatial_diag.get('nn_enabled', False)}",
            f"- Ripley's K enabled: {spatial_diag.get('ripley_enabled', False)}",
            f"- Ripley radii (m): {', '.join(str(v) for v in spatial_diag.get('ripley_radii', []))}",
            f"- Ripley rows: {spatial_diag.get('ripley_rows', 0)}",
            f"- KDE enabled: {spatial_diag.get('kde_enabled', False)}",
            f"- KDE bandwidths (m): {', '.join(str(v) for v in spatial_diag.get('kde_bandwidths', []))}",
            f"- KDE rows: {spatial_diag.get('kde_rows', 0)}",
        ]
    )

    taph_stats = summary.get("taphonomy", {})
    lines.extend(["", "## Taphonomic context (terrain-intrinsic)"])
    if taph_stats:
        for context_name, context_count in taph_stats.items():
            lines.append(f"- {context_name}: {context_count}")
    else:
        lines.append("- Taphonomic classification disabled or unavailable.")

    lines.extend(["", "## Output files"])
    if output_paths:
        for key in sorted(output_paths):
            lines.append(f"- {key}: `{output_paths[key]}`")
    else:
        lines.append("- No output files tracked.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved run report: %s", report_path)
    return report_path


def plot_histograms(
    df: pd.DataFrame, config: Dict[str, Any], run_id: str, logger: logging.Logger
) -> None:
    """Generate histograms of key variables."""
    fig_dir = Path(config["paths"]["output_dir"]) / "figures" / "histograms"
    fig_dir.mkdir(parents=True, exist_ok=True)
    variables = ["elev_mean", "slope_mean", "curvature_mean"]
    variables.extend([c for c in df.columns if c.startswith("tpi") and c.endswith("_mean")])
    variables.extend([c for c in df.columns if c.startswith("flowacc") and c.endswith("_mean")])
    variables.extend([c for c in df.columns if c.startswith("drain_dist") and c.endswith("_mean")])

    for var in variables:
        if var not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(df[var].dropna(), bins=30, color="#4c72b0", alpha=0.8)
        plt.title(f"Histogram: {var}")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        out_path = fig_dir / f"{var}_{run_id}.svg"
        plt.tight_layout()
        plt.savefig(out_path, format="svg")
        plt.close()
        logger.info("Saved histogram: %s", out_path)


def plot_cluster_prob_histogram(
    df: pd.DataFrame, config: Dict[str, Any], run_id: str, logger: logging.Logger
) -> None:
    """Plot histogram of cluster membership probabilities."""
    if "cluster_prob" not in df.columns:
        return
    fig_dir = Path(config["paths"]["output_dir"]) / "figures" / "histograms"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(df["cluster_prob"].dropna(), bins=30, color="#55a868", alpha=0.8)
    plt.title("Histogram: cluster_prob")
    plt.xlabel("cluster_prob")
    plt.ylabel("Frequency")
    out_path = fig_dir / f"cluster_prob_{run_id}.svg"
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    logger.info("Saved cluster_prob histogram: %s", out_path)


def plot_boxplots(
    df: pd.DataFrame, config: Dict[str, Any], run_id: str, logger: logging.Logger
) -> None:
    """Generate boxplots by material layer and cluster."""
    fig_dir = Path(config["paths"]["output_dir"]) / "figures" / "boxplots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    material_field = get_material_field(config)
    variables = ["elev_mean", "slope_mean"]

    for var in variables:
        if var not in df.columns:
            continue
        if material_field in df.columns:
            plt.figure(figsize=(8, 4))
            if sns is not None:
                sns.boxplot(data=df, x=material_field, y=var)
            else:
                df.boxplot(column=var, by=material_field, rot=45)
                plt.suptitle("")
            plt.title(f"{var} by {material_field}")
            plt.tight_layout()
            out_path = fig_dir / f"{var}_by_{material_field}_{run_id}.svg"
            plt.savefig(out_path, format="svg")
            plt.close()
            logger.info("Saved boxplot: %s", out_path)

        if "cluster" in df.columns:
            plt.figure(figsize=(8, 4))
            filtered = df[df["cluster"] != -1]
            if filtered.empty:
                plt.close()
                continue
            if sns is not None:
                sns.boxplot(data=filtered, x="cluster", y=var)
            else:
                filtered.boxplot(column=var, by="cluster", rot=45)
                plt.suptitle("")
            plt.title(f"{var} by cluster")
            plt.tight_layout()
            out_path = fig_dir / f"{var}_by_cluster_{run_id}.svg"
            plt.savefig(out_path, format="svg")
            plt.close()
            logger.info("Saved boxplot: %s", out_path)


def plot_pairplot(
    df: pd.DataFrame, config: Dict[str, Any], run_id: str, logger: logging.Logger
) -> None:
    """Generate pairplot/scatter matrix for a subset of variables."""
    fig_dir = Path(config["paths"]["output_dir"]) / "figures" / "pairplots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    candidates = [c for c in df.columns if c.endswith("_mean")]
    selected = candidates[:6]
    if len(selected) < 2:
        return
    plot_df = df[selected].dropna()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000, random_state=config["random"]["seed"])
    out_path = fig_dir / f"pairplot_{run_id}.svg"
    if sns is not None:
        grid = sns.pairplot(plot_df)
        grid.savefig(out_path)
    else:
        pd.plotting.scatter_matrix(plot_df, figsize=(8, 8))
        plt.tight_layout()
        plt.savefig(out_path, format="svg")
    plt.close("all")
    logger.info("Saved pairplot: %s", out_path)
