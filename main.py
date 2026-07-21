#!/usr/bin/env python3
"""
Landscape analysis pipeline for archaeological point data.

This script reads a DEM index and archaeological material points, derives
topographic metrics at micro-topographic scales, summarizes raster values in a
buffer around each point to account for positional accuracy, and performs
hierarchical clustering to explore archaeological and taphonomic structure.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pyproj import CRS

from generate_reports import (
    plot_boxplots,
    plot_cluster_prob_histogram,
    plot_histograms,
    plot_pairplot,
    run_cluster_diagnostics,
    safe_write_gpkg,
    save_cluster_summaries,
    save_tables,
    write_run_report,
)
from hdbscan_clustering import HDBSCAN_AVAILABLE, run_clustering
from raster_processing import (
    _GDAL_AVAILABLE,
    _compute_derivatives_for_tile_worker,
    build_dem_mosaic_vrt,
    classify_taphonomic_context,
    compute_derivatives_for_tile,
    read_dem_index,
    sample_points_by_tile,
    scale_to_tag,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Landscape analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must define a YAML mapping.")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values and required paths."""
    required_top = [
        "paths",
        "crs",
        "raster",
        "sampling",
        "clustering",
        "features_sub",
        "cluster_diagnostics",
        "plots",
        "output",
        "logging",
        "random",
    ]
    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    paths = config["paths"]
    for k in ["dem_index", "points_gpkg", "output_dir", "rasters_dir"]:
        if k not in paths:
            raise ValueError(f"Missing paths.{k} in config")

    dem_index = Path(paths["dem_index"])
    points_gpkg = Path(paths["points_gpkg"])
    if not dem_index.exists():
        raise FileNotFoundError(f"DEM index CSV not found: {dem_index}")
    if not points_gpkg.exists():
        raise FileNotFoundError(f"Points GeoPackage not found: {points_gpkg}")

    analysis_crs = config["crs"].get("analysis_crs")
    if not analysis_crs:
        raise ValueError("crs.analysis_crs must be defined in config")
    CRS.from_user_input(analysis_crs)

    raster_cfg = config["raster"]
    if raster_cfg.get("target_resolution", 0) <= 0:
        raise ValueError("raster.target_resolution must be > 0")
    tpi_scales = raster_cfg.get("tpi_scales", [])
    if not isinstance(tpi_scales, list) or any(s <= 0 for s in tpi_scales):
        raise ValueError("raster.tpi_scales must be a list of positive values")

    if config["sampling"].get("buffer_radius", 0) <= 0:
        raise ValueError("sampling.buffer_radius must be > 0")

    clustering = config["clustering"]
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is required for clustering but is not installed.")
    if not _GDAL_AVAILABLE:
        raise ImportError(
            "GDAL Python bindings (osgeo.gdal) are required to build the elevation "
            "mosaic used by the directional path-cost clustering metric."
        )
    if clustering.get("min_cluster_size", 0) <= 0 or clustering.get("min_samples", 0) <= 0:
        raise ValueError("clustering.min_cluster_size and min_samples must be > 0")
    path_cfg = clustering.get("directional_path_cost", {})
    if not isinstance(path_cfg, dict):
        raise ValueError("clustering.directional_path_cost must be a mapping.")
    if float(path_cfg.get("cutoff_radius_m", 300.0)) <= 0:
        raise ValueError("clustering.directional_path_cost.cutoff_radius_m must be > 0.")
    if int(path_cfg.get("n_samples", 20)) < 2:
        raise ValueError("clustering.directional_path_cost.n_samples must be >= 2.")
    for key in ["uphill_penalty", "downhill_penalty"]:
        if float(path_cfg.get(key, 0.0)) < 0:
            raise ValueError(f"clustering.directional_path_cost.{key} must be >= 0.")
    if float(path_cfg.get("unreachable_distance_factor", 10.0)) <= 0:
        raise ValueError(
            "clustering.directional_path_cost.unreachable_distance_factor must be > 0."
        )

    post_cfg = clustering.get("postprocess", {})
    if post_cfg.get("enforce_max_cluster_diameter", False):
        if float(post_cfg.get("max_cluster_diameter_m", 0.0)) <= 0:
            raise ValueError(
                "clustering.postprocess.max_cluster_diameter_m must be > 0 when enabled."
            )
        if int(post_cfg.get("min_subcluster_size", 1)) <= 0:
            raise ValueError(
                "clustering.postprocess.min_subcluster_size must be > 0 when enabled."
            )

    diag_cfg = config["cluster_diagnostics"]
    if diag_cfg.get("enable", False):
        ripley_radii = diag_cfg.get("ripley", {}).get("radii", [])
        if any(r <= 0 for r in ripley_radii):
            raise ValueError("cluster_diagnostics.ripley.radii must be positive.")
        bandwidths = diag_cfg.get("density", {}).get("bandwidths", [])
        if any(b <= 0 for b in bandwidths):
            raise ValueError("cluster_diagnostics.density.bandwidths must be positive.")

    point_filter = config.get("point_filter", {})
    if point_filter:
        if not isinstance(point_filter, dict):
            raise ValueError("point_filter must be a mapping.")
        if point_filter.get("enable", False):
            field = point_filter.get("field")
            if not isinstance(field, str) or not field.strip():
                raise ValueError("point_filter.field must be a non-empty string.")
            exclude_values = point_filter.get("exclude_values", [])
            if not isinstance(exclude_values, list) or not exclude_values:
                raise ValueError("point_filter.exclude_values must be a non-empty list.")

    taph_cfg = config.get("taphonomy", {})
    if taph_cfg:
        if not isinstance(taph_cfg, dict):
            raise ValueError("taphonomy must be a mapping.")
        if taph_cfg.get("enable", False):
            if float(taph_cfg.get("critical_slope_deg", 10.0)) <= 0:
                raise ValueError("taphonomy.critical_slope_deg must be > 0.")
            for key in ["slope_smooth_radius_m", "coarse_tpi_radius_m", "tpi_sd_threshold"]:
                default = {"slope_smooth_radius_m": 15.0, "coarse_tpi_radius_m": 100.0, "tpi_sd_threshold": 0.5}[key]
                if float(taph_cfg.get(key, default)) <= 0:
                    raise ValueError(f"taphonomy.{key} must be > 0.")
            if not _GDAL_AVAILABLE:
                raise ImportError(
                    "taphonomy.enable requires GDAL Python bindings (osgeo.gdal) "
                    "to build the elevation mosaic, but they are not installed."
                )


def normalize_text(
    value: Any,
    case_sensitive: bool = False,
    strip_whitespace: bool = True,
    normalize_accents: bool = True,
) -> str:
    """Normalize text values to compare categorical filters robustly."""
    text = "" if value is None else str(value)
    if strip_whitespace:
        text = text.strip()
    if normalize_accents:
        text = "".join(
            ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
        )
    if not case_sensitive:
        text = text.casefold()
    return text


def apply_point_filter(
    points: gpd.GeoDataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Apply optional point exclusion filter defined in config.point_filter."""
    loaded_points = int(len(points))
    cfg = config.get("point_filter", {})
    if not cfg or not cfg.get("enable", False):
        return points, {
            "enabled": False,
            "field": None,
            "exclude_values": [],
            "loaded_points": loaded_points,
            "removed_points": 0,
            "remaining_points": loaded_points,
        }

    field = cfg.get("field")
    exclude_values = cfg.get("exclude_values", [])
    case_sensitive = bool(cfg.get("case_sensitive", False))
    strip_whitespace = bool(cfg.get("strip_whitespace", True))
    normalize_accents = bool(cfg.get("normalize_accents", True))

    if field not in points.columns:
        logger.warning(
            "Point filter enabled but field '%s' is missing. No points excluded.", field
        )
        return points, {
            "enabled": True,
            "field": field,
            "exclude_values": list(exclude_values),
            "loaded_points": loaded_points,
            "removed_points": 0,
            "remaining_points": loaded_points,
            "missing_field": True,
        }

    exclude_normalized = {
        normalize_text(v, case_sensitive, strip_whitespace, normalize_accents)
        for v in exclude_values
    }
    series_normalized = points[field].map(
        lambda v: normalize_text(v, case_sensitive, strip_whitespace, normalize_accents)
    )
    remove_mask = series_normalized.isin(exclude_normalized)
    removed = int(remove_mask.sum())
    if removed == 0:
        logger.info("Point filter removed 0 points using field '%s'.", field)
        return points, {
            "enabled": True,
            "field": field,
            "exclude_values": list(exclude_values),
            "loaded_points": loaded_points,
            "removed_points": 0,
            "remaining_points": loaded_points,
        }

    filtered = points.loc[~remove_mask].copy()
    logger.info(
        "Point filter removed %d points using field '%s'. Remaining: %d.",
        removed,
        field,
        len(filtered),
    )
    if filtered.empty:
        raise ValueError("Point filter removed all points; adjust point_filter in config.")
    return filtered, {
        "enabled": True,
        "field": field,
        "exclude_values": list(exclude_values),
        "loaded_points": loaded_points,
        "removed_points": removed,
        "remaining_points": int(len(filtered)),
    }


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configure console and optional file logging."""
    level = config["logging"].get("level", "INFO").upper()
    logger = logging.getLogger("landscape_analysis")
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if config["logging"].get("log_to_file"):
        log_path = Path(config["logging"].get("log_file", "run.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Create output directories if they do not exist."""
    output_dir = Path(config["paths"]["output_dir"])
    rasters_dir = Path(config["paths"]["rasters_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    (rasters_dir / "dem_resampled").mkdir(parents=True, exist_ok=True)
    if config["raster"].get("slope"):
        (rasters_dir / "slope").mkdir(parents=True, exist_ok=True)
    for scale in config["raster"].get("tpi_scales", []):
        (rasters_dir / f"tpi{scale_to_tag(scale)}").mkdir(parents=True, exist_ok=True)


def read_points(
    config: Dict[str, Any], logger: logging.Logger
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Read point layer and reproject to analysis CRS."""
    points_path = Path(config["paths"]["points_gpkg"])
    layer = config["paths"].get("points_layer")
    if layer:
        gdf = gpd.read_file(points_path, layer=layer)
    else:
        gdf = gpd.read_file(points_path)
    analysis_crs = CRS.from_user_input(config["crs"]["analysis_crs"])
    if gdf.crs is None:
        raise ValueError("Point layer CRS is undefined.")
    gdf = gdf.to_crs(analysis_crs)
    logger.info("Loaded %d points.", len(gdf))
    gdf, filter_stats = apply_point_filter(gdf, config, logger)
    return gdf, filter_stats


def assign_tiles(
    points: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame, logger: logging.Logger
) -> gpd.GeoDataFrame:
    """Assign each point to an intersecting DEM tile (first match if multiple)."""
    points = points.copy()
    points["pt_index"] = np.arange(len(points))
    try:
        joined = gpd.sjoin(points, tiles, how="left", predicate="intersects")
    except TypeError:  # pragma: no cover - older geopandas
        joined = gpd.sjoin(points, tiles, how="left", op="intersects")

    joined = joined.sort_values("pt_index")
    joined = joined.drop_duplicates(subset="pt_index", keep="first").set_index("pt_index")
    points = points.set_index("pt_index")
    points = points.join(joined[["tile_id", "dem_path", "dem_crs"]])
    points = points.reset_index(drop=True)

    missing = points["tile_id"].isna().sum()
    if missing:
        logger.warning("%d points could not be assigned to a DEM tile.", missing)
        points = points[points["tile_id"].notna()].copy()
    return points



def run_pipeline(config: Dict[str, Any], logger: logging.Logger) -> None:
    """Run the full analysis pipeline."""
    ensure_output_dirs(config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths: Dict[str, Path] = {}

    tiles = read_dem_index(config, logger)
    points, filter_stats = read_points(config, logger)
    points = assign_tiles(points, tiles, logger)
    points_assigned_tiles = int(len(points))

    valid_tiles: List[Tuple[str, Path]] = []
    for tile_id, group in points.groupby("tile_id"):
        dem_path = Path(group["dem_path"].iloc[0])
        if not dem_path.exists():
            logger.warning("DEM path missing: %s", dem_path)
            continue
        valid_tiles.append((tile_id, dem_path))

    tile_rasters: Dict[str, Dict[str, Path]] = {}
    configured_workers = config.get("performance", {}).get("max_workers")
    max_workers = int(configured_workers) if configured_workers else (os.cpu_count() or 1)
    max_workers = min(max_workers, len(valid_tiles)) if valid_tiles else 0

    if max_workers <= 1:
        for tile_id, dem_path in valid_tiles:
            tile_rasters[tile_id] = compute_derivatives_for_tile(
                tile_id, dem_path, config, logger
            )
    else:
        logger.info(
            "Computing DEM derivatives for %d tiles using %d parallel workers.",
            len(valid_tiles),
            max_workers,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _compute_derivatives_for_tile_worker,
                    (tile_id, str(dem_path), config),
                ): tile_id
                for tile_id, dem_path in valid_tiles
            }
            for future in as_completed(futures):
                tile_id, outputs, records = future.result()
                for level, msg in records:
                    getattr(logger, level)(msg)
                tile_rasters[tile_id] = outputs

    points = sample_points_by_tile(points, tile_rasters, config, logger)

    # The elevation mosaic backs both the directional path-cost clustering
    # metric (always active) and the taphonomic process-zone raster.
    elevation_mosaic_path = build_dem_mosaic_vrt(Path(config["paths"]["rasters_dir"]), logger)

    points, taph_stats = classify_taphonomic_context(
        points, config, logger, elevation_mosaic_path
    )

    cluster_labels, cluster_prob, cluster_metrics = run_clustering(
        points, config, logger, elevation_mosaic_path
    )
    cluster_prob = np.where(cluster_labels == -1, 0.0, cluster_prob)
    logger.info(
        "Cluster membership probability mean=%.3f std=%.3f",
        float(np.mean(cluster_prob)),
        float(np.std(cluster_prob)),
    )

    points["cluster"] = cluster_labels
    points["cluster_prob"] = cluster_prob

    if config["output"].get("save_gpkg"):
        gpkg_path = Path(config["output"]["gpkg_path"])
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
        layer = config["output"].get("gpkg_layer", "points_enriched")
        safe_write_gpkg(points, gpkg_path, layer, logger)
        logger.info("Saved enriched GeoPackage: %s", gpkg_path)
        output_paths["gpkg"] = gpkg_path

    geom_col = points.geometry.name
    table_df = pd.DataFrame(points.drop(columns=geom_col))
    output_paths.update(save_tables(table_df, config, run_id, logger))
    diagnostics = run_cluster_diagnostics(points, config, logger)
    cluster_diag_df = diagnostics["cluster"] if diagnostics is not None else None
    ripley_df = diagnostics["ripley"] if diagnostics is not None else None
    density_df = diagnostics["density"] if diagnostics is not None else None
    summary_path = save_cluster_summaries(
        points, config, run_id, logger, diagnostics=cluster_diag_df
    )
    output_paths["cluster_summary"] = summary_path
    if cluster_diag_df is not None and not cluster_diag_df.empty:
        diag_path = (
            Path(config["paths"]["output_dir"])
            / "tables"
            / f"cluster_diagnostics_{run_id}.csv"
        )
        cluster_diag_df.to_csv(diag_path, index=False)
        logger.info("Saved cluster diagnostics: %s", diag_path)
        output_paths["cluster_diagnostics"] = diag_path
    if ripley_df is not None and not ripley_df.empty:
        ripley_path = (
            Path(config["paths"]["output_dir"])
            / "tables"
            / f"ripley_k_{run_id}.csv"
        )
        ripley_df.to_csv(ripley_path, index=False)
        logger.info("Saved Ripley's K diagnostics: %s", ripley_path)
        output_paths["ripley_k"] = ripley_path
    if density_df is not None and not density_df.empty:
        density_path = (
            Path(config["paths"]["output_dir"])
            / "tables"
            / f"kde_diagnostics_{run_id}.csv"
        )
        density_df.to_csv(density_path, index=False)
        logger.info("Saved KDE diagnostics: %s", density_path)
        output_paths["kde_diagnostics"] = density_path

    if config["plots"].get("make_histograms"):
        plot_histograms(points, config, run_id, logger)
    if config["plots"].get("make_cluster_prob_hist"):
        plot_cluster_prob_histogram(points, config, run_id, logger)
    if config["plots"].get("make_boxplots"):
        plot_boxplots(points, config, run_id, logger)
    if config["plots"].get("make_pairplot"):
        plot_pairplot(points, config, run_id, logger)

    n_noise = int(np.sum(cluster_labels == -1))
    n_clusters = int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
    figures_root = Path(config["paths"]["output_dir"]) / "figures"
    figure_files = (
        list(figures_root.rglob(f"*_{run_id}.svg")) if figures_root.exists() else []
    )
    summary = {
        "points_loaded": int(filter_stats.get("loaded_points", len(points))),
        "points_after_filter": int(filter_stats.get("remaining_points", len(points))),
        "points_assigned_tiles": points_assigned_tiles,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / len(cluster_labels)) if len(cluster_labels) else 0.0,
        "cluster_prob_mean": float(np.mean(cluster_prob)) if len(cluster_prob) else 0.0,
        "cluster_metrics": cluster_metrics,
        "point_filter": filter_stats,
        "taphonomy": taph_stats,
        "figure_count": len(figure_files),
        "spatial_diagnostics": {
            "nn_enabled": bool(config.get("cluster_diagnostics", {}).get("nn", {}).get("compute", False)),
            "ripley_enabled": bool(config.get("cluster_diagnostics", {}).get("ripley", {}).get("compute", False)),
            "ripley_radii": [
                float(r)
                for r in config.get("cluster_diagnostics", {}).get("ripley", {}).get("radii", [])
            ],
            "ripley_rows": int(len(ripley_df)) if ripley_df is not None else 0,
            "kde_enabled": bool(config.get("cluster_diagnostics", {}).get("density", {}).get("compute", False)),
            "kde_bandwidths": [
                float(b)
                for b in config.get("cluster_diagnostics", {}).get("density", {}).get("bandwidths", [])
            ],
            "kde_rows": int(len(density_df)) if density_df is not None else 0,
        },
    }
    if config.get("output", {}).get("save_report", True):
        report_path = write_run_report(config, run_id, logger, summary, output_paths)
        output_paths["report"] = report_path


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)
    logger = setup_logging(config)
    set_random_seed(int(config["random"]["seed"]))
    run_pipeline(config, logger)


if __name__ == "__main__":
    main()
