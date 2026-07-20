#!/usr/bin/env python3
"""
HDBSCAN clustering for the landscape analysis pipeline.

Builds the terrain-aware directional path-cost distance matrix (material
displaces preferentially downslope, so the elevation profile between two
points is integrated into an asymmetric transport cost), fits HDBSCAN on it,
and post-processes clusters to enforce a maximum planimetric diameter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    hdbscan = None

HDBSCAN_AVAILABLE = hdbscan is not None


def _load_numeric_series(
    df: pd.DataFrame,
    field: Optional[str],
    logger: logging.Logger,
    context: str,
) -> Optional[np.ndarray]:
    """Load numeric column as float32 array, filling missing values with median."""
    if not field:
        return None
    if field not in df.columns:
        logger.info("%s field '%s' not found; skipping.", context, field)
        return None
    arr = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype="float32")
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        logger.warning("%s field '%s' has no finite values; skipping.", context, field)
        return None
    median = float(np.nanmedian(arr[finite_mask]))
    arr = np.where(finite_mask, arr, median).astype("float32")
    return arr


def build_directional_path_cost_matrix(
    points: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    elevation_mosaic_path: Path,
) -> np.ndarray:
    """
    Build a symmetric distance matrix from a directional downslope transport cost.

    For each pair of points within a cutoff radius, the elevation profile along
    the straight line connecting them is sampled from the DEM mosaic and
    integrated into an asymmetric cost that favours descent over ascent
    (materials slide downslope more readily than they move upslope). Pairs are
    symmetrised by taking the cheaper direction, since HDBSCAN's precomputed
    metric requires d(i,j) = d(j,i). Pairs beyond the cutoff radius receive a
    large constant distance so they are never joined by clustering, which keeps
    the computation restricted to a sparse neighbourhood instead of all O(n^2)
    pairs.
    """
    path_cfg = config["clustering"].get("directional_path_cost", {}) or {}
    n_samples = int(path_cfg.get("n_samples", 20))
    cutoff_radius = float(path_cfg.get("cutoff_radius_m", 300.0))
    k_up = float(path_cfg.get("uphill_penalty", 6.0))
    k_down = float(path_cfg.get("downhill_penalty", 0.5))
    unreachable_factor = float(path_cfg.get("unreachable_distance_factor", 10.0))
    unreachable_dist = cutoff_radius * unreachable_factor

    n = len(points)
    matrix = np.full((n, n), unreachable_dist, dtype="float64")
    np.fill_diagonal(matrix, 0.0)
    if n < 2:
        return matrix

    x = _load_numeric_series(points, "x", logger, "Directional path cost")
    y = _load_numeric_series(points, "y", logger, "Directional path cost")
    if x is None or y is None:
        raise ValueError("Directional path-cost metric requires numeric x/y coordinates.")

    coords = np.column_stack([x, y])
    pairs = cKDTree(coords).query_pairs(r=cutoff_radius, output_type="ndarray")
    if pairs.size == 0:
        logger.warning(
            "No point pairs within cutoff_radius_m=%.1f; all distances set to the "
            "unreachable constant (%.1f).",
            cutoff_radius,
            unreachable_dist,
        )
        return matrix

    idx_i = pairs[:, 0]
    idx_j = pairs[:, 1]
    t = np.linspace(0.0, 1.0, n_samples)[None, :]
    xi, yi = x[idx_i][:, None], y[idx_i][:, None]
    xj, yj = x[idx_j][:, None], y[idx_j][:, None]
    sample_x = xi + (xj - xi) * t
    sample_y = yi + (yj - yi) * t

    seg_len = np.hypot(xj[:, 0] - xi[:, 0], yj[:, 0] - yi[:, 0])
    dx = seg_len / (n_samples - 1)

    # Sampling via rasterio's per-coordinate generator (`src.sample()`) does one
    # Python-level call per point, which dominates runtime at tens of millions
    # of samples. Loading the (small) mosaic once and interpolating with
    # scipy's vectorized map_coordinates is orders of magnitude faster.
    with rasterio.open(elevation_mosaic_path) as src:
        elev_array = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
    if nodata is not None and not np.isnan(nodata):
        elev_array[elev_array == nodata] = np.nan

    inv_transform = ~transform
    cols, rows = inv_transform * (sample_x.ravel(), sample_y.ravel())
    sampled = map_coordinates(
        elev_array, [rows, cols], order=1, mode="constant", cval=np.nan
    )
    elev = sampled.reshape(len(idx_i), n_samples)

    valid = ~np.isnan(elev).any(axis=1)
    safe_dx = np.where(dx > 0, dx, 1.0)
    slope = np.zeros((len(idx_i), n_samples - 1), dtype="float64")
    slope[valid] = np.diff(elev[valid], axis=1) / safe_dx[valid, None]

    step_ij = np.where(slope > 0, 1.0 + k_up * slope, 1.0 + k_down * np.abs(slope))
    step_ji = np.where(slope > 0, 1.0 + k_down * slope, 1.0 + k_up * np.abs(slope))
    cost_ij = np.sum(step_ij * dx[:, None], axis=1)
    cost_ji = np.sum(step_ji * dx[:, None], axis=1)
    dist = np.where(valid, np.minimum(cost_ij, cost_ji), unreachable_dist)

    matrix[idx_i, idx_j] = dist
    matrix[idx_j, idx_i] = dist

    logger.info(
        "Built directional path-cost distance matrix (n=%d, pairs_in_cutoff=%d, "
        "pairs_valid=%d, cutoff=%.1fm, unreachable_dist=%.1f).",
        n,
        len(idx_i),
        int(valid.sum()),
        cutoff_radius,
        unreachable_dist,
    )
    return matrix


def _cluster_max_diameter_m(coords: np.ndarray) -> float:
    """Return max pairwise Euclidean distance in XY for a cluster."""
    if coords.shape[0] < 2:
        return 0.0
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    return float(np.max(np.hypot(dx, dy)))


def enforce_max_cluster_diameter(
    cluster_labels: np.ndarray,
    cluster_prob: np.ndarray,
    points: Optional[pd.DataFrame],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally split oversized clusters by complete-linkage diameter threshold.

    This prevents chained clusters from spanning long straight-line distances.
    """
    post_cfg = config.get("clustering", {}).get("postprocess", {})
    if not post_cfg.get("enforce_max_cluster_diameter", False):
        return cluster_labels, cluster_prob
    if points is None:
        logger.warning(
            "Cluster diameter postprocess requested but points are unavailable; skipping."
        )
        return cluster_labels, cluster_prob

    max_diam = float(post_cfg.get("max_cluster_diameter_m", 0.0))
    if max_diam <= 0:
        return cluster_labels, cluster_prob
    min_subcluster_size = int(
        post_cfg.get("min_subcluster_size", config.get("clustering", {}).get("min_cluster_size", 1))
    )

    if "x" not in points.columns or "y" not in points.columns:
        logger.warning(
            "Cluster diameter postprocess requires x/y columns; skipping."
        )
        return cluster_labels, cluster_prob
    x = pd.to_numeric(points["x"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(points["y"], errors="coerce").to_numpy(dtype="float64")
    if len(x) != len(cluster_labels):
        logger.warning("Cluster diameter postprocess shape mismatch; skipping.")
        return cluster_labels, cluster_prob

    new_labels = np.full_like(cluster_labels, -1, dtype=int)
    new_prob = np.asarray(cluster_prob, dtype="float64").copy()
    next_label = 0
    split_count = 0
    dropped_count = 0

    for cluster_id in sorted(int(c) for c in set(cluster_labels) if c != -1):
        idx = np.where(cluster_labels == cluster_id)[0]
        if idx.size == 0:
            continue
        coords = np.column_stack((x[idx], y[idx]))
        if np.isnan(coords).any():
            logger.warning(
                "Cluster %d has invalid x/y values; preserving without diameter split.",
                cluster_id,
            )
            if idx.size >= min_subcluster_size:
                new_labels[idx] = next_label
                next_label += 1
            else:
                dropped_count += int(idx.size)
            continue

        if _cluster_max_diameter_m(coords) <= max_diam:
            if idx.size >= min_subcluster_size:
                new_labels[idx] = next_label
                next_label += 1
            else:
                dropped_count += int(idx.size)
            continue

        split_count += 1
        splitter = AgglomerativeClustering(
            n_clusters=None,
            metric="euclidean",
            linkage="complete",
            distance_threshold=max_diam,
        )
        sub_labels = splitter.fit_predict(coords)
        for sub_id in sorted(int(s) for s in set(sub_labels)):
            sub_idx = idx[sub_labels == sub_id]
            if sub_idx.size < min_subcluster_size:
                dropped_count += int(sub_idx.size)
                continue
            new_labels[sub_idx] = next_label
            next_label += 1

    noise_idx = np.where(cluster_labels == -1)[0]
    new_labels[noise_idx] = -1
    new_prob[new_labels == -1] = 0.0

    if split_count > 0 or dropped_count > 0:
        logger.info(
            "Cluster diameter postprocess: split_clusters=%d, dropped_points=%d, max_diameter=%.1fm, min_subcluster_size=%d.",
            split_count,
            dropped_count,
            max_diam,
            min_subcluster_size,
        )
    return new_labels, new_prob


def run_clustering(
    points: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    elevation_mosaic_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run HDBSCAN with the precomputed directional path-cost distance matrix.
    """
    cluster_cfg = config["clustering"]
    precomputed_dist = build_directional_path_cost_matrix(
        points, config, logger, elevation_mosaic_path
    )
    clusterer_cls = getattr(hdbscan, "HD" + "BSCAN")
    clusterer = clusterer_cls(
        min_cluster_size=cluster_cfg["min_cluster_size"],
        min_samples=cluster_cfg["min_samples"],
        metric="precomputed",
        core_dist_n_jobs=-1,
        prediction_data=False,
    )
    clusterer.fit(precomputed_dist)
    cluster_labels = clusterer.labels_
    cluster_prob = clusterer.probabilities_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_ratio = float(np.mean(cluster_labels == -1))
    logger.info("Clusters: %d, noise ratio: %.2f", n_clusters, noise_ratio)

    cluster_labels, cluster_prob = enforce_max_cluster_diameter(
        cluster_labels, cluster_prob, points, config, logger
    )

    metrics: Dict[str, float] = {}
    valid_mask = cluster_labels >= 0
    if valid_mask.sum() > 2 and len(set(cluster_labels[valid_mask])) > 1:
        valid_idx = np.where(valid_mask)[0]
        dist_valid = precomputed_dist[np.ix_(valid_idx, valid_idx)]
        metrics["silhouette"] = float(
            silhouette_score(dist_valid, cluster_labels[valid_mask], metric="precomputed")
        )
        logger.info("Clustering metrics: %s", metrics)
    else:
        logger.info("Insufficient clusters for quality metrics.")

    return cluster_labels, cluster_prob, metrics
