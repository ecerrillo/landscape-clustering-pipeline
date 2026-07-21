#!/usr/bin/env python3
"""
Raster processing for the landscape analysis pipeline.

Covers DEM tile reading, reprojection/resampling, geomorphometric derivatives
(slope, TPI), buffer-based sampling of those rasters at artefact locations, the seamless
elevation mosaic used by clustering, and the terrain-intrinsic taphonomic
process-zone classification.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.transform import xy as transform_xy
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from shapely import wkt
from shapely.geometry import Point, box
from shapely.ops import transform as shapely_transform

# GDAL is only needed for the directional_path_cost elevation mosaic. Importing
# osgeo.gdal eagerly registers GDAL's built-in Arrow/Parquet drivers, which
# claim the process-wide Arrow C++ 'file' filesystem factory; if pyarrow later
# tries to register its own, it raises ArrowKeyError. Checking availability via
# find_spec (rather than importing) avoids loading GDAL until it is actually
# needed, in build_dem_mosaic_vrt.
try:
    _GDAL_AVAILABLE = importlib.util.find_spec("osgeo.gdal") is not None
except ModuleNotFoundError:
    _GDAL_AVAILABLE = False


DEFAULT_NODATA = -9999.0
STAT_ORDER = ["mean"]


def scale_to_tag(scale: float) -> str:
    """Create a safe string tag for a scale value."""
    if float(scale).is_integer():
        return str(int(scale))
    return str(scale).replace(".", "p")


def make_tile_id(dem_path: str) -> str:
    """Generate a stable tile identifier based on DEM path."""
    stem = Path(dem_path).stem
    digest = hashlib.sha1(dem_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def read_dem_index(config: Dict[str, Any], logger: logging.Logger) -> gpd.GeoDataFrame:
    """Read DEM index CSV and return tile geometries in analysis CRS."""
    index_path = Path(config["paths"]["dem_index"])
    df = pd.read_csv(index_path)
    required = {"dem", "crs", "minx", "miny", "maxx", "maxy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DEM index missing columns: {missing}")

    analysis_crs = CRS.from_user_input(config["crs"]["analysis_crs"])
    records = []
    for _, row in df.iterrows():
        dem_path = row["dem"]
        dem_crs = row["crs"]
        if pd.isna(dem_path) or pd.isna(dem_crs):
            continue
        if "geometry" in df.columns and isinstance(row.get("geometry"), str):
            geom = wkt.loads(row["geometry"])
        else:
            geom = box(row["minx"], row["miny"], row["maxx"], row["maxy"])

        tile_crs = CRS.from_user_input(dem_crs)
        if tile_crs != analysis_crs:
            transformer = Transformer.from_crs(tile_crs, analysis_crs, always_xy=True)
            geom = shapely_transform(transformer.transform, geom)

        records.append(
            {
                "tile_id": make_tile_id(dem_path),
                "dem_path": dem_path,
                "dem_crs": dem_crs,
                "geometry": geom,
            }
        )

    tiles = gpd.GeoDataFrame(records, geometry="geometry", crs=analysis_crs)
    logger.info("Loaded %d DEM tiles from index.", len(tiles))
    return tiles


def reproject_resample_dem(
    src_path: Path,
    dst_path: Path,
    dst_crs: CRS,
    target_resolution: float,
    logger: logging.Logger,
) -> Path:
    """Reproject and resample a DEM to the analysis CRS and target resolution."""
    if dst_path.exists():
        logger.debug("Resampled DEM exists: %s", dst_path)
        return dst_path

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution,
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=src.nodata,
        )
        data = np.empty((height, width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=src.nodata,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)
    logger.info("Resampled DEM saved: %s", dst_path)
    return dst_path


def read_raster_as_float(src: rasterio.io.DatasetReader) -> np.ndarray:
    """Read a raster band as float32 with nodata converted to NaN."""
    arr = src.read(1).astype("float32")
    nodata = src.nodata
    if nodata is not None:
        if np.isnan(nodata):
            return arr
        arr[arr == nodata] = np.nan
    return arr


def write_raster(
    path: Path, data: np.ndarray, profile: Dict[str, Any], nodata: float
) -> None:
    """Write a single-band float raster with nodata."""
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    out = data.astype("float32")
    out = np.where(np.isnan(out), nodata, out)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)


def compute_slope(elev: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """Compute slope (degrees)."""
    dz_dy, dz_dx = np.gradient(elev, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad)


def circular_kernel(radius: int) -> np.ndarray:
    """Create a circular kernel for TPI calculation."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype("float32")


def compute_tpi(elev: np.ndarray, radius_pixels: int) -> np.ndarray:
    """
    Compute Topographic Position Index (TPI) using a circular kernel.

    TPI is computed at micro-topographic scales (e.g. 10-20 m) relevant to
    archaeological material dispersion and short-distance transport.
    """
    kernel = circular_kernel(radius_pixels)
    valid = ~np.isnan(elev)
    filled = np.where(valid, elev, 0.0)
    weights = convolve(valid.astype("float32"), kernel, mode="constant", cval=0.0)
    summed = convolve(filled, kernel, mode="constant", cval=0.0)
    mean = np.divide(summed, weights, out=np.full_like(summed, np.nan), where=weights > 0)
    return elev - mean


def compute_derivatives_for_tile(
    tile_id: str,
    dem_path: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Compute derivative rasters for a tile, skipping existing outputs."""
    rasters_dir = Path(config["paths"]["rasters_dir"])
    resampled_dir = rasters_dir / "dem_resampled"
    analysis_crs = CRS.from_user_input(config["crs"]["analysis_crs"])
    target_res = float(config["raster"]["target_resolution"])

    resampled_path = resampled_dir / f"{tile_id}.tif"
    resampled_path = reproject_resample_dem(
        dem_path, resampled_path, analysis_crs, target_res, logger
    )

    outputs: Dict[str, Path] = {"elev": resampled_path}
    with rasterio.open(resampled_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        elev = read_raster_as_float(src)
        res_x, res_y = src.res

        if config["raster"].get("slope"):
            slope = compute_slope(elev, res_x, res_y)
            slope_path = rasters_dir / "slope" / f"{tile_id}.tif"
            if not slope_path.exists():
                write_raster(slope_path, slope, profile, nodata)
                logger.info("Saved slope raster: %s", slope_path)
            outputs["slope"] = slope_path

        for scale in config["raster"].get("tpi_scales", []):
            scale = float(scale)
            radius_px = max(1, int(round(scale / target_res)))
            # Micro-topography at 10-20 m captures local relief relevant to short transport.
            tpi_path = rasters_dir / f"tpi{scale_to_tag(scale)}" / f"{tile_id}.tif"
            if not tpi_path.exists():
                tpi = compute_tpi(elev, radius_px)
                write_raster(tpi_path, tpi, profile, nodata)
                logger.info("Saved TPI raster: %s", tpi_path)
            outputs[f"tpi{scale_to_tag(scale)}"] = tpi_path

    return outputs


class _LogCollector:
    """Buffer log records produced inside a worker process.

    Worker processes spawned via ProcessPoolExecutor cannot share the parent
    logger's handlers (a FileHandler's open file descriptor is not picklable
    under the 'spawn' start method), so each tile's log lines are collected
    here and replayed through the real logger once the worker returns.
    """

    def __init__(self) -> None:
        self.records: List[Tuple[str, str]] = []

    def _log(self, level: str, msg: str, *args: Any) -> None:
        self.records.append((level, msg % args if args else msg))

    def info(self, msg: str, *args: Any) -> None:
        self._log("info", msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._log("warning", msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._log("debug", msg, *args)


def _compute_derivatives_for_tile_worker(
    args: Tuple[str, str, Dict[str, Any]]
) -> Tuple[str, Dict[str, Path], List[Tuple[str, str]]]:
    """Picklable entry point run in each worker process for one DEM tile."""
    tile_id, dem_path_str, config = args
    collector = _LogCollector()
    outputs = compute_derivatives_for_tile(tile_id, Path(dem_path_str), config, collector)
    return tile_id, outputs, collector.records


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute the buffer mean from a 1D array (the only statistic any
    downstream feature actually consumes; see STAT_ORDER)."""
    if values.size == 0:
        return {"mean": np.nan}
    return {"mean": float(np.nanmean(values))}


def raster_stats_for_point(
    src: rasterio.io.DatasetReader,
    point: Point,
    buffer_radius: float,
) -> Dict[str, float]:
    """Sample raster values within a buffer radius around a point.

    Buffer-based sampling handles ±2 m positional accuracy by summarizing
    micro-topographic conditions around each location.
    """
    buffer_geom = point.buffer(buffer_radius)
    minx, miny, maxx, maxy = buffer_geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, src.transform)
    # Clip to the dataset extent so the window transform matches the data that
    # src.read() actually returns for buffers overhanging the raster edge.
    try:
        window = window.intersection(Window(0, 0, src.width, src.height))
    except (WindowError, ValueError):
        return {stat: np.nan for stat in STAT_ORDER}
    data = src.read(1, window=window, masked=True)
    if data.size == 0:
        return {stat: np.nan for stat in STAT_ORDER}

    window_transform = src.window_transform(window)
    rows, cols = np.indices(data.shape)
    xs, ys = transform_xy(window_transform, rows, cols, offset="center")
    xs = np.array(xs)
    ys = np.array(ys)
    if xs.ndim == 1 and xs.size == data.size:
        xs = xs.reshape(data.shape)
        ys = ys.reshape(data.shape)
    dist = np.sqrt((xs - point.x) ** 2 + (ys - point.y) ** 2)
    mask = dist <= buffer_radius
    values = data.filled(np.nan)[mask]
    values = values[~np.isnan(values)]
    return compute_stats(values)


def sample_points_by_tile(
    points: gpd.GeoDataFrame,
    tile_rasters: Dict[str, Dict[str, Path]],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Sample raster statistics for points grouped by tile."""
    buffer_radius = float(config["sampling"]["buffer_radius"])
    results = pd.DataFrame(index=points.index)
    results["x"] = points.geometry.x
    results["y"] = points.geometry.y

    for tile_id, idxs in points.groupby("tile_id").groups.items():
        if tile_id not in tile_rasters:
            logger.warning("Missing rasters for tile %s", tile_id)
            continue
        raster_paths = tile_rasters[tile_id]
        with ExitStack() as stack:
            datasets = {
                name: stack.enter_context(rasterio.open(path))
                for name, path in raster_paths.items()
            }
            for idx in idxs:
                point = points.at[idx, "geometry"]
                for name, src in datasets.items():
                    stats = raster_stats_for_point(src, point, buffer_radius)
                    for stat_name, stat_val in stats.items():
                        results.at[idx, f"{name}_{stat_name}"] = stat_val

    enriched = points.join(results)
    return enriched


def build_dem_mosaic_vrt(rasters_dir: Path, logger: logging.Logger) -> Path:
    """Build a seamless elevation VRT mosaic from all resampled DEM tiles.

    A VRT lets path-cost sampling read elevation continuously across tile
    boundaries instead of being confined to each point's assigned tile.
    """
    if not _GDAL_AVAILABLE:
        raise ImportError("GDAL Python bindings (osgeo.gdal) are required to build the DEM mosaic.")
    from osgeo import gdal  # imported lazily; see _GDAL_AVAILABLE note above

    resampled_dir = rasters_dir / "dem_resampled"
    tile_paths = sorted(str(p) for p in resampled_dir.glob("*.tif"))
    if not tile_paths:
        raise ValueError("No resampled DEM tiles found to build elevation mosaic.")
    with rasterio.open(tile_paths[0]) as ref:
        tile_nodata = ref.nodata if ref.nodata is not None else DEFAULT_NODATA
    vrt_path = resampled_dir / "mosaic.vrt"
    # Set nodata explicitly so gaps between non-contiguous tiles (uncovered by any
    # source) are masked out rather than silently filled with 0 by GDAL's default.
    vrt_options = gdal.BuildVRTOptions(srcNodata=tile_nodata, VRTNodata=tile_nodata)
    vrt_ds = gdal.BuildVRT(str(vrt_path), tile_paths, options=vrt_options)
    if vrt_ds is None:
        raise RuntimeError("gdal.BuildVRT failed to build the elevation mosaic.")
    vrt_ds = None  # noqa: F841 - flush/close the dataset handle
    logger.info("Built elevation mosaic VRT from %d tiles: %s", len(tile_paths), vrt_path)
    return vrt_path


PROCESS_ZONE_CODES = {0: "undetermined", 1: "stable", 2: "source", 3: "transit", 4: "accumulation"}


def _focal_mean_fft(arr: np.ndarray, radius_px: int) -> np.ndarray:
    """NaN-aware circular focal mean via FFT convolution (fast on large mosaics)."""
    kernel = circular_kernel(radius_px).astype("float64")
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype("float64")
    num = fftconvolve(filled, kernel, mode="same")
    den = fftconvolve(valid.astype("float64"), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


def compute_process_zone_raster(
    elevation_mosaic_path: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """
    Compute a spatially coherent taphonomic process-zone raster from the DEM mosaic.

    Instead of classifying artefact locations independently from noisy
    exact-point raster values (which produces salt-and-pepper class patterns
    with no spatial coherence), the terrain itself is classified into
    contiguous process zones following slope-position logic (Weiss 2001;
    catena / erosion-transport-deposition cascade), and artefacts inherit the
    zone they fall in. Inputs are smoothed at landform scale: slope is
    focal-averaged (default 15 m radius) and relative landscape position is a
    coarse-scale TPI (default 100 m radius) standardised by its global
    standard deviation. Cell rules (theta = critical slope; t = TPI SD
    threshold):

    - upper positions (z >= +t):   steep -> source;  gentle -> stable
    - mid positions   (|z| < t):   steep -> transit; gentle -> stable
    - lower positions (z <= -t):   steep -> transit; gentle -> accumulation

    The raster is written to rasters/taphonomy/process_zones.tif (uint8 codes
    per PROCESS_ZONE_CODES) so the zoning can be inspected directly in a GIS
    against hillshade and field knowledge.
    """
    taph_cfg = config.get("taphonomy", {})
    theta = float(taph_cfg.get("critical_slope_deg", 10.0))
    slope_smooth_radius = float(taph_cfg.get("slope_smooth_radius_m", 15.0))
    coarse_tpi_radius = float(taph_cfg.get("coarse_tpi_radius_m", 100.0))
    sd_threshold = float(taph_cfg.get("tpi_sd_threshold", 0.5))

    with rasterio.open(elevation_mosaic_path) as src:
        elev = src.read(1).astype("float64")
        nodata = src.nodata
        profile = src.profile.copy()
        res_x, res_y = src.res
    if nodata is not None and not np.isnan(nodata):
        elev[elev == nodata] = np.nan

    dz_dy, dz_dx = np.gradient(elev, res_y, res_x)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    slope_radius_px = max(1, int(round(slope_smooth_radius / res_x)))
    tpi_radius_px = max(1, int(round(coarse_tpi_radius / res_x)))
    slope_smooth = _focal_mean_fft(slope_deg, slope_radius_px)
    tpi_coarse = elev - _focal_mean_fft(elev, tpi_radius_px)

    tpi_sd = float(np.nanstd(tpi_coarse))
    if tpi_sd <= 0 or not np.isfinite(tpi_sd):
        raise ValueError("Coarse TPI has zero variance; cannot classify process zones.")
    z = tpi_coarse / tpi_sd

    zones = np.zeros(elev.shape, dtype="uint8")
    valid = np.isfinite(z) & np.isfinite(slope_smooth)
    steep = slope_smooth >= theta
    upper = valid & (z >= sd_threshold)
    lower = valid & (z <= -sd_threshold)
    mid = valid & ~upper & ~lower

    zones[upper & steep] = 2      # source
    zones[upper & ~steep] = 1     # stable
    zones[mid & steep] = 3        # transit
    zones[mid & ~steep] = 1       # stable
    zones[lower & steep] = 3      # transit
    zones[lower & ~steep] = 4     # accumulation

    out_dir = Path(config["paths"]["rasters_dir"]) / "taphonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    zones_path = out_dir / "process_zones.tif"
    zone_profile = profile.copy()
    zone_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", driver="GTiff")
    with rasterio.open(zones_path, "w", **zone_profile) as dst:
        dst.write(zones, 1)

    tpi_path = out_dir / "tpi_coarse.tif"
    tpi_profile = profile.copy()
    tpi_profile.update(dtype="float32", count=1, nodata=DEFAULT_NODATA, compress="deflate", driver="GTiff")
    with rasterio.open(tpi_path, "w", **tpi_profile) as dst:
        dst.write(np.where(np.isfinite(tpi_coarse), tpi_coarse, DEFAULT_NODATA).astype("float32"), 1)

    zone_counts = {PROCESS_ZONE_CODES[c]: int(n) for c, n in zip(*np.unique(zones, return_counts=True))}
    logger.info(
        "Process-zone raster written: %s (theta=%.1f deg, tpi_radius=%.0fm, sd_threshold=%.2f, tpi_sd=%.2fm, cells=%s)",
        zones_path,
        theta,
        coarse_tpi_radius,
        sd_threshold,
        tpi_sd,
        zone_counts,
    )
    return zones_path


def classify_taphonomic_context(
    points: gpd.GeoDataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    elevation_mosaic_path: Optional[Path],
) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
    """
    Assign each point the taphonomic process zone it falls in.

    Zones come from a smoothed, spatially coherent terrain classification (see
    compute_process_zone_raster); the classification of a point therefore
    depends only on the landform it sits on, never on other recorded artefacts
    or on single-pixel raster noise.
    """
    taph_cfg = config.get("taphonomy", {})
    if not taph_cfg or not taph_cfg.get("enable", False):
        return points, {}
    if elevation_mosaic_path is None:
        logger.warning("Taphonomic classification skipped; elevation mosaic unavailable.")
        return points, {}

    zones_path = compute_process_zone_raster(elevation_mosaic_path, config, logger)

    with rasterio.open(zones_path) as src:
        coords = [(geom.x, geom.y) for geom in points.geometry]
        sampled = np.array([v[0] for v in src.sample(coords)], dtype="int32")

    context = np.array([PROCESS_ZONE_CODES.get(int(c), "undetermined") for c in sampled], dtype=object)
    points = points.copy()
    points["taph_context"] = context

    counts = {str(k): int(v) for k, v in pd.Series(context).value_counts().items()}
    logger.info("Taphonomic context assigned from process zones: %s", counts)
    return points, counts
