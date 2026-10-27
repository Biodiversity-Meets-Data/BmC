import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import random
import os

from osgeo import gdal
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import dask
import rasterio
from rasterio.session import Session
from rasterio.session import AWSSession
import rioxarray
import xarray as xr

from bmc.utils.logger import log_execution

logger = logging.getLogger(__name__)

# Base settings safe for ALL remote HTTP/VSI datasets (CHELSA, CDSE, STAC, etc.)
BASE_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",  # Restore: Reduces network round-trips
    "GDAL_HTTP_TIMEOUT": "30",                    # Restore: Prevents hung threads
    "GDAL_HTTP_MAX_RETRY": "5",                   # Restore: Fast C-level retries
    "GDAL_HTTP_RETRY_DELAY": "3",
    "VSI_CACHE": "FALSE",                         # Restore: Prevents RAM bloat
}

# CDSE / S3-specific overrides
CDSE_GDAL_ENV = {
    **BASE_GDAL_ENV,
    "AWS_VIRTUAL_HOSTING": "FALSE",  # Required specifically for CDSE path-style routing
}

def _fetch_single_raster(
    path: str, 
    geom: Union[Tuple[float, float, float, float], Any], 
    gdal_env: Optional[Dict[str, str]] = None,
    geom_crs: Optional[str] = None,
    max_retries: int = 3,
    cloud_session: Optional[Session] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Optional[xr.DataArray]]:
    """
    Thread-safe worker to fetch and clip a remote raster (CDSE S3 or CHELSA VSI/HTTP).
    """
    gdal_env = gdal_env or BASE_GDAL_ENV

    # --- Path Normalization ---
    # Convert /vsis3/ paths to s3:// when AWSSession is present
    target_path = path
    if cloud_session and isinstance(cloud_session, AWSSession):
        if target_path.startswith("/vsis3/"):
            target_path = target_path.replace("/vsis3/", "s3://")

    # Network jitter to prevent thread stampedes
    time.sleep(random.uniform(0.1, 1.0))

    # Build environmental context
    env_context = rasterio.Env(session=cloud_session, **gdal_env) if cloud_session else rasterio.Env(**gdal_env)

    for attempt in range(1, max_retries + 1):
        try:
            with env_context:
                with dask.config.set(scheduler='single-threaded'):
                    with rioxarray.open_rasterio(target_path, chunks=True, masked=True) as src:
                        
                        # Route clipping based on geometry type
                        if isinstance(geom, tuple) and len(geom) == 4:
                            # Pass geom_crs if provided and different from raster CRS
                            if geom_crs and src.rio.crs and geom_crs != src.rio.crs.to_string():
                                clipped_lazy = src.rio.clip_box(*geom, crs=geom_crs)
                            else:
                                clipped_lazy = src.rio.clip_box(*geom)
                        else:
                            geometries = geom if isinstance(geom, (list, tuple)) else [geom]
                            clipped_lazy = src.rio.clip(geometries, crs=geom_crs, drop=True)

                        # Download the clipped region into memory
                        clipped = clipped_lazy.compute()

                        # Structural Integrity Check
                        if clipped.size == 0:
                            raise ValueError(f"Clipping resulted in 0 pixels for bounds {geom}.")

                        return path, clipped

        except Exception as e:
            if attempt < max_retries:
                backoff_time = 5.0 + random.uniform(0.0, 3.0)
                log_execution(
                    logger, 
                    f"Network anomaly on '{path}'. Retrying in {backoff_time:.1f}s (Attempt {attempt}/{max_retries}). Error: {e}", 
                    logging.WARNING
                )
                time.sleep(backoff_time)
            else:
                log_execution(
                    logger, 
                    f"FATAL: Failed to fetch asset '{path}' after {max_retries} attempts. Reason: {e}", 
                    logging.ERROR
                )
                return path, None


def parallel_fetch_rasters(
    paths: Union[str, List[str]],
    geom: Union[Tuple[float, float, float, float], Any],
    max_workers: int = 10,
    cloud_session: Optional[Session] = None,
    gdal_config: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    geom_crs: Optional[str] = None
) -> Dict[str, xr.DataArray]:
    """
    Executes highly parallelized, optimized extractions across a generalized list of rasters.

    This utility spins up a concurrent thread pool to fetch, clip, and load raster 
    assets from any list of URIs or local paths. It acts as a universal geospatial 
    I/O engine completely decoupled from specific dataset metadata.

    Parameters
    ----------
    paths : str or list of str
        A single file path/URI or a list of file paths/URIs to be downloaded and clipped.
    geom : tuple of float or Any (Shapely geometry, GeoJSON dict)
        The spatial geometry used to clip the raster. Accepts either a rectangular 
        bounding box tuple (min_x, min_y, max_x, max_y) or a complex polygon geometry.
    max_workers : int, optional
        The maximum number of concurrent threads to spawn for network I/O. 
        Default is 10.
    cloud_session : rasterio.session.Session, optional
        An authenticated cloud session (e.g., AWSSession, GSSession) to use for fetching.
        If None, the environment configures for anonymous public bucket access.
    gdal_config : dict, optional
        Overrides for the GDAL environment configuration. If None, a highly 
        optimized default configuration for Cloud Optimized GeoTIFF (COG) 
        HTTP streaming is applied. Default is None.
    geom_crs : str, optional
        The coordinate reference system of the provided polygon geometry 
        (e.g., "EPSG:4326"). Required if passing a polygon that differs from 
        the source raster's CRS. Default is None.

    Returns
    -------
    dict
        A dictionary mapping the original file path (str) directly to its 
        in-memory `xarray.DataArray`.
    """
    # Start with the base GDAL configuration or the optimized default
    gdal_env = (gdal_config or BASE_GDAL_ENV).copy()

    # Apply dynamic Configuration based on active Session
    if cloud_session:
        if "AWS_VIRTUAL_HOSTING" not in gdal_env:
            gdal_env["AWS_VIRTUAL_HOSTING"] = "FALSE"
        if "AWS_HTTPS" not in gdal_env:
            gdal_env["AWS_HTTPS"] = "YES"
    else:
        gdal_env["AWS_NO_SIGN_REQUEST"] = "YES"

    if isinstance(paths, str):
        paths = [paths]

    geom_type = "bounding box" if isinstance(geom, tuple) and len(geom) == 4 else "polygon mask"
    log_execution(
        logger, 
        f"Initiating parallel raster fetch across {len(paths)} assets using {max_workers} workers (Geometry: {geom_type})...", 
        logging.INFO
    )

    fetched_arrays: Dict[str, xr.DataArray] = {}
    failed_fetches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            # Explicitly pass all keyword/positional arguments safely, including logger and max_retries
            executor.submit(
                _fetch_single_raster, 
                path=path, 
                geom=geom, 
                gdal_env=gdal_env, 
                geom_crs=geom_crs, 
                max_retries=max_retries, 
                cloud_session=cloud_session, 
                logger=logger
            ): path
            for path in paths
        }    
        
        for future in as_completed(futures):
            path, data = future.result()
            if data is not None:
                fetched_arrays[path] = data
            else:
                failed_fetches += 1

    if failed_fetches > 0:
        log_execution(
            logger, 
            f"Parallel ingest completed with {failed_fetches} network fetch failures.", 
            logging.WARNING
        )
    else:
        log_execution(
            logger, 
            "Parallel ingest completed successfully with zero fetch failures.", 
            logging.INFO
        )
    return fetched_arrays


def filter_and_group_catalog(catalog_path, bbox, start_date, end_date, group_cols=None):
    """
    Filters a GeoParquet catalog spatially and temporally, optionally grouping 
    results to aggregate S3 paths for Virtual Raster (VRT) generation.

    This function enables rapid preprocessing of large Earth observation 
    catalogs. By passing specific grouping columns (like 'datetime' and 
    'proj:epsg'), it clusters intersecting tiles into consistent coordinate 
    and temporal systems, which is required for building valid GDAL VRTs.

    Parameters
    ----------
    catalog_path : str
        The file path to the input GeoParquet catalog.
    bbox : list or tuple of float
        The spatial bounding box in the format [west, south, east, north]. 
        Coordinates must match the CRS of the catalog (typically EPSG:4326).
    start_date : str or datetime.datetime
        The start of the temporal bounding box (inclusive). Accepts standard 
        date strings (e.g., '2017-01-01').
    end_date : str or datetime.datetime
        The end of the temporal bounding box (inclusive). Accepts standard 
        date strings (e.g., '2020-12-31').
    group_cols : list of str, optional
        Columns by which to group the filtered dataframe. Defaults to None.

    Returns
    -------
    dict or list
        If `group_cols` is provided, returns a dictionary where keys are the 
        unique grouping combinations and values are lists of `s3_href` strings. 
        If `group_cols` is None, returns a flat list of all matching `s3_href` 
        strings. Returns an empty dict or list if no intersections are found.

    Examples
    --------
    Extracting a flat list of tiles for a specific region:
    
    >>> bbox = [3.5, 50.5, 4.0, 51.0]
    >>> filter_and_group_catalog("catalog.parquet", bbox, "2018-01-01", "2018-12-31")
    ['s3://eodata/tile1.tif', 's3://eodata/tile2.tif']

    Grouping tiles by year and CRS for mosaicking:
    
    >>> filter_and_group_catalog("catalog.parquet", bbox, "2017-01-01", "2020-12-31", ['datetime', 'proj:epsg'])
    {(Timestamp('2018-01-01'), 3035): ['s3://eodata/t1.tif', ...], ...}
    """
    # --- 1. Load the Catalog ---
    # Read the pre-indexed GeoParquet file into memory. 
    # This format allows spatial operations directly on the dataframe.
    gdf = gpd.read_parquet(catalog_path)
    
    # --- 2. Spatial Filter ---
    # Construct a Shapely polygon from the bounding box [w, s, e, n]
    search_geom = box(*bbox)
    
    # Filter the GeoDataFrame for any geometries that intersect the search box.
    # We use .copy() to decouple the filtered data from the original dataframe, 
    # preventing pandas SettingWithCopyWarnings during downstream manipulations.
    filtered_gdf = gdf[gdf.intersects(search_geom)].copy()
    
    # --- 3. Temporal Filter ---
    # Convert input strings to pandas Timestamp objects for reliable comparisons
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Filter for records that fall on or between the start and end dates
    filtered_gdf = filtered_gdf[
        (filtered_gdf['datetime'] >= start) & 
        (filtered_gdf['datetime'] <= end)
    ]
    
    # Early exit if the combined spatial/temporal filters yield no results
    if filtered_gdf.empty:
        print("No matching tiles found for the given criteria.")
        return {} if group_cols else []

    # --- 4. Grouping Logic ---
    if group_cols:
        # Group the dataframe by the requested columns (e.g., year, EPSG).
        # We select only the 's3_href' column and apply list() to collapse 
        # all matching S3 paths for each group into a single list element.
        grouped = filtered_gdf.groupby(group_cols)['s3_href'].apply(list)
        
        # Convert the resulting pandas Series into a standard Python dictionary.
        # Format: {(group_val_1, group_val_2): ['s3://...', 's3://...']}
        return grouped.to_dict()
    
    # If no grouping was requested, return a flat Python list of all matching hrefs
    return filtered_gdf['s3_href'].tolist()


from typing import Optional
from osgeo import gdal
import os
import rasterio
from rasterio.session import Session

def s3_to_vrt(
    s3_hrefs, 
    vrt_path, 
    cloud_session=None,
    export_tif_path=None, 
    resample_alg="near",
    creation_options=None,
    output_srs=None,
    output_bounds=None,
    env_kwargs=None
):
    """
    Constructs a GDAL Virtual Raster (VRT) from a list of S3 raster URIs, 
    enabling lightweight, virtual mosaicking and optional local GeoTIFF export.

    This function configures GDAL's Virtual File System (`/vsis3/`) to stream 
    remote cloud raster assets dynamically. By creating a VRT, GDAL maps the 
    spatial footprints of input rasters without downloading pixel data upfront. 
    If an `export_tif_path` is specified, it streams and compresses the mosaic 
    into a local GeoTIFF file.

    Parameters
    ----------
    s3_hrefs : list of str
        A list of S3 URIs (e.g., `['s3://bucket/raster1.tif', ...]`) to be 
        mosaicked into the virtual dataset.
    vrt_path : str
        The output target path for the VRT file. Can be a physical file path 
        (e.g., `./mosaic.vrt`) or an in-memory virtual file (e.g., `/vsimem/mosaic.vrt`).
    cloud_session : rasterio.session.Session, optional
        An authenticated cloud session (e.g., AWSSession) to use for fetching.
        If None, GDAL configures for anonymous public bucket access.
    export_tif_path : str, optional
        An optional local file path to write a compiled GeoTIFF raster. If 
        None, only the VRT is constructed. Defaults to None.
    resample_alg : str, optional
        The resampling algorithm used when pixel interpolation or reprojection 
        occurs (e.g., `'near'`, `'bilinear'`, `'cubic'`). Defaults to `'near'`, 
        which preserves discrete categorical class values.
    creation_options : list of str, optional
        Custom driver creation options passed to `gdal.TranslateOptions` if 
        exporting a GeoTIFF. Defaults to `["COMPRESS=DEFLATE", "TILED=YES", 
        "BIGTIFF=IF_SAFER"]`.
    output_srs : str, optional
        The target spatial reference system (e.g., "EPSG:3035"). Overrides GDAL's 
        strict metadata matching to prevent heterogeneous projection errors.
    output_bounds : tuple, optional
        The bounding box for the output VRT in (minX, minY, maxX, maxY) format.
    env_kwargs : dict, optional
        Explicit GDAL configuration variables (e.g., AWS_S3_ENDPOINT).

    Returns
    -------
    bool
        Returns True if the VRT (and optional GeoTIFF) was generated successfully; 
        returns False if the input list is empty or GDAL fails to build the VRT.
    """
    if not s3_hrefs:
        print("No S3 URIs provided. Skipping VRT construction.")
        return False

    # --- 1. Standardize Prefixes ---
    # Convert 's3://' paths into GDAL '/vsis3/' virtual filesystem paths
    vsi_paths = [p.replace("s3://", "/vsis3/") for p in s3_hrefs]

    print(f"Constructing VRT from {len(vsi_paths)} remote raster source(s)...")

    # --- 2. Build the GDAL Environment Context ---
    env_kwargs = env_kwargs or {}
    gdal_env = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'AWS_VIRTUAL_HOSTING': 'FALSE' if cloud_session else 'TRUE'
    }
    
    # Merge custom routing (like AWS_S3_ENDPOINT for Copernicus)
    gdal_env.update(env_kwargs)
    
    if not cloud_session:
        gdal_env['AWS_NO_SIGN_REQUEST'] = 'YES'
    else:
        # Extract credentials natively from Rasterio's AWSSession to bridge the gap 
        # to osgeo.gdal C++ bindings, which execute independently of rasterio.Env
        try:
            creds = cloud_session._creds
            if creds:
                gdal_env['AWS_ACCESS_KEY_ID'] = creds.access_key
                gdal_env['AWS_SECRET_ACCESS_KEY'] = creds.secret_key
                if hasattr(creds, 'token') and creds.token:
                    gdal_env['AWS_SESSION_TOKEN'] = creds.token
        except Exception as e:
            print(f"Warning: Failed to extract credentials from AWSSession: {e}")

    # --- 3. Enforce Configuration across GDAL & Rasterio ---
    # Explicitly set the C++ configuration for the osgeo.gdal native module
    for k, v in gdal_env.items():
        gdal.SetConfigOption(k, str(v))

    try:
        # Scrub restricted credential keys before passing to Rasterio
        # Rasterio will extract them natively from the cloud_session object
        rasterio_safe_env = {
            k: v for k, v in gdal_env.items() 
            if k not in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
        }
        
        # Secure the VRT generation step inside an active session context for rioxarray
        env_context = rasterio.Env(session=cloud_session, **rasterio_safe_env) if cloud_session else rasterio.Env(**rasterio_safe_env)

        with env_context:
            # --- 4. Construct Virtual Raster (VRT) ---
            vrt_options = gdal.BuildVRTOptions(
                resampleAlg=resample_alg,
                addAlpha=False,
                outputSRS=output_srs,
                outputBounds=output_bounds,
                resolution='highest',
                options=["-allow_projection_difference"]
            )
            
            vrt_ds = gdal.BuildVRT(vrt_path, vsi_paths, options=vrt_options)
            
            if vrt_ds is None:
                print("Failed to construct GDAL VRT dataset.")
                return False

            # --- 5. Optional GeoTIFF Export ---
            if export_tif_path:
                print(f"Translating virtual mosaic to physical raster: {export_tif_path}")
                
                target_dir = os.path.dirname(os.path.abspath(export_tif_path))
                os.makedirs(target_dir, exist_ok=True)
                
                if creation_options is None:
                    creation_options = [
                        "COMPRESS=DEFLATE",
                        "TILED=YES",
                        "BIGTIFF=IF_SAFER"
                    ]
                    
                translate_options = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=creation_options
                )
                
                gdal.Translate(export_tif_path, vrt_ds, options=translate_options)
                
                file_size_mb = os.path.getsize(export_tif_path) / (1024 * 1024)
                print(f"Export complete. File Size: {file_size_mb:.2f} MB")

    finally:
        # --- 6. Memory Management & Cleanup ---
        vrt_ds = None 
        
        # Wipe global GDAL variables to prevent credential contamination between datasets
        for k in gdal_env.keys():
            gdal.SetConfigOption(k, None)
        
        # Automatically unlink /vsimem/ virtual memory files upon completion to prevent memory leaks
        if vrt_path.startswith("/vsimem/"):
            gdal.Unlink(vrt_path)
            
    return True