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
import rioxarray
import xarray as xr

from bmc.utils.logger import log_execution

logger = logging.getLogger(__name__)

# Highly optimized default environment for streaming Cloud Optimized GeoTIFFs (COGs) via HTTP/S3
DEFAULT_GDAL_ENV = {
    # 1. Prevents GDAL from scanning the entire remote bucket directory for matching file layouts.
    #    Treating the folder as an 'EMPTY_DIR' avoids massive directory parsing latency penalties.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",

    # 2. Restricts network probes strictly to .tif assets. Prevents GDAL from hunting for 
    #    non-existent sidecar files (like .tfw, .aux.xml, or .ovr) which wastes valuable network roundtrips.
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",

    # 3. Disables GDAL's internal global RAM caching mechanism for virtual files. Since our thread pool 
    #    immediately processes, clips, and frees arrays, turning this off prevents creeping RAM bloat.
    "VSI_CACHE": "FALSE",                    

    # 4. Sets the maximum connection retry threshold. If the host server experiences a transient spike 
    #    or issues a 429/503 rate limit throttle, the thread will attempt to re-connect up to 10 times.
    "GDAL_HTTP_MAX_RETRY": "10", 

    # 5. The backoff rest interval (in seconds) between retry attempts, giving remote servers space 
    #    to recover before the thread aggressively polls the resource endpoint again.
    "GDAL_HTTP_RETRY_DELAY": "3",

    # 6. Slashes the maximum connection timeout window to 30 seconds. Prevents an un-responsive 
    #    network socket or dead server from permanently hanging an execution thread indefinitely.
    "GDAL_HTTP_TIMEOUT": "30",

    # 7. Instructs GDAL to look ahead at byte-range requests. If the script requests blocks that 
    #    are close together, it merges them into one larger sequential stream request, bypassing 
    #    individual HTTP header handshake latencies.
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES"
}


def _fetch_single_raster(
    path: str, 
    geom: Union[Tuple[float, float, float, float], Any], 
    gdal_env: Dict[str, str],
    geom_crs: Optional[str] = None,
    max_retries = 3
) -> Tuple[str, Optional[xr.DataArray]]:
    """
    Low-level thread worker to stream and clip a single remote or local raster asset.

    Parameters
    ----------
    path : str
        The absolute path or virtual file system URI (e.g., '/vsicurl/...', '/vsis3/...') 
        pointing to the target raster asset.
    geom : tuple of float or Any (Shapely geometry, GeoJSON dict)
        The spatial geometry used to clip the raster. If a 4-element tuple is provided, 
        it is treated as a bounding box (min_x, min_y, max_x, max_y). Otherwise, it is 
        treated as a polygon mask.
    gdal_env : dict
        A dictionary of GDAL environment variables used to configure 
        the `rasterio.Env` context for optimized streaming.
    geom_crs : str, optional
        The coordinate reference system of the provided polygon geometry 
        (e.g., "EPSG:4326"). Required if passing a polygon that differs from 
        the source raster's CRS. Default is None.

    Returns
    -------
    tuple
        A 2-element tuple containing:
        - path (str): The original file path, passed back for downstream matching.
        - clipped (xarray.DataArray or None): The clipped spatiotemporal array in memory. 
          Returns None if a network or I/O failure occurs.
    """
    # Clean fallback: use the module-level constant if no specific override is passed down
    gdal_env = gdal_env or DEFAULT_GDAL_ENV
    # --- Network Jitter ---
    # Sleep for a random fraction of a second to prevent the overloading the server
    # of 32 threads hitting the server at the exact same millisecond.
    time.sleep(random.uniform(0.5, 3.0))

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.Env(**gdal_env):
                # Enforce single-threaded Dask computation within this specific worker
                with dask.config.set(scheduler='single-threaded'):
                    da = rioxarray.open_rasterio(path, chunks=True, masked=True)
                    
                    # Route based on geometry type
                    if isinstance(geom, tuple) and len(geom) == 4:
                        clipped = da.rio.clip_box(*geom).compute()
                    else:
                        geometries = geom if isinstance(geom, (list, tuple)) else [geom]
                        clipped = da.rio.clip(geometries, crs=geom_crs, drop=True).compute()
                        
                    # ==========================================
                    # INTEGRITY ARMOR (Segfault Prevention)
                    # ==========================================
                    # Structural Check: Did the timeout result in a 0-pixel array?
                    if clipped.size == 0:
                        raise ValueError(f"Array returned 0 pixels. File header is corrupted or truncated.")
                    
                    # Ghost Data Check: Did the timeout silently fill the array with NaNs?
                    if clipped.isnull().all():
                        raise ValueError(f"Array contains 100% missing data (Silent connection drop).")

                    # If it passes the gauntlet, return the healthy array and exit the loop
                    return path, clipped
                    
        except Exception as e:
            if attempt < max_retries:
                # --- 3. The 10-Second Base Wait + Jitter ---
                # 10 second strict wait to let the firewall cool down, plus 
                # up to 5 seconds of random jitter to stagger the retry requests.
                backoff_time = 10.0 + random.uniform(0.0, 5.0)  
                
                log_execution(
                    logger, 
                    f"Network anomaly on '{path}'. Retrying in {backoff_time:.1f}s (Attempt {attempt}/{max_retries}). Error: {e}", 
                    logging.WARNING
                )
                time.sleep(backoff_time)
            else:
                log_execution(
                    logger, 
                    f"FATAL: Failed to fetch asset '{path}' after {max_retries} attempts. Server is unreachable. Reason: {e}", 
                    logging.ERROR
                )
                return path, None


def parallel_fetch_rasters(
    paths: Union[str, List[str]],
    geom: Union[Tuple[float, float, float, float], Any],
    max_workers: int = 10,
    credentials: Optional[Dict[str, Any]] = None,
    gdal_config: Optional[Dict[str, str]] = None,
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
    credentials : dict, optional
        A dictionary containing S3 authentication keys. Expected keys are 
        's3_access_key' (or 'aws_access_key_id'), 's3_secret_key' 
        (or 'aws_secret_access_key'), and 's3_endpoint'. If None or missing 
        keys, the environment configures for anonymous public bucket access.
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
    gdal_env = (gdal_config or DEFAULT_GDAL_ENV).copy()

    # Apply dynamic S3 Credential Resolution
    if credentials:
        access_key = credentials.get("s3_access_key") or credentials.get("aws_access_key_id")
        secret_key = credentials.get("s3_secret_key") or credentials.get("aws_secret_access_key")
        endpoint = credentials.get("s3_endpoint") or credentials.get("endpoint")
        
        if endpoint:
            gdal_env["AWS_S3_ENDPOINT"] = endpoint

        if access_key and secret_key:
            gdal_env["AWS_ACCESS_KEY_ID"] = access_key
            gdal_env["AWS_SECRET_ACCESS_KEY"] = secret_key
            
            # Allow the credentials dictionary to optionally override kwargs
            is_virtual = credentials.get("virtual_hosting", False)
            is_https = credentials.get("use_https", True)
            
            gdal_env["AWS_VIRTUAL_HOSTING"] = "TRUE" if is_virtual else "FALSE"
            gdal_env["AWS_HTTPS"] = "YES" if is_https else "NO"
            
            # Ensure no sign request isn't accidentally active for private buckets
            gdal_env.pop("AWS_NO_SIGN_REQUEST", None)
        else:
            gdal_env["AWS_NO_SIGN_REQUEST"] = "YES"
    elif not any(k.startswith("AWS_") for k in gdal_env):
        # Fallback to anonymous requests if no credentials or AWS overrides exist
        gdal_env["AWS_NO_SIGN_REQUEST"] = "YES"

    if isinstance(paths, str):
        paths = [paths]

    geom_type = "bounding box" if isinstance(geom, tuple) and len(geom) == 4 else "polygon mask"
    log_execution(
        logger, 
        f"Initiating parallel raster fetch across {len(paths)} assets using {max_workers} workers (Geometry: {geom_type})...", 
        logging.INFO)

    fetched_arrays: Dict[str, xr.DataArray] = {}
    failed_fetches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            # Pass the dynamically resolved gdal_env to each worker
            executor.submit(_fetch_single_raster, path, geom, gdal_env, geom_crs): path
            for path in paths}     
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

def s3_to_vrt(
    s3_hrefs, 
    vrt_path, 
    credentials: Optional[Dict[str, Any]] = None,
    export_tif_path=None, 
    resample_alg="near",
    virtual_hosting=False,
    use_https=True,
    creation_options=None
):
    """
    Constructs a GDAL Virtual Raster (VRT) from a list of S3 raster URIs, 
    enabling lightweight, virtual mosaicking and optional local GeoTIFF export.

    This function configures GDAL's Virtual File System (`/vsis3/`) to stream 
    remote cloud raster assets dynamically[cite: 2]. By creating a VRT, GDAL maps the 
    spatial footprints of input rasters without downloading pixel data upfront[cite: 2]. 
    If an `export_tif_path` is specified, it streams and compresses the mosaic 
    into a local GeoTIFF file[cite: 2].

    Parameters
    ----------
    s3_hrefs : list of str
        A list of S3 URIs (e.g., `['s3://bucket/raster1.tif', ...]`) to be 
        mosaicked into the virtual dataset.
    vrt_path : str
        The output target path for the VRT file. Can be a physical file path 
        (e.g., `./mosaic.vrt`) or an in-memory virtual file (e.g., `/vsimem/mosaic.vrt`).
    credentials : dict, optional
        A dictionary containing S3 authentication keys. Expected keys are 
        's3_access_key' (or 'aws_access_key_id'), 's3_secret_key' 
        (or 'aws_secret_access_key'), and 's3_endpoint'. If None or missing 
        keys, GDAL configures for anonymous public bucket access.
    export_tif_path : str, optional
        An optional local file path to write a compiled GeoTIFF raster. If 
        None, only the VRT is constructed. Defaults to None.
    resample_alg : str, optional
        The resampling algorithm used when pixel interpolation or reprojection 
        occurs (e.g., `'near'`, `'bilinear'`, `'cubic'`). Defaults to `'near'`, 
        which preserves discrete categorical class values[cite: 2].
    virtual_hosting : bool, optional
        Toggles whether GDAL uses bucket-virtual hosting (True) or path-style 
        addressing (False). Defaults to False.
    use_https : bool, optional
        Toggles whether GDAL uses HTTPS (True) or HTTP (False) protocol for 
        fetching S3 objects. Defaults to True.
    creation_options : list of str, optional
        Custom driver creation options passed to `gdal.TranslateOptions` if 
        exporting a GeoTIFF. Defaults to `["COMPRESS=DEFLATE", "TILED=YES", 
        "BIGTIFF=IF_SAFER"]`[cite: 2].

    Returns
    -------
    bool
        Returns True if the VRT (and optional GeoTIFF) was generated successfully; 
        returns False if the input list is empty or GDAL fails to build the VRT[cite: 2].
    """
    if not s3_hrefs:
        print("No S3 URIs provided. Skipping VRT construction.")
        return False

    # --- 1. Global GDAL Configuration ---
    # Disable directory listing on open to avoid latency/timeouts on vast S3 object stores[cite: 2]
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    
    # Reset any previous authentication state in the global GDAL configuration
    gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'NO')
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', '')
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', '')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', '')

    if credentials:
        access_key = credentials.get("s3_access_key") or credentials.get("aws_access_key_id")
        secret_key = credentials.get("s3_secret_key") or credentials.get("aws_secret_access_key")
        endpoint = credentials.get("s3_endpoint") or credentials.get("endpoint")
        
        if endpoint:
            gdal.SetConfigOption('AWS_S3_ENDPOINT', endpoint)
            
        if access_key and secret_key:
            gdal.SetConfigOption('AWS_ACCESS_KEY_ID', access_key)
            gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', secret_key)
            
            # Allow the credentials dictionary to optionally override kwargs
            is_virtual = credentials.get("virtual_hosting", virtual_hosting)
            is_https = credentials.get("use_https", use_https)
            
            gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'TRUE' if is_virtual else 'FALSE')
            gdal.SetConfigOption('AWS_HTTPS', 'YES' if is_https else 'NO')
        else:
            gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'YES')
    else:
        # Public/Anonymous access
        gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'YES')

    # --- 2. Standardize Prefixes ---
    # Convert 's3://' paths into GDAL '/vsis3/' virtual filesystem paths[cite: 2]
    vsi_paths = [p.replace("s3://", "/vsis3/") for p in s3_hrefs]

    # --- 3. Construct Virtual Raster (VRT) ---
    print(f"Constructing VRT from {len(vsi_paths)} remote raster source(s)...")
    
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg=resample_alg,
        addAlpha=False
    )
    
    vrt_ds = gdal.BuildVRT(vrt_path, vsi_paths, options=vrt_options)
    
    if vrt_ds is None:
        print("Failed to construct GDAL VRT dataset.")
        return False

    # --- 4. Optional GeoTIFF Export ---
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

    # --- 5. Memory Management & Cleanup ---
    vrt_ds = None 
    
    # Automatically unlink /vsimem/ virtual memory files upon completion to prevent memory leaks[cite: 2]
    if vrt_path.startswith("/vsimem/"):
        gdal.Unlink(vrt_path)
        
    return True