import numpy as np
import rasterio
from rasterio.session import Session
from pyproj import CRS, Transformer
from typing import Tuple, Optional
import logging
from bmc.utils.logger import log_execution


def detect_remote_crs_and_res(file_path: str, cloud_session: Optional[Session] = None) -> Tuple[str, float]:
    """
    Extracts the Coordinate Reference System (CRS) and spatial resolution from 
    a local or remote GeoTIFF without loading pixel arrays into memory.

    This function performs a lightweight HTTP GET Range request (or local header read) 
    using Rasterio. It reads only the structural metadata of the file, making it 
    highly efficient for querying cloud-hosted datasets (e.g., S3 Cloud Optimized 
    GeoTIFFs) prior to establishing a spatial processing pipeline.

    Parameters
    ----------
    file_path : str
        The absolute local file path or remote virtual file system URL 
        (e.g., 's3://bucket/file.tif', '/vsicurl/http://...', or local 'data.tif') 
        pointing to the target raster.
    cloud_session : rasterio.session.Session, optional
        An authenticated cloud session (e.g., AWSSession) to use for fetching.
        If None, relies on the default unauthenticated environment.

    Returns
    -------
    crs_str : str
        The Coordinate Reference System represented as a standard string 
        (e.g., "EPSG:4326"). Defaults to "EPSG:4326" if the file lacks CRS metadata.
    res_x : float
        The native spatial resolution (pixel size) of the raster along the X-axis, 
        in the units of the detected CRS (e.g., degrees or meters).

    Examples
    --------
    >>> crs, res = detect_remote_crs_and_res("s3://chelsa/tas_01_2000.tif")
    >>> print(crs, res)
    'EPSG:4326', 0.0083333333
    """
    env_kwargs = {}
    
    if cloud_session:
        # Only disable virtual hosting. Do NOT set AWS_HTTPS="YES".
        # The session automatically injects the full 'https://...' endpoint_url.
        env_kwargs["AWS_VIRTUAL_HOSTING"] = "FALSE"

    env_context = rasterio.Env(session=cloud_session, **env_kwargs) if cloud_session else rasterio.Env()
    
    with env_context:
        with rasterio.open(file_path) as src:
            crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
            res_x = src.res[0]  
            
    return crs_str, res_x


def build_safe_fetch_envelope(
    target_crs: str,
    target_bounds: Tuple[float, float, float, float],
    source_crs: str,
    source_resolution: float,
    pixel_buffer: int = 5,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float, float, float]:
    """
    Constructs a densified, buffered bounding box in the source coordinate space 
    to guarantee full encapsulation of a target grid region during reprojection.

    When converting between coordinate systems (e.g., planar meters to spherical degrees), 
    straight bounding box edges curve and distort. This function densifies the perimeter 
    of the target box using linear interpolation, projects the dense point-cloud into the 
    source CRS, and extracts the absolute outer envelope. It then expands this envelope 
    by a defined pixel buffer to prevent boundary starvation when applying multi-pixel 
    GDAL resampling kernels (like cubic or average) downstream.

    Parameters
    ----------
    target_crs : str
        The Coordinate Reference System string of the destination grid 
        (e.g., "EPSG:3035").
    target_bounds : tuple of float
        The bounding box coordinates in the target CRS formatted as 
        (minx, miny, maxx, maxy).
    source_crs : str
        The Coordinate Reference System string of the raw data being fetched 
        (e.g., "EPSG:4326").
    source_resolution : float
        The size of a single source pixel in native source CRS units (e.g., degrees). 
        Used to calculate the exact buffer distance.
    pixel_buffer : int, optional
        The number of native source pixels to add as an outer safety margin. A buffer 
        of 5 is recommended to accommodate typical GDAL interpolation windows. 
        Default is 5.
    logger : logging.Logger, optional
        A standard Python logger instance to record execution progress. Default is None.

    Returns
    -------
    safe_bounds : tuple of float
        The buffered, mathematically safe bounding box in the source CRS, 
        formatted as (src_minx, src_miny, src_maxx, src_maxy).

    Raises
    ------
    ValueError
        If the spatial transformation yields entirely non-finite coordinates, indicating 
        the target bounds fall entirely outside the mathematically valid domain of 
        the source projection.

    Notes
    -----
    If the `source_crs` is determined to be a geographic coordinate system (measured in 
    degrees), the function enforces strict global guardrails, clamping the output bounds 
    between -180.0/180.0 (Longitude) and -90.0/90.0 (Latitude) to prevent downstream 
    API request failures.

    Examples
    --------
    Calculate a safe WGS84 fetching envelope for a metric target grid in Europe:
    
    >>> target_box = (3800000.0, 2900000.0, 3900000.0, 3000000.0)
    >>> safe_env = build_safe_fetch_envelope(
    ...     target_crs="EPSG:3035",
    ...     target_bounds=target_box,
    ...     source_crs="EPSG:4326",
    ...     source_resolution=0.008333,
    ...     pixel_buffer=5
    ... )
    >>> print(safe_env)
    (4.19833, 50.71833, 5.66833, 51.65500)
    """
    log_execution(logger, f"Computing safe fetch envelope ({target_crs} -> {source_crs})...", logging.INFO)

    minx, miny, maxx, maxy = target_bounds
    num_points = 100 

    # Vectorized Perimeter Densification
    bx = np.linspace(minx, maxx, num_points)
    by = np.full(num_points, miny)
    rx = np.full(num_points, maxx)
    ry = np.linspace(miny, maxy, num_points)
    tx = np.linspace(maxx, minx, num_points)
    ty = np.full(num_points, maxy)
    lx = np.full(num_points, minx)
    ly = np.linspace(maxy, miny, num_points)

    perimeter_x = np.concatenate([bx, rx, tx, lx])
    perimeter_y = np.concatenate([by, ry, ty, ly])

    # Coordinate Transformation
    transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    src_x, src_y = transformer.transform(perimeter_x, perimeter_y)

    valid_mask = np.isfinite(src_x) & np.isfinite(src_y)
    if not np.any(valid_mask):
        raise ValueError("Failed to project bounds. Ensure coordinates match the CRS domain.")
        
    src_x, src_y = src_x[valid_mask], src_y[valid_mask]

    src_minx, src_maxx = float(np.min(src_x)), float(np.max(src_x))
    src_miny, src_maxy = float(np.min(src_y)), float(np.max(src_y))

    # Apply Resampling Safety Buffer
    buffer_padding = source_resolution * pixel_buffer
    
    safe_minx = src_minx - buffer_padding
    safe_maxx = src_maxx + buffer_padding
    safe_miny = src_miny - buffer_padding
    safe_maxy = src_maxy + buffer_padding

    # Geographic Domain Guardrails
    src_crs_obj = CRS.from_string(source_crs)
    if src_crs_obj.is_geographic:
        safe_minx = max(-180.0, safe_minx)
        safe_maxx = min(180.0, safe_maxx)
        safe_miny = max(-90.0, safe_miny)
        safe_maxy = min(90.0, safe_maxy)

    return (safe_minx, safe_miny, safe_maxx, safe_maxy)

def build_envelope_from_file(
    target_crs: str,
    target_bounds: Tuple[float, float, float, float],
    source_file_path: str,
    pixel_buffer: int = 5,
    logger: Optional[logging.Logger] = None,
    cloud_session: Optional[Session] = None
) -> Tuple[float, float, float, float]:
    """
    Convenience wrapper that automatically pings a remote or local file to detect 
    its spatial parameters before calculating a safe fetching envelope.

    This function combines `detect_remote_crs_and_res` and `build_safe_fetch_envelope` 
    into a single step.

    Parameters
    ----------
    target_crs : str
        The Coordinate Reference System string of the destination grid (e.g., "EPSG:3035").
    target_bounds : tuple of float
        The bounding box coordinates in the target CRS (minx, miny, maxx, maxy).
    source_file_path : str
        The absolute local file path or remote URL to the target raster.
    pixel_buffer : int, optional
        The number of native source pixels to add as an outer safety margin. Default is 5.
    logger : logging.Logger, optional
        Logger instance. Default is None.
    cloud_session : rasterio.session.Session, optional
        An authenticated cloud session (e.g., AWSSession) to use for fetching.

    Returns
    -------
    safe_bounds : tuple of float
        The buffered, mathematically safe bounding box in the detected source CRS.
    """
    log_execution(logger, f"Pinging '{source_file_path}' to detect spatial metadata...", logging.INFO)
    
    # 1. Execute the network/disk I/O step using the secure session
    detected_crs, detected_res = detect_remote_crs_and_res(source_file_path, cloud_session=cloud_session)
    
    log_execution(logger, f"Detected Source CRS: {detected_crs} | Res: {detected_res}", logging.INFO)

    # 2. Execute the pure mathematical step
    return build_safe_fetch_envelope(
        target_crs=target_crs,
        target_bounds=target_bounds,
        source_crs=detected_crs,
        source_resolution=detected_res,
        pixel_buffer=pixel_buffer,
        logger=logger
    )