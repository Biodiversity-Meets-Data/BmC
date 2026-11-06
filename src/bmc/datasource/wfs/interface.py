import logging
from numbers import Real
from typing import Optional, Sequence
from urllib.parse import urlparse

import geopandas as gpd
import requests


def _validate_url(url: Optional[str]) -> str:
    """
    Validate and format a WFS endpoint URL.

    Parameters
    ----------
    url : str, optional
        The URL string to validate. Must be a standard HTTP or HTTPS URL.

    Returns
    -------
    str
        The stripped and validated URL string.

    Raises
    ------
    ValueError
        If the URL is not a string, is empty, or lacks a valid HTTP/HTTPS scheme and network location.
    """
    # Ensure the URL is actually a string and not empty after removing whitespace
    if not isinstance(url, str) or not url.strip():
        raise ValueError("'wfs_url' is required and must be a non-empty string.")
    
    # Clean the string of any leading/trailing whitespace
    url = url.strip()
    
    # Parse the URL into components to check its protocol and domain
    parsed = urlparse(url)
    
    # Ensure the protocol is either http or https, and a domain/host (netloc) is present
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid 'wfs_url' '{url}': expected a http(s) URL such as 'https://host/path'.")
    
    return url

def _validate_bbox(bbox: Sequence[float]) -> tuple:
    """
    Validate a spatial bounding box against EPSG:4326 coordinate boundaries.

    Parameters
    ----------
    bbox : sequence of float
        A sequence representing the bounding box as (long_min, lat_min, long_max, lat_max).

    Returns
    -------
    tuple
        A tuple containing the validated (long_min, lat_min, long_max, lat_max).

    Raises
    ------
    ValueError
        If the bounding box does not contain exactly 4 numeric values, or if the 
        coordinates fall outside the valid ranges for EPSG:4326 (WGS 84).
    """
    # Check that bbox is a list or tuple, has exactly 4 items, and all items are numeric (excluding booleans)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(
        isinstance(v, Real) and not isinstance(v, bool) for v in bbox
    ):
        raise ValueError("'bbox' must be a sequence of four numbers: (long_min, lat_min, long_max, lat_max).")
    
    # Unpack the coordinates for logical boundary checking
    long_min, lat_min, long_max, lat_max = bbox
    
    # Ensure coordinates are within global WGS 84 limits and mins are strictly less than maxes
    if not (-180 <= long_min < long_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise ValueError(f"'bbox' {tuple(bbox)} is not a valid EPSG:4326 extent.")
    
    return tuple(bbox)


def fetch_wfs(
    wfs_url: Optional[str] = None,
    type_name: Optional[str] = None,
    output_format: str = "application/json",
    max_features: Optional[int] = None,
    bbox: Optional[Sequence[float]] = None,
    timeout: float = 60.0,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Fetches features from a WFS endpoint (e.g. GeoServer) as a GeoDataFrame.

    Parameters
    ----------
    wfs_url : str
        Base URL of the WFS endpoint. Required; must be a valid http(s) URL.
    type_name : str
        Feature type name, e.g. 'workspace:layer'. Required.
    output_format : str, default 'application/json'
        WFS output format. The response must be GeoJSON.
    max_features : int, optional
        Maximum number of features to request. Must be a positive integer.
    bbox : sequence of float, optional
        (long_min, lat_min, long_max, lat_max) in EPSG:4326.
    timeout : float, default 60.0
        Request timeout in seconds.
    logger : logging.Logger, optional
        Logger for execution telemetry.

    Returns
    -------
    geopandas.GeoDataFrame
        Features in EPSG:4326 (empty if the endpoint returned no features).

    Raises
    ------
    ValueError
        If a required parameter is missing or any parameter is invalid, or if
        the response is not valid GeoJSON.
    requests.HTTPError
        If the endpoint returns an HTTP error status.
    """
    # 1. Parameter validation (before any network access)
    wfs_url = _validate_url(wfs_url)

    if not isinstance(type_name, str) or not type_name.strip():
        raise ValueError("'type_name' is required and must be a non-empty string.")
    type_name = type_name.strip()

    if not isinstance(output_format, str) or not output_format.strip():
        raise ValueError("'output_format' must be a non-empty string.")

    if max_features is not None and (
        not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0
    ):
        raise ValueError("'max_features' must be a positive integer or None.")

    if bbox is not None:
        bbox = _validate_bbox(bbox)

    # 2. Request
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": output_format,
    }
    if max_features is not None:
        params["maxFeatures"] = max_features
    if bbox is not None:
        params["bbox"] = ",".join(str(v) for v in bbox) + ",EPSG:4326"

    response = requests.get(wfs_url, params=params, timeout=timeout)
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as ex:
        raise ValueError(
            f"Response from '{wfs_url}' is not valid JSON (check 'type_name' and 'output_format'): {ex}"
        ) from ex

    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"Response from '{wfs_url}' is not a GeoJSON FeatureCollection.")

    # 3. Build GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

    if logger:
        if gdf.empty:
            logger.warning(f"WFS request for '{type_name}' at {wfs_url} returned no features.")
        else:
            logger.info(f"Fetched {len(gdf)} features of '{type_name}' from {wfs_url}.")
    return gdf

def fetch_wfs(
    wfs_url: Optional[str] = None,
    type_name: Optional[str] = None,
    output_format: str = "application/json",
    max_features: Optional[int] = None,
    bbox: Optional[Sequence[float]] = None,
    timeout: float = 60.0,
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Fetches feature data from a Web Feature Service (WFS) endpoint into a GeoDataFrame.

    Parameters
    ----------
    wfs_url : str
        Base URL of the WFS endpoint. Required; must be a valid HTTP or HTTPS URL.
    type_name : str
        The specific feature type name to request from the server, typically formatted 
        as 'workspace:layer'. Required.
    output_format : str, default 'application/json'
        The format requested from the WFS server. The server's response must ultimately 
        be valid GeoJSON for GeoPandas to parse it.
    max_features : int, optional
        The maximum number of features the server should return. Must be a strictly 
        positive integer.
    bbox : sequence of float, optional
        A geographic bounding box (long_min, lat_min, long_max, lat_max) in the 
        EPSG:4326 coordinate reference system to limit the spatial query.
    timeout : float, default 60.0
        The maximum time in seconds to wait for the server response before aborting.
    logger : logging.Logger, optional
        An optional logger instance for recording telemetry and request outcomes.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the requested features, explicitly set to the 
        EPSG:4326 coordinate reference system. Returns an empty GeoDataFrame if 
        the server responds successfully but provides no features.

    Raises
    ------
    ValueError
        If any provided parameters fail pre-flight validation, if the server response 
        cannot be parsed as JSON, or if the JSON is not a valid GeoJSON FeatureCollection.
    requests.HTTPError
        If the HTTP request to the WFS server returns an error status code (e.g., 404, 500).
    """
    # ---------------------------------------------------------
    # Parameter validation (Executes before any network I/O)
    # ---------------------------------------------------------
    
    # Validate the base WFS URL
    wfs_url = _validate_url(wfs_url)

    # Validate that the layer name is provided and is a valid string
    if not isinstance(type_name, str) or not type_name.strip():
        raise ValueError("'type_name' is required and must be a non-empty string.")
    type_name = type_name.strip()

    # Ensure the requested output format is valid
    if not isinstance(output_format, str) or not output_format.strip():
        raise ValueError("'output_format' must be a non-empty string.")

    # If max_features is provided, ensure it is a valid, positive integer limit
    if max_features is not None and (
        not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0
    ):
        raise ValueError("'max_features' must be a positive integer or None.")

    # If a spatial filter is requested, strictly validate the coordinate bounds
    if bbox is not None:
        bbox = _validate_bbox(bbox)

    # ---------------------------------------------------------
    # WFS Network Request Construction and Execution
    # ---------------------------------------------------------
    
    # Construct the base WFS GetFeature query parameters based on OGC standards
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": output_format,
    }
    
    # Append optional limits if provided by the user
    if max_features is not None:
        params["maxFeatures"] = max_features
        
    # Append bounding box filter if provided. Format must match OGC spec: minX,minY,maxX,maxY,CRS
    if bbox is not None:
        params["bbox"] = ",".join(str(v) for v in bbox) + ",EPSG:4326"

    # Execute the HTTP GET request with a timeout to prevent hanging
    response = requests.get(wfs_url, params=params, timeout=timeout)
    
    # Raise an exception immediately if the server returned an HTTP error code
    response.raise_for_status()

    # ---------------------------------------------------------
    # Data Parsing and GeoDataFrame Construction
    # ---------------------------------------------------------
    
    # Attempt to decode the response payload into a Python dictionary
    try:
        data = response.json()
    except ValueError as ex:
        # If JSON parsing fails, the server likely returned an XML error or HTML page
        raise ValueError(
            f"Response from '{wfs_url}' is not valid JSON (check 'type_name' and 'output_format'): {ex}"
        ) from ex

    # Verify the structure of the JSON matches the expected GeoJSON standard
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"Response from '{wfs_url}' is not a GeoJSON FeatureCollection.")

    # Convert the GeoJSON features list into a tabular GeoDataFrame, strictly assigning WGS 84
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

    # ---------------------------------------------------------
    # Telemetry and Return
    # ---------------------------------------------------------
    
    # Log the outcome if a logger was provided
    if logger:
        if gdf.empty:
            logger.warning(f"WFS request for '{type_name}' at {wfs_url} returned no features.")
        else:
            logger.info(f"Fetched {len(gdf)} features of '{type_name}' from {wfs_url}.")
            
    return gdf