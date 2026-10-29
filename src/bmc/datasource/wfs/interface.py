import logging
from numbers import Real
from typing import Optional, Sequence
from urllib.parse import urlparse

import geopandas as gpd
import requests


def _validate_url(url: Optional[str]) -> str:
    """Return the stripped URL if it is a regular http(s) URL, else raise ValueError."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("'wfs_url' is required and must be a non-empty string.")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid 'wfs_url' '{url}': expected a http(s) URL such as 'https://host/path'.")
    return url


def _validate_bbox(bbox: Sequence[float]) -> tuple:
    """Validate a (long_min, lat_min, long_max, lat_max) bounding box in EPSG:4326."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(
        isinstance(v, Real) and not isinstance(v, bool) for v in bbox
    ):
        raise ValueError("'bbox' must be a sequence of four numbers: (long_min, lat_min, long_max, lat_max).")
    long_min, lat_min, long_max, lat_max = bbox
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