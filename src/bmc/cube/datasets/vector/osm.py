from bmc.cube.spatiotemporal.vector import *

# import modules can be cleaned up

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd

try:  # osmnx >= 2.0 exposes the public module; older versions only the private one
    from osmnx.errors import InsufficientResponseError
except ImportError:
    from osmnx._errors import InsufficientResponseError


class osmCube(vectorCube):
    """
    Child class dedicated to downloading and processing OpenStreetMap (OSM)
    vector features for spatiotemporal vector data cubes.

    The class reads the ``sources.osm`` block of the pipeline recipe, downloads
    the configured feature layers via the Overpass API (through OSMnx), filters
    them by geometry type and attribute list, and returns them as a single
    GeoDataFrame in EPSG:4326 with an ``osm_layer`` column identifying the
    originating query.

    Methods
    -------
    _extract_bbox(recipe)
        Reads the WGS84 bounding box from the recipe and converts it to the
        (west, south, east, north) order expected by OSMnx.
    _download_layers(bbox, osm_cfg, logger=None)
        Executes all configured OSM queries and merges the results.
    _cache_path(recipe, bbox, osm_cfg)
        Derives a deterministic GeoPackage path for the query configuration.
    fetch_data(recipe, logger=None, downloaded_filepath=None, **kwargs)
        Loads data from file/cache or downloads it, and returns a GeoDataFrame.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_bbox(self, recipe: dict) -> Tuple[float, float, float, float]:
        """
        Reads the WGS84 bounding box from ``recipe['spatial']['bbox']``.

        Returns
        -------
        tuple
            (west, south, east, north), as expected by ``ox.features_from_bbox``.

        Raises
        ------
        ValueError
            If any of the four bounds is missing.
        """
        bbox_cfg = recipe.get("spatial", {}).get("bbox", {})
        keys = ("long_min", "lat_min", "long_max", "lat_max")
        values = tuple(bbox_cfg.get(k) for k in keys)
        if any(v is None for v in values):
            raise ValueError(
                f"Recipe 'spatial.bbox' must define all of {keys}; got {bbox_cfg}."
            )
        return values

    # ------------------------------------------------------------------
    # Grid resolution
    # ------------------------------------------------------------------

    def resolve_target_grid(
        self,
        spatial_cfg: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> str:
        """
        Resolves the master grid key (e.g. 'EEA_10km') from the recipe's
        ``spatial`` block, falling back to 'EEA_1km' if the combination is
        not registered.
        """
        grid_base = spatial_cfg.get("target_grid", "EEA")
        resolution = spatial_cfg.get("target_resolution", "1km")
        constructed_key = f"{grid_base}_{resolution}"
        if constructed_key in self.GRID_REGISTRY:
            return constructed_key

        log_execution(
            logger,
            f"Grid '{constructed_key}' not in GRID_REGISTRY; falling back to 'EEA_1km'.",
            logging.WARNING,
        )
        return "EEA_1km"
    
    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_layers(
        self,
        bbox: Tuple[float, float, float, float],
        osm_cfg: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> gpd.GeoDataFrame:
        """
        Executes all configured OSM queries and merges the results.

        Parameters
        ----------
        bbox : tuple
            (west, south, east, north) in EPSG:4326.
        osm_cfg : dict
            The ``sources.osm`` block of the recipe.
        logger : logging.Logger, optional
            Logger instance for recording execution telemetry.

        Returns
        -------
        geopandas.GeoDataFrame
            All layers concatenated, with an ``osm_layer`` column. Empty if no
            layer returned features.

        Raises
        ------
        ValueError
            If no queries are configured.
        RuntimeError
            If every query failed (as opposed to returning no features).
        """
        queries = osm_cfg.get("queries", {})
        if not queries:
            raise ValueError("Recipe 'sources.osm.queries' is missing or empty.")

        geometry_types = osm_cfg.get("geometry_types")
        attributes = osm_cfg.get("attributes")

        frames: List[gpd.GeoDataFrame] = []
        empty_layers: List[str] = []
        failed_layers: List[str] = []

        for layer_name, tags in queries.items():
            log_execution(logger, f"Fetching OSM layer '{layer_name}'...", logging.INFO)

            try:
                gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
            except InsufficientResponseError:
                log_execution(logger, f"No OSM features found for query: {tags}", logging.WARNING)
                empty_layers.append(layer_name)
                continue
            except Exception as ex:
                log_execution(
                    logger,
                    f"OSM query '{layer_name}' failed: {type(ex).__name__}: {ex}",
                    logging.ERROR,
                )
                failed_layers.append(layer_name)
                continue

            # Keep only configured geometry types
            if geometry_types:
                gdf = gdf[gdf.geometry.geom_type.isin(geometry_types)].copy()

            if gdf.empty:
                log_execution(
                    logger,
                    f"Layer '{layer_name}': features found, but none match geometry types {geometry_types}.",
                    logging.WARNING,
                )
                empty_layers.append(layer_name)
                continue

            # Keep only requested attributes that actually exist
            if attributes:
                keep_cols = [c for c in attributes if c in gdf.columns]
                if "geometry" not in keep_cols:
                    keep_cols.append("geometry")
                gdf = gdf[keep_cols]

            # Flatten OSMnx MultiIndex (element, id) into regular columns
            gdf = gdf.reset_index()
            gdf["osm_layer"] = layer_name
            frames.append(gdf)

        log_execution(
            logger,
            f"OSM download finished: {len(frames)} successful, "
            f"{len(empty_layers)} empty {empty_layers}, {len(failed_layers)} failed {failed_layers}.",
            logging.INFO,
        )

        if not frames:
            if failed_layers and not empty_layers:
                raise RuntimeError(f"All OSM queries failed: {failed_layers}")
            log_execution(logger, "No OSM features retrieved; returning empty GeoDataFrame.", logging.WARNING)
            return gpd.GeoDataFrame({"osm_layer": []}, geometry=[], crs="EPSG:4326")

        merged = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        return merged

    def _cache_path(self, recipe: dict, bbox: Tuple[float, ...], osm_cfg: Dict[str, Any]) -> Path:
        """Derives a deterministic GeoPackage path from the query configuration."""
        signature = json.dumps(
            {
                "bbox": bbox,
                "queries": osm_cfg.get("queries", {}),
                "geometry_types": osm_cfg.get("geometry_types"),
                "attributes": osm_cfg.get("attributes"),
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        target_dir = Path(recipe.get("base_dir", "./downloads")) / "osm"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"osm_{digest}.gpkg"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        recipe: dict,
        logger: Optional[logging.Logger] = None,
        downloaded_filepath: str = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Loads OSM data from file/cache or downloads it, and returns a GeoDataFrame.

        Resolution order: (1) ``downloaded_filepath`` (pre-downloaded by the
        orchestrator), (2) ``sources.osm.local_file_path`` (user-supplied file),
        (3) cached GeoPackage for identical query settings, (4) fresh download
        via Overpass, which is then cached.

        Parameters
        ----------
        recipe : dict
            The full pipeline YAML execution recipe.
        logger : logging.Logger, optional
            Logger instance for recording execution telemetry.
        downloaded_filepath : str, optional
            Explicit path to a previously downloaded or local dataset.
        **kwargs : dict
            Additional arguments absorbed safely from the orchestrator.

        Returns
        -------
        geopandas.GeoDataFrame
            OSM features in EPSG:4326 with an ``osm_layer`` column.
        """
        osm_cfg = recipe.get("sources", {}).get("osm", {})

        # 1./2. Pre-downloaded or user-supplied file
        source_path = downloaded_filepath or osm_cfg.get("local_file_path")
        if source_path and os.path.exists(source_path):
            log_execution(logger, f"Loading OSM data from {source_path}...", logging.INFO)
            gdf = gpd.read_file(source_path)
            return gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")

        # 3./4. Cache or synchronous download
        bbox = self._extract_bbox(recipe)
        cache_file = self._cache_path(recipe, bbox, osm_cfg)

        if cache_file.exists():
            log_execution(logger, f"Loading cached OSM data from {cache_file}...", logging.INFO)
            return gpd.read_file(cache_file)

        log_execution(logger, "No pre-downloaded OSM file provided. Initiating download...", logging.WARNING)
        gdf = self._download_layers(bbox, osm_cfg, logger)

        if not gdf.empty:
            gdf.to_file(cache_file, driver="GPKG")
            log_execution(logger, f"Saved {len(gdf):,} OSM features to {cache_file}.", logging.INFO)

        return gdf