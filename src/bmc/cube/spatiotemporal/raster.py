import os
import json
import gc
import time
import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from rioxarray.enum import Convention
import rasterio
from rasterio.warp import transform_bounds

from datetime import datetime, timezone
import shapely
import shapely.geometry
from shapely.geometry import box
try:
    import pystac
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False

from bmc.utils.logger import log_execution, ResourceProfiler
from bmc.utils.spatial import build_envelope_from_file
from bmc.utils.io import parallel_fetch_rasters

from bmc.engine.raster import rasterEngine
from bmc.cube.spatiotemporal.cube import dataCube

class rasterCube(rasterEngine, dataCube, ABC):
    """
    Base class for constructing multidimensional ecological data lakes/cubes.

    This class provides the fundamental spatial physics, directory generation, 
    logging, and core GDAL/xarray processing engines required to build 
    spatiotemporal data cubes. It handles the ingestion, out-of-core warping, 
    and N-dimensional alignment of raw raster data (e.g., WEkEO, CHELSA) 
    to rigidly defined master grids.

    Attributes
    ----------
    GRID_REGISTRY : dict
        A master registry of supported coordinate reference systems (CRS) and 
        their exact spatial boundaries. Used to ensure flawless mathematical 
        alignment across disparate datasets.
    GDAL_RESAMPLERS : dict
        Internal mapping of human-readable resampler string keys to their 
        corresponding GDAL C++ integer constants.

    Methods
    -------
    resolve_target_grid(spatial_cfg, logger)
        Translates user-defined configurations into a validated grid registry key.
    generate_execution_plan(recipe, logger)
        Translates the YAML recipe into a standardized data fetching DataFrame queue.
    parse_metadata(row, da)
        Extracts dimensional metadata from the queue and injects it into Z-axes.
    get_resample_rule(variable_name)
        Determines the appropriate GDAL spatial resampling algorithm for a variable.
    generate_raster_stac_item(level_reprojected_paths, recipe, dataset_name, output_dir, item_id=None, logger=None)
        Converts the processed variables into a fully compliant STAC Item with embedded metadata.
    process_cube(recipe, max_workers=10, logger=None)
        The universal out-of-core spatial processing loop that triggers fetching and warping.
    """  
    def __init__(self):
        super().__init__()

    #################################
    # Interface & helper functions  #
    #################################

    def _parse_res_to_meters(self, res_str: str) -> float:
        """
        Converts a resolution string (e.g., '10m', '1km') into a float in meters.
        
        Parameters
        ----------
        res_str : str
            The resolution string extracted from the recipe or data source.
            
        Returns
        -------
        float
            The numerical representation of the resolution in metric units.
        """
        res_str = res_str.lower().strip()
        if 'km' in res_str:
            return float(res_str.replace('km', '')) * 1000
        elif 'm' in res_str:
            return float(res_str.replace('m', ''))
        else:
            # Fallback for unexpected formats (like arc-seconds) mapping to degrees
            return 999999.0

    def _resolve_query_resolution(
        self, 
        strategy: str, 
        available_res: List[str], 
        logger: Optional[logging.Logger] = None
    ) -> str:
        """
        Determines the single best resolution string to use based on a strategy.

        Parameters
        ----------
        strategy : str
            Options: 'highest' (smallest meters), 'lowest' (largest meters), 
            or a specific string matching an available resolution.
        available_res : list of str
            The unique resolution strings found in the remote inventory.
        logger : logging.Logger, optional
            Logger instance to record fallback decisions.
            
        Returns
        -------
        str
            The strictly selected resolution string.
        """
        if strategy not in ['highest', 'lowest'] and strategy in available_res:
            return strategy
            
        res_map = {self._parse_res_to_meters(r): r for r in available_res}
        
        if not res_map:
            return "UNKNOWN"

        if strategy == 'highest':
            # Smallest physical distance equals highest spatial resolution
            return res_map[min(res_map.keys())]
        elif strategy == 'lowest':
            # Largest physical distance equals lowest spatial resolution
            return res_map[max(res_map.keys())]
        else:
            # Safe fallback if user explicitly requests an unavailable resolution
            best_guess = res_map[min(res_map.keys())]
            log_execution(
                logger, 
                f"Requested query resolution '{strategy}' not found. Falling back to highest available: {best_guess}", 
                logging.WARNING
            )
            return best_guess

    #################################
    #     Abstract Definitions      #
    #################################

    @abstractmethod
    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: logging.Logger) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key.

        This abstract method must interpret the ``spatial_cfg`` block from the YAML 
        recipe and map it to a physically safe, mathematically supported grid framework 
        present in the base class's ``GRID_REGISTRY``. 

        Parameters
        ----------
        spatial_cfg : dict
            The 'spatial' configuration dictionary extracted from the execution recipe. 
        logger : logging.Logger
            The logger instance used to record validation steps or fallback warnings.

        Returns
        -------
        str
            The validated dictionary key (e.g., 'EEA_1km') required to query the 
            parent class's GRID_REGISTRY.
        """
        pass

    @abstractmethod
    def get_resample_rule(self, variable_name: str) -> str:
        """
        Determines the appropriate GDAL spatial resampling algorithm for a variable.

        Different physical and ecological variables require strictly different mathematical 
        algorithms to prevent data corruption (e.g., keeping categories discrete).

        Parameters
        ----------
        variable_name : str
            The physical variable currently being warped (e.g., 'pr', 'Corine Land Cover 2018').

        Returns
        -------
        str
            The requested GDAL resampling string ('nearest', 'bilinear', 'average', etc.).
        """
        pass

    @abstractmethod
    def get_auxiliary_dimensions(self, recipe: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Parses the execution recipe to dynamically identify active auxiliary dimensions.

        This abstract method forces dataset-specific child classes (e.g., ``chelsaCube``) 
        to act as translators between the user's YAML recipe and the engine's flat 
        catalog schema. By overriding this method, the child class extracts dataset-specific 
        parameters (such as climate models, emission scenarios, or baseline periods) 
        and maps them to their exact corresponding column names in the GeoParquet catalog.

        Parameters
        ----------
        recipe : dict
            The parsed YAML execution plan containing the user's configurations for 
            spatial domains, temporal bounds, and dataset-specific selections.

        Returns
        -------
        dict of str to list
            A dictionary where each key perfectly matches a flat column name in the 
            GeoParquet catalog (e.g., ``'cmip6:model'``), and each value is a list of 
            the requested categorical values (e.g., ``['GFDL-ESM4', 'MPI-ESM1-2-HR']``). 
            If no auxiliary dimensions are active, returns an empty dictionary.

        Notes
        -----
        The keys returned by this dictionary are dynamically appended to the Z-axis 
        grouping logic inside :meth:`generate_execution_plan`. Therefore, any key 
        returned here *must* exist as a column in the dataset's GeoParquet file, 
        otherwise the pipeline will log a warning and ignore the dimension.
        """
        pass

    #################################
    #      Core Processing Loops    #
    #################################

    def _apply_recipe_attributes(self, recipe: Dict[str, Any]) -> None:
        """
        Dynamically extracts dataset-specific parameters from an execution recipe 
        and maps them to instance attributes required by the base engine.

        This method parses the nested configuration dictionary (typically originating 
        from a user-defined YAML file) to determine which specific processing levels 
        and physical variables are activated for the current dataset. It also handles 
        overriding default systemic paths (like the catalog location) if the user 
        has provided a custom path in the recipe.

        Parameters
        ----------
        recipe : Dict[str, Any]
            The full, parsed configuration dictionary containing execution directives. 
            It is expected to have a 'sources' key that maps to dataset-specific 
            configurations (e.g., matching `self.dataset_name`).

        Returns
        -------
        None
            This method operates in place and modifies instance attributes directly.

        Attributes Modified
        -------------------
        catalog_path : str
            Overwritten if a custom 'catalog_path' is explicitly defined within the 
            dataset's recipe block.
        levels : list of str
            Populated with the names of all levels where the 'include' flag is True.
        variables : list of str
            Populated with a flattened, deduplicated list of all active variable 
            names found within the included levels.

        Notes
        -----
        The expected dictionary structure within the recipe is:
        sources:
        <dataset_name>:
            catalog_path: "/optional/custom/path"
            levels:
            <level_name>:
                include: True
                variables:
                <var_1>: True
                <var_2>: False
        """
        # Isolate the configuration block specific to this dataset instance.
        # Fall back to an empty dictionary to prevent KeyError if the dataset is missing.
        dataset_cfg = recipe.get('sources', {}).get(self.dataset_name, {})
        
        # 1. Catalog Path Override
        # Check if the user explicitly provided a custom path for the STAC/Parquet catalog.
        # If present, overwrite the default path instantiated during class initialization.
        if 'catalog_path' in dataset_cfg:
            self.catalog_path = dataset_cfg['catalog_path']
        
        # Reset internal tracking lists to ensure idempotency if called multiple times
        self.levels = []
        self.variables = []
        
        # 2. Extract Active Levels and Variables
        # Iterate through the defined structural levels (e.g., 'crop_types', 'forest_type')
        for level, settings in dataset_cfg.get('levels', {}).items():
            
            # Only process levels explicitly flagged for inclusion in the pipeline
            if settings.get('include', False):
                self.levels.append(level)
                
                # Extract all target variables nested under this specific level.
                # We filter the dictionary comprehension to keep only those marked as True.
                active_vars = [
                    var_name for var_name, is_active 
                    in settings.get('variables', {}).items() 
                    if is_active
                ]
                self.variables.extend(active_vars)
                
        # 3. Deduplicate Variables
        # If different levels share variable abbreviations, or if the recipe is malformed,
        # converting to a set and back to a list ensures unique elements for the processing queue.
        self.variables = list(set(self.variables))

    def generate_execution_plan(
        self, recipe: Dict[str, Any], logger: logging.Logger
    ) -> pd.DataFrame:
        """
        Constructs a tabular execution manifest outlining the required spatial assets.

        This method parses the user-provided YAML recipe and filters the underlying 
        spatial catalog (e.g., Parquet file) based on requested variables, spatial 
        bounding boxes, temporal ranges, and auxiliary dimensions. It acts as the 
        query planner for the orchestrator.

        Parameters
        ----------
        recipe : Dict[str, Any]
            The master pipeline configuration dictionary parsed from YAML.
        logger : logging.Logger
            The logger instance for tracking telemetry and pipeline warnings.

        Returns
        -------
        pd.DataFrame
            A fully filtered and grouped Pandas DataFrame containing the required 
            asset URIs (`vsi_path`), their types (`tile` vs `slice`), and temporal/spatial 
            metadata necessary for GDAL retrieval. Returns an empty DataFrame if no 
            assets match the query constraints.
        """
        # --- 0. Initialize Dynamic Attributes ---
        # Binds metadata variables and custom arguments specified in the recipe to the class instance
        self._apply_recipe_attributes(recipe)

        # --- 1. Extract Config from Recipe ---
        # Default to a global bounding box if none is explicitly provided in the spatial block
        bbox_cfg = recipe.get("spatial", {}).get("bbox", {})
        bbox = (
            bbox_cfg.get("long_min", -180),
            bbox_cfg.get("lat_min", -90),
            bbox_cfg.get("long_max", 180),
            bbox_cfg.get("lat_max", 90),
        )

        # Default temporal bounds to a broad range if absent
        temp_cfg = recipe.get("temporal", {})
        start_date = f"{temp_cfg.get('start_year', 1900)}-01-01"
        end_date = f"{temp_cfg.get('end_year', 2100)}-12-31"

        # Construct exact Shapely geometry for spatial intersection queries
        search_geom = shapely.geometry.box(*bbox)
        start = pd.to_datetime(start_date, utc=True)
        end = pd.to_datetime(end_date, utc=True)

        static_levels = getattr(self, "temporally_static_levels", [])
        aux_dims = self.get_auxiliary_dimensions(recipe)

        levels = self.levels if isinstance(self.levels, list) else [self.levels]
        variables = (
            self.variables if isinstance(self.variables, list) else [self.variables]
        )

        manifest_chunks = []

        # --- Process Level-by-Level ---
        # Catalogs are often partitioned by temporal resolution (e.g., 'annual' vs 'monthly')
        for level in levels:

            # --- 2. Load ONLY the catalog file for the current level ---
            catalog_dict = getattr(self, "catalogs", {})
            unified_catalog = getattr(self, "catalog", "")

            # Resolve the physical filepath of the Parquet catalog
            if isinstance(catalog_dict, dict) and level in catalog_dict:
                # Target found in dictionary mapping (e.g., CLMS modular datasets)
                level_catalog_path = catalog_dict[level]
            elif isinstance(unified_catalog, str) and unified_catalog.endswith('.parquet'):
                # Target is a single, unified master catalog file (e.g., CHELSA)
                level_catalog_path = unified_catalog
            else:
                # Fallback to standard directory structure if self.catalog is a directory path
                base_path = getattr(self, "catalog_path", unified_catalog)
                level_catalog_path = f"{base_path}/{level}_catalog.parquet"

            try:
                level_gdf = gpd.read_parquet(level_catalog_path)
                log_execution(
                    logger,
                    f"Loaded catalog for level '{level}' ({len(level_gdf)} rows) from {level_catalog_path}",
                    logging.INFO,
                )
            except Exception as e:
                log_execution(
                    logger,
                    f"Failed to load catalog for level '{level}' ({level_catalog_path}): {e}",
                    logging.WARNING,
                )
                continue

            # Schema Validation: Ensure the catalog adheres to the required strict SpatioTemporal schema
            required_cols = [
                "geometry",
                "datetime",
                "level",
                "variable",
                "vsi_path",
            ]
            missing = [c for c in required_cols if c not in level_gdf.columns]
            if missing:
                log_execution(
                    logger,
                    f"Catalog '{level}' missing columns: {missing}. Skipping.",
                    logging.ERROR,
                )
                continue

            # --- 3. Filter Variables ---
            # Drop all records that are not explicitly requested in the YAML recipe
            level_gdf = level_gdf[
                (level_gdf["level"] == level)
                & (level_gdf["variable"].isin(variables))
            ].copy()
            if level_gdf.empty:
                continue

            # --- 4. Auxiliary Dimension Filter & Grouping Injection ---
            # Defines how records should be grouped together into a single time-slice
            grouping_cols = ["level", "variable", "datetime"]

            if aux_dims:
                for dim_col, values in aux_dims.items():
                    if dim_col in level_gdf.columns:
                        if not isinstance(values, list):
                            values = [values]
                        # Retain records matching requested dimensions or explicitly marked as NaN (applies everywhere)
                        if not level_gdf[dim_col].isna().all():
                            level_gdf = level_gdf[
                                level_gdf[dim_col].isin(values)
                                | level_gdf[dim_col].isna()
                            ]
                            if dim_col not in grouping_cols:
                                grouping_cols.append(dim_col)

            if level_gdf.empty:
                continue

            # --- 5. Spatial Filter ---
            # Perform bounding box intersection using the GeoPandas STRTree index
            level_gdf = level_gdf[level_gdf.intersects(search_geom)].copy()
            if level_gdf.empty:
                continue

            # --- 6. Temporal Filter ---
            # Discard records outside the requested timeframe (unless the data is strictly temporally static)
            level_gdf["datetime"] = pd.to_datetime(level_gdf["datetime"], utc=True)
            if level not in static_levels:
                level_gdf = level_gdf[
                    (level_gdf["datetime"] >= start)
                    & (level_gdf["datetime"] <= end)
                ]

            if level_gdf.empty:
                continue

            # --- 7. Grouping & Manifest Construction ---
            # Condense identical temporal/spatial slices into a single row. 
            # If multiple spatial tiles exist for the same timestamp, their URLs are bundled into a list.
            agg_rules = {"vsi_path": list}

            if "proj:epsg" in level_gdf.columns:
                agg_rules["proj:epsg"] = lambda x: list(
                    set([int(v) for v in x if pd.notna(v)])
                )
            if "nodata" in level_gdf.columns:
                agg_rules["nodata"] = (
                    lambda x: list(set(x))[0] if len(set(x)) > 0 else None
                )

            level_manifest = (
                level_gdf.groupby(grouping_cols, dropna=True)
                .agg(agg_rules)
                .reset_index()
            )
            manifest_chunks.append(level_manifest)

        # --- 8. Assemble Final Manifest ---
        if not manifest_chunks:
            return pd.DataFrame()

        manifest = pd.concat(manifest_chunks, ignore_index=True)

        rename_map = {"datetime": "time", "proj:epsg": "epsg_codes"}
        manifest = manifest.rename(
            columns={k: v for k, v in rename_map.items() if k in manifest.columns}
        )

        # Dynamically determine the pipeline routing strategy.
        # 'tile': Distributed multi-file spatial blocks requiring GDAL VRT mosaicking.
        # 'slice': Single global/regional files requiring straightforward GDAL windowing.
        manifest['asset_type'] = manifest['vsi_path'].apply(
            lambda paths: 'tile' if len(paths) > 1 else 'slice'
        )
        manifest['path_type'] = 'remote'
        
        # --- Safely execute the dataset-specific provider resolver ---
        # Used downstream to apply specific S3 / Cloud authentication headers
        if hasattr(self, 'resolve_provider'):
            manifest['provider_id'] = manifest['vsi_path'].apply(self.resolve_provider)
        else:
            manifest['provider_id'] = None

        return manifest

    def generate_raster_stac_item(
        self,
        level_reprojected_paths: Dict[str, Dict[str, str]],
        recipe: dict,
        dataset_name: str,
        output_dir: str,
        item_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> 'pystac.Item':
        """
        Converts the generated raster variables and level file paths into a fully 
        compliant STAC Item, embedding spatial grid specifications, resample algorithms, 
        and dynamically extracting active multi-dimensional metadata coordinates.
        """
        if not HAS_PYSTAC:
            raise ImportError("The 'pystac' library is required to generate STAC items. Run: pip install pystac")

        log_execution(logger, f"=== Assembling STAC Item for {dataset_name.upper()} (Raster) ===", logging.INFO)

        # ---------------------------------------------------------------------
        # 1. Spatial Domain Resolution & Coordinate Reference
        # ---------------------------------------------------------------------
        spatial_cfg = recipe.get("spatial", {})
        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)
        grid_info = self.GRID_REGISTRY[target_grid_key]
        target_crs = grid_info["crs"]
        target_res = grid_info["resolution"]

        bbox_cfg = spatial_cfg.get("bbox", {})
        wgs84_bounds = [
            float(min(bbox_cfg.get("long_min", 0.0), bbox_cfg.get("long_max", 0.0))),
            float(min(bbox_cfg.get("lat_min", 0.0), bbox_cfg.get("lat_max", 0.0))),
            float(max(bbox_cfg.get("long_min", 0.0), bbox_cfg.get("long_max", 0.0))),
            float(max(bbox_cfg.get("lat_min", 0.0), bbox_cfg.get("lat_max", 0.0)))
        ]
        footprint = shapely.geometry.mapping(shapely.box(*wgs84_bounds))

        # ---------------------------------------------------------------------
        # 2. Extract Temporal Domain
        # ---------------------------------------------------------------------
        temporal_cfg = recipe.get("temporal", {})
        start_year = temporal_cfg.get("start_year")
        end_year = temporal_cfg.get("end_year")

        if start_year and end_year:
            start_dt = datetime(int(start_year), 1, 1, tzinfo=timezone.utc)
            end_dt = datetime(int(end_year), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        else:
            start_dt = end_dt = datetime.now(timezone.utc)

        # ---------------------------------------------------------------------
        # 3. Compile STAC Provenance & Generalized Dimension Extraction
        # ---------------------------------------------------------------------
        export_format = recipe.get("export_as", {}).get("format", "netcdf").lower()
        cube_name = recipe.get("cube_name", "raster_cube")

        stac_properties = {
            "cube:dataset": dataset_name,
            "cube:grid_registry_key": target_grid_key,
            "cube:crs": str(target_crs),
            "cube:resolution": target_res,
            "cube:export_format": export_format,
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
        }

        # Dynamically extract active dimensions (e.g., ensemble, scenario) generated by parse_metadata
        for level, variables in level_reprojected_paths.items():
            if not variables:
                continue
            
            # Grab the first available variable file for this processing level to inspect its schema
            sample_var = list(variables.keys())[0]
            sample_path = variables[sample_var]
            
            if os.path.exists(sample_path):
                engine = "zarr" if sample_path.endswith('.zarr') or os.path.isdir(sample_path) else "netcdf4"
                try:
                    # Open lazily without loading data into memory
                    with xr.open_dataset(sample_path, engine=engine) as ds:
                        # Filter out spatial structures, the variable itself, and root proxy coordinates
                        ignore_dims = {'x', 'y', 'lon', 'lat', 'spatial_ref', 'crs', 'band', 'projection', sample_var}
                        
                        for coord in ds.coords:
                            if coord not in ignore_dims:
                                # Extract unique coordinate values
                                raw_vals = pd.unique(ds.coords[coord].values)
                                
                                # Cast values to pure Python types for safe JSON serialization
                                if np.issubdtype(raw_vals.dtype, np.datetime64):
                                    clean_vals = [str(pd.Timestamp(v).isoformat()) for v in raw_vals]
                                else:
                                    clean_vals = [v.item() if hasattr(v, 'item') else str(v) for v in raw_vals]
                                
                                # Inject into the STAC item
                                stac_properties[f"cube:{level}_{coord}"] = clean_vals
                                
                except Exception as e:
                    log_execution(logger, f"Failed to dynamically extract dimensions from {sample_path}: {e}", logging.WARNING)

        # ---------------------------------------------------------------------
        # 4. Construct Item Base
        # ---------------------------------------------------------------------
        stac_id = item_id or f"{cube_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        item = pystac.Item(
            id=stac_id,
            geometry=footprint,
            bbox=wgs84_bounds,
            datetime=start_dt,
            properties=stac_properties
        )

        # ---------------------------------------------------------------------
        # 5. Link Physical Data Assets (The actual files mapped per variable)
        # ---------------------------------------------------------------------
        for level, variables in level_reprojected_paths.items():
            for var_name, file_path in variables.items():
                if not file_path or not os.path.exists(file_path):
                    continue

                is_zarr = file_path.endswith('.zarr') or os.path.isdir(file_path)
                media_type = "application/x-zarr" if is_zarr else "application/x-netcdf"
                
                resample_rule = self.get_resample_rule(var_name)
                asset_key = f"{level}_{var_name}"
                
                item.add_asset(
                    asset_key,
                    pystac.Asset(
                        href=file_path,
                        media_type=media_type,
                        roles=["data", "datacube"],
                        title=f"{var_name.upper()} ({level.title()})",
                        description=f"Spatially reprojected raster array warped using '{resample_rule}' resampling.",
                        extra_fields={
                            "processing:level": level,
                            "processing:variable": var_name,
                            "processing:resample_rule": resample_rule
                        }
                    )
                )

        log_execution(logger, f"=== STAC Item Assembly Complete: {stac_id} ===", logging.INFO)
        return item

    def parse_metadata(self, row: pd.Series, da: xr.DataArray) -> Tuple[str, xr.DataArray]:
        """
        Universal metadata parser that dynamically expands 2D spatial arrays into 
        N-dimensional DataArrays using metadata parameters from an execution manifest.

        This method bridges the gap between static 2D raster ingestion (via rioxarray) 
        and N-dimensional datacube assembly. By interpreting the columns of the execution 
        manifest row, it intelligently determines the primary Z-axis (e.g., 'time' or 'depth') 
        and promotes the 2D array into that dimensional space. Any remaining metadata columns 
        are attached as parallel non-dimensional coordinates, ensuring downstream aggregations 
        maintain full contextual awareness.

        Parameters
        ----------
        row : pd.Series
            A single row from the execution manifest DataFrame containing structural 
            routing flags (e.g., 'level', 'variable') and dynamic metadata dimensions 
            (e.g., 'time', 'depth', 'model').
        da : xr.DataArray
            The raw 2D spatial DataArray (Y, X) loaded from the physical raster or VRT.

        Returns
        -------
        tuple of (str, xr.DataArray)
            level : str
                The processing family grouping (e.g., 'climatologies', 'crop_types') extracted 
                from the row, used for downstream routing.
            da : xr.DataArray
                The enriched N-dimensional DataArray (Z, Y, X), with the Z-dimension and 
                auxiliary coordinates correctly assigned.

        Notes
        -----
        The resolution of the primary Z-dimension follows a strict hierarchy:
        1. **Time:** If a 'time' column exists, it becomes the primary dimension. Timezone 
        awareness is stripped to prevent serialization errors in NetCDF/Zarr formats.
        2. **Single Auxiliary Dimension:** If no time exists but exactly one valid metadata 
        column is present (e.g., 'depth'), that column becomes the primary dimension.
        3. **Multiple Auxiliary Dimensions:** If multiple columns exist without time (e.g., 
        a climate projection with 'model' and 'scenario'), the primary dimension defaults 
        to a composite string named 'projection'.
        4. **Static Fallback:** If no metadata exists, the data is assigned a generic 'index' 
        dimension.
        """
        level = row['level']
        
        # Assign the physical variable name to the DataArray for consistent xarray merging
        da.name = row['variable']

        # Define structural columns that dictate pipeline routing rather than scientific dimensions.
        # 'nodata' is explicitly blacklisted so it is used strictly for physical masking/warping
        # and never attached as a scientific coordinate to the final xarray object.
        reserved_cols = {'level', 'variable', 'vsi_path', 'path_type', 'asset_type', 'epsg_codes', 'nodata'}
        
        # Extract all active metadata columns present in this specific execution row,
        # ignoring system columns and NaN values (which indicate the dimension doesn't apply to this dataset).
        active_meta = {
            col: row[col] for col in row.index 
            if col not in reserved_cols and pd.notna(row[col])
        }

        # --- 1. Determine Primary Z-Dimension and Coordinates ---
        
        if 'time' in active_meta:
            # Time is the universally preferred primary dimension for earth observation data
            dim_name = 'time'
            raw_time = active_meta['time']
            
            # Strip timezone info. Formats like NetCDF and Zarr frequently crash or fail to 
            # concatenate properly if datetime objects maintain explicit UTC timezone localization.
            coord_val = raw_time.tz_localize(None) if hasattr(raw_time, 'tz') and raw_time.tz is not None else raw_time
            
            # All other metadata (e.g., 'satellite', 'orbit') become auxiliary scalar coordinates
            aux_coords = {k: v for k, v in active_meta.items() if k != 'time'}

        elif len(active_meta) == 1:
            # Single auxiliary dimension (e.g., a static DEM broken down by 'depth' or 'elevation')
            dim_name = list(active_meta.keys())[0]
            coord_val = list(active_meta.values())[0]
            aux_coords = {}

        elif len(active_meta) > 1:
            # Multiple non-temporal auxiliary dimensions (e.g., CMIP6 'model' + 'scenario').
            # Creates a composite primary dimension ('projection') while retaining the individual 
            # components as auxiliary coordinates.
            dim_name = 'projection'
            coord_val = "_".join(str(v) for v in active_meta.values())
            aux_coords = active_meta

        else:
            # Static 2D raster fallback (e.g., a single universal land cover mask)
            dim_name = 'index'
            coord_val = 0
            aux_coords = {}

        # --- 2. Expand Array & Assign Coordinates ---
        
        # Promote the 2D array (Y, X) to a 3D array (Z, Y, X) and assign the primary coordinate value.
        da = da.expand_dims(dim_name).assign_coords({dim_name: [coord_val]})

        # Attach auxiliary metadata as parallel scalar coordinates explicitly tied to the new Z-dimension.
        # Casting values to string ensures safe serialization to NetCDF/Zarr.
        for meta_key, meta_val in aux_coords.items():
            da = da.assign_coords({meta_key: (dim_name, [str(meta_val)])})

        # Rioxarray automatically injects a generic 'band' dimension (size 1) when opening standard GeoTIFFs.
        # We squeeze this out to prevent 4D arrays (Z, band, Y, X) and unnecessary dimension clutter.
        if "band" in da.coords:
            da = da.squeeze("band", drop=True)

        return level, da

    def process_cube(
        self, 
        recipe: Dict[str, Any], 
        max_workers: int = 10,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        The universal out-of-core spatial processing switchboard.

        This acts as the main entry point for data generation. It reads the execution plan, 
        initiates asynchronous network fetches to cloud buckets, routes execution to either 
        chunked Dask materialization (slices) or VRT-based mosaicking (tiles) based on the 
        data's detected spatial architecture, and finally serializes a PySTAC item.

        Parameters
        ----------
        recipe : Dict[str, Any]
            The master pipeline configuration dictionary parsed from YAML.
        max_workers : int, optional
            The maximum number of concurrent threads/processes allowed for network 
            I/O and GDAL C++ warping, by default 10.
        logger : Optional[logging.Logger], optional
            The logger instance for tracking telemetry.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A nested dictionary mapping hierarchical levels and variable names to their 
            final materialized NetCDF/Zarr filepaths on disk (e.g., `{'monthly': {'tas': '.../tas.nc'}}`).
        """
        import math

        # Unpack execution context and base directory structures
        ctx, logger, tracker = self.initialize_pipeline(recipe, logger=logger)
        base_dir, cube_name, dataset_name = ctx["base_dir"], ctx["cube_name"], ctx["dataset_name"]
        spatial_cfg = ctx["spatial_cfg"]

        execution_plan = self.generate_execution_plan(recipe, logger)
        if execution_plan.empty:
            log_execution(logger, "Terminating pipeline: no candidate asset catalog generated.", logging.WARNING)
            return {}

        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)
        grid_info = self.GRID_REGISTRY[target_grid_key]
        target_crs, target_res = grid_info["crs"], grid_info["resolution"]
        
        # Resolve target bounding coordinates
        bbox_cfg = spatial_cfg.get('bbox', {})
        wgs84_bounds = (
            min(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            min(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0)),
            max(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            max(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0))
        )
        target_bounds = transform_bounds("EPSG:4326", target_crs, *wgs84_bounds)
        
        # --- 1. Dynamic Session & Kwargs for the Initial Ping ---
        # Ping the remote server with the first file to establish bounds and connection
        sample_row = execution_plan.iloc[0]
        sample_file_path = sample_row['vsi_path'][0] if isinstance(sample_row['vsi_path'], list) else sample_row['vsi_path']
        
        provider_id = sample_row.get('provider_id')
        if isinstance(provider_id, pd.Series):
            provider_id = provider_id.iloc[0]

        active_session = None
        env_kwargs = {}
        
        # Unpack the tuple to extract the AWSSession AND the GDAL environment kwargs
        if hasattr(self, 'get_cloud_session') and provider_id:
            session_result = self.get_cloud_session(provider_id)
            if isinstance(session_result, tuple):
                active_session, env_kwargs = session_result
            else:
                active_session = session_result
        
        # Inject the custom CDSE kwargs (like AWS_S3_ENDPOINT) into GDAL explicitly 
        with rasterio.Env(**env_kwargs):
            source_bbox = build_envelope_from_file(
                target_crs, 
                target_bounds, 
                sample_file_path, 
                5, 
                logger,
                cloud_session=active_session
            )

        level_reprojected_paths: Dict[str, Dict[str, str]] = {lvl: {} for lvl in execution_plan['level'].unique()}
        grouped_plan = execution_plan.groupby(['level', 'variable'])
        
        # Process the plan iteratively per variable to avoid memory saturation
        for (level, var_name), group_df in grouped_plan:
            log_execution(logger, f"\nProcessing Level: '{level}' | Variable: '{var_name}'...", logging.INFO)
            tracker.log_usage(f"START Processing {var_name}")
            
            target_paths = group_df['vsi_path'].explode().unique().tolist()

            # --- 2. Dynamic Session & Kwargs for the Worker Threads ---
            group_provider_id = group_df['provider_id'].iloc[0] if 'provider_id' in group_df.columns else None
            
            active_session = None
            env_kwargs = {}
            if hasattr(self, 'get_cloud_session') and group_provider_id:
                session_result = self.get_cloud_session(group_provider_id)
                if isinstance(session_result, tuple):
                    active_session, env_kwargs = session_result
                else:
                    active_session = session_result
            
            num_files = len(target_paths)
            if num_files > 0:
                # Sub-linear thread allocation prevents network saturation and DDOS-like behavior against HTTP stores
                calc_workers = int(math.sqrt(num_files))
                dynamic_workers = max(1, min(calc_workers, num_files, max_workers))
            else:
                dynamic_workers = 1
            
            log_execution(
                logger, 
                f"Initiating network fetch for {num_files} asset(s) using {dynamic_workers} worker(s) "
                f"(Sub-linear sqrt({num_files}) thread allocation)...", 
                logging.INFO
            )
            
            with tracker.track_strain(f"Network Fetch ({var_name})"):
                # Apply explicit kwargs like AWS_VIRTUAL_HOSTING="FALSE" to the network fetchers
                with rasterio.Env(**env_kwargs):
                    raw_fetched_data = parallel_fetch_rasters(
                        paths=target_paths, 
                        geom=source_bbox, 
                        max_workers=dynamic_workers,
                        cloud_session=active_session 
                    )

            # Check internal routing flag computed during plan generation
            is_tiled = (group_df['asset_type'] == 'tile').any()

            if is_tiled:
                # Route tiled data (multiple spatial chunks per timestep) through VRT builder
                log_execution(logger, "Processing data through tiled mosaic pipeline", logging.INFO)
                output_path = self._process_tiled_mosaic(
                    group_df=group_df,
                    var_name=var_name,
                    level=level,
                    ctx=ctx,
                    target_bounds=target_bounds,     
                    target_grid_key=target_grid_key,
                    target_crs=target_crs,            
                    target_res=target_res,            
                    max_workers=max_workers,          
                    tracker=tracker,                  
                    logger=logger,
                    cloud_session=active_session,
                    source_bbox=source_bbox,
                    env_kwargs=env_kwargs  # Pass kwargs downstream
                )
                
                if output_path:
                    level_reprojected_paths[level][var_name] = output_path
                    
            else:
                # Route single global slices directly to chunked windowing
                log_execution(logger, "Processing data through chunked timeseries pipeline", logging.INFO)
                output_path = self._process_chunked_timeseries(
                    group_df=group_df,
                    var_name=var_name,
                    level=level,
                    ctx=ctx,
                    target_bounds=target_bounds,
                    target_grid_key=target_grid_key,
                    target_crs=target_crs,
                    target_res=target_res,
                    raw_fetched_data=raw_fetched_data,
                    max_workers=max_workers,
                    tracker=tracker,
                    logger=logger,
                    cloud_session=active_session,
                    env_kwargs=env_kwargs  # Pass kwargs downstream
                )
                if output_path:
                    level_reprojected_paths[level][var_name] = output_path

            tracker.log_usage(f"END Processing {var_name}")
            
        # --- STAC Finalization ---
        # Generate PySTAC item describing the successfully materialized cube structure
        stac_assets_dir = os.path.join(base_dir, cube_name, "meta", "STAC_assets")
        os.makedirs(stac_assets_dir, exist_ok=True)
        stac_item = self.generate_raster_stac_item(level_reprojected_paths, recipe, dataset_name, stac_assets_dir, logger=logger)
        
        stac_filepath = os.path.join(stac_assets_dir, f"{dataset_name}_stac.json")
        with open(stac_filepath, "w") as f:
            json.dump(stac_item.to_dict(), f, indent=4)
            
        return level_reprojected_paths

    def _execute_standard_warp(
        self, snapped_list, rule: str, cache_dir: str, target_grid_key: str, 
        max_workers: int, cfg: dict, tracker, logger, cloud_session=None, env_kwargs=None
    ) -> List[str]:
        """
        Executes standard coordinate reprojection and interpolation (warping) on continuous data.

        Dynamically chooses between native C++ GDAL multi-threading (for large dense matrices) 
        and Python ThreadPool execution (for rapid, small parallel slices) based on matrix size.

        Parameters
        ----------
        snapped_list : List[xr.DataArray]
            List of 2D Xarray spatial matrices to be reprojected.
        rule : str
            GDAL interpolation rule (e.g., 'bilinear', 'nearest', 'average').
        cache_dir : str
            Filesystem location to store the interim warped TIF shards.
        target_grid_key : str
            The internal registry ID representing the CRS and physical layout parameters.
        max_workers : int
            The maximum permitted concurrent threads.
        cfg : dict
            Performance configuration dictionary (e.g., slice thresholds).
        tracker : Any
            Memory and CPU usage telemetry tracker.
        logger : logging.Logger
            Execution logger.
        cloud_session : AWSSession, optional
            Active authenticated Boto3/Rasterio cloud session.
        env_kwargs : dict, optional
            Additional GDAL environment variables (e.g., HTTP retry settings).

        Returns
        -------
        List[str]
            A list of physical disk paths pointing to the successfully warped interim TIF slices.
        """
        import os
        import math
        import numpy as np
        import rasterio
        from concurrent.futures import ThreadPoolExecutor

        # Calculate uncompressed memory footprint to decide on execution backend
        slice_size_mb = snapped_list[0].nbytes / (1024 * 1024)
        env_kwargs = env_kwargs or {}

        def _warp_worker(da_2d: xr.DataArray, index: int, gdal_threads: str) -> str:
            """Internal worker for serializing, sanitizing, and reprojecting a single slice."""
            worker_kwargs = {
                "GDAL_NUM_THREADS": gdal_threads,
                "VSI_CACHE": "FALSE", 
                "GTIFF_SRS_SOURCE": "EPSG",
                "GDAL_HTTP_MAX_RETRY": "5",          # Increase from default 3 to mitigate remote disconnects
                "GDAL_HTTP_RETRY_DELAY": "3",        # Wait 3 seconds between retries
                "GDAL_HTTP_TIMEOUT": "30"            # Allow 30 seconds for a response
            }
            
            # Safely merge custom endpoints while keeping raw credentials out of rasterio.Env 
            # to prevent accidental leakage into GDAL internal logs
            safe_env = {k: v for k, v in env_kwargs.items() if k not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]}
            worker_kwargs.update(safe_env)
            
            env_context = rasterio.Env(session=cloud_session, **worker_kwargs) if cloud_session else rasterio.Env(**worker_kwargs)

            with env_context:
                try:
                    # Sanitize nodata values to prevent integer overflow issues during float scaling
                    nodata_val = da_2d.rio.nodata
                    if nodata_val is not None:
                        if np.issubdtype(da_2d.dtype, np.integer):
                            limits = np.iinfo(da_2d.dtype)
                            if not (limits.min <= nodata_val <= limits.max):
                                da_2d = da_2d.astype('float32')
                        da_2d.rio.write_nodata(nodata_val, inplace=True)

                    # Remove conflicting legacy NetCDF fill values which cause rioxarray export crashes
                    if '_FillValue' in da_2d.attrs:
                        del da_2d.attrs['_FillValue']

                    da_2d = self._sanitize_spatial_geometry(da_2d, default_crs="EPSG:4326", logger=None)
                    out_filepath = os.path.join(cache_dir, f"slice_{index:04d}.tif")
                    
                    self.affine_reproject(
                        input_data=da_2d, 
                        output_filepath=out_filepath, 
                        grid_name=target_grid_key, 
                        resample_keyword=rule, 
                        num_threads=gdal_threads, 
                        logger=None  
                    )
                    return out_filepath
                except Exception as e:
                    log_execution(logger, f"CRITICAL: GDAL failed to warp slice {index}. Error: {e}", logging.ERROR)
                    return ""

        warped_tif_paths = []
        num_slices = len(snapped_list)

        # Route to ThreadPoolExecutor if slices are small enough to be rapidly processed in pure Python threads
        if slice_size_mb < cfg.get("large_slice_threshold_mb", 50.0):
            if num_slices < cfg.get("min_slices_for_parallel", 16):
                with tracker.track_strain(f"Main-Thread Sequential Warp ({snapped_list[0].name})"):
                    for i, da in enumerate(snapped_list):
                        result = _warp_worker(da, i, "1")
                        if result != "": warped_tif_paths.append(result)
            else:
                calc_workers = int(math.sqrt(num_slices))
                dynamic_workers = max(1, min(calc_workers, max_workers, cfg.get("max_safe_threadpool_workers", 8)))
                with tracker.track_strain(f"Parallel ThreadPool Warp ({snapped_list[0].name})"):
                    with ThreadPoolExecutor(max_workers=dynamic_workers) as executor:
                        futures = [executor.submit(_warp_worker, da, i, "1") for i, da in enumerate(snapped_list)]
                        for f in futures:
                            res = f.result()
                            if res != "": warped_tif_paths.append(res)
        else:
            # For massive matrices, delegate concurrency entirely to the internal C++ GDAL multi-threading engine
            num_gdal_threads = str(min(os.cpu_count() or 4, max_workers))
            with tracker.track_strain(f"C++ Multi-Core Native Warp ({snapped_list[0].name})"):
                for i, da in enumerate(snapped_list):
                    result_path = _warp_worker(da, i, num_gdal_threads)
                    if result_path != "": warped_tif_paths.append(result_path)

        return warped_tif_paths

    def _execute_fractional_warp(
        self, snapped_list: list, target_classes: dict, cache_dir: str, 
        target_grid_key: str, tracker, logger, cloud_session=None, env_kwargs=None
    ) -> dict:
        """
        Executes highly specialized fractional warping for categorical/classification datasets.

        Unlike standard interpolation (which destroys categorical integrity), this extracts 
        each valid classification integer as an isolated boolean layer, spatially warps it, 
        and computes exact areal representation percentages in the target resolution pixel.

        Parameters
        ----------
        snapped_list : list of xr.DataArray
            List of 2D Xarray categorical matrices.
        target_classes : dict
            Dictionary mapping raw integer values to their semantic string classes.
        cache_dir : str
            Directory to store interim single-class TIF output shards.
        target_grid_key : str
            Target CRS and dimension parameters string key.
        tracker : Any
            Memory and CPU usage telemetry tracker.
        logger : logging.Logger
            Execution logger.
        cloud_session : AWSSession, optional
            Authenticated cloud session for remote data reads.
        env_kwargs : dict, optional
            Additional environment arguments to pass to GDAL.

        Returns
        -------
        dict
            A dictionary mapping each class integer to a list of its warped 
            fractional TIF filepath locations.
        """
        import rasterio
        
        class_paths = {class_int: [] for class_int in target_classes.keys()}
        worker_kwargs = {
            "GDAL_NUM_THREADS": "ALL_CPUS",      # Utilize all cores for heavy fractional map-reduce
            "VSI_CACHE": "FALSE", 
            "GTIFF_SRS_SOURCE": "EPSG",
            "GDAL_HTTP_MAX_RETRY": "5",          
            "GDAL_HTTP_RETRY_DELAY": "3",        # Wait 3 seconds between retries
            "GDAL_HTTP_TIMEOUT": "30"            # Allow 30 seconds for a response
        }
        if env_kwargs:
            safe_env = {k: v for k, v in env_kwargs.items() if k not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]}
            worker_kwargs.update(safe_env)
            
        if not cloud_session:
            worker_kwargs["AWS_NO_SIGN_REQUEST"] = "YES"

        env_context = rasterio.Env(session=cloud_session, **worker_kwargs) if cloud_session else rasterio.Env(**worker_kwargs)

        with tracker.track_strain(f"Fractional Warping ({len(target_classes)} classes)"):
            with env_context: 
                for i, da_2d in enumerate(snapped_list):
                    for class_int, class_name in target_classes.items():
                        out_filepath = os.path.join(cache_dir, f"slice_{i:04d}_cls{class_int}.tif")
                        
                        # Defers computation to the child class implementation (e.g. clms.py)
                        result_path = self.compute_class_fraction(
                            source_data=da_2d, class_value=class_int, grid_name=target_grid_key,
                            output_filepath=out_filepath, logger=logger
                        )
                        class_paths[class_int].append(result_path)
                    
        return class_paths
    
    def _process_tiled_mosaic(
        self, group_df, var_name, level, ctx, target_bounds, target_grid_key, 
        target_crs, target_res, max_workers, tracker, logger, cloud_session=None, source_bbox=None, env_kwargs=None
    ):
        """
        Processes distributed spatial tiles by virtually grouping them into an actionable mosaic.

        Uses GDAL Virtual Rasters (VRTs) to wrap multiple individual spatial files (e.g. Eurostat grids) 
        into a single logical monolithic array representation prior to feeding it into the alignment pipeline.

        Parameters
        ----------
        group_df : pd.DataFrame
            The DataFrame partition containing all URLs for a specific temporal timestep.
        var_name : str
            The name of the variable being mosaicked (e.g., 'TCD', 'landcover').
        level : str
            The temporal hierarchy grouping (e.g., 'monthly', 'clc').
        ctx : dict
            Master context dictionary containing structural directory locations.
        target_bounds : tuple
            Projected destination bounding box (minx, miny, maxx, maxy).
        target_grid_key : str
            Target resolution and CRS identifier.
        target_crs : Any
            The specific coordinate reference system to project into.
        target_res : float
            The pixel size of the target space.
        max_workers : int
            Allowed concurrent processes.
        tracker : Any
            Telemetry and usage tracking module.
        logger : logging.Logger
            Execution logging output module.
        cloud_session : AWSSession, optional
            Authenticated cloud session for S3 queries.
        source_bbox : tuple, optional
            Bounding box in the source projection for safe VRT clipping.
        env_kwargs : dict, optional
            Safe dictionary of GDAL HTTP environment variables.

        Returns
        -------
        Optional[str]
            Path to the final assembled multidimensional output format (Zarr/NetCDF), or None on failure.
        """
        import os
        import rioxarray
        import rasterio
        from bmc.utils.io import s3_to_vrt

        env_kwargs = env_kwargs or {}
        base_dir = ctx["base_dir"]
        cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        da_list = []
        
        for idx, row in group_df.iterrows():
            tile_paths = row['vsi_path']
            if not isinstance(tile_paths, list):
                tile_paths = [tile_paths]
                
            if not tile_paths:
                continue
                
            vrt_path = os.path.join(cache_dir, f"slice_mosaic_{idx}.vrt")
            
            with tracker.track_strain(f"VRT Construction ({var_name} slice {idx})"):
                # Clean the env_kwargs to ensure rasterio.Env accepts them cleanly
                safe_env = {k: v for k, v in env_kwargs.items() if k not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]}
                with rasterio.Env(session=cloud_session, **safe_env) if cloud_session else rasterio.Env(**safe_env):
                    # Construct a virtual raster locally bridging the remote distributed tiles
                    success = s3_to_vrt(
                        s3_hrefs=tile_paths, vrt_path=vrt_path, cloud_session=cloud_session,
                        resample_alg="near", output_bounds=source_bbox, env_kwargs=env_kwargs
                    )
                if not success:
                    log_execution(logger, f"Failed to build VRT at {vrt_path}.", logging.ERROR)
                    continue
            
            # Keep the context open so rioxarray can safely query the remote tiles referenced by the VRT
            env_context = rasterio.Env(session=cloud_session, **safe_env) if cloud_session else rasterio.Env(**safe_env)
            with env_context:
                vrt_da = rioxarray.open_rasterio(vrt_path, chunks=True)
                _, structured_da = self.parse_metadata(row, vrt_da)
                da_list.append(structured_da)
            
        return self._align_warp_and_export(
            da_list=da_list, var_name=var_name, level=level, ctx=ctx, target_bounds=target_bounds,
            target_grid_key=target_grid_key, target_crs=target_crs, target_res=target_res,
            max_workers=max_workers, tracker=tracker, logger=logger, cloud_session=cloud_session, env_kwargs=env_kwargs
        )

    def _process_chunked_timeseries(
        self, group_df, var_name: str, level: str, ctx: dict, 
        target_bounds: tuple, target_grid_key: str, 
        target_crs, target_res: float, raw_fetched_data: dict, 
        max_workers: int, tracker, logger, cloud_session=None, env_kwargs=None
    ):
        """
        Processes standard monolithic files representing entire regional/global slices.

        Unlike the mosaicking pathway, these source assets cover the full geographic 
        extent requested and thus are immediately funneled into Xarray metadata assignment 
        prior to parallel alignment.

        Parameters
        ----------
        group_df : pd.DataFrame
            DataFrame of slice paths mapped by timestamp.
        var_name : str
            Variable name identifier (e.g. 'tas', 'qkndvi').
        level : str
            Temporal resolution (e.g. 'monthly', 'annual').
        ctx : dict
            Master directory and configuration context block.
        target_bounds : tuple
            Bounding envelope coordinates.
        target_grid_key : str
            Destination configuration registry key.
        target_crs : Any
            EPSG output projection space.
        target_res : float
            Pixel size mapping target.
        raw_fetched_data : dict
            Dictionary mapping remote filepaths to their lazily opened Xarray DataArrays.
        max_workers : int
            Thread limitation allocation.
        tracker : Any
            Telemetry usage logger.
        logger : logging.Logger
            Logging stream destination.
        cloud_session : AWSSession, optional
            Injected cloud authentication environment.
        env_kwargs : dict, optional
            Custom GDAL HTTP argument dictionary.

        Returns
        -------
        Optional[str]
            Output path for the serialized Zarr/NetCDF array, or None if extraction failed.
        """
        import rasterio
        
        env_kwargs = env_kwargs or {}
        safe_env = {k: v for k, v in env_kwargs.items() if k not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]}
        
        da_list = []
        
        # Determine the environment context dynamically to protect lazy Dask evaluations when triggered
        env_context = rasterio.Env(session=cloud_session, **safe_env) if cloud_session else rasterio.Env(**safe_env)
        
        with env_context:
            for _, row in group_df.iterrows():
                path_key = row['vsi_path'][0] if isinstance(row['vsi_path'], list) else row['vsi_path']
                raw_da = raw_fetched_data.get(path_key)
                
                if raw_da is not None:
                    _, structured_da = self.parse_metadata(row, raw_da)
                    da_list.append(structured_da)
                    
            if not da_list:
                log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
                return None
                
            return self._align_warp_and_export(
                da_list=da_list,
                var_name=var_name,
                level=level,
                ctx=ctx,
                target_bounds=target_bounds,
                target_grid_key=target_grid_key,
                target_crs=target_crs,
                target_res=target_res,
                max_workers=max_workers,
                tracker=tracker,
                logger=logger,
                cloud_session=cloud_session,
                env_kwargs=env_kwargs
            )

    def _align_warp_and_export(
        self, da_list: List[xr.DataArray], var_name: str, level: str, ctx: Dict[str, Any], 
        target_bounds: Tuple[float, float, float, float], target_grid_key: str, 
        target_crs: Any, target_res: float, max_workers: int, tracker: Any, 
        logger: logging.Logger, cloud_session=None, env_kwargs=None
    ) -> Optional[str]:
        """
        Acts as the central coordination hub bridging raw opened data arrays to the final GDAL warp calls.

        Ensures all DataArrays are dimensionally conformal (Z-axis sorting and coordinate stacking), 
        determines categorical vs standard pipeline routing, coordinates the warp phase, and triggers 
        the ultimate file assembly and export function.

        Parameters
        ----------
        da_list : List[xr.DataArray]
            List of raw opened Xarray slices lacking final structural alignments.
        var_name : str
            Variable name being processed.
        level : str
            Hierarchical level descriptor (e.g. 'annual').
        ctx : dict
            Master output directory context.
        target_bounds : Tuple[float, float, float, float]
            Desired spatial projection boundaries.
        target_grid_key : str
            Master grid ID mapping.
        target_crs : Any
            Target EPSG coordinate reference object.
        target_res : float
            Pixel size parameter.
        max_workers : int
            Total maximum allowed threads for underlying C++/Python pools.
        tracker : Any
            Memory tracking diagnostic payload handler.
        logger : logging.Logger
            Logging stream destination.
        cloud_session : AWSSession, optional
            Authenticated cloud hook for S3.
        env_kwargs : dict, optional
            Additional GDAL HTTP runtime attributes.

        Returns
        -------
        Optional[str]
            Path to the completed output file, or None if the sequence failed.
        """
        
        if not da_list:
            log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
            return None

        base_dir = ctx["base_dir"]
        
        # --- 1. Z-Axis Alignment & Coordinate Extraction ---
        # Ensure perfect planar conformance by broadcasting the x/y basis of the first 
        # slice across the stack before extracting the primary temporal 'Z' tracking dimension.
        base_x, base_y = da_list[0].coords['x'], da_list[0].coords['y']
        snapped_list = [d.assign_coords(x=base_x, y=base_y) for d in da_list]
        z_dim = [dim for dim in snapped_list[0].dims if dim not in ['x', 'y']][0]
        
        # Chronological / sequential sorting ensures identical output parity
        snapped_list.sort(key=lambda da: da.coords[z_dim].values[0])
        
        z_vals = np.array([da[z_dim].values for da in snapped_list]).flatten()
        full_meta_coords = {z_dim: z_vals}
        
        # Extract arbitrary supplemental coordinates (like EPSG tracking or source flags)
        for k in snapped_list[0].coords.keys():
            if k not in ['x', 'y', 'spatial_ref', z_dim]:
                meta_vector = np.array([da[k].values for da in snapped_list]).flatten()
                full_meta_coords[k] = (z_dim, meta_vector)
        
        rule = self.get_resample_rule(var_name)
        cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
        os.makedirs(cache_dir, exist_ok=True)

        # --- 2. Pipeline Routing ---
        if rule == "class_fraction":
            # Extract categorical integers natively provided by the child class
            target_classes = getattr(self, "class_mapping", {}).get(var_name, {})
            if not target_classes:
                log_execution(logger, f"No class mapping found for {var_name}. Aborting fraction warp.", logging.ERROR)
                return None
                
            warped_tif_paths = self._execute_fractional_warp(
                snapped_list, target_classes, cache_dir, target_grid_key, tracker, logger,
                cloud_session=cloud_session, env_kwargs=env_kwargs
            )
        else:
            warped_tif_paths = self._execute_standard_warp(
                snapped_list, rule, cache_dir, target_grid_key, max_workers, 
                ctx.get("execution_config", {}), tracker, logger, cloud_session, env_kwargs=env_kwargs
            )

        # --- 3. Materialization & Export ---
        return self._assemble_and_export_cube(
            warped_tif_paths, var_name, level, ctx, z_dim, full_meta_coords, 
            target_bounds, target_crs, target_res, rule, target_classes if rule == "class_fraction" else None, 
            tracker, logger
        )

    def _assemble_and_export_cube(
        self, warped_paths: Union[List[str], Dict[int, List[str]]], var_name: str, level: str, 
        ctx: Dict[str, Any], z_dim: str, full_meta_coords: dict, target_bounds: Tuple[float, float, float, float], 
        target_crs: Any, target_res: float, rule: str, target_classes: Optional[Dict[int, str]], 
        tracker: Any, logger: logging.Logger
    ) -> Optional[str]:
        """
        Stacks intermediate warped shards into a monolithic multidimensional array and exports to disk.

        This function reads the serialized interim TIF files, concatenates them structurally along 
        the temporal 'Z' axis, enforces CF-Compliant (Climate and Forecast) metadata schemas 
        (particularly critical for fractional class dictionaries), materializes the lazy Dask 
        graph into RAM, and orchestrates the final disk write (NetCDF/Zarr).

        Notes
        -----
        NetCDF Export Concurrency Warning:
        NetCDF (backed by HDF5) heavily relies on POSIX file locking mechanisms. If `max_workers` 
        is set aggressively high (>1) when exporting to `netcdf` format, silent write-collisions 
        and race conditions may occur when chunks attempt to write simultaneously. This is 
        frequently visualized as perfectly square nodata "voids" or white holes dropped entirely 
        from the output cube. To guarantee robust writes without dropping chunks, use Zarr formatting, 
        or lower the orchestrator's concurrency limit (`max_workers=1`).

        Parameters
        ----------
        warped_paths : Union[List[str], Dict[int, List[str]]]
            Flat list of shard paths (for continuous data) or a nested mapping linking 
            categorical integer keys to their respective list of shard paths (fractional data).
        var_name : str
            The active variable name.
        level : str
            Pipeline hierarchy string grouping (e.g. 'annual').
        ctx : dict
            Orchestration execution parameters and configuration.
        z_dim : str
            The name assigned to the temporal z-axis stack (e.g. 'time').
        full_meta_coords : dict
            Supplemental dimension coordinates to be appended prior to export.
        target_bounds : Tuple[float, float, float, float]
            Precise clipping boundaries strictly enforced on the assembled array.
        target_crs : Any
            The EPSG Coordinate system embedded in the CF metadata.
        target_res : float
            Resolution cell size assigned to CF metadata tracking.
        rule : str
            The logical resample rule previously used to shape the data ('class_fraction' forces categorical schema logic).
        target_classes : Optional[Dict[int, str]]
            The integer-to-string mapping used for annotating CF categorical flag_values.
        tracker : Any
            System memory telemetry monitor.
        logger : logging.Logger
            Logging output destination.

        Returns
        -------
        Optional[str]
            Path to the completed NetCDF/Zarr file payload, or None on critical error.
        """
        import os
        import gc
        import time
        import json
        import xarray as xr
        import rioxarray
        from rioxarray.enum import Convention

        base_dir, cube_name, dataset_name = ctx["base_dir"], ctx["cube_name"], ctx["dataset_name"]
        export_format = ctx["export_format"]
        cfg = ctx.get("execution_config", {})
        block_size = cfg.get("gdal_warp_block_size", 2048)
        
        def _load_and_stack(paths: List[str]) -> xr.DataArray:
            """Internal closure dynamically generating and squeezing Dask-backed xarray slices."""
            slices = []
            for p in paths:
                da = rioxarray.open_rasterio(p, chunks={'x': block_size, 'y': block_size})
                if 'band' in da.dims: 
                    da = da.squeeze('band', drop=True)
                slices.append(da)
            return xr.concat(slices, dim=z_dim) if slices else None

        # Assemble and structure fractional datasets (Categorical Classifications)
        if rule == "class_fraction":
            class_das = []
            for class_int, paths in warped_paths.items():
                if not paths: continue
                z_stacked = _load_and_stack(paths)
                if z_stacked is not None:
                    # Assign the native integer coordinate natively to allow discrete masking downstream
                    z_stacked = z_stacked.expand_dims('class_id').assign_coords({'class_id': [class_int]})
                    class_das.append(z_stacked)
            if not class_das:
                log_execution(logger, f"No warped slices produced for {var_name}.", logging.ERROR)
                return None
            
            # Concatenate along the new integer dimension (Yielding a true 4D (x, y, time, class_id) array)
            combined_da = xr.concat(class_das, dim='class_id')
            
            # --- INJECT CF-COMPLIANT METADATA ---
            if target_classes:
                # 1. CF attributes should represent the entire dictionary namespace (Flag Bindings)
                all_class_ints = list(target_classes.keys())
                all_class_strs = [target_classes[c] for c in all_class_ints]
                # CF strictly requires spaces to be replaced by underscores for flag mappings
                safe_strings = [s.replace(" ", "_") for s in all_class_strs]
                
                combined_da.attrs['flag_values'] = all_class_ints
                combined_da.attrs['flag_meanings'] = " ".join(safe_strings)
                combined_da.attrs['raw_class_mapping'] = json.dumps(target_classes)

                # 2. The parallel textual coordinate MUST match the exact length of the actually assembled dimension
                actual_class_ints = combined_da['class_id'].values.tolist()
                actual_class_strs = [target_classes[c] for c in actual_class_ints]
                combined_da = combined_da.assign_coords(class_name=("class_id", actual_class_strs))
                
        # Assemble standard continuous arrays
        else:
            if not warped_paths:
                log_execution(logger, f"No warped slices produced for {var_name}.", logging.ERROR)
                return None
            combined_da = _load_and_stack(warped_paths)
            
        if combined_da is None:
            return None

        # Broadcast required dimensions and rigid limits before physical execution
        combined_da.name = var_name
        combined_da = combined_da.assign_coords(full_meta_coords)
        combined_da = combined_da.rio.clip_box(*target_bounds)

        # Triggers materialization of the lazy Dask Graph into system RAM
        with tracker.track_strain(f"Dask Materialization ({var_name})"):
            combined_da = combined_da.load()
            
        combined_da = combined_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        combined_da = combined_da.rio.write_crs(target_crs, convention=Convention.CF)
        combined_da = combined_da.rio.write_transform(convention=Convention.CF)
        combined_da.attrs['crs'] = str(target_crs)
        combined_da.attrs['res'] = float(target_res)

        level_dir = os.path.join(base_dir, cube_name, dataset_name, level)
        os.makedirs(level_dir, exist_ok=True)
        export_ds = combined_da.to_dataset(name=var_name)

        # NetCDF export strictness: Forces arbitrarily typed 'object' columns/attributes into proper strings
        for v_key in export_ds.variables:
            if export_ds[v_key].dtype == 'O':
                export_ds[v_key] = export_ds[v_key].astype(str)

        # Execute final serialized write commands to disk
        if export_format == 'zarr':
            var_cache_path = os.path.join(level_dir, f"{var_name}.zarr")
            export_ds.to_zarr(var_cache_path, mode='w')
        else:
            var_cache_path = os.path.join(level_dir, f"{var_name}.nc")
            export_ds.to_netcdf(var_cache_path, engine="netcdf4", format="NETCDF4")
        
        # Purge dangling caches to prevent memory leaks across loops
        xr.backends.file_manager.FILE_CACHE.clear()
        del combined_da, export_ds
        gc.collect()

        time.sleep(cfg.get("warp_cache_cleanup_delay_sec", 30))
        
        return var_cache_path