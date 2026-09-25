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

    def generate_execution_plan(self, recipe: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
        """
        Filters and groups a standardized GeoParquet catalog to generate a pipeline-ready execution manifest.

        This method queries a local or remote GeoParquet catalog based on spatial, temporal, 
        and categorical constraints defined in the execution recipe. It processes the catalog 
        level-by-level to ensure that auxiliary dimensions (e.g., specific sensor bands or 
        quality flags) are tightly scoped to their relevant datasets without creating NaN 
        conflicts across the broader manifest. The output is a highly structured DataFrame 
        that dictates exactly which files to fetch and how they should be routed through 
        the processing pipeline (e.g., as individual slices or mosaicked tiles).

        Parameters
        ----------
        recipe : Dict[str, Any]
            The parsed YAML configuration dictionary containing execution directives, 
            including spatial bounding boxes, temporal ranges, and active variables.
        logger : logging.Logger
            The execution logger used to record inventory fetching progress, schema 
            validation results, and potential missing data errors.

        Returns
        -------
        pd.DataFrame
            A standardized execution manifest detailing the processing queue. 
            Returns an empty DataFrame if the catalog is missing, invalid, or yields no results.
            Expected columns include:
            - 'level' (str): Processing family grouping (e.g., 'climatologies').
            - 'variable' (str): Physical variable name (e.g., 'temperature').
            - 'vsi_path' (list of str): GDAL-compatible data URIs for the targeted assets.
            - 'time' (datetime): The temporal coordinate for the data chunk.
            - 'asset_type' (str): Pipeline routing flag ('slice' for single files, 'tile' for multi-file mosaics).
            - 'path_type' (str): Storage indicator (hardcoded to 'remote' for S3 streams).
            - [Auxiliary Columns]: Any dynamically injected dimensions (e.g., 'band', 'orbit').
        """
        # --- 0. Initialize Dynamic Attributes ---
        # Hydrate the class instance with dataset-specific parameters from the recipe
        # (e.g., extracting active levels, variables, and potentially a custom catalog path).
        self._apply_recipe_attributes(recipe)

        # --- 1. Extract Config from Recipe ---
        # Safely extract spatial constraints, defaulting to a global bounding box if absent.
        bbox_cfg = recipe.get("spatial", {}).get("bbox", {})
        bbox = (
            bbox_cfg.get("long_min", -180), 
            bbox_cfg.get("lat_min", -90), 
            bbox_cfg.get("long_max", 180), 
            bbox_cfg.get("lat_max", 90)
        )
        
        # Safely extract temporal constraints, defaulting to a broad historical/future range.
        temp_cfg = recipe.get("temporal", {})
        start_date = f"{temp_cfg.get('start_year', 1900)}-01-01"
        end_date = f"{temp_cfg.get('end_year', 2100)}-12-31"

        # --- 2. Load Catalog ---
        catalog_path = getattr(self, 'catalog_path', None)
        if not catalog_path:
            log_execution(logger, "Fatal: 'catalog_path' attribute not set.", logging.ERROR)
            return pd.DataFrame()

        try:
            # GeoPandas natively handles reading partitioned parquet directories or single files.
            gdf = gpd.read_parquet(catalog_path)
        except Exception as e:
            log_execution(logger, f"Failed to load catalog: {e}", logging.ERROR)
            return pd.DataFrame()

        # --- 3. Strict Schema Validation ---
        # Ensure the catalog contains the absolute minimum required metadata to build a pipeline queue.
        required_cols = ['geometry', 'datetime', 'level', 'variable', 'vsi_path']
        missing = [c for c in required_cols if c not in gdf.columns]
        if missing:
            log_execution(logger, f"Catalog missing mandatory columns: {missing}. Aborting.", logging.ERROR)
            return pd.DataFrame()

        log_execution(logger, f"Initial catalog rows loaded: {len(gdf)}", logging.INFO)

        # Pre-fetch and parse configurations to avoid redundant overhead inside the level loop.
        search_geom = shapely.geometry.box(*bbox)
        start = pd.to_datetime(start_date, utc=True)
        end = pd.to_datetime(end_date, utc=True)
        
        # Identify levels that do not have a temporal component (e.g., static digital elevation models).
        static_levels = getattr(self, 'temporally_static_levels', [])
        aux_dims = self.get_auxiliary_dimensions(recipe)
        
        # Ensure levels and variables are iterable lists, even if defined as single strings.
        levels = self.levels if isinstance(self.levels, list) else [self.levels]
        variables = self.variables if isinstance(self.variables, list) else [self.variables]
        
        manifest_chunks = []

        # --- Process Level-by-Level ---
        # We iterate by level because auxiliary dimensions (like 'band' or 'polarization') 
        # might apply to one level but not another. Processing them together would cause NaN conflicts.
        for level in levels:
            
            # --- 4. Core Categorical Filter (Isolated per level) ---
            # Isolate the data for the current level and only keep the requested variables.
            level_gdf = gdf[(gdf['level'] == level) & (gdf['variable'].isin(variables))].copy()
            if level_gdf.empty:
                continue

            # --- 5. Auxiliary Dimension Filter & Grouping Injection (Scoped) ---
            # Define the base grouping keys required to assemble a unique processing unit.
            grouping_cols = ['level', 'variable', 'datetime']
            
            if aux_dims:
                for dim_col, values in aux_dims.items():
                    if dim_col in level_gdf.columns:
                        if not isinstance(values, list):
                            values = [values]
                        
                        # Prevent filtering if the auxiliary column exists but is entirely NaN for this specific level.
                        if not level_gdf[dim_col].isna().all():
                            # Keep rows matching the requested aux values, OR rows that don't use this dimension (NaN).
                            level_gdf = level_gdf[level_gdf[dim_col].isin(values) | level_gdf[dim_col].isna()]
                            
                            # Dynamically inject the auxiliary dimension into the groupby clause so slices aren't conflated.
                            if dim_col not in grouping_cols:
                                grouping_cols.append(dim_col)
            
            if level_gdf.empty:
                continue

            # --- 6. Spatial Filter ---
            # Retain only items whose bounding boxes intersect the requested spatial region.
            level_gdf = level_gdf[level_gdf.intersects(search_geom)].copy()
            if level_gdf.empty:
                continue

            # --- 7. Temporal Filter ---
            level_gdf['datetime'] = pd.to_datetime(level_gdf['datetime'], utc=True)
            
            # Skip time filtering for datasets known to be temporally static.
            if level not in static_levels:
                level_gdf = level_gdf[(level_gdf['datetime'] >= start) & (level_gdf['datetime'] <= end)]
            
            if level_gdf.empty:
                continue

            # --- 8. Grouping & Manifest Construction (Per Level) ---
            # Define how to aggregate rows that belong to the same processing unit (same variable, time, aux dims).
            agg_rules = {'vsi_path': list}  # Collapse all individual tile paths into a single list for VRT processing.
            
            if 'proj:epsg' in level_gdf.columns:
                agg_rules['proj:epsg'] = lambda x: list(set(x))
            if 'nodata' in level_gdf.columns:
                # Safely extract the primary NoData value, if one exists.
                agg_rules['nodata'] = lambda x: list(set(x))[0] if len(set(x)) > 0 else None

            # Group the data. dropna=True is safe here because we explicitly excluded 
            # completely NaN auxiliary columns from `grouping_cols` in Step 5.
            level_manifest = level_gdf.groupby(grouping_cols, dropna=True).agg(agg_rules).reset_index()
            manifest_chunks.append(level_manifest)

        # --- 9. Reassemble the Final Manifest ---
        if not manifest_chunks:
            return pd.DataFrame()

        # Stitch the isolated level manifests back into a single unified queue.
        manifest = pd.concat(manifest_chunks, ignore_index=True)

        # Map internal STAC/GeoParquet nomenclature to the pipeline's expected naming conventions.
        rename_map = {'datetime': 'time', 'proj:epsg': 'epsg_codes'}
        manifest = manifest.rename(columns={k: v for k, v in rename_map.items() if k in manifest.columns})

        # Dynamically determine the pipeline routing strategy.
        # If a grouping resulted in multiple VSI paths, it spans multiple spatial tiles and requires a VRT ('tile').
        # If it contains only a single path, it can be processed directly as a standalone array ('slice').
        manifest['asset_type'] = manifest['vsi_path'].apply(
            lambda paths: 'tile' if len(paths) > 1 else 'slice'
        )
        
        # Explicitly flag these assets as remote to trigger the appropriate GDAL virtual file system configs downstream.
        manifest['path_type'] = 'remote'

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
        Routes execution to either chunked Dask materialization or VRT-based streaming 
        based on the detected structure of the incoming data assets.
        """
        import math

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
        
        bbox_cfg = spatial_cfg.get('bbox', {})
        wgs84_bounds = (
            min(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            min(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0)),
            max(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            max(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0))
        )
        target_bounds = transform_bounds("EPSG:4326", target_crs, *wgs84_bounds)
        
        # Safely extract the first scalar URI string from the grouped list
        sample_file_path = execution_plan.iloc[0]['vsi_path'][0]
        source_bbox = build_envelope_from_file(target_crs, target_bounds, sample_file_path, 5, logger)

        level_reprojected_paths: Dict[str, Dict[str, str]] = {lvl: {} for lvl in execution_plan['level'].unique()}
        grouped_plan = execution_plan.groupby(['level', 'variable'])
        
        for (level, var_name), group_df in grouped_plan:
            log_execution(logger, f"\nProcessing Level: '{level}' | Variable: '{var_name}'...", logging.INFO)
            tracker.log_usage(f"START Processing {var_name}")
            
            target_paths = group_df['vsi_path'].explode().unique().tolist()
            
            # --- Inline Network Fetch Phase ---
            num_files = len(target_paths)
            if num_files > 0:
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
                with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"):
                    raw_fetched_data = parallel_fetch_rasters(target_paths, source_bbox, dynamic_workers)

            # --- Pipeline Routing Switchboard ---
            is_tiled = any(str(p).endswith('.zip') for p in target_paths)

            if is_tiled:
                log_execution(logger, "Processing data through tiled mosaic pipeline", logging.INFO)
                output_paths = self._process_tiled_mosaic(
                    group_df=group_df,
                    var_name=var_name,
                    level=level,
                    ctx=ctx,
                    target_grid_key=target_grid_key,
                    logger=logger
                )
                if isinstance(output_paths, list):
                    for i, p in enumerate(output_paths):
                        level_reprojected_paths[level][f"{var_name}_{i}"] = p
                elif output_paths:
                    level_reprojected_paths[level][var_name] = output_paths
                    
            else:
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
                    logger=logger
                )
                if output_path:
                    level_reprojected_paths[level][var_name] = output_path

            tracker.log_usage(f"END Processing {var_name}")
            
        # --- STAC Finalization ---
        stac_assets_dir = os.path.join(base_dir, cube_name, "meta", "STAC_assets")
        os.makedirs(stac_assets_dir, exist_ok=True)
        stac_item = self.generate_raster_stac_item(level_reprojected_paths, recipe, dataset_name, stac_assets_dir, logger=logger)
        
        stac_filepath = os.path.join(stac_assets_dir, f"{dataset_name}_stac.json")
        with open(stac_filepath, "w") as f:
            json.dump(stac_item.to_dict(), f, indent=4)
            
        return level_reprojected_paths


    def _process_tiled_mosaic(
        self, group_df, var_name, level, ctx, target_bounds, target_grid_key, 
        target_crs, target_res, max_workers, tracker, logger):
        """
        Executes the spatial VRT pipeline for multi-file spatial grids (tiles).
        Builds a Virtual Raster per N-dimensional slice, injects auxiliary coordinates, 
        and routes to the universal warping engine.
        """
        import os
        from osgeo import gdal
        import rioxarray

        base_dir = ctx["base_dir"]
        cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        da_list = []
        
        # 1. Assemble VRTs per multidimensional slice
        for idx, row in group_df.iterrows():
            tile_paths = row['vsi_path']
            if not isinstance(tile_paths, list):
                tile_paths = [tile_paths]
                
            if not tile_paths:
                continue
                
            vrt_path = os.path.join(cache_dir, f"slice_mosaic_{idx}.vrt")
            
            with tracker.track_strain(f"VRT Construction ({var_name} slice {idx})"):
                # Dynamically mosaic the tiles belonging to this specific datetime/dimension
                gdal.BuildVRT(vrt_path, tile_paths)
                
            # Lazy load the VRT blueprint
            vrt_da = rioxarray.open_rasterio(vrt_path, chunks=True)
            
            # Inject auxiliary coordinates (time, model, scenario, etc.)
            _, structured_da = self.parse_metadata(row, vrt_da)
            da_list.append(structured_da)
            
        # 2. Delegate to the unified alignment, warping, and materialization engine
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
            logger=logger
        )

    def _process_chunked_timeseries(
        self, group_df: pd.DataFrame, var_name: str, level: str, ctx: Dict[str, Any], 
        target_bounds: Tuple[float, float, float, float], target_grid_key: str, 
        target_crs: Any, target_res: float, raw_fetched_data: Dict[str, xr.DataArray], 
        max_workers: int, tracker: Any, logger: logging.Logger
    ) -> Optional[str]:
        """
        Processes standalone 2D slice data by injecting dimensional metadata and 
        routing the structured arrays to the universal warping engine.

        This method acts as a bridge between raw, unstructured 2D rasters (like 
        individual GeoTIFFs downloaded from an API) and the N-dimensional data cube. 
        It iterates through the execution manifest, pulls the corresponding raw array 
        from memory, assigns contextual coordinates (like time or depth), and 
        delegates the heavy lifting of spatial alignment to the warping engine.

        Parameters
        ----------
        group_df : pd.DataFrame
            The execution manifest partition containing metadata for the specific 
            variable and level being processed.
        var_name : str
            The physical variable name (e.g., 'precipitation', 'temperature').
        level : str
            The pipeline processing level or categorical family.
        ctx : Dict[str, Any]
            Global execution context containing paths, configurations, and state variables.
        target_bounds : Tuple[float, float, float, float]
            The spatial bounding box (min_x, min_y, max_x, max_y) for the final output.
        target_grid_key : str
            The identifier for the target spatial grid to which the data will be warped.
        target_crs : Any
            The target Coordinate Reference System (typically a pyproj.CRS or EPSG string).
        target_res : float
            The spatial resolution in the target CRS units.
        raw_fetched_data : Dict[str, xr.DataArray]
            An in-memory dictionary mapping VSI paths to their raw rioxarray DataArrays.
        max_workers : int
            The maximum number of concurrent threads permitted for spatial processing.
        tracker : Any
            The performance/resource tracker instance for logging CPU and memory strain.
        logger : logging.Logger
            The execution logger.

        Returns
        -------
        Optional[str]
            The absolute filepath to the finalized, materialized Zarr or NetCDF4 cube, 
            or None if processing fails or returns empty data.
        """
        da_list = []
        
        # --- 1. Inject auxiliary coordinates into raw memory-loaded DataArrays ---
        for _, row in group_df.iterrows():
            # The execution plan groups paths into lists; for chunked timeseries, 
            # we extract the singular path string representing this specific slice.
            path_key = row['vsi_path'][0] if isinstance(row['vsi_path'], list) else row['vsi_path']
            
            # Retrieve the pre-loaded 2D array from the fetcher's dictionary
            raw_da = raw_fetched_data.get(path_key)
            
            if raw_da is not None:
                # Promote the 2D array to an N-dimensional array using the row's metadata
                _, structured_da = self.parse_metadata(row, raw_da)
                da_list.append(structured_da)
                
        # Fast exit if upstream fetchers failed or returned empty subsets
        if not da_list:
            log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
            return None
            
        # --- 2. Delegate to the unified alignment, warping, and materialization engine ---
        # The list of temporally/structurally labeled 2D arrays is now handed off 
        # to be physically aligned to the target projection and spatial grid.
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
            logger=logger
        )
    
def _align_warp_and_export(
        self, 
        da_list: List[xr.DataArray], 
        var_name: str, 
        level: str, 
        ctx: Dict[str, Any], 
        target_bounds: Tuple[float, float, float, float], 
        target_grid_key: str, 
        target_crs: Any, 
        target_res: float, 
        max_workers: int, 
        tracker: Any, 
        logger: logging.Logger
    ) -> Optional[str]:
        """
        Universal spatial alignment, parallel warping, and cube materialization engine.

        This engine standardizes a list of disparate 2D DataArrays by reprojecting 
        them to a common grid, assembling them into an N-dimensional structure, 
        clipping them to the target bounds, and serializing them to disk. It employs 
        dynamic routing to choose between native GDAL multi-threading or Python-level 
        ThreadPool executors based on hardware constraints and slice sizes to optimize 
        throughput without causing out-of-memory (OOM) crashes.

        Parameters
        ----------
        da_list : List[xr.DataArray]
            A list of structural 2D DataArrays, each containing Z-axis coordinates (e.g., time).
        var_name : str
            The physical variable name (e.g., 'temperature').
        level : str
            The pipeline processing level or categorical family.
        ctx : Dict[str, Any]
            Global execution context containing paths, configurations, and state variables.
        target_bounds : Tuple[float, float, float, float]
            The spatial bounding box (min_x, min_y, max_x, max_y) for the final output.
        target_grid_key : str
            The identifier for the target spatial grid.
        target_crs : Any
            The target Coordinate Reference System.
        target_res : float
            The spatial resolution in the target CRS units.
        max_workers : int
            The maximum number of concurrent threads permitted for spatial processing.
        tracker : Any
            The performance/resource tracker instance.
        logger : logging.Logger
            The execution logger.

        Returns
        -------
        Optional[str]
            The absolute filepath to the finalized Zarr or NetCDF4 cube, or None if failed.
        """
        if not da_list:
            log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
            return None

        base_dir, cube_name, dataset_name = ctx["base_dir"], ctx["cube_name"], ctx["dataset_name"]
        export_format = ctx["export_format"]
        cfg = ctx.get("execution_config", {})

        # --- 1. Z-Axis Alignment & Coordinate Extraction ---
        # Ensure all incoming slices share exactly the same base X/Y coordinate instances
        # to prevent xarray from attempting dense outer joins during concatenation.
        base_x, base_y = da_list[0].coords['x'], da_list[0].coords['y']
        snapped_list = [d.assign_coords(x=base_x, y=base_y) for d in da_list]
        
        # Identify the primary Z-dimension (anything that isn't spatial)
        z_dim = [dim for dim in snapped_list[0].dims if dim not in ['x', 'y']][0]
        
        # Sort chronologically or sequentially by the Z-dimension to ensure data integrity
        snapped_list.sort(key=lambda da: da.coords[z_dim].values[0])
        
        # Extract all primary and auxiliary coordinates to reapply post-warp
        z_vals = np.array([da[z_dim].values for da in snapped_list]).flatten()
        full_meta_coords = {z_dim: z_vals}
        
        for k in snapped_list[0].coords.keys():
            if k not in ['x', 'y', 'spatial_ref', z_dim]:
                meta_vector = np.array([da[k].values for da in snapped_list]).flatten()
                full_meta_coords[k] = (z_dim, meta_vector)
        
        # Determine mathematical resampling rule (e.g., 'nearest' for classes, 'bilinear' for continuous)
        rule = self.get_resample_rule(var_name)
        
        # Establish a temporary disk cache for warped 2D slices to keep RAM usage low
        cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
        os.makedirs(cache_dir, exist_ok=True)

        # --- 2. Parallel 2D Warping Setup ---
        slice_size_mb = snapped_list[0].nbytes / (1024 * 1024)

        def _warp_worker(da_2d: xr.DataArray, index: int, gdal_threads: str) -> str:
            """Internal worker for serializing, sanitizing, and reprojecting a single slice."""
            # VSI_CACHE="FALSE" prevents thread locks during concurrent I/O
            with rasterio.Env(GDAL_NUM_THREADS=gdal_threads, VSI_CACHE="FALSE", GTIFF_SRS_SOURCE="EPSG"):
                try:
                    # Sanitize NoData Values to prevent GDAL casting errors (e.g., float NaNs in Int8 arrays)
                    nodata_val = da_2d.rio.nodata
                    if nodata_val is not None:
                        if np.issubdtype(da_2d.dtype, np.integer):
                            limits = np.iinfo(da_2d.dtype)
                            if not (limits.min <= nodata_val <= limits.max):
                                da_2d = da_2d.astype('float32')
                        da_2d.rio.write_nodata(nodata_val, inplace=True)

                    # Remove conflicting NetCDF FillValue attributes before GDAL processing
                    if '_FillValue' in da_2d.attrs:
                        del da_2d.attrs['_FillValue']

                    # Ensure standard grid configurations and spatial references exist
                    da_2d = self._sanitize_spatial_geometry(da_2d, default_crs="EPSG:4326", logger=None)
                    out_filepath = os.path.join(cache_dir, f"slice_{index:04d}.tif")
                    
                    # Delegate the physical pixel transformation to the core warping method
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

        # --- 3. Warp Execution Routing ---
        # Heuristic engine: determines the safest concurrent strategy based on file sizes and limits.
        if slice_size_mb < cfg.get("large_slice_threshold_mb", 50.0):
            # If there are very few files, threading overhead exceeds processing time. Run sequentially.
            if num_slices < cfg.get("min_slices_for_parallel", 16):
                with tracker.track_strain(f"Main-Thread Sequential Warp ({var_name})"):
                    for i, da in enumerate(snapped_list):
                        result = _warp_worker(da, i, "1")
                        if result != "": warped_tif_paths.append(result)
            else:
                # Modest-sized files benefit significantly from Python-level ThreadPools
                calc_workers = int(math.sqrt(num_slices))
                dynamic_workers = max(1, min(calc_workers, max_workers, cfg.get("max_safe_threadpool_workers", 8)))
                with tracker.track_strain(f"Parallel ThreadPool Warp ({var_name})"):
                    with ThreadPoolExecutor(max_workers=dynamic_workers) as executor:
                        futures = [executor.submit(_warp_worker, da, i, "1") for i, da in enumerate(snapped_list)]
                        for f in futures:
                            res = f.result()
                            if res != "": warped_tif_paths.append(res)
        else:
            # For massive individual grids (e.g., >50MB per slice), Python threading risks OOM.
            # Instead, run slices sequentially in Python, but allow GDAL to use C++ multi-threading underneath.
            num_gdal_threads = str(min(os.cpu_count() or 4, max_workers))
            with tracker.track_strain(f"C++ Multi-Core Native Warp ({var_name})"):
                for i, da in enumerate(snapped_list):
                    result_path = _warp_worker(da, i, num_gdal_threads)
                    if result_path != "": warped_tif_paths.append(result_path)

        # --- 4. 3D Reassembly & Dask Materialization ---
        aligned_slices = []
        block_size = cfg.get("gdal_warp_block_size", 2048)
        
        # Load the physical warped files back into memory as chunked dask arrays
        for tif_path in warped_tif_paths:
            warped_2d = rioxarray.open_rasterio(
                tif_path, 
                chunks={'x': block_size, 'y': block_size}
            )
            # Purge rioxarray's default 1-sized band dimension
            if 'band' in warped_2d.dims:
                warped_2d = warped_2d.squeeze('band', drop=True)
            aligned_slices.append(warped_2d)

        if not aligned_slices:
            log_execution(logger, f"No warped slices produced for {var_name}.", logging.ERROR)
            return None

        # Assemble the N-dimensional cube
        combined_da = xr.concat(aligned_slices, dim=z_dim)
        combined_da.name = var_name
        
        # Re-attach the temporal and auxiliary coordinates that were stripped during GDAL warping
        combined_da = combined_da.assign_coords(full_meta_coords)
        
        # Clip off any nodata padding created during reprojection bounding-box expansion
        combined_da = combined_da.rio.clip_box(*target_bounds)
        
        # Force Dask to compute the graph and load the final array entirely into RAM
        with tracker.track_strain(f"Dask Materialization ({var_name})"):
            combined_da = combined_da.load()
            
        # Ensure CF-compliant spatial attributes are baked into the DataArray
        combined_da = combined_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        combined_da = combined_da.rio.write_crs(target_crs, convention=Convention.CF)
        combined_da = combined_da.rio.write_transform(convention=Convention.CF)
        combined_da.attrs['crs'] = str(target_crs)
        combined_da.attrs['res'] = float(target_res)

        # --- 5. Export ---
        level_dir = os.path.join(base_dir, cube_name, dataset_name, level)
        os.makedirs(level_dir, exist_ok=True)
        export_ds = combined_da.to_dataset(name=var_name)

        # NetCDF/Zarr engines cannot serialize generic Python 'object' types; force string casting
        for v_key in export_ds.variables:
            if export_ds[v_key].dtype == 'O':
                export_ds[v_key] = export_ds[v_key].astype(str)

        # Serialize the materialized dataset to the user's requested format
        if export_format == 'zarr':
            var_cache_path = os.path.join(level_dir, f"{var_name}.zarr")
            export_ds.to_zarr(var_cache_path, mode='w')
        else:
            var_cache_path = os.path.join(level_dir, f"{var_name}.nc")
            export_ds.to_netcdf(var_cache_path, engine="netcdf4", format="NETCDF4")
        
        # Memory cleanup: purge xarray backend file locks and force garbage collection
        xr.backends.file_manager.FILE_CACHE.clear()
        del snapped_list, combined_da, export_ds
        gc.collect()

        # Brief pause to ensure OS-level file handlers have released the warp cache
        time.sleep(cfg.get("warp_cache_cleanup_delay_sec", 30))
        
        return var_cache_path