import os
import logging
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

from bmc.cube.spatiotemporal import spatiotemporal_cube
from bmc.utils.logger import log_execution


class chelsa_cube(spatiotemporal_cube):
    """
    Child class dedicated to the single-pass generation of high-resolution 
    CHELSA V2.1 spatiotemporal data cubes.

    Inherits from `spatiotemporal_cube`. This processing engine utilizes external 
    generalized spatial and I/O utilities to dynamically calculate safe fetching 
    envelopes, stream raw Cloud Optimized GeoTIFF matrices, structure complex 
    MultiIndex CMIP6 metadata, and utilize the parent class affine reprojection 
    engine to guarantee perfect grid alignment across all environmental data sources.

    Attributes
    ----------
    resample_rules : dict
        A strict internal routing dictionary mapping physical environmental variables 
        (e.g., 'tas', 'pr', 'bio12') to their physically appropriate GDAL resampling 
        algorithms (e.g., continuous gradients via `bilinear` vs. volume conservation 
        via `average`).

    Methods
    -------
    resolve_target_grid(spatial_cfg, logger)
        Translates user spatial requests into physically safe native target grids.
    generate_execution_plan(recipe, logger, catalog_path_override=None)
        Filters the CHELSA GeoParquet inventory to generate a precise execution queue.
    get_resample_rule(variable_name)
        Maps variables to strict GDAL resampling rules (e.g., volume-conserving averages).
    parse_metadata(row, da)
        Translates flat tabular metadata into rich, multi-dimensional Xarray coordinates.
    apply_multi_index(level, dataset)
        Compiles independent dimensional coordinates into a vendor-specific MultiIndex.

    Notes
    -----
    When extracting raw spatial bounds, this class explicitly pads the spatial envelope 
    with a 5-pixel buffer. This physical "scaffolding" prevents edge-starvation 
    anomalies during the GDAL affine reprojection phase. Once the matrix is snapped 
    to the master grid, the excess border is mathematically clipped away to perfectly 
    match the user's requested bounding box.
    
    Furthermore, because standard GeoTIFF I/O operations inherently flatten and destroy 
    complex Z-axis Xarray dimensions, this engine utilizes a "Metadata Catch and Release" 
    architecture. Complex coordinate frameworks (such as CMIP6 ensembles or time series) 
    are stripped immediately prior to disk-cached warping, and surgically re-injected 
    upon reload to preserve data integrity for downstream MultiIndex compilation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the CHELSA spatiotemporal data cube engine rules.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed directly to the parent class.
        """
        # Pass any external keyword arguments up to the spatiotemporal_cube parent class
        super().__init__(**kwargs)

        # Define a strict dictionary mapping physical variables to GDAL interpolation rules
        self.resample_rules = {
            'tas': 'bilinear',       # Air temperature requires smooth bilinear gradients
            'tasmax': 'bilinear',    # Maximum temperature requires smooth gradients
            'tasmin': 'bilinear',    # Minimum temperature requires smooth gradients
            'sfcWind': 'bilinear',   # Wind speed requires smooth gradients
            'vpd': 'bilinear',       # Vapor pressure deficit requires smooth gradients
            'pr': 'average',         # Precipitation flux requires volume-conserving averages
            'rsds': 'average',       # Solar radiation flux requires volume-conserving averages
            'clt': 'average',        # Cloud cover percentage requires cell-area averages
            'hurs': 'average',       # Relative humidity percentage requires cell-area averages
            'pet': 'average'         # Potential evapotranspiration flux requires averages
        }

    def resolve_target_grid(
        self, 
        spatial_cfg: Dict[str, Any], 
        logger: Optional[logging.Logger] = None
    ) -> str:
        """
        Intersects target spatial configurations to enforce minimum atmospheric resolutions.
        Enforces 1km for metric grids (EEA, GEA) and 15sec for WGS84 degree grids.
        """
        log_execution(logger, "Resolving target grid base for CHELSA data cube generation...", logging.INFO)
        
        # Extract user requests
        raw_grid = str(spatial_cfg.get('target_grid', 'EEA'))
        raw_res = str(spatial_cfg.get('target_resolution', '1km'))
        
        # 0. Direct Registry Check
        if raw_grid in self.GRID_REGISTRY:
            log_execution(logger, f"Successfully mapped CHELSA processing to exact registry grid '{raw_grid}'.", logging.INFO)
            return raw_grid
            
        grid_base_upper = raw_grid.upper()

        # 1. Standardize the base grid names and set safe defaults
        if grid_base_upper == "EEA":
            clean_base = "EEA"
            default_res = "1km"
        elif grid_base_upper in ["GEA", "GLOBAL_EQUALAREA", "GLOBAL_EQUAL_AREA"]:
            clean_base = "Global_EqualArea"
            default_res = "1km"
        elif grid_base_upper in ["WGS84", "GLOBAL_WGS84"]:
            clean_base = "Global_WGS84"
            default_res = "30sec"  # CHELSA native standard
        else:
            clean_base = raw_grid 
            default_res = "1km"

        # 2. Enforce Minimum Resolutions based on Coordinate System Families
        if clean_base in ["EEA", "Global_EqualArea"]:
            # --- Metric Grids ---
            res_in_meters = self._parse_res_to_meters(raw_res)
            
            if res_in_meters < 1000.0:
                log_execution(
                    logger, 
                    f"Requested metric resolution '{raw_res}' is too fine. Clamping to '{default_res}'.", 
                    logging.WARNING
                )
                final_res = default_res
            else:
                final_res = raw_res

        elif clean_base == "Global_WGS84":
            # --- Angular Grids ---
            too_fine_wgs84 = ["3sec", "7_5sec"]
            allowed_wgs84 = ["15sec", "30sec", "5min"]
            
            if raw_res in too_fine_wgs84:
                log_execution(
                    logger, 
                    f"Requested angular resolution '{raw_res}' is too fine. Clamping to minimum allowable '15sec'.", 
                    logging.WARNING
                )
                final_res = "15sec"
            elif raw_res in allowed_wgs84:
                final_res = raw_res
            else:
                log_execution(
                    logger, 
                    f"Unrecognized WGS84 resolution '{raw_res}'. Falling back to safe default: '{default_res}'.", 
                    logging.WARNING
                )
                final_res = default_res
        else:
            # Fallback for unrecognized grid bases
            final_res = raw_res

        # 3. Construct and validate the final master key
        resolved_grid = f"{clean_base}_{final_res}"

        if resolved_grid not in self.GRID_REGISTRY:
            fallback = f"{clean_base}_{default_res}"
            if fallback not in self.GRID_REGISTRY:
                fallback = "EEA_1km" # Absolute safety net
            
            log_execution(
                logger, 
                f"Constructed grid '{resolved_grid}' does not exist in registry. Falling back to safe default: '{fallback}'.", 
                logging.WARNING
            )
            resolved_grid = fallback
        else:
            log_execution(logger, f"Successfully mapped CHELSA processing to registry grid '{resolved_grid}'.", logging.INFO)
        
        return resolved_grid

    def generate_execution_plan(
        self,
        recipe: Dict[str, Any], 
        logger: logging.Logger,
        *,  # Forces subsequent arguments to be keyword-only to protect the parent signature
        catalog_path_override: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Translates the YAML execution recipe into Pandas boolean queries against 
        the CHELSA GeoParquet database to generate an asset processing queue.

        Parameters
        ----------
        recipe : dict
            The fully loaded and parsed YAML configuration recipe dictating the 
            spatiotemporal bounds and requested variables.
        logger : logging.Logger
            The logger instance used to record catalog intersection progress, 
            connection status, and the final asset queue count.
        catalog_path_override : str, optional
            Direct path to the GeoParquet STAC catalog. If provided, this overrides 
            both the path specified in the YAML recipe and the system default.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the queued assets to be processed, with columns 
            for level, variable, date, month, time_range, ensemble, scenario, and vsi_path. 
            Returns an empty DataFrame if no assets match or if the catalog is missing.
        """
        log_execution(logger, "Intersecting configuration recipe against CHELSA inventory...", logging.INFO)
        
        # =================================================================
        # 1. HIERARCHY OF PATH RESOLUTION
        # Resolves the catalog path by checking the override argument first, 
        # falling back to the YAML recipe, and finally using a system default.
        # =================================================================
        recipe_path = recipe.get('sources', {}).get('chelsa', {}).get('catalog_path')
        default_path = "../../../meta/chelsa_gp_stac/chelsa_master.parquet"
        
        final_parquet_path = catalog_path_override or recipe_path or default_path
        
        if not os.path.exists(final_parquet_path):
            log_execution(logger, f"FATAL: GeoParquet database not found at '{final_parquet_path}'", logging.ERROR)
            return pd.DataFrame()

        log_execution(logger, f"Reading GeoParquet catalog from: {final_parquet_path}", logging.INFO)
        
        try:
            # Load the entire GeoParquet database into memory in milliseconds
            stac_gdf = gpd.read_parquet(final_parquet_path)
        except Exception as e:
            log_execution(logger, f"Failed to load GeoParquet Catalog: {e}", logging.ERROR)
            return pd.DataFrame()

        # =================================================================
        # 2. CONFIGURATION EXTRACTION & TEMPORAL BOUNDS
        # =================================================================
        chelsa_cfg = recipe.get('sources', {}).get('chelsa', {})
        if not chelsa_cfg.get('enabled', False):
            log_execution(logger, "CHELSA processing explicitly disabled in recipe.", logging.INFO)
            return pd.DataFrame()

        # Parse temporal bounds for continuous data filtering
        temp_cfg = recipe.get('temporal', {})
        start_date = pd.to_datetime(f"{temp_cfg.get('start_year')}-{temp_cfg.get('start_month', 1):02d}-01", utc=True)
        end_date = pd.to_datetime(f"{temp_cfg.get('end_year')}-{temp_cfg.get('end_month', 12):02d}-28", utc=True)

        execution_queue = []
        levels_cfg = chelsa_cfg.get('levels', {})

        # =================================================================
        # 3. TRANSLATE YAML RULES TO PANDAS BOOLEAN MASKS
        # =================================================================
        for level, settings in levels_cfg.items():
            # Skip levels explicitly disabled in the YAML
            if not settings.get('include', False):
                continue

            # Extract only the variables marked as 'true'
            active_vars = [var for var, is_active in settings.get('variables', {}).items() if is_active]
            if not active_vars:
                continue

            # Create the base boolean mask targeting the specific level and variables
            base_mask = (stac_gdf['chelsa:level'] == level) & (stac_gdf['chelsa:variable'].isin(active_vars))

            # --- Rule Set A: Continuous Time Series ---
            if level in ['daily', 'monthly', 'annual']:
                # Ensure the STAC datetime column is formatted for precise comparison
                stac_gdf['datetime'] = pd.to_datetime(stac_gdf['datetime'], utc=True)
                
                # Filter assets strictly within the requested temporal window
                time_mask = (stac_gdf['datetime'] >= start_date) & (stac_gdf['datetime'] <= end_date)
                chunk = stac_gdf[base_mask & time_mask]

            # --- Rule Set B: Static Projections (CMIP6 / Baseline) ---
            elif level in ['climatologies', 'bioclim']:
                active_ranges = [tr for tr, state in settings.get('time_ranges', {}).items() if state]
                active_ensembles = [ens for ens, state in settings.get('ensembles', {}).items() if state]
                active_scenarios = [scen for scen, state in settings.get('scenarios', {}).items() if state]

                # Initialize an all-True mask to iteratively apply constraints
                static_mask = pd.Series(True, index=stac_gdf.index)
                
                if active_ranges:
                    static_mask &= stac_gdf['chelsa:time_range'].isin(active_ranges)
                
                # Apply exact match filtering for climate models and SSP scenarios
                if active_ensembles:
                    static_mask &= stac_gdf['cmip6:model'].isin(active_ensembles)
                    
                if active_scenarios:
                    static_mask &= stac_gdf['cmip6:scenario'].isin(active_scenarios)
                    
                chunk = stac_gdf[base_mask & static_mask]

            # =================================================================
            # 4. EXTRACT ASSET URLS AND BUILD THE WORKER QUEUE
            # =================================================================
            for _, row in chunk.iterrows():
                var_name = row.get("chelsa:variable")
                
                # Extract the actual URL from the nested STAC asset dictionary
                assets_dict = row.get("assets", {})
                vsi_path = assets_dict.get(var_name, {}).get("href") if isinstance(assets_dict, dict) else None
                
                # Failsafe: skip if the file path is missing from the database
                if not vsi_path:
                    continue

                # Standardize the row layout for the downstream execution pipeline
                execution_queue.append({
                    'level': level,
                    'variable': var_name,
                    'date': row.get('datetime'),
                    'month': row.get("chelsa:month"),
                    'time_range': row.get("chelsa:time_range"),
                    'ensemble': row.get("cmip6:model", "historical"),
                    'scenario': row.get("cmip6:scenario", "historical"),
                    'vsi_path': vsi_path
                })

        # =================================================================
        # 5. FINALIZE AND RETURN
        # =================================================================
        final_df = pd.DataFrame(execution_queue)
        
        if final_df.empty:
            log_execution(logger, "GeoParquet Query returned 0 assets. Check YAML constraints.", logging.WARNING)
        else:
            log_execution(logger, f"GeoParquet Intersection successful: queued {len(final_df)} target assets.", logging.INFO)
      
        return final_df

    def get_resample_rule(self, variable_name: str) -> str:
        """
        Determines the appropriate GDAL spatial resampling algorithm for a variable.

        Different physical and ecological variables require strictly different 
        mathematical algorithms during affine reprojection. Child classes must map 
        variable strings to valid GDAL resampling strings to prevent data corruption 
        (e.g., ensuring categorical land cover classes are never interpolated).

        Parameters
        ----------
        variable_name : str
            The name of the physical variable or product type currently being warped 
            (e.g., 'pr', 'Corine Land Cover 2018').

        Returns
        -------
        str
            The GDAL resampling string. Valid options include 'nearest', 'bilinear', 
            'cubic', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3', 'sum', 'rms'.
        """
        var_str = str(variable_name)
        
        # Check the class dictionary for a registered rule, defaulting to bilinear
        rule = self.resample_rules.get(var_str, 'bilinear')
        
        # Implement safety override: composite precipitation models MUST conserve volume
        if 'bio' in var_str.lower():
            if var_str.lower() in ['bio12', 'bio13', 'bio14']:
                rule = 'average'
            else:
                rule = 'bilinear'
                
        return rule

    def parse_metadata(self, row: pd.Series, da: xr.DataArray) -> Tuple[str, xr.DataArray]:
        """
        Extracts dataset-specific metadata and injects it as dimensional coordinates.

        Raw Cloud-Optimized GeoTIFFs downloaded from remote storage are fundamentally 2D 
        and lack complex contextual metadata. This method expands the 2D spatial array 
        into a 3D or 4D array by parsing the metadata from the execution plan row (e.g., 
        extracting a year from a filename, or parsing CMIP6 ensembles) and assigning 
        those values to a new Z-axis dimension (like 'time' or 'projection').

        Parameters
        ----------
        row : pd.Series
            A single record from the execution plan DataFrame containing the contextual 
            metadata associated with the fetched array.
        da : xarray.DataArray
            The raw, mathematically flattened 2D spatial array returned by the 
            parallel network fetcher.

        Returns
        -------
        tuple
            A 2-element tuple containing:
            - ``level`` (str): The processing family grouping string.
            - ``da`` (xarray.DataArray): The structurally augmented 3D/4D DataArray 
              ready for Z-axis concatenation.
        """
        level = row['level']
        vsi_path = row['vsi_path']
        da.name = row['variable']
        
        # Attempt to get a clean filename, fallback to splitting the S3 URL
        filename = row.get('filename') if pd.notna(row.get('filename')) else vsi_path.split('/')[-1]
        file_parts = filename.replace('.tif', '').split('_')

        # 1. Climatologies (Decadal averages)
        if level == 'climatologies':
            ens = str(row['ensemble']) if pd.notna(row.get('ensemble')) else 'historical'
            scen = str(row['scenario']) if pd.notna(row.get('scenario')) else 'historical'
            tr = str(row['time_range']) if pd.notna(row.get('time_range')) else '1981-2010'
            
            m = 1
            for part in file_parts:
                if part.isdigit() and len(part) == 2 and 1 <= int(part) <= 12:
                    m = int(part)
                    break
            
            coord_val = f"{ens}_{scen}_{tr}_{m:02d}"
            dim_name = "projection"
            
        # 2. Bioclimatic variables
        elif level == 'bioclim':
            ens = str(row['ensemble']) if pd.notna(row.get('ensemble')) else 'historical'
            scen = str(row['scenario']) if pd.notna(row.get('scenario')) else 'historical'
            tr = str(row['time_range']) if pd.notna(row.get('time_range')) else 'unknown'
            
            coord_val = f"{ens}_{scen}_{tr}"
            dim_name = "projection"

        # 3. Continuous timeseries (Daily / Monthly)
        elif pd.notna(row.get('date')) and level in ['daily', 'monthly', 'annual']:
            # Strip the timezone to ensure compatibility with Zarr/NetCDF serialization
            raw_date = row['date']
            if hasattr(raw_date, 'tz') and raw_date.tz is not None:
                coord_val = raw_date.tz_localize(None)
            else:
                coord_val = raw_date                
            dim_name = "time"
        
        # 4. Fallback for older datasets
        elif pd.notna(row.get('year')):
            coord_val = pd.to_datetime(f"{int(row['year'])}-01-01")
            dim_name = "time"
            
        else:
            coord_val = row.name 
            dim_name = "index"

        # Expand the flat 2D array into 3D
        da = da.expand_dims(dim_name).assign_coords({dim_name: [coord_val]})

        # Inject the independent MultiIndex components as parallel coordinates
        if dim_name == "projection":
            if level == 'climatologies':
                da = da.assign_coords({
                    "ensemble": (dim_name, [ens]),
                    "scenario": (dim_name, [scen]),
                    "time_range": (dim_name, [tr]),
                    "month": (dim_name, [m])
                })
            elif level == 'bioclim':
                da = da.assign_coords({
                    "ensemble": (dim_name, [ens]),
                    "scenario": (dim_name, [scen]),
                    "time_range": (dim_name, [tr])
                })

        # Strip rioxarray's default 1-size band coordinate
        if "band" in da.coords:
            da = da.squeeze("band").drop_vars("band", errors="ignore")

        return level, da

    def apply_multi_index(self, level: str, dataset: xr.Dataset) -> xr.Dataset:
        """
        Compiles independent dimensional coordinates into a vendor-specific MultiIndex.

        After the parent engine completes spatial warping and restores basic Z-axis 
        coordinates, some highly complex datasets (such as multidimensional climate 
        scenarios) require bundling individual string coordinates into a formalized 
        Pandas/Xarray MultiIndex. Child classes implement this to finalize the 
        Dataset structure.

        Parameters
        ----------
        level : str
            The processing family grouping string (e.g., 'climatologies', 'bioclim') 
            which dictates whether a MultiIndex is necessary.
        dataset : xarray.Dataset
            The fully warped, spatially aligned, and basic-coordinate-restored Dataset.

        Returns
        -------
        xarray.Dataset
            The finalized Dataset, optionally containing a `.set_index()` MultiIndex 
            on the Z-axis (e.g., grouping ensemble, scenario, and time_range).
        """
        # Only process datasets that were assigned the 'projection' dimension 
        # in the parse_metadata phase
        if "projection" in dataset.dims:
            
            if level == 'climatologies':
                # Bundle the 4 independent metadata arrays into a true MultiIndex
                dataset = dataset.set_index(projection=["ensemble", "scenario", "time_range", "month"])
                
            elif level == 'bioclim':
                # Bundle the 3 independent metadata arrays into a true MultiIndex
                dataset = dataset.set_index(projection=["ensemble", "scenario", "time_range"])
                
        return dataset