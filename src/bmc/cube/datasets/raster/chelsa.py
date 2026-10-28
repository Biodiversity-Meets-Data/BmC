import os
from pathlib import Path
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

import bmc
from bmc.cube.spatiotemporal.raster import *
from bmc.utils.logger import log_execution


class chelsaCube(rasterCube):
    """
    Processing engine dedicated to the generation of high-resolution 
    CHELSA V2.1 spatiotemporal data cubes.

    Inherits from `rasterCube`. This class is tailored to handle the specific 
    metadata and physical characteristics of the CHELSA climatological dataset. 
    It dynamically resolves safe spatial grids, extracts complex auxiliary 
    climate dimensions (such as GCM ensembles and SSP scenarios), and enforces 
    variable-specific GDAL resampling algorithms to maintain scientific validity 
    during affine reprojections.

    Attributes
    ----------
    dataset_name : str
        The dynamically derived lowercase name of the dataset (e.g., 'chelsa').
    catalog_path : str
        The absolute file path to the local GeoParquet catalog tracking the 
        CHELSA asset inventory.
    temporally_static_levels : list of str
        A list of processing levels (e.g., 'climatologies', 'bioclim') that represent 
        static temporal aggregations rather than dynamic timeseries slices.
    resample_rules : dict
        A strict routing dictionary mapping physical environmental variables 
        (e.g., 'tas', 'pr') to their physically appropriate GDAL resampling 
        algorithms (e.g., continuous gradients via `bilinear` vs. volume 
        conservation via `average`).

    Methods
    -------
    resolve_target_grid(spatial_cfg, logger=None)
        Translates user spatial requests into physically safe native target grids, 
        clamping excessively fine resolutions.
    get_resample_rule(variable_name)
        Maps variable strings to strict GDAL resampling rules to prevent data 
        corruption during warping.
    get_auxiliary_dimensions(recipe)
        Parses the user recipe to deduplicate and aggregate active CHELSA-specific 
        dimensions (time ranges, models, scenarios).

    Notes
    -----
    The CHELSA dataset requires careful handling of resampling algorithms. 
    Temperature-based continuous variables (e.g., 'tas', 'tasmax') use 'bilinear' 
    interpolation, whereas flux and volume-based variables (e.g., 'pr', 'bio12') 
    must use 'average' to conserve total mass over a given spatial grid area.
    """

    def __init__(self, **kwargs):
        # Zero-argument safe initialization for the orchestrator
        super().__init__(**kwargs)
        
       # 1. Dynamically derive the dataset name
        self.dataset_name = self.__class__.__name__.replace('Cube', '').lower()
        
        # 2. Anchor to the repo root using pathlib
        # __file__     = BmC/src/bmc/__init__.py
        # parents[0]   = BmC/src/bmc
        # parents[1]   = BmC/src
        # parents[2]   = BmC (The actual repository root)
        repo_root = Path(bmc.__file__).parents[2]
        
        self.catalog = os.path.join(
            repo_root, 
            "meta", 
            self.dataset_name, 
            f"{self.dataset_name}_catalog.parquet"
        )

        # --- Declare Temporal Physics ---
        self.temporally_static_levels = ['climatologies', 'bioclim']

        # 3. Define Resample Rules natively
        self.resample_rules = {
            'tas': 'bilinear',
            'tasmax': 'bilinear',
            'tasmin': 'bilinear',
            'sfcWind': 'bilinear',
            'vpd': 'bilinear',
            'pr': 'average',
            'rsds': 'average',
            'clt': 'average',
            'hurs': 'average',
            'pet': 'average'
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
        
        raw_grid = str(spatial_cfg.get('target_grid', 'EEA'))
        raw_res = str(spatial_cfg.get('target_resolution', '1km'))
        registry = getattr(self, 'GRID_REGISTRY', {})
        
        # 0. Early Exit: Exact match provided natively
        if raw_grid in registry:
            log_execution(logger, f"Successfully mapped to exact registry grid '{raw_grid}'.", logging.INFO)
            return raw_grid
            
        # 1. Base Grid Standardization via Dictionary Mapping
        grid_aliases = {
            "EEA": ("EEA", "1km"),
            "GEA": ("Global_EqualArea", "1km"),
            "GLOBAL_EQUALAREA": ("Global_EqualArea", "1km"),
            "GLOBAL_EQUAL_AREA": ("Global_EqualArea", "1km"),
            "WGS84": ("Global_WGS84", "30sec"),
            "GLOBAL_WGS84": ("Global_WGS84", "30sec")
        }
        
        clean_base, default_res = grid_aliases.get(raw_grid.upper(), (raw_grid, "1km"))
        final_res = raw_res

        # 2. Enforce Minimum Resolutions based on Coordinate System Families
        if clean_base in ["EEA", "Global_EqualArea"]:
            if self._parse_res_to_meters(raw_res) < 1000.0:
                log_execution(logger, f"Metric resolution '{raw_res}' too fine. Clamping to '{default_res}'.", logging.WARNING)
                final_res = default_res

        elif clean_base == "Global_WGS84":
            if raw_res in ["3sec", "7_5sec"]:
                log_execution(logger, f"Angular resolution '{raw_res}' too fine. Clamping to '15sec'.", logging.WARNING)
                final_res = "15sec"
            elif raw_res not in ["15sec", "30sec", "5min"]:
                log_execution(logger, f"Unrecognized WGS84 resolution '{raw_res}'. Falling back to '{default_res}'.", logging.WARNING)
                final_res = default_res

        # 3. Final Validation & Fallback Construction
        resolved_grid = f"{clean_base}_{final_res}"

        if resolved_grid not in registry:
            fallback = f"{clean_base}_{default_res}"
            resolved_grid = fallback if fallback in registry else "EEA_1km"
            log_execution(logger, f"Constructed grid missing from registry. Falling back to safe default: '{resolved_grid}'.", logging.WARNING)
        else:
            log_execution(logger, f"Successfully mapped CHELSA processing to registry grid '{resolved_grid}'.", logging.INFO)
        
        return resolved_grid

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
            (e.g., 'pr').

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

    def get_auxiliary_dimensions(self, recipe: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Parses the execution recipe to identify and aggregate active CHELSA-specific 
        auxiliary dimensions across structural levels.

        This method scans the configuration for climate projection parameters—specifically 
        temporal reference periods ('time_ranges'), climate models ('ensembles'), and 
        Shared Socioeconomic Pathways ('scenarios'). It deduplicates these requests 
        across the 'climatologies' and 'bioclim' levels and maps them to the standard 
        column names expected by the catalog manifest.

        Parameters
        ----------
        recipe : Dict[str, Any]
            The parsed YAML configuration dictionary containing execution directives. 
            Expected to have a 'sources' key that maps to this specific dataset's 
            configuration.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping the standardized auxiliary dimension names to a 
            deduplicated list of requested values.
            Possible keys include:
            - 'time_range' : List of temporal strings (e.g., ['1981-2010', '2041-2070']).
            - 'model' : List of climate model identifiers (e.g., ['MPI-ESM1-2-HR']).
            - 'scenario' : List of SSP scenarios (e.g., ['ssp126', 'ssp585']).

        Notes
        -----
        - Uses sets internally during extraction to ensure that if multiple levels 
        (e.g., both 'climatologies' and 'bioclim') request the same model or 
        scenario, it is only represented once in the final filter array.
        - Note the nomenclature mapping: the recipe key 'ensembles' is mapped to 
        the catalog column 'model'.
        """
        aux_dims = {}
        
        # Isolate the configuration block specific to the CHELSA dataset.
        chelsa_cfg = recipe.get('sources', {}).get(self.dataset_name, {})
        levels_cfg = chelsa_cfg.get('levels', {})
        
        # Initialize sets to automatically deduplicate requested dimensions if they 
        # are declared multiple times across different levels.
        active_ranges = set()
        active_ensembles = set()
        active_scenarios = set()
        
        # Iterate exclusively through the structural levels that utilize climate 
        # projection and historical baseline dimensions.
        for level in ['climatologies', 'bioclim']:
            settings = levels_cfg.get(level, {})
            
            # Only extract dimensions if the level itself is flagged for processing
            if settings.get('include', False):
                # Extract dictionary keys where the boolean value is True
                active_ranges.update(
                    [tr for tr, state in settings.get('time_ranges', {}).items() if state]
                )
                active_ensembles.update(
                    [ens for ens, state in settings.get('ensembles', {}).items() if state]
                )
                active_scenarios.update(
                    [scen for scen, state in settings.get('scenarios', {}).items() if state]
                )
                
        # Map the extracted YAML configuration keys to the exact column names 
        # present in the underlying GeoParquet catalog.
        if active_ranges:
            aux_dims['time_range'] = list(active_ranges)
            
        if active_ensembles:
            # 'ensembles' in the user YAML corresponds to the 'model' column in the catalog
            aux_dims['model'] = list(active_ensembles)
            
        if active_scenarios:
            aux_dims['scenario'] = list(active_scenarios)
            
        return aux_dims