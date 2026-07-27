import pandas as pd
import xarray as xr
import logging
from typing import Dict, Any, Tuple, Optional
import os
from hda import Client


from bmc.cube.spatiotemporal.legacy.spatiotemporal import spatiotemporal_cube
from bmc.utils.logger import log_execution

class wekeo_cube(spatiotemporal_cube):
    """
    Orchestration layer for processing WEkEO (Copernicus Land Monitoring Service) Earth Observation data.

    This class acts as a specialized data cube generator inheriting from `spatiotemporal_cube`. 
    It bridges the gap between abstract user configurations defined in a YAML recipe and the 
    physical operations required to fetch, align, and reconstruct multi-dimensional data cubes 
    from the WEkEO STAC catalog. It introduces a critical metadata "smuggling" technique 
    to preserve complex categorical attributes through out-of-core GDAL C++ warping processes.

    Attributes
    ----------
    dataset_resolution_strategy : str or None
        The resolution override strategy (e.g., "highest", "lowest", or a specific metric 
        like "10m") extracted from the WEkEO YAML configuration during plan generation. 
        Utilized during grid resolution to dynamically select the correct target grid.

    Methods
    -------
    generate_execution_plan(recipe, logger, catalog_path_override=None)
        Intersects YAML spatiotemporal constraints against a GeoParquet STAC catalog 
        to generate a standardized asset fetching queue.
    resolve_target_grid(spatial_cfg, logger)
        Resolves dynamic target resolutions against the global configuration and strictly 
        validates them against the core engine's grid registry.
    parse_metadata(row, da)
        Injects temporal axes, sanitizes redundant spatial dimensions, and structurally encodes 
        categorical metadata into the variable's string name before GDAL processing.
    get_resample_rule(variable_name)
        Decodes the structurally smuggled variable name to determine the mathematically safe 
        GDAL resampling algorithm.
    apply_multi_index(level, dataset)
        Unpacks the smuggled string metadata post-warp and expands the flattened variables 
        back into a formalized N-dimensional data cube.

    Notes
    -----
    The underlying GDAL C++ bindings inherently strip Python `xarray` coordinates during 
    spatial affine reprojection. To circumvent this, the `wekeo_cube` encodes metadata 
    (such as `aggregation` and `layer_class`) directly into the `DataArray` name string 
    using a double-underscore delimiter (`__`) in `parse_metadata`. 
    This string safely passes through GDAL, and is subsequently split apart by 
    `apply_multi_index` to restore the correct dimensions.
    """
    def __init__(self):
        super().__init__()

    def _initialize_wekeo_client(self):
        """Authenticates the WEkEO client using the .env file."""
        
        print("Authenticating with WEkEO...")
        wekeo_user = os.getenv("HDA_USER")
        wekeo_pass = os.getenv("HDA_PASSWORD")
        
        if not wekeo_user or not wekeo_pass:
            raise ValueError("Authentication failed: HDA_USER or HDA_PASSWORD missing from .env file!")
            
        self.wekeo_client = Client(user=wekeo_user, password=wekeo_pass)

    def generate_execution_plan(
        self,
        recipe: Dict[str, Any], 
        logger: logging.Logger,
        *,  # Forces subsequent arguments to be keyword-only to protect the parent signature
        catalog_path_override: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Translates the YAML execution recipe into a standardized data fetching queue by 
        intersecting user constraints against the WEkEO GeoParquet database.

        This method acts as the primary translation layer between the abstract user request 
        (e.g., "Give me Grassland data for 2018") and the physical data lake. It loads 
        the SpatioTemporal Asset Catalog (STAC) stored as a GeoParquet file, parses the 
        requested variables and temporal bounds from the recipe, and applies highly 
        optimized Pandas boolean masking to filter out irrelevant files. The result is 
        a flat, standardized execution queue ready for out-of-core parallel downloading 
        and spatial warping.

        Parameters
        ----------
        recipe : Dict[str, Any]
            The fully loaded and parsed YAML configuration dictionary. Must contain the 
            'sources' and 'temporal' schema blocks to dictate the spatiotemporal bounds 
            and requested specific variables.
        logger : logging.Logger
            The active logger instance used to record catalog intersection progress, 
            connection/loading status, and the final extracted asset queue count.
        catalog_path_override : Optional[str], default=None
            A direct absolute or relative path to the GeoParquet STAC catalog. If provided, 
            this explicitly overrides both the path specified in the YAML recipe and the 
            hardcoded system default.

        Returns
        -------
        pd.DataFrame
            A standardized execution queue containing the exact physical assets to be processed. 
            The DataFrame contains the following enforced columns:
            - `level` (str): The processing family or collection group (e.g., 'GRA').
            - `variable` (str): The core scientific variable name (e.g., 'Grassland').
            - `year` (int): The temporal year of the specific data slice.
            - `aggregation` (str): The mathematical representation (e.g., 'coverage', 'max').
            - `layer_class` (str): The categorical class label, if applicable.
            - `vsi_path` (str): The direct file path or Virtual File System (VSI) URL to the GeoTIFF.
            
            Returns an empty DataFrame if no assets match the constraints or if the 
            catalog file is missing/corrupted.

        Notes
        -----
        The method executes in five distinct phases:
        1. Hierarchy of path resolution (Override -> Recipe -> Default).
        2. Configuration extraction and temporal bounding.
        3. Vectorized Pandas boolean mask generation.
        4. Asset URL extraction and queue standardization.
        5. Finalization and DataFrame construction.
        """
        import geopandas as gpd
        
        log_execution(logger, "Intersecting configuration recipe against WEkEO inventory...", logging.INFO)
        
        # =================================================================
        # 1. HIERARCHY OF PATH RESOLUTION
        # Resolves the catalog path by checking the override argument first, 
        # falling back to the YAML recipe, and finally using a system default.
        # =================================================================
        
        # Attempt to safely extract the catalog path defined within the user's YAML file
        recipe_path = recipe.get('sources', {}).get('wekeo', {}).get('catalog_path')
        
        # Hardcoded fallback path assuming standard repository execution directory
        default_path = "../../../meta/wekeo_lake/wekeo_lake_catalog.parquet"
        
        # Resolve the final path using Python's truthy 'or' evaluation (first non-None/non-empty wins)
        final_parquet_path = catalog_path_override or recipe_path or default_path
        
        # Validate physical existence of the file before attempting to load it into memory
        if not os.path.exists(final_parquet_path):
            log_execution(logger, f"FATAL: GeoParquet database not found at '{final_parquet_path}'", logging.ERROR)
            return pd.DataFrame()

        log_execution(logger, f"Reading GeoParquet catalog from: {final_parquet_path}", logging.INFO)
        
        try:
            # Load the entire GeoParquet database into a GeoDataFrame. 
            # Parquet is highly optimized, so reading this takes milliseconds even for millions of rows.
            stac_gdf = gpd.read_parquet(final_parquet_path)
        except Exception as e:
            # Catch generic reading errors (e.g., corrupted file, missing compression libraries)
            log_execution(logger, f"Failed to load GeoParquet Catalog: {e}", logging.ERROR)
            return pd.DataFrame()

        # =================================================================
        # 2. CONFIGURATION EXTRACTION & TEMPORAL BOUNDS
        # Parses the structural rules defined by the user in the YAML.
        # =================================================================
        
        wekeo_cfg = recipe.get('sources', {}).get('wekeo', {})
        
        # Fast-fail if the WEkEO section is explicitly turned off in the config
        if not wekeo_cfg.get('enabled', False):
            log_execution(logger, "WEkEO processing explicitly disabled in recipe.", logging.INFO)
            return pd.DataFrame()

        # Capture the dataset-specific resolution override (e.g., "highest", "10m", or None)
        # This is saved as an instance attribute so the spatial_engine can access it later during warping
        self.dataset_resolution_strategy = wekeo_cfg.get("dataset_resolution")
        
        # Extract the global temporal bounding window. 
        # Provide extremely wide defaults (2000-2050) to prevent crashing if the user omits them.
        temp_cfg = recipe.get('temporal', {})
        start_year = temp_cfg.get('start_year', 2000)
        end_year = temp_cfg.get('end_year', 2050)

        execution_queue = []
        categories = wekeo_cfg.get('categories', {})

        # =================================================================
        # 3. TRANSLATE YAML RULES TO PANDAS BOOLEAN MASKS
        # Converts nested dictionary logic into flat lists for fast vector math.
        # =================================================================
        
        active_collections = []
        active_variables = []
        
        # Iterate through the nested config to build flat lists of approved items
        for cat_name, cat_data in categories.items():
            # Check if the broad collection category (e.g., "GRA", "TCF") is enabled
            if cat_data.get("include", False):
                active_collections.append(cat_name)
                
                # Check which specific datasets within that category are enabled
                datasets = cat_data.get("datasets", {})
                for var_name, var_data in datasets.items():
                    if var_data.get("include", False):
                        active_variables.append(var_name)

        # Fast-fail if the user left WEkEO 'enabled' but turned off all actual data layers
        if not active_collections or not active_variables:
            log_execution(logger, "No active WEkEO collections or variables found in recipe.", logging.WARNING)
            return pd.DataFrame()

        # Construct the vectorized boolean mask. 
        # This acts as a sieve, dropping any rows that do not strictly meet all four criteria.
        base_mask = (
            (stac_gdf['collection'].isin(active_collections)) &  # Must belong to an active collection
            (stac_gdf['variable'].isin(active_variables)) &      # Must be an actively requested variable
            (stac_gdf['year'] >= start_year) &                   # Must occur on or after the start year
            (stac_gdf['year'] <= end_year)                       # Must occur on or before the end year
        )
        
        # Apply the mask to slice the GeoDataFrame
        chunk = stac_gdf[base_mask]

        # =================================================================
        # 4. EXTRACT ASSET URLS AND BUILD THE WORKER QUEUE
        # Maps the raw database columns to the parent engine's expected format.
        # =================================================================
        
        for _, row in chunk.iterrows():
            # Extract the actual system or network path to the physical GeoTIFF file
            vsi_path = row.get("href")
            
            # Failsafe: Skip the row if the file path is structurally missing or null
            if not pd.notna(vsi_path) or not vsi_path:
                continue

            # Append a clean, standardized dictionary to the execution list.
            # This formatting perfectly aligns with what `spatiotemporal_cube.process_cube` expects.
            execution_queue.append({
                'level': row.get("collection"),     # Level determines the final output cube grouping (e.g., "GRA" cube)
                'variable': row.get("variable"),    # Variable defines the specific scientific layer
                'year': row.get("year"),            # Year becomes the Z-axis coordinate
                'aggregation': row.get("aggregation"), # Will be mapped into the dimension structure
                'layer_class': row.get("layer_class"), # Will be mapped into the dimension structure
                'vsi_path': vsi_path                # The critical path passed to GDAL/rioxarray for reading
            })

        # =================================================================
        # 5. FINALIZE AND RETURN
        # =================================================================
        
        # Convert the standardized list of dictionaries back into a highly readable Pandas DataFrame
        final_df = pd.DataFrame(execution_queue)
        
        # Provide final logging telemetry to help the user debug their YAML constraints if 0 assets matched
        if final_df.empty:
            log_execution(logger, "GeoParquet Query returned 0 assets. Check YAML constraints.", logging.WARNING)
        else:
            log_execution(logger, f"GeoParquet Intersection successful: queued {len(final_df)} target assets.", logging.INFO)
      
        return final_df

    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: logging.Logger) -> str:
        """
        Dynamically resolves and validates the target spatial grid configuration.

        This method acts as the spatial gatekeeper for the WEkEO cube pipeline. It merges 
        the global spatial configuration (e.g., a default target of "EEA_100m") with any 
        dataset-specific resolution overrides defined in the YAML recipe (e.g., requesting 
        the "highest" available resolution). It parses dynamic strategies, maps them against 
        the physically supported frameworks in the core `GRID_REGISTRY`, and delegates 
        final structural validation to the parent spatial engine.

        Parameters
        ----------
        spatial_cfg : Dict[str, Any]
            The 'spatial' configuration dictionary extracted from the execution recipe. 
            Expected to contain fundamental constraints such as 'target_grid' (e.g., "EEA") 
            and 'target_resolution' (e.g., "1km").
        logger : logging.Logger
            The active logger instance used to record resolution overrides, fallback 
            warnings, and the finalized grid key.

        Returns
        -------
        str
            The strictly validated dictionary key (e.g., 'EEA_10km', 'Global_WGS84_30sec') 
            that mathematically aligns with a predefined footprint in the `GRID_REGISTRY`.

        Raises
        ------
        ValueError
            Inherited from `resolve_grid_registry_key`. Raised if the final concatenated 
            grid key does not explicitly exist in the `GRID_REGISTRY`.
        """
        # =================================================================
        # 1. EXTRACT GLOBAL SPATIAL DEFAULTS
        # Pull the baseline geographic target from the YAML recipe.
        # =================================================================
        
        # Base grid network from global spatial config (e.g., "EEA" or "Global_EqualArea")
        target_grid = spatial_cfg.get("target_grid")
        
        # Default spatial resolution from global spatial config (e.g., "100m")
        final_resolution = spatial_cfg.get("target_resolution")
        
        # =================================================================
        # 2. APPLY DATASET-SPECIFIC OVERRIDES
        # Check if the user specified a WEkEO-specific resolution strategy 
        # (captured earlier during the `generate_execution_plan` phase).
        # =================================================================
        
        if hasattr(self, 'dataset_resolution_strategy') and self.dataset_resolution_strategy:
            strategy = self.dataset_resolution_strategy
            
            # --- Dynamic Strategies ("highest" or "lowest") ---
            # These require scanning the registry to find out what distances are actually supported.
            if strategy in ['highest', 'lowest']:
                
                # Construct the search prefix to isolate grids belonging to the requested family.
                # e.g., If target_grid is "EEA", the prefix is "EEA_"
                prefix = f"{target_grid}_"
                
                # List comprehension to dynamically extract all valid resolution strings 
                # (e.g., "100m", "250m", "10km") that belong to the chosen grid family.
                allowed_resolutions = [
                    key.replace(prefix, "") 
                    for key in self.GRID_REGISTRY.keys()
                    if key.startswith(prefix)
                ]
                
                if allowed_resolutions:
                    # Delegate the mathematical distance comparison (e.g., 100m vs 10km) 
                    # to the parent spatial engine to figure out which string is the "highest".
                    final_resolution = self._resolve_query_resolution(
                        strategy=strategy, 
                        available_res=allowed_resolutions, 
                        logger=logger
                    )
                else:
                    # Failsafe: If the registry is misconfigured or missing the grid family entirely.
                    log_execution(
                        logger, 
                        f"Could not find any predefined resolutions for grid family '{target_grid}' in GRID_REGISTRY. "
                        f"Falling back to global default.", 
                        logging.WARNING
                    )
            
            # --- Hardcoded Strategies (e.g., "10km") ---
            else:
                # If the user explicitly defined an override string instead of a dynamic strategy.
                final_resolution = strategy
                
            log_execution(logger, f"Applied WEkEO resolution override: {strategy} -> {final_resolution}", logging.INFO)

        # =================================================================
        # 3. STRICT STRUCTURAL VALIDATION
        # =================================================================
        
        # Combine the base grid and the newly resolved resolution, then pass them 
        # to the parent spatial engine. This guarantees the key exists and will 
        # instantly halt execution if an unsupported grid is requested.
        grid_key = self.resolve_grid_registry_key(target_grid, final_resolution, logger)
        
        log_execution(logger, f"WEkEO target grid safely finalized as: {grid_key}", logging.INFO)
        
        return grid_key
    
    def parse_metadata(self, row: pd.Series, da: xr.DataArray) -> Tuple[str, xr.DataArray]:
        """
        Extracts dataset-specific metadata and structurally prepares the array for spatial processing.

        Raw GeoTIFFs fetched from remote storage are inherently flat spatial grids 
        lacking multidimensional awareness. This method translates those flat grids 
         into temporal slices by injecting a 'time' dimension. Crucially, it employs a 
        string-encoding strategy to "smuggle" critical metadata (like aggregations and classes) 
        through the parent engine's GDAL C++ warping functions, which natively strip complex 
        Python coordinates.

        Parameters
        ----------
        row : pd.Series
            A single record from the standardized execution queue containing the contextual 
            STAC metadata (e.g., year, aggregation, layer_class) for this specific asset.
        da : xr.DataArray
            The raw, lazily loaded spatial array returned by the out-of-core fetcher.

        Returns
        -------
        Tuple[str, xr.DataArray]
            A two-element tuple containing:
            - The `level` grouping string (e.g., "GRA") used by the parent engine to route 
              arrays into specific data cubes.
            - The structurally enriched `DataArray`, renamed with the smuggled metadata and 
              expanded along the correct temporal axis.
        """
        
        # =================================================================
        # 1. TEMPORAL METADATA EXTRACTION
        # Parse the contextual metadata required for multidimensional stacking.
        # =================================================================
        
        # Extract the parent processing family (e.g., 'TCF', 'GRA')
        level = row['level']
        
        # Isolate the observation year and lock it to a standard January 1st Pandas datetime.
        # This guarantees standard xarray time-series alignment later.
        year = int(row['year'])
        time_coord = pd.to_datetime(f"{year}-01-01")
        
        # =================================================================
        # 2. DIMENSIONAL SANITIZATION & EXPANSION
        # Clean the array structure to prevent out-of-core streaming failures.
        # =================================================================
        
        # Safely remove the redundant 'band' dimension automatically added by rioxarray.
        # Standard GeoTIFF formats crash if attempting to write 4-dimensional data 
        # (e.g., time, band, y, x). Squeezing 'band' forces the array to a flat 2D (y, x).
        if 'band' in da.dims:
            da = da.squeeze('band').drop_vars('band', errors='ignore')
            
        # Expand the now 2D array into a true 3D temporal slice (time, y, x)
        structured_da = da.expand_dims(time=[time_coord])
        
        # =================================================================
        # 3. METADATA SMUGGLING (THE GDAL WORKAROUND)
        # Encode complex categorical properties directly into the variable name.
        # =================================================================
        
        # Safely extract the aggregation logic and the specific categorical class.
        # Defaulting to 'unknown' prevents the pipeline from crashing on unexpected STAC schemas.
        aggregation = row.get('aggregation', 'unknown')
        layer_class = row.get('layer_class', 'unknown')
        
        # Create a highly unique routing string utilizing a double-underscore delimiter.
        # Example output: "Forest_Type__coverage__Broadleaved_forest"
        # 
        # WHY: The parent `spatiotemporal_cube` passes this array to GDAL for spatial snapping.
        # GDAL does not understand xarray's custom scalar coordinates and will drop them. 
        # By baking the metadata directly into the array's core name, the metadata safely 
        # survives the C++ transformation. The `apply_multi_index` function will later crack 
        # this string open and restore the 'aggregation' and 'layer_class' dimensions.
        structured_da.name = f"{row['variable']}__{aggregation}__{layer_class}"
        
        # Return the target cube grouping key and the safely encoded multidimensional array
        return level, structured_da

    def get_resample_rule(self, variable_name: str) -> str:
        """
        Determines the appropriate GDAL spatial resampling algorithm by decoding 
        the structurally smuggled routing name.

        During out-of-core affine reprojection, different physical and ecological 
        variables require strictly different mathematical algorithms to prevent 
        data corruption. Because standard GDAL C++ bindings strip away 
        complex xarray coordinates, the `parse_metadata` method previously encoded 
        the requested aggregation strategy directly into the variable's string name 
        (e.g., "Forest_Type__coverage__Broadleaved_forest"). 
        
        This method acts as the decoder ring, splitting that string, extracting the 
        user's requested aggregation, and safely translating it into an exact GDAL 
        resampler keyword compatible with the parent spatial engine.

        Parameters
        ----------
        variable_name : str
            The heavily encoded variable name containing the scientific variable, 
            the aggregation method, and the class label separated by double underscores 
            (e.g., "Tree_Cover_Density__max__unknown").

        Returns
        -------
        str
            The strictly validated GDAL resampling string used for out-of-core warping. 
            Returns "average" for fractional coverages, the direct statistical string 
            for continuous variables (e.g., "max"), or safely falls back to "nearest" 
            if the string is improperly formatted.
        """
        
        # =================================================================
        # 1. DECODE THE SMUGGLED METADATA
        # =================================================================
        
        # Parse the string format using the delimiter: "Variable__Aggregation__Class"
        # Example: "Grassland__coverage__Grassland" becomes ['Grassland', 'coverage', 'Grassland']
        parts = variable_name.split("__")
        
        # Safely check if the string actually contained the smuggled metadata.
        # If it only has 1 part, it's a raw variable name and bypasses the specific logic.
        if len(parts) >= 2:
            
            # Isolate the aggregation rule (the second element) and standardize to lowercase
            agg_rule = parts[1].lower()
            
            # =================================================================
            # 2. DISCRETE FRACTIONAL COVERAGE ROUTING
            # =================================================================
            
            # When downsampling discrete categorical maps (like Land Cover) into fractions,
            # we must conserve the physical spatial volume (the amount of area covered).
            # The mathematical arithmetic mean ("average") perfectly calculates the 
            # continuous percentage of a binary mask within a new, larger pixel footprint.
            if agg_rule == 'coverage':
                return "average"
                
            # =================================================================
            # 3. CONTINUOUS STATISTICAL ROUTING
            # =================================================================
            
            # For continuous variables (like Tree Cover Density), the requested statistical 
            # aggregations natively match their GDAL C++ integer constant string names 
            # defined in `spatial_engine.GDAL_RESAMPLERS`.
            if agg_rule in ['max', 'min', 'average', 'rms']:
                return agg_rule
                
        # =================================================================
        # 4. SAFETY FALLBACK
        # =================================================================
        
        # If the variable lacks a defined rule, default to "nearest" (Nearest Neighbor).
        # This is the safest global fallback because it strictly prevents interpolation, 
        # ensuring categorical classes are not accidentally corrupted by floating-point math.
        return "nearest"

    def apply_multi_index(self, level: str, dataset: xr.Dataset) -> xr.Dataset:
        """
        Executes post-warp dimensionality reduction and reconstructs the finalized multi-dimensional data cube.

        Because the core spatial engine relies on GDAL C++ bindings for out-of-core affine 
        reprojection, any complex multidimensional structures (like specific class categories 
        or aggregation strategies) are flattened into independent 2D/3D variables. 
        This method acts as the "unpacker" to the `parse_metadata` method's "packer". 
        It reads the smuggled metadata baked into the temporary variable names, promotes 
        those strings back into actual dimensional coordinates, and concatenates the separated 
        arrays back into a unified, mathematically clean N-dimensional `xarray.Dataset`.

        Parameters
        ----------
        level : str
            The processing family grouping string (e.g., 'GRA', 'TCF') being actively processed. 
            While not strictly used for WEkEO index building, it fulfills the parent class's 
            abstract method signature requirements.
        dataset : xr.Dataset
            The fully spatially warped, but dimensionally flat, Dataset returned by the 
            spatial engine. Variables will have structurally encoded names 
            (e.g., 'Tree_Cover_Density__max__unknown').

        Returns
        -------
        xr.Dataset
            The finalized, scientifically structured multidimensional Dataset. 
            Variables are restored to their base names (e.g., 'Tree_Cover_Density') and 
            expanded along strict categorical dimensions (e.g., 'layer_class' or 'aggregation').
        """
        
        # =================================================================
        # 1. INITIALIZATION
        # =================================================================
        
        # Dictionary to hold the grouped 3D/4D arrays before final concatenation.
        # Structure will be: {'Base_Variable': [DataArray_1, DataArray_2, ...]}
        grouped_arrays = {}
        
        # =================================================================
        # 2. DECODING & DIMENSIONAL EXPANSION
        # Iterate over all the flattened, disjointed variables returned from GDAL.
        # =================================================================
        
        for temp_var_name, da in dataset.data_vars.items():
            
            # Crack open the smuggled string using the double-underscore delimiter.
            # Example: "Forest_Type__coverage__Broadleaved_forest"
            parts = str(temp_var_name).split("__")
            
            # Safely extract the components. 
            # If the string was perfectly formatted, it yields exactly 3 parts. 
            # If not, it falls back to 'unknown' to prevent index errors.
            base_var = parts[0]
            agg_val = parts[1] if len(parts) > 1 else 'unknown'
            class_val = parts[2] if len(parts) > 2 else 'unknown'
            
            # --- Expand the DataArray along a new Z-axis dimension ---
            # If the data is a discrete fractional mask (routed via 'coverage'),
            # the meaningful dimensional axis is the target class (e.g., Broadleaved vs Coniferous).
            if agg_val == 'coverage':
                # The 3D array (time, y, x) becomes 4D (layer_class, time, y, x)
                da = da.expand_dims(layer_class=[class_val])
                
            # If the data is a continuous variable (routed via 'max', 'average', etc.),
            # the meaningful dimensional axis is the statistical aggregation method.
            else:
                # The 3D array (time, y, x) becomes 4D (aggregation, time, y, x)
                da = da.expand_dims(aggregation=[agg_val])
            
            # =================================================================
            # 3. VARIABLE GROUPING
            # =================================================================
            
            # Strip the routing string off the data array so its internal name 
            # reverts to the clean scientific variable (e.g., "Forest_Type").
            da.name = base_var
            
            # Drop the newly expanded array into the grouping dictionary under its base name.
            if base_var not in grouped_arrays:
                grouped_arrays[base_var] = []
            grouped_arrays[base_var].append(da)
            
        # =================================================================
        # 4. DATASET RECONSTRUCTION (CONCATENATION)
        # =================================================================
        
        # Initialize the fresh, empty Dataset that will hold the final cube.
        final_dataset = xr.Dataset()
        
        for base_var, arrays in grouped_arrays.items():
            
            # Dynamically determine which dimension to glue the arrays together along.
            # It checks the first array in the grouped list to see if it was expanded 
            # along 'layer_class' (discrete) or 'aggregation' (continuous).
            concat_dim = 'layer_class' if 'layer_class' in arrays[0].dims else 'aggregation'
            
            # Use xarray's high-performance concat tool to smash the list of 4D slices 
            # together into a unified multidimensional variable, and assign it to the Dataset.
            # Example: Three arrays of size (1, time, y, x) become one array of size (3, time, y, x).
            final_dataset[base_var] = xr.concat(arrays, dim=concat_dim)
            
        return final_dataset