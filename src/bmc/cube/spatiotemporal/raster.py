import os
import gc
import time
import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rioxarray.enum import Convention
import rasterio
from rasterio.warp import transform_bounds

from bmc.utils.logger import log_execution, ResourceProfiler
from bmc.utils.spatial import build_envelope_from_file
from bmc.utils.io import parallel_fetch_rasters

from bmc.engine.raster import raster_engine

class raster_cube(raster_engine, ABC):
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
    _GDAL_RESAMPLERS : dict
        Internal mapping of human-readable resampler string keys to their 
        corresponding GDAL C++ integer constants.
    _RESAMPLER_DECODER : dict
        Internal reverse-mapping of GDAL integer constants back to human-readable 
        resampler strings, utilized primarily for clear execution logging.

    Methods
    -------
    generate_cube_recipe(config_path, logger=None)
        Parses a YAML configuration file and generates a standardized execution recipe.
    resolve_grid_registry_key(target_grid, target_resolution, logger=None)
        Dynamically constructs and validates the master grid key from user configuration.
    gather_tifs_from_zips(source_directory, target_directory, logger=None)
        Iterates through zip archives and extracts .tif files into a flattened directory.
    cleanup_raw_storage(recipe, logger=None)
        Safely purges the raw data directory based on the user configuration.
    build_virtual_mosaic(input_folder, output_vrt_path, logger=None)
        Creates a lightweight Virtual Raster (VRT) blueprint from multiple GeoTIFF tiles.
    process_virtual_mosaic(vrt_path, strategy, grid_name, output_dir_or_file, logger=None, **kwargs)
        Routes a VRT blueprint to either standard reprojection or categorical fractional coverage.
    affine_reproject(input_data, output_filepath, grid_name, resample_keyword='bilinear', compress_mode='lzw', memory_limit_bytes=4096, logger=None)
        Performs out-of-core spatial reprojection and snapping to a strictly defined master grid.
    calculate_fractional_coverages(ds, grid_name, output_dir, class_values=None, class_mapping=None, file_prefix='fractional', logger=None)
        Calculates fractional coverage of categorical classes and exports single-band COGs.
    export_to_cog(ds, output_filepath, compress_mode='deflate', logger=None)
        Exports a lazy xarray object to disk as a Cloud Optimized GeoTIFF (COG).
    da_layer_constructor(data_layer_func, param)
        General layer constructor that fetches all slices for a layer sequentially.
    da_layer_constructor_concurrent(layer_func, param, max_workers=4)
        General layer constructor that fetches all slices for a layer concurrently.
    da_concat(data_arrays, dim_name, coordinates)
        Combines a stack of 2D data arrays into a 3D data array along a new dimension.    
    
    Notes
    -----
    The choice of GDAL resampling algorithm during affine reprojection is critical 
    for spatial accuracy. Below is a guide to the supported resamplers and their 
    optimal ecological use cases:

    Categorical & Discrete Data (e.g., Land Cover, Forest Type):
    * nearestNeighbour : Assigns the value of the single closest source pixel, 
      preserving original discrete values without interpolation.
    * mode : Assigns the most frequently occurring value among contributing pixels. 
      The mathematical standard for downsampling categorical data.

    Continuous Data Smoothing (e.g., Elevation, Temperature):
    * bilinear : Distance-weighted average of the 4 closest source pixels.
    * cubic : Distance-weighted cubic polynomial curve over the 16 nearest pixels.
    * cubicSpline : 2D B-spline mathematical function over the 16 nearest pixels. 
      Heavily smooths data and prevents "overshoot" (Runge's phenomenon). The 
      gold standard for realistic, continuous gradients.
    * lanczos : Complex windowed sinc function over the 36 nearest source pixels. 
      Preserves high-frequency details and sharpness.

    Continuous Data Statistical Aggregation (Downsampling):
    * average : Arithmetic mean of all valid intersecting source pixels.
    * max / min : Highest or lowest data value within the target footprint.
    * med : Exact middle value (50th percentile) of contributing pixels.
    * q1 / q3 : First (25th) or third (75th) quartile of contributing pixels.
    * sum : Addition of all valid intersecting source pixels.
    * rms : Root Mean Square (quadratic mean). Emphasizes higher magnitude values.
    """  
    def __init__(self):
        pass

    #################################
    # Interface & helper functions  #
    #################################

 
    def _parse_res_to_meters(self, res_str: str) -> float:
        """
        Converts a resolution string (e.g., '10m', '1km') into a float in meters.
        
        This helper is required for mathematical comparisons between different 
        available raw data resolutions.
        """
        res_str = res_str.lower().strip()
        if 'km' in res_str:
            return float(res_str.replace('km', '')) * 1000
        elif 'm' in res_str:
            return float(res_str.replace('m', ''))
        else:
            # Fallback for unexpected formats (like arc-seconds)
            # You can extend this logic as needed for Global_WGS84 grids
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
            or a specific value like '20m'.
        available_res : list of str
            The unique resolution strings found in the inventory for a specific product.
        """
        if strategy not in ['highest', 'lowest'] and strategy in available_res:
            return strategy
            
        # Create a mapping: {meters: 'string_name'}
        res_map = {self._parse_res_to_meters(r): r for r in available_res}
        
        if not res_map:
            return "UNKNOWN"

        if strategy == 'highest':
            # Smallest distance = Highest resolution
            return res_map[min(res_map.keys())]
        elif strategy == 'lowest':
            # Largest distance = Lowest resolution
            return res_map[max(res_map.keys())]
        else:
            # If a specific res was requested but isn't available, 
            # we default to 'highest' and log a warning.
            best_guess = res_map[min(res_map.keys())]
            log_execution(
                logger, 
                f"Requested query resolution '{strategy}' not found. Falling back to highest available: {best_guess}", 
                logging.WARNING
            )
            return best_guess

    #################################
    #     General Cube pipeline     #
    #################################

    @abstractmethod
    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: logging.Logger) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key.

        This abstract method must interpret the ``spatial_cfg`` block from the YAML 
        recipe and map it to a physically safe, mathematically supported grid framework 
        present in the base class's ``GRID_REGISTRY``. Child classes must handle 
        vendor-specific logic here, such as safely degrading sub-kilometer CHELSA requests 
        to native 1km atmospheric scales, or allowing high-resolution WEkEO requests to pass.

        Parameters
        ----------
        spatial_cfg : dict
            The 'spatial' configuration dictionary extracted from the execution recipe. 
            Expected to contain keys such as 'target_grid' and 'target_resolution'.
        logger : logging.Logger
            The logger instance used to record validation steps or fallback warnings 
            if a requested resolution is actively overridden by the child class.

        Returns
        -------
        str
            The validated dictionary key (e.g., 'EEA_1km', 'Global_EqualArea_100m') 
            required to query the parent class's ``GRID_REGISTRY``.
        """
        pass

    @abstractmethod
    def generate_execution_plan(self, recipe: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
        """
        Translates the execution recipe into a standardized data fetching queue.

        This method bridges the gap between the user's abstract configuration (e.g., 
        "Give me temperature for 2000-2010") and the vendor's actual data lake. Child 
        classes must implement the logic to query their specific catalogs (whether via 
        a local CSV inventory, directory crawling, or a live STAC API) and filter assets 
        based on spatial, temporal, and categorical constraints.

        Parameters
        ----------
        recipe : dict
            The fully loaded and parsed YAML configuration recipe dictating the 
            spatiotemporal bounds and requested variables.
        logger : logging.Logger
            The logger instance used to record catalog intersection progress, connection 
            status (if using STAC), and the final asset queue count.

        Returns
        -------
        pd.DataFrame
            A standardized execution queue. To be compatible with the parent processing 
            engine, the DataFrame must contain at minimum the following columns:
            - ``level`` (str): The processing family (e.g., 'daily', 'TCF').
            - ``variable`` (str): The specific scientific variable (e.g., 'tas', 'Tree Cover Density').
            - ``vsi_path`` (str): The direct /vsicurl/ or local path to the raw GeoTIFF.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def process_cube(
        self, 
        recipe: Dict[str, Any], 
        max_workers: int = 10,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        The universal out-of-core spatial processing loop.
        
        This method executes the spatiotemporal pipeline by fetching raw spatial 
        data, harmonizing spatial bounds, and leveraging a hybrid multithreaded worker 
        pool to project individual 2D slices. 
        
        It utilizes a "Disk-Spilling Data Lake" architecture: finished multidimensional 
        variables are immediately written to a nested directory structure on disk 
        and destroyed from RAM. The function returns a lightweight path catalog 
        instead of a monolithic memory object.

        Parameters
        ----------
        recipe : dict
            The parsed YAML configuration dictating the bounds, grids, and targets.
        max_workers : int, optional
            The maximum number of parallel threads to use for both network fetching 
            and GDAL reprojection. Defaults to 10.
        logger : logging.Logger, optional
            Logger instance for execution tracking. If None, one is automatically created.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A catalog dictionary mapping processing levels to their explicitly 
            generated files on disk (e.g., {'bioclim': {'bio01': './path/to/bio01.nc'}}).
        """
        
        # ==========================================
        # 1. Initialization & Spatial Framework
        # ==========================================
        paths_cfg = recipe.get('paths', {})
        base_dir = paths_cfg.get('base_dir') or recipe.get('base_dir', './cubing_output/')
        cube_name = recipe.get('cube_name', 'bmd_default_cube')
        
        dataset_name = recipe.get('dataset_name', self.__class__.__name__.lower().replace('_cube', ''))
        export_format = recipe.get('export_as', {}).get('format', 'netcdf').lower()
        
        if logger is None:
            log_dir = os.path.join(base_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir, 'spatiotemporal_cube_generation.log')
            logger = self._setup_pipeline_logger(logger_name="spatiotemporal_cube", log_filepath=log_filepath)
            self.logger = logger

        tracker = ResourceProfiler(log_dir=os.path.join(base_dir, 'logs'))
        log_execution(logger, "You are using the correct version", logging.INFO)
        log_execution(logger, "\n=== Initiating Out-of-Core Data Cube Generation ===", logging.INFO)
        
        execution_plan = self.generate_execution_plan(recipe, logger)
        if execution_plan.empty:
            log_execution(logger, "Terminating pipeline: no candidate asset catalog generated.", logging.WARNING)
            return {}

        spatial_cfg = recipe.get('spatial', {})
        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)
        grid_info = self.GRID_REGISTRY[target_grid_key]
        target_crs = grid_info["crs"]
        target_res = grid_info["resolution"]
        
        bbox_cfg = spatial_cfg.get('bbox', {})
        wgs84_bounds = (
            min(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            min(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0)),
            max(bbox_cfg.get('long_min', 0), bbox_cfg.get('long_max', 0)),
            max(bbox_cfg.get('lat_min', 0), bbox_cfg.get('lat_max', 0))
        )
        target_bounds = transform_bounds("EPSG:4326", target_crs, *wgs84_bounds)

        log_execution(logger, f"\n--- Target Spatial Framework Initialized ---", logging.INFO)
        log_execution(logger, f"  Master Grid : {target_grid_key} ({target_crs})", logging.INFO)
        log_execution(logger, f"  Resolution  : {target_res} (Native CRS Units)", logging.INFO)
        log_execution(logger, f"  WGS84 Bounds: [MinLon: {wgs84_bounds[0]:.4f}, MinLat: {wgs84_bounds[1]:.4f}, MaxLon: {wgs84_bounds[2]:.4f}, MaxLat: {wgs84_bounds[3]:.4f}]", logging.INFO)
        log_execution(logger, f"  Proj Bounds : [MinX: {target_bounds[0]:.2f}, MinY: {target_bounds[1]:.2f}, MaxX: {target_bounds[2]:.2f}, MaxY: {target_bounds[3]:.2f}]\n", logging.INFO)

        sample_file_path = execution_plan.iloc[0]['vsi_path']
        source_bbox = build_envelope_from_file(
            target_crs=target_crs,
            target_bounds=target_bounds,
            source_file_path=sample_file_path,
            pixel_buffer=5,
            logger=logger
        )

        level_reprojected_paths: Dict[str, Dict[str, str]] = {
            lvl: {} for lvl in execution_plan['level'].unique()
        }

        grouped_plan = execution_plan.groupby(['level', 'variable'])
        
        # ==========================================
        # 2. Main Processing Loop (Per Variable)
        # ==========================================
        for (level, var_name), group_df in grouped_plan:
            log_execution(logger, f"\nProcessing Level: '{level}' | Variable: '{var_name}'...", logging.INFO)
            tracker.log_usage(f"START Processing {var_name}")
            
            # --- Network Fetch ---
            target_paths = group_df['vsi_path'].unique().tolist()
            
            # Dynamic Worker Safeguard: Cap threads to the exact number of requested files
            num_files = len(target_paths)
            dynamic_workers = min(num_files, max_workers)
            
            log_execution(logger, f"Initiating network fetch for {num_files} asset(s) using {dynamic_workers} worker(s)...", logging.INFO)
            
            with tracker.track_strain(f"Network Fetch ({var_name})"):
                raw_fetched_data = parallel_fetch_rasters(target_paths, source_bbox, dynamic_workers)
            
            # --- Metadata Injection ---
            da_list = []
            for _, row in group_df.iterrows():
                raw_da = raw_fetched_data.get(row['vsi_path'])
                if raw_da is not None:
                    _, structured_da = self.parse_metadata(row, raw_da)
                    da_list.append(structured_da)
                    
            if not da_list:
                log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
                continue
                
            base_x, base_y = da_list[0].coords['x'], da_list[0].coords['y']
            snapped_list = [d.assign_coords(x=base_x, y=base_y) for d in da_list]
            
            z_dim = [dim for dim in snapped_list[0].dims if dim not in ['x', 'y']][0]
            snapped_list.sort(key=lambda da: da.coords[z_dim].values[0])
            
            log_execution(logger, f"  -> Compiling master metadata coordinates for {var_name}...", logging.INFO)
            z_vals = np.array([da[z_dim].values for da in snapped_list]).flatten()
            full_meta_coords = {z_dim: z_vals}
            
            for k in snapped_list[0].coords.keys():
                if k not in ['x', 'y', 'spatial_ref', z_dim]:
                    meta_vector = np.array([da[k].values for da in snapped_list]).flatten()
                    full_meta_coords[k] = (z_dim, meta_vector)
            
            rule = self.get_resample_rule(var_name)
            cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
            os.makedirs(cache_dir, exist_ok=True)

            # ==========================================
            # 3. Hybrid 2D Slice Warping (The Engine)
            # ==========================================
            # Calculate the physical RAM footprint of a single 2D slice. 
            # This metric acts as the trigger switch between our two parallelization architectures.
            slice_size_mb = snapped_list[0].nbytes / (1024 * 1024)
            log_execution(logger, f"  -> Estimated single slice size: {slice_size_mb:.2f} MB", logging.INFO)

            def _warp_worker(da_2d: xr.DataArray, index: int, gdal_threads: str) -> str:
                """
                Worker function executed during both Fast-Path and Safe-Path architectures.
                
                CRITICAL C++ SANDBOXING:
                We wrap the execution in `rasterio.Env()`. This creates an isolated local 
                state for GDAL's underlying C binaries. By dynamically passing `GDAL_NUM_THREADS`, 
                we allow the orchestrator to dictate whether GDAL uses 1 core (during Python Threading) 
                or all available cores (during GDAL Sequential Threading).
                """
                with rasterio.Env(GDAL_NUM_THREADS=gdal_threads, VSI_CACHE="FALSE"):
                    try:
                        nodata_val = da_2d.rio.nodata
                        if nodata_val is not None:
                            if np.issubdtype(da_2d.dtype, np.integer):
                                limits = np.iinfo(da_2d.dtype)
                                if not (limits.min <= nodata_val <= limits.max):
                                    da_2d = da_2d.astype('float32')
                            da_2d.rio.write_nodata(nodata_val, inplace=True)

                        if '_FillValue' in da_2d.attrs:
                            del da_2d.attrs['_FillValue']

                        da_2d = self._sanitize_spatial_geometry(da_2d, default_crs="EPSG:4326", logger=None)
                        out_filepath = os.path.join(cache_dir, f"slice_{index:04d}.tif")
                        
                        self.affine_reproject(
                            input_data=da_2d, 
                            output_filepath=out_filepath, 
                            grid_name=target_grid_key, 
                            resample_keyword=rule, 
                            num_threads=gdal_threads, # EXPLICITLY PASSED
                            logger=None  
                        )
                        return out_filepath
                    except Exception as e:
                        # GRACEFUL DEGRADATION:
                        # Catch network tile corruption gracefully so the pipeline survives.
                        log_execution(logger, f"CRITICAL: GDAL failed to warp slice {index}. Error: {e}", logging.ERROR)
                        return ""

            warped_tif_paths = []
            
            # =================================================================================
            # ARCHITECTURE ROUTER
            # The script checks the payload size and dynamically chooses the safest strategy.
            # =================================================================================
            if slice_size_mb < 50.0:
                # ---------------------------------------------------------
                # FAST PATH: Parallel Slices (Python-Level Threading)
                # ---------------------------------------------------------
                # Small payloads avoid C++ Threading Chunking Overhead by routing multiple
                # slices simultaneously across Python threads.
                log_execution(logger, "  -> Small payload detected. Utilizing high-speed parallel slice warping.", logging.INFO)
                num_python_threads = min(os.cpu_count() or 4, len(snapped_list), max_workers)
                
                with tracker.track_strain(f"Parallel Slice Warp ({var_name})"):
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=num_python_threads) as executor:
                        futures = [
                            executor.submit(_warp_worker, da, i, "1") 
                            for i, da in enumerate(snapped_list)
                        ]
                        warped_tif_paths = [p for p in (f.result() for f in futures) if p != ""]
            
            else:
                # ---------------------------------------------------------
                # SAFE PATH: Parallel Pixels (C++ Level Threading)
                # ---------------------------------------------------------
                # Massive continental payloads route sequentially through Python to avoid 
                # C++ RAM collision (Segmentation Faults), but unlock maximum GDAL CPU cores natively.
                log_execution(logger, "  -> Massive payload detected. Utilizing ultra-stable sequential multi-core warping.", logging.INFO)
                num_gdal_threads = str(min(os.cpu_count() or 4, max_workers))
                
                with tracker.track_strain(f"Sequential Multi-Core Warp ({var_name})"):
                    for i, da in enumerate(snapped_list):
                        result_path = _warp_worker(da, i, num_gdal_threads)
                        if result_path != "":
                            warped_tif_paths.append(result_path)

            # ==========================================
            # 4. Robust 3D Re-assembly 
            # ==========================================
            aligned_slices = []
            log_execution(logger, f"  -> Reassembling 3D {var_name} cube from warped slices...", logging.INFO)
            
            for tif_path in warped_tif_paths:
                warped_2d = rioxarray.open_rasterio(tif_path, chunks=True)
                if 'band' in warped_2d.dims:
                    warped_2d = warped_2d.squeeze('band', drop=True)
                aligned_slices.append(warped_2d)

            combined_da = xr.concat(aligned_slices, dim=z_dim)
            combined_da.name = var_name
            combined_da = combined_da.assign_coords(full_meta_coords)
            combined_da = combined_da.rio.clip_box(*target_bounds)
            
            with tracker.track_strain(f"Dask Materialization ({var_name})"):
                combined_da = combined_da.load()
                
            # ==========================================
            # THE DISK SPILL 
            # ==========================================
            log_execution(logger, f"  -> Spilling {var_name} to nested directory cache...", logging.INFO)
            
            # --- CRS RE-ASSERTION BLOCK (CF CONVENTION) ---
            # xarray operations (concat, clip, load) frequently strip CF-compliant metadata.
            # Here we utilize rioxarray's native CF convention enforcer to rebuild the 
            # spatial_ref coordinate, grid_mapping, and GeoTransform arrays that QGIS requires.

            # Ensure dimensions are recognized as spatial
            combined_da = combined_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            
            # Write the CRS strictly using the CF Convention
            combined_da = combined_da.rio.write_crs(
                target_crs, 
                convention=Convention.CF
            )
            
            # Explicitly write the GeoTransform matrix (Critical for NetCDF in QGIS)
            combined_da = combined_da.rio.write_transform(convention=Convention.CF)
            
            # Fallback root attributes for non-CF compliant parsers
            combined_da.attrs['crs'] = str(target_crs)
            combined_da.attrs['res'] = float(target_res)

            level_dir = os.path.join(base_dir, cube_name, dataset_name, level)
            os.makedirs(level_dir, exist_ok=True)
            
            export_ds = combined_da.to_dataset(name=var_name)
            
            if export_format == 'zarr':
                var_cache_path = os.path.join(level_dir, f"{var_name}.zarr")
                export_ds.to_zarr(var_cache_path, mode='w')
            else:
                var_cache_path = os.path.join(level_dir, f"{var_name}.nc")
                export_ds.to_netcdf(var_cache_path, format="NETCDF4")
            
            level_reprojected_paths[level][var_name] = var_cache_path
            
            xr.backends.file_manager.FILE_CACHE.clear()
            del raw_fetched_data
            del da_list
            del snapped_list
            del aligned_slices
            del combined_da
            del export_ds
            import gc
            gc.collect()

            cooldown_seconds = 30
            log_execution(logger, f"  -> Network cooldown: Sleeping for {cooldown_seconds}s...", logging.INFO)
            import time
            time.sleep(cooldown_seconds)

            tracker.log_usage(f"END Processing {var_name}")

        # ==========================================
        # 5. Pipeline Finalization 
        # ==========================================
        log_execution(logger, "\nValidating Generated Data Cube...", logging.INFO)
        
        total_files = 0
        for level, variables in level_reprojected_paths.items():
            if not variables: continue
            log_execution(logger, f"  -> Level '{level}' successfully generated {len(variables)} independent variable files.", logging.INFO)
            total_files += len(variables)
            
        log_execution(logger, f"=== Data Cube Generation Complete ({total_files} files written to disk) ===", logging.INFO)
        
        return level_reprojected_paths
