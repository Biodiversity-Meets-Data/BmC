import logging
from typing import Optional, Union, Dict, Any, List, Tuple,  Callable
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
import time 

import yaml
import sys
import os
import json
import shutil
import zipfile
import glob
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from rioxarray import set_options
from rioxarray.enum import Convention
from osgeo import gdal
import pyproj
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import shapely

import geopandas as gpd
from typing import Union, Tuple, Optional
import logging
from bmc.utils.logger import log_execution

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
try:
    import dask_geopandas as dask_gpd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
try:
    import pystac
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False


from bmc.utils.spatial import build_envelope_from_file
from bmc.utils.io import parallel_fetch_rasters
from bmc.utils.logger import log_execution, ResourceProfiler

from bmc.engine.legacy.spatial import spatial_engine
from bmc.engine.legacy.spatial import spatial_vector_engine

class spatiotemporal_cube(spatial_engine, ABC):
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
            with tracker.track_strain(f"Network Fetch ({var_name})"):
                raw_fetched_data = parallel_fetch_rasters(target_paths, source_bbox, max_workers)
            
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

class spatiotemporal_vector_cube(spatial_vector_engine, ABC):

    def fetch_vector_by_bbox(
        self,
        file_path: str, 
        target_grid_name: str,
        target_bbox: Tuple[float, float, float, float],
        engine: str = 'auto',
        use_dask: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> Union[gpd.GeoDataFrame, 'dask_gpd.GeoDataFrame']:
        """
        Fetches a spatial subset of a remote or local vector dataset.
        
        Parameters
        ----------
        file_path : str
            URL or local path to the vector file (e.g., .parquet, .gpkg, .fgb).
        target_grid_name : str
            The precise registry key of the target grid to calculate projection curvature.
        target_bbox : tuple
            Strict bounding box in the format (minx, miny, maxx, maxy).
        engine : str
            'auto', 'geopandas', or 'duckdb'. 
        use_dask : bool
            If True, returns a lazy dask-geopandas dataframe for out-of-core processing.
        """
        is_parquet = file_path.lower().endswith(('.parquet', '.geoparquet'))
        is_remote = file_path.startswith(('http', 's3://'))
        
        # 1. Generate the padded envelope BEFORE routing to prevent edge starvation across ALL engines
        safe_bbox = self.build_safe_fetch_envelope(
            target_grid_name=target_grid_name, 
            target_bounds=target_bbox, # The strict metric template bounds
            source_crs_or_grid="EPSG:4326", 
            pixel_buffer=2, # Small buffer to capture curved edge polygons
            logger=logger
        )
        safe_minx, safe_miny, safe_maxx, safe_maxy = safe_bbox
        
        # ---------------------------------------------------------
        # DASK OUT-OF-CORE ENGINE
        # ---------------------------------------------------------
        if use_dask:
            if not HAS_DASK:
                raise ImportError("dask_geopandas is required for out-of-core processing.")
            if not is_parquet:
                raise ValueError("Dask-GeoPandas currently best supports GeoParquet for distributed reads.")
                
            log_execution(logger, f"Lazy loading massive GeoParquet via Dask: {file_path}", logging.INFO)
            import shapely.geometry
            ddf = dask_gpd.read_parquet(file_path)
            return ddf.clip(shapely.geometry.box(*safe_bbox))

        # ---------------------------------------------------------
        # AUTO-DETECTION ROUTER
        # ---------------------------------------------------------
        if engine == 'auto':
            # DuckDB is generally vastly superior for remote Parquet files
            if HAS_DUCKDB and is_parquet:
                log_execution(logger, "Auto-routing to DuckDB (Optimized for Parquet).", logging.INFO)
                engine = 'duckdb'
            # For local GPKG/FGB, native GeoPandas/GDAL is usually fast enough
            else:
                log_execution(logger, "Auto-routing to native GeoPandas engine.", logging.INFO)
                engine = 'geopandas'

        # ---------------------------------------------------------
        # DUCKDB ANALYTICAL ENGINE
        # ---------------------------------------------------------
        if engine == 'duckdb':
            if not HAS_DUCKDB:
                raise ImportError("duckdb is required to use the analytical query engine.")
                
            log_execution(logger, f"Executing DuckDB spatial pushdown on: {file_path}", logging.INFO)
            
            duckdb.execute("INSTALL spatial; LOAD spatial;")
            if is_remote:
                duckdb.execute("INSTALL httpfs; LOAD httpfs;") 
            
            from_clause = f"read_parquet('{file_path}')" if is_parquet else f"ST_Read('{file_path}')"
            
            # Pass the safely padded envelope to the ST_MakeEnvelope spatial filter
            query = f"""
                SELECT * 
                FROM {from_clause}
                WHERE ST_Intersects(
                    geometry, 
                    ST_MakeEnvelope({safe_minx}, {safe_miny}, {safe_maxx}, {safe_maxy})
                )
            """
            # Execute the query and return the spatial dataframe
            return gpd.GeoDataFrame(duckdb.query(query).df(), geometry='geometry')

        # ---------------------------------------------------------
        # NATIVE GEOPANDAS / GDAL ENGINE
        # ---------------------------------------------------------
        if engine == 'geopandas':
            log_execution(logger, f"Fetching subset using native engine: {file_path}", logging.INFO)
            
            if is_parquet:
                return gpd.read_parquet(file_path, bbox=safe_bbox)
            else:
                if is_remote:
                    file_path = file_path.replace("s3://", "/vsis3/") if file_path.startswith("s3://") else f"/vsicurl/{file_path}"
                return gpd.read_file(file_path, bbox=safe_bbox)
    
    @abstractmethod
    def fetch_data(self, recipe: Dict[str, Any], logger: Optional[logging.Logger] = None) -> gpd.GeoDataFrame:
        """
        Translates the execution recipe into raw vector data retrieval.
        Child classes must implement this to query their specific catalogs and return a GeoDataFrame.
        """
        pass

    @abstractmethod
    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: logging.Logger) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key.
        """
        pass

    def _apply_cf_temporal_standards(
    self, 
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    time_cols: Optional[List[str]] = None
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Internal: Converts split temporal columns into CF-compliant ISO-8601 datetimes.
        Safely processes and returns either DataFrames or GeoDataFrames.
        """
        if time_cols is None:
            time_cols = ["year", "month", "day"]
            
        # Safeguard: check if all specified time columns exist in the DataFrame
        missing_cols = [col for col in time_cols if col not in df.columns]
        if missing_cols:
            return df

        work_df = df.copy()
        
        # Build datetime string dynamically depending on the provided temporal resolution
        if len(time_cols) == 3:
            y, m, d = time_cols
            dt_str = work_df[y].astype(str) + '-' + work_df[m].astype(str).str.zfill(2) + '-' + work_df[d].astype(str).str.zfill(2)
        elif len(time_cols) == 2:
            y, m = time_cols
            dt_str = work_df[y].astype(str) + '-' + work_df[m].astype(str).str.zfill(2) + '-01'
        elif len(time_cols) == 1:
            y = time_cols[0]
            dt_str = work_df[y].astype(str) + '-01-01'
        else:
            # Fallback if an empty list or invalid format is passed
            return work_df

        # Convert to datetime and drop the old split columns
        work_df['datetime'] = pd.to_datetime(dt_str, format='%Y-%m-%d', errors='coerce')
        work_df = work_df.drop(columns=time_cols)
            
        return work_df

    def process_cube(
        self, 
        recipe: Dict[str, Any], 
        dataset_name: str, 
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> 'pystac.Item':        
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        processing_mode = source_cfg.get("processing_mode", "vector").lower()
        time_cols = source_cfg.get("time_cols", ["year", "month", "day"])

        # =====================================================================
        # 1. DEFINE DIRECTORY STRUCTURE (cube_dir/dataset_dir)
        # =====================================================================
        base_dir = recipe.get("base_dir", "./")
        cube_name = recipe.get("cube_name", "default_cube")
        
        # Build the nested directory path
        cube_dir = os.path.join(base_dir, cube_name)
        dataset_dir = os.path.join(cube_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Pre-allocate expected filepaths
        source_geom_path = os.path.join(dataset_dir, f"{dataset_name}_source_geometries.parquet")
        unaggregated_path = os.path.join(dataset_dir, f"{dataset_name}_unaggregated.parquet")
        aggregated_path = os.path.join(dataset_dir, f"{dataset_name}_aggregated.parquet")

        log_execution(logger, f"\n=== Initiating {dataset_name.upper()} Generation ===", logging.INFO)
        log_execution(logger, f"Processing Mode: {processing_mode.upper()}", logging.INFO)
        log_execution(logger, f"Output Directory: {dataset_dir}", logging.INFO)

        # =====================================================================
        # 2. FETCH & BUILD BASE TOPOLOGY DATA
        # =====================================================================
        raw_gdf = self.fetch_data(recipe, logger=logger, **kwargs)
        if raw_gdf.empty:
            log_execution(logger, f"Terminating pipeline: fetched {dataset_name} dataset is empty.", logging.WARNING)
            return None

        # Handle Eject Button (Raw Mode)
        if processing_mode == "raw":
            raw_path = os.path.join(dataset_dir, f"{dataset_name}_raw.parquet")
            log_execution(logger, f"Bypassing spatial engine. Exporting RAW data to {raw_path}...", logging.INFO)
            raw_gdf = self._apply_cf_temporal_standards(raw_gdf, time_cols)
            raw_gdf.to_parquet(raw_path)
            # Cannot generate a valid spatial STAC item without a grid index
            return raw_gdf

        # Sanitize Data
        sanitized_gdf = self.sanitize_geometries(raw_gdf, force_multi=False, logger=logger)

        # =====================================================================
        # 3. RESOLVE MASTER GRID & BOUNDING BOX
        # =====================================================================
        spatial_cfg = recipe.get('spatial', {})
        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)

        target_bbox = None
        if spatial_cfg.get('use_bbox', False) and 'bbox' in spatial_cfg:
            from rasterio.warp import transform_bounds
            log_execution(logger, "Extracting padded global bounding box from recipe...", logging.INFO)
            bbox_cfg = spatial_cfg['bbox']
            
            grid_info = self.GRID_REGISTRY[target_grid_key]
            target_crs = grid_info["crs"]
            target_res = grid_info["resolution"]
            
            wgs84_bounds = (
                bbox_cfg["long_min"], bbox_cfg["lat_min"], 
                bbox_cfg["long_max"], bbox_cfg["lat_max"]
            )
            strict_bbox = transform_bounds("EPSG:4326", target_crs, *wgs84_bounds)
            target_bbox = (
                strict_bbox[0] - target_res, strict_bbox[1] - target_res,
                strict_bbox[2] + target_res, strict_bbox[3] + target_res
            )
        
        # =====================================================================
        # 4. DYNAMIC SPATIAL ROUTING
        # =====================================================================
        if processing_mode == "api_cube":
            api_cfg = source_cfg.get("api_cube_config", {})
            transformed_data = self.transform_cellCollection_to_template(
                source_gdf=sanitized_gdf,
                target_grid_name=target_grid_key,
                value_column="occurrenceCount",
                data_type=api_cfg.get("data_type", "discrete"),
                method=api_cfg.get("spatial_method", "intersect"),
                target_bbox=target_bbox,
                logger=logger
            )

        elif processing_mode == "vector":
            vector_cfg = source_cfg.get("vector_processing", {})
            topology = vector_cfg.get("topology", "point")
            mapping_mode = vector_cfg.get("mapping_mode", "fractional")
            spatial_method = vector_cfg.get("spatial_method", "intersect")
            
            if topology == "point":
                transformed_data = self.map_points_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    output_col="grid_idx",
                    method=spatial_method,
                    target_bbox=target_bbox,
                    logger=logger
                )
            elif topology == "point_cloud":
                transformed_data = self.map_point_cloud_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    geom_column=topology,
                    output_col="grid_idx",
                    mode=mapping_mode,
                    classify_method=spatial_method,
                    target_bbox=target_bbox,
                    logger=logger
                )
            elif topology == "polygon":
                transformed_data = self.map_polygon_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    geom_column="geometry",
                    output_col="grid_idx",
                    target_bbox=target_bbox,
                    logger=logger
                )
            else:
                raise ValueError(f"Unsupported topology '{topology}' specified in recipe.")

        # =====================================================================
        # 5. DYNAMIC QA/QC VALIDATION & SOURCE EXPORTS
        # =====================================================================
        aggregate_cfg = source_cfg.get("aggregate", recipe.get("aggregate", {}))
        
        if processing_mode == "vector":
            qa_report = self.validate_vector_transformation(
                orig_gdf=sanitized_gdf,
                targ_gdf=transformed_data,
                recipe=recipe,
                dataset_name=dataset_name,
                logger=logger
            )

            validation_dir = os.path.join(dataset_dir, 'validation_report')
            os.makedirs(validation_dir, exist_ok=True)
            report_path = os.path.join(validation_dir, f"{dataset_name}_qa_qc_failures.json")
            
            import json
            with open(report_path, 'w') as f:
                json.dump(qa_report, f, indent=4)
                
            log_execution(logger, f"Validation report saved to {report_path}", logging.INFO)
            
        else:
            log_execution(logger, "Relational table output detected. Skipping geometric vector QA/QC.", logging.INFO)
            
        # Export the Source Geometries if relational (i.e. shattered vectors converted to pure pandas tables)
        if not isinstance(transformed_data, gpd.GeoDataFrame):
            log_execution(logger, f"Exporting source geometries to {source_geom_path}...", logging.INFO)
            source_export = sanitized_gdf.copy()
            source_export["src_uid"] = source_export.index 
            
            extra_geom_cols = [
                c for c in source_export.columns 
                if c != source_export.geometry.name and (c == "point_cloud" or getattr(source_export[c], "dtype", None) == "geometry")
            ]
            if extra_geom_cols:
                source_export = source_export.drop(columns=extra_geom_cols)

            source_export = self._apply_cf_temporal_standards(source_export, time_cols)
            source_export.to_parquet(source_geom_path)
        else:
            source_geom_path = None # Prevents STAC from referencing missing file

        # Track final paths to pass to STAC
        final_unaggregated_path = None
        final_aggregated_path = None

        # Export unaggregated shattered fractions if requested (or if no aggregation is set)
        if aggregate_cfg.get("export_unaggregated", False) or not aggregate_cfg:
            log_execution(logger, f"Exporting unaggregated relational table to {unaggregated_path}...", logging.INFO)
            unagg_export = self._apply_cf_temporal_standards(transformed_data, time_cols)
            if "grid_idx" in unagg_export.columns:
                unagg_export = unagg_export.dropna(subset=["grid_idx"]).reset_index(drop=True)
            unagg_export.to_parquet(unaggregated_path)
            final_unaggregated_path = unaggregated_path

        # =====================================================================
        # 6. AGGREGATION & FINAL EXPORT
        # =====================================================================
        if processing_mode == "vector" and aggregate_cfg:
            transformed_data = self.aggregate_vector_cube(
                data=transformed_data,
                recipe=recipe,
                dataset_name=dataset_name,
                logger=logger
            )
            
            if "grid_idx" in transformed_data.columns:
                out_of_bounds_count = transformed_data["grid_idx"].isna().sum()
                if out_of_bounds_count > 0:
                    log_execution(logger, f"Filtering out {out_of_bounds_count} records that fell outside the target bounding box...", logging.INFO)
                    transformed_data = transformed_data.dropna(subset=["grid_idx"]).reset_index(drop=True)

            log_execution(logger, f"Exporting final aggregated spatial dataset to {aggregated_path}...", logging.INFO)
            transformed_data = self._apply_cf_temporal_standards(transformed_data, time_cols)
            transformed_data.to_parquet(aggregated_path)
            final_aggregated_path = aggregated_path

        log_execution(logger, f"=== {dataset_name.upper()} Generation Complete ===", logging.INFO)
        
        # =====================================================================
        # 7. GENERATE AND RETURN STAC ITEM
        # =====================================================================
        stac_id = f"{cube_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        stac_item = self.generate_vector_stac_item(
            df=transformed_data,
            recipe=recipe,
            dataset_name=dataset_name,
            source_geom_path=source_geom_path,
            output_dir=dataset_dir,               # Places the grid dimension table cleanly next to the parquet files
            unaggregated_path=final_unaggregated_path,
            aggregated_path=final_aggregated_path,
            item_id=stac_id,
            logger=logger
        )

        return stac_item
    
    def aggregate_vector_cube(
    self,
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    recipe: Dict[str, Any],
    dataset_name: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
        """
        Dedicated aggregation engine for Vector processing mode.
        Dynamically groups by recipe-defined dimensions and executes math 
        according to the metrics map, safely handling fractional weights.
        """
        dataset_cfg = recipe.get("sources", {}).get(dataset_name, {})
        aggregate_cfg = dataset_cfg.get("aggregate", recipe.get("aggregate", {}))

        if not aggregate_cfg:
            log_execution(logger, "No 'aggregate' block found. Returning vector data as-is.", logging.WARNING)
            return data

        # 1. Resolve Grouping Dimensions
        recipe_groups = aggregate_cfg.get("group_by_columns", [])
        
        spatial_col = "grid_id" if "grid_id" in data.columns else "grid_idx"
        group_cols = [spatial_col] + recipe_groups if spatial_col not in recipe_groups else recipe_groups
        actual_group_cols = [col for col in group_cols if col in data.columns]
        
        log_execution(logger, f"Aggregating vector data by dimensions: {actual_group_cols}...", logging.INFO)

        # 2. Determine Spatial Weights for Additive Math
        fraction_col = next((c for c in ["areal_fraction", "fraction"] if c in data.columns), None)
        if fraction_col:
            log_execution(logger, f"Fractional geometries detected. Spatial multiplier: '{fraction_col}'.", logging.INFO)
            data['_base_weight'] = data[fraction_col]
        else:
            log_execution(logger, "Discrete classification detected. Spatial multiplier set to 1.0.", logging.INFO)
            data['_base_weight'] = 1.0

        # 3. Initialize Cube
        aggregated = data[actual_group_cols].drop_duplicates().set_index(actual_group_cols)

        # 4. Execute YAML Metrics Map
        metrics = aggregate_cfg.get("metrics", [])
        for metric in metrics:
            col = metric.get("column")
            method = metric.get("method", "nunique")
            weighted = metric.get("weighted", False)
            new_name = metric.get("rename", f"{col}_{method}")
            
            if col not in data.columns:
                log_execution(logger, f"Metric column '{col}' missing from data payload. Skipping.", logging.WARNING)
                continue
                
            log_execution(logger, f"Aggregating '{col}' -> '{new_name}' (Method: {method}, Weighted: {weighted})", logging.INFO)
                
            # Apply fractional multiplier for additive metrics to conserve mass
            if weighted and method in ['sum', 'mean']:
                temp_col = f"_weighted_{col}"
                # Catch null abundance counts and default to 1.0 to preserve the presence
                numeric_series = pd.to_numeric(data[col], errors='coerce').fillna(1.0 if method == 'sum' else 0.0)
                data[temp_col] = data['_base_weight'] * numeric_series
                aggregated[new_name] = data.groupby(actual_group_cols, dropna=False)[temp_col].agg(method)
            else:
                # Standard metrics (e.g., nunique for Observers and Species)
                aggregated[new_name] = data.groupby(actual_group_cols, dropna=False)[col].agg(method)

        # 5. Cleanup memory
        cols_to_drop = [c for c in data.columns if c.startswith('_weighted_') or c == '_base_weight']
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        log_execution(logger, f"Vector aggregation complete. Yielded {len(aggregated)} final cells.", logging.INFO)
        
        return aggregated.reset_index()

    def generate_vector_stac_item(
        self,
        df: pd.DataFrame,
        recipe: dict,
        dataset_name: str,
        source_geom_path: str,
        output_dir: str,
        unaggregated_path: Optional[str] = None,
        aggregated_path: Optional[str] = None,
        template_zarr_path: Optional[str] = None,
        item_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Converts the relational output of process_cube into a fully compliant STAC Item, 
        embedding topology provenance (e.g., point cloud distribution) and aggregation configurations.
        
        This method generates a Spatial Dimension Table mapping unique grid_idx values 
        to their physical cell geometries, and binds all generated physical files 
        (source geometries, unaggregated fractions, aggregated cubes) as STAC Assets.

        Parameters
        ----------
        df : pandas.DataFrame
            The final dataframe (aggregated or unaggregated) containing 'grid_idx' 
            to derive the spatial footprint.
        recipe : dict
            The pipeline YAML configuration to extract grid definitions and provenance metadata.
        dataset_name : str
            The name of the dataset (e.g., 'gbif').
        source_geom_path : str
            Filepath to the saved original source GeoParquet.
        output_dir : str
            Directory to save the dynamically generated Spatial Dimension GeoParquet.
        unaggregated_path : str, optional
            Filepath to the unaggregated relational fractions/classifications Parquet.
        aggregated_path : str, optional
            Filepath to the final aggregated vector cube Parquet.
        template_zarr_path : str, optional
            Filepath to the Zarr template, if generated.
        item_id : str, optional
            A unique identifier for the STAC Item. Defaults to dataset_name_timestamp.
        logger : logging.Logger, optional
            Pipeline logger.

        Returns
        -------
        pystac.Item
            The assembled STAC Item ready to be added to the bmd_cube Collection.
        """
        if not HAS_PYSTAC:
            raise ImportError("The 'pystac' library is required to generate STAC items. Run: pip install pystac")

        if logger: logger.info(f"=== Assembling STAC Item for {dataset_name.upper()} ===")

        # =====================================================================
        # 1. EXTRACT SPATIAL SPECS & TEMPORAL BOUNDS
        # =====================================================================
        spatial_cfg = recipe.get("spatial", {})
        target_grid = spatial_cfg.get("target_grid", "EEA")
        target_res = spatial_cfg.get("target_resolution", "1km")
        grid_name = f"{target_grid}_{target_res}"
        
        spec = self.GRID_REGISTRY[grid_name]
        res = spec["resolution"]
        master_minx, _, master_maxx, master_maxy = spec["bounds"]

        # Derive accurate temporal bounds from the actual dataset contents
        if 'datetime' in df.columns and not df['datetime'].isna().all():
            start_dt = df['datetime'].min().to_pydatetime().replace(tzinfo=timezone.utc)
            # Push end_dt to the end of the final month/year if applicable
            end_dt = df['datetime'].max().to_pydatetime().replace(tzinfo=timezone.utc)
        elif 'year' in df.columns:
            start_year = int(df['year'].min())
            end_year = int(df['year'].max())
            start_dt = datetime(start_year, 1, 1, tzinfo=timezone.utc)
            # If month is present, set to end of that month, else end of year
            end_month = int(df['month'].max()) if 'month' in df.columns else 12
            if end_month == 12:
                end_dt = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            else:
                # Handle end of specific month safely
                import calendar
                last_day = calendar.monthrange(end_year, end_month)[1]
                end_dt = datetime(end_year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        else:
            start_dt = end_dt = datetime.now(timezone.utc)

        # =====================================================================
        # 2. EXTRACT PROCESSING PROVENANCE METADATA
        # =====================================================================
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        vector_cfg = source_cfg.get("vector_processing", {})
        aggregate_cfg = source_cfg.get("aggregate", {})
        
        topology = vector_cfg.get("topology", "point")
        mapping_mode = vector_cfg.get("mapping_mode", "classification")
        topology_cfg = vector_cfg.get("topology_config", {}).get(topology, {})

        stac_properties = {
            "cube:dataset": dataset_name,
            "cube:grid_registry_key": grid_name,
            "processing:topology": topology,
            "processing:mapping_mode": mapping_mode,
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
        }

        # Encode topology-specific parameters
        if topology == "point_cloud":
            stac_properties["processing:distribution"] = topology_cfg.get("distribution", "gaussian")
            stac_properties["processing:n_passes"] = topology_cfg.get("n_passes", 30)
            stac_properties["processing:random_seed"] = topology_cfg.get("random_seed", None)
        elif topology == "polygon":
            stac_properties["processing:quad_segs"] = topology_cfg.get("quad_segs", 8)

        # Encode aggregation parameters if an aggregation run was performed
        if aggregate_cfg:
            stac_properties["processing:aggregation_groups"] = aggregate_cfg.get("group_by_columns", [])
            stac_properties["processing:aggregation_metrics"] = aggregate_cfg.get("metrics", [])

        # =====================================================================
        # 3. GENERATE THE SPATIAL DIMENSION TABLE (CELL GEOMETRIES)
        # =====================================================================
        if logger: logger.info("Generating discrete Spatial Dimension Table from unique grid IDs...")
        
        # Accommodate 'grid_idx' or 'grid_id' naming conventions
        spatial_col = "grid_idx" if "grid_idx" in df.columns else "grid_id"
        unique_grid_ids = df[spatial_col].dropna().unique()
        
        total_global_cols = int(round((master_maxx - master_minx) / res))
        global_rows = unique_grid_ids // total_global_cols
        global_cols = unique_grid_ids % total_global_cols

        x_centers = master_minx + (global_cols * res) + (res / 2.0)
        y_centers = master_maxy - (global_rows * res) - (res / 2.0)

        half_res = res / 2.0
        polygons = shapely.box(
            x_centers - half_res, 
            y_centers - half_res, 
            x_centers + half_res, 
            y_centers + half_res
        )

        grid_gdf = gpd.GeoDataFrame(
            {spatial_col: unique_grid_ids}, 
            geometry=polygons, 
            crs=spec["crs"]
        )

        grid_mapping_path = os.path.join(output_dir, f"{dataset_name}_{grid_name}_dimension_table.parquet")
        grid_gdf.to_parquet(grid_mapping_path)
        if logger: logger.info(f"Spatial Dimension Table saved to: {grid_mapping_path}")

        # =====================================================================
        # 4. CALCULATE STAC-COMPLIANT BBOX (EPSG:4326)
        # =====================================================================
        wgs84_bounds = grid_gdf.to_crs("EPSG:4326").total_bounds
        bbox = [float(wgs84_bounds[0]), float(wgs84_bounds[1]), float(wgs84_bounds[2]), float(wgs84_bounds[3])]
        footprint = shapely.geometry.mapping(shapely.box(*bbox))

        # =====================================================================
        # 5. CONSTRUCT THE PYSTAC ITEM
        # =====================================================================
        stac_id = item_id or f"{dataset_name}_vector_cube_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        item = pystac.Item(
            id=stac_id,
            geometry=footprint,
            bbox=bbox,
            datetime=start_dt,
            properties=stac_properties
        )

        # =====================================================================
        # 6. ATTACH THE ASSETS (THE STAR SCHEMA LINKS)
        # =====================================================================
        
        # Asset 1: Original Source Geometries (Always Present)
        item.add_asset(
            "source_geometries",
            pystac.Asset(
                href=source_geom_path,
                media_type="application/x-parquet",
                roles=["metadata", "spatial-index"],
                title=f"Source Geometries ({topology.replace('_', ' ').title()})",
                description="Original vector records containing src_uid and generated spatial topologies."
            )
        )

        # Asset 2: Unaggregated Relational Data (If Exported)
        if unaggregated_path and os.path.exists(unaggregated_path):
            item.add_asset(
                "unaggregated_data",
                pystac.Asset(
                    href=unaggregated_path,
                    media_type="application/x-parquet",
                    roles=["data"],
                    title="Unaggregated Relational Data",
                    description=f"Raw {mapping_mode} mapping linking source records to grid cell identifiers."
                )
            )

        # Asset 3: Final Aggregated Vector Cube (If Generated)
        if aggregated_path and os.path.exists(aggregated_path):
            item.add_asset(
                "aggregated_cube",
                pystac.Asset(
                    href=aggregated_path,
                    media_type="application/x-parquet",
                    roles=["data"],
                    title="Aggregated Vector Cube",
                    description="Final multidimensional grouped cube summarizing observations and spatial weights."
                )
            )

        # Asset 4: The Spatial Dimension Table (Grid Cells)
        item.add_asset(
            "grid_geometries",
            pystac.Asset(
                href=grid_mapping_path,
                media_type="application/x-parquet",
                roles=["metadata", "spatial-index"],
                title=f"{grid_name} Spatial Dimension Table",
                description="Explicit GeoParquet linking discrete grid identifiers to physical grid cell polygons."
            )
        )

        # Asset 5: (Optional) The Zarr Template
        if template_zarr_path and os.path.exists(template_zarr_path):
            item.add_asset(
                "zarr_template",
                pystac.Asset(
                    href=template_zarr_path,
                    media_type="application/x-zarr",
                    roles=["metadata"],
                    title=f"{grid_name} Zarr Blueprint",
                    description="Empty N-dimensional Zarr array defining the master grid topology."
                )
            )

        if logger: logger.info(f"=== STAC Item Assembly Complete: {stac_id} ===")
        return item