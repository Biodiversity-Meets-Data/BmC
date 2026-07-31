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
import xarray as xr
import rioxarray
from rioxarray.enum import Convention
import rasterio
from rasterio.warp import transform_bounds

from datetime import datetime, timezone
import shapely
import shapely.geometry

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
    apply_multi_index(level, dataset)
        Compiles independent dimensional coordinates into a MultiIndex structure.
    generate_raster_stac_item(level_reprojected_paths, recipe, dataset_name, output_dir, item_id=None, logger=None)
        Converts the processed variables into a fully compliant STAC Item with embedded metadata.
    process_cube(recipe, max_workers=10, logger=None)
        The universal out-of-core spatial processing loop that triggers fetching and warping.
    """  
    def __init__(self):
        pass

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
    def generate_execution_plan(self, recipe: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
        """
        Translates the execution recipe into a standardized data fetching queue.

        Bridges the gap between abstract user configurations and specific remote STAC 
        catalogs or local inventories.

        Parameters
        ----------
        recipe : dict
            The parsed YAML configuration recipe dictating spatiotemporal constraints.
        logger : logging.Logger
            The execution logger recording the inventory fetching process.

        Returns
        -------
        pandas.DataFrame
            A standardized execution queue requiring 'level', 'variable', and 'vsi_path' columns.
        """
        pass

    @abstractmethod
    def parse_metadata(self, row: pd.Series, da: xr.DataArray) -> Tuple[str, xr.DataArray]:
        """
        Extracts dataset-specific metadata and injects it as dimensional coordinates.

        Raw Cloud-Optimized GeoTIFFs downloaded from remote storage are fundamentally 2D. 
        This method expands the 2D spatial array into a 3D or 4D array by parsing the 
        metadata from the execution plan row (e.g., year, ensemble).

        Parameters
        ----------
        row : pd.Series
            A single record from the execution plan DataFrame containing contextual metadata.
        da : xarray.DataArray
            The raw 2D spatial array returned by the network fetcher.

        Returns
        -------
        tuple
            A 2-element tuple containing the processing family 'level' and the structurally 
            augmented 3D/4D DataArray ready for Z-axis concatenation.
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
    def apply_multi_index(self, level: str, dataset: xr.Dataset) -> xr.Dataset:
        """
        Compiles independent dimensional coordinates into a vendor-specific MultiIndex.

        Parameters
        ----------
        level : str
            The processing family grouping string (e.g., 'climatologies').
        dataset : xarray.Dataset
            The fully warped and spatially aligned 3D Dataset.

        Returns
        -------
        xarray.Dataset
            The finalized Dataset containing an optional Pandas MultiIndex on the Z-axis.
        """
        pass

    #################################
    #      Core Processing Loops    #
    #################################

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

    def process_cube(
        self, 
        recipe: Dict[str, Any], 
        max_workers: int = 10,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        The universal out-of-core spatial processing loop.
        
        Executes the spatiotemporal pipeline by fetching raw spatial data, 
        harmonizing bounding boxes, and leveraging a dynamic hybrid multithreaded 
        worker pool to execute C++ GDAL warping across 2D slices.
        
        Parameters
        ----------
        recipe : dict
            The parsed YAML configuration map driving extraction.
        max_workers : int, optional
            The maximum number of parallel Python threads to allocate for networking.
        logger : logging.Logger, optional
            Execution logger stream.

        Returns
        -------
        dict
            Nested directory structure catalog mapping levels to file paths.
        """
        # =====================================================================
        # 1. PIPELINE INITIALIZATION & CONTEXT BOOTSTRAPPING
        # =====================================================================
        # Provisions the directory structure and attaches system hardware telemetrics
        ctx, logger, tracker = self.initialize_pipeline(recipe, logger=logger)
        
        base_dir = ctx["base_dir"]
        cube_name = ctx["cube_name"]
        dataset_name = ctx["dataset_name"]
        export_format = ctx["export_format"]
        spatial_cfg = ctx["spatial_cfg"]

        # Delegate execution plan building to the specific vendor child class
        execution_plan = self.generate_execution_plan(recipe, logger)
        if execution_plan.empty:
            log_execution(logger, "Terminating pipeline: no candidate asset catalog generated.", logging.WARNING)
            return {}

        # Translate spatial specs into mathematical master grids
        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)
        grid_info = self.GRID_REGISTRY[target_grid_key]
        target_crs = grid_info["crs"]
        target_res = grid_info["resolution"]
        
        # Calculate WGS84 and Target-Native coordinate bounding boxes
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

        # Utilize an arbitrary sample file from the execution plan to build a densified, 
        # mathematically padded fetching envelope to prevent spatial edge starvation.
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
        
        # =====================================================================
        # 2. MAIN PROCESSING LOOP (Iterates over Variable Types)
        # =====================================================================
        for (level, var_name), group_df in grouped_plan:
            log_execution(logger, f"\nProcessing Level: '{level}' | Variable: '{var_name}'...", logging.INFO)
            tracker.log_usage(f"START Processing {var_name}")
            
            # --- Network Fetch Phase ---
            target_paths = group_df['vsi_path'].unique().tolist()
            
            # DYNAMIC WORKER SAFEGUARD (Sub-Linear Thread Scaling)
            # Scaling purely linearly crashes the OS networking stack. We scale 
            # at sqrt(N) to maximize connection speed while throttling socket abuse.
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
                raw_fetched_data = parallel_fetch_rasters(target_paths, source_bbox, dynamic_workers)
            
            # --- Dimensional Metadata Injection Phase ---
            # Loop over every downloaded slice and extract the metadata (e.g. Month, Year) 
            # from the plan row to build the 3D Z-axis coordinate matrices.
            da_list = []
            for _, row in group_df.iterrows():
                raw_da = raw_fetched_data.get(row['vsi_path'])
                if raw_da is not None:
                    _, structured_da = self.parse_metadata(row, raw_da)
                    da_list.append(structured_da)
                    
            if not da_list:
                log_execution(logger, f"No valid data returned for {var_name}. Skipping.", logging.WARNING)
                continue
                
            # Align the arbitrary mathematical bases
            base_x, base_y = da_list[0].coords['x'], da_list[0].coords['y']
            snapped_list = [d.assign_coords(x=base_x, y=base_y) for d in da_list]
            
            # Sort arrays over the chronological or vertical axis to ensure NetCDF alignment
            z_dim = [dim for dim in snapped_list[0].dims if dim not in ['x', 'y']][0]
            snapped_list.sort(key=lambda da: da.coords[z_dim].values[0])
            
            log_execution(logger, f"  -> Compiling master metadata coordinates for {var_name}...", logging.INFO)
            z_vals = np.array([da[z_dim].values for da in snapped_list]).flatten()
            full_meta_coords = {z_dim: z_vals}
            
            # Flatten non-spatial coordinates into indexable 1D arrays
            for k in snapped_list[0].coords.keys():
                if k not in ['x', 'y', 'spatial_ref', z_dim]:
                    meta_vector = np.array([da[k].values for da in snapped_list]).flatten()
                    full_meta_coords[k] = (z_dim, meta_vector)
            
            # Define specific ecological resampling algorithm based on the variable content
            rule = self.get_resample_rule(var_name)
            cache_dir = os.path.join(base_dir, "warp_cache", level, var_name)
            os.makedirs(cache_dir, exist_ok=True)

            # =====================================================================
            # 3. HYBRID 2D SLICE WARPING (The C++ Mathematical Engine)
            # =====================================================================
            # Calculates the approximate physical RAM footprint required by one slice.
            # If the payload is too large, doing parallel Python threads will crash 
            # due to Out Of Memory (OOM) killer terminating the process.
            slice_size_mb = snapped_list[0].nbytes / (1024 * 1024)
            log_execution(logger, f"  -> Estimated single slice size: {slice_size_mb:.2f} MB", logging.INFO)

            def _warp_worker(da_2d: xr.DataArray, index: int, gdal_threads: str) -> str:
                """
                Worker function executing GDAL isolated routines across varying threading structures.
                Wraps runtime into rasterio.Env() to ensure C-bindings don't collide in parallel.
                """
                with rasterio.Env(GDAL_NUM_THREADS=gdal_threads, VSI_CACHE="FALSE"):
                    try:
                        nodata_val = da_2d.rio.nodata
                        if nodata_val is not None:
                            # Catch data limits integer wrapping that breaks float representation mapping
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
                            num_threads=gdal_threads, 
                            logger=None  
                        )
                        return out_filepath
                    except Exception as e:
                        log_execution(logger, f"CRITICAL: GDAL failed to warp slice {index}. Error: {e}", logging.ERROR)
                        return ""

            warped_tif_paths = []
            num_slices = len(snapped_list)
            MIN_SLICES_FOR_PYTHON_PARALLEL = 16
            MAX_SAFE_THREADPOOL_WORKERS = 8
            LARGE_SLICE_THRESHOLD_MB = 50.0

            if slice_size_mb < LARGE_SLICE_THRESHOLD_MB:
                if num_slices < MIN_SLICES_FOR_PYTHON_PARALLEL:
                    # PATH A: MAIN-THREAD SEQUENTIAL 
                    # If we only have a few small slices (e.g. 12 month climatologies), 
                    # the overhead of spawning Python Threads is slower than doing it directly.
                    log_execution(
                        logger, 
                        f"  -> Executing sequentially on main thread for 100% stability...", 
                        logging.INFO
                    )
                    with tracker.track_strain(f"Main-Thread Sequential Warp ({var_name})"):
                        for i, da in enumerate(snapped_list):
                            result = _warp_worker(da, i, "1")
                            if result != "": warped_tif_paths.append(result)

                else:
                    # PATH B: PYTHON THREADPOOL
                    # Optimal for high volume, small payload slices. Threads manage parallel I/O.
                    calc_workers = int(math.sqrt(num_slices))
                    dynamic_workers = max(1, min(calc_workers, max_workers, MAX_SAFE_THREADPOOL_WORKERS))
                    
                    log_execution(
                        logger, 
                        f"  -> Spawning Python ThreadPool with {dynamic_workers} worker(s)...", 
                        logging.INFO
                    )

                    with tracker.track_strain(f"Parallel ThreadPool Warp ({var_name})"):
                        with ThreadPoolExecutor(max_workers=dynamic_workers) as executor:
                            futures = [executor.submit(_warp_worker, da, i, "1") for i, da in enumerate(snapped_list)]
                            for f in futures:
                                res = f.result()
                                if res != "": warped_tif_paths.append(res)

            else:
                # PATH C: NATIVE C++ MULTI-THREADING 
                # If payloads are enormous (>50MB), we execute sequentially in Python to save memory, 
                # but tell the GDAL C++ engine to utilize all system CPUs internally.
                num_gdal_threads = str(min(os.cpu_count() or 4, max_workers))
                log_execution(
                    logger, 
                    f"  -> Large payload detected. Routing with native GDAL C++ threads ({num_gdal_threads})...", 
                    logging.INFO
                )

                with tracker.track_strain(f"C++ Multi-Core Native Warp ({var_name})"):
                    for i, da in enumerate(snapped_list):
                        result_path = _warp_worker(da, i, num_gdal_threads)
                        if result_path != "": warped_tif_paths.append(result_path)

            # =====================================================================
            # 4. ROBUST 3D RE-ASSEMBLY & DASK MATERIALIZATION
            # =====================================================================
            aligned_slices = []
            log_execution(logger, f"  -> Reassembling 3D {var_name} cube from warped slices...", logging.INFO)
            
            for tif_path in warped_tif_paths:
                warped_2d = rioxarray.open_rasterio(tif_path, chunks=True)
                # Squeeze dummy bands out of single-band arrays to prevent dimension conflicts
                if 'band' in warped_2d.dims:
                    warped_2d = warped_2d.squeeze('band', drop=True)
                aligned_slices.append(warped_2d)

            # Concatenate back into a single 3D stack based on our metadata Z-axis (e.g., 'time')
            combined_da = xr.concat(aligned_slices, dim=z_dim)
            combined_da.name = var_name
            combined_da = combined_da.assign_coords(full_meta_coords)
            
            # Explicitly force a final mathematical clip to the exact bounding box
            combined_da = combined_da.rio.clip_box(*target_bounds)
            
            # Execute the Dask graph into Physical RAM
            with tracker.track_strain(f"Dask Materialization ({var_name})"):
                combined_da = combined_da.load()
                
            # =====================================================================
            # 5. THE DISK SPILL (Writing NetCDF/Zarr to Disk)
            # =====================================================================
            log_execution(logger, f"  -> Spilling {var_name} to nested directory cache...", logging.INFO)
            
            # Apply strict Climate and Forecast (CF) spatial conventions
            combined_da = combined_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            combined_da = combined_da.rio.write_crs(target_crs, convention=Convention.CF)
            combined_da = combined_da.rio.write_transform(convention=Convention.CF)
            combined_da.attrs['crs'] = str(target_crs)
            combined_da.attrs['res'] = float(target_res)

            level_dir = os.path.join(base_dir, cube_name, dataset_name, level)
            os.makedirs(level_dir, exist_ok=True)
            
            export_ds = combined_da.to_dataset(name=var_name)

            # Prevent NetCDF4 freezing on Object-type coordinates (common for string ensembles)
            for v_key in export_ds.variables:
                if export_ds[v_key].dtype == 'O':
                    export_ds[v_key] = export_ds[v_key].astype(str)

            if export_format == 'zarr':
                var_cache_path = os.path.join(level_dir, f"{var_name}.zarr")
                export_ds.to_zarr(var_cache_path, mode='w')
            else:
                var_cache_path = os.path.join(level_dir, f"{var_name}.nc")
                export_ds.to_netcdf(var_cache_path, engine="netcdf4", format="NETCDF4")
            
            level_reprojected_paths[level][var_name] = var_cache_path
            
            # Aggressive garbage collection to prevent Memory Leaks over 100+ variables
            xr.backends.file_manager.FILE_CACHE.clear()
            del raw_fetched_data, da_list, snapped_list, combined_da, export_ds
            gc.collect()

            # Network Cooldown allows STAC/WEkEO API socket closures to process fully
            cooldown_seconds = 30
            log_execution(logger, f"  -> Network cooldown: Sleeping for {cooldown_seconds}s...", logging.INFO)
            time.sleep(cooldown_seconds)
            tracker.log_usage(f"END Processing {var_name}")
            
        # =====================================================================
        # 6. PIPELINE FINALIZATION & STAC EXPORT
        # =====================================================================
        log_execution(logger, "\nValidating Generated Data Cube...", logging.INFO)
        total_files = 0
        for level, variables in level_reprojected_paths.items():
            if variables: 
                log_execution(logger, f"  -> Level '{level}' successfully generated {len(variables)} independent variable files.", logging.INFO)
                total_files += len(variables)
            
        log_execution(logger, f"=== Data Cube Generation Complete ({total_files} files written to disk) ===", logging.INFO)
        
        stac_assets_dir = os.path.join(base_dir, cube_name, "meta", "STAC_assets")
        os.makedirs(stac_assets_dir, exist_ok=True)
        
        # Instantiate STAC collection building routine
        stac_item = self.generate_raster_stac_item(
            level_reprojected_paths=level_reprojected_paths,
            recipe=recipe,
            dataset_name=dataset_name,
            output_dir=stac_assets_dir,
            logger=logger
        )

        # Write the JSON dictionary to the STAC_assets directory
        stac_filepath = os.path.join(stac_assets_dir, f"{dataset_name}_stac.json")
        with open(stac_filepath, "w") as f:
            json.dump(stac_item.to_dict(), f, indent=4)
            
        log_execution(logger, f"Exported STAC item to {stac_filepath}", logging.INFO)

        return level_reprojected_paths