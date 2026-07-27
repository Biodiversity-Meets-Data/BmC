import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
import gc
import os
import zipfile
import shutil
import glob
import uuid
from pathlib import Path

os.environ["GDAL_CACHEMAX"] = "1024"      # Give GDAL 1GB of RAM for caching
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS" # Unleash multi-threading
os.environ["VSI_CACHE"] = "TRUE"          # Optimize virtual file reading
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR" # Speeds up file discovery

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from rioxarray import set_options
from rioxarray.enum import Convention
from osgeo import gdal
from shapely.geometry import box
import geopandas as gpd

from bmc.utils.spatial import build_safe_fetch_envelope
from bmc.utils.logger import log_execution
from bmc.engine.spatial import spatial_engine

class spatiotemporal_lake(spatial_engine, ABC):
 
    def export_to_cog(
        self, 
        ds: Union[xr.DataArray, xr.Dataset], 
        output_filepath: str, 
        compress_mode: str = "deflate",
        logger: Optional[logging.Logger] = None
    ) -> str:
        """
        Exports an xarray object to disk as a Cloud Optimized GeoTIFF (COG).
        Uses a two-step multi-threaded process to avoid Dask single-core bottlenecks.

        Parameters
        ----------
        ds : xarray.DataArray or xarray.Dataset
            The spatial data to be exported.
        output_filepath : str
            The target file path (must end in .tif).
        compress_mode : str, optional
            The compression algorithm. 'deflate' is highly recommended for SDM data.
        logger : logging.Logger, optional
            Logger for recording execution progress.

        Returns
        -------
        output_filepath : str
            The path to the successfully created COG.
        """
        log_execution(logger, f"Exporting to Cloud Optimized GeoTIFF: {output_filepath}", logging.INFO)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        
        # 1. Define a temporary standard GTiff path
        base_name, ext = os.path.splitext(output_filepath)
        temp_tif = f"{base_name}_temp_streaming{ext}"

        try:
            # 2. Stream Dask chunks to a standard tiled GTiff (Fast & Multi-core friendly)
            ds.rio.to_raster(
                temp_tif,
                driver="GTiff",
                compress=compress_mode,
                tiled=True,
                windowed=True
            )
            
            log_execution(logger, "Temporary GTiff stream complete. Building COG pyramids across all cores...", logging.INFO)

            # 3. Use GDAL C++ bindings to aggressively build the COG using all available CPU threads
            gdal.UseExceptions()
            gdal.Translate(
                output_filepath, 
                temp_tif, 
                format="COG", 
                creationOptions=[
                    f"COMPRESS={compress_mode.upper()}", 
                    "NUM_THREADS=ALL_CPUS",
                    "BIGTIFF=YES"  # Safety net for massive continent-wide mosaics
                ]
            )
            
            log_execution(logger, "COG successfully generated.", logging.INFO)
            return output_filepath

        except Exception as e:
            log_execution(logger, f"Failed to export COG: {e}", logging.ERROR, exc_info=True)
            raise
            
        finally:
            # 4. Always clean up the temporary physical file to save disk space
            if os.path.exists(temp_tif):
                try:
                    os.remove(temp_tif)
                except OSError as cleanup_error:
                    log_execution(logger, f"Warning: Could not delete temporary file {temp_tif}: {cleanup_error}", logging.WARNING)

    def gather_tifs_from_zips(
        self, 
        source_directory: str, 
        target_directory: str, 
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Iterates through zip archives in a source directory and extracts 
        only the .tif files, flattening them into a single target directory.

        This method acts as a data preparation step, isolating raw spatial 
        arrays from nested archive structures and auxiliary metadata files 
        prior to Virtual Raster (VRT) mosaic construction.

        Parameters
        ----------
        source_directory : str
            The path to the directory containing the downloaded .zip archives.
        target_directory : str
            The destination directory where extracted .tif files will be saved.
            If the directory does not exist, it will be created.
        logger : logging.Logger, optional
            The logger instance for recording execution progress and errors. 
            Defaults to None.

        Returns
        -------
        None
        """
        # Set up the paths
        source_path = Path(source_directory)
        target_path = Path(target_directory)
        
        # Create the target directory if it doesn't exist yet
        target_path.mkdir(parents=True, exist_ok=True)

        # Find all zip files in the source folder
        zip_files = list(source_path.glob("*.zip"))
        log_execution(logger, f"Found {len(zip_files)} zip files. Starting extraction...", logging.INFO)

        # Iterate over each zip file
        for zip_file_path in zip_files:
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    
                    # Look at every file hidden inside the zip archive
                    for file_info in zip_ref.infolist():
                        
                        # Check if the file is a .tif
                        if file_info.filename.endswith('.tif'):
                            
                            # Extract the filename without any internal zip folder structure
                            tif_filename = Path(file_info.filename).name 
                            destination_file = target_path / tif_filename
                            
                            # Copy the .tif file to the shared folder
                            with zip_ref.open(file_info) as source_file:
                                with open(destination_file, 'wb') as target_file:
                                    shutil.copyfileobj(source_file, target_file)
            except zipfile.BadZipFile:
                log_execution(logger, f"Warning: {zip_file_path.name} is corrupted or not a valid zip file.", logging.WARNING)
                continue
            except Exception as e:
                log_execution(logger, f"Error extracting from {zip_file_path.name}: {e}", logging.ERROR)

        log_execution(logger, "All .tif files have been gathered successfully!", logging.INFO)

    def cleanup_raw_storage(self, recipe: Dict, logger: Optional[logging.Logger] = None) -> None:
        """
        Safely purges the raw data directory based on the user configuration.

        This function checks the execution recipe for the `keep_raw` parameter.
        If `keep_raw` is explicitly set to False, it will aggressively remove 
        the raw downloads directory to free up disk space. It uses `ignore_errors=True` 
        during removal to prevent phantom file locks (e.g., [WinError 32] on Windows) 
        from crashing the pipeline. If `keep_raw` is True or missing, the raw 
        data is safely retained.

        Parameters
        ----------
        recipe : dict
            The parsed execution configuration dictionary. It is expected to contain 
            the `keep_raw` boolean under `recipe['raw_config']['keep_raw']` and the 
            target directory path under `recipe['paths']['raw_dir']`.
        logger : logging.Logger, optional
            The logger instance used to record the execution and cleanup status. 
            Default is None.

        Returns
        -------
        None
            This function does not return a value.
        """
        keep_raw = recipe.get('raw_config', {}).get('keep_raw', True)
        raw_dir = recipe.get('paths', {}).get('raw_dir')

        if not keep_raw and raw_dir and os.path.exists(raw_dir):
            if logger:
                logger.info(f"keep_raw is False. Purging raw data directory: {raw_dir}")
            try:
                # Delete an entire directory tree; path must point to a directory (but not a symbolic link to a directory). 
                shutil.rmtree(raw_dir, ignore_errors=True)
                if logger:
                    logger.info("Raw data successfully deleted.")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not completely delete raw directory: {e}")
        elif keep_raw:
            if logger:
                logger.info("keep_raw is True. Retaining raw downloaded data.")

    def build_fractional_cogs(self, source_path: str, grid_name, output_dir, class_values, logger, file_prefix="fractional", class_mapping=None):
        """
        Lake Orchestrator: Iterates classes, computes fractions to disk, and bakes COGs directly.
        """
        import uuid
        
        for cls in class_values:
            cls_int = int(cls)
            
            if class_mapping and cls_int in class_mapping:
                cls_name = str(class_mapping[cls_int]).replace(" ", "_").replace("/", "_").replace("-", "_")
            else:
                cls_name = f"class_{cls_int}"

            # --- THE FIX ---
            # 1. Create a safe physical temp file EXPLICITLY on the target hard drive
            # Bypassing tempfile.mkstemp entirely to guarantee it stays off the RAM disk.
            temp_path = os.path.join(output_dir, f"temp_frac_{cls_name}_{uuid.uuid4().hex[:8]}.tif")

            final_cog_path = os.path.join(output_dir, f"{file_prefix}_{cls_name}.tif")

            try:
                if logger:
                    logger.info(f"Processing spatial math for class: {cls_name}")
                
                # 2. Hand the VRT string path down to the pure rasterio/GDAL math primitive
                self.compute_class_fraction(source_path, cls_int, grid_name, temp_path, logger)
                
                # 3. Use GDAL C++ bindings to aggressively build the COG directly from the temp file
                gdal.UseExceptions()
                gdal.Translate(
                    final_cog_path, 
                    temp_path, 
                    format="COG", 
                    creationOptions=[
                        "COMPRESS=DEFLATE", 
                        "NUM_THREADS=ALL_CPUS",
                        "BIGTIFF=YES"
                    ]
                )
                
                if logger:
                    logger.info(f"Successfully baked COG: {final_cog_path}")

            except Exception as e:
                if logger:
                    logger.error(f"Pipeline failure on class {cls_name}: {e}", exc_info=True)
                raise
            finally:
                # 4. Clean up the physical file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        if logger:
                            logger.warning(f"Could not delete temp file {temp_path}: {e}")

    def build_virtual_mosaic(
        self,
        input_folder: str, 
        output_vrt_path: str, 
        logger: Optional[logging.Logger] = None
    ) -> Optional[str]:
        """
        Creates a lightweight Virtual Raster (VRT) blueprint from multiple GeoTIFF tiles.

        This method discovers all `.tif` files in a folder and creates an XML-based `.vrt` 
        file, mosaicking them together at their native resolution.

        Parameters
        ----------
        input_folder : str
            The directory containing the raw `.tif` tiles.
        output_vrt_path : str
            The destination file path for the blueprint (must end in `.vrt`).
        logger : logging.Logger, optional
            The logger instance to use for recording execution messages. Default is ``None``.

        Returns
        -------
        str or None
            The file path to the generated `.vrt` file. Returns ``None`` if no tiles were found.
        """
        # Enable errors
        gdal.UseExceptions()
        
        tif_files = glob.glob(f"{input_folder}/*.tif")
        if not tif_files:
            log_execution(logger, f"No .tif files found in '{input_folder}'.", logging.WARNING)
            return None

        log_execution(logger, f"Found {len(tif_files)} files. Building VRT blueprint...", logging.INFO)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_vrt_path)), exist_ok=True)

            # Build the VRT without any resampling options
            vrt = gdal.BuildVRT(output_vrt_path, tif_files)
            
            # Critical: Flush cache and destroy the python object to force GDAL to write the XML to disk
            vrt.FlushCache()
            vrt = None 

            log_execution(logger, f"Virtual mosaic successfully saved to: {output_vrt_path}", logging.INFO)
            return output_vrt_path

        except Exception as e:
            log_execution(logger, f"Error building virtual mosaic: {e}", logging.ERROR, exc_info=True)
            raise

        
    def process_virtual_mosaic(
        self, 
        vrt_path: str, 
        strategy: str, 
        grid_name: str,
        output_dir_or_file: str,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> Union[xr.DataArray, xr.Dataset, None]:
        """
        Routes a Virtual Raster (VRT) blueprint to the appropriate spatial processing algorithm.

        This dispatcher function delays heavy pixel-crunching until the exact mathematical 
        strategy is determined. It handles the I/O handoff, routing the lightweight XML 
        blueprint to either a standard GDAL affine reprojection or a categorical 
        fractional coverage calculator.

        Parameters
        ----------
        vrt_path : str
            The file path to the source `.vrt` file.
        strategy : str
            The processing algorithm to apply. Valid options: 'reproject', 'coverage'.
        grid_name : str
            The key of the target grid defined in the class grid registry.
        output_dir_or_file : str
            If strategy is 'reproject', this should be the output file path (.tif).
            If strategy is 'coverage', this should be the output directory.
        logger : logging.Logger, optional
            The logger instance to use for recording execution messages. Default is ``None``.
        **kwargs : dict
            Keyword arguments passed directly to the chosen processing function 
            (e.g., class_values, class_mapping, file_prefix, resample_keyword, compress_mode).

        Returns
        -------
        xarray.DataArray or xarray.Dataset or None
            The lazily loaded result of the processing step if 'reproject'. 
            Returns None if 'coverage' is used, as it bakes directly to disk.
        """
        if not os.path.exists(vrt_path):
            raise FileNotFoundError(f"VRT blueprint not found at: {vrt_path}")

        log_execution(logger, f"Initializing '{strategy}' processing pipeline for VRT...", logging.INFO)

        try:
            if strategy.lower() == 'reproject':
                return self.affine_reproject(
                    input_data=vrt_path, 
                    output_filepath=output_dir_or_file, 
                    grid_name=grid_name, 
                    logger=logger, 
                    **kwargs
                )
                
            elif strategy.lower() == 'coverage':
                target_classes = kwargs.get('class_values', [])
                file_prefix = kwargs.get('file_prefix', 'fractional')
                class_mapping = kwargs.get('class_mapping', None)
                
                # Pass the physical string path directly, avoiding Dask and RAM overhead entirely.
                self.build_fractional_cogs(
                    source_path=vrt_path, 
                    grid_name=grid_name, 
                    output_dir=output_dir_or_file, 
                    class_values=target_classes, 
                    logger=logger,
                    file_prefix=file_prefix,
                    class_mapping=class_mapping
                )
                return None
                
            else:
                raise ValueError(f"Unknown processing strategy: '{strategy}'. Must be 'reproject' or 'coverage'.")

        except Exception as e:
            log_execution(logger, f"Pipeline failure during {strategy}: {e}", logging.ERROR, exc_info=True)
            raise

    @abstractmethod
    def fetch_raw_data(self, recipe: Dict[str, Any], logger: logging.Logger) -> Any:
        """
        Connects to the vendor API or remote server and downloads the raw source files.
        
        Parameters
        ----------
        recipe : dict
            The configuration dictionary containing API credentials, spatial/temporal 
            bounds, and requested datasets.
        logger : logging.Logger
            Logger for recording API connection status and download progress.
        """
        pass

    @abstractmethod
    def build_datalake(self, recipe: Dict[str, Any], logger: logging.Logger) -> List[str]:
        """
        Translates raw downloaded files into standardized Cloud Optimized GeoTIFFs (COGs).
        
        This method should orchestrate the `spatial_engine` tools to mosaic raw tiles, 
        calculate fractional coverages, and save the final standardized arrays to the 
        local data lake directory.

        Parameters
        ----------
        recipe : dict
            The configuration dictionary dictating resampling strategies and output paths.
        logger : logging.Logger
            Logger for recording spatial transformations and I/O operations.

        Returns
        -------
        list of str
            A list of file paths pointing to the successfully generated COGs.
        """
        pass

    @abstractmethod
    def validate_datalake(self, base_dir: str, logger: logging.Logger) -> bool:
        """
        Performs strict mathematical and ecological QA validations on the generated lake.
        
        Parameters
        ----------
        base_dir : str
            The root directory of the finalized data lake.
        logger : logging.Logger
            Logger for recording QA test results and flagging physics violations.

        Returns
        -------
        bool
            True if the data passes all physics checks without corruption, False otherwise.
        """
        pass

    @abstractmethod
    def generate_catalog(self, target_dir: str, logger: logging.Logger) -> str:
        """
        Builds the final inventory file that downstream runtime cubes will query.
        
        Parameters
        ----------
        target_dir : str
            The root directory where the catalog file should be saved.
        logger : logging.Logger
            Logger for recording catalog generation.

        Returns
        -------
        str
            The file path to the newly generated catalog (e.g., CSV, Parquet).
        """
        pass