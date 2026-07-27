import os
import yaml
import logging
from typing import Optional

from .datasets.raster.chelsa import chelsa_cube
from .datasets.vector.gbif import gbif_cube
from bmc.datasource.gbif import sql
from bmc.utils.logger import log_execution

class bmd_cube:
    def __init__(self):
        # Dispatch table mapping YAML keys to their respective class objects
        self._source_map = {
            "chelsa": chelsa_cube,
            "gbif": gbif_cube
        }

    def _load_recipe(self, recipe_file, recipe_path):
        """Private helper method to load and parse the YAML recipe."""
        recipe_filepath = recipe_file if os.path.isabs(recipe_file) else os.path.join(recipe_path, recipe_file)
        with open(recipe_filepath) as f:
            return yaml.safe_load(f)

    def process_recipe(self, recipe_file, recipe_path, max_workers=10, logger: Optional[logging.Logger] = None):
        """
        Orchestrates the data cube generation by reading the recipe and delegating
        processing to the respective dataset engines. The output is inherently 
        structured as a nested directory tree on disk.
        """
        recipe = self._load_recipe(recipe_file, recipe_path)
        sources = recipe.get("sources", {})

        base_dir = recipe.get("base_dir", ".")
        cube_name = recipe.get("cube_name", "default_cube")
        
        # Determine the root directory for this specific cube hierarchy
        cube_dir = os.path.join(base_dir, cube_name)
        if not os.path.exists(cube_dir):
            log_execution(logger, f"Destination directory '{cube_dir}' not found. Creating path...", logging.INFO)
            os.makedirs(cube_dir, exist_ok=True)

        gbif_download_key = None
        gbif_inst = None
        
        # =================================================================
        # PHASE 1: ASYNCHRONOUS GBIF SUBMISSION
        # If GBIF is requested, submit the query to their servers immediately 
        # so it compiles while we process local datasets.
        # =================================================================
        gbif_cfg = sources.get("gbif")
        # Default to True if 'enabled' is missing from the GBIF block
        if gbif_cfg is not None and gbif_cfg.get("enabled", True):
            log_execution(logger, "--- Submitting GBIF Query (Running in Background) ---", logging.INFO)
            gbif_inst = self._source_map["gbif"]()
            query = gbif_inst.generate_gbif_query_from_recipe(recipe, logger=logger)
            
            raw_key_response = sql.submit_gbif_query(query)
            
            # Safeguard: Extract the string if submit_gbif_query returns a dictionary
            gbif_download_key = raw_key_response.get("key") if isinstance(raw_key_response, dict) else raw_key_response

        # =================================================================
        # PHASE 2: PROCESS ALL OTHER DATASETS
        # Execute raster/vector operations while GBIF generates on the server.
        # =================================================================
        for source_name, cube_class in self._source_map.items():
            if source_name == "gbif":
                continue # Skip GBIF, handled in Phase 3
            
            # Check if this specific source is present and enabled in the recipe
            source_cfg = sources.get(source_name, {})
            if source_cfg.get("enabled", False):
                log_execution(logger, f"Initializing {source_name.upper()} cube generation...", logging.INFO)
                
                # Instantiate the mapped class dynamically
                cube_inst = cube_class()
                
                # The engine handles STAC generation and exporting directly 
                # to the nested directory structure on disk.
                cube_inst.process_cube(recipe, max_workers=max_workers, logger=logger)

        # =================================================================
        # PHASE 3: FETCH AND PROCESS GBIF
        # =================================================================
        if gbif_download_key and gbif_inst:
            log_execution(logger, "--- Fetching and Processing GBIF Data ---", logging.INFO)
            
            # Save raw downloads to a shared sibling 'downloads' folder
            download_dir = os.path.join(base_dir, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            
            # Fetch the download (this blocks until the file is ready on the GBIF server)
            download_info = sql.fetch_gbif_download(gbif_download_key, target_dir=download_dir)
            
            # Safeguard: Extract the path string if fetch_gbif_download returns a dictionary
            local_zip_path = download_info.get("path") if isinstance(download_info, dict) else download_info
            
            log_execution(logger, "Initializing GBIF cube generation...", logging.INFO)
            gbif_inst.process_cube(
                recipe=recipe, 
                dataset_name="gbif",
                downloaded_filepath=local_zip_path,
                logger=logger
            )

        log_execution(logger, f"=== BMD Cube Generation Complete. Nested DataTree available at: {cube_dir} ===", logging.INFO)
        return None