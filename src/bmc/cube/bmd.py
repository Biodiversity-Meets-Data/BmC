import os
import yaml
import logging
from typing import Optional, List, Dict, Any
import shutil

import importlib
import pkgutil
import inspect

try:
    import pystac
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False

from .datasets.raster.chelsa import chelsaCube
from .datasets.vector.gbif import gbifCube
from bmc.datasource.gbif import sql
from bmc.utils.logger import log_execution
from bmc.utils.provenance import generate_provenance_metadata

class bmdCube:
    """
    The central orchestrator for constructing multi-dimensional spatiotemporal data cubes.

    The `bmd_cube` class manages the end-to-end execution of ecological data pipelines.
    It parses YAML configuration recipes, dispatches execution to specific raster 
    (e.g., CHELSA) and vector (e.g., GBIF) engines, manages asynchronous data fetching, 
    and unifies the distinct outputs into a nested directory structure bound together 
    by a formal SpatioTemporal Asset Catalog (STAC).

    Methods
    -------
    _load_recipe(recipe_file, recipe_path)
        Loads and parses the YAML configuration recipe.
    _dump_recipe(recipe, meta_dir, logger=None)
        Exports a copy of the executed recipe to the metadata directory.
    _export_provenance(recipe, meta_dir, logger=None)
        Generates and saves full execution provenance and environment metadata.
    _collect_stac_item(res, dataset_name, cube_dir, logger=None)
        Retrieves generated PySTAC Items from memory or disk.
    _create_unifying_stac_catalog(stac_items, cube_name, meta_dir, logger=None)
        Wraps disparate dataset PySTAC Items into a cohesive PySTAC Catalog.
    process_recipe(recipe_file, recipe_path, max_workers=10, logger=None, credentials=None)
        Executes the full pipeline: triggers asynchronous downloads, processes data, and writes the output tree.
    """
    
    def __init__(self) -> None:
        """
        Initialize the bmd orchestrator and build the dynamic engine dispatcher.

        Triggers dynamic discovery to discover and map available raster and vector
        dataset engines located within the `bmc.cube.datasets` sub-packages.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._source_map: Dict[str, Any] = self._build_source_map()

    def _build_source_map(self) -> Dict[str, Any]:
        """
        Dynamically discover and register Cube engine classes from dataset modules.

        Scans the `raster` and `vector` package directories inside `bmc.cube.datasets` 
        for classes adhering to the `{datasource}Cube` naming convention (e.g., `chelsaCube`, 
        `gbifCube`). Strips the 'Cube' suffix to populate the lookup dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary mapping lowercased datasource string keys (e.g., 'chelsa', 
            'gbif') to their corresponding uninstantiated class objects.

        Raises
        ------
        ImportError
            If any of the dataset target packages cannot be dynamically imported.
        """
        source_map: Dict[str, Any] = {}

        # Resolve base package path (resolves to 'bmc.cube' when imported)
        base_pkg = __package__ if __package__ else "bmc.cube"

        packages_to_scan = [
            f"{base_pkg}.datasets.raster",
            f"{base_pkg}.datasets.vector",
        ]

        for pkg_name in packages_to_scan:
            try:
                # 1. Import the dataset package (e.g., bmc.cube.datasets.raster)
                pkg = importlib.import_module(pkg_name)

                if not hasattr(pkg, "__path__"):
                    continue

                # 2. Iterate over modules inside the package directory (chelsa.py, gbif.py, etc.)
                for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
                    full_module_name = f"{pkg_name}.{module_name}"
                    module = importlib.import_module(full_module_name)

                    # 3. Inspect module classes matching the {datasource}Cube convention
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if name.endswith("Cube") and obj.__module__ == full_module_name:
                            datasource_key = name[:-4].lower()  # Converts 'chelsaCube' -> 'chelsa'
                            source_map[datasource_key] = obj

            except ImportError as e:
                logging.warning(f"Could not load package '{pkg_name}' during dynamic discovery: {e}")

        return source_map

    def _load_recipe(self, recipe_file: str, recipe_path: str) -> Dict[str, Any]:
        """
        Private helper method to safely load and parse the YAML configuration recipe.

        Parameters
        ----------
        recipe_file : str
            The name or absolute path of the YAML configuration file.
        recipe_path : str
            The base directory containing the recipe file (if `recipe_file` is relative).

        Returns
        -------
        dict
            The parsed dictionary representation of the YAML recipe.
        """
        # Dynamically build the path depending on whether the user provided an absolute or relative reference
        recipe_filepath = recipe_file if os.path.isabs(recipe_file) else os.path.join(recipe_path, recipe_file)
        with open(recipe_filepath) as f:
            return yaml.safe_load(f)

    def _export_provenance(self, recipe: Dict[str, Any], meta_dir: str, logger: Optional[logging.Logger] = None) -> str:
        """
        Generates and saves full execution provenance metadata.
        
        Captures Python environment packages, OS constraints, execution timestamps, 
        and internal system metadata to generate a rigid `provenance_metadata.json` 
        audit log.

        Parameters
        ----------
        recipe : dict
            The execution configuration dictionary.
        meta_dir : str
            The directory to output the JSON provenance log.
        logger : logging.Logger, optional
            Pipeline execution logger.

        Returns
        -------
        str
            The physical filepath to the generated provenance JSON file.
        """
        os.makedirs(meta_dir, exist_ok=True)
        
        # Isolate paths logic: Construct a temporary dictionary routing the base_dir 
        # output directly to the meta_dir so the provenance generator writes it properly.
        provenance_recipe = recipe.copy()
        if "paths" not in provenance_recipe or not isinstance(provenance_recipe["paths"], dict):
            provenance_recipe["paths"] = {}
        provenance_recipe["paths"]["base_dir"] = meta_dir

        log_execution(logger, "--- Generating Execution & Environment Provenance ---", logging.INFO)
        provenance_path = generate_provenance_metadata(
            recipe=provenance_recipe,
            target_package="bmc",
            logger=logger
        )
        return provenance_path

    def _collect_stac_item(
        self, 
        res: Any, 
        dataset_name: str, 
        cube_dir: str, 
        logger: Optional[logging.Logger] = None
    ) -> Optional['pystac.Item']:
        """
        Helper method to collect a PySTAC Item either directly from processing outputs
        or by scanning expected output directories for serialized JSON files.

        Parameters
        ----------
        res : Any
            The return object from a processing engine's `process_cube` method.
        dataset_name : str
            The identifier of the dataset (e.g., 'chelsa', 'gbif').
        cube_dir : str
            The root directory of the generated data cube.
        logger : logging.Logger, optional
            Pipeline execution logger.

        Returns
        -------
        pystac.Item or None
            The parsed PySTAC Item, or None if it cannot be found/STAC is uninstalled.
        """
        if not HAS_PYSTAC:
            return None

        # Scenario 1: Engine natively returned an instantiated PySTAC Item object in RAM (e.g., vector_cube)
        if isinstance(res, pystac.Item):
            return res

        # Scenario 2: Engine dumped STAC to disk and returned paths (e.g., raster_cube limits memory overhead)
        # Search candidate hierarchical locations based on the metadata schema
        candidate_paths = [
            # Updated: Look in the new centralized STAC_assets folder first
            os.path.join(cube_dir, "meta", "STAC_assets", f"{dataset_name}_stac.json"),
            # Legacy fallbacks just in case older datasets haven't been re-run yet
            os.path.join(cube_dir, "meta", f"{dataset_name}_stac.json"),
            os.path.join(cube_dir, dataset_name, "meta", f"{dataset_name}_stac.json")
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    item = pystac.Item.from_file(path)
                    log_execution(logger, f"Collected STAC Item for '{dataset_name}' from {path}", logging.INFO)
                    
                    # Add this line to delete the unlinked, redundant file once it is safely in memory
                    os.remove(path) 
                    
                    return item
                except Exception as e:
                    log_execution(logger, f"Failed to parse STAC Item at {path}: {e}", logging.WARNING)

    def _create_unifying_stac_catalog(
        self, 
        stac_items: List['pystac.Item'], 
        cube_name: str, 
        meta_dir: str, 
        logger: Optional[logging.Logger] = None
    ) -> Optional['pystac.Catalog']:
        """
        Constructs and exports a unifying STAC Catalog containing all items 
        from the processed raster and vector dataset cubes.

        Parameters
        ----------
        stac_items : list of pystac.Item
            The gathered metadata items representing individual datasets.
        cube_name : str
            The human-readable identifier for the parent catalog.
        meta_dir : str
            The destination directory to save `catalog.json`.
        logger : logging.Logger, optional
            Pipeline execution logger.

        Returns
        -------
        pystac.Catalog or None
            The hierarchical PySTAC Catalog object, or None if dependencies/items are missing.
        """
        if not HAS_PYSTAC:
            log_execution(logger, "PySTAC not installed. Skipping unifying STAC catalog generation.", logging.WARNING)
            return None

        if not stac_items:
            log_execution(logger, "No STAC items available to catalog. Skipping unified STAC creation.", logging.WARNING)
            return None

        catalog_id = f"{cube_name}_catalog"
        log_execution(logger, f"--- Constructing Unifying STAC Catalog ({catalog_id}) ---", logging.INFO)

        catalog = pystac.Catalog(
            id=catalog_id,
            description=f"Unified STAC catalog aggregating all sub-cube STAC items for '{cube_name}'."
        )

        for item in stac_items:
            catalog.add_item(item)

        # Updated: Create the dedicated STAC directory inside meta_dir
        stac_out_dir = os.path.join(meta_dir, "STAC")
        os.makedirs(stac_out_dir, exist_ok=True)

        # Ensure all internal asset pointers are resolved relatively to maintain catalog portability
        catalog_path = os.path.join(stac_out_dir, "catalog.json")
        catalog.normalize_hrefs(stac_out_dir)
        catalog.save(pystac.CatalogType.SELF_CONTAINED)

        log_execution(logger, f"Exported unifying STAC Catalog to: {catalog_path}", logging.INFO)
        
        # Clean up the temporary STAC_assets folder and its contents
        temp_assets_dir = os.path.join(meta_dir, "STAC_assets")
        if os.path.exists(temp_assets_dir):
            try:
                import shutil
                shutil.rmtree(temp_assets_dir)
                log_execution(logger, f"Successfully removed temporary STAC assets directory: {temp_assets_dir}", logging.INFO)
            except Exception as e:
                log_execution(logger, f"Failed to remove temporary STAC assets directory {temp_assets_dir}: {e}", logging.WARNING)

        return catalog

    def process_recipe(
        self, 
        recipe_file: str, 
        recipe_path: str, 
        max_workers: int = 10, 
        logger: Optional[logging.Logger] = None,
        creds: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Orchestrates the data cube generation by reading the recipe and delegating
        processing to the respective dataset engines. 

        This pipeline features an asynchronous execution flow: It triggers heavy 
        backend tasks (like GBIF API generation) immediately, runs synchronous localized 
        raster tasks (like CHELSA warping) while waiting, and finally resolves the 
        blocking downloads once the raster generation finishes.

        Parameters
        ----------
        recipe_file : str
            Filename of the configuration target.
        recipe_path : str
            Directory path containing the configuration YAML.
        max_workers : int, optional
            Maximum Python ThreadPool/GDAL C++ cores available to assign to tasks. Default 10.
        logger : logging.Logger, optional
            The master orchestrator tracking execution flow.
        creds : dict, optional
            API credentials dictionary (e.g., gbif_username, gbif_password, gbif_email) 
            passed downstream to remote fetch tasks.
        """
        recipe = self._load_recipe(recipe_file, recipe_path)
        sources = recipe.get("sources", {})

        base_dir = recipe.get("base_dir", ".")
        cube_name = recipe.get("cube_name", "default_cube")
        
        # Determine the root directory and master meta directory for this specific cube hierarchy
        cube_dir = os.path.join(base_dir, cube_name)
        meta_dir = os.path.join(cube_dir, "meta")
        if not os.path.exists(cube_dir):
            log_execution(logger, f"Destination directory '{cube_dir}' not found. Creating path...", logging.INFO)
            os.makedirs(cube_dir, exist_ok=True)

        # =================================================================
        # PHASE 0: EXPORT RECIPE AND PROVENANCE METADATA
        # =================================================================
        # Archive structural context before execution prevents missing context upon crash.
        self._export_provenance(recipe, meta_dir, logger=logger)

        stac_items: List['pystac.Item'] = []
        gbif_download_key = None
        gbif_inst = None
        
        # =================================================================
        # PHASE 1: ASYNCHRONOUS GBIF SUBMISSION
        # =================================================================
        # GBIF SQL API queries can take anywhere from a few minutes to hours depending 
        # on data volume. We compile and submit the query immediately so the remote 
        # GBIF servers begin processing it while our local machine concurrently 
        # downloads and warps raster datasets.
        gbif_cfg = sources.get("gbif")
        gbif_local_path = gbif_cfg.get("local_file_path") if gbif_cfg else None
        
        if gbif_cfg is not None and gbif_cfg.get("enabled", True):
            gbif_inst = self._source_map["gbif"]()
            
            if gbif_local_path:
                log_execution(logger, f"--- Local GBIF data detected: {gbif_local_path}. Bypassing server request. ---", logging.INFO)
            else:
                log_execution(logger, "--- Submitting GBIF Query (Running in Background) ---", logging.INFO)
                query = gbif_inst.generate_gbif_query_from_recipe(recipe, logger=logger)
                
                # Pass the creds dictionary exactly as expected by the original SQL utility
                raw_key_response = sql.submit_gbif_query(query, creds=creds)
                gbif_download_key = raw_key_response.get("key") if isinstance(raw_key_response, dict) else raw_key_response

        # =================================================================
        # PHASE 2: PROCESS ALL OTHER DATASETS (RASTER / OTHER VECTORS)
        # =================================================================
        # Iterates through the dispatch map and executes active raster engines (e.g. CHELSA)
        for source_name, cube_class in self._source_map.items():
            if source_name == "gbif":
                continue # Skip GBIF, as it is uniquely deferred to Phase 3
            
            source_cfg = sources.get(source_name, {})
            if source_cfg.get("enabled", False):
                log_execution(logger, f"Initializing {source_name.upper()} cube generation...", logging.INFO)
                
                # Instantiate the engine dynamically and process synchronous arrays
                cube_inst = cube_class()
                res = cube_inst.process_cube(recipe, max_workers=max_workers, logger=logger)
                
                # Gather the generated metadata structures
                item = self._collect_stac_item(res, source_name, cube_dir, logger)
                if item:
                    stac_items.append(item)

        # =================================================================
        # PHASE 3: FETCH AND PROCESS GBIF (VECTOR)
        # =================================================================
        # Now that raster processing has completed and local system load has reduced, 
        # we resolve the GBIF dataset.
        if gbif_inst:
            if gbif_local_path:
                log_execution(logger, "--- Processing Local GBIF Data ---", logging.INFO)
                local_data_path = gbif_local_path
                
            elif gbif_download_key:
                log_execution(logger, "--- Fetching and Processing GBIF Data ---", logging.INFO)
                download_dir = os.path.join(base_dir, "downloads")
                os.makedirs(download_dir, exist_ok=True)
                
                # The fetch_gbif_download command uses internal exponential backoff to 
                # poll the API. This thread will now block execution safely until the 
                # payload is fully ready and downloaded to the local hard drive.
                # Pass credentials down for authentication retrieval
                download_info = sql.fetch_gbif_download(gbif_download_key, target_dir=download_dir, max_time=10800, creds=creds)
                local_data_path = download_info.get("path") if isinstance(download_info, dict) else download_info
            else:
                local_data_path = None

            if local_data_path:
                log_execution(logger, "Initializing GBIF cube generation...", logging.INFO)
                # Map geometries using the retrieved payload
                res = gbif_inst.process_cube(
                    recipe=recipe, 
                    dataset_name="gbif",
                    downloaded_filepath=local_data_path,
                    logger=logger
                )
                
                item = self._collect_stac_item(res, "gbif", cube_dir, logger)
                if item:
                    stac_items.append(item)

        # =================================================================
        # PHASE 4: CONSTRUCT UNIFYING STAC CATALOG
        # =================================================================
        # Encapsulate the generated independent datasets into a unified hierarchical catalog
        self._create_unifying_stac_catalog(stac_items, cube_name, meta_dir, logger)

        log_execution(logger, f"=== BMD Cube Generation Complete. Nested DataTree available at: {cube_dir} ===", logging.INFO)
        return None