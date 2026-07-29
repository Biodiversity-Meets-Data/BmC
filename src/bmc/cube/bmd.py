import os
import yaml
import logging
from typing import Optional, List, Dict, Any

try:
    import pystac
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False

from .datasets.raster.chelsa import chelsa_cube
from .datasets.vector.gbif import gbif_cube
from bmc.datasource.gbif import sql
from bmc.utils.logger import log_execution
from bmc.utils.provenance import generate_provenance_metadata

class bmd_cube:
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
    
    def __init__(self):
        """
        Initializes the bmd_cube orchestrator and establishes the dynamic engine dispatcher.
        """
        # Dispatch table mapping YAML dataset string keys to their respective uninstantiated class objects.
        # This allows dynamic scaling of the pipeline as more data sources are added.
        self._source_map = {
            "chelsa": chelsa_cube,
            "gbif": gbif_cube
        }

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

    def _dump_recipe(self, recipe: Dict[str, Any], meta_dir: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Dumps a frozen copy of the executed recipe dictionary to the metadata directory.
        
        This guarantees reproducibility by preserving the exact configurations used to 
        generate the data cube, even if the original user YAML file is later modified.

        Parameters
        ----------
        recipe : dict
            The parsed configuration dictionary to serialize.
        meta_dir : str
            The target directory where the `recipe.yaml` file will be saved.
        logger : logging.Logger, optional
            Pipeline execution logger.
        """
        os.makedirs(meta_dir, exist_ok=True)
        recipe_dump_path = os.path.join(meta_dir, "recipe.yaml")
        
        # Serialize the dictionary back to YAML cleanly
        with open(recipe_dump_path, "w") as f:
            yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
            
        log_execution(logger, f"Dumped executed recipe to: {recipe_dump_path}", logging.INFO)

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
            os.path.join(cube_dir, "meta", f"{dataset_name}_stac.json"),
            os.path.join(cube_dir, dataset_name, "meta", f"{dataset_name}_stac.json")
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    item = pystac.Item.from_file(path)
                    log_execution(logger, f"Collected STAC Item for '{dataset_name}' from {path}", logging.INFO)
                    return item
                except Exception as e:
                    log_execution(logger, f"Failed to parse STAC Item at {path}: {e}", logging.WARNING)

        return None

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

        # Ensure all internal asset pointers are resolved relatively to maintain catalog portability
        catalog_path = os.path.join(meta_dir, "catalog.json")
        catalog.normalize_hrefs(meta_dir)
        catalog.save(pystac.CatalogType.SELF_CONTAINED)

        log_execution(logger, f"Exported unifying STAC Catalog to: {catalog_path}", logging.INFO)
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
        self._dump_recipe(recipe, meta_dir, logger=logger)
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
                download_info = sql.fetch_gbif_download(gbif_download_key, target_dir=download_dir, creds=creds)
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