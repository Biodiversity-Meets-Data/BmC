import os
import logging
from typing import Dict, Any, Tuple, Optional
from bmc.utils.logger import log_execution, ResourceProfiler

class dataCube():
    """
    Parent base class for all spatiotemporal data cubes (Raster and Vector).
    
    Centralizes logging creation, recipe configuration parsing, directory structure 
    allocation, and resource profiling across processing engines.
    
    Methods
    -------
    _setup_pipeline_logger(logger_name, log_filepath, level=logging.INFO)
        Instantiates and configures a dedicated dual-stream logger for a specific cube run.
    initialize_pipeline(recipe, dataset_name=None, logger=None)
        Parses initial recipe metadata, provisions output directories, and boots logging/profiling.
    """
    def __init__(self):
        # Standard execution configuration stored directly in the base class
        self.execution_config = {
            "min_slices_for_parallel": 16,
            "max_safe_threadpool_workers": 8,
            "large_slice_threshold_mb": 50.0,
            "gdal_warp_block_size": 2048,
            "warp_cache_cleanup_delay_sec": 30
        }

    def update_config_from_recipe(self, recipe: Dict[str, Any], logger: logging.Logger):
        """
        Scans the recipe for an execution component. Updates class options if found,
        otherwise logs the standard configuration being used.
        """
        exec_block = recipe.get("execution", {})
        
        # Filter for recognized execution keys to avoid polluting the config dictionary
        overrides = {k: v for k, v in exec_block.items() if k in self.execution_config}
        
        if overrides:
            self.execution_config.update(overrides)
            log_execution(
                logger, 
                f"Alternative execution component detected. Updated configuration with: {overrides}", 
                logging.INFO
            )
        else:
            log_execution(
                logger, 
                f"No alternative execution component present in recipe. Utilizing standard execution configuration: {self.execution_config}", 
                logging.INFO
            )

    def _setup_pipeline_logger(
        self, 
        logger_name: str, 
        log_filepath: str, 
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Instantiates and configures a dedicated logger for a specific cube run.

        Parameters
        ----------
        logger_name : str
            The internal identifier for the logger instance.
        log_filepath : str
            The absolute or relative path where the log file will be saved.
        level : int, optional
            The logging severity threshold. Default is ``logging.INFO``.

        Returns
        -------
        logging.Logger
            A configured standard Python Logger instance containing both 
            File and Stream (Console) handlers.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Prevent duplicate handlers if the logger already exists in memory
        if not logger.handlers:
            os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
            
            # 1. File Handler Setup: Appends to log file safely
            file_handler = logging.FileHandler(log_filepath, mode='a')
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # 2. Console Handler Setup: Streams output to stdout
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def initialize_pipeline(
        self, 
        recipe: Dict[str, Any], 
        dataset_name: Optional[str] = None, 
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Dict[str, Any], logging.Logger, ResourceProfiler]:
        """
        Parses initial recipe metadata, provisions output directories, and boots logging/profiling.

        Parameters
        ----------
        recipe : dict
            The loaded YAML execution plan governing spatial domains and datasets.
        dataset_name : str, optional
            Explicit datasource name. Fallback defaults to class name or recipe specification.
        logger : logging.Logger, optional
            Pre-existing logger instance. If None, a dataset-specific logger is initialized.

        Returns
        -------
        tuple
            A 3-element tuple containing:
            - ``ctx`` (Dict[str, Any]): Normalized pipeline context variables (directories, config).
            - ``logger`` (logging.Logger): The active tracking logger.
            - ``tracker`` (ResourceProfiler): Hardware telemetry tracker instance.
        """
        # 1. Extract foundational directory paths from recipe
        paths_cfg = recipe.get('paths', {})
        base_dir = paths_cfg.get('base_dir') or recipe.get('base_dir', './cubing_output/')
        cube_name = recipe.get('cube_name', 'bmd_default_cube')
        
        # 2. Dynamically resolve the dataset name for folder structures
        if not dataset_name:
            dataset_name = recipe.get('dataset_name') or self.__class__.__name__.lower().replace('_cube', '')

        # 3. Construct nested directory topology
        cube_dir = os.path.join(base_dir, cube_name)
        log_dir = os.path.join(cube_dir, 'logs')
        dataset_dir = os.path.join(cube_dir, dataset_name)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # 4. Initialize per-datasource logger if not provided by orchestrator
        if logger is None:
            log_filename = f"{cube_name}_{dataset_name}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            logger = self._setup_pipeline_logger(
                logger_name=f"{cube_name}_{dataset_name}", 
                log_filepath=log_filepath
            )

        self.logger = logger
        tracker = ResourceProfiler(log_dir=log_dir)

        # 5. Scan and Update Execution Configurations
        self.update_config_from_recipe(recipe, logger)

        # 6. Standardized context block generation for downstream consumption
        ctx = {
            "base_dir": base_dir,
            "cube_name": cube_name,
            "dataset_name": dataset_name,
            "cube_dir": cube_dir,
            "dataset_dir": dataset_dir,
            "log_dir": log_dir,
            "spatial_cfg": recipe.get('spatial', {}),
            "export_format": recipe.get('export_as', {}).get('format', 'netcdf').lower()
        }

        log_execution(logger, f"\n=== Initiating Data Cube Pipeline: '{cube_name}' | Dataset: '{dataset_name}' ===", logging.INFO)
        log_execution(logger, f"  Cube Directory   : {cube_dir}", logging.INFO)
        log_execution(logger, f"  Output Directory : {dataset_dir}", logging.INFO)
        log_execution(logger, f"  Log File         : {os.path.join(log_dir, f'{cube_name}_{dataset_name}.log')}", logging.INFO)

        return ctx, logger, tracker