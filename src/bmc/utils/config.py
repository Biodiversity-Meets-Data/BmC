import os
import yaml
import logging
from typing import Dict, Any, Optional
from bmc.utils.logger import log_execution

# Establish a fallback module-level logger for this specific configuration file
module_logger = logging.getLogger(__name__)

def read_recipe(yaml_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Safely loads, resolves, and validates a YAML execution recipe.

    This utility parses a configuration text file into a native Python dictionary,
    enforces structural schema requirements, and translates relative path paths
    into absolute system paths to prevent downstream tracking failures.

    Parameters
    ----------
    yaml_path : str
        The absolute or relative file system path to the target .yaml recipe file.
    logger : logging.Logger, optional
        An active Python logger instance to capture validation events. If None, 
        fallback logging will default to the file's internal module-level logger.
        Default is None.

    Returns
    -------
    dict
        The fully loaded, verified, and path-resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified YAML configuration file does not exist on disk.
    ValueError
        If critical top-level keys ('base_dir' or 'spatial') are missing from the file.
    """
    # Fallback to the module logger instance if no external runtime logger was passed down
    active_logger = logger or module_logger

    # Log the initialization of the configuration file extraction sequence
    log_execution(active_logger, f"Attempting to load configuration recipe from path: {yaml_path}", logging.INFO)
    
    # Verify the physical existence of the file path before opening a stream
    if not os.path.exists(yaml_path):
        # Log a critical failure message to assist with remote debugging traces
        log_execution(active_logger, f"Configuration recipe file could not be found at: {yaml_path}", logging.CRITICAL)
        # Raise an explicit exception to halt initialization immediately
        raise FileNotFoundError(f"Recipe file not found: {yaml_path}")

    # Secure an open text read stream utilizing an explicit context manager block
    with open(yaml_path, 'r') as file:
        try:
            # Convert raw text configuration parameters into a structured Python dictionary
            recipe = yaml.safe_load(file)
        except yaml.YAMLError as e:
            # Capture formatting syntax violations such as illegal tabs or key indentations
            log_execution(active_logger, f"Failed to parse target file format due to syntax error: {e}", logging.CRITICAL)
            # Re-raise the structural parsing error to inform the master orchestration layer
            raise

    # Verify that the fundamental data tracking destination directory is defined
    if 'base_dir' not in recipe:
        # Abort processing if the core directory parameters are missing
        raise ValueError("Invalid Recipe Configuration: Missing critical top-level key 'base_dir'.")

    # Verify that the geographic coordinate constraints block is defined
    if 'spatial' not in recipe:
        # Abort processing if the geographic bounding coordinates are completely missing
        raise ValueError("Invalid Recipe Configuration: Missing critical top-level key 'spatial'.")

    # Isolate the user-defined output directory string, defaulting to standard locations if empty
    raw_base_path = recipe.get('base_dir', './outputs/')
    
    # Resolve the provided path string into a strict absolute reference to secure file writes
    recipe['base_dir'] = os.path.abspath(raw_base_path)

    # Log the successful compilation and schema validation of the target blueprint layout
    log_execution(active_logger, "Configuration recipe parsed, verified, and normalized successfully.", logging.INFO)
    
    # Hand the finalized, safe dictionary back to the processing pipeline loop
    return recipe
