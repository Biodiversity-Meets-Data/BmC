import os
import re
from pathlib import Path
import json
import logging
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from rasterio.session import AWSSession

import bmc
from bmc.utils.logger import log_execution
from bmc.utils.meta import fetch_meta
from bmc.cube.spatiotemporal.raster import *

class clmsCube(rasterCube):
    """
    A child class of `rasterCube` designed to process Copernicus Land Monitoring Service 
    (CLMS) datasets from authenticated S3 buckets.

    This class manages the ingestion, authentication, and spatial processing of CLMS 
    data products. It handles multiple discrete catalog files, enforces physical resolution 
    caps native to CLMS products (e.g., preventing 100m source data from being artificially 
    upscaled to 10m), and maps raw integer pixel values to semantic categorical strings 
    using a dedicated class mapping for fractional spatial resampling.

    Attributes
    ----------
    cdse_credentials_path : str
        Filesystem path to the JSON file containing Copernicus Data Space Ecosystem (CDSE) credentials.
    private_credentials_path : str or None
        Filesystem path to the JSON file containing private AWS S3 credentials.
    dataset_name : str
        The normalized string identifier for this dataset engine (resolved to 'clms').
    cloud_session : AWSSession or None
        Placeholder for the active cloud session (dynamically resolved per-request).
    catalog_dir : str
        The root directory path containing the individual CLMS Parquet catalog files.
    catalog_path : str
        Alias for `catalog_dir` used by parent classes.
    catalogs : dict of str to str
        A mapping of CLMS sub-level names (e.g., 'crop_types') to their specific catalog file paths.
    temporally_static_levels : list
        List of levels that lack a time dimension (empty by default for CLMS).
    class_mapping : dict of str to dict
        A nested dictionary mapping variable names to their integer-to-string categorical definitions.
    """
    def __init__(
        self,
        cdse_s3_credentials: str,
        private_s3_credentials: str = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the CLMS dataset engine, clears global AWS environment variables, 
        and sets up the multi-catalog routing matrix.

        Parameters
        ----------
        cdse_s3_credentials : str
            Absolute or relative path to the CDSE authentication JSON file.
        private_s3_credentials : str, optional
            Absolute or relative path to the private S3 authentication JSON file.
        *args
            Variable length argument list passed to the `rasterCube` parent.
        **kwargs
            Arbitrary keyword arguments passed to the `rasterCube` parent.
        """
        super().__init__(*args, **kwargs)

        self.cdse_credentials_path = cdse_s3_credentials
        self.private_credentials_path = private_s3_credentials
        
        # Dynamically extract the dataset name from the class definition (clmsCube -> clms)
        self.dataset_name = self.__class__.__name__.replace("Cube", "").lower()

        # Wipe any lingering global AWS pollution from the environment.
        # This prevents credentials from a different dataset engine (e.g., CHELSA) 
        # from interfering with CLMS's specialized S3 endpoint routing.
        for key in ["AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_VIRTUAL_HOSTING"]:
            os.environ.pop(key, None)
            
        # CLMS pulls from both 'eodata' (CDSE) and private buckets, requiring dynamic per-request sessions
        self.cloud_session = None

        # Anchor to the repo root using pathlib to ensure robust cross-platform path resolution
        repo_root = Path(bmc.__file__).parents[2]
        self.catalog_dir = os.path.join(repo_root, "meta", self.dataset_name)
        self.catalog_path = self.catalog_dir

        # Explicitly map all individual CLMS catalogs.
        # CLMS is modular, so each product family gets its own Parquet file for faster I/O.
        self.catalogs = {
            "clcplus": os.path.join(self.catalog_dir, "clcplus_instance_catalog.parquet"),
            "crop_types": os.path.join(self.catalog_dir, "crop_types_catalog.parquet"),
            "cropping_patterns": os.path.join(self.catalog_dir, "cropping_patterns_catalog.parquet"),
            "dominant_leaf_type": os.path.join(self.catalog_dir, "dominant_leaf_type_catalog.parquet"),
            "forest_type": os.path.join(self.catalog_dir, "forest_type_catalog.parquet"),
            "grassland_herbaceous": os.path.join(self.catalog_dir, "grassland_and_herbaceous_catalog.parquet"),
            "grassland_mowing": os.path.join(self.catalog_dir, "grassland_mowing_events_catalog.parquet"),
            "tree_cover_density": os.path.join(self.catalog_dir, "tree_cover_density_catalog.parquet"),
        }

        # Route CORINE legacy catalogs based on available credentials.
        # If private credentials are provided, use the remote catalog; otherwise, fallback to local.
        if private_s3_credentials is None:
            self.catalogs["clc"] = os.path.join(self.catalog_dir, "clc_catalog_local.parquet")
        else:
            self.catalogs["clc"] = os.path.join(self.catalog_dir, "clc_catalog.parquet")

        self.temporally_static_levels = []

        # Central registry mapping integer raster pixel values to their semantic classifications.
        # Required for CF-Compliant NetCDF export and the categorical fractional warping pipeline.
        self.class_mapping = {
        'CPFLP': {
            0: 'No fallow land', 
            1: 'Fallow', 
            65535: 'Outside area'
        },
        'CPSCT': {
            0: 'No annual cropland',
            1: 'Short summer CC',
            2: 'Long summer CC',
            3: 'Short winter CC',
            4: 'Long winter CC',
            65526: '65526',
            65527: '65527',
            65530: '65530',
            65531: '65531',
            65532: '65532',
            65533: '65533',
            65534: '65534',
            65535: 'Outside area'
        },
        'CTY': {
            0: 'No Cropland',
            1110: 'Wheat',
            1120: 'Barley',
            1130: 'Maize',
            1140: 'Rice',
            1150: 'Other Cereals',
            1210: 'Fresh Vegetables',
            1220: 'Dry Pulses',
            1310: 'Potatoes',
            1320: 'Sugar Beet',
            1410: 'Sunflower',
            1420: 'Soybeans',
            1430: 'Rapeseed',
            1440: 'Flax cotton and hemp',
            2100: 'Grapes',
            2200: 'Olives',
            2310: 'Fruits',
            2320: 'Nuts',
            3100: 'Unclassified arable crop',
            3200: 'Unclassified permanent crop',
            65535: 'Outside area'
        },
        'DLT': {
            0: 'all non-tree covered areas',
            1: 'broadleaved trees',
            2: 'coniferous trees',
            255: 'outside area'
        },
        'DLTC': {
            0: 'unchanged areas with no tree cover',
            1: 'new broadleaved cover',
            2: 'new coniferous cover',
            3: 'loss of broadleaved cover',
            4: 'loss of coniferous cover',
            10: 'unchanged areas with tree cover',
            255: 'outside area'
        },
        'FTY': {
            0: 'all non-forest areas',
            1: 'broadleaved forest',
            2: 'coniferous forest',
            255: 'outside area'
        },
        'GRA': {
            0: 'all non-grassland areas in reference year',
            1: 'grassland',
            255: 'outside area'
        },
        'HER': {
            0: 'non-grassland in reference year',
            1: 'permanent and temporary grassland in reference year',
            255: 'outside area'
        },
        'LULUCF-INSTANCE': {
            11: 'S Burnt areas',
            12: 'S Settlements',
            13: 'S Green urban areas',
            14: 'S Other settlements',
            21: 'FL Burnt areas',
            22: 'FL Transitional woodland',
            23: 'FL Deciduous',
            24: 'FL Coniferous/Evergreen',
            25: 'FL Other forestland',
            31: 'CL Burnt areas',
            32: 'CL Annual crops',
            33: 'CL Perennial crops',
            34: 'CL Other cropland',
            41: 'GL Burnt areas',
            42: 'GL Pastures',
            43: 'GL Shrubs',
            44: 'GL Natural grassland',
            45: 'GL Other grassland',
            51: 'WL Wetland managed',
            52: 'WL Wetland unmanaged',
            53: 'WL Water managed',
            54: 'WL Water unmanaged',
            55: 'WL Burnt areas',
            61: 'OL Bare soil and rocks',
            62: 'OL Permanent snow and ice',
            63: 'OL Lichens and mosses',
            64: 'OL Other land',
            254: 'Unclassified',
            255: 'No data'
        },
        'PLOUGH': {
            0: 'Indication of ploughing in current year',
            1: '1 year since last indication of ploughing',
            2: '2 years since last indication of ploughing',
            3: '3 years since last indication of ploughing',
            4: '4 years since last indication of ploughing',
            5: '5 years since last indication of ploughing',
            6: '6 years since last indication of ploughing',
            100: 'Change of herbaceous cover',
            253: 'no ploughing information',
            255: 'outside area'
        },
        'landcover': {
            1: 'Continuous urban fabric',
            2: 'Discontinuous urban fabric',
            3: 'Industrial or commercial units',
            4: 'Road and rail networks and associated land',
            5: 'Port areas',
            6: 'Airports',
            7: 'Mineral extraction sites',
            8: 'Dump sites',
            9: 'Construction sites',
            10: 'Green urban areas',
            11: 'Sport and leisure facilities',
            12: 'Non-irrigated arable land',
            13: 'Permanently irrigated land',
            14: 'Rice fields',
            15: 'Vineyards',
            16: 'Fruit trees and berry plantations',
            17: 'Olive groves',
            18: 'Pastures',
            19: 'Annual crops associated with permanent crops',
            20: 'Complex cultivation patterns',
            21: 'Land principally occupied by agriculture with significant areas of natural vegetation',
            22: 'Agro-forestry areas',
            23: 'Broad-leaved forest',
            24: 'Coniferous forest',
            25: 'Mixed forest',
            26: 'Natural grasslands',
            27: 'Moors and heathland',
            28: 'Sclerophyllous vegetation',
            29: 'Transitional woodland-shrub',
            30: 'Beaches - dunes - sands',
            31: 'Bare rocks',
            32: 'Sparsely vegetated areas',
            33: 'Burnt areas',
            34: 'Glaciers and perpetual snow',
            35: 'Inland marshes',
            36: 'Peat bogs',
            37: 'Salt marshes',
            38: 'Salines',
            39: 'Intertidal flats',
            40: 'Water courses',
            41: 'Water bodies',
            42: 'Coastal lagoons',
            43: 'Estuaries',
            44: 'Sea and ocean',
            48: 'NODATA'
        }
    }
        
    def resolve_provider(self, paths: list) -> Optional[str]:
        """
        Determines the appropriate cloud provider ID from a list of URIs.

        Analyzes the target URL hostname or bucket signature to route the request 
        to the correct credential payload (CDSE vs Private AWS).

        Parameters
        ----------
        paths : list of str
            List of VSI paths or URIs pointing to remote assets.

        Returns
        -------
        str or None
            The provider ID string ('cdse' or 'private'), or None if unmatched.
        """
        if not paths:
            return None
            
        first_path = str(paths[0])
        
        # 'eodata' is the standard bucket identifier for the Copernicus Data Space Ecosystem
        if "eodata" in first_path:
            return "cdse"
        # BII custom internal staging/production bucket
        elif "bii-bmd-workflow-engine-production" in first_path:
            return "private"
            
        return None

    def get_cloud_session(self, provider_id: str) -> Tuple[Any, Dict[str, str]]:
        """
        Dynamically generates an isolated AWSSession and specific GDAL environment 
        keyword arguments (kwargs) for a designated provider.

        This ensures that non-standard S3-compatible endpoints (like CDSE's eodata 
        bucket) receive the exact HTTP headers and endpoint overrides they require 
        without polluting the global GDAL configuration environment.

        Parameters
        ----------
        provider_id : str
            The internal identifier of the target provider ('private' or 'cdse').

        Returns
        -------
        Tuple[AWSSession or None, Dict[str, str]]
            A tuple containing the authenticated Boto3/Rasterio cloud session object, 
            and a dictionary of critical environment variables to be passed to `rasterio.Env`.
        """
        from rasterio.session import AWSSession
        import json
        import os
        
        if provider_id == "private" and self.private_credentials_path:
            expanded_path = os.path.expanduser(self.private_credentials_path)
            if os.path.exists(expanded_path):
                with open(expanded_path) as f:
                    priv = json.load(f)
                
                # Native AWS Bucket: Omit endpoint_url, provide only standard region
                session = AWSSession(
                    aws_access_key_id=priv.get("s3_access_key", priv.get("aws_access_key_id")),
                    aws_secret_access_key=priv.get("s3_secret_key", priv.get("aws_secret_access_key")),
                    region_name=priv.get("region_name", "eu-north-1")
                )
                return session, {}  # No explicit GDAL kwargs needed for native AWS infrastructure
            
        if provider_id == "cdse" and self.cdse_credentials_path:
            expanded_path = os.path.expanduser(self.cdse_credentials_path)
            if os.path.exists(expanded_path):
                with open(expanded_path) as f:
                    cdse = json.load(f)
                
                # Custom Copernicus Bucket: Requires the endpoint protocol to be stripped 
                # so it can be injected directly into GDAL's internal C++ S3 driver configuration.
                raw_endpoint = cdse.get("s3_endpoint", "https://eodata.dataspace.copernicus.eu")
                clean_endpoint = raw_endpoint.replace("https://", "").replace("http://", "").rstrip("/")
                
                session = AWSSession(
                    aws_access_key_id=cdse.get("s3_access_key", cdse.get("aws_access_key_id")),
                    aws_secret_access_key=cdse.get("s3_secret_key", cdse.get("aws_secret_access_key"))
                )
                env_kwargs = {
                    "AWS_S3_ENDPOINT": clean_endpoint,
                    "AWS_VIRTUAL_HOSTING": "FALSE",  # CDSE eodata uses path-style access, not virtual-hosted
                    "AWS_HTTPS": "YES"
                }
                return session, env_kwargs
            
        return None, {}

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def resolve_target_grid(
        self, 
        spatial_cfg: dict, 
        logger: logging.Logger = None,
        variables: list = None
    ) -> str:
        """
        Resolves the user's requested target grid by standardizing aliases, 
        and enforces physical resolution caps based on the native source datasets.

        Certain datasets within CLMS are generated at differing base resolutions 
        (e.g., standard HR layers are 10m, but LULUCF CLC+ instances are 100m). 
        This method prevents the orchestrator from attempting to "upscale" coarse data 
        beyond its true physical resolution limits.

        Parameters
        ----------
        spatial_cfg : dict
            The spatial configuration dictionary block from the YAML recipe.
        logger : logging.Logger, optional
            Execution logger for validation telemetry.
        variables : list, optional
            List of requested variable strings. Used to determine variable-specific caps.

        Returns
        -------
        str
            The strictly validated grid key matching an entry in `GRID_REGISTRY`.

        Raises
        ------
        ValueError
            If the requested grid base/resolution combination does not exist in the registry.
        """
        if variables is None:
            variables = []
            
        if logger:
            logger.info("Resolving target grid base for CLMS data cube generation...")
            
        registry = getattr(self, 'GRID_REGISTRY', {})
        
        # 0. Early Exit: Exact match provided natively by the user (e.g., 'EEA_1km')
        exact_candidate = str(spatial_cfg.get('target_grid', ''))
        if exact_candidate in registry:
            if logger:
                logger.info(f"Successfully mapped to exact registry grid '{exact_candidate}'.")
            return exact_candidate

        # 1. Base Grid Standardization via Dictionary Mapping
        # Standardize common aliases (e.g. "GEA" -> "Global_EqualArea")
        raw_grid = str(spatial_cfg.get('target_grid', 'EEA')).upper()
        raw_res = str(spatial_cfg.get('target_resolution', '10m'))
        
        grid_aliases = {
            "EEA": "EEA",
            "GEA": "Global_EqualArea",
            "GLOBAL_EQUALAREA": "Global_EqualArea",
            "GLOBAL_EQUAL_AREA": "Global_EqualArea",
            "WGS84": "Global_WGS84",
            "GLOBAL_WGS84": "Global_WGS84"
        }
        
        base_grid = grid_aliases.get(raw_grid, raw_grid)
        
        # 2. Reconstruct the combined grid key
        grid_key = f"{base_grid}_{raw_res}"
        
        if grid_key not in registry:
            raise ValueError(
                f"Target grid '{grid_key}' not found in GRID_REGISTRY. "
                f"Available grids: {list(registry.keys())}"
            )
            
        target_res = registry[grid_key]["resolution"]
        crs = registry[grid_key]["crs"]
        
        # 3. Determine the native resolution cap based on requested CLMS variables
        is_degree = (crs == "EPSG:4326")
        
        # Default CLMS High Resolution Layer base resolution cap is 10 meters
        base_limit = 0.00008333333333333333 if is_degree else 10.0
        
        # The 'clcplus' (LULUCF-INSTANCE) product has a natively coarser base resolution of 100m.
        # If requested alongside HR layers, the entire output grid must be capped at 100m minimum.
        if "LULUCF-INSTANCE" in variables:
            base_limit = 0.0008333333333333333 if is_degree else 100.0
            
        # 4. Enforce the cap: if requested grid cell size is physically smaller than the base_limit, downgrade it
        if target_res < (base_limit * 0.99):
            if logger:
                logger.warning(
                    f"Requested grid '{grid_key}' ({target_res}) is finer than the "
                    f"source data cap ({base_limit}). Downgrading to the finest allowable source limit."
                )
            
            # Filter the registry for grids in the exact same projection that satisfy the physical base limit
            valid_grids = {
                k: v for k, v in registry.items() 
                if v["crs"] == crs and v["resolution"] >= (base_limit * 0.99)
            }
            
            # Sort by resolution and pick the finest remaining valid grid
            grid_key = sorted(valid_grids.items(), key=lambda item: item[1]["resolution"])[0][0]
            
        if logger:
            logger.info(f"Resolved target grid to: {grid_key}")
            
        return grid_key

    def get_resample_rule(self, variable_name: str) -> str:
        """
        Determines the appropriate spatial resampling algorithm for a variable.

        Examines the requested variable against the engine's internal `class_mapping`. 
        If the variable is strictly categorical (e.g., Crop Type), it routes execution 
        to the specialized fractional coverage warping engine. Otherwise, it defaults 
        to standard continuous interpolation.

        Parameters
        ----------
        variable_name : str
            The name of the variable being processed.

        Returns
        -------
        str
            The resampling rule string ('class_fraction' for categorical datasets, 
            or 'bilinear' for continuous datasets).
        """
        # Strip potential temporal or string suffixes if the upstream pipeline appends them
        base_var = variable_name.split('_20')[0] if '_20' in variable_name else variable_name
        
        # Route strictly mapped categorical arrays (e.g., CTY, FTY) to the fractional engine
        if hasattr(self, 'class_mapping') and base_var in self.class_mapping:
            return "class_fraction"
            
        # Default to continuous bilinear resampling for confidence/density layers (e.g., Tree Cover Density)
        return "bilinear"

    def get_auxiliary_dimensions(self, recipe: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Retrieves dataset-specific auxiliary dimension configurations from the recipe.

        Parameters
        ----------
        recipe : dict
            The master pipeline configuration YAML dictionary.

        Returns
        -------
        dict
            A dictionary of auxiliary dimensions mapped to their configured lists of allowable values.
        """
        dataset_cfg = recipe.get('sources', {}).get(getattr(self, 'dataset_name', 'clms'), {})
        return dataset_cfg.get("auxiliary_dimensions", {})