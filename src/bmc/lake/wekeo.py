import os
import json
import re
import logging
import urllib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import gc

os.environ["GDAL_CACHEMAX"] = "1024"      # Give GDAL 1GB of RAM for caching
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS" # Unleash multi-threading
os.environ["VSI_CACHE"] = "TRUE"          # Optimize virtual file reading
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR" # Speeds up file discovery

import pandas as pd
import xarray as xr
import rioxarray

import hda 

from bmc.lake.spatiotemporal import spatiotemporal_lake
from bmc.utils.logger import log_execution
from bmc.utils.meta import fetch_meta
from bmc.datasource.wekeo.interface import generate_wekeo_query, get_dataset_variables, build_data_inventory

class wekeo_lake(spatiotemporal_lake):
    """
    Offline ingestion engine for building WEkEO (Copernicus DIAS) data lakes.

    Inherits from `spatiotemporal_lake`. This class connects to the WEkEO HDA API, 
    translates abstract recipes into queries, downloads raw zip archives, mosaicks 
    them into VRTs, and uses the spatial engine to bake them into continuous COGs 
    or fractional categorical layers. Finally, it validates the physics and generates 
    a runtime catalog.
    """
    
    DATASET_MAP = {
        "TCF": "EO:EEA:DAT:HRL:TCF",
        "GRA": "EO:EEA:DAT:HRL:GRA",
        "IMP": "EO:EEA:DAT:HRL:IMP",
        "SLF": "EO:EEA:DAT:HRL:SLF",
        "CRL": "EO:EEA:DAT:HRL:CRL", 
        "CORINE": "EO:EEA:DAT:CORINE"
    } 

    _CLC_LEVEL_3 = {
        111: "Continuous urban fabric", 112: "Discontinuous urban fabric",
        121: "Industrial or commercial units", 122: "Road and rail networks and associated land",
        123: "Port areas", 124: "Airports", 131: "Mineral extraction sites",
        132: "Dump sites", 133: "Construction sites", 141: "Green urban areas",
        142: "Sport and leisure facilities",
        211: "Non-irrigated arable land", 212: "Permanently irrigated land",
        213: "Rice fields", 221: "Vineyards", 222: "Fruit trees and berry plantations",
        223: "Olive groves", 231: "Pastures", 241: "Annual crops associated with permanent crops",
        242: "Complex cultivation patterns", 243: "Land principally occupied by agriculture",
        244: "Agro-forestry areas",
        311: "Broad-leaved forest", 312: "Coniferous forest", 313: "Mixed forest",
        321: "Natural grasslands", 322: "Moors and heathland", 323: "Sclerophyllous vegetation",
        324: "Transitional woodland-shrub", 331: "Beaches, dunes, sands",
        332: "Bare rocks", 333: "Sparsely vegetated areas", 334: "Burnt areas",
        335: "Glaciers and perpetual snow",
        411: "Inland marshes", 412: "Peat bogs", 421: "Salt marshes",
        422: "Salines", 423: "Intertidal flats",
        511: "Water courses", 512: "Water bodies", 521: "Coastal lagoons",
        522: "Estuaries", 523: "Sea and ocean",
        48: "Unclassified / Unverifiable", 128: "No data", 255: "Outside area / No data"
    }

    CATEGORICAL_CLASSES = {
        "Crop Types": {
            0: "No crop / Non-agricultural",
            1110: "Wheat", 1120: "Barley", 1130: "Maize", 1140: "Rice", 
            1150: "Other_Cereals", 1210: "Fresh_Vegetables", 1220: "Dry_Pulses", 
            1310: "Potatoes", 1320: "Sugar_Beet", 1410: "Sunflower", 1420: "Soybeans", 
            1430: "Rapeseed", 1440: "Flax_Cotton_Hemp", 2100: "Grapes", 2200: "Olives", 
            2310: "Fruits", 2320: "Nuts", 3100: "Unclassified_Arable", 
            3200: "Unclassified_Permanent", 254: "Unclassifiable", 255: "No data"
        },
        "Secondary Crops Type": { # Re-using Crop Types dict for secondary
            0: "No crop / Non-agricultural",
            1110: "Wheat", 1120: "Barley", 1130: "Maize", 1140: "Rice", 
            1150: "Other_Cereals", 254: "Unclassifiable", 255: "No data"
        },
        "Corine Land Cover 1990": _CLC_LEVEL_3,
        "Corine Land Cover 2000": _CLC_LEVEL_3,
        "Corine Land Cover 2006": _CLC_LEVEL_3,
        "Corine Land Cover 2012": _CLC_LEVEL_3,
        "Corine Land Cover 2018": _CLC_LEVEL_3,
        "Forest Type": {0: "Non-forest areas", 1: "Broadleaved forest", 2: "Coniferous forest", 254: "Unclassifiable", 255: "No data"},
        "Dominant Leaf Type": {0: "All non-tree areas", 1: "Broadleaved", 2: "Coniferous", 254: "Unclassifiable", 255: "No data"},
        "Impervious Built-up": {0: "Non-built-up area", 1: "Built-up area", 254: "Unclassifiable", 255: "No data"},
        "Grassland": {0: "Non-grassland", 1: "Grassland", 254: "Unclassifiable", 255: "No data"},
        "Ploughing Indicator": {0: "No ploughing detected", 1: "Ploughing detected", 254: "Unclassifiable", 255: "No data"},
        "Forest Mask": {0: "Non-forest", 1: "Forest", 254: "Unclassifiable", 255: "No data"},
        "Crop Mask": {0: "Non-crop", 1: "Crop", 254: "Unclassifiable", 255: "No data"},
        "Small Woody Features": {0: "Non-SWF", 1: "Small Woody Feature presence", 254: "Unclassifiable", 255: "No data"},
        "Fallow Land Presence": {0: "No fallow land", 1: "Fallow land present", 254: "Unclassifiable", 255: "No data"}
    }
    
    def __init__(self, hdarc_path: str = "./hdarc"):
        super().__init__()
        self.wekeo_logger = None
        self._load_credentials_to_memory(hdarc_path)

    def _load_credentials_to_memory(self, filepath: str):
        """Reads a local hdarc file and injects it into environment variables."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('url:'):
                        os.environ['HDA_URL'] = line.split('url:')[1].strip()
                    elif line.startswith('user:'):
                        os.environ['HDA_USER'] = line.split('user:')[1].strip()
                    elif line.startswith('password:'):
                        os.environ['HDA_PASSWORD'] = line.split('password:')[1].strip()
            # --- VERIFICATION STEP ---
            # Safely fetch the loaded variables
            loaded_user = os.environ.get('HDA_USER')
            loaded_pwd = os.environ.get('HDA_PASSWORD')
            
            # Create a masked password (e.g., "my_password" becomes "my***rd")
            if loaded_pwd and len(loaded_pwd) > 4:
                masked_pwd = f"{loaded_pwd[:2]}***{loaded_pwd[-2:]}"
            else:
                masked_pwd = "***"

            # Print or log the confirmation
            verification_msg = f"SUCCESS: WEkEO credentials loaded for user '{loaded_user}' (Password: {masked_pwd})"
            
            if self.wekeo_logger:
                self.wekeo_logger.info(verification_msg)
            else:
                print(verification_msg)
                
        else:
            warning_msg = f"WARNING: Could not find hdarc file at {filepath}"
            if self.wekeo_logger:
                self.wekeo_logger.warning(warning_msg)
            else:
                print(warning_msg)

    def fetch_raw_data(self, recipe: Dict[str, Any], logger: logging.Logger) -> Tuple[Any, List[Dict]]:
        """
        Authenticates the HDA client and builds the API execution queue.
        """
        log_execution(logger, "Authenticating WEkEO API and building query queue...", logging.INFO)
        
        # Initialize WEkEO client (Mocked here - replace with your actual hda.Client init)
        wekeo_client = hda.Client()
        
        metadata_schema = dict(zip(
            [key for key in self.DATASET_MAP.keys()],
            [get_dataset_variables(self.DATASET_MAP[key], wekeo_client)["constraints"][0] for key in self.DATASET_MAP.keys()]
        ))

        # Re-using your exact intersection logic
        execution_queue = self.intersect_config_with_dataframe(
            recipe=recipe, 
            wekeo_client=wekeo_client, 
            api_schema=metadata_schema, 
            logger=logger
        )
        return wekeo_client, execution_queue

    def build_datalake(self, recipe: Dict[str, Any], logger: logging.Logger) -> List[str]:
        """
        Executes the queue: Downloads zips, extracts TIFs, builds VRTs, and bakes COGs.
        """
        log_execution(logger, "Running pipeline in synchronous sequential mode (GDAL C++ Threading enabled).", logging.INFO)

        raw_dir = recipe.get("paths", {}).get("raw_dir", "./raw_wekeo")
        base_dir = recipe.get("paths", {}).get("base_dir", "./wekeo_lake")
        lake_name = recipe.get("lake_name", "defaultWekeoLake")

        if logger is None:
            log_dir = os.path.join(base_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            logger = self._setup_pipeline_logger(logger_name=lake_name, log_filepath=os.path.join(log_dir, f'{lake_name}_generation.log'))
            self.logger = logger

        wekeo_config = recipe.get("sources", {}).get("wekeo", {})
        if not wekeo_config.get("enabled", False):
            return []

        target_grid_key = recipe.get("spatial", {}).get("target_grid_key", "EEA_100m")
        resampling_config = recipe.get("spatial", {}).get("resampling_strategies", {})
        target_grid, target_res = target_grid_key.split("_", 1) if "_" in target_grid_key else (target_grid_key, "unknown")

        wekeo_client, execution_queue = self.fetch_raw_data(recipe, logger)
        if not execution_queue:
            return []

        generated_cogs = []

        for query in execution_queue:
            product_type = query.get("productType", "Unknown_Product")
            clean_product = product_type.replace(' ', '_').replace('/', '_')
            year = query.get("year", "AllYears")
            dataset_cat = query.get("dataset_id", "").split(":")[-1]
            safe_name = f"{clean_product}_{year}"
            
            log_execution(logger, f"\n{'='*50}\nBuilding Lake Layer: {safe_name}\n{'='*50}", logging.INFO)
            isolated_raw_dir = os.path.join(raw_dir, "wekeo", query.get("dataset_id", "").replace(":", "_"), clean_product)
            os.makedirs(isolated_raw_dir, exist_ok=True)
            
            download_dir = self._fetch_unpack_query(query, wekeo_client, isolated_raw_dir, logger)
            if not download_dir: continue

            tif_folder = os.path.join(download_dir, "tif_files")
            vrt_path = os.path.join(tif_folder, f"{safe_name}.vrt")
            if not self.build_virtual_mosaic(tif_folder, vrt_path, logger): continue

            if product_type in self.CATEGORICAL_CLASSES:
                if resampling_config.get('discrete', 'coverage') == 'coverage':
                    layer_dir = os.path.join(base_dir, dataset_cat, clean_product, "coverage")
                    os.makedirs(layer_dir, exist_ok=True)
                    
                    raw_class_dict = self.CATEGORICAL_CLASSES[product_type]
                    class_mapping = {k: v for k, v in raw_class_dict.items() if k not in [48, 128, 254, 255]}
                    
                    self.process_virtual_mosaic(
                        vrt_path=vrt_path, strategy='coverage', grid_name=target_grid_key,
                        output_dir_or_file=layer_dir, logger=logger, class_values=list(class_mapping.keys()),
                        class_mapping=class_mapping, file_prefix=f"{clean_product}_{year}_{target_grid}_{target_res}"
                    )
                    gc.collect()
            else:
                cont_stats = resampling_config.get('continuous', ['average'])
                for stat in cont_stats if isinstance(cont_stats, list) else [cont_stats]:
                    layer_dir = os.path.join(base_dir, dataset_cat, clean_product, stat.lower())
                    os.makedirs(layer_dir, exist_ok=True)
                    
                    file_name = f"{clean_product}_{year}_{target_grid}_{target_res}_{stat.lower()}.tif"
                    final_cog_path = os.path.join(layer_dir, file_name)
                    temp_tif_path = os.path.join(layer_dir, f"temp_{file_name}")
                    
                    result_xr = self.process_virtual_mosaic(vrt_path=vrt_path, strategy='reproject', grid_name=target_grid_key, output_dir_or_file=temp_tif_path, logger=logger, resample_keyword=stat)
                    self.export_to_cog(result_xr, final_cog_path, logger=logger)
                    generated_cogs.append(final_cog_path)
                    
                    result_xr.close()
                    del result_xr
                    if os.path.exists(temp_tif_path): os.remove(temp_tif_path)
                    gc.collect()

        self.cleanup_raw_storage(recipe=recipe, logger=logger)
        return generated_cogs
    
    def validate_datalake(
    self, 
    base_dir: str, 
    tolerance: float = 0.0001, 
    logger: Optional[logging.Logger] = None
) -> bool:
        """
        Dynamically crawls a structured data lake and performs strict mathematical 
        and ecological QA validations on all detected continuous and categorical datasets.
        """
        base_path = Path(base_dir)
        log_execution(logger, f"\n=== Initiating Global QA Sweep on: {base_path.absolute()} ===", logging.INFO)
        
        if not base_path.exists():
            log_execution(logger, f"[!] Target directory does not exist: {base_dir}", logging.ERROR)
            return False

        all_passed = True
        lake_catalog = {}
        
        for tif_path in base_path.rglob("*.tif"):
            parts = tif_path.parts
            
            # Anchor to the data lake root folder name to index subdirectories reliably
            if base_path.name in parts:
                base_idx = parts.index(base_path.name)
                if len(parts) - base_idx < 4: 
                    continue
                
                category = parts[base_idx + 1]     # e.g., 'HRL' or 'TCF'
                product = parts[base_idx + 2]      # e.g., 'Tree_Cover_Density'
                aggregation = parts[base_idx + 3]  # e.g., 'average', 'min', 'coverage'
            else:
                if len(parts) < 4: continue
                category, product, aggregation = parts[-4], parts[-3], parts[-2]
            
            year_match = re.search(r'(19\d{2}|20\d{2})', tif_path.name)
            year = int(year_match.group(1)) if year_match else "AllYears"
            
            if category not in lake_catalog: lake_catalog[category] = {}
            if product not in lake_catalog[category]: lake_catalog[category][product] = {}
            if year not in lake_catalog[category][product]: lake_catalog[category][product][year] = {}
            
            if aggregation == 'coverage':
                if 'coverage' not in lake_catalog[category][product][year]:
                    lake_catalog[category][product][year]['coverage'] = []
                lake_catalog[category][product][year]['coverage'].append(tif_path)
            else:
                lake_catalog[category][product][year][aggregation] = tif_path

        def load_da(path):
            return rioxarray.open_rasterio(path, masked=True, chunks={'x': 2048, 'y': 2048}).squeeze()

        # Execute Validations
        for category, products in lake_catalog.items():
            for product, years in products.items():
                for year, layers in years.items():
                    log_execution(logger, f"\n--- Validating: {category} | {product} | {year} ---", logging.INFO)
                    
                    # ==========================================
                    # TEST A: Categorical Fractional Boundaries & Sums
                    # ==========================================
                    if 'coverage' in layers:
                        fraction_das = []
                        for frac_path in layers['coverage']:
                            da = load_da(frac_path)
                            fraction_das.append(da)
                            class_name = frac_path.stem.split(f"{year}_")[-1] 
                            
                            under_0 = (da < -tolerance).sum().compute().item()
                            over_1 = (da > (1.0 + tolerance)).sum().compute().item()
                            
                            if under_0 == 0 and over_1 == 0:
                                log_execution(logger, f"    -> {class_name} Bounds [0,1]: [PASSED]", logging.INFO)
                            else:
                                log_execution(logger, f"    -> {class_name} Bounds [0,1]: [FAILED] ({under_0} < 0, {over_1} > 1)", logging.WARNING)
                                all_passed = False
                        
                        if len(fraction_das) > 1:
                            total_coverage = sum([da.fillna(0) for da in fraction_das])
                            valid_mask = sum([da.notnull() for da in fraction_das]) > 0
                            over_100 = ((total_coverage > (1.0 + tolerance)) & valid_mask).sum().compute().item()
                            
                            if over_100 == 0:
                                log_execution(logger, f"    -> Aggregate Sum <= 100%: [PASSED]", logging.INFO)
                            else:
                                log_execution(logger, f"    -> Aggregate Sum <= 100%: [FAILED] ({over_100} pixels violated)", logging.WARNING)
                                all_passed = False

                    # ==========================================
                    # TEST B: Continuous Mathematical Inequalities
                    # ==========================================
                    else:
                        da_min = load_da(layers.get('min')) if 'min' in layers else None
                        da_avg = load_da(layers.get('average')) if 'average' in layers else None
                        da_rms = load_da(layers.get('rms')) if 'rms' in layers else None
                        da_max = load_da(layers.get('max')) if 'max' in layers else None
                        
                        if da_min is not None and da_avg is not None:
                            min_gt_avg = (da_min > (da_avg + tolerance)).sum().compute().item()
                            if min_gt_avg == 0:
                                log_execution(logger, "    -> Math Check (Min <= Avg): [PASSED]", logging.INFO)
                            else:
                                log_execution(logger, f"    -> Math Check (Min <= Avg): [FAILED] ({min_gt_avg} violations)", logging.WARNING)
                                all_passed = False
                        
                        if da_avg is not None and da_rms is not None:
                            avg_gt_rms = (da_avg > (da_rms + tolerance)).sum().compute().item()
                            if avg_gt_rms == 0:
                                log_execution(logger, "    -> Math Check (Avg <= RMS): [PASSED]", logging.INFO)
                            else:
                                log_execution(logger, f"    -> Math Check (Avg <= RMS): [FAILED] ({avg_gt_rms} violations)", logging.WARNING)
                                all_passed = False

                        if da_rms is not None and da_max is not None:
                            rms_gt_max = (da_rms > (da_max + tolerance)).sum().compute().item()
                            if rms_gt_max == 0:
                                log_execution(logger, "    -> Math Check (RMS <= Max): [PASSED]", logging.INFO)
                            else:
                                log_execution(logger, f"    -> Math Check (RMS <= Max): [FAILED] ({rms_gt_max} violations)", logging.WARNING)
                                all_passed = False

        # ==========================================
        # TEST C: Global Cross-Layer Ecological Logic
        # ==========================================
        log_execution(logger, "\n--- Running Cross-Layer Dependencies ---", logging.INFO)
        
        if 'TCF' in lake_catalog and 'Tree_Cover_Density' in lake_catalog['TCF'] and 'Forest_Type' in lake_catalog['TCF']:
            for year in lake_catalog['TCF']['Tree_Cover_Density'].keys():
                if year in lake_catalog['TCF']['Forest_Type']:
                    max_tcd_path = lake_catalog['TCF']['Tree_Cover_Density'][year].get('max')
                    forest_coverages = lake_catalog['TCF']['Forest_Type'][year].get('coverage', [])
                    
                    if max_tcd_path and forest_coverages:
                        tcd_max = load_da(max_tcd_path)
                        total_forest = sum([load_da(p).fillna(0) for p in forest_coverages])
                        
                        impossible_forests = ((total_forest > 0.01) & (tcd_max == 0)).sum().compute().item()
                        if impossible_forests == 0:
                            log_execution(logger, f"  [TCF] {year} | 0% Max Density -> 0% Forest Fraction: [PASSED]", logging.INFO)
                        else:
                            log_execution(logger, f"  [TCF] {year} | 0% Max Density -> 0% Forest Fraction: [FAILED] ({impossible_forests} violations)", logging.WARNING)
                            all_passed = False

        if 'IMP' in lake_catalog and 'Imperviousness_Density' in lake_catalog['IMP'] and 'Impervious_Built-up' in lake_catalog['IMP']:
            for year in lake_catalog['IMP']['Imperviousness_Density'].keys():
                if year in lake_catalog['IMP']['Impervious_Built-up']:
                    max_imp_path = lake_catalog['IMP']['Imperviousness_Density'][year].get('max')
                    builtup_coverages = lake_catalog['IMP']['Impervious_Built-up'][year].get('coverage', [])
                    
                    if max_imp_path and builtup_coverages:
                        imp_max = load_da(max_imp_path)
                        total_built = sum([load_da(p).fillna(0) for p in builtup_coverages])
                        
                        impossible_built = ((total_built > 0.01) & (imp_max == 0)).sum().compute().item()
                        if impossible_built == 0:
                            log_execution(logger, f"  [IMP] {year} | 0% Max Density -> 0% Built-up Fraction: [PASSED]", logging.INFO)
                        else:
                            log_execution(logger, f"  [IMP] {year} | 0% Max Density -> 0% Built-up Fraction: [FAILED] ({impossible_built} violations)", logging.WARNING)
                            all_passed = False

        status_msg = "PASSED" if all_passed else "FAILED (See Warnings)"
        log_execution(logger, f"\n=== Global QA Sweep Complete: {status_msg} ===", logging.INFO if all_passed else logging.WARNING)
        
        return all_passed

    def generate_catalog(self, target_dir: str, logger: Optional[logging.Logger] = None) -> str:
        """
        Crawls the finalized COGs and writes a STAC-compliant GeoParquet catalog.
        """
        log_execution(logger, f"Generating cloud-native STAC GeoParquet catalog for lake at: {target_dir}", logging.INFO)
    
        base_path = Path(target_dir)
        catalog_records = []
        
        for tif_path in base_path.rglob("*.tif"):
            parts = tif_path.parts
            
            # Anchor to the data lake root folder name to index subdirectories reliably
            if base_path.name in parts:
                base_idx = parts.index(base_path.name)
                if len(parts) - base_idx < 4: 
                    continue
                category = parts[base_idx + 1]     # e.g., 'HRL' or 'TCF'
                product = parts[base_idx + 2]      # e.g., 'Tree_Cover_Density'
                aggregation = parts[base_idx + 3]  # e.g., 'average', 'coverage'
            else:
                if len(parts) < 4: 
                    continue
                category, product, aggregation = parts[-4], parts[-3], parts[-2]
            
            # Parse timestamp string from filename layout
            year_match = re.search(r'(19\d{2}|20\d{2})', tif_path.name)
            if year_match:
                year_str = year_match.group(1)
                datetime_val = f"{year_str}-01-01T00:00:00Z" 
            else:
                datetime_val = None

            # Extract the specific layer/class from the filename (e.g., 'Non_grassland')
            if "100m_" in tif_path.stem:
                extracted_suffix = tif_path.stem.split("100m_")[-1]
            else:
                # Fallback to grabbing everything after the last underscore
                extracted_suffix = tif_path.stem.split("_")[-1]

            # Prevent redundancy: if the suffix is just the aggregation method (e.g., 'average'),
            # leave layer_class empty, as it's a continuous variable.
            layer_class = None if extracted_suffix == aggregation else extracted_suffix

            # Inspect the physical file to inherit its exact spatial properties
            try:
                with rioxarray.open_rasterio(tif_path) as src:
                    epsg_code = src.rio.crs.to_epsg() if src.rio.crs else None
                    
                    # Transform raster bounds into WGS84 bounding box coordinates
                    bounds_wgs84 = src.rio.transform_bounds("EPSG:4326")
                    long_min, lat_min, long_max, lat_max = bounds_wgs84
                    
                    # Generate standard Shapely geometry representing the spatial footprint
                    geometry_obj = box(long_min, lat_min, long_max, lat_max)
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Could not read metadata for footprint extraction on {tif_path.name}: {e}")
                continue

            # Append standard STAC Asset metadata dictionary structure
            catalog_records.append({
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": tif_path.stem,
                "geometry": geometry_obj,
                "bbox": [long_min, lat_min, long_max, lat_max],
                "properties": {
                    "datetime": datetime_val,
                    "collection": category,
                    "variable": product,
                    "aggregation": aggregation,
                    "layer_class": layer_class,  # Safely added without continuous redundancy
                    "proj:epsg": epsg_code,
                    "level": category, 
                    "year": int(year_str) if year_match else None
                },
                "assets": {
                    "data": {
                        "href": str(tif_path.absolute()),
                        "type": "image/tiff; application=geotiff; profile=cloud-optimized"
                    }
                }
            })
            
        if not catalog_records:
            log_execution(logger, "No items found to inventory.", logging.WARNING)
            return ""

        # Flatten the records directly onto the root level for clean GeoParquet columns
        flattened_records = []
        for record in catalog_records:
            flat = {
                "stac_version": record["stac_version"],
                "type": record["type"],
                "id": record["id"],
                "geometry": record["geometry"],
                "bbox": record["bbox"],
                "href": record["assets"]["data"]["href"],
                "mime_type": record["assets"]["data"]["type"]
            }
            
            # REMOVED 'prop_' PREFIX: Keeps columns aligned with direct stac-geoparquet conventions
            for prop_k, prop_v in record["properties"].items():
                flat[prop_k] = prop_v
                
            flattened_records.append(flat)

        # Wrap as a standard GeoDataFrame
        gdf = gpd.GeoDataFrame(flattened_records, geometry="geometry", crs="EPSG:4326")
        
        # Export the finalized Parquet database file
        catalog_path = os.path.join(target_dir, "wekeo_lake_catalog.parquet")
        gdf.to_parquet(catalog_path, index=False)
        
        log_execution(logger, f"STAC GeoParquet catalog generated successfully with {len(gdf)} assets at: {catalog_path}", logging.INFO)
        return catalog_path

    def _fetch_unpack_query(self, wekeo_query: dict, wekeo_client: object, base_dir: str = "wekeo", logger: Optional[logging.Logger] = None) -> Optional[str]:
        """
        Executes a formatted query against the WEkEO Harmonized Data Access (HDA) API, 
        downloads the compressed results, and unpacks the internal .tif files.

        Parameters
        ----------
        wekeo_query : dict
            The JSON-like dictionary payload containing the specific dataset, product, 
            spatial, and temporal parameters to be requested from the API.
        wekeo_client : object
            An initialized and authenticated instance of the WEkEO HDA client used 
            to send the search and download requests.
        base_dir : str, optional
            The root directory where the downloaded data will be stored. A dataset-specific 
            subdirectory will be created inside this path. Default is "wekeo".
        logger : logging.Logger, optional
            The logger instance to use for recording execution progress, bugs, and API 
            responses. If None, it defaults to the class's internal `wekeo_logger`.

        Returns
        -------
        download_dir : str or None
            The absolute or relative path to the directory where the unzipped .tif files 
            have been saved. Returns None if the API responds with an empty result set.

        Notes
        -----
        This is a private method meant to be called sequentially during the pipeline's 
        execution phase. It includes a built-in safety net to catch silent empty responses 
        from the API (where the API returns 200 OK but provides no actual data links).

        See Also
        --------
        gather_tifs_from_zips : Utility function called internally to extract the files.
        """
        if not logger:
            logger = self.wekeo_logger
            
        dataset_id = wekeo_query.get("dataset_id", "UNKNOWN_DATASET")
        log_execution(logger, f"===={dataset_id}====", logging.INFO)
    
        pretty_query = json.dumps(wekeo_query, indent=4)
        indented_block = "\n    " + pretty_query.replace("\n", "\n    ")
        log_execution(logger, f"Executing search query...{indented_block}", logging.INFO)
        
        # Execute the search query
        response = wekeo_client.search(wekeo_query)
        
        # Check if response has results or is fundamentally empty
        try:
            results = getattr(response, 'results', [])
            is_empty = not response or len(results) == 0
        except Exception:
            results = []
            is_empty = False

        if is_empty:
            log_execution(logger, f"EMPTY RESULT FROM QUERY", logging.WARNING)
            return None
        
        if results:
            total_files = len(results)
            log_execution(logger, f"Query returned {total_files} files. Handing off to HDA bulk downloader...", logging.INFO)
        else:
            log_execution(logger, f"Downloading response group as a bulk fallback to {base_dir}...", logging.INFO)
            
        try:
            # Let the hda client manage its own internal concurrent downloads
            response.download(download_dir=base_dir)
        except Exception as e:
            log_execution(logger, f"HDA download stream failed or was interrupted: {e}", logging.ERROR)
            return None
        
        # Extract the tif files from their zipped folders
        self.gather_tifs_from_zips(
            source_directory=base_dir,
            target_directory=os.path.join(base_dir, "tif_files"),
            logger=logger
        )
        
        return base_dir

    def intersect_config_with_dataframe(
            self,
            recipe: Dict[str, Any],
            wekeo_client: Any,
            api_schema: dict,
            inventory_filename: str = "wekeo_data_inventory.csv",
            logger: Optional[logging.Logger] = None
        ) -> List[Dict[str, Any]]:
            """
            Generates an execution queue of valid WEkEO API queries by intersecting 
            the user configuration with a verified ground-truth dataset inventory.

            This method attempts to load a pre-compiled dataset inventory from the 
            bundled `meta/` directory. If the file is missing, it automatically falls 
            back to dynamically crawling the WEkEO API to reconstruct it. Once the 
            inventory is loaded, it checks if requested datasets support bounding box 
            queries, gracefully degrading to full-extent downloads if they do not.

            Parameters
            ----------
            recipe : Dict[str, Any]
                The parsed YAML configuration dictionary containing the user's spatial, 
                temporal, and dataset requirements.
            wekeo_client : Any
                The initialized WEkEO HDA API client, used to perform the fallback 
                inventory crawl if the static catalog is missing.
            api_schema : dict
                A dictionary containing the categories, products, and available 
                configurations (extracted via WEkEO constraints) used to build the 
                fallback inventory.
            inventory_filename : str, optional
                The name of the static metadata catalog file to look for in the `meta/` 
                directory. Default is 'wekeo_data_inventory.csv'.
            logger : logging.Logger, optional
                The logger instance used to record execution status and warnings. 
                Default is None.

            Returns
            -------
            List[Dict[str, Any]]
                A list of dictionary payloads formatted as valid WEkEO HDA queries, 
                ready to be passed to the API downloader.

            Raises
            ------
            ValueError
                If the requested configuration products cannot be validated against 
                the ground-truth inventory.
            """
            # Try fetching static meta, fallback to dynamic build
            try:
                df = fetch_meta(inventory_filename, logger=logger)
                log_execution(logger, f"Successfully loaded pre-compiled inventory: '{inventory_filename}'", logging.INFO)
                
            except FileNotFoundError:
                log_execution(logger, f"Inventory '{inventory_filename}' not found in meta directory. Constructing dynamically...", logging.WARNING)
                
                # Resolve the path to the meta directory so the new file is saved properly
                base_dir = Path(__file__).resolve().parents[3]
                target_path = base_dir / 'meta' / inventory_filename
                
                df = build_data_inventory(
                    api_schema=api_schema,
                    dataset_map=self.DATASET_MAP,
                    wekeo_client=wekeo_client,
                    output_filepath=str(target_path),
                    logger=logger
                )

            # Validate user config against the resulting DataFrame
            if not self._validate_config_products(recipe, df, logger):
                raise ValueError("Recipe validation failed. Check inventory logs.")
            
            wekeo_config = recipe.get('sources', {}).get('wekeo', {})
            global_q_res = wekeo_config.get('query_resolution', 'highest') 
            
            # Extract Global Constraints
            temp_cfg = recipe.get('temporal', {})
            start_year, end_year = temp_cfg.get('start_year'), temp_cfg.get('end_year')
            temporal_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

            # Extract Target Bounding Box
            user_bbox = None
            raw_spatial = recipe.get('raw_config', {}).get('spatial', {})
            use_bbox = raw_spatial.get('use_bbox', False)
            
            if use_bbox:
                bbox_dict = recipe.get('spatial', {}).get('bbox')
                if bbox_dict:
                    user_bbox = [
                        bbox_dict['long_min'], 
                        bbox_dict['lat_min'], 
                        bbox_dict['long_max'], 
                        bbox_dict['lat_max']
                    ]
                else:
                    log_execution(logger, "Warning: 'use_bbox' is True, but no coordinates were found.", logging.WARNING)
                
            execution_queue = []
            datasets_config = wekeo_config.get('datasets', {})
            
            # 3. Build the specific execution queue
            for category, cat_config in datasets_config.items():
                if not cat_config or not cat_config.get('include', False): 
                    continue
                    
                # Logic for Format-based layers (CORINE)
                if 'format' in cat_config:
                    target_format = cat_config.get('format')
                    strict_df = temporal_df[(temporal_df['category'] == category) & 
                                            (temporal_df['format'] == target_format)]
                    
                    for _, row in strict_df.iterrows():
                        supports_bbox = bool(row['bbox'])
                        query_bbox = user_bbox if supports_bbox else None
                        
                        if user_bbox and not supports_bbox:
                            log_execution(logger, f"Note: API BBOX not supported for '{row['productType']}'. Will download full extent.", logging.INFO)

                        execution_queue.append(generate_wekeo_query(
                            dataset_id=row['dataset_id'], 
                            product_type=row['productType'],
                            data_format=target_format, 
                            bbox=query_bbox
                        ))
                        log_execution(logger, f"Queued: {row['productType']} ({row['year']}) at {target_format}", logging.INFO)

                # Logic for Resolution-based layers (HRL, etc.)
                elif 'productTypes' in cat_config:
                    strategy = cat_config.get('query_resolution', global_q_res)
                    
                    for product in cat_config['productTypes']:
                        product_df = temporal_df[temporal_df['productType'] == product]
                        
                        if product_df.empty:
                            log_execution(logger, f"Skipped '{product}': No data found for specified years.", logging.WARNING)
                            continue
                            
                        # Resolve raw resolution to query
                        available_res = product_df['resolution'].unique()
                        resolved_res = self._resolve_query_resolution(strategy, available_res, logger)
                        
                        final_df = product_df[product_df['resolution'] == resolved_res]
                        
                        for year in sorted(final_df['year'].unique()):
                            row = final_df[final_df['year'] == year].iloc[0]
                            
                            supports_bbox = bool(row['bbox'])
                            query_bbox = user_bbox if supports_bbox else None
                            
                            if user_bbox and not supports_bbox:
                                log_execution(logger, f"Note: API BBOX not supported for '{product}' ({year}). Will download full extent.", logging.INFO)

                            execution_queue.append(generate_wekeo_query(
                                dataset_id=row['dataset_id'], 
                                product_type=product,
                                year=str(year), 
                                resolution=resolved_res, 
                                bbox=query_bbox
                            ))
                            log_execution(logger, f"Queued: {product} ({year}) at {resolved_res}", logging.INFO)
                      
            return execution_queue

    def _validate_config_products(
            self,
            recipe: Dict[str, Any], 
            inventory_df: pd.DataFrame, 
            logger: Optional[logging.Logger] = None
            ) -> bool:
        """
        Validates user-requested WEkEO product types against the ground-truth API inventory.

        This internal method acts as a pre-execution safeguard. It ensures that the 
        products requested in the YAML configuration actually exist in the compiled 
        metadata inventory before the pipeline attempts to query the API. It also 
        validates specific resolution overrides, issuing a warning if an exact 
        resolution match is unavailable.

        Parameters
        ----------
        recipe : dict
            A dictionary containing the parsed execution recipe. Expected to contain 
            nested configuration under `['sources']['wekeo']`.
        inventory_df : pandas.DataFrame
            The ground-truth API inventory loaded as a DataFrame. Must contain at least 
            the columns 'productType' and 'resolution'.
        logger : logging.Logger, optional
            The logger instance used to record validation steps, warnings for missing 
            resolutions, and errors for missing products. Default is None.

        Returns
        -------
        bool
            True if all requested products are found in the inventory, or if the WEkEO 
            pipeline is explicitly disabled. False if one or more requested products 
            are completely missing from the inventory. 
            
            Note: Invalid resolution targets only generate a warning to the logger and 
            do not cause the function to return False, as the resolver will fallback 
            to the best available option.
        """
        is_valid = True
        # Retrieve requested wekeo section
        wekeo_config = recipe.get('sources', {}).get('wekeo', {})
        
        if not wekeo_config.get('enabled', False):
            log_execution(logger, "WEkEO pipeline is disabled in recipe. Skipping validation.", logging.INFO)
            return True
        
        datasets_config = wekeo_config.get('datasets', {})
        # Generate list of products that can be requested
        available_products = set(inventory_df['productType'].dropna().unique())
        
        for category, cat_config in datasets_config.items():
            if not cat_config or not cat_config.get('include', False):
                continue
                
            requested_products = cat_config.get('productTypes', [])
            
            for product in requested_products:
                if product not in available_products:
                    log_execution(logger, f"Mismatch in [{category}]: Product '{product}' not in inventory.", logging.WARNING)
                    is_valid = False
                else:
                    # New: Check if the specific resolution override exists (if provided)
                    target_res = cat_config.get('query_resolution')
                    if target_res and target_res not in ['highest', 'lowest']:
                        prod_res = inventory_df[inventory_df['productType'] == product]['resolution'].unique()
                        if target_res not in prod_res:
                            log_execution(logger, f"Warning: Specific resolution '{target_res}' for '{product}' not in inventory. Resolver will pick best available.", logging.WARNING)          
        return is_valid