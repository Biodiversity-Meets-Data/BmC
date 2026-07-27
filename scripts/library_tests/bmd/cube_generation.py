import yaml
import os
import logging
import faulthandler
from bmc.cube.bmd import bmd_cube

faulthandler.enable()

# Enable GDAL/CURL verbose logging[cite: 8]
os.environ["CPL_DEBUG"] = "ON"
os.environ["CPL_CURL_VERBOSE"] = "YES"

# Set up the logger to capture the execution flow
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dynamically resolve the absolute path to the catalog based on this script's location[cite: 8]
script_dir = os.path.dirname(os.path.abspath(__file__))
catalog_path = os.path.abspath(os.path.join(script_dir, "../../../meta/chelsa_gp_stac/chelsa_master.parquet"))

# Ensure we have a directory to save our test cubes and configs
output_dir = os.path.abspath(os.path.join(script_dir, "../test_cubes/"))
os.makedirs(output_dir, exist_ok=True)

# =======================================================================
# 1. COMBINED RECIPE DEFINITION
# =======================================================================
combined_recipe_eea = f"""
base_dir: {output_dir}
cube_name: combined_test_cube_eea1km_netcdf

export_as:
  format: netcdf

spatial:
  target_grid: EEA
  target_resolution: 1km
  use_bbox: true
  bbox:
      long_min: 3.376891
      long_max: 4.804775
      lat_min: 50.583876
      lat_max: 51.386124

temporal:
  start_year: 2018
  start_month: 1
  end_year: 2018
  end_month: 12

sources:
  # ---------------------------------------------------------------------
  # CHELSA (Raster Processing)
  # ---------------------------------------------------------------------
  chelsa:
      enabled: true
      catalog_path: {catalog_path}
      levels:
        monthly:
          include: true
          variables:
            tas: true
            tasmax: false
            tasmin: false
            pr: true
        annual:
          include: true
          variables:
            tas: true
            tasmax: false
            tasmin: false
            pr: false
        climatologies:
          include: true
          time_ranges:
            "1981-2010": true
            "2011-2040": false
            "2041-2070": false
          ensembles:
            historical: true
            GFDL-ESM4: true
            IPSL-CM6A-LR: true
          scenarios:
            historical: true
            ssp126: true
            ssp370: true
            ssp585: true
          variables:
            tas: true
            tasmax: false
            tasmin: false
            pr: true
        bioclim:
          include: true
          time_ranges:
            1981-2010: true
            2011-2040: true
            2041-2070: true
          ensembles:
            historical: true
            GFDL-ESM4: true
            IPSL-CM6A-LR: true
          scenarios:
            historical: true
            ssp126: true
            ssp370: true
            ssp585: true
          variables:
            bio01: true
            bio02: true

  # ---------------------------------------------------------------------
  # GBIF (Vector Processing)
  # ---------------------------------------------------------------------
  gbif:
    enabled: true
    query_filters:
      taxon_keys: ['G59D', '4F6YZ', '3Y9W2'] # Subset for faster testing
      record_type: presence
      default_Uncertainty: 1000
      max_uncertainty: 1000
      exclude_issues:
        - ZERO_COORDINATE
        - COORDINATE_OUT_OF_RANGE
        - COUNTRY_COORDINATE_MISMATCH
    taxonomy:
      col_backbone: true
    columns: ["year", "month", "recordedby"]
    time_cols: ["year", "month"]
    processing_mode: "vector"
    vector_processing:
      topology: "point_cloud"
      mapping_mode: "fractional"
      spatial_method: "intersect"
      topology_config:
        point_cloud:
          n_passes: 25 # Reduced passes for testing
          distribution: "gaussian"
          random_seed: 42
    aggregate:
      export_unaggregated: false
      group_by_columns: ["specieskey", "year", "month"]
      metrics:
        - column: "fraction"
          method: "sum"
          weighted: false
          rename: "expected_occurrences"
"""

# =======================================================================
# 2. GENERATE YAML FILE
# =======================================================================
# Parse the string into a dictionary[cite: 8]
recipe = yaml.safe_load(combined_recipe_eea)

# Dump the dictionary directly to the file[cite: 8]
yaml_filename = "combined_test_cube_eea1km_netcdf.yaml"
yaml_path = os.path.join(script_dir, yaml_filename)

with open(yaml_path, "w", encoding="utf-8") as file:
    yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)
    
logger.info(f"YAML file successfully generated at: {yaml_path}")

# =======================================================================
# 3. EXECUTE ORCHESTRATOR
# =======================================================================
def run_combined_suite():
    print(f"\n=======================================================")
    print(f" RUNNING COMBINED TEST: CHELSA + GBIF")
    print(f"=======================================================")
    
    # Instantiate the unified orchestrator 
    orchestrator = bmd_cube()
    
    # Trigger the asynchronous pipeline
    # It will submit GBIF, process CHELSA, then fetch/process GBIF
    orchestrator.process_recipe(
        recipe_file=yaml_filename, 
        recipe_path=script_dir, 
        max_workers=8,
        logger=logger
    )

if __name__ == "__main__":
    run_combined_suite()