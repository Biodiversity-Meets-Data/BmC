import yaml
import os
import glob
import logging
import faulthandler
from bmc.cube.bmd import bmd_cube

faulthandler.enable()

# Set up the logger to capture the execution flow
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dynamically resolve paths based on this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
catalog_path = os.path.abspath(os.path.join(script_dir, "../../../meta/chelsa_gp_stac/chelsa_master.parquet"))
output_dir = os.path.abspath(os.path.join(script_dir, "test_cubes/"))
os.makedirs(output_dir, exist_ok=True)
local_zip_path = "test_cubes/downloads"
# =======================================================================
# TEMPLATE CREDENTIALS
# =======================================================================
# Dictionary for passing API credentials dynamically to the pipeline
creds = {
    "GBIF_USER": "username",
    "GBIF_PWD": "password",
    "GBIF_MAIL": "email"
}
# =======================================================================
# 1. TEST 1: EEA GRID | NETCDF EXPORT | ASYNCHRONOUS GBIF DOWNLOAD
# =======================================================================
recipe_eea_netcdf = f"""
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
  end_month: 12 # Full single year (12 slices per monthly variable)

sources:
  chelsa:
    enabled: true
    catalog_path: {catalog_path}
    levels:
      monthly:
        include: true
        variables:
          pr: true
          tas: true
      bioclim:
        include: true
        time_ranges:
          "1981-2010": true
          "2011-2040": true
          "2041-2070": true
        ensembles:
          historical: true
          GFDL-ESM4: true
          MPI-ESM1-2-HR: true
        scenarios:
          historical: true
          ssp126: true
          ssp585: true
        variables:
          bio01: true  # Mean Annual Temp
          bio12: true  # Annual Precipitation

  gbif:
    enabled: true
    local_file_path: "{local_zip_path}"
    query_filters:
      taxon_keys: ['G59D', '4F6YZ', '3Y9W2'] 
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
          n_passes: 25 
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
# 2. TEST 2: GLOBAL WGS84 GRID | NETCDF EXPORT | LOCAL GBIF BYPASS
# =======================================================================
def get_wgs84_recipe_string(local_zip_path):
    return f"""
base_dir: {output_dir}
cube_name: combined_test_cube_wgs84_netcdf

export_as:
  format: netcdf

spatial:
  target_grid: Global_WGS84
  target_resolution: 30sec
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
  chelsa:
      enabled: true
      catalog_path: {catalog_path}
      levels:
        monthly:
          include: true
          variables:
            pr: true
            tas: true
        bioclim:
          include: true
          time_ranges:
            "1981-2010": true
            "2011-2040": true
            "2041-2070": true
          ensembles:
            historical: true
            GFDL-ESM4: true
            MPI-ESM1-2-HR: true
          scenarios:
            historical: true
            ssp126: true
            ssp585: true
          variables:
            bio01: true  # Mean Annual Temp
            bio12: true

  gbif:
    enabled: true
    local_file_path: "{local_zip_path}"
    columns: ["year", "month", "recordedby"]
    time_cols: ["year", "month"]
    processing_mode: "vector"
    vector_processing:
      topology: "polygon"
      mapping_mode: "fractional"
      spatial_method: "intersect"
      topology_config:
        polygon:
          quad_segs: 8
    aggregate:
      export_unaggregated: false
      group_by_columns: ["specieskey", "year", "month"]
      metrics:
        - column: "areal_fraction"
          method: "sum"
          weighted: false
          rename: "expected_occurrences"
"""

# =======================================================================
# 3. TEST 3: GLOBAL EQUAL AREA GRID | ZARR EXPORT | LOCAL GBIF BYPASS
# =======================================================================
def get_equalarea_zarr_recipe_string(local_zip_path):
    return f"""
base_dir: {output_dir}
cube_name: combined_test_cube_equalarea_zarr

export_as:
  format: zarr

spatial:
  target_grid: Global_EqualArea
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
  chelsa:
      enabled: true
      catalog_path: {catalog_path}
      levels:
        monthly:
          include: true
          variables:
            pr: true
            tas: true
        bioclim:
          include: true
          time_ranges:
            "1981-2010": true
            "2011-2040": true
            "2041-2070": true
          ensembles:
            historical: true
            GFDL-ESM4: true
            MPI-ESM1-2-HR: true
          scenarios:
            historical: true
            ssp126: true
            ssp585: true
          variables:
            bio01: true  # Mean Annual Temp
            bio12: true

  gbif:
    enabled: true
    local_file_path: "{local_zip_path}"
    columns: ["year", "month", "recordedby"]
    time_cols: ["year", "month"]
    processing_mode: "vector"
    vector_processing:
      topology: "point_cloud"
      mapping_mode: "fractional"
      spatial_method: "intersect"
      topology_config:
        point_cloud:
          n_passes: 10
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
# 4. EXECUTE ORCHESTRATOR SUITE
# =======================================================================
def run_combined_suite():
    orchestrator = bmd_cube()
    """
    # ---------------------------------------------------------
    # RUN TEST 1: EEA Grid (NetCDF + Async GBIF Download)
    # ---------------------------------------------------------
    print(f"\n=======================================================")
    print(f" RUNNING TEST 1: EEA (1km) | NETCDF | ASYNC GBIF")
    print(f"=======================================================")
    
    recipe_1 = yaml.safe_load(recipe_eea_netcdf)
    yaml_1_filename = "combined_test_1_eea_netcdf.yaml"
    yaml_1_path = os.path.join(script_dir, yaml_1_filename)

    with open(yaml_1_path, "w", encoding="utf-8") as file:
        yaml.dump(recipe_1, file, default_flow_style=False, sort_keys=False)
        
    orchestrator.process_recipe(
        recipe_file=yaml_1_filename, 
        recipe_path=script_dir, 
        max_workers=8,
        creds=creds
    )
    """
    # ---------------------------------------------------------
    # FIND THE DOWNLOADED ZIP FILE FOR TESTS 2 & 3
    # ---------------------------------------------------------
    downloads_dir = os.path.join(output_dir, "downloads")
    zip_files = glob.glob(os.path.join(downloads_dir, "*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Test 1 failed to download a GBIF zip file to {downloads_dir}.")
    
    latest_zip = max(zip_files, key=os.path.getmtime).replace("\\", "/")
    logger.info(f"Located downloaded GBIF dataset for subsequent tests: {latest_zip}")

    # ---------------------------------------------------------
    # RUN TEST 2: Global_WGS84 Grid (NetCDF + Local GBIF Bypass)
    # ---------------------------------------------------------
    print(f"\n=======================================================")
    print(f" RUNNING TEST 2: GLOBAL_WGS84 (30sec) | NETCDF | LOCAL GBIF")
    print(f"=======================================================")
    
    recipe_2 = yaml.safe_load(get_wgs84_recipe_string(latest_zip))
    yaml_2_filename = "combined_test_2_wgs84_netcdf.yaml"
    yaml_2_path = os.path.join(script_dir, yaml_2_filename)

    with open(yaml_2_path, "w", encoding="utf-8") as file:
        yaml.dump(recipe_2, file, default_flow_style=False, sort_keys=False)

    orchestrator.process_recipe(
        recipe_file=yaml_2_filename, 
        recipe_path=script_dir, 
        max_workers=8
    )

    # ---------------------------------------------------------
    # RUN TEST 3: Global_EqualArea Grid (Zarr + Local GBIF Bypass)
    # ---------------------------------------------------------
    print(f"\n=======================================================")
    print(f" RUNNING TEST 3: GLOBAL_EQUALAREA (1km) | ZARR | LOCAL GBIF")
    print(f"=======================================================")
    
    recipe_3 = yaml.safe_load(get_equalarea_zarr_recipe_string(latest_zip))
    yaml_3_filename = "combined_test_3_equalarea_zarr.yaml"
    yaml_3_path = os.path.join(script_dir, yaml_3_filename)

    with open(yaml_3_path, "w", encoding="utf-8") as file:
        yaml.dump(recipe_3, file, default_flow_style=False, sort_keys=False)

    orchestrator.process_recipe(
        recipe_file=yaml_3_filename, 
        recipe_path=script_dir, 
        max_workers=8
    )
    
    print("\n=== All Combined Test Suites Completed Successfully ===")

if __name__ == "__main__":
    run_combined_suite()