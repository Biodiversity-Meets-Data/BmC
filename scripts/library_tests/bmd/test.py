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
output_dir = os.path.abspath(os.path.join(script_dir, "../test_cubes/"))
os.makedirs(output_dir, exist_ok=True)

# =======================================================================
# REALISTIC PAYLOAD TEMPLATE 
# (Full year monthly + Multidimensional Bioclim Matrix)
# =======================================================================
def get_recipe(name, export_format):
    return f"""
base_dir: {output_dir}
cube_name: {name}

export_as:
  format: {export_format}

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
    enabled: false
"""

# =======================================================================
# EXECUTE ORCHESTRATOR
# =======================================================================
def run_combined_suite():
    orchestrator = bmd_cube()

    # ---------------------------------------------------------
    # RUN TEST 1: REALISTIC NETCDF EXPORT
    # ---------------------------------------------------------
    print(f"\n=======================================================")
    print(f" RUNNING TEST 1: REALISTIC MULTI-SLICE NETCDF EXPORT")
    print(f"=======================================================")
    yaml_1_filename = "test_1_realistic_nc.yaml"
    yaml_1_path = os.path.join(script_dir, yaml_1_filename)
    with open(yaml_1_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml.safe_load(get_recipe("test_1_realistic_nc", "netcdf")), file, default_flow_style=False, sort_keys=False)
        
    orchestrator.process_recipe(recipe_file=yaml_1_filename, recipe_path=script_dir, max_workers=8, logger=logger)

    # ---------------------------------------------------------
    # RUN TEST 2: REALISTIC ZARR EXPORT
    # ---------------------------------------------------------
    print(f"\n=======================================================")
    print(f" RUNNING TEST 2: REALISTIC MULTI-SLICE ZARR EXPORT")
    print(f"=======================================================")
    yaml_2_filename = "test_2_realistic_zarr.yaml"
    yaml_2_path = os.path.join(script_dir, yaml_2_filename)
    with open(yaml_2_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml.safe_load(get_recipe("test_2_realistic_zarr", "zarr")), file, default_flow_style=False, sort_keys=False)
        
    orchestrator.process_recipe(recipe_file=yaml_2_filename, recipe_path=script_dir, max_workers=8, logger=logger)
    
    print("\n=== All Combined Test Suites Completed Successfully ===")

if __name__ == "__main__":
    run_combined_suite()