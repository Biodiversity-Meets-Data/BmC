import yaml
import os
import logging
from bmc.cube.datasets.raster.chelsa import chelsa_cube

import faulthandler
faulthandler.enable()

import os
os.environ["CPL_DEBUG"] = "ON"
os.environ["CPL_CURL_VERBOSE"] = "YES"

# Dynamically resolve the absolute path to the catalog based on this script's location
# __file__ is the path to cube_generation.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to hit the BmC root, then down into the meta folder
catalog_path = os.path.abspath(os.path.join(script_dir, "../../../meta/chelsa_gp_stac/chelsa_master.parquet"))


chelsa_recipe_eea=f"""
base_dir: ../test_cubes/
cube_name: chelsa_test_cube_eea1km_netcdf

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
        # Select Base Periods / Projection Horizons
        time_ranges:
          "1981-2010": true  # Historical observation baseline
          "2011-2040": false # Near-term future
          "2041-2070": false # Mid-term future
        ensembles:
          historical: true   # REQUIRED if "1981-2010" is true
          GFDL-ESM4: true
          IPSL-CM6A-LR: true
        scenarios:
          historical: true   # REQUIRED if "1981-2010" is true
          ssp126: true      # Highly optimistic / Strong mitigation
          ssp370: true      # Middle of the road
          ssp585: true      # Highly pessimistic / Fossil-fueled development          
        variables:
          tas: true     # Mean air temperature
          tasmax: false # Maximum air temperature
          tasmin: false # Minimum air temperature
          pr: true      # Precipitation amount
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
"""
# 1. Parse the string into a dictionary
recipe = yaml.safe_load(chelsa_recipe_eea)

# 2. Dump the dictionary directly to the file (just for your records)
with open("chelsa_test_cube_eea1km_netcdf.yaml", "w", encoding="utf-8") as file:
    yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)
    
print("YAML file successfully generated!")

# 3. Instantiate the engine ONCE
engine = chelsa_cube()

# 4. Process the cube using the parsed dictionary
engine.process_cube(recipe=recipe, max_workers=8)

# =========================================================
# WGS 84 RUN
# =========================================================
chelsa_recipe_wgs84 = f"""
base_dir: ../test_cubes/
cube_name: chelsa_test_cube_wgs84_netcdf

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
"""

recipe_wgs84 = yaml.safe_load(chelsa_recipe_wgs84)

yaml_filename = "chelsa_test_cube_wgs84_netcdf.yaml"
with open(yaml_filename, "w", encoding="utf-8") as file:
    yaml.dump(recipe_wgs84, file, default_flow_style=False, sort_keys=False)

print("WGS84 (Global_WGS84_30sec) YAML file successfully generated!")

# Pass the parsed dictionary, not the filename
engine.process_cube(recipe=recipe_wgs84, max_workers=8)

# =========================================================
# EQUAL AREA RUN
# =========================================================
chelsa_recipe_equalarea = f"""
base_dir: ../test_cubes/
cube_name: chelsa_test_cube_equalarea_1km_netcdf

export_as:
  format: netcdf

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
"""

recipe_equalarea = yaml.safe_load(chelsa_recipe_equalarea)

yaml_filename = "chelsa_test_cube_equalarea_1km_netcdf.yaml"
with open(yaml_filename, "w", encoding="utf-8") as file:
    yaml.dump(recipe_equalarea, file, default_flow_style=False, sort_keys=False)

print("Global Equal Area (Global_EqualArea_1km) YAML file successfully generated!")

# Pass the parsed dictionary, not the filename
engine.process_cube(recipe=recipe_equalarea, max_workers=8)

# =========================================================
# ZARR RUN
# =========================================================
chelsa_recipe_zarr = f"""
base_dir: ../test_cubes/
cube_name: chelsa_test_cube_eea1km_zarr

export_as:
  format: zarr

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
  chelsa:
    enabled: true
    catalog_path: {catalog_path}
    levels:
      monthly:
        include: true
        variables:
          tas: true
          pr: true
      annual:
        include: true
        variables:
          tas: true
      bioclim:
        include: true
        time_ranges:
          1981-2010: true
        ensembles:
          historical: true
        scenarios:
          historical: true
        variables:
          bio01: true
          bio02: true
"""

recipe_zarr = yaml.safe_load(chelsa_recipe_zarr)

yaml_filename = "chelsa_test_cube_eea1km_zarr.yaml"
with open(yaml_filename, "w", encoding="utf-8") as file:
    yaml.dump(recipe_zarr, file, default_flow_style=False, sort_keys=False)

print("Zarr (EEA_1km) YAML file successfully generated!")

# Pass the parsed dictionary, not the filename
engine.process_cube(recipe=recipe_zarr, max_workers=8)