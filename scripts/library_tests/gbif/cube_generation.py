import yaml
import os
import copy
import logging
import faulthandler

from bmc.cube import gbif
from bmc.datasource.gbif import sql

faulthandler.enable()

# Ensure we have a directory to save our test cubes and configs
output_dir = "../test_cubes/"
os.makedirs(output_dir, exist_ok=True)

credentials = {"GBIF_USER":"nbilliet",
               "GBIF_EMAIL":"niels.billiet@plantentuinmeise.be",
               "GBIF_PWD": "xax9yahc"}

# =======================================================================
# 1. BASE RECIPE DEFINITION (Raw Points / Vector)
# =======================================================================
base_yaml = f"""
base_dir: {output_dir}
cube_name: gbif_test_suite # This will be dynamically overwritten per test

spatial:
  target_grid: EEA
  target_resolution: 1km
  use_bbox: true  
  bbox:
      long_min: 2.591733
      long_max: 5.883999
      lat_min: 50.680797
      lat_max: 51.963900

temporal:
  start_year: 2004
  start_month: 1
  end_year: 2025
  end_month: 12

sources:
  gbif:
    query_filters:
      taxon_keys: ['G59D', '4F6YZ', '3Y9W2', '5B9T3', '4HPXM']
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
"""

base_recipe = yaml.safe_load(base_yaml)

# =======================================================================
# 2. API CUBE RECIPE DEFINITION
# =======================================================================
recipe_api = copy.deepcopy(base_recipe)
recipe_api["sources"]["gbif"]["processing_mode"] = "api_cube"
recipe_api["sources"]["gbif"]["api_cube_config"] = {
    "data_type": "discrete",
    "spatial_method": "intersect"
}
recipe_api["sources"]["gbif"]["aggregate"] = {
    "group_by_columns": ["year", "month", "speciesKey"],
    "metrics": [{"column": "gbifID", "method": "count", "rename": "total_occurrences"}]
}

# =======================================================================
# 3. DUAL DATA DOWNLOAD (API Cube vs Vector Points)
# =======================================================================
engine = gbif.gbif_cube()
download_dir = os.path.join(output_dir, "downloads")

print("--- Submitting GBIF Queries ---")
vector_query = engine.generate_gbif_query_from_recipe(base_recipe)
api_query = engine.generate_gbif_query_from_recipe(recipe_api)

#vector_key = sql.submit_gbif_query(vector_query, creds=credentials)
#api_key = sql.submit_gbif_query(api_query, creds=credentials)

print("\n--- Fetching GBIF Downloads ---")
#vector_zip_path = sql.fetch_gbif_download(vector_key, target_dir=download_dir)['path']
vector_zip_path = "../test_cubes/downloads/0013211-260721160103020.zip"
#api_zip_path = sql.fetch_gbif_download(api_key, target_dir=download_dir)['path']
api_zip_path = "../test_cubes/downloads/0013212-260721160103020.zip"
print(f"Vector Data downloaded to: {vector_zip_path}")
print(f"API Cube Data downloaded to: {api_zip_path}")

# =======================================================================
# 4. TEST EXECUTION HELPER
# =======================================================================
def run_test_case(test_name: str, recipe: dict, zip_path: str):
    print(f"\n=======================================================")
    print(f" RUNNING TEST: {test_name.upper()}")
    print(f"=======================================================")
    
    # Force the engine to create a uniquely named folder for each test!
    recipe["cube_name"] = test_name
    
    yaml_filename = os.path.join(output_dir, f"{test_name}.yaml")
    with open(yaml_filename, "w", encoding="utf-8") as file:
        yaml.dump(recipe, file, default_flow_style=False, sort_keys=False)
        
    engine.process_cube(
        recipe=recipe, 
        dataset_name="gbif", 
        downloaded_filepath=zip_path
    )

# =======================================================================
# 5. TEST SUITE: EEA PROCESSING MODES
# =======================================================================

# --- Test 5A: Raw Point Export ---
recipe_raw = copy.deepcopy(base_recipe)
recipe_raw["sources"]["gbif"]["processing_mode"] = "raw"
run_test_case("gbif_test_00_raw_points", recipe_raw, vector_zip_path)


# --- Test 5B: GBIF API Cube (Server-side aggregation) ---
# Note: This is the ONLY test that uses the api_zip_path
run_test_case("gbif_test_01_eea_api_cube", recipe_api, api_zip_path)


# --- Test 5C: Vector Point Classification ---
recipe_pt = copy.deepcopy(base_recipe)
recipe_pt["sources"]["gbif"]["processing_mode"] = "vector"
recipe_pt["sources"]["gbif"]["vector_processing"] = {
    "topology": "point",
    "mapping_mode": "classification",
    "spatial_method": "intersect"
}
recipe_pt["sources"]["gbif"]["aggregate"] = {
    "group_by_columns": ["specieskey", "year", "month"],
    "metrics": [{"column": "gbifid", "method": "nunique", "weighted": False, "rename": "observation_count"}]
}
run_test_case("gbif_test_02_eea_point_discrete", recipe_pt, vector_zip_path)


# --- Test 5D: Vector Point Cloud (Fractional Jittering) ---
recipe_pc = copy.deepcopy(base_recipe)
recipe_pc["sources"]["gbif"]["processing_mode"] = "vector"
recipe_pc["sources"]["gbif"]["vector_processing"] = {
    "topology": "point_cloud",
    "mapping_mode": "fractional",
    "spatial_method": "intersect",
    "topology_config": {
        "point_cloud": {"n_passes": 50, "distribution": "gaussian", "random_seed": 42}
    }
}
recipe_pc["sources"]["gbif"]["aggregate"] = {
    "export_unaggregated": True,
    "group_by_columns": ["specieskey", "year", "month"],
    # Point clouds use the "fraction" column
    "metrics": [
        {"column": "fraction", "method": "sum", "weighted": False, "rename": "expected_occurrences"}
    ]
}
run_test_case("gbif_test_03_eea_pointcloud_fractional", recipe_pc, vector_zip_path)


# --- Test 5E: Vector Polygon (Fractional Buffer Sharding) ---
recipe_poly = copy.deepcopy(base_recipe)
recipe_poly["sources"]["gbif"]["processing_mode"] = "vector"
recipe_poly["sources"]["gbif"]["vector_processing"] = {
    "topology": "polygon",
    "mapping_mode": "fractional",
    "spatial_method": "intersect",
    "topology_config": {
        "polygon": {"quad_segs": 8}
    }
}
recipe_poly["sources"]["gbif"]["aggregate"] = {
    "export_unaggregated": True,
    "group_by_columns": ["specieskey", "year", "month"],
    # Polygons use the "areal_fraction" column
    "metrics": [
        {"column": "areal_fraction", "method": "sum", "weighted": False, "rename": "expected_occurrences"}
    ]
}
run_test_case("gbif_test_04_eea_polygon_fractional", recipe_poly, vector_zip_path)


# =======================================================================
# 6. TEST SUITE: CRS REPROJECTIONS
# =======================================================================

# --- Test 6A: Global WGS84 (30sec) ---
recipe_wgs84 = copy.deepcopy(recipe_pc)
recipe_wgs84["spatial"]["target_grid"] = "Global_WGS84"
recipe_wgs84["spatial"]["target_resolution"] = "30sec"
run_test_case("gbif_test_05_wgs84_pointcloud", recipe_wgs84, vector_zip_path)


# --- Test 6B: Global Equal Area (1km) ---
recipe_gea = copy.deepcopy(recipe_pc)
recipe_gea["spatial"]["target_grid"] = "Global_EqualArea"
recipe_gea["spatial"]["target_resolution"] = "1km"
run_test_case("gbif_test_06_equalarea_pointcloud", recipe_gea, vector_zip_path)

print("\n=== All GBIF Test Suites Completed Successfully ===")