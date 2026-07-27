import yaml
import os
import logging
from bmc.lake.wekeo import wekeo_lake

lake_recipe = """
paths:
  base_dir: "/storage/niels/bmc/debug/"
  raw_dir: "/storage/niels/bmc/debug/raw/"

lake_name: "micro_lake"

raw_config:
  keep_raw: false
  spatial:
    use_bbox: false

spatial:
  target_grid_key: "EEA_10km"
  resampling_strategies:
    continuous: ["average", "max", "min", "rms"]
    discrete: "coverage" 
  bbox:
    long_min: 3.0
    lat_min: 50.0
    long_max: 6.5
    lat_max: 52.5
temporal:
  start_year: 2017
  start_month: 1
  end_year: 2018
  end_month: 12

sources:
  wekeo:
    enabled: true
    datasets:
      TCF:
        include: true
        productTypes:
          - "Dominant Leaf Type"
          - "Tree Cover Density"
      CORINE:
        include: true
        productTypes:
          - "Corine Land Cover 2018"
"""

recipe = yaml.safe_load(lake_recipe)
lake_dir = recipe["paths"]["base_dir"]

# 1. Initialize your engine
lake_engine = wekeo_lake()

# 2. AUTOMATIC INITIALIZATION: Build the lake sequentially using GDAL C++ threads
generated_cogs = lake_engine.build_datalake(recipe, logger=None)

# 3. EXTRACTION: Capture the automatically constructed logger from the engine instance
auto_logger = lake_engine.logger

# 4. ORCHESTRATION: Pass the captured logger downstream
is_valid = lake_engine.validate_datalake(lake_dir, logger=auto_logger)

if is_valid:
    lake_engine.generate_catalog(lake_dir, logger=auto_logger)
else:
    if auto_logger:
        auto_logger.error("Datalake QA validation failed! Catalog creation aborted.")