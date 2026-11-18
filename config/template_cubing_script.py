import os
import json
from pathlib import Path
from bmc.cube.orchestrator.bmd import bmdCube

# =====================================================================
# OPTIONAL: CREDENTIAL GENERATION (Uncomment to create credential files)
# =====================================================================
# creds_dir = Path.home()
# 
# cdse_creds_data = {
#     "s3_endpoint": "https://eodata.dataspace.copernicus.eu",
#     "s3_access_key": "yourAccessKey",
#     "s3_secret_key": "yourSecretKey"
# }
# with open(creds_dir / "cdse.json", "w") as f:
#     json.dump(cdse_creds_data, f, indent=4)
# 
# gbif_creds_data = {
#     "GBIF_USER": "yourAccountName", 
#     "GBIF_PWD": "yourPassword", 
#     "GBIF_MAIL": "yourEmail"
# }
# with open(creds_dir / "gbif.json", "w") as f:
#     json.dump(gbif_creds_data, f, indent=4)
# 
# private_creds_data = {
#     "s3_access_key": "yourAccessKey", 
#     "s3_secret_key": "yourSecretKey",
#     "s3_endpoint": "https://s3.eu-north-1.amazonaws.com"
# }
# with open(creds_dir / "bmd.json", "w") as f:
#     json.dump(private_creds_data, f, indent=4)
# =====================================================================

# 1. Resolve credential paths
cdse_creds = Path.home() / "cdse.json"
private_creds = Path.home() / "bmd.json"
gbif_creds = Path.home() / "gbif.json"

active_credentials = {
    "gbif_credentials": str(gbif_creds),
    "cdse_s3_credentials": str(cdse_creds),
    "private_s3_credentials": str(private_creds)
}

# 2. Define your bounding box [long_min, lat_min, long_max, lat_max]
custom_bbox = [4.326938, 50.723880, 4.561834, 50.847357]

# 3. Define the execution recipe
tutorial_cube = {
    "base_dir": "./bmdCube_tutorial_output/",
    "cube_name": "eeaGrid100m_cube",
    "export_as": {
        "format": "netcdf"
    },
    "spatial": {
        "target_grid": "EEA",
        "target_resolution": "100m",
        "use_bbox": True,
        "bbox": {
            "long_min": custom_bbox[0],
            "lat_min": custom_bbox[1],
            "long_max": custom_bbox[2],
            "lat_max": custom_bbox[3]
        }
    },
    "temporal": {
        "start_year": 2018,
        "start_month": 1,
        "end_year": 2020,
        "end_month": 12,
        "time_aggregation": "monthly"
    },
    "sources": {
        "clms": {
            "enabled": True,
            "base_s3_path": "s3://eodata/CLMS/",
            "levels": {
                "clc": {
                    "include": True,
                    "variables": {
                        "landcover": True
                    }
                },
                "clcplus": {
                    "include": True,
                    "variables": {
                        "LULUCF-INSTANCE": True
                    }
                },
                "crop_types": {
                    "include": True,
                    "variables": {
                        "CTY": True
                    }
                },
                "dominant_leaf_type": {
                    "include": True,
                    "variables": {
                        "DLT": True
                    }
                },
                "forest_type": {
                    "include": True,
                    "variables": {
                        "FTY": True
                    }
                },
                "grassland_herbaceous": {
                    "include": True,
                    "variables": {
                        "GRA": True,
                        "HER": True
                    }
                },
                "tree_cover_density": {
                    "include": True,
                    "variables": {
                        "TCD": True
                    }
                }
            }
        }
    }
}

# 4. Execute the pipeline
if __name__ == "__main__":
    orchestrator = bmdCube()
    orchestrator.process_recipe(tutorial_cube, creds=active_credentials)