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
        "gbif": {
            "local_file_path": None,
            "query_filters": {
                "fetch_all_taxa": True,
                "record_type": "presence",
                "default_Uncertainty": 1000,
                "max_uncertainty": 5000,
                "exclude_issues": [
                    "ZERO_COORDINATE",
                    "COORDINATE_OUT_OF_RANGE",
                    "COUNTRY_COORDINATE_MISMATCH"
                ]
            },
            "taxonomy": {
                "lowest_level": None
            },
            "columns": ["year", "month", "recordedby"],
            "time_cols": ["year", "month"],
            "processing_mode": "vector",
            "vector_processing": {
                "topology": "point_cloud",
                "mapping_mode": "classify",
                "spatial_method": "intersect",
                "topology_config": {
                    "point_cloud": {
                        "n_passes": 200,
                        "distribution": "uniform",
                        "random_seed": 42
                    }
                }
            },
            "aggregate": {
                "export_unaggregated": True,
                "group_by_columns": ["year", "month", "speciesKey"],
                "metrics": [
                    {
                        "column": "speciesKey",
                        "method": "count",
                        "weighted": False,
                        "rename": "occurrences"
                    },
                    {
                        "column": "recordedBy",
                        "method": "nunique",
                        "weighted": False,
                        "rename": "distinct_observers"
                    }
                ]
            }
        },
        "chelsa": {
            "enabled": True,
            "catalog_path": "C:/Users/niels/Documents/Repositories/BmC/meta/chelsa_gp_stac/chelsa_master.parquet",
            "levels": {
                "monthly": {
                    "include": True,
                    "variables": {
                        "tas": True,
                        "tasmax": True,
                        "tasmin": True
                    }
                },
                "annual": {
                    "include": True,
                    "variables": {
                        "qkndvi": True,
                        "mymd": True
                    }
                },
                "climatologies": {
                    "include": True,
                    "time_ranges": {
                        "1981-2010": True,
                        "2011-2040": True,
                        "2041-2070": True
                    },
                    "ensembles": {
                        "historical": True,
                        "GFDL-ESM4": True,
                        "IPSL-CM6A-LR": True,
                        "MPI-ESM1-2-HR": True,
                        "MRI-ESM2-0": True,
                        "UKESM1-0-LL": True
                    },
                    "scenarios": {
                        "historical": True,
                        "ssp126": True,
                        "ssp370": True,
                        "ssp585": True
                    },
                    "variables": {
                        "tas": True,
                        "tasmax": True,
                        "tasmin": True
                    }
                },
                "bioclim": {
                    "include": True,
                    "time_ranges": {
                        "1981-2010": True,
                        "2011-2040": True,
                        "2041-2070": True,
                        "2071-2100": True
                    },
                    "ensembles": {
                        "historical": True,
                        "GFDL-ESM4": True,
                        "IPSL-CM6A-LR": True,
                        "MPI-ESM1-2-HR": True,
                        "MRI-ESM2-0": True,
                        "UKESM1-0-LL": True
                    },
                    "scenarios": {
                        "historical": True,
                        "ssp126": True,
                        "ssp370": True,
                        "ssp585": True
                    },
                    "variables": {
                        "bio01": True,
                        "bio02": True,
                        "bio03": True,
                        "bio04": True,
                        "bio05": True,
                        "bio06": True,
                        "bio07": True,
                        "bio08": True,
                        "bio09": True,
                        "bio10": True,
                        "bio11": True,
                        "bio12": True,
                        "bio13": True,
                        "bio14": True,
                        "bio15": True,
                        "bio16": True,
                        "bio17": True,
                        "bio18": True,
                        "bio19": True,
                    }
                }
            }
        },
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
