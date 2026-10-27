import logging
import os
import getpass
from pathlib import Path
import faulthandler

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bmc.cube.orchestrator.bmd import bmdCube

faulthandler.enable()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Base directory setup
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "tiling_branch"
output_dir.mkdir(parents=True, exist_ok=True)

# Bounding box definition: [long_min, lat_min, long_max, lat_max]
bbox = [4.26888,50.71863,4.59503,50.91390]

# =======================================================================
# RECIPE DEFINITION
# =======================================================================
dev_recipe = {
    "base_dir": str(output_dir),
    "cube_name": "tilingBranch_test_eea250m_netcdf",
    "export_as": {"format": "netcdf"},
    "spatial": {
        "target_grid": "EEA",
        "target_resolution": "250m",
        "use_bbox": True,
        "bbox": {
            "long_min": bbox[0],
            "lat_min": bbox[1],
            "long_max": bbox[2],
            "lat_max": bbox[3],
        },
    },
    "temporal": {
        "start_year": 2018,
        "start_month": 1,
        "end_year": 2018,
        "end_month": 12,
    },
    "sources": {
        "clms": {
            "enabled": True,
            "levels": {
                "clc": {
                    "include": True,
                    "variables": {"landcover": True},
                },
                "clcplus_instance": {
                    "include": True,
                    "variables": {"LULUCF-INSTANCE": True},
                },
                "grassland_and_herbaceous": {
                    "include": True,
                    "variables": {"GRA": True},
                },
                "tree_cover_density": {
                    "include": True,
                    "variables": {"TCD": True},
                },
                "forest_type": {
                    "include": True,
                    "variables": {
                        "FADSL": True,
                        "FTY": True,
                    },
                },
                "crop_types": {
                    "include": True,
                    "variables": {"CTY": True},
                },
            },
        },
        "chelsa": {
            "enabled": True,
            "levels": {
                "monthly": {
                    "include": True,
                    "variables": {
                        "tas": True,
                        "pr": True,
                        "spei12": True,
                        "spi12": True,
                    },
                },
                "annual": {
                    "include": True,
                    "variables": {
                        "qkndvi": True,
                    },
                },
            },  
        }    
    }
}

class_mapping = {
    'CPFLP': {0: 'No fallow land', 1: 'Fallow', 65535: 'Outside area'},
    'CPSCT': {0: 'No annual cropland', 1: 'Short summer CC', 2: 'Long summer CC', 3: 'Short winter CC', 4: 'Long winter CC', 65526: '65526', 65527: '65527', 65530: '65530', 65531: '65531', 65532: '65532', 65533: '65533', 65534: '65534', 65535: 'Outside area'},
    'CTY': {0: 'No Cropland', 1110: 'Wheat', 1120: 'Barley', 1130: 'Maize', 1140: 'Rice', 1150: 'Other Cereals', 1210: 'Fresh Vegetables', 1220: 'Dry Pulses', 1310: 'Potatoes', 1320: 'Sugar Beet', 1410: 'Sunflower', 1420: 'Soybeans', 1430: 'Rapeseed', 1440: 'Flax cotton and hemp', 2100: 'Grapes', 2200: 'Olives', 2310: 'Fruits', 2320: 'Nuts', 3100: 'Unclassified arable crop', 3200: 'Unclassified permanent crop', 65535: 'Outside area'},
    'DLT': {0: 'all non-tree covered areas', 1: 'broadleaved trees', 2: 'coniferous trees', 255: 'outside area'},
    'DLTC': {0: 'unchanged areas with no tree cover', 1: 'new broadleaved cover', 2: 'new coniferous cover', 3: 'loss of broadleaved cover', 4: 'loss of coniferous cover', 10: 'unchanged areas with tree cover', 255: 'outside area'},
    'FTY': {0: 'all non-forest areas', 1: 'broadleaved forest', 2: 'coniferous forest', 255: 'outside area'},
    'GRA': {0: 'all non-grassland areas in reference year', 1: 'grassland', 255: 'outside area'},
    'HER': {0: 'non-grassland in reference year', 1: 'permanent and temporary grassland in reference year', 255: 'outside area'},
    'LULUCF-INSTANCE': {11: 'S Burnt areas', 12: 'S Settlements', 13: 'S Green urban areas', 14: 'S Other settlements', 21: 'FL Burnt areas', 22: 'FL Transitional woodland', 23: 'FL Deciduous', 24: 'FL Coniferous/Evergreen', 25: 'FL Other forestland', 31: 'CL Burnt areas', 32: 'CL Annual crops', 33: 'CL Perennial crops', 34: 'CL Other cropland', 41: 'GL Burnt areas', 42: 'GL Pastures', 43: 'GL Shrubs', 44: 'GL Natural grassland', 45: 'GL Other grassland', 51: 'WL Wetland managed', 52: 'WL Wetland unmanaged', 53: 'WL Water managed', 54: 'WL Water unmanaged', 55: 'WL Burnt areas', 61: 'OL Bare soil and rocks', 62: 'OL Permanent snow and ice', 63: 'OL Lichens and mosses', 64: 'OL Other land', 254: 'Unclassified', 255: 'No data'},
    'PLOUGH': {0: 'Indication of ploughing in current year', 1: '1 year since last indication of ploughing', 2: '2 years since last indication of ploughing', 3: '3 years since last indication of ploughing', 4: '4 years since last indication of ploughing', 5: '5 years since last indication of ploughing', 6: '6 years since last indication of ploughing', 100: 'Change of herbaceous cover', 253: 'no ploughing information', 255: 'outside area'},
    'landcover': {1: 'Continuous urban fabric', 2: 'Discontinuous urban fabric', 3: 'Industrial or commercial units', 4: 'Road and rail networks and associated land', 5: 'Port areas', 6: 'Airports', 7: 'Mineral extraction sites', 8: 'Dump sites', 9: 'Construction sites', 10: 'Green urban areas', 11: 'Sport and leisure facilities', 12: 'Non-irrigated arable land', 13: 'Permanently irrigated land', 14: 'Rice fields', 15: 'Vineyards', 16: 'Fruit trees and berry plantations', 17: 'Olive groves', 18: 'Pastures', 19: 'Annual crops associated with permanent crops', 20: 'Complex cultivation patterns', 21: 'Land principally occupied by agriculture with significant areas of natural vegetation', 22: 'Agro-forestry areas', 23: 'Broad-leaved forest', 24: 'Coniferous forest', 25: 'Mixed forest', 26: 'Natural grasslands', 27: 'Moors and heathland', 28: 'Sclerophyllous vegetation', 29: 'Transitional woodland-shrub', 30: 'Beaches - dunes - sands', 31: 'Bare rocks', 32: 'Sparsely vegetated areas', 33: 'Burnt areas', 34: 'Glaciers and perpetual snow', 35: 'Inland marshes', 36: 'Peat bogs', 37: 'Salt marshes', 38: 'Salines', 39: 'Intertidal flats', 40: 'Water courses', 41: 'Water bodies', 42: 'Coastal lagoons', 43: 'Estuaries', 44: 'Sea and ocean', 48: 'NODATA'}
}

if __name__ == "__main__":
    # =======================================================================
    # CREDENTIALS PROMPT
    # =======================================================================


    cdse_creds = Path.home() / "cdse.json"
    private_creds = Path.home() / "bmd.json"

    active_credentials = {
        "cdse_s3_credentials": str(cdse_creds),
        "private_s3_credentials": str(private_creds)
    }

    # =======================================================================
    # EXECUTION
    # =======================================================================
    logger.info("Initializing orchestrator...")
    orchestrator = bmdCube()

    logger.info("Processing recipe for cube: %s", dev_recipe["cube_name"])
    orchestrator.process_recipe(dev_recipe, creds=active_credentials)
    logger.info("Cube generation finished successfully.")

    # =======================================================================
    # VISUALIZATION
    # =======================================================================
    cube_dir = output_dir / dev_recipe["cube_name"]
    img_dir = cube_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    nc_files = list(cube_dir.rglob("*.nc"))
    logger.info(f"Found {len(nc_files)} NetCDF files to visualize.")

    for nc_path in nc_files:
        var_name = nc_path.stem
        
        with xr.open_dataset(nc_path) as ds:
            actual_var = var_name if var_name in ds else list(ds.data_vars)[0]
            da = ds[actual_var].load()

        fig, ax = plt.subplots(figsize=(10, 8))
        title_suffix = ""
        
        # CATEGORICAL DATA (CLMS Class Fractions)
        if "class_id" in da.dims:
            valid_classes = [c for c in da.coords["class_id"].values if c not in [0, 255, 65535]]
            da_valid = da.sel(class_id=valid_classes) if valid_classes else da
            
            plot_da = da_valid.idxmax(dim="class_id", skipna=True)
            max_fraction = da_valid.max(dim="class_id", skipna=True)
            title_suffix = "\n(Dominant Class)"
            
            plot_da = plot_da.where(max_fraction > 0)
            
            alpha_data = max_fraction.values.copy()
            if np.nanmax(alpha_data) > 1.0:
                alpha_data = alpha_data / 100.0
            alpha_data = np.clip(np.nan_to_num(alpha_data, nan=0.0), 0.0, 1.0)
            
            unique_vals = np.unique(plot_da.values[~np.isnan(plot_da.values)])
            unique_vals = sorted([int(v) for v in unique_vals])
            
            mapped_da = xr.zeros_like(plot_da) * np.nan
            for i, val in enumerate(unique_vals):
                mapped_da = xr.where(plot_da == val, i, mapped_da)
                
            cmap_size = max(len(unique_vals), 1)
            cmap = plt.get_cmap("tab20", cmap_size)
            
            mapped_da.plot(
                ax=ax, cmap=cmap, vmin=-0.5, vmax=cmap_size - 0.5, 
                add_colorbar=False, alpha=alpha_data
            )
            
            mapping_dict = class_mapping.get(var_name, {})
            patches = []
            for i, val in enumerate(unique_vals):
                label_text = mapping_dict.get(val, f"Class {val}")
                patches.append(mpatches.Patch(color=cmap(i), label=f"{val}: {label_text}"))
                
            if patches:
                ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                          fontsize=9, frameon=False, title="Classes")
                plt.subplots_adjust(right=0.7)
                
        # TIME SERIES DATA (CHELSA Monthly)
        elif "time" in da.dims and da.sizes["time"] > 1:
            plot_da = da.mean(dim="time", keep_attrs=True)
            title_suffix = "\n(Annual Average)"
            cmap = "magma" if "tas" in actual_var.lower() else "viridis"
            plot_da.plot(ax=ax, cmap=cmap, add_colorbar=True)
            
        # SINGLE SLICE / CONTINUOUS (TCD, qkndvi, GBIF)
        else:
            plot_da = da.squeeze().where(da.squeeze() > 0)
            cmap = "Greens" if "tcd" in actual_var.lower() else "viridis"
            plot_da.plot(ax=ax, cmap=cmap, add_colorbar=True)

        ax.set_title(f"{actual_var}{title_suffix}", fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("X (EPSG:3035)")
        ax.set_ylabel("Y (EPSG:3035)")

        out_file = img_dir / f"{actual_var}_map.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        logger.info(f"Saved visualization: {out_file.name}")
        
    logger.info("All visual inspections generated in the 'img/' directory.")