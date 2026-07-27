from .chelsa import chelsa_cube
from .wekeo import wekeo_cube
from xarray import DataTree, open_datatree
import xarray as xr
import os
import yaml
import pandas as pd

class bmd_cube:
    def __init__(self):
        # A single dictionary to hold layers grouped by their source dataset
        self.cube_data = {} 
        self.data_tree = None
        
        # Dispatch table mapping YAML keys to their respective class objects
        self._source_map = {
            "chelsa": chelsa_cube,
            "wekeo": wekeo_cube
            # "gbif": gbif_cube  <-- commented out until refactored
        }

    def _load_recipe(self, recipe_file, recipe_path):
        """Private helper method to load and parse the YAML recipe."""
        recipe_filepath = recipe_file if os.path.isabs(recipe_file) else os.path.join(recipe_path, recipe_file)
        with open(recipe_filepath) as f:
            return yaml.safe_load(f)

    def generate_bmd_data(self, recipe_file, recipe_path, max_workers=10):
        recipe = self._load_recipe(recipe_file, recipe_path)
        sources = recipe.get("sources", {})

        base_dir = recipe.get("base_dir", ".")
        if not os.path.exists(base_dir):
            print(f"Notice: Destination directory '{base_dir}' not found. Creating path...")
            os.makedirs(base_dir, exist_ok=True)

        # Dynamically iterate over the registered sources in our dispatch table
        for source_name, cube_class in self._source_map.items():
            
            # Check if this specific source is present and enabled in the recipe
            source_cfg = sources.get(source_name, {})
            if source_cfg.get("enabled", False):
                print(f"Initializing {source_name.upper()} cube generation...")
                
                # Instantiate the mapped class dynamically
                cube_inst = cube_class()
                
                # process_cube returns a dictionary: Dict[str, xr.Dataset]
                result_dict = cube_inst.process_cube(recipe, max_workers=max_workers)
                
                # Guard against empty dictionaries (e.g., if no assets matched the recipe constraints)
                if not result_dict:
                    print(f"Warning: {source_name.upper()} processing yielded no layers. Check logs for details.")
                    continue
                    
                # Directly assign the returned dictionary to our source-specific branch
                self.cube_data[source_name] = result_dict

        return None
    
    def construct_datatree(self, recipe_file, recipe_path) -> None:
        recipe = self._load_recipe(recipe_file, recipe_path)
        cube_name = recipe.get("cube_name", "bmd_default_cube")
        
        # Assemble a fresh DataTree
        self.data_tree = DataTree(name=cube_name)
        
        # Iterate over our source dictionaries (the new branches)
        for source_name, layers in self.cube_data.items():
            
            # Create a dedicated branch for the data source
            self.data_tree[source_name] = DataTree(name=source_name)
            
            # Attach individual layers as leaves to this branch
            for layer_name, ds in layers.items():
                node = DataTree(dataset=ds, name=layer_name)
                self.data_tree[source_name][layer_name] = node

    def export_tree(self, recipe_file, recipe_path):
        recipe = self._load_recipe(recipe_file, recipe_path)
        
        # Extract base properties dynamically
        base_dir = recipe.get("base_dir", ".")
        cube_name = recipe.get("cube_name", "bmd_default_cube")
        export_cfg = recipe.get("export", {})
        
        as_tree = export_cfg.get("as_tree", False)
        as_nested_dir = export_cfg.get("as_nested_dir", True)

        if not self.data_tree:
            raise ValueError("self.data_tree is empty, please use the self.construct_datatree method to build a tree")

        # =================================================================
        # PRE-PROCESSING: Strip MultiIndexes for NetCDF Compatibility
        # NetCDF cannot serialize pandas MultiIndex objects natively.
        # We must reset them to standard 1D auxiliary coordinate arrays.
        # =================================================================
        for branch in self.data_tree:
            for leaf in self.data_tree[branch]:
                ds = self.data_tree[branch][leaf].ds
                
                # Dynamically find any dimension that is a pandas MultiIndex
                m_indexes = [
                    dim for dim, idx in ds.indexes.items() 
                    if isinstance(idx, pd.MultiIndex)
                ]
                
                if m_indexes:
                    # reset_index flattens the MultiIndex back to standard coordinates
                    self.data_tree[branch][leaf].ds = ds.reset_index(m_indexes)

        # =================================================================
        # EXPORT ROUTINES
        # =================================================================
        
        # Export as a single DataTree NetCDF
        if as_tree:
            os.makedirs(base_dir, exist_ok=True)
            file_path = os.path.join(base_dir, f"{cube_name}.nc")
            self.data_tree.to_netcdf(file_path, format="NETCDF4")
            print(f"Exported DataTree to single file: {file_path}")

        # Export as a nested directory structure
        if as_nested_dir:
            cube_folder = os.path.join(base_dir, cube_name)
            os.makedirs(cube_folder, exist_ok=True)
            
            # Iterate over branches (e.g., 'chelsa', 'wekeo')
            for branch in self.data_tree:
                branch_dir = os.path.join(cube_folder, branch)
                os.makedirs(branch_dir, exist_ok=True)
                
                # Iterate over leaves (the individual xarray datasets)
                for leaf in self.data_tree[branch]:
                    leaf_file_path = os.path.join(branch_dir, f"{leaf}.nc")
                    self.data_tree[branch][leaf].ds.to_netcdf(leaf_file_path, format="NETCDF4")
            print(f"Exported DataTree to nested directory: {cube_folder}")

        return None      

    def import_tree(self, target_path: str):
        """
        Imports the DataTree directly from a path. Automatically detects whether 
        the target is a single NetCDF file or a nested directory structure, 
        and restores any flattened MultiIndexes.
        """
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"The specified path does not exist: {target_path}")

        # =================================================================
        # 1. IMPORT ROUTINES
        # =================================================================
        if os.path.isfile(target_path) and target_path.endswith(".nc"):
            self.data_tree = open_datatree(target_path, format="NETCDF4")
            print(f"Imported DataTree from single file: {target_path}")
            
        elif os.path.isdir(target_path):
            cube_name = os.path.basename(os.path.normpath(target_path))
            self.data_tree = DataTree(name=cube_name)
            
            for branch in sorted(os.listdir(target_path)):
                branch_path = os.path.join(target_path, branch)
                if not os.path.isdir(branch_path):
                    continue
                
                self.data_tree[branch] = DataTree(name=branch)
                
                for leaf_file in sorted(os.listdir(branch_path)):
                    if leaf_file.endswith(".nc"):
                        leaf_name = leaf_file.replace(".nc", "")
                        leaf_path = os.path.join(branch_path, leaf_file)
                        
                        ds = xr.open_dataset(leaf_path, engine="netcdf4")
                        self.data_tree[branch][leaf_name] = DataTree(name=leaf_name, dataset=ds)
                        
            print(f"Imported DataTree from nested directory: {target_path}")
            
        else:
            raise ValueError("Target path must be either a .nc file or a valid directory.")

        # =================================================================
        # 2. POST-PROCESSING: Restore MultiIndexes
        # Automatically detects 1D auxiliary coordinates mapped to the 
        # 'projection' dimension and zips them back into a MultiIndex.
        # =================================================================
        for branch in self.data_tree:
            for leaf in self.data_tree[branch]:
                ds = self.data_tree[branch][leaf].ds
                
                # Check if this specific dataset uses the 'projection' framework
                if 'projection' in ds.dims:
                    
                    # Dynamically find all 1D coordinates that belong solely to 'projection'
                    multi_coords = [
                        coord_name for coord_name, coord_da in ds.coords.items()
                        if list(coord_da.dims) == ['projection'] and coord_name != 'projection'
                    ]
                    
                    # If we found auxiliary coordinates, rebuild the MultiIndex!
                    if multi_coords:
                        self.data_tree[branch][leaf].ds = ds.set_index(projection=multi_coords)

        return self.data_tree
    """
    def import_tree(self, cube_dir, cube_name):
        cube_filepath = os.path.join(cube_dir, cube_name)
        self.data_tree = open_datatree(cube_filepath, format="NETCDF4")
        for branch in self.data_tree:
            for leaf in self.data_tree[branch]:
                for attr in self.data_tree[branch][leaf].ds.attrs:
                    if attr=="__orig_var_name__":
                        self.data_tree[branch][leaf].ds = self.load_sparse(self.data_tree[branch][leaf].ds)
        return self.data_tree

    Currently we will focus on using the data tree in the prototype endproduct

    def export_group(self):
        out_dir = os.path.join(self.cube_dir, self.cube_name)
        os.makedirs(out_dir, exist_ok=True)
        #Iterate over both dictionairies in the same way
        for (group, datasets), (group, names) in zip(self.groups.items(), self.group_names.items()):
            group_path = os.path.join(out_dir, group)
            os.makedirs(group_path, exist_ok=True)
            for name, ds in zip(names, datasets):
                file_name = self.cube_name+"_"+name
                file_path = os.path.join(group_path, file_name)
                file_path+=".nc"
                for name, da, in ds.data_vars.items():
                ds.to_netcdf(file_path, mode="w", format="NETCDF4")

    def import_group(self, cube_name, cube_path):
        # Build and validate root folder
        root = os.path.join(cube_path, cube_name)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"No folder at {root!r}")
        group = {}
        group_names = {}
        # Iterate over subfolders (groups)
        for group in sorted(os.listdir(root)):
            gdir = os.path.join(root, group)
            if not os.path.isdir(gdir):
                continue
            group[group] = []
            group_names[group] = []
            # Iterate over files in each group
            for fname in sorted(os.listdir(gdir)):
                base, ext = os.path.splitext(fname)
                prefix = f"{cube_name}_"
                # Skip non-NetCDF or wrong prefix
                if ext.lower() != ".nc" or not base.startswith(prefix):
                    continue
                key = base[len(prefix):]
                full = os.path.join(gdir, fname)
                # Open NetCDF
                ds = xr.open_dataset(full,engine="netcdf4",decode_cf=True,decode_times=True,use_cftime=True)
                group_names[group].append(key)
                group[group].append(ds)
        return group, group_names
    """