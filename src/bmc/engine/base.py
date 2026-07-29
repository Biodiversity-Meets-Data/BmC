import logging
import math
import numpy as np
import xarray as xr
from typing import Optional, Tuple
from pyproj import CRS, Transformer
from bmc.utils.logger import log_execution

class base_spatial_grid():
    """
    The foundational spatial truth of the spatiotemporal pipeline.
    
    This base class acts as the central source of truth for the physical laws 
    of the pipeline. It holds the definitive registry of all supported master grids 
    (Coordinate Reference Systems, resolutions, and absolute bounding boxes). 
    
    By abstracting this into a foundational class, both the raster-based 
    ``spatial_engine`` and the vector-based ``vector_spatial_engine`` inherit 
    the exact same mathematical blueprints, guaranteeing perfect 1-to-1 pixel 
    alignment when bridging continuous and discrete datasets.

    Attributes
    ----------
    GRID_REGISTRY : dict
        A master dictionary mapping human-readable grid keys (e.g., "EEA_10km") 
        to their rigid spatial definitions. Each definition contains:
        * ``crs``: The EPSG code string.
        * ``resolution``: The size of a single pixel in native CRS units.
        * ``bounds``: The definitive absolute extent (minx, miny, maxx, maxy).
    """

    GRID_REGISTRY = {
        # ---------------------------------------------------------
        # EEA Reference Grid (EPSG:3035) - Metric
        # ---------------------------------------------------------
        "EEA_10m": {"crs": "EPSG:3035", "resolution": 10, "bounds": (2000000, 1000000, 6000000, 5500000)},
        "EEA_100m": {"crs": "EPSG:3035", "resolution": 100, "bounds": (2000000, 1000000, 6000000, 5500000)},
        "EEA_250m": {"crs": "EPSG:3035", "resolution": 250, "bounds": (2000000, 1000000, 6000000, 5500000)},
        "EEA_500m": {"crs": "EPSG:3035", "resolution": 500, "bounds": (2000000, 1000000, 6000000, 5500000)},
        "EEA_1km":  {"crs": "EPSG:3035", "resolution": 1000, "bounds": (2000000, 1000000, 6000000, 5500000)},
        "EEA_10km": {"crs": "EPSG:3035", "resolution": 10000, "bounds": (2000000, 1000000, 6000000, 5500000)},

        # ---------------------------------------------------------
        # Global Equal Area (EPSG:6933) - Metric
        # ---------------------------------------------------------
        "Global_EqualArea_10m": {"crs": "EPSG:6933", "resolution": 10, "bounds": (-17367530, -7314540, 17367530, 7314540)},
        "Global_EqualArea_100m": {"crs": "EPSG:6933", "resolution": 100, "bounds": (-17367530, -7314540, 17367530, 7314540)},
        "Global_EqualArea_250m": {"crs": "EPSG:6933", "resolution": 250, "bounds": (-17367530, -7314540, 17367530, 7314540)},
        "Global_EqualArea_500m": {"crs": "EPSG:6933", "resolution": 500, "bounds": (-17367530, -7314540, 17367530, 7314540)},
        "Global_EqualArea_1km":  {"crs": "EPSG:6933", "resolution": 1000, "bounds": (-17367530, -7314540, 17367530, 7314540)},
        "Global_EqualArea_10km": {"crs": "EPSG:6933", "resolution": 10000, "bounds": (-17367530, -7314540, 17367530, 7314540)},

        # ---------------------------------------------------------
        # Global WGS84 (EPSG:4326) - Decimal Degrees
        # ---------------------------------------------------------
        "Global_WGS84_0_3sec": {"crs": "EPSG:4326", "resolution": 0.00008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
        "Global_WGS84_3sec": {"crs": "EPSG:4326", "resolution": 0.0008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
        "Global_WGS84_7_5sec": {"crs": "EPSG:4326", "resolution": 0.0020833333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
        "Global_WGS84_15sec": {"crs": "EPSG:4326", "resolution": 0.004166666666666667, "bounds": (-180.0, -90.0, 180.0, 90.0)},
        "Global_WGS84_30sec": {"crs": "EPSG:4326", "resolution": 0.008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
        "Global_WGS84_5min": {"crs": "EPSG:4326", "resolution": 0.08333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)}
    }

    def resolve_grid_registry_key(
        self, 
        target_grid: str, 
        target_resolution: str, 
        logger: Optional[logging.Logger] = None
        ) -> str:
        """
        Dynamically constructs and validates the master grid key from user configuration.

        Parameters
        ----------
        target_grid : str
            The base coordinate reference system identifier (e.g., "EEA", "Global_WGS84").
        target_resolution : str
            The spatial resolution string (e.g., "100m", "10km", "30sec").
        logger : logging.Logger, optional
            The logger instance to record the error if the key doesn't exist. Default is None.

        Returns
        -------
        grid_key : str
            The validated dictionary key used to access `self.GRID_REGISTRY`.

        Raises
        ------
        ValueError
            If the concatenated string does not match a predefined grid.
        """
        # Construct the key
        grid_key = f"{target_grid}_{target_resolution}"
        
        # Validate existence in the registry
        if grid_key not in self.GRID_REGISTRY:
            available = "\n - ".join(self.GRID_REGISTRY.keys())
            error_msg = (
                f"\n[Spatial Config Error] Attempted to build grid key '{grid_key}', "
                f"but it does not exist in the registry.\n\n"
                f"Available Grids:\n - {available}"
            )
            log_execution(logger, error_msg, logging.ERROR)
            raise ValueError(error_msg)    
            
        return grid_key

    def build_safe_fetch_envelope(
        self,
        target_grid_name: str,
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
        source_crs_or_grid: str = "EPSG:4326",
        source_resolution: Optional[float] = None,
        pixel_buffer: int = 5,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[float, float, float, float]:
        """
        Constructs a densified, buffered source envelope guaranteed to fully encapsulate 
        a target grid region without causing edge starvation or NaN boundary artifacts.

        Parameters
        ----------
        target_grid_name : str
            The key of the destination grid defined in `GRID_REGISTRY` (e.g., "EEA_1km").
        target_bounds : tuple of float, optional
            Specific sub-region bounding box in target CRS units: (minx, miny, maxx, maxy). 
            If omitted, defaults to the master grid's full definitive bounds.
        source_crs_or_grid : str, optional
            Either a target key from `GRID_REGISTRY` (e.g., "Global_WGS84_30sec") or a 
            standard CRS string (e.g., "EPSG:4326"). Default is "EPSG:4326".
        source_resolution : float, optional
            The size of a single source pixel in native source CRS units. Automatically 
            inferred if `source_crs_or_grid` exists in the registry.
        pixel_buffer : int, optional
            Number of native source pixels added as an outer safety padding to support 
            multi-pixel GDAL resampling kernels. Default is 5.
        logger : logging.Logger, optional
            Logger instance for recording execution metadata. Default is None.

        Returns
        -------
        tuple of float
            The safe outer envelope in the source coordinate space: 
            (src_minx, src_miny, src_maxx, src_maxy).

        Raises
        ------
        KeyError
            If the requested `target_grid_name` does not exist in the registry.
        ValueError
            If spatial transformation yields entirely non-finite coordinates.
        """
        log_execution(logger, f"Computing safe fetch envelope for target grid '{target_grid_name}'...", logging.INFO)

        # 1. Resolve Target Grid Configurations
        if target_grid_name not in self.GRID_REGISTRY:
            raise KeyError(f"Target grid '{target_grid_name}' not found in GRID_REGISTRY.")
            
        target_spec = self.GRID_REGISTRY[target_grid_name]
        target_crs = target_spec["crs"]
        
        if target_bounds is None:
            target_bounds = target_spec["bounds"]
            log_execution(logger, "Specific target_bounds omitted. Encapsulating full master grid extent.", logging.INFO)

        # 2. Resolve Source Data Configurations
        if source_crs_or_grid in self.GRID_REGISTRY:
            src_spec = self.GRID_REGISTRY[source_crs_or_grid]
            actual_source_crs = src_spec["crs"]
            if source_resolution is None:
                source_resolution = src_spec["resolution"]
        else:
            actual_source_crs = source_crs_or_grid
            if source_resolution is None:
                source_resolution = self.GRID_REGISTRY.get("Global_WGS84_30sec", {}).get(
                    "resolution", 0.008333333333333333
                )
                log_execution(
                    logger,
                    f"source_resolution omitted for custom CRS '{actual_source_crs}'. "
                    f"Applying default fallback: {source_resolution}",
                    logging.WARNING
                )

        # 3. Vectorized Perimeter Densification (Creates points along the boundary)
        minx, miny, maxx, maxy = target_bounds
        num_points = 100 

        bx = np.linspace(minx, maxx, num_points)
        by = np.full(num_points, miny)

        rx = np.full(num_points, maxx)
        ry = np.linspace(miny, maxy, num_points)

        tx = np.linspace(maxx, minx, num_points)
        ty = np.full(num_points, maxy)

        lx = np.full(num_points, minx)
        ly = np.linspace(maxy, miny, num_points)

        perimeter_x = np.concatenate([bx, rx, tx, lx])
        perimeter_y = np.concatenate([by, ry, ty, ly])

        # 4. Perform Coordinate Transformation
        transformer = Transformer.from_crs(target_crs, actual_source_crs, always_xy=True)
        src_x, src_y = transformer.transform(perimeter_x, perimeter_y)

        # Validate projection output
        valid_mask = np.isfinite(src_x) & np.isfinite(src_y)
        if not np.any(valid_mask):
            raise ValueError(
                f"Failed to project target bounds from {target_crs} to {actual_source_crs}. "
                "Ensure target coordinates fall within allowable projection definitions."
            )
            
        src_x, src_y = src_x[valid_mask], src_y[valid_mask]

        src_minx, src_maxx = float(np.min(src_x)), float(np.max(src_x))
        src_miny, src_maxy = float(np.min(src_y)), float(np.max(src_y))

        # 5. Apply Resampling Safety Buffer
        buffer_padding = source_resolution * pixel_buffer
        
        safe_minx = src_minx - buffer_padding
        safe_maxx = src_maxx + buffer_padding
        safe_miny = src_miny - buffer_padding
        safe_maxy = src_maxy + buffer_padding

        # 6. Apply Geographic Domain Guardrails (Restricts coordinates to valid degree bounds if needed)
        src_crs_obj = CRS.from_string(actual_source_crs)
        if src_crs_obj.is_geographic:
            safe_minx = max(-180.0, safe_minx)
            safe_maxx = min(180.0, safe_maxx)
            safe_miny = max(-90.0, safe_miny)
            safe_maxy = min(90.0, safe_maxy)

        log_execution(
            logger,
            f"Safe Source Envelope ({actual_source_crs}): "
            f"({safe_minx:.5f}, {safe_miny:.5f}, {safe_maxx:.5f}, {safe_maxy:.5f})",
            logging.INFO
        )
            
        return (safe_minx, safe_miny, safe_maxx, safe_maxy)

    def create_aligned_raster_template(
        self, 
        sample_bbox: Tuple[float, float, float, float], 
        grid_name: str
    ) -> Tuple[xr.DataArray, Tuple[float, float, float, float]]:
        """
        Generates an empty, mathematically rigid xarray DataArray template perfectly 
        snapped to a predefined master grid.

        Parameters
        ----------
        sample_bbox : tuple of float
            The localized region of interest bounds in the format 
            ``(minx, miny, maxx, maxy)``. 
        grid_name : str
            The precise dictionary key corresponding to the target grid defined 
            in the ``GRID_REGISTRY`` (e.g., "EEA_1km").

        Returns
        -------
        template : xarray.DataArray
            A 2D spatial matrix filled with zeros (dtype: int32). 
        aligned_bbox : tuple of float
            The newly expanded, grid-snapped bounding box in the format 
            ``(aligned_minx, aligned_miny, aligned_maxx, aligned_maxy)``.

        Raises
        ------
        KeyError
            If the requested ``grid_name`` does not exist within the class 
            ``GRID_REGISTRY``.
        """
        if grid_name not in self.GRID_REGISTRY:
            raise KeyError(f"Grid '{grid_name}' not found in registry.")
            
        master = self.GRID_REGISTRY[grid_name]
        res = master["resolution"]
        master_minx, master_miny, master_maxx, master_maxy = master["bounds"]
        
        s_minx, s_miny, s_maxx, s_maxy = sample_bbox
        
        # 1. Snap strictly to the Master Grid intervals (growing outward)
        aligned_minx = master_minx + math.floor((s_minx - master_minx) / res) * res
        aligned_miny = master_miny + math.floor((s_miny - master_miny) / res) * res
        aligned_maxx = master_minx + math.ceil((s_maxx - master_minx) / res) * res
        aligned_maxy = master_miny + math.ceil((s_maxy - master_miny) / res) * res
        
        # 2. Calculate integer dimensions safely
        width = max(1, int(round((aligned_maxx - aligned_minx) / res)))
        height = max(1, int(round((aligned_maxy - aligned_miny) / res)))
        
        # 3. Generate spatial coordinates (Pixel Centers)
        x_coords = aligned_minx + (np.arange(width) + 0.5) * res
        y_coords = aligned_maxy - (np.arange(height) + 0.5) * res
        
        # 4. Dynamically determine spatial units from the CRS
        crs_obj = CRS.from_string(master["crs"])
        spatial_unit = "degrees" if crs_obj.is_geographic else "meters"
        
        # 5. Create the DataArray template with robust metadata attributes
        template = xr.DataArray(
            data=np.zeros((height, width), dtype=np.int32), 
            coords={"y": y_coords, "x": x_coords},
            dims=("y", "x"),
            attrs={
                "grid_registry_key": grid_name,
                "res": res,
                "spatial_unit": spatial_unit
            }
        )
        
        # 6. Inject CF-compliant spatial topology
        template = template.rio.write_crs(master["crs"])
        template.attrs["crs"] = str(master["crs"])
        
        return template, (aligned_minx, aligned_miny, aligned_maxx, aligned_maxy)

    def calculate_deterministic_global_indices(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        grid_name: str,
        logger: Optional[logging.Logger] = None
    ) -> np.ndarray:
        """
        Converts local 2D spatial coordinates into a deterministic, globally 
        consistent 1D index based on a master grid's absolute origin.

        Parameters
        ----------
        x_coords : numpy.ndarray
            A 1D array or flattened mesh of pixel X centers (easting/longitude).
        y_coords : numpy.ndarray
            A 1D array or flattened mesh of pixel Y centers (northing/latitude).
        grid_name : str
            The key of the master grid defined in `GRID_REGISTRY` to fetch 
            the absolute bounds and resolution.
        logger : logging.Logger, optional
            Logger instance for execution tracking.

        Returns
        -------
        numpy.ndarray
            An array of int64 global indices.
        """
        if grid_name not in self.GRID_REGISTRY:
            raise KeyError(f"Target grid '{grid_name}' not found in GRID_REGISTRY.")
            
        master_spec = self.GRID_REGISTRY[grid_name]
        res = master_spec["resolution"]
        master_minx, _, master_maxx, master_maxy = master_spec["bounds"]
        
        # 1. Calculate total absolute columns in the entire master grid
        total_global_cols = int(round((master_maxx - master_minx) / res))
        
        # 2. Calculate absolute row/col offsets from the top-left master origin
        global_cols = np.floor((x_coords - master_minx) / res).astype(np.int64)
        global_rows = np.floor((master_maxy - y_coords) / res).astype(np.int64)
        
        # 3. Flatten into a rigid, globally consistent 1D index
        global_grid_ids = (global_rows * total_global_cols) + global_cols
        
        return global_grid_ids