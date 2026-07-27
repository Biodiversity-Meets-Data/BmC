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
    # ~10m at the equator (0.3 arc-seconds)
    "Global_WGS84_0_3sec": {"crs": "EPSG:4326", "resolution": 0.00008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
    # ~100m at the equator (3 arc-seconds)
    "Global_WGS84_3sec": {"crs": "EPSG:4326", "resolution": 0.0008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
    # ~250m at the equator (7.5 arc-seconds)
    "Global_WGS84_7_5sec": {"crs": "EPSG:4326", "resolution": 0.0020833333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
    # ~500m at the equator (15 arc-seconds)
    "Global_WGS84_15sec": {"crs": "EPSG:4326", "resolution": 0.004166666666666667, "bounds": (-180.0, -90.0, 180.0, 90.0)},
    # ~1km at the equator (30 arc-seconds)
    "Global_WGS84_30sec": {"crs": "EPSG:4326", "resolution": 0.008333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)},
    # ~10km at the equator (5 arc-minutes)
    "Global_WGS84_5min": {"crs": "EPSG:4326", "resolution": 0.08333333333333333, "bounds": (-180.0, -90.0, 180.0, 90.0)}}

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
        grid_key = f"{target_grid}_{target_resolution}"
        
        if grid_key not in self.GRID_REGISTRY:
            available = "\n - ".join(self.GRID_REGISTRY.keys())
            error_msg = (
                f"\n[Spatial Config Error] Attempted to build grid key '{grid_key}', "
                f"but it does not exist in the registry.\n\n"
                f"Available Grids:\n - {available}"
            )
            # Log the critical error before stopping execution
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

        This universal method resolves target grid geometries against the internal registry, 
        densifies the outer perimeter using vectorized linear interpolation to capture 
        projection curvature, applies spatial transformation to the native source coordinate 
        space, and buffers the resulting envelope outward to supply sufficient edge pixels 
        for multi-pixel GDAL resampling kernels.

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

        Examples
        --------
        Case 1: Standard Master Grid Extraction
        Deriving the fetching envelope for an entire master grid extent. By omitting 
        ``target_bounds``, the method automatically defaults to the full definitive bounds 
        of the requested target grid registry entry.

        >>> safe_wgs_envelope = cube.build_safe_fetch_envelope(
        ...     target_grid_name="EEA_1km",
        ...     source_crs_or_grid="Global_WGS84_30sec",
        ...     pixel_buffer=5,
        ...     logger=pipeline_logger
        ... )
        >>> print(safe_wgs_envelope)
        (-11.45833, 34.04166, 31.95833, 72.87500)

        Case 2: Localized Sub-Region Ingestion
        Calculating a highly precise, buffered source envelope for a specific localized subset 
        (e.g., a localized study area in Belgium defined in metric EPSG:3035 coordinates). 
        Adding a 5-pixel buffer safely feeds downstream cubic or average C++ warping kernels.

        >>> study_area_3035 = (3800000, 2900000, 3900000, 3000000)
        >>> safe_subset_envelope = cube.build_safe_fetch_envelope(
        ...     target_grid_name="EEA_100m",
        ...     target_bounds=study_area_3035,
        ...     source_crs_or_grid="Global_WGS84_30sec",
        ...     pixel_buffer=5
        ... )
        >>> print(safe_subset_envelope)
        (4.19833, 50.71833, 5.66833, 51.65500)

        Case 3: Custom Source Resolution Fallback
        Querying an unlisted custom CRS string while manually supplying the native source 
        pixel spacing to ensure exact metric-to-degree buffer translation.

        >>> custom_envelope = cube.build_safe_fetch_envelope(
        ...     target_grid_name="Global_EqualArea_1km",
        ...     source_crs_or_grid="EPSG:4326",
        ...     source_resolution=0.0041666667,  # ~500m source pixels
        ...     pixel_buffer=8
        ... )
        """
        log_execution(logger, f"Computing safe fetch envelope for target grid '{target_grid_name}'...", logging.INFO)

        # Resolve Target Grid Configurations
        if target_grid_name not in self.GRID_REGISTRY:
            raise KeyError(f"Target grid '{target_grid_name}' not found in GRID_REGISTRY.")
            
        target_spec = self.GRID_REGISTRY[target_grid_name]
        target_crs = target_spec["crs"]
        
        if target_bounds is None:
            target_bounds = target_spec["bounds"]
            log_execution(logger, "Specific target_bounds omitted. Encapsulating full master grid extent.", logging.INFO)

        # Resolve Source Data Configurations
        if source_crs_or_grid in self.GRID_REGISTRY:
            src_spec = self.GRID_REGISTRY[source_crs_or_grid]
            actual_source_crs = src_spec["crs"]
            if source_resolution is None:
                source_resolution = src_spec["resolution"]
        else:
            actual_source_crs = source_crs_or_grid
            if source_resolution is None:
                # Fallback check: Assume CHELSA/WorldClim ~30 arc-second base spacing
                source_resolution = self.GRID_REGISTRY.get("Global_WGS84_30sec", {}).get(
                    "resolution", 0.008333333333333333
                )
                log_execution(
                    logger,
                    f"source_resolution omitted for custom CRS '{actual_source_crs}'. "
                    f"Applying default 30 arc-second fallback: {source_resolution}",
                    logging.WARNING
                )

        # 3. Vectorized Perimeter Densification
        minx, miny, maxx, maxy = target_bounds
        num_points = 100  # Granularity per edge

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

        # 6. Apply Geographic Domain Guardrails
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

        This method acts as a spatial blueprint generator. It takes a loose, localized 
        bounding box and forces it to expand outward until its edges perfectly intersect 
        the integer-aligned pixel boundaries of the master grid. It then generates a 2D 
        matrix of zeros (with embedded CF-compliant spatial coordinates) that downstream 
        functions can use as a canvas for raster reprojection or vector fractional burning.

        Parameters
        ----------
        sample_bbox : tuple of float
            The localized region of interest bounds in the format 
            ``(minx, miny, maxx, maxy)``. These bounds must already be projected 
            into the native CRS of the target grid.
        grid_name : str
            The precise dictionary key corresponding to the target grid defined 
            in the ``GRID_REGISTRY`` (e.g., "EEA_1km").

        Returns
        -------
        template : xarray.DataArray
            A 2D spatial matrix filled with zeros (dtype: int32). Embedded attributes 
            include the mathematically derived `x` and `y` pixel center coordinates, 
            the rioxarray spatial reference topology, and critical metadata strings 
            (``crs``, ``res``, ``spatial_unit``).
        aligned_bbox : tuple of float
            The newly expanded, grid-snapped bounding box in the format 
            ``(aligned_minx, aligned_miny, aligned_maxx, aligned_maxy)``.

        Raises
        ------
        KeyError
            If the requested ``grid_name`` does not exist within the class 
            ``GRID_REGISTRY``.

        Notes
        -----
        The mathematical snapping relies on the absolute origin (``master_minx``, 
        ``master_miny``) defined in the registry. 
        
        It utilizes ``math.floor()`` for the minimum coordinates and ``math.ceil()`` 
        for the maximum coordinates. This guarantees that the localized bounding box 
        only ever grows outward, preventing edge-starvation where geometries resting 
        on the absolute boundary might otherwise be clipped.
        """
        if grid_name not in self.GRID_REGISTRY:
            raise KeyError(f"Grid '{grid_name}' not found in registry.")
            
        master = self.GRID_REGISTRY[grid_name]
        res = master["resolution"]
        master_minx, master_miny, master_maxx, master_maxy = master["bounds"]
        
        # sample_bbox is (minx, miny, maxx, maxy)
        s_minx, s_miny, s_maxx, s_maxy = sample_bbox
        
        # 1. Snap strictly to the Master Grid intervals
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
        
        # 6. Inject CF-compliant spatial topology FIRST
        template = template.rio.write_crs(master["crs"])
        
        # 7. Assign the text attribute explicitly 
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

        This guarantees that the same physical spatial cell will always yield 
        the exact same integer ID, regardless of the localized bounding box 
        or processing chunk being evaluated.

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
        
        # Calculate total absolute columns in the entire master grid
        total_global_cols = int(round((master_maxx - master_minx) / res))
        
        # Calculate absolute row/col offsets from the top-left master origin
        global_cols = np.floor((x_coords - master_minx) / res).astype(np.int64)
        global_rows = np.floor((master_maxy - y_coords) / res).astype(np.int64)
        
        # Flatten into a rigid, globally consistent 1D index
        global_grid_ids = (global_rows * total_global_cols) + global_cols
        
        return global_grid_ids