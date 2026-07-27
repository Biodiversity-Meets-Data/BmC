import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint
from shapely.geometry.collection import GeometryCollection
from shapely import force_2d
import logging
from typing import Optional


class spatial_vector_engine(base_spatial_grid):
    def coordinate_to_geometry(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        uncert_col: Optional[str] = None,
        output_type: str = "polygon",
        input_crs: str = "EPSG:4326",
        on_missing_uncertainty: str = "fallback",
        quad_segs: int = 8,
        point_cloud_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> gpd.GeoDataFrame:
        """
        Converts tabular spatial records into a GeoDataFrame, optionally applying
        a geometric buffer or generating Monte Carlo point clouds based on coordinate 
        uncertainty.
    
        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe containing tabular coordinate data and
            associated uncertainty measurements.
        x_col : str
            The exact column name containing X coordinates (Longitude/Easting).
        y_col : str
            The exact column name containing Y coordinates (Latitude/Northing).
        uncert_col : str, optional
            The column name containing coordinate uncertainty in meters.
        output_type : {'point', 'polygon', 'point_cloud'}, optional
            The desired geometric output topology. Default is 'polygon'.
            * 'point': Constructs single Point geometries.
            * 'polygon': Creates uncertainty buffers dynamically in local UTM/UPS zones.
            * 'point_cloud': Delegates to generate_spatial_point_clouds to produce MultiPoint 
              jitter clouds while preserving original centroids.
        input_crs : str, optional
            The EPSG code or identifier for source coordinates. Default is "EPSG:4326".
        on_missing_uncertainty : {'fallback', 'raise'}, optional
            Controls behavior when `output_type` is 'polygon' or 'point_cloud' but 
            `uncert_col` is missing.
            - 'fallback': log a warning and degrade to 'point' geometry generation.
            - 'raise': raise a ValueError immediately.
        quad_segs : int, optional
            Quarter-circle segment resolution for polygon buffering. Default is 8.
        point_cloud_config : dict, optional
            Configuration dictionary for point cloud generation when output_type='point_cloud'.
            Supports both direct parameter keys and STAC processing extension keys:
            - n_passes / processing:n_passes (default: 30)
            - distribution / processing:distribution (default: 'gaussian')
            - random_state / random_seed / processing:random_seed (default: None)
            - output_col (default: 'point_cloud')
        logger : logging.Logger, optional
            Instance of standard Python logger. Default is None.
    
        Returns
        -------
        geopandas.GeoDataFrame
            A transformed GeoDataFrame assigned to the input_crs.
        """
        # =========================================================================
        # PARAMETER VALIDATION
        # =========================================================================
        valid_outputs = ("point", "polygon", "point_cloud")
        if output_type not in valid_outputs:
            raise ValueError(f"output_type must be one of {valid_outputs}, got '{output_type}'")
    
        if on_missing_uncertainty not in ("fallback", "raise"):
            raise ValueError("on_missing_uncertainty must be either 'fallback' or 'raise'")
    
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot construct geometries.")
    
        coord_nan_mask = df[x_col].isna() | df[y_col].isna()
        if coord_nan_mask.any():
            raise ValueError(
                f"Found {int(coord_nan_mask.sum())} row(s) with NaN in '{x_col}' or "
                f"'{y_col}'. Resolve or filter these rows before calling coordinate_to_geometry."
            )
    
        # =========================================================================
        # PIPELINE SAFETY: UNCERTAINTY COLUMN CHECK FOR POLYGON / POINT_CLOUD
        # =========================================================================
        if output_type in ("polygon", "point_cloud"):
            if not uncert_col or uncert_col not in df.columns:
                if on_missing_uncertainty == "raise":
                    raise ValueError(
                        f"uncert_col '{uncert_col}' not found in input data and "
                        f"output_type='{output_type}' was requested. Pass a valid "
                        "uncert_col, or set on_missing_uncertainty='fallback' to "
                        "degrade to point geometry."
                    )
                log_execution(
                    logger,
                    f"Uncertainty column '{uncert_col}' not found in input data. "
                    "Defaulting to 'point' geometry generation.",
                    level=logging.WARNING,
                )
                output_type = "point"
    
        log_execution(
            logger,
            f"Converting tabular coordinates to {output_type.upper()} geometries in {input_crs}...",
            level=logging.INFO,
        )
    
        # =========================================================================
        # INITIALIZE BASE SPATIAL TOPOLOGY
        # =========================================================================
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[x_col], df[y_col]),
            crs=input_crs,
        )
        
        # Drop redundant raw coordinate columns
        gdf = gdf.drop(columns=[x_col, y_col])
    
        # =========================================================================
        # STRATEGY 1: POINT TOPOLOGY
        # =========================================================================
        if output_type == "point":
            log_execution(logger, "Point generation complete.", level=logging.INFO)
            return gdf

        # =========================================================================
        # STRATEGY 2: POINT CLOUD TOPOLOGY
        # =========================================================================
        if output_type == "point_cloud":
            log_execution(logger, "Delegating to generate_spatial_point_clouds...", level=logging.INFO)
            
            # Extract configuration parameters (supporting standard & STAC metadata keys)
            cfg = point_cloud_config or {}
            
            n_passes = cfg.get("n_passes", cfg.get("processing:n_passes", 30))
            distribution = cfg.get("distribution", cfg.get("processing:distribution", "gaussian"))
            random_state = cfg.get("random_state", cfg.get("random_seed", cfg.get("processing:random_seed", None)))
            cloud_out_col = cfg.get("output_col", "point_cloud")

            return self.generate_spatial_point_clouds(
                gdf=gdf,
                n_passes=n_passes,
                uncertainty_col=uncert_col,
                output_col=cloud_out_col,
                distribution=distribution,
                random_state=random_state,
                logger=logger,
            )

        # =========================================================================
        # STRATEGY 3: POLYGON TOPOLOGY (BUFFERING)
        # =========================================================================
        missing_count = gdf[uncert_col].isna().sum()
        if missing_count > 0:
            log_execution(
                logger,
                f"Filling {missing_count} missing uncertainty values with 0m to prevent buffering failure.",
                level=logging.WARNING,
            )
            gdf[uncert_col] = gdf[uncert_col].fillna(0)
    
        negative_mask = gdf[uncert_col] < 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            log_execution(
                logger,
                f"Clipping {negative_count} negative uncertainty value(s) to 0m.",
                level=logging.WARNING,
            )
            gdf[uncert_col] = gdf[uncert_col].clip(lower=0)
    
        log_execution(logger, "Applying metric coordinate uncertainty buffers...", level=logging.INFO)
    
        if gdf.crs.is_geographic:
            log_execution(
                logger,
                "Geographic CRS detected. Dynamically mapping records to localized UTM/UPS zones...",
                level=logging.INFO,
            )
    
            gdf_wgs84 = gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs("EPSG:4326")
    
            longitudes = gdf_wgs84.geometry.x
            latitudes = gdf_wgs84.geometry.y
    
            UPS_NORTH_EPSG = 32661
            UPS_SOUTH_EPSG = 32761
    
            north_polar_mask = latitudes >= 84.0
            south_polar_mask = latitudes < -80.0
            utm_mask = ~(north_polar_mask | south_polar_mask)
    
            zone_epsg = pd.Series(index=gdf.index, dtype="int64")
            zone_epsg.loc[north_polar_mask] = UPS_NORTH_EPSG
            zone_epsg.loc[south_polar_mask] = UPS_SOUTH_EPSG
    
            if utm_mask.any():
                utm_zones = np.clip(
                    ((longitudes[utm_mask] + 180) / 6).astype(int) + 1, 1, 60
                )
                epsg_prefixes = np.where(latitudes[utm_mask] >= 0, 32600, 32700)
                zone_epsg.loc[utm_mask] = epsg_prefixes + utm_zones
    
            zone_col = "_coord_to_geom_zone_epsg"
            gdf[zone_col] = zone_epsg.astype(int)
    
            log_execution(logger, "Batch-processing polygons within localized metric zones...", level=logging.INFO)
            buffered_chunks = []
    
            for zone_epsg_code, group in gdf.groupby(zone_col):
                group_metric = group.to_crs(f"EPSG:{zone_epsg_code}")
                group_metric["geometry"] = group_metric.geometry.buffer(
                    group_metric[uncert_col], resolution=quad_segs
                )
                buffered_chunks.append(group_metric.to_crs(input_crs))
    
            gdf_polygon = gpd.GeoDataFrame(
                pd.concat(buffered_chunks).sort_index(), crs=input_crs
            )
            gdf_polygon = gdf_polygon.drop(columns=[zone_col])
        else:
            log_execution(logger, "Projected CRS detected. Buffering directly in native units.", level=logging.INFO)
            gdf_polygon = gdf.copy()
            gdf_polygon["geometry"] = gdf_polygon.geometry.buffer(
                gdf_polygon[uncert_col], resolution=quad_segs
            )
    
        log_execution(logger, "Coordinate uncertainty polygon generation complete.", level=logging.INFO)
    
        return gdf_polygon    
    
    def sanitize_geometries(
    self, 
    gdf: gpd.GeoDataFrame, 
    allowed_types: Optional[List[str]] = None,
    force_multi: bool = True,
    deduplicate: bool = False,
    make_valid_method: str = "linework",
    logger: Optional[logging.Logger] = None
) -> gpd.GeoDataFrame:
        """
        Cleans, flattens, normalizes, and validates dirty vector geometries using 
        highly optimized vectorized Shapely C-operations where possible.

        Operational Mechanics:
        1. Removes completely null, missing, or structurally empty geometries.
        2. Drops the Z/M coordinates to enforce a strict 2D planar workspace.
        3. Isolates topologically invalid shapes (e.g., self-intersections) and heals them.
        4. Unpacks messy GeometryCollections, extracting only the highest-dimensional assets.
        5. Runs a safety re-validation check to drop unresolvable geometric anomalies.
        6. Homogenizes features into their Multi* variants to prevent downstream schema mismatches.
        7. Optionally drops exact geometric duplicates and filters by case-sensitive geometry types.
        8. Resets the index to guarantee safe table merges and joins downstream.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The raw input vector dataset containing potentially corrupt or mixed geometries.
        allowed_types : list of str, optional
            Case-sensitive geometry types allowed in the final output layer 
            (e.g., ['Polygon', 'MultiPolygon']). **If left empty or None, all valid 
            geometry types are permitted to pass through.**
        force_multi : bool, default True
            If True, normalizes single/atomic geometries to their Multi* counterparts 
            (e.g., Polygon -> MultiPolygon) to ensure index/schema uniformity.
        deduplicate : bool, default False
            If True, executes an exact spatial lookup to drop duplicate geometry rows.
        make_valid_method : str, default 'linework'
            The underlying GEOS algorithm used for fixing broken topologies. 
            Options are 'linework' (standard) or 'structure' (preserves grid lines better).
        logger : logging.Logger, optional
            Active pipeline logger instance for streaming system telemetry.

        Returns
        -------
        geopandas.GeoDataFrame
            A pristine, topologically valid, schema-homogenized dataset with a clean contiguous index.
        """
        log_execution(logger, "Initiating vector geometry sanitization...", level=logging.INFO)
        
        # -------------------------------------------------------------------------
        # STRUCTURAL INPUT GUARDRAILS & TYPO PROTECTION
        # -------------------------------------------------------------------------
        if gdf.empty:
            log_execution(logger, "[WARNING] Input GeoDataFrame is empty. Returning empty copy.", level=logging.WARNING)
            return gdf.copy()

        # Define the strict internal vocabulary recognized by Shapely's .geom_type property
        VALID_GEOM_TYPES = {
            "Point", "MultiPoint", "LineString", "MultiLineString", 
            "Polygon", "MultiPolygon", "GeometryCollection"
        }
        
        # If a filter list is provided, validate it upfront to prevent silent runtime failure
        if allowed_types:
            for t in allowed_types:
                if t not in VALID_GEOM_TYPES:
                    raise ValueError(
                        f"Invalid type '{t}' in allowed_types. Must match standard Shapely case-sensitive "
                        f"vocabulary: {list(VALID_GEOM_TYPES)}"
                    )

        initial_count = len(gdf)
        
        # -------------------------------------------------------------------------
        # PURGE NULLS AND STRUCTURALLY EMPTY GEOMETRIES (THE GHOSTS)
        # -------------------------------------------------------------------------
        # dropna avoids triggering the messy GeoPandas warning when dropping missing geometries
        gdf = gdf.dropna(subset=['geometry'])
        # Strip empty representations like "Polygon()" which contain no vertices
        gdf = gdf[~gdf.geometry.is_empty].copy()
        
        dropped_empty = initial_count - len(gdf)
        if dropped_empty > 0:
            log_execution(logger, f"Dropped {dropped_empty} empty or null geometries.", level=logging.INFO)

        # -------------------------------------------------------------------------
        # VECTORIZED FORCE 2D PLANAR GEOMETRIES (DROP Z/M AXES)
        # -------------------------------------------------------------------------
        # Fast C-Array Operation: Flattens 3D/4D dimensions down to native X and Y coordinates
        gdf.geometry = shapely.force_2d(gdf.geometry.values)

        # -------------------------------------------------------------------------
        # VECTORIZED TOPOLOGY HEALING (THE BOWTIES)
        # -------------------------------------------------------------------------
        # Isolate only the invalid rows to save processing cycles over large datasets
        invalid_mask = ~gdf.geometry.is_valid.values
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            log_execution(logger, f"Healing {invalid_count} topologically invalid geometries via '{make_valid_method}'...", level=logging.INFO)
            # Vectorized array healing; generates structural representations or collections if needed
            healed_geoms = shapely.make_valid(gdf.loc[invalid_mask, 'geometry'].values, method=make_valid_method)
            gdf.loc[invalid_mask, 'geometry'] = healed_geoms

        # -------------------------------------------------------------------------
        # UNPACK GEOMETRYCOLLECTIONS (SCOPED STRICTLY TO HEALED ROW SUBSET)
        # -------------------------------------------------------------------------
        healed_subset = gdf.loc[invalid_mask]
        collection_mask = (healed_subset.geometry.geom_type == "GeometryCollection").values
        
        if collection_mask.any():
            collection_indices = healed_subset.index[collection_mask]
            log_execution(logger, f"Unpacking {len(collection_indices)} complex GeometryCollections...", level=logging.INFO)
            
            audited_dropped_parts = 0
            audited_dropped_features = 0
            updated_geometries = []

            # Iterate only through the row indices flagged as structural collections
            for idx in collection_indices:
                geom = gdf.loc[idx, 'geometry']
                parts = list(geom.geoms)
                
                # Sort individual parts based on their structural complexity/topological dimensions
                polygons = [p for p in parts if p.geom_type in ['Polygon', 'MultiPolygon']]
                lines = [p for p in parts if p.geom_type in ['LineString', 'MultiLineString']]
                points = [p for p in parts if p.geom_type in ['Point', 'MultiPoint']]
                
                selected_geom = None
                dropped_count = 0
                
                # Prioritize Polygons > Lines > Points to pick the dominant dimension
                if polygons:
                    # shapely.get_parts recursively flattens internal MultiPolygons to avoid an atomic list crash
                    flattened_polys = shapely.get_parts(polygons)
                    selected_geom = shapely.MultiPolygon(flattened_polys)
                    dropped_count = len(lines) + len(points)
                elif lines:
                    flattened_lines = shapely.get_parts(lines)
                    selected_geom = shapely.MultiLineString(flattened_lines)
                    dropped_count = len(points)
                elif points:
                    flattened_points = shapely.get_parts(points)
                    selected_geom = shapely.MultiPoint(flattened_points)

                # Keep track of dropped dimension fragments for audit logs
                if dropped_count > 0:
                    audited_dropped_parts += dropped_count
                    audited_dropped_features += 1
                    
                updated_geometries.append(selected_geom)
                
            # Log a warning to make sure any loss of small slivers/lines is completely auditable
            if audited_dropped_parts > 0:
                log_execution(
                    logger, 
                    f"[AUDIT] Discarded {audited_dropped_parts} lower-dimension sliver parts across "
                    f"{audited_dropped_features} split features to preserve topological dimensionality.", 
                    level=logging.WARNING
                )
                
            gdf.loc[collection_indices, 'geometry'] = updated_geometries

        # -------------------------------------------------------------------------
        # PIPELINE SAFETY RE-VALIDATION CHECK
        # -------------------------------------------------------------------------
        # Double-check that the GEOS algorithms didn't generate any unresolvable geometric artifacts
        post_healing_invalid = ~gdf.geometry.is_valid.values
        if post_healing_invalid.any():
            failed_count = post_healing_invalid.sum()
            log_execution(
                logger, 
                f"[CRITICAL] {failed_count} features failed post-healing validation check. Purging unresolvable structures.", 
                level=logging.ERROR
            )
            gdf = gdf[~post_healing_invalid].copy()

        # -------------------------------------------------------------------------
        #: SCHEMA NORMALIZATION (ENFORCE TYPE HOMOGENEITY)
        # -------------------------------------------------------------------------
        # Converts singular primitives (e.g. Polygon) into single-element Multi-primitives.
        # Prevents runtime crashes when exporting to rigid vector formats (like shapefiles/Parquet schemas).
        if force_multi and not gdf.empty:
            gdf.geometry = gdf.geometry.apply(
                lambda g: shapely.multipoints([g]) if g.geom_type == 'Point' else (
                        shapely.multilinestrings([g]) if g.geom_type == 'LineString' else (
                        shapely.multipolygons([g]) if g.geom_type == 'Polygon' else g))
            )

        # -------------------------------------------------------------------------
        # GEOMETRY DEDUPLICATION (OPTIONAL)
        # -------------------------------------------------------------------------
        if deduplicate and not gdf.empty:
            pre_dedup = len(gdf)
            gdf = gdf.drop_duplicates(subset=['geometry']).copy()
            dedup_delta = pre_dedup - len(gdf)
            if dedup_delta > 0:
                log_execution(logger, f"Deduplication removed {dedup_delta} exact geometry overlaps.", level=logging.INFO)

        # -------------------------------------------------------------------------
        # TYPE ENFORCEMENT
        # -------------------------------------------------------------------------
        # Note: If allowed_types was left as None/Empty, this whole block is skipped
        # and all valid geometry types are safely permitted to remain in the dataset.
        if allowed_types and not gdf.empty:
            type_mask = gdf.geometry.geom_type.isin(allowed_types).values
            dropped_type = len(gdf) - type_mask.sum()
            gdf = gdf[type_mask].copy()
            if dropped_type > 0:
                log_execution(
                    logger, 
                    f"Filtered out {dropped_type} features not matching allowed types: {allowed_types}", 
                    level=logging.INFO
                )

        # -------------------------------------------------------------------------
        # RESET INDEX FOR SEAMLESS MERGES
        # -------------------------------------------------------------------------
        # Clears fragmented, non-contiguous indexing caused by row-dropping operations
        gdf = gdf.reset_index(drop=True)

        # -------------------------------------------------------------------------
        # PIPELINE TERMINATION ASSESSMENT
        # -------------------------------------------------------------------------
        final_count = len(gdf)
        if final_count == 0:
            # Elevated warning level flags that the sanitization stripped everything
            log_execution(
                logger, 
                f"[WARNING] Geometry sanitization reduced feature count from {initial_count} to 0 rows. Pipeline halted.", 
                level=logging.WARNING
            )
        else:
            log_execution(logger, f"Sanitization complete. Final feature count: {final_count}", level=logging.INFO)
            
        return gdf

    def _validate_geom_column(
        self,
        gdf: gpd.GeoDataFrame,
        geom_column: str,
        allowed_types: set,
        context: str,
    ) -> None:
        """Shared precondition checks used by every geometry-type mapping function."""
        if geom_column not in gdf.columns:
            raise ValueError(f"geom_column '{geom_column}' not found in the input GeoDataFrame.")

        # Safely cast the column to a GeoSeries. 
        # This is strictly required because if `geom_column` is not the *active* 
        # geometry column, pandas treats it as a standard Series of objects, 
        # which lacks the .is_empty and .geom_type spatial accessors.
        geo_col = gpd.GeoSeries(gdf[geom_column])

        null_mask = geo_col.isna() | geo_col.is_empty
        if null_mask.any():
            raise ValueError(
                f"{context}: found {int(null_mask.sum())} null/empty geometries in "
                f"'{geom_column}'. Sanitize input (e.g. via sanitize_geometries) first."
            )

        # Dropna ensures missing geometries (which evaluate to None for geom_type) 
        # don't trigger a false positive in the unsupported type check.
        geom_types = set(geo_col.geom_type.dropna().unique())
        if not geom_types <= allowed_types:
            raise ValueError(
                f"{context}: expected geometry types {allowed_types} in '{geom_column}', "
                f"found unsupported type(s): {geom_types - allowed_types}"
            )

    def _build_target_grid(
        self,
        target_grid_name: str,
        source_crs,
        source_bounds: Tuple[float, float, float, float],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[gpd.GeoDataFrame, str, float]:
        """
        Shared grid-blueprint builder used by every geometry-type mapping function.

        Returns
        -------
        target_grid_gdf : geopandas.GeoDataFrame
            Columns ['grid_idx', 'geometry'], one row per aligned template cell.
        target_crs : str
            The resolved CRS of the target grid.
        res : float
            The resolution (cell size) of the target grid, in target_crs units.
        """
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]

        if target_bbox is not None:
            log_execution(logger, f"Using explicitly provided target bounding box: {target_bbox}", level=logging.INFO)
            dst_bbox = target_bbox
        else:
            dst_bbox = transform_bounds(source_crs, target_crs, *source_bounds)

        template_da, _ = self.create_aligned_raster_template(dst_bbox, target_grid_name)
        res = template_da.attrs["res"]

        x_centers = template_da.x.values
        y_centers = template_da.y.values

        if x_centers.size == 0 or y_centers.size == 0:
            raise ValueError(
                "create_aligned_raster_template produced a 0-cell grid. This usually "
                "means the source bounding box has zero width/height in an axis "
                "(e.g. a single-point dataset landing exactly on a grid line). "
                f"dst_bbox={dst_bbox}, grid_name={target_grid_name}."
            )

        xx, yy = np.meshgrid(x_centers, y_centers)
        x_flat, y_flat = xx.ravel(), yy.ravel()
        half_res = res / 2.0

        # Vectorized cell construction (shapely >= 2.0) instead of a per-cell
        # Python-level list comprehension -- meaningfully faster once grids scale
        # into the hundreds of thousands / millions of cells.
        cell_polygons = shapely.box(
            x_flat - half_res, y_flat - half_res, x_flat + half_res, y_flat + half_res
        )

        target_grid_gdf = gpd.GeoDataFrame(
            {"grid_idx": np.arange(len(cell_polygons))},
            geometry=cell_polygons,
            crs=target_crs,
        )
        return target_grid_gdf, target_crs, res

    def _ensure_crs(
            self,
            gdf: gpd.GeoDataFrame,
            target_crs: Union[str, int, CRS],
            logger: Optional[logging.Logger] = None
        ) -> gpd.GeoDataFrame:
        """
        Validates the CRS of a GeoDataFrame against a target CRS. 
        Reprojects the data only if a mismatch is detected.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The input spatial dataframe to validate.
        target_crs : str, int, or pyproj.CRS
            The expected coordinate reference system (e.g., "EPSG:3035", 3035).
        logger : logging.Logger, optional
            Logger to record reprojection events or missing CRS warnings.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame guaranteed to be in the target CRS.
        
        Raises
        ------
        ValueError
            If the input GeoDataFrame has no defined CRS.
        """
        if gdf.crs is None:
            raise ValueError(
                "Input GeoDataFrame lacks a defined CRS. Cannot safely reproject "
                "or map to the target template."
            )

        # Standardize the target CRS to a pyproj CRS object for robust comparison
        target = CRS.from_user_input(target_crs)

        # GeoPandas handles CRS equality checks intelligently via pyproj
        if gdf.crs == target:
            return gdf
        
        if logger:
            logger.info(
                f"CRS mismatch detected. Reprojecting from {gdf.crs.name} "
                f"to {target.name}..."
            )
            
        return gdf.to_crs(target)

    def generate_spatial_point_clouds(
        self,
        gdf: gpd.GeoDataFrame,
        n_passes: int = 30,
        uncertainty_col: str = "coordinateuncertaintyinmeters",
        output_col: str = "point_cloud",
        distribution: str = "gaussian",
        random_state: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> gpd.GeoDataFrame:
        """
        Generate memory-efficient spatial point clouds around feature centroids.
    
        Computes randomized probability clouds using flat, vectorized NumPy arrays
        to handle massive datasets without exploding memory. Every record receives 
        the exact same number of sampling passes, ensuring strict conservation of 
        statistical weight (1 occurrence = 1.0 probability mass) regardless of 
        the spatial extent of its uncertainty.
    
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The input spatial dataset containing geometries and uncertainty values.
            Must not contain null or empty geometries.
        n_passes : int, default 30
            The exact number of jittered points to generate per feature. Keeping
            this uniform across all records guarantees identical statistical weight.
        uncertainty_col : str, default 'coordinateuncertaintyinmeters'
            The name of the column containing the uncertainty radius in meters.
        output_col : str, default 'point_cloud'
            The name of the new column where the generated MultiPoint clouds will
            be stored.
        distribution : {'gaussian', 'uniform'}, default 'gaussian'
            The probability distribution used to scatter the points.
            * 'gaussian': Concentrates points near the centroid. Uncertainty is
            treated as a 3-sigma radius (sigma = uncertainty / 3).
            * 'uniform': Scatters points evenly across the entire uncertainty disk.
        random_state : int, optional
            Seed for a local, independent random generator. Pass this for
            reproducible STAC outputs.
        logger : logging.Logger, optional
            Logger instance for execution tracking.
    
        Returns
        -------
        gpd.GeoDataFrame
            A copy of the input GeoDataFrame with the new `output_col` geometry
            column, along with `passes` and `weight_per_point` columns.
        """
        # =========================================================================
        # FAIL-FAST VALIDATION
        # =========================================================================
        if distribution not in ("gaussian", "uniform"):
            raise ValueError("distribution must be either 'uniform' or 'gaussian'")
    
        if n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    
        if uncertainty_col not in gdf.columns:
            raise ValueError(f"uncertainty_col '{uncertainty_col}' not found in input GeoDataFrame.")
    
        if gdf.crs is None:
            log_execution(
                logger,
                "Input GeoDataFrame has no CRS defined; assuming projected units in "
                "meters for jitter offsets. Set a CRS explicitly if this is not the case.",
                level=logging.WARNING,
            )
    
        if gdf.empty:
            result_gdf = gdf.copy()
            result_gdf[output_col] = None
            result_gdf["passes"] = 0
            result_gdf["weight_per_point"] = 0.0
            return result_gdf
    
        null_geom_mask = gdf.geometry.isna() | gdf.geometry.is_empty
        if null_geom_mask.any():
            raise ValueError(
                f"Found {int(null_geom_mask.sum())} null/empty geometries in gdf. "
                "Sanitize geometries before generating point clouds."
            )
    
        # Initialize isolated random state for perfect STAC reproducibility
        rng = np.random.default_rng(random_state)
    
        # Extract base coordinates safely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            centroids = gdf.geometry.centroid
            x_coords = centroids.x.values
            y_coords = centroids.y.values
    
        # Swap out NaN uncertainty values for 0 (no jitter).
        uncertainties = gdf[uncertainty_col].fillna(0).values.astype(float)
    
        negative_mask = uncertainties < 0
        if negative_mask.any():
            log_execution(
                logger,
                f"Clipping {int(negative_mask.sum())} negative uncertainty value(s) to 0m.",
                level=logging.WARNING,
            )
            uncertainties = np.clip(uncertainties, 0, None)
    
        n_features = len(gdf)
        n_expanded = n_features * n_passes
    
        log_execution(
            logger,
            f"Generating strictly {n_passes} point(s) per feature "
            f"({n_expanded:,} total points across {n_features:,} records).",
            level=logging.INFO,
        )
    
        # =========================================================================
        # GENERATE EXPANDED ARRAYS IN PURE NUMPY
        # =========================================================================
        expanded_uncertainties = np.repeat(uncertainties, n_passes)
        expanded_x = np.repeat(x_coords, n_passes)
        expanded_y = np.repeat(y_coords, n_passes)
    
        # =========================================================================
        # CHOOSE PROBABILITY DISTRIBUTION
        # =========================================================================
        if distribution == "uniform":
            angles = rng.uniform(0, 2 * np.pi, n_expanded)
            radii_modifiers = np.sqrt(rng.uniform(0, 1, n_expanded))
            actual_radii = expanded_uncertainties * radii_modifiers
    
            delta_x = actual_radii * np.cos(angles)
            delta_y = actual_radii * np.sin(angles)
    
        else:  # "gaussian"
            sigma = expanded_uncertainties / 3.0
            delta_x = rng.normal(loc=0.0, scale=sigma, size=n_expanded)
            delta_y = rng.normal(loc=0.0, scale=sigma, size=n_expanded)
    
        # =========================================================================
        # GEOGRAPHIC CRS CONVERSION (Meters -> Degrees)
        # =========================================================================
        is_geographic = gdf.crs is not None and gdf.crs.is_geographic
        if is_geographic:
            meters_per_deg_lat = 111320.0
            near_pole_mask = np.abs(expanded_y) > 89.9
            if near_pole_mask.any():
                log_execution(
                    logger,
                    f"Clamping latitude for longitude-scale calculation near the poles "
                    f"(>89.9°) for {int(near_pole_mask.sum())} point(s).",
                    level=logging.WARNING,
                )
            lat_for_scale = np.clip(expanded_y, -89.9, 89.9)
            meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_for_scale))
    
            delta_x = delta_x / meters_per_deg_lon
            delta_y = delta_y / meters_per_deg_lat
    
        new_x = expanded_x + delta_x
        new_y = expanded_y + delta_y
    
        if is_geographic:
            new_x = ((new_x + 180.0) % 360.0) - 180.0
    
        # =========================================================================
        # COMPRESS BACK TO MULTIPOINTS (Reshape Optimization)
        # =========================================================================
        # Because every row has strictly `n_passes` points, we no longer need to 
        # compute variable slice indices. We reshape instantly in C-contiguous memory.
        coords_2d = np.column_stack((new_x, new_y))
        grouped_coords = coords_2d.reshape(n_features, n_passes, 2)
        
        from shapely.geometry import MultiPoint
        multipoints = [MultiPoint(pts) for pts in grouped_coords]
    
        # =========================================================================
        # ATTACH TO DATAFRAME
        # =========================================================================
        result_gdf = gdf.copy()
        result_gdf[output_col] = multipoints
    
        # Track conservation of mass
        result_gdf["passes"] = n_passes
        result_gdf["weight_per_point"] = 1.0 / n_passes
    
        return result_gdf

    def _compute_home_cell_mapping(
        self,
        reference_geom: gpd.GeoSeries,
        uid_values,
        uid_col_name: str,
        tree: KDTree,
        grid_idx_values: np.ndarray,
        res: float,
        output_col_name: str = "centroid_grid_idx",
    ) -> pd.DataFrame:
        """
        Shared helper: nearest target-grid-cell centroid to each reference
        geometry's own centroid. 
        """
        epsilon = res * 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            centroids = reference_geom.centroid
            coords = np.column_stack([centroids.x.values + epsilon, centroids.y.values + epsilon])
    
        _, nearest_idx = tree.query(coords)
        return pd.DataFrame({
            uid_col_name: uid_values,
            output_col_name: grid_idx_values[nearest_idx],
        })
    
    def map_points_to_template(
        self,
        source_gdf: gpd.GeoDataFrame,
        target_grid_name: str,
        geom_column: str = "geometry",
        output_col: str = "grid_idx",
        method: str = "intersect",
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> gpd.GeoDataFrame:
        if method not in ("intersect", "kdtree"):
            raise ValueError("method must be 'kdtree' or 'intersect'.")
    
        # FETCH AND VALIDATE CRS EARLY
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)
    
        if source_gdf.empty:
            log_execution(logger, "map_points_to_template: source_gdf is empty; returning an empty copy.", level=logging.WARNING)
            result = source_gdf.copy()
            result[output_col] = pd.array([], dtype="Int64")
            return result
    
        self._validate_geom_column(source_gdf, geom_column, {"Point"}, "map_points_to_template")
    
        if output_col in source_gdf.columns:
            log_execution(logger, f"output_col '{output_col}' already exists in source_gdf and will be overwritten.", level=logging.WARNING)
    
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        # OVERWRITE LOCAL INDICES WITH DETERMINISTIC GLOBAL INDICES
        grid_centroids = target_grid_gdf.geometry.centroid
        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )
    
        _uid_col = "_map_pts_src_uid_tmp"
        work_df = source_gdf[[geom_column]].copy()
        work_df[_uid_col] = source_gdf.index
        # No need for .to_crs() here; it is already guaranteed by _ensure_crs
        work_df = work_df.set_geometry(geom_column, crs=source_gdf.crs)
    
        if method == "kdtree":
            log_execution(logger, "Mapping simple points via KDTree...", level=logging.INFO)
            tree = KDTree(np.column_stack([grid_centroids.x.values, grid_centroids.y.values]))
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                geom_coords = np.column_stack([work_df.geometry.x, work_df.geometry.y])
    
            _, nearest_idx = tree.query(geom_coords)
            mapping = pd.DataFrame({
                _uid_col: work_df[_uid_col].values,
                output_col: target_grid_gdf["grid_idx"].values[nearest_idx],
            })
    
        else:
            log_execution(logger, "Mapping simple points via spatial intersection...", level=logging.INFO)
            joined = gpd.sjoin(work_df, target_grid_gdf, how="inner", predicate="intersects")
    
            best_match = (
                joined.sort_values(
                    by=[_uid_col, "grid_idx"],
                    ascending=[True, True],
                    kind="mergesort",
                ).drop_duplicates(subset=[_uid_col])
            )
            mapping = best_match[[_uid_col, "grid_idx"]].rename(columns={"grid_idx": output_col})
    
            unmatched_count = work_df[_uid_col].nunique() - mapping[_uid_col].nunique()
            if unmatched_count > 0:
                log_execution(logger, f"{unmatched_count} record(s) fell outside the grid extent.", level=logging.WARNING)
    
        result_gdf = source_gdf.copy()
        result_gdf[_uid_col] = result_gdf.index
        result_gdf = result_gdf.merge(mapping, on=_uid_col, how="left")
        result_gdf.index = source_gdf.index
        result_gdf = result_gdf.drop(columns=[_uid_col])
    
        if result_gdf[output_col].isna().any():
            result_gdf[output_col] = result_gdf[output_col].astype("Int64")
    
        return result_gdf
    
    def map_point_cloud_to_template(
        self,
        source_gdf: gpd.GeoDataFrame,
        target_grid_name: str,
        geom_column: str = "point_cloud",
        output_col: str = "grid_idx",
        mode: str = "fractional",
        classify_method: str = "intersect",
        fraction_col: str = "fraction",
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        
        if mode not in ("fractional", "classify"):
            raise ValueError("mode must be 'fractional' or 'classify'.")
        if mode == "classify" and classify_method not in ("intersect", "kdtree"):
            raise ValueError("classify_method must be 'kdtree' or 'intersect'.")

        # 1. CAPTURE ORIGINAL CRS BEFORE REPROJECTION
        orig_crs = source_gdf.crs

        # 2. FETCH TARGET CRS AND ALIGN ACTIVE GEOMETRY
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)

        # Validate that the point cloud column exists and contains MultiPoints
        self._validate_geom_column(source_gdf, geom_column, {"MultiPoint", "Point"}, "map_point_cloud_to_template")

        # Validate that the active geometry column contains the original Points
        original_geom_name = source_gdf.geometry.name
        self._validate_geom_column(
            source_gdf, original_geom_name, {"Point"}, "map_point_cloud_to_template (original geometry)"
        )

        from geopandas.array import GeometryDtype as _GeometryDtype
        
        # Identify formal geometry columns
        geometry_cols = [c for c in source_gdf.columns if isinstance(source_gdf[c].dtype, _GeometryDtype)]
        
        # Explicitly exclude the generated point cloud column from the preserved attributes.
        # Since it is stored as an object list of shapely MultiPoints rather than a true GeometryDtype,
        # it previously sneaked past the filter and crashed the PyArrow Parquet exporter.
        preserve_cols = [
            c for c in source_gdf.columns 
            if c not in geometry_cols and c != geom_column
        ]

        # Handle empty incoming dataframes
        if source_gdf.empty:
            log_execution(logger, "map_point_cloud_to_template: source_gdf is empty; returning an empty result.", level=logging.WARNING)
            if mode == "classify":
                result = source_gdf.copy()
                result[output_col] = pd.array([], dtype="Int64")
                result["src_grid_idx"] = pd.array([], dtype="Int64")
                
                # Drop the heavy secondary point_cloud geometry column before returning
                if geom_column in result.columns and geom_column != result.geometry.name:
                    result = result.drop(columns=[geom_column])
                    
                return result
            return pd.DataFrame(columns=[output_col, "src_grid_idx", fraction_col] + preserve_cols)

        if output_col in source_gdf.columns:
            log_execution(logger, f"output_col '{output_col}' already exists in source_gdf and will be overwritten.", level=logging.WARNING)

        # Build the strictly aligned template mesh
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        # Overwrite local indices with deterministic global indices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            grid_centroids = target_grid_gdf.geometry.centroid

        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )

        # 3. CONSTRUCT WORK_DF AND REPROJECT SECONDARY POINT CLOUD COLUMN IF CRS CHANGED
        _uid_col = "_map_cloud_src_uid_tmp"
        work_df = source_gdf[[geom_column]].copy()
        work_df[_uid_col] = source_gdf.index

        if orig_crs != target_crs:
            work_df[geom_column] = gpd.GeoSeries(work_df[geom_column], crs=orig_crs).to_crs(target_crs)

        work_df = work_df.set_geometry(geom_column, crs=target_crs)

        # 4. TRACK ORIGINAL SOURCE CENTROIDS (Calculates src_grid_idx)
        log_execution(logger, "Mapping original source points to determine true home grid cells...", level=logging.INFO)
        orig_pts = source_gdf[[original_geom_name]].copy()
        orig_pts = orig_pts.set_geometry(original_geom_name, crs=target_crs)

        tree = KDTree(np.column_stack([grid_centroids.x.values, grid_centroids.y.values]))
        grid_idx_values = target_grid_gdf["grid_idx"].values

        # Map the original active geometry (the source point) to the grid
        centroid_mapping = self._compute_home_cell_mapping(
            reference_geom=orig_pts.geometry,
            uid_values=source_gdf.index,
            uid_col_name=_uid_col,
            tree=tree,
            grid_idx_values=grid_idx_values,
            res=res,
            output_col_name="src_grid_idx"
        )

        # =====================================================================
        # CLASSIFICATION STRATEGY (KD-Tree)
        # =====================================================================
        if mode == "classify" and classify_method == "kdtree":
            log_execution(logger, "Classifying point clouds via mathematical cloud centroid...", level=logging.INFO)
            mapping = self._compute_home_cell_mapping(
                reference_geom=work_df.geometry,
                uid_values=work_df[_uid_col].values,
                uid_col_name=_uid_col,
                tree=tree,
                grid_idx_values=grid_idx_values,
                res=res,
                output_col_name=output_col,
            )
            mapping = mapping.merge(centroid_mapping, on=_uid_col, how="left")

            result_gdf = source_gdf.copy()
            result_gdf[_uid_col] = result_gdf.index
            result_gdf = result_gdf.merge(mapping, on=_uid_col, how="left").drop(columns=[_uid_col])
            result_gdf.index = source_gdf.index
            
            # Enforce nullable integer typing
            for col in (output_col, "src_grid_idx"):
                if result_gdf[col].isna().any():
                    result_gdf[col] = result_gdf[col].astype("Int64")
                    
            # Drop the heavy secondary point_cloud geometry column before returning
            if geom_column in result_gdf.columns and geom_column != result_gdf.geometry.name:
                result_gdf = result_gdf.drop(columns=[geom_column])
                
            return result_gdf

        # =====================================================================
        # INTERSECTION STRATEGY (Batched for Memory Safety)
        # =====================================================================
        # Define a safe batch limit. 10,000 MultiPoints * 100 passes = 1 million geometries per batch
        chunk_size = 10000 
        num_chunks = math.ceil(len(work_df) / chunk_size)
        
        counts_list = []
        true_totals_list = []

        log_execution(logger, f"Exploding and intersecting clouds in {num_chunks} memory-safe batches...", level=logging.INFO)

        for i in range(num_chunks):
            # Extract the current batch
            chunk = work_df.iloc[i * chunk_size : (i + 1) * chunk_size]
            
            # Explode only this specific chunk
            exploded_chunk = chunk.explode(index_parts=False).reset_index(drop=True)
            
            # Aggregate the true total passes for this chunk
            true_totals_chunk = exploded_chunk.groupby(_uid_col).size().rename("true_total_passes").reset_index()
            true_totals_list.append(true_totals_chunk)
            
            # Perform the spatial intersection against the rigid mesh
            joined_chunk = gpd.sjoin(exploded_chunk, target_grid_gdf, how="inner", predicate="intersects")
            counts_chunk = joined_chunk.groupby([_uid_col, "grid_idx"]).size().reset_index(name="pt_count")
            counts_list.append(counts_chunk)
            
            # Explicitly force garbage collection to keep RAM perfectly flat across batches
            del chunk, exploded_chunk, joined_chunk
            gc.collect()

        # Combine the lightweight aggregated results
        counts = pd.concat(counts_list, ignore_index=True)
        true_totals = pd.concat(true_totals_list, ignore_index=True)

        if mode == "classify":
            log_execution(logger, "Assigning clouds to grid cell with maximum point density...", level=logging.INFO)
            # Find the grid cell that captured the most points for each source feature
            best_match = (
                counts.sort_values(
                    by=[_uid_col, "pt_count", "grid_idx"],
                    ascending=[True, False, True],
                    kind="mergesort",
                ).drop_duplicates(subset=[_uid_col])
            )
            mapping = best_match[[_uid_col, "grid_idx"]].rename(columns={"grid_idx": output_col})
            mapping = mapping.merge(centroid_mapping, on=_uid_col, how="left")

            unmatched_count = work_df[_uid_col].nunique() - mapping[_uid_col].dropna().nunique()
            if unmatched_count > 0:
                log_execution(
                    logger,
                    f"{unmatched_count} point cloud(s) had no intersecting grid cell and "
                    f"received a null '{output_col}' assignment.",
                    level=logging.WARNING,
                )

            result_gdf = source_gdf.copy()
            result_gdf[_uid_col] = result_gdf.index
            result_gdf = result_gdf.merge(mapping, on=_uid_col, how="left").drop(columns=[_uid_col])
            result_gdf.index = source_gdf.index
            
            for col in (output_col, "src_grid_idx"):
                if result_gdf[col].isna().any():
                    result_gdf[col] = result_gdf[col].astype("Int64")
                    
            # Drop the heavy secondary point_cloud geometry column before returning
            if geom_column in result_gdf.columns and geom_column != result_gdf.geometry.name:
                result_gdf = result_gdf.drop(columns=[geom_column])
                
            return result_gdf

        # =====================================================================
        # FRACTIONAL STRATEGY
        # =====================================================================
        log_execution(logger, "Calculating fractional probability weights per grid cell...", level=logging.INFO)
        counts = counts.merge(true_totals, on=_uid_col, how="left")
        counts[fraction_col] = counts["pt_count"] / counts["true_total_passes"]

        per_record_total_fraction = counts.groupby(_uid_col)[fraction_col].sum()
        partial_coverage_count = int((per_record_total_fraction < 0.99).sum())
        if partial_coverage_count > 0:
            log_execution(
                logger,
                f"{partial_coverage_count} point cloud(s) have some jittered points falling "
                "outside the grid extent; their returned fractions sum to < 1.0.",
                level=logging.WARNING,
            )

        fully_unmatched = len(source_gdf) - counts[_uid_col].nunique()
        if fully_unmatched > 0:
            log_execution(logger, f"{fully_unmatched} point cloud(s) fell entirely outside the grid.", level=logging.WARNING)

        # Preserve src_uid in the returned table for QA/QC and aggregation linkage
        counts = counts.rename(columns={"grid_idx": output_col, _uid_col: "src_uid"})
        result_columns = [output_col, "src_grid_idx", fraction_col, "src_uid"] + preserve_cols

        centroid_mapping_renamed = centroid_mapping.rename(columns={_uid_col: "src_uid"})
        
        source_meta = source_gdf[preserve_cols].reset_index()
        idx_col_name = source_gdf.index.name or "index"
        source_meta = source_meta.rename(columns={idx_col_name: "src_uid"})

        result_df = counts[[output_col, "src_uid", fraction_col]].merge(
            centroid_mapping_renamed, on="src_uid", how="left"
        ).merge(
            source_meta,
            on="src_uid",
            how="left",
        )
        result_df = result_df.reset_index(drop=True)

        return result_df[result_columns]
        
    def map_polygon_to_template(
        self,
        source_gdf: gpd.GeoDataFrame,
        target_grid_name: str,
        geom_column: str = "geometry",
        output_col: str = "grid_idx",
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        min_areal_fraction: float = 1e-6,
        include_centroid_tracking: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        
        # =====================================================================
        # STEP 1: CRS VALIDATION AND GEOMETRY HYGIENE
        # =====================================================================
        # Ensure the source data is projected to exactly match the target grid's CRS.
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)
    
        # Verify that the active geometry column actually contains polygons.
        self._validate_geom_column(source_gdf, geom_column, {"Polygon", "MultiPolygon"}, "map_polygon_to_template")
    
        # A rigid topological check: overlay math will catastrophically fail if polygons cross over themselves.
        invalid_mask = ~source_gdf[geom_column].is_valid
        if invalid_mask.any():
            raise ValueError(
                f"map_polygon_to_template: found {int(invalid_mask.sum())} topologically "
                f"invalid geometries in '{geom_column}'. Run sanitize_geometries first."
            )
    
        from pyproj import CRS as _CRS
        from geopandas.array import GeometryDtype as _GeometryDtype
        
        # Determine if the target grid is measured in degrees (geographic) or meters (projected).
        # We use EPSG:6933 (Global Equal Area) as a fallback to calculate accurate metric areas for geographic grids.
        is_geographic_grid = _CRS.from_user_input(target_crs).is_geographic
        EQUAL_AREA_CRS = "EPSG:6933" 

        # Separate spatial columns from attribute columns so we can pass attributes through to the final table.
        geometry_cols = [c for c in source_gdf.columns if isinstance(source_gdf[c].dtype, _GeometryDtype)]
        preserve_cols = [c for c in source_gdf.columns if c not in geometry_cols]
        
        # Define the exact schema of the final output table, explicitly including 'src_uid' for downstream QA/QC.
        result_columns = [output_col, "src_uid"] + (["centroid_grid_idx"] if include_centroid_tracking else []) + ["areal_fraction"] + preserve_cols
    
        # Handle edge case: empty dataframes bypass processing safely.
        if source_gdf.empty:
            log_execution(logger, "map_polygon_to_template: source_gdf is empty; returning an empty result.", level=logging.WARNING)
            return pd.DataFrame(columns=result_columns)
    
        if output_col in source_gdf.columns:
            log_execution(logger, f"output_col '{output_col}' already exists in source_gdf and will be overwritten in the result.", level=logging.WARNING)
    
        # =====================================================================
        # STEP 2: BUILD THE PRISTINE TARGET GRID MESH
        # =====================================================================
        # Generate the strict mathematical blueprint for the target area.
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        # Assign deterministic global IDs to every cell based on its distance from the master grid origin.
        grid_centroids = target_grid_gdf.geometry.centroid
        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )
    
        # =====================================================================
        # STEP 3: CALCULATE PRE-INTERSECT BASELINE AREAS
        # =====================================================================
        log_execution(logger, "Preparing polygons and calculating baseline continuous metric areas...", level=logging.INFO)
        _uid_col = "_map_poly_src_uid_tmp"
        _area_col = "_map_poly_source_area_tmp"
    
        source_df = source_gdf.copy()
        source_df[_uid_col] = source_df.index
        source_df = source_df.set_geometry(geom_column, crs=source_gdf.crs)
        
        # Calculate the mathematical area of the original whole polygon before it gets shattered.
        if is_geographic_grid:
            source_df[_area_col] = source_df.to_crs(EQUAL_AREA_CRS).geometry.area
        else:
            source_df[_area_col] = source_df.geometry.area
    
        # Purge microscopic degenerate geometries that evaluate to 0 area to prevent divide-by-zero errors later.
        zero_area_mask = source_df[_area_col] <= 0
        if zero_area_mask.any():
            log_execution(
                logger,
                f"Dropping {int(zero_area_mask.sum())} source record(s) with zero or negative area (degenerate geometry).",
                level=logging.WARNING,
            )
            source_df = source_df[~zero_area_mask]
    
        # =====================================================================
        # STEP 4: TRACK ORIGINAL CENTROIDS (Optional)
        # =====================================================================
        # Determines which grid cell physically contains the mathematical center of the polygon.
        centroid_mapping = None
        if include_centroid_tracking:
            log_execution(logger, "Mapping polygon centroids to determine home grid cells...", level=logging.INFO)
            tree = KDTree(np.column_stack([grid_centroids.x.values, grid_centroids.y.values]))
            centroid_mapping = self._compute_home_cell_mapping(
                reference_geom=source_df.geometry,
                uid_values=source_df[_uid_col].values,
                uid_col_name=_uid_col,
                tree=tree,
                grid_idx_values=target_grid_gdf["grid_idx"].values,
                res=res,
            )
    
        # =====================================================================
        # STEP 5: MEMORY-SAFE BATCHED INTERSECTION (SHATTERING)
        # =====================================================================
        # Intersecting hundreds of thousands of polygons against a dense grid mesh requires massive RAM.
        # We slice the input dataset into chunks to keep memory usage flat.
        import math
        import gc
        chunk_size = 10000
        num_chunks = math.ceil(len(source_df) / chunk_size)
        intersections_list = []
        
        log_execution(logger, f"Executing polygon network fragmentation against template mesh in {num_chunks} memory-safe batches...", level=logging.INFO)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*keep_geom_type.*")
            
            for i in range(num_chunks):
                chunk = source_df.iloc[i * chunk_size : (i + 1) * chunk_size]
                
                # Perform strict topological overlay. Polygons are physically cut by grid cell boundaries.
                intersection_chunk = gpd.overlay(
                    chunk[[_uid_col, _area_col, geom_column]],
                    target_grid_gdf,
                    how="intersection",
                )
                
                if not intersection_chunk.empty:
                    intersections_list.append(intersection_chunk)
                    
                # Explicitly force Python's garbage collector to release the heavy geometric data from RAM.
                del chunk, intersection_chunk
                gc.collect()
    
        if not intersections_list:
            log_execution(logger, "No intersections found between source polygons and target template.", level=logging.WARNING)
            return pd.DataFrame(columns=result_columns)
            
        # Combine all the shattered fragments from the batches back together.
        intersections = pd.concat(intersections_list, ignore_index=True)
    
        # =====================================================================
        # STEP 6: CALCULATE FRACTIONAL YIELDS
        # =====================================================================
        # Calculate the area of each resulting fragment.
        if is_geographic_grid:
            intersections["intersect_area"] = intersections.to_crs(EQUAL_AREA_CRS).geometry.area
        else:
            intersections["intersect_area"] = intersections.geometry.area

        # The fragment's mathematical weight is its new shattered area divided by its original whole area.
        intersections["areal_fraction"] = intersections["intersect_area"] / intersections[_area_col]
        
        # Drop microscopic slivers created by topological drift to keep the dataset lightweight.
        intersections = intersections[intersections["areal_fraction"] > min_areal_fraction]
    
        # =====================================================================
        # STEP 7: LOGGING DIAGNOSTICS & SCHEMA FORMATTING
        # =====================================================================
        per_record_total_fraction = intersections.groupby(_uid_col)["areal_fraction"].sum()
        partial_coverage_count = int((per_record_total_fraction < 0.99).sum())
        if partial_coverage_count > 0:
            log_execution(
                logger,
                f"{partial_coverage_count} polygon(s) have some area falling outside the "
                "grid extent; their returned areal_fraction values sum to < 1.0 for that record.",
                level=logging.WARNING,
            )
    
        unmatched_count = source_df[_uid_col].nunique() - intersections[_uid_col].nunique()
        if unmatched_count > 0:
            log_execution(
                logger,
                f"{unmatched_count} source record(s) had no intersection with the target grid.",
                level=logging.WARNING,
            )
    
        intersections = intersections.rename(columns={"grid_idx": output_col})
    
        # Build the final table by merging the fragments with their original attribute data.
        result = intersections[[_uid_col, output_col, "areal_fraction"]]
        
        if include_centroid_tracking:
            result = result.merge(centroid_mapping, on=_uid_col, how="left")
            
        result = result.merge(
            source_gdf[preserve_cols].reset_index().rename(columns={source_gdf.index.name or "index": _uid_col}),
            on=_uid_col,
            how="left",
        )
        
        # Rename the tracking ID to exactly 'src_uid' so the QA/QC engine and parqeut exporters recognize it
        result = result.rename(columns={_uid_col: "src_uid"})
        
        # Drop the dataframe index to ensure a clean, zero-indexed output table
        result = result.reset_index(drop=True)
    
        return result[result_columns]
    
    def transform_cellCollection_to_template(
        self, 
        source_gdf: gpd.GeoDataFrame, 
        target_grid_name: str, 
        value_column: str, 
        data_type: str = 'discrete', 
        method: str = 'kdtree',
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None) -> gpd.GeoDataFrame:
        """
        Transforms data from a source GeoDataFrame to match a strictly aligned master grid.

        Parameters:
        -----------
        source_gdf : geopandas.GeoDataFrame
            The input data to transform. Must have a defined CRS.
        target_grid_name : str
            The key of the template grid defined in `self.GRID_REGISTRY`.
        value_column : str
            The name of the numeric column to aggregate.
        data_type : str
            'discrete' (sum whole integers/majority rule) or 'continuous' (areal weighting).
        method : str
            'kdtree' (representative point snapping) or 'intersect' (geometric clipping).
        target_bbox : tuple of float, optional
            A specific bounding box (minx, miny, maxx, maxy) in the target CRS to force 
            the grid generation. If None, it dynamically calculates from the source data.
        """
        if source_gdf.crs is None:
            raise ValueError("Source GeoDataFrame is missing a CRS. Cannot project.")

        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]

        # 1. Determine the Bounding Box
        if target_bbox is not None:
            if logger: logger.info(f"Using explicitly provided target bounding box: {target_bbox}")
            dst_bbox = target_bbox
        else:
            if logger: logger.info("No target_bbox provided. Dynamically calculating from source data extent...")
            src_bounds = source_gdf.total_bounds 
            dst_bbox = transform_bounds(source_gdf.crs, target_crs, *src_bounds)

        # 2. Fetch the pristine blueprint
        template_da, _ = self.create_aligned_raster_template(dst_bbox, target_grid_name)
        res = template_da.attrs["res"]
        
        x_centers = template_da.x.values
        y_centers = template_da.y.values

        # 3. Build the target grid mesh (Pristine Squares)
        if logger: logger.info(f"Building {res}m target mesh in {target_crs}...")
        
        xx, yy = np.meshgrid(x_centers, y_centers)
        x_flat, y_flat = xx.flatten(), yy.flatten()
        half_res = res / 2.0
        
        polygons = [box(x - half_res, y - half_res, x + half_res, y + half_res) for x, y in zip(x_flat, y_flat)]
        
        target_grid_gdf = gpd.GeoDataFrame(
            {'grid_id': np.arange(len(polygons))}, 
            geometry=polygons, 
            crs=target_crs
        )

        # 4. Prepare Source Data Attributes
        source_df = source_gdf.copy()
        source_df['src_uid'] = source_df.index 
        
        ignore_cols = ['geometry', value_column, 'src_uid', 'source_area']
        preserve_cols = [col for col in source_df.columns if col not in ignore_cols]
        group_cols = ['grid_id'] + preserve_cols

        # ==========================================
        # STRATEGY A: KDTREE (Representative Point Snapping)
        # ==========================================
        if method == 'kdtree':
            if data_type != 'discrete':
                raise ValueError("KDTree routing mathematically requires 'discrete' data_type.")
            if logger: logger.info("Executing KDTree snapping via representative points...")
                
            source_points = source_df.copy()
            source_points["geometry"] = source_points.geometry.representative_point()
            source_points = source_points.to_crs(target_crs)
            
            src_coords = np.column_stack([source_points.geometry.x, source_points.geometry.y])
            tgt_coords = np.column_stack([x_flat, y_flat]) 
            
            tree = KDTree(tgt_coords)
            _, matched_grid_ids = tree.query(src_coords)
            source_df['grid_id'] = matched_grid_ids
            
            aggregated = source_df.groupby(group_cols)[value_column].sum().reset_index()

        # ==========================================
        # STRATEGY B: INTERSECT (Geometric Overlay & Areal Weighting)
        # ==========================================
        elif method == 'intersect':
            import warnings
            if logger: logger.info("Executing precise geometric overlay (Warning: RAM intensive)...")
            
            source_reprojected = source_df.to_crs(target_crs)
            source_reprojected['source_area'] = source_reprojected.geometry.area
            
            # Temporarily mute the benign keep_geom_type overlay warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                intersections = gpd.overlay(source_reprojected, target_grid_gdf, how='intersection')
            
            if intersections.empty:
                return target_grid_gdf.iloc[0:0].copy()
                
            intersections['intersect_area'] = intersections.geometry.area

            if data_type == 'continuous':
                intersections['weight'] = intersections['intersect_area'] / intersections['source_area']
                intersections['weighted_val'] = intersections[value_column] * intersections['weight']
                aggregated = intersections.groupby(group_cols)['weighted_val'].sum().reset_index()
                aggregated.rename(columns={'weighted_val': value_column}, inplace=True)

            elif data_type == 'discrete':
                idx_max = intersections.sort_values('intersect_area', ascending=False).groupby('src_uid').head(1)
                aggregated = idx_max.groupby(group_cols)[value_column].sum().reset_index()
            else:
                raise ValueError("Invalid data_type for intersect. Choose 'continuous' or 'discrete'.")
                
        else:
            raise ValueError("Invalid method. Choose 'kdtree' or 'intersect'.")

        # 5. Re-attach the pristine target grid geometries
        if logger: logger.info("Merging aggregated attributes back to pristine template grid...")
        target_geometries = target_grid_gdf[['grid_id', 'geometry']]
        result_grid = target_geometries.merge(aggregated, on='grid_id', how='inner')
        
        return result_grid
    
    def validate_vector_transformation(
        self,
        orig_gdf: gpd.GeoDataFrame, 
        targ_gdf: Union[gpd.GeoDataFrame, pd.DataFrame], 
        recipe: dict, 
        dataset_name: str,
        logger: Optional[logging.Logger] = None
    ) -> dict:
        """
        Validates the mathematical and topological integrity of a spatial transformation.

        This function dynamically extracts constraints (resolution, bounding box, 
        uncertainty) from the pipeline's YAML recipe and checks the transformed data 
        against the original raw geometries to detect mathematical leaks, hallucinations, 
        or topological drift.

        Parameters
        ----------
        orig_gdf : geopandas.GeoDataFrame
            The original, un-transformed geometries loaded from the source file. Must 
            contain the source geometries and the coordinate uncertainty column.
        targ_gdf : geopandas.GeoDataFrame or pandas.DataFrame
            The transformed dataset outputted by the pipeline (e.g., intersected polygons, 
            shattered point clouds, or KDTree-snapped points).
        recipe : dict
            The parsed YAML configuration dictionary used to dynamically extract the 
            target resolution, bounding box bounds, default uncertainty, topology, 
            and mapping mode.
        dataset_name : str, default "gbif"
            The primary key inside the `sources` block of the recipe indicating which 
            configuration to parse.
        logger : logging.Logger, optional
            The active logger instance. Used to emit warnings with specific failure counts.

        Returns
        -------
        dict
            A dictionary containing lists of original indices (`src_uid` or DataFrame 
            index) for records that failed validation.
            - 'interior_mass_failures' : list
            - 'boundary_mass_failures' : list
            - 'drift_failures' : list

        Notes
        -----
        To prevent false positives caused by bounding box clipping, the function first 
        implements Dynamic Spatial Zoning. It classifies every original feature as 
        either 'Interior' (wholly contained) or 'Boundary' (uncertainty radius bleeds 
        over the strict bounding box edge).

        The function then applies validation criteria based on the selected mapping mode:

        1. Conservation of Mass (Fractional Topologies)
        Triggered when `mapping_mode = "fractional"`.
        When features are shattered across a grid, the sum of their resulting 
        fragments must equal their geometrically expected mass.
        
        - Expected Mass Calculation: The algorithm calculates the exact expected yield 
            by intersecting the feature's geometry (or its uncertainty buffer, if it is a 
            point) with the strict bounding box before grid shattering occurs.
        
        - Interior Rule (Tolerance = 0.01): The output mass must equal the expected 
            mass (1.0) with an absolute tolerance of 1%. 
            *Why 0.01?* When GeoPandas calculates fractional intersections across hundreds 
            of grid cells, microscopic floating-point rounding errors and invalid ring 
            sliver deletions occur. 1% accounts for strict floating-point drift while 
            catching genuine algorithmic leaks.
        
        - Boundary Rule (Tolerance = 0.02): The output mass must equal the expected 
            clipped mass with an absolute tolerance of 2%.
            *Why 0.02?* Point cloud mapping is probabilistic (e.g., 100 Gaussian scatter 
            passes). The random scattering of points over a border boundary will closely 
            approximate, but not perfectly mirror, the mathematical area of a perfect 
            circle buffer. A 2% tolerance accommodates this statistical variance.

        2. Maximum Topological Drift (Classification Topologies)
        Triggered when `mapping_mode = "classification"`.
        When a continuous point is forced into a discrete grid cell, its coordinates 
        shift. This ensures the spatial join algorithm did not assign a point to a 
        vastly incorrect cell.
        
        - Rule: The Euclidean distance between the original coordinate and the assigned 
            grid cell centroid must not exceed: (Coordinate Uncertainty) + (Grid Cell Diagonal).
        
        - Justification: The maximum physical extent of a point is its uncertainty 
            radius. The maximum mathematical shift within a correct square grid cell is 
            the cell's diagonal (sqrt(2 * s^2)). Exceeding the sum of these extremes 
            indicates a fundamentally invalid spatial mapping.
        """
        log_execution(logger, "=== Initiating Dynamic Vector QA/QC Profiling ===", level=logging.INFO)
            
        # =====================================================================
        # 1. RECIPE EXTRACTION
        # =====================================================================
        # Drill down into the specific data source configurations within the recipe
        spatial_cfg = recipe.get("spatial", {})
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        vector_cfg = source_cfg.get("vector_processing", {})
        query_filters = source_cfg.get("query_filters", {})
        
        # Extract the requested spatial processing parameters
        topology = vector_cfg.get("topology", "point")
        mapping_mode = vector_cfg.get("mapping_mode", "classification")
        default_uncert = float(query_filters.get("default_Uncertainty", 1000.0))
        
        # Extract the strict target bounding box bounds
        bbox_cfg = spatial_cfg.get("bbox", {})
        target_bbox = (
            bbox_cfg.get("long_min"), 
            bbox_cfg.get("lat_min"), 
            bbox_cfg.get("long_max"), 
            bbox_cfg.get("lat_max")
        )
        
        # Parse the user-defined resolution string to calculate physical cell dimensions in meters
        res_str = str(spatial_cfg.get("target_resolution", "1km")).lower()
        if res_str.endswith("km"):
            res_meters = float(res_str.replace("km", "")) * 1000
        elif res_str.endswith("m"):
            res_meters = float(res_str.replace("m", ""))
        else:
            res_meters = 1000.0 # Fallback default
            
        # Calculate the maximum possible distance a point can shift within a single square cell
        cell_diagonal = math.sqrt(2 * (res_meters ** 2))

        # =====================================================================
        # 2. SPATIAL SETUP
        # =====================================================================
        # Ensure all geometries are in a projected (metric) CRS. This is strictly required 
        # for accurate area measurements and Euclidean distance calculations.
        if orig_gdf.crs.to_epsg() == 4326:
            proj_crs = orig_gdf.estimate_utm_crs()
            orig_proj = orig_gdf.to_crs(proj_crs)
            
            # Safely reproject target data ONLY if it is a GeoDataFrame
            if isinstance(targ_gdf, gpd.GeoDataFrame) and not targ_gdf.empty:
                targ_proj = targ_gdf.to_crs(proj_crs)
            else:
                targ_proj = targ_gdf.copy()
        else:
            orig_proj = orig_gdf.copy()
            targ_proj = targ_gdf.copy()
            proj_crs = orig_proj.crs

        # Construct a geometric representation of the rigid bounding box to calculate intersection boundaries
        bbox_poly = box(*target_bbox)
        bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs="EPSG:4326").to_crs(proj_crs)
        strict_bounds = bbox_gdf.geometry.iloc[0]

        # Locate the uncertainty column and fill NaN values with the default specified in the recipe
        uncert_col = next((c for c in orig_proj.columns if 'uncertainty' in c.lower()), 'coordinateuncertaintyinmeters')
        uncertainties = orig_proj[uncert_col].fillna(default_uncert).astype(float)
        
        # Perform Spatial Zoning: Determine if a feature's physical area bleeds over the target bounding box
        dist_to_boundary = orig_proj.geometry.distance(strict_bounds.exterior)
        is_inside = orig_proj.geometry.within(strict_bounds)
        
        # A boundary feature is either outside the box but its uncertainty reaches inside, 
        # or inside the box but its uncertainty reaches outside.
        is_boundary = dist_to_boundary <= uncertainties
        
        # An interior feature is safely contained with no edge bleeding.
        is_interior = is_inside & (dist_to_boundary > uncertainties)

        orig_proj['qa_zone'] = np.where(is_interior, 'Interior', 
                                      np.where(is_boundary, 'Boundary', 'Exterior'))
        
        # Initialize failure tracking lists
        interior_failures_idx = []
        boundary_failures_idx = []
        drift_failures_idx = []

        # =====================================================================
        # 3. CHECK A: CONSERVATION OF MASS (Fractional Mapping)
        # =====================================================================
        # Triggered when geometries (polygons or point clouds) are shattered across multiple cells
        if mapping_mode == "fractional":
            weight_col = "areal_fraction" if "areal_fraction" in targ_proj.columns else "fraction"
            
            # Identify the foreign key linking shattered target cells back to the original source row
            link_col = 'src_uid' if 'src_uid' in targ_proj.columns else targ_proj.index.name
            if link_col is None or link_col not in targ_proj.columns:
                targ_proj['src_uid'] = targ_proj.index
                link_col = 'src_uid'

            # Sum the resulting fractions to determine the pipeline's total output mass per original geometry
            mass_yield = targ_proj.groupby(link_col)[weight_col].sum().rename("output_mass")
            
            # Calculate the Pre-Intersect Expected Mass
            # If the input geometry is a raw point (0 area), we must buffer it by its uncertainty 
            # radius to accurately simulate what percentage of its "cloud" falls inside the grid.
            proxy_geoms = orig_proj.geometry.copy()
            zero_area_mask = proxy_geoms.area == 0
            if zero_area_mask.any():
                proxy_geoms.loc[zero_area_mask] = proxy_geoms[zero_area_mask].buffer(uncertainties[zero_area_mask])
                
            # Divide the area inside the box by the total area to get the mathematically perfect expected mass
            orig_proj['expected_mass'] = proxy_geoms.intersection(strict_bounds).area / proxy_geoms.area
            
            # Join the expected mass with the actual pipeline output mass
            qa_df = orig_proj[['qa_zone', 'expected_mass']].join(mass_yield, how='left')
            qa_df['output_mass'] = qa_df['output_mass'].fillna(0.0)

            # 1. Interior Check: Should be exactly 1.0 (with a tiny 1% allowance for floating-point math)
            interior_failures = qa_df[(qa_df['qa_zone'] == 'Interior') & 
                                      (~np.isclose(qa_df['output_mass'], 1.0, atol=0.01))]
            interior_failures_idx = interior_failures.index.tolist()
            
            # 2. Boundary Check: Output mass must match the geometrically expected clipped mass
            # We allow a slightly wider 2% tolerance here to accommodate probabilistic Monte Carlo point clouds
            boundary_failures = qa_df[(qa_df['qa_zone'] == 'Boundary') & 
                                      (~np.isclose(qa_df['output_mass'], qa_df['expected_mass'], atol=0.02))]
            boundary_failures_idx = boundary_failures.index.tolist()

            if not interior_failures.empty:
                log_execution(logger, f"MASS LEAK: {len(interior_failures)} interior features failed Conservation of Mass.", level=logging.WARNING)
            if not boundary_failures.empty:
                log_execution(logger, f"BOUNDARY LEAK/HALLUCINATION: {len(boundary_failures)} edge features did not yield their geometrically expected mass.", level=logging.WARNING)

        # =====================================================================
        # 4. CHECK B: TOPOLOGICAL DRIFT (Classification)
        # =====================================================================
        # Triggered when continuous points are snapped into discrete grid cells using nearest-neighbor/intersection logic
        elif topology in ["point", "point_cloud"] and mapping_mode == "classification":
            if isinstance(targ_proj, gpd.GeoDataFrame) and 'grid_idx' in targ_proj.columns and 'geometry' in targ_proj.columns:
                drift_check = targ_proj.copy()
                
                # Align target geometries back to original geometries to measure the shift
                if drift_check.index.name != orig_proj.index.name:
                    drift_check = drift_check.join(orig_proj[['geometry', uncert_col]], lsuffix='_targ', rsuffix='_orig')
                
                if 'geometry_orig' in drift_check.columns:
                    # Calculate absolute Euclidean distance shifted
                    drift_check['drift_dist'] = drift_check['geometry_targ'].distance(drift_check['geometry_orig'])
                    
                    # A valid snap can shift the point up to its uncertainty radius PLUS the diagonal of the target cell.
                    # Anything further implies a projection failure or KDTree assignment error.
                    drift_check['allowed_drift'] = drift_check[uncert_col].fillna(default_uncert).astype(float) + cell_diagonal
                    
                    drift_failures = drift_check[drift_check['drift_dist'] > drift_check['allowed_drift']]
                    drift_failures_idx = drift_failures.index.tolist()
                    
                    if not drift_failures.empty:
                        log_execution(logger, f"DRIFT EXCEEDED: {len(drift_failures)} points snapped further than max allowance.", level=logging.WARNING)

        log_execution(logger, "=== Dynamic Vector QA/QC Complete ===", level=logging.INFO)
            
        return {
            "interior_mass_failures": interior_failures_idx,
            "boundary_mass_failures": boundary_failures_idx,
            "drift_failures": drift_failures_idx
        }