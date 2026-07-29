import gc
import logging
import math
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import KDTree
from pyproj import CRS
from rasterio.warp import transform_bounds
from shapely import force_2d, make_valid
from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint, box
from shapely.geometry.collection import GeometryCollection
from typing import Optional, List, Union, Tuple, Dict, Any
from bmc.engine.base import base_spatial_grid
from bmc.utils.logger import log_execution

class vector_engine(base_spatial_grid):
    """
    The fundamental vector spatial transformation and mapping engine.

    This class handles vector-specific geometrical operations, including converting coordinates
    to geometries, sanitizing topologies, generating probabilistic point clouds, and mapping
    vector data (points, point clouds, polygons) directly onto a mathematically rigid master grid template.

    Methods
    -------
    coordinate_to_geometry(df, x_col, y_col, uncert_col=None, output_type="polygon", input_crs="EPSG:4326", on_missing_uncertainty="fallback", quad_segs=8, point_cloud_config=None, logger=None)
        Converts tabular spatial records into a GeoDataFrame, optionally applying buffers or Monte Carlo point clouds.
    sanitize_geometries(gdf, allowed_types=None, force_multi=True, deduplicate=False, make_valid_method="linework", logger=None)
        Cleans, flattens, normalizes, and validates dirty vector geometries.
    _validate_geom_column(gdf, geom_column, allowed_types, context)
        Shared precondition checks to ensure column existence, non-emptiness, and geometry type constraints.
    _build_target_grid(target_grid_name, source_crs, source_bounds, target_bbox=None, logger=None)
        Shared grid-blueprint builder mapping localized domains to target grid specs.
    _ensure_crs(gdf, target_crs, logger=None)
        Validates the CRS of a GeoDataFrame against a target CRS and reprojects if necessary.
    generate_spatial_point_clouds(gdf, n_passes=30, uncertainty_col="coordinateuncertaintyinmeters", output_col="point_cloud", distribution="gaussian", random_state=None, logger=None)
        Generates memory-efficient spatial point clouds around feature centroids based on coordinate uncertainty.
    _compute_home_cell_mapping(reference_geom, uid_values, uid_col_name, tree, grid_idx_values, res, output_col_name="centroid_grid_idx")
        Calculates the nearest target-grid-cell mapping relative to centroid location using a KDTree.
    map_points_to_template(source_gdf, target_grid_name, geom_column="geometry", output_col="grid_idx", method="intersect", target_bbox=None, logger=None)
        Executes point geometries intersection directly mapped over a fixed alignment grid.
    map_point_cloud_to_template(source_gdf, target_grid_name, geom_column="point_cloud", output_col="grid_idx", mode="fractional", classify_method="intersect", fraction_col="fraction", target_bbox=None, logger=None)
        Computes probabilistic point clouds distribution intersects mapping fractional arrays on a spatial topology blueprint.
    map_polygon_to_template(source_gdf, target_grid_name, geom_column="geometry", output_col="grid_idx", target_bbox=None, min_areal_fraction=1e-6, include_centroid_tracking=True, logger=None)
        Maps polygon datasets by perfectly fracturing and distributing intersection weights onto a strict grid blueprint.
    map_cellCollection_to_template(source_gdf, target_grid_name, target_bbox=None, logger=None)
        Maps pre-aggregated, perfectly aligned grid cell polygons directly to the master template grid.
    validate_vector_transformation(orig_gdf, targ_gdf, recipe, dataset_name, logger=None)
        Validates the mathematical and topological integrity of a spatial transformation (e.g. tracking mass conservation and drift).
    """
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
        input_crs : str, optional
            The EPSG code or identifier for source coordinates. Default is "EPSG:4326".
        on_missing_uncertainty : {'fallback', 'raise'}, optional
            Controls behavior when uncertainty data is missing.
        quad_segs : int, optional
            Quarter-circle segment resolution for polygon buffering. Default is 8.
        point_cloud_config : dict, optional
            Configuration dictionary for point cloud generation.
        logger : logging.Logger, optional
            Instance of standard Python logger.
    
        Returns
        -------
        geopandas.GeoDataFrame
            A transformed GeoDataFrame assigned to the input_crs.
        """
        # =========================================================================
        # 1. PARAMETER VALIDATION
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
            raise ValueError("Found missing values in coordinates.")
    
        # =========================================================================
        # 2. PIPELINE SAFETY: UNCERTAINTY COLUMN CHECK
        # =========================================================================
        if output_type in ("polygon", "point_cloud"):
            if not uncert_col or uncert_col not in df.columns:
                if on_missing_uncertainty == "raise":
                    raise ValueError("Uncertainty column missing.")
                log_execution(logger, "Uncertainty column missing. Defaulting to 'point'.", level=logging.WARNING)
                output_type = "point"
    
        # =========================================================================
        # 3. INITIALIZE BASE SPATIAL TOPOLOGY
        # =========================================================================
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[x_col], df[y_col]),
            crs=input_crs,
        )
        gdf = gdf.drop(columns=[x_col, y_col])
    
        # =========================================================================
        # 4. GEOMETRY GENERATION
        # =========================================================================
        if output_type == "point":
            return gdf

        if output_type == "point_cloud":
            cfg = point_cloud_config or {}
            return self.generate_spatial_point_clouds(
                gdf=gdf,
                n_passes=cfg.get("n_passes", 30),
                uncertainty_col=uncert_col,
                output_col=cfg.get("output_col", "point_cloud"),
                distribution=cfg.get("distribution", "gaussian"),
                random_state=cfg.get("random_state", None),
                logger=logger,
            )

        # Polygon processing with buffering logic
        gdf[uncert_col] = gdf[uncert_col].fillna(0).clip(lower=0)
    
        if gdf.crs.is_geographic:
            # Map elements into local UPS/UTM zones prior to metric buffering
            gdf_wgs84 = gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs("EPSG:4326")
            
            latitudes = gdf_wgs84.geometry.y
            longitudes = gdf_wgs84.geometry.x
            
            zone_epsg = pd.Series(index=gdf.index, dtype="int64")
            utm_mask = ~( (latitudes >= 84.0) | (latitudes < -80.0) )
            
            if utm_mask.any():
                utm_zones = np.clip(((longitudes[utm_mask] + 180) / 6).astype(int) + 1, 1, 60)
                epsg_prefixes = np.where(latitudes[utm_mask] >= 0, 32600, 32700)
                zone_epsg.loc[utm_mask] = epsg_prefixes + utm_zones
    
            zone_col = "_coord_to_geom_zone_epsg"
            gdf[zone_col] = zone_epsg.astype(int)
    
            buffered_chunks = []
            for zone_epsg_code, group in gdf.groupby(zone_col):
                group_metric = group.to_crs(f"EPSG:{zone_epsg_code}")
                group_metric["geometry"] = group_metric.geometry.buffer(
                    group_metric[uncert_col], resolution=quad_segs
                )
                buffered_chunks.append(group_metric.to_crs(input_crs))
    
            gdf_polygon = gpd.GeoDataFrame(pd.concat(buffered_chunks).sort_index(), crs=input_crs)
            gdf_polygon = gdf_polygon.drop(columns=[zone_col])
        else:
            gdf_polygon = gdf.copy()
            gdf_polygon["geometry"] = gdf_polygon.geometry.buffer(
                gdf_polygon[uncert_col], resolution=quad_segs
            )
    
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
        Cleans, flattens, normalizes, and validates dirty vector geometries.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The raw input vector dataset.
        allowed_types : list of str, optional
            Case-sensitive geometry types allowed in the final output layer.
        force_multi : bool, default True
            If True, normalizes atomic geometries to their Multi* counterparts.
        deduplicate : bool, default False
            If True, removes geometric duplicates.
        make_valid_method : str, default 'linework'
            GEOS algorithm to fix broken topologies ('linework' or 'structure').
        logger : logging.Logger, optional
            Active pipeline logger instance.

        Returns
        -------
        geopandas.GeoDataFrame
            A valid, schema-homogenized dataset.
        """
        # 1. Validations and typings
        if gdf.empty:
            return gdf.copy()
            
        VALID_GEOM_TYPES = {
            "Point", "MultiPoint", "LineString", "MultiLineString", 
            "Polygon", "MultiPolygon", "GeometryCollection"
        }
        
        if allowed_types:
            for t in allowed_types:
                if t not in VALID_GEOM_TYPES:
                    raise ValueError(f"Invalid type '{t}' in allowed_types.")

        # 2. Purge Nulls
        gdf = gdf.dropna(subset=['geometry'])
        gdf = gdf[~gdf.geometry.is_empty].copy()

        # 3. Force 2D Planar Geometry
        gdf.geometry = shapely.force_2d(gdf.geometry.values)

        # 4. Topology Healing
        invalid_mask = ~gdf.geometry.is_valid.values
        if invalid_mask.any():
            healed_geoms = shapely.make_valid(gdf.loc[invalid_mask, 'geometry'].values, method=make_valid_method)
            gdf.loc[invalid_mask, 'geometry'] = healed_geoms

        # 5. Geometry Collections Handling
        healed_subset = gdf.loc[invalid_mask]
        collection_mask = (healed_subset.geometry.geom_type == "GeometryCollection").values
        
        if collection_mask.any():
            collection_indices = healed_subset.index[collection_mask]
            updated_geometries = []
            for idx in collection_indices:
                geom = gdf.loc[idx, 'geometry']
                parts = list(geom.geoms)
                
                polygons = [p for p in parts if p.geom_type in ['Polygon', 'MultiPolygon']]
                lines = [p for p in parts if p.geom_type in ['LineString', 'MultiLineString']]
                points = [p for p in parts if p.geom_type in ['Point', 'MultiPoint']]
                
                if polygons:
                    selected_geom = shapely.MultiPolygon(shapely.get_parts(polygons))
                elif lines:
                    selected_geom = shapely.MultiLineString(shapely.get_parts(lines))
                elif points:
                    selected_geom = shapely.MultiPoint(shapely.get_parts(points))
                
                updated_geometries.append(selected_geom)
            gdf.loc[collection_indices, 'geometry'] = updated_geometries

        # 6. Final validity check
        post_healing_invalid = ~gdf.geometry.is_valid.values
        if post_healing_invalid.any():
            gdf = gdf[~post_healing_invalid].copy()

        # 7. Homogenize types
        if force_multi and not gdf.empty:
            gdf.geometry = gdf.geometry.apply(
                lambda g: shapely.multipoints([g]) if g.geom_type == 'Point' else (
                          shapely.multilinestrings([g]) if g.geom_type == 'LineString' else (
                          shapely.multipolygons([g]) if g.geom_type == 'Polygon' else g))
            )

        # 8. Filter and Format Final
        if deduplicate and not gdf.empty:
            gdf = gdf.drop_duplicates(subset=['geometry']).copy()
            
        if allowed_types and not gdf.empty:
            type_mask = gdf.geometry.geom_type.isin(allowed_types).values
            gdf = gdf[type_mask].copy()

        return gdf.reset_index(drop=True)

    def _validate_geom_column(
        self,
        gdf: gpd.GeoDataFrame,
        geom_column: str,
        allowed_types: set,
        context: str,
    ) -> None:
        """
        Shared precondition checks to ensure column existence, non-emptiness,
        and geometry type constraints.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The input layer to test.
        geom_column : str
            The column containing target geometry objects.
        allowed_types : set
            A set containing permissible string geometry type values.
        context : str
            Logging string prefix for error messages.
        
        Raises
        ------
        ValueError
            If validation criteria fails.
        """
        if geom_column not in gdf.columns:
            raise ValueError(f"geom_column '{geom_column}' not found.")

        geo_col = gpd.GeoSeries(gdf[geom_column])
        if (geo_col.isna() | geo_col.is_empty).any():
            raise ValueError(f"{context}: found null/empty geometries in '{geom_column}'.")

        geom_types = set(geo_col.geom_type.dropna().unique())
        if not geom_types <= allowed_types:
            raise ValueError(f"{context}: expected {allowed_types}, found {geom_types - allowed_types}")

    def _build_target_grid(
        self,
        target_grid_name: str,
        source_crs: Union[str, CRS],
        source_bounds: Tuple[float, float, float, float],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[gpd.GeoDataFrame, str, float]:
        """
        Shared grid-blueprint builder mapping localized domains to target grid specs.

        Parameters
        ----------
        target_grid_name : str
            The master grid registry name.
        source_crs : str or pyproj.CRS
            The source coordinate reference system.
        source_bounds : tuple
            Extents representing the min/max X and Y dimensions.
        target_bbox : tuple, optional
            A predefined rigid boundary overriding source boundary transformation.
        logger : logging.Logger, optional
            Logger output context.
            
        Returns
        -------
        target_grid_gdf : geopandas.GeoDataFrame
            The aligned empty grid cells dataframe.
        target_crs : str
            Resolved Target grid CRS.
        res : float
            Mathematical cell resolution.
        """
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        dst_bbox = target_bbox if target_bbox is not None else transform_bounds(source_crs, target_crs, *source_bounds)
        
        # Pull empty canvas from Raster Engine helper logic
        template_da, _ = self.create_aligned_raster_template(dst_bbox, target_grid_name)
        res = template_da.attrs["res"]

        x_centers = template_da.x.values
        y_centers = template_da.y.values

        xx, yy = np.meshgrid(x_centers, y_centers)
        x_flat, y_flat = xx.ravel(), yy.ravel()
        half_res = res / 2.0

        # High speed box array constructions
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
        """
        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame lacks a defined CRS.")

        target = CRS.from_user_input(target_crs)
        if gdf.crs == target:
            return gdf
        
        if logger:
            logger.info(f"CRS mismatch detected. Reprojecting from {gdf.crs.name} to {target.name}...")
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
    
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            The input spatial dataset containing geometries and uncertainty values.
        n_passes : int, default 30
            Number of points generated per record.
        uncertainty_col : str, default 'coordinateuncertaintyinmeters'
            Name of column containing the radial standard uncertainty in meters.
        output_col : str, default 'point_cloud'
            Column mapping target destination for points cluster output.
        distribution : {'gaussian', 'uniform'}, default 'gaussian'
            Probability distribution pattern mapped on generated offsets.
        random_state : int, optional
            Seed to align reproducibility standards.
        logger : logging.Logger, optional
            Contextual event recording hook.
    
        Returns
        -------
        gpd.GeoDataFrame
            Modified spatial dataframe.
        """
        if distribution not in ("gaussian", "uniform"):
            raise ValueError("distribution must be either 'uniform' or 'gaussian'")
    
        # 1. Fetch metadata states
        rng = np.random.default_rng(random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            centroids = gdf.geometry.centroid
            x_coords = centroids.x.values
            y_coords = centroids.y.values
    
        uncertainties = gdf[uncertainty_col].fillna(0).values.astype(float)
        uncertainties = np.clip(uncertainties, 0, None)
    
        n_features = len(gdf)
        n_expanded = n_features * n_passes
    
        expanded_uncertainties = np.repeat(uncertainties, n_passes)
        expanded_x = np.repeat(x_coords, n_passes)
        expanded_y = np.repeat(y_coords, n_passes)
    
        # 2. Assign coordinate scatter distributions
        if distribution == "uniform":
            angles = rng.uniform(0, 2 * np.pi, n_expanded)
            radii_modifiers = np.sqrt(rng.uniform(0, 1, n_expanded))
            actual_radii = expanded_uncertainties * radii_modifiers
            delta_x = actual_radii * np.cos(angles)
            delta_y = actual_radii * np.sin(angles)
        else: 
            sigma = expanded_uncertainties / 3.0
            delta_x = rng.normal(loc=0.0, scale=sigma, size=n_expanded)
            delta_y = rng.normal(loc=0.0, scale=sigma, size=n_expanded)
    
        # 3. Reproject local differences dynamically mapped across geographic spheres
        is_geographic = gdf.crs is not None and gdf.crs.is_geographic
        if is_geographic:
            lat_for_scale = np.clip(expanded_y, -89.9, 89.9)
            meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat_for_scale))
            delta_x = delta_x / meters_per_deg_lon
            delta_y = delta_y / 111320.0
    
        new_x = expanded_x + delta_x
        new_y = expanded_y + delta_y
    
        if is_geographic:
            new_x = ((new_x + 180.0) % 360.0) - 180.0
    
        # 4. Collapse Points output
        coords_2d = np.column_stack((new_x, new_y))
        grouped_coords = coords_2d.reshape(n_features, n_passes, 2)
        multipoints = [MultiPoint(pts) for pts in grouped_coords]
    
        result_gdf = gdf.copy()
        result_gdf[output_col] = multipoints
        result_gdf["passes"] = n_passes
        result_gdf["weight_per_point"] = 1.0 / n_passes
    
        return result_gdf

    def _compute_home_cell_mapping(
        self,
        reference_geom: gpd.GeoSeries,
        uid_values: np.ndarray,
        uid_col_name: str,
        tree: KDTree,
        grid_idx_values: np.ndarray,
        res: float,
        output_col_name: str = "centroid_grid_idx",
    ) -> pd.DataFrame:
        """
        Calculates the nearest target-grid-cell mapping relative to centroid location.
        
        Parameters
        ----------
        reference_geom : gpd.GeoSeries
            Dataset feature geometries holding reference.
        uid_values : numpy.ndarray
            Array listing indices linking mapping configurations.
        uid_col_name : str
            Internal ID indexing column assignment naming convention.
        tree : scipy.spatial.KDTree
            Compiled Tree structure querying neighbors.
        grid_idx_values : numpy.ndarray
            Grid assignment array matching tree elements to mapped results.
        res : float
            Mathematical metric cell dimensional scaling limit.
        output_col_name : str, default 'centroid_grid_idx'
            Named configuration of final ID indexing format.
            
        Returns
        -------
        pandas.DataFrame
            Mapping table resolving IDs directly against grid allocations.
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
        """
        Executes point geometries intersection directly mapped over a fixed alignment grid.

        Parameters
        ----------
        source_gdf : geopandas.GeoDataFrame
            Initial dataset to evaluate against spatial templates.
        target_grid_name : str
            Grid blueprint reference ID dictating mesh allocations.
        geom_column : str, default 'geometry'
            Reference mapped geometrical field mapping.
        output_col : str, default 'grid_idx'
            Appended resulting column targeting mapped grid index string.
        method : {'intersect', 'kdtree'}, default 'intersect'
            Spatial join function mechanics evaluating grid mesh overlap points.
        target_bbox : tuple of float, optional
            A boundary crop bounding operation processing ranges.
        logger : logging.Logger, optional
            Event capturing configuration handler.

        Returns
        -------
        geopandas.GeoDataFrame
            Appended result mapping indexed features mapped accurately relative to spatial cell intersections.
        """
        # 1. Parameter Checking & Validation setup
        if method not in ("intersect", "kdtree"):
            raise ValueError("method must be 'kdtree' or 'intersect'.")
    
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)
    
        if source_gdf.empty:
            result = source_gdf.copy()
            result[output_col] = pd.array([], dtype="Int64")
            return result
    
        self._validate_geom_column(source_gdf, geom_column, {"Point"}, "map_points_to_template")
    
        # 2. Build aligned mesh components
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        grid_centroids = target_grid_gdf.geometry.centroid
        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )
    
        # 3. Processing method branch evaluations
        _uid_col = "_map_pts_src_uid_tmp"
        work_df = source_gdf[[geom_column]].copy()
        work_df[_uid_col] = source_gdf.index
        work_df = work_df.set_geometry(geom_column, crs=source_gdf.crs)
    
        if method == "kdtree":
            tree = KDTree(np.column_stack([grid_centroids.x.values, grid_centroids.y.values]))
            geom_coords = np.column_stack([work_df.geometry.x, work_df.geometry.y])
            _, nearest_idx = tree.query(geom_coords)
            mapping = pd.DataFrame({
                _uid_col: work_df[_uid_col].values,
                output_col: target_grid_gdf["grid_idx"].values[nearest_idx],
            })
        else:
            joined = gpd.sjoin(work_df, target_grid_gdf, how="inner", predicate="intersects")
            best_match = (
                joined.sort_values(
                    by=[_uid_col, "grid_idx"],
                    ascending=[True, True],
                    kind="mergesort",
                ).drop_duplicates(subset=[_uid_col])
            )
            mapping = best_match[[_uid_col, "grid_idx"]].rename(columns={"grid_idx": output_col})
    
        # 4. Integrate mapping output parameters mapping configuration dataset
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
        """
        Computes probabilistic point clouds distribution intersects mapping accurately 
        fractional mappings directly on the configured spatial topology blueprint templates.

        Parameters
        ----------
        source_gdf : geopandas.GeoDataFrame
            Initial dataframe defining cluster coordinate lists.
        target_grid_name : str
            ID mapped directly linking master blueprint mapping specs.
        geom_column : str, default 'point_cloud'
            Mapping column holding spatial clustering features lists.
        output_col : str, default 'grid_idx'
            Assignment configuration field holding grid mesh indices output values.
        mode : {'fractional', 'classify'}, default 'fractional'
            Method distributing calculation formats relative mapping cluster output data structures.
        classify_method : {'intersect', 'kdtree'}, default 'intersect'
            Engine evaluation method tracking index mapping logic routines.
        fraction_col : str, default 'fraction'
            Fraction weights array parameter tracking distributed weighting allocation logic arrays.
        target_bbox : tuple of float, optional
            A boundary bounding operation limit array parameters range limiting features calculations.
        logger : logging.Logger, optional
            Logger execution parameter output link mapping string status variables.

        Returns
        -------
        geopandas.GeoDataFrame or pandas.DataFrame
            The structured mapping matrix holding allocated cloud geometry spatial mappings matrix coordinates lists.
        """
        # 1. State Setup and parameter assertions checking 
        if mode not in ("fractional", "classify"):
            raise ValueError("mode must be 'fractional' or 'classify'.")

        orig_crs = source_gdf.crs
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)

        self._validate_geom_column(source_gdf, geom_column, {"MultiPoint", "Point"}, "map_point_cloud_to_template")
        original_geom_name = source_gdf.geometry.name
        self._validate_geom_column(source_gdf, original_geom_name, {"Point"}, "map_point_cloud_to_template (original)")

        from geopandas.array import GeometryDtype as _GeometryDtype
        geometry_cols = [c for c in source_gdf.columns if isinstance(source_gdf[c].dtype, _GeometryDtype)]
        preserve_cols = [c for c in source_gdf.columns if c not in geometry_cols and c != geom_column]

        if source_gdf.empty:
            if mode == "classify":
                result = source_gdf.copy()
                result[output_col] = pd.array([], dtype="Int64")
                result["src_grid_idx"] = pd.array([], dtype="Int64")
                if geom_column in result.columns and geom_column != result.geometry.name:
                    result = result.drop(columns=[geom_column])
                return result
            return pd.DataFrame(columns=[output_col, "src_grid_idx", fraction_col] + preserve_cols)

        # 2. Extract and define global mappings
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        grid_centroids = target_grid_gdf.geometry.centroid
        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )

        _uid_col = "_map_cloud_src_uid_tmp"
        work_df = source_gdf[[geom_column]].copy()
        work_df[_uid_col] = source_gdf.index

        if orig_crs != target_crs:
            work_df[geom_column] = gpd.GeoSeries(work_df[geom_column], crs=orig_crs).to_crs(target_crs)

        work_df = work_df.set_geometry(geom_column, crs=target_crs)

        # 3. Source Mapping Indexation
        orig_pts = source_gdf[[original_geom_name]].copy()
        orig_pts = orig_pts.set_geometry(original_geom_name, crs=target_crs)

        tree = KDTree(np.column_stack([grid_centroids.x.values, grid_centroids.y.values]))
        centroid_mapping = self._compute_home_cell_mapping(
            reference_geom=orig_pts.geometry,
            uid_values=source_gdf.index,
            uid_col_name=_uid_col,
            tree=tree,
            grid_idx_values=target_grid_gdf["grid_idx"].values,
            res=res,
            output_col_name="src_grid_idx"
        )

        # 4. Target Tree classification branch limits mapping arrays setup configs parameters
        if mode == "classify" and classify_method == "kdtree":
            mapping = self._compute_home_cell_mapping(
                reference_geom=work_df.geometry,
                uid_values=work_df[_uid_col].values,
                uid_col_name=_uid_col,
                tree=tree,
                grid_idx_values=target_grid_gdf["grid_idx"].values,
                res=res,
                output_col_name=output_col,
            )
            mapping = mapping.merge(centroid_mapping, on=_uid_col, how="left")

            result_gdf = source_gdf.copy()
            result_gdf[_uid_col] = result_gdf.index
            result_gdf = result_gdf.merge(mapping, on=_uid_col, how="left").drop(columns=[_uid_col])
            result_gdf.index = source_gdf.index
            
            for col in (output_col, "src_grid_idx"):
                if result_gdf[col].isna().any():
                    result_gdf[col] = result_gdf[col].astype("Int64")
                    
            if geom_column in result_gdf.columns and geom_column != result_gdf.geometry.name:
                result_gdf = result_gdf.drop(columns=[geom_column])
            return result_gdf

        # 5. Batched Intersect execution logic mappings 
        chunk_size = 10000 
        num_chunks = math.ceil(len(work_df) / chunk_size)
        counts_list, true_totals_list = [], []

        for i in range(num_chunks):
            chunk = work_df.iloc[i * chunk_size : (i + 1) * chunk_size]
            exploded_chunk = chunk.explode(index_parts=False).reset_index(drop=True)
            
            true_totals_chunk = exploded_chunk.groupby(_uid_col).size().rename("true_total_passes").reset_index()
            true_totals_list.append(true_totals_chunk)
            
            joined_chunk = gpd.sjoin(exploded_chunk, target_grid_gdf, how="inner", predicate="intersects")
            counts_chunk = joined_chunk.groupby([_uid_col, "grid_idx"]).size().reset_index(name="pt_count")
            counts_list.append(counts_chunk)
            
            del chunk, exploded_chunk, joined_chunk
            gc.collect()

        counts = pd.concat(counts_list, ignore_index=True)
        true_totals = pd.concat(true_totals_list, ignore_index=True)

        if mode == "classify":
            best_match = (
                counts.sort_values(
                    by=[_uid_col, "pt_count", "grid_idx"],
                    ascending=[True, False, True],
                    kind="mergesort",
                ).drop_duplicates(subset=[_uid_col])
            )
            mapping = best_match[[_uid_col, "grid_idx"]].rename(columns={"grid_idx": output_col})
            mapping = mapping.merge(centroid_mapping, on=_uid_col, how="left")

            result_gdf = source_gdf.copy()
            result_gdf[_uid_col] = result_gdf.index
            result_gdf = result_gdf.merge(mapping, on=_uid_col, how="left").drop(columns=[_uid_col])
            result_gdf.index = source_gdf.index
            
            for col in (output_col, "src_grid_idx"):
                if result_gdf[col].isna().any():
                    result_gdf[col] = result_gdf[col].astype("Int64")
                    
            if geom_column in result_gdf.columns and geom_column != result_gdf.geometry.name:
                result_gdf = result_gdf.drop(columns=[geom_column])
                
            return result_gdf

        counts = counts.merge(true_totals, on=_uid_col, how="left")
        counts[fraction_col] = counts["pt_count"] / counts["true_total_passes"]

        counts = counts.rename(columns={"grid_idx": output_col, _uid_col: "src_uid"})
        result_columns = [output_col, "src_grid_idx", fraction_col, "src_uid"] + preserve_cols

        centroid_mapping_renamed = centroid_mapping.rename(columns={_uid_col: "src_uid"})
        source_meta = source_gdf[preserve_cols].reset_index()
        source_meta = source_meta.rename(columns={source_gdf.index.name or "index": "src_uid"})

        result_df = counts[[output_col, "src_uid", fraction_col]].merge(
            centroid_mapping_renamed, on="src_uid", how="left"
        ).merge(
            source_meta,
            on="src_uid",
            how="left",
        )
        return result_df.reset_index(drop=True)[result_columns]
        
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
        """
        Maps polygon datasets perfectly fracturing relative grid intersection weights indices configurations layouts logic grids structures configurations constraints output mapping matrix table vectors parameters constraints layout parameters output maps constraints grids structures mapping limits matrices limits mapping parameters matrix parameters limits lists mapped matrices grids output lists boundaries bounds parameters variables geometries shapes tables limits values limits matrix parameters mapping lists mappings structures.

        Parameters
        ----------
        source_gdf : geopandas.GeoDataFrame
            Initial geometries lists constraints mappings structure bounds matrices mappings structures mapped bounds bounds mapped geometry formats matrices boundaries mapping mappings bounding mapping output logic boundaries grids mapped limits mapping limits parameters matrix output structure bounds.
        target_grid_name : str
            Grid ID string reference mapping geometry shapes intersections.
        geom_column : str, default 'geometry'
            Column containing dataset bounding boxes constraints polygons parameters matrices formats parameters matrices formats mappings.
        output_col : str, default 'grid_idx'
            Grid index output allocation matrix formats string parameters configuration variables configurations mapping logic configurations setups logic mappings tables arrays logic.
        target_bbox : tuple of float, optional
            Boundary list dimensions limits matrices forms constraints layouts tables lists limits matrix lists.
        min_areal_fraction : float, default 1e-6
            Lower mapping matrix threshold lists dropping empty array parameters structures.
        include_centroid_tracking : bool, default True
            Adds mapped bounding matrix features forms constraints forms mapping limits matrix parameters constraints layouts geometry mapping.
        logger : logging.Logger, optional
            Event streaming string limit matrix layouts arrays output configurations forms matrix string structures limits logging constraints layouts formats parameters forms setups arrays maps logic arrays variables bounds matrices variables.

        Returns
        -------
        pandas.DataFrame
            Shattered fragments bounding box geometries geometries mapping setups setups arrays maps limitations mapping structures matrices formats parameters mapped setups arrays matrices shapes boundaries limitations parameters.
        """
        # =====================================================================
        # 1. CRS VALIDATION AND GEOMETRY HYGIENE
        # =====================================================================
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)
        self._validate_geom_column(source_gdf, geom_column, {"Polygon", "MultiPolygon"}, "map_polygon_to_template")
    
        if (~source_gdf[geom_column].is_valid).any():
            raise ValueError("Topologically invalid geometries detected.")
    
        from pyproj import CRS as _CRS
        from geopandas.array import GeometryDtype as _GeometryDtype
        
        is_geographic_grid = _CRS.from_user_input(target_crs).is_geographic
        EQUAL_AREA_CRS = "EPSG:6933" 

        geometry_cols = [c for c in source_gdf.columns if isinstance(source_gdf[c].dtype, _GeometryDtype)]
        preserve_cols = [c for c in source_gdf.columns if c not in geometry_cols]
        result_columns = [output_col, "src_uid"] + (["centroid_grid_idx"] if include_centroid_tracking else []) + ["areal_fraction"] + preserve_cols
    
        if source_gdf.empty:
            return pd.DataFrame(columns=result_columns)
    
        # =====================================================================
        # 2. BUILD PRISTINE TARGET GRID MESH
        # =====================================================================
        target_grid_gdf, _, res = self._build_target_grid(
            target_grid_name=target_grid_name,
            source_crs=source_gdf.crs,
            source_bounds=source_gdf.total_bounds,
            target_bbox=target_bbox,
            logger=logger,
        )

        grid_centroids = target_grid_gdf.geometry.centroid
        target_grid_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=grid_centroids.x.values,
            y_coords=grid_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )
    
        # =====================================================================
        # 3. CALCULATE PRE-INTERSECT BASELINE AREAS
        # =====================================================================
        _uid_col = "_map_poly_src_uid_tmp"
        _area_col = "_map_poly_source_area_tmp"
    
        source_df = source_gdf.copy()
        source_df[_uid_col] = source_df.index
        source_df = source_df.set_geometry(geom_column, crs=source_gdf.crs)
        
        if is_geographic_grid:
            source_df[_area_col] = source_df.to_crs(EQUAL_AREA_CRS).geometry.area
        else:
            source_df[_area_col] = source_df.geometry.area
    
        source_df = source_df[source_df[_area_col] > 0]
    
        # =====================================================================
        # 4. TRACK ORIGINAL CENTROIDS
        # =====================================================================
        centroid_mapping = None
        if include_centroid_tracking:
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
        # 5. BATCHED INTERSECTION (SHATTERING)
        # =====================================================================
        chunk_size = 10000
        num_chunks = math.ceil(len(source_df) / chunk_size)
        intersections_list = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*keep_geom_type.*")
            for i in range(num_chunks):
                chunk = source_df.iloc[i * chunk_size : (i + 1) * chunk_size]
                intersection_chunk = gpd.overlay(
                    chunk[[_uid_col, _area_col, geom_column]],
                    target_grid_gdf,
                    how="intersection",
                )
                if not intersection_chunk.empty:
                    intersections_list.append(intersection_chunk)
                del chunk, intersection_chunk
                gc.collect()
    
        if not intersections_list:
            return pd.DataFrame(columns=result_columns)
            
        intersections = pd.concat(intersections_list, ignore_index=True)
    
        # =====================================================================
        # 6. CALCULATE FRACTIONAL YIELDS
        # =====================================================================
        if is_geographic_grid:
            intersections["intersect_area"] = intersections.to_crs(EQUAL_AREA_CRS).geometry.area
        else:
            intersections["intersect_area"] = intersections.geometry.area

        intersections["areal_fraction"] = intersections["intersect_area"] / intersections[_area_col]
        intersections = intersections[intersections["areal_fraction"] > min_areal_fraction]
    
        intersections = intersections.rename(columns={"grid_idx": output_col})
    
        # =====================================================================
        # 7. SCHEMA FORMATTING
        # =====================================================================
        result = intersections[[_uid_col, output_col, "areal_fraction"]]
        if include_centroid_tracking:
            result = result.merge(centroid_mapping, on=_uid_col, how="left")
            
        result = result.merge(
            source_gdf[preserve_cols].reset_index().rename(columns={source_gdf.index.name or "index": _uid_col}),
            on=_uid_col,
            how="left",
        )
        
        result = result.rename(columns={_uid_col: "src_uid"}).reset_index(drop=True)
        return result[result_columns]
    
    def map_cellCollection_to_template(
        self, 
        source_gdf: gpd.GeoDataFrame, 
        target_grid_name: str, 
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        logger: Optional[logging.Logger] = None
    ) -> gpd.GeoDataFrame:
        """
        Maps pre-aggregated, perfectly aligned grid cell polygons directly to the 
        master template grid.

        Parameters
        ----------
        source_gdf : geopandas.GeoDataFrame
            DataFrame containing raw target constraints geometry lists setups formats arrays configurations limits matrices bounds bounding arrays formats geometries constraints configurations configurations matrix.
        target_grid_name : str
            Dictionary key indicating target index mesh mappings mapping lists string structures formats constraints array parameters configuration parameters mapped setups.
        target_bbox : tuple of float, optional
            Area domain bounding boundary array variables bounds bounds parameters formats limits forms array arrays geometries structures boundaries boundaries parameters mappings forms limitations constraints tables lists limits parameters mappings forms geometries bounds variables strings variables string strings setup tables strings matrices lists boundaries parameters mappings strings.
        logger : logging.Logger, optional
            String pipeline events setup structures limitations logging matrix matrix parameters matrices configurations matrices formats bounds variables forms setup matrices matrices boundaries forms lists matrix parameters structures strings boundaries forms.

        Returns
        -------
        geopandas.GeoDataFrame
            Grid bounds mapped string string configurations formats lists strings parameters lists setups.
        """
        if source_gdf.crs is None:
            raise ValueError("Source GeoDataFrame is missing a CRS. Cannot project.")

        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        source_gdf = self._ensure_crs(source_gdf, target_crs, logger)

        log_execution(logger, "Mapping pre-gridded API cells directly to master grid indices...", logging.INFO)

        # 1. Extract centroid locations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cell_centroids = source_gdf.geometry.centroid

        # 2. Compute absolute indexes locations deterministic forms mappings formats array mapping matrix string configurations bounds string string forms arrays boundaries matrices tables limitations setups matrices lists parameters boundaries setups strings forms formats limits bounding variables.
        source_gdf["grid_idx"] = self.calculate_deterministic_global_indices(
            x_coords=cell_centroids.x.values,
            y_coords=cell_centroids.y.values,
            grid_name=target_grid_name,
            logger=logger
        )

        return source_gdf
    
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

        Parameters
        ----------
        orig_gdf : geopandas.GeoDataFrame
            The original, un-transformed geometries loaded from the source file.
        targ_gdf : geopandas.GeoDataFrame or pandas.DataFrame
            The transformed dataset outputted by the pipeline.
        recipe : dict
            The parsed YAML configuration dictionary used to dynamically extract the 
            target resolution, bounding box bounds, default uncertainty, topology, 
            and mapping mode.
        dataset_name : str
            The primary key inside the `sources` block of the recipe indicating which 
            configuration to parse.
        logger : logging.Logger, optional
            The active logger instance.

        Returns
        -------
        dict
            Dictionary tracking transformation failures for pipeline integrity.
        """
        log_execution(logger, "=== Initiating Dynamic Vector QA/QC Profiling ===", level=logging.INFO)
            
        # =====================================================================
        # 1. RECIPE EXTRACTION
        # =====================================================================
        spatial_cfg = recipe.get("spatial", {})
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        vector_cfg = source_cfg.get("vector_processing", {})
        query_filters = source_cfg.get("query_filters", {})
        
        topology = vector_cfg.get("topology", "point")
        mapping_mode = vector_cfg.get("mapping_mode", "classification")
        default_uncert = float(query_filters.get("default_Uncertainty", 1000.0))
        
        bbox_cfg = spatial_cfg.get("bbox", {})
        target_bbox = (
            bbox_cfg.get("long_min"), 
            bbox_cfg.get("lat_min"), 
            bbox_cfg.get("long_max"), 
            bbox_cfg.get("lat_max")
        )
        
        res_str = str(spatial_cfg.get("target_resolution", "1km")).lower()
        res_meters = float(res_str.replace("km", "")) * 1000 if res_str.endswith("km") else float(res_str.replace("m", "")) if res_str.endswith("m") else 1000.0
        cell_diagonal = math.sqrt(2 * (res_meters ** 2))

        # =====================================================================
        # 2. SPATIAL SETUP
        # =====================================================================
        if orig_gdf.crs.to_epsg() == 4326:
            proj_crs = orig_gdf.estimate_utm_crs()
            orig_proj = orig_gdf.to_crs(proj_crs)
            targ_proj = targ_gdf.to_crs(proj_crs) if isinstance(targ_gdf, gpd.GeoDataFrame) and not targ_gdf.empty else targ_gdf.copy()
        else:
            orig_proj = orig_gdf.copy()
            targ_proj = targ_gdf.copy()
            proj_crs = orig_proj.crs

        bbox_poly = box(*target_bbox)
        bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_poly]}, crs="EPSG:4326").to_crs(proj_crs)
        strict_bounds = bbox_gdf.geometry.iloc[0]

        uncert_col = next((c for c in orig_proj.columns if 'uncertainty' in c.lower()), 'coordinateuncertaintyinmeters')
        uncertainties = orig_proj[uncert_col].fillna(default_uncert).astype(float)
        
        dist_to_boundary = orig_proj.geometry.distance(strict_bounds.exterior)
        is_inside = orig_proj.geometry.within(strict_bounds)
        is_boundary = dist_to_boundary <= uncertainties
        is_interior = is_inside & (dist_to_boundary > uncertainties)

        orig_proj['qa_zone'] = np.where(is_interior, 'Interior', 
                                      np.where(is_boundary, 'Boundary', 'Exterior'))
        
        interior_failures_idx, boundary_failures_idx, drift_failures_idx = [], [], []

        # =====================================================================
        # 3. CHECK A: CONSERVATION OF MASS (Fractional Mapping)
        # =====================================================================
        if mapping_mode == "fractional":
            weight_col = "areal_fraction" if "areal_fraction" in targ_proj.columns else "fraction"
            link_col = 'src_uid' if 'src_uid' in targ_proj.columns else targ_proj.index.name or 'src_uid'
            if link_col not in targ_proj.columns:
                targ_proj['src_uid'] = targ_proj.index

            mass_yield = targ_proj.groupby(link_col)[weight_col].sum().rename("output_mass")
            
            proxy_geoms = orig_proj.geometry.copy()
            zero_area_mask = proxy_geoms.area == 0
            if zero_area_mask.any():
                proxy_geoms.loc[zero_area_mask] = proxy_geoms[zero_area_mask].buffer(uncertainties[zero_area_mask])
                
            orig_proj['expected_mass'] = proxy_geoms.intersection(strict_bounds).area / proxy_geoms.area
            qa_df = orig_proj[['qa_zone', 'expected_mass']].join(mass_yield, how='left').fillna(0.0)

            interior_failures = qa_df[(qa_df['qa_zone'] == 'Interior') & (~np.isclose(qa_df['output_mass'], 1.0, atol=0.01))]
            interior_failures_idx = interior_failures.index.tolist()
            
            boundary_failures = qa_df[(qa_df['qa_zone'] == 'Boundary') & (~np.isclose(qa_df['output_mass'], qa_df['expected_mass'], atol=0.02))]
            boundary_failures_idx = boundary_failures.index.tolist()

        # =====================================================================
        # 4. CHECK B: TOPOLOGICAL DRIFT (Classification)
        # =====================================================================
        elif topology in ["point", "point_cloud"] and mapping_mode == "classification":
            if isinstance(targ_proj, gpd.GeoDataFrame) and 'grid_idx' in targ_proj.columns and 'geometry' in targ_proj.columns:
                drift_check = targ_proj.copy()
                
                if drift_check.index.name != orig_proj.index.name:
                    drift_check = drift_check.join(orig_proj[['geometry', uncert_col]], lsuffix='_targ', rsuffix='_orig')
                
                if 'geometry_orig' in drift_check.columns:
                    drift_check['drift_dist'] = drift_check['geometry_targ'].distance(drift_check['geometry_orig'])
                    drift_check['allowed_drift'] = drift_check[uncert_col].fillna(default_uncert).astype(float) + cell_diagonal
                    drift_failures = drift_check[drift_check['drift_dist'] > drift_check['allowed_drift']]
                    drift_failures_idx = drift_failures.index.tolist()

        return {
            "interior_mass_failures": interior_failures_idx,
            "boundary_mass_failures": boundary_failures_idx,
            "drift_failures": drift_failures_idx
        }