import os
import json
import logging
from typing import Optional, Union, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
import shapely
import shapely.geometry
from rasterio.warp import transform_bounds

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import dask_geopandas as dask_gpd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import pystac
    HAS_PYSTAC = True
except ImportError:
    HAS_PYSTAC = False

from bmc.engine.vector import vector_engine
from bmc.utils.logger import log_execution

class vector_cube(vector_engine, ABC):

    def fetch_vector_by_bbox(
        self,
        file_path: str, 
        target_grid_name: str,
        target_bbox: Tuple[float, float, float, float],
        engine: str = 'auto',
        use_dask: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> Union[gpd.GeoDataFrame, 'dask_gpd.GeoDataFrame']:
        """
        Fetches a spatial subset of a remote or local vector dataset.
        
        Parameters
        ----------
        file_path : str
            URL or local path to the vector file (e.g., .parquet, .gpkg, .fgb).
        target_grid_name : str
            The precise registry key of the target grid to calculate projection curvature.
        target_bbox : tuple
            Strict bounding box in the format (minx, miny, maxx, maxy).
        engine : str
            'auto', 'geopandas', or 'duckdb'. 
        use_dask : bool
            If True, returns a lazy dask-geopandas dataframe for out-of-core processing.
        """
        is_parquet = file_path.lower().endswith(('.parquet', '.geoparquet'))
        is_remote = file_path.startswith(('http', 's3://'))
        
        # 1. Generate the padded envelope BEFORE routing to prevent edge starvation across ALL engines
        safe_bbox = self.build_safe_fetch_envelope(
            target_grid_name=target_grid_name, 
            target_bounds=target_bbox, # The strict metric template bounds
            source_crs_or_grid="EPSG:4326", 
            pixel_buffer=2, # Small buffer to capture curved edge polygons
            logger=logger
        )
        safe_minx, safe_miny, safe_maxx, safe_maxy = safe_bbox
        
        # ---------------------------------------------------------
        # DASK OUT-OF-CORE ENGINE
        # ---------------------------------------------------------
        if use_dask:
            if not HAS_DASK:
                raise ImportError("dask_geopandas is required for out-of-core processing.")
            if not is_parquet:
                raise ValueError("Dask-GeoPandas currently best supports GeoParquet for distributed reads.")
                
            log_execution(logger, f"Lazy loading massive GeoParquet via Dask: {file_path}", logging.INFO)
            import shapely.geometry
            ddf = dask_gpd.read_parquet(file_path)
            return ddf.clip(shapely.geometry.box(*safe_bbox))

        # ---------------------------------------------------------
        # AUTO-DETECTION ROUTER
        # ---------------------------------------------------------
        if engine == 'auto':
            # DuckDB is generally vastly superior for remote Parquet files
            if HAS_DUCKDB and is_parquet:
                log_execution(logger, "Auto-routing to DuckDB (Optimized for Parquet).", logging.INFO)
                engine = 'duckdb'
            # For local GPKG/FGB, native GeoPandas/GDAL is usually fast enough
            else:
                log_execution(logger, "Auto-routing to native GeoPandas engine.", logging.INFO)
                engine = 'geopandas'

        # ---------------------------------------------------------
        # DUCKDB ANALYTICAL ENGINE
        # ---------------------------------------------------------
        if engine == 'duckdb':
            if not HAS_DUCKDB:
                raise ImportError("duckdb is required to use the analytical query engine.")
                
            log_execution(logger, f"Executing DuckDB spatial pushdown on: {file_path}", logging.INFO)
            
            duckdb.execute("INSTALL spatial; LOAD spatial;")
            if is_remote:
                duckdb.execute("INSTALL httpfs; LOAD httpfs;") 
            
            from_clause = f"read_parquet('{file_path}')" if is_parquet else f"ST_Read('{file_path}')"
            
            # Pass the safely padded envelope to the ST_MakeEnvelope spatial filter
            query = f"""
                SELECT * 
                FROM {from_clause}
                WHERE ST_Intersects(
                    geometry, 
                    ST_MakeEnvelope({safe_minx}, {safe_miny}, {safe_maxx}, {safe_maxy})
                )
            """
            # Execute the query and return the spatial dataframe
            return gpd.GeoDataFrame(duckdb.query(query).df(), geometry='geometry')

        # ---------------------------------------------------------
        # NATIVE GEOPANDAS / GDAL ENGINE
        # ---------------------------------------------------------
        if engine == 'geopandas':
            log_execution(logger, f"Fetching subset using native engine: {file_path}", logging.INFO)
            
            if is_parquet:
                return gpd.read_parquet(file_path, bbox=safe_bbox)
            else:
                if is_remote:
                    file_path = file_path.replace("s3://", "/vsis3/") if file_path.startswith("s3://") else f"/vsicurl/{file_path}"
                return gpd.read_file(file_path, bbox=safe_bbox)
    
    @abstractmethod
    def fetch_data(self, recipe: Dict[str, Any], logger: Optional[logging.Logger] = None) -> gpd.GeoDataFrame:
        """
        Translates the execution recipe into raw vector data retrieval.
        Child classes must implement this to query their specific catalogs and return a GeoDataFrame.
        """
        pass

    @abstractmethod
    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: logging.Logger) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key.
        """
        pass

    def _apply_cf_temporal_standards(
    self, 
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    time_cols: Optional[List[str]] = None
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Internal: Converts split temporal columns into CF-compliant ISO-8601 datetimes.
        Safely processes and returns either DataFrames or GeoDataFrames.
        """
        if time_cols is None:
            time_cols = ["year", "month", "day"]
            
        # Safeguard: check if all specified time columns exist in the DataFrame
        missing_cols = [col for col in time_cols if col not in df.columns]
        if missing_cols:
            return df

        work_df = df.copy()
        
        # Build datetime string dynamically depending on the provided temporal resolution
        if len(time_cols) == 3:
            y, m, d = time_cols
            dt_str = work_df[y].astype(str) + '-' + work_df[m].astype(str).str.zfill(2) + '-' + work_df[d].astype(str).str.zfill(2)
        elif len(time_cols) == 2:
            y, m = time_cols
            dt_str = work_df[y].astype(str) + '-' + work_df[m].astype(str).str.zfill(2) + '-01'
        elif len(time_cols) == 1:
            y = time_cols[0]
            dt_str = work_df[y].astype(str) + '-01-01'
        else:
            # Fallback if an empty list or invalid format is passed
            return work_df

        # Convert to datetime and drop the old split columns
        work_df['datetime'] = pd.to_datetime(dt_str, format='%Y-%m-%d', errors='coerce')
        work_df = work_df.drop(columns=time_cols)
            
        return work_df

    def process_cube(
        self, 
        recipe: Dict[str, Any], 
        dataset_name: str, 
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> 'pystac.Item':        
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        processing_mode = source_cfg.get("processing_mode", "vector").lower()
        time_cols = source_cfg.get("time_cols", ["year", "month", "day"])

        # =====================================================================
        # 1. DEFINE DIRECTORY STRUCTURE (cube_dir/dataset_dir)
        # =====================================================================
        base_dir = recipe.get("base_dir", "./")
        cube_name = recipe.get("cube_name", "default_cube")
        
        # Build the nested directory path
        cube_dir = os.path.join(base_dir, cube_name)
        dataset_dir = os.path.join(cube_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Pre-allocate expected filepaths
        source_geom_path = os.path.join(dataset_dir, f"{dataset_name}_source_geometries.parquet")
        unaggregated_path = os.path.join(dataset_dir, f"{dataset_name}_unaggregated.parquet")
        aggregated_path = os.path.join(dataset_dir, f"{dataset_name}_aggregated.parquet")

        log_execution(logger, f"\n=== Initiating {dataset_name.upper()} Generation ===", logging.INFO)
        log_execution(logger, f"Processing Mode: {processing_mode.upper()}", logging.INFO)
        log_execution(logger, f"Output Directory: {dataset_dir}", logging.INFO)

        # =====================================================================
        # 2. FETCH & BUILD BASE TOPOLOGY DATA
        # =====================================================================
        raw_gdf = self.fetch_data(recipe, logger=logger, **kwargs)
        if raw_gdf.empty:
            log_execution(logger, f"Terminating pipeline: fetched {dataset_name} dataset is empty.", logging.WARNING)
            return None

        # Handle Eject Button (Raw Mode)
        if processing_mode == "raw":
            raw_path = os.path.join(dataset_dir, f"{dataset_name}_raw.parquet")
            log_execution(logger, f"Bypassing spatial engine. Exporting RAW data to {raw_path}...", logging.INFO)
            raw_gdf = self._apply_cf_temporal_standards(raw_gdf, time_cols)
            raw_gdf.to_parquet(raw_path)
            # Cannot generate a valid spatial STAC item without a grid index
            return raw_gdf

        # Sanitize Data
        sanitized_gdf = self.sanitize_geometries(raw_gdf, force_multi=False, logger=logger)

        # =====================================================================
        # 3. RESOLVE MASTER GRID & BOUNDING BOX
        # =====================================================================
        spatial_cfg = recipe.get('spatial', {})
        target_grid_key = self.resolve_target_grid(spatial_cfg, logger)

        target_bbox = None
        if spatial_cfg.get('use_bbox', False) and 'bbox' in spatial_cfg:
            from rasterio.warp import transform_bounds
            log_execution(logger, "Extracting padded global bounding box from recipe...", logging.INFO)
            bbox_cfg = spatial_cfg['bbox']
            
            grid_info = self.GRID_REGISTRY[target_grid_key]
            target_crs = grid_info["crs"]
            target_res = grid_info["resolution"]
            
            wgs84_bounds = (
                bbox_cfg["long_min"], bbox_cfg["lat_min"], 
                bbox_cfg["long_max"], bbox_cfg["lat_max"]
            )
            strict_bbox = transform_bounds("EPSG:4326", target_crs, *wgs84_bounds)
            target_bbox = (
                strict_bbox[0] - target_res, strict_bbox[1] - target_res,
                strict_bbox[2] + target_res, strict_bbox[3] + target_res
            )
        
        # =====================================================================
        # 4. DYNAMIC SPATIAL ROUTING
        # =====================================================================
        if processing_mode == "api_cube":
            transformed_data = self.map_cellCollection_to_template(
                source_gdf=sanitized_gdf,
                target_grid_name=target_grid_key,
                target_bbox=target_bbox,
                logger=logger
            )

        elif processing_mode == "vector":
            vector_cfg = source_cfg.get("vector_processing", {})
            topology = vector_cfg.get("topology", "point")
            mapping_mode = vector_cfg.get("mapping_mode", "fractional")
            spatial_method = vector_cfg.get("spatial_method", "intersect")
            
            if topology == "point":
                transformed_data = self.map_points_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    output_col="grid_idx",
                    method=spatial_method,
                    target_bbox=target_bbox,
                    logger=logger
                )
            elif topology == "point_cloud":
                transformed_data = self.map_point_cloud_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    geom_column=topology,
                    output_col="grid_idx",
                    mode=mapping_mode,
                    classify_method=spatial_method,
                    target_bbox=target_bbox,
                    logger=logger
                )
            elif topology == "polygon":
                transformed_data = self.map_polygon_to_template(
                    source_gdf=sanitized_gdf,
                    target_grid_name=target_grid_key,
                    geom_column="geometry",
                    output_col="grid_idx",
                    target_bbox=target_bbox,
                    logger=logger
                )
            else:
                raise ValueError(f"Unsupported topology '{topology}' specified in recipe.")

        # =====================================================================
        # 5. DYNAMIC QA/QC VALIDATION & SOURCE EXPORTS
        # =====================================================================
        aggregate_cfg = source_cfg.get("aggregate", recipe.get("aggregate", {}))
        
        if processing_mode == "vector":
            qa_report = self.validate_vector_transformation(
                orig_gdf=sanitized_gdf,
                targ_gdf=transformed_data,
                recipe=recipe,
                dataset_name=dataset_name,
                logger=logger
            )

            validation_dir = os.path.join(dataset_dir, 'validation_report')
            os.makedirs(validation_dir, exist_ok=True)
            report_path = os.path.join(validation_dir, f"{dataset_name}_qa_qc_failures.json")
            
            import json
            with open(report_path, 'w') as f:
                json.dump(qa_report, f, indent=4)
                
            log_execution(logger, f"Validation report saved to {report_path}", logging.INFO)
            
        else:
            log_execution(logger, "Relational table output detected. Skipping geometric vector QA/QC.", logging.INFO)
            
        # Export the Source Geometries if relational (i.e. shattered vectors converted to pure pandas tables)
        if not isinstance(transformed_data, gpd.GeoDataFrame):
            log_execution(logger, f"Exporting source geometries to {source_geom_path}...", logging.INFO)
            source_export = sanitized_gdf.copy()
            source_export["src_uid"] = source_export.index 
            
            extra_geom_cols = [
                c for c in source_export.columns 
                if c != source_export.geometry.name and (c == "point_cloud" or getattr(source_export[c], "dtype", None) == "geometry")
            ]
            if extra_geom_cols:
                source_export = source_export.drop(columns=extra_geom_cols)

            source_export = self._apply_cf_temporal_standards(source_export, time_cols)
            source_export.to_parquet(source_geom_path)
        else:
            source_geom_path = None # Prevents STAC from referencing missing file

        # Track final paths to pass to STAC
        final_unaggregated_path = None
        final_aggregated_path = None

        # Export unaggregated shattered fractions if requested (or if no aggregation is set)
        if aggregate_cfg.get("export_unaggregated", False) or not aggregate_cfg:
            log_execution(logger, f"Exporting unaggregated relational table to {unaggregated_path}...", logging.INFO)
            unagg_export = self._apply_cf_temporal_standards(transformed_data, time_cols)
            if "grid_idx" in unagg_export.columns:
                unagg_export = unagg_export.dropna(subset=["grid_idx"]).reset_index(drop=True)
            unagg_export.to_parquet(unaggregated_path)
            final_unaggregated_path = unaggregated_path

        # =====================================================================
        # 6. AGGREGATION & FINAL EXPORT
        # =====================================================================
        if processing_mode == "vector" and aggregate_cfg:
            transformed_data = self.aggregate_vector_cube(
                data=transformed_data,
                recipe=recipe,
                dataset_name=dataset_name,
                logger=logger
            )
            
            if "grid_idx" in transformed_data.columns:
                out_of_bounds_count = transformed_data["grid_idx"].isna().sum()
                if out_of_bounds_count > 0:
                    log_execution(logger, f"Filtering out {out_of_bounds_count} records that fell outside the target bounding box...", logging.INFO)
                    transformed_data = transformed_data.dropna(subset=["grid_idx"]).reset_index(drop=True)

            log_execution(logger, f"Exporting final aggregated spatial dataset to {aggregated_path}...", logging.INFO)
            transformed_data = self._apply_cf_temporal_standards(transformed_data, time_cols)
            transformed_data.to_parquet(aggregated_path)
            final_aggregated_path = aggregated_path

        log_execution(logger, f"=== {dataset_name.upper()} Generation Complete ===", logging.INFO)
        
        # =====================================================================
        # 7. GENERATE AND RETURN STAC ITEM
        # =====================================================================
        stac_id = f"{cube_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        stac_item = self.generate_vector_stac_item(
            df=transformed_data,
            recipe=recipe,
            dataset_name=dataset_name,
            source_geom_path=source_geom_path,
            output_dir=dataset_dir,               # Places the grid dimension table cleanly next to the parquet files
            unaggregated_path=final_unaggregated_path,
            aggregated_path=final_aggregated_path,
            item_id=stac_id,
            logger=logger
        )

        # ---------------------------------------------------------------------
        # Export STAC to the meta directory
        # ---------------------------------------------------------------------
        meta_dir = os.path.join(cube_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        stac_filepath = os.path.join(meta_dir, f"{dataset_name}_stac.json")
        
        with open(stac_filepath, "w") as f:
            json.dump(stac_item.to_dict(), f, indent=4)
            
        log_execution(logger, f"Exported STAC item to {stac_filepath}", logging.INFO)

        return stac_item
    
    def aggregate_vector_cube(
    self,
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    recipe: Dict[str, Any],
    dataset_name: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
        """
        Dedicated aggregation engine for Vector processing mode.
        Dynamically groups by recipe-defined dimensions and executes math 
        according to the metrics map, safely handling fractional weights.
        """
        dataset_cfg = recipe.get("sources", {}).get(dataset_name, {})
        aggregate_cfg = dataset_cfg.get("aggregate", recipe.get("aggregate", {}))

        if not aggregate_cfg:
            log_execution(logger, "No 'aggregate' block found. Returning vector data as-is.", logging.WARNING)
            return data

        # 1. Resolve Grouping Dimensions
        recipe_groups = aggregate_cfg.get("group_by_columns", [])
        
        spatial_col = "grid_id" if "grid_id" in data.columns else "grid_idx"
        group_cols = [spatial_col] + recipe_groups if spatial_col not in recipe_groups else recipe_groups
        actual_group_cols = [col for col in group_cols if col in data.columns]
        
        log_execution(logger, f"Aggregating vector data by dimensions: {actual_group_cols}...", logging.INFO)

        # 2. Determine Spatial Weights for Additive Math
        fraction_col = next((c for c in ["areal_fraction", "fraction"] if c in data.columns), None)
        if fraction_col:
            log_execution(logger, f"Fractional geometries detected. Spatial multiplier: '{fraction_col}'.", logging.INFO)
            data['_base_weight'] = data[fraction_col]
        else:
            log_execution(logger, "Discrete classification detected. Spatial multiplier set to 1.0.", logging.INFO)
            data['_base_weight'] = 1.0

        # 3. Initialize Cube
        aggregated = data[actual_group_cols].drop_duplicates().set_index(actual_group_cols)

        # 4. Execute YAML Metrics Map
        metrics = aggregate_cfg.get("metrics", [])
        for metric in metrics:
            col = metric.get("column")
            method = metric.get("method", "nunique")
            weighted = metric.get("weighted", False)
            new_name = metric.get("rename", f"{col}_{method}")
            
            if col not in data.columns:
                log_execution(logger, f"Metric column '{col}' missing from data payload. Skipping.", logging.WARNING)
                continue
                
            log_execution(logger, f"Aggregating '{col}' -> '{new_name}' (Method: {method}, Weighted: {weighted})", logging.INFO)
                
            # Apply fractional multiplier for additive metrics to conserve mass
            if weighted and method in ['sum', 'mean']:
                temp_col = f"_weighted_{col}"
                # Catch null abundance counts and default to 1.0 to preserve the presence
                numeric_series = pd.to_numeric(data[col], errors='coerce').fillna(1.0 if method == 'sum' else 0.0)
                data[temp_col] = data['_base_weight'] * numeric_series
                aggregated[new_name] = data.groupby(actual_group_cols, dropna=False)[temp_col].agg(method)
            else:
                # Standard metrics (e.g., nunique for Observers and Species)
                aggregated[new_name] = data.groupby(actual_group_cols, dropna=False)[col].agg(method)

        # 5. Cleanup memory
        cols_to_drop = [c for c in data.columns if c.startswith('_weighted_') or c == '_base_weight']
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        log_execution(logger, f"Vector aggregation complete. Yielded {len(aggregated)} final cells.", logging.INFO)
        
        return aggregated.reset_index()

    def generate_vector_stac_item(
        self,
        df: pd.DataFrame,
        recipe: dict,
        dataset_name: str,
        source_geom_path: str,
        output_dir: str,
        unaggregated_path: Optional[str] = None,
        aggregated_path: Optional[str] = None,
        template_zarr_path: Optional[str] = None,
        item_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Converts the relational output of process_cube into a fully compliant STAC Item, 
        embedding topology provenance (e.g., point cloud distribution) and aggregation configurations.
        
        This method generates a Spatial Dimension Table mapping unique grid_idx values 
        to their physical cell geometries, and binds all generated physical files 
        (source geometries, unaggregated fractions, aggregated cubes) as STAC Assets.

        Parameters
        ----------
        df : pandas.DataFrame
            The final dataframe (aggregated or unaggregated) containing 'grid_idx' 
            to derive the spatial footprint.
        recipe : dict
            The pipeline YAML configuration to extract grid definitions and provenance metadata.
        dataset_name : str
            The name of the dataset (e.g., 'gbif').
        source_geom_path : str
            Filepath to the saved original source GeoParquet.
        output_dir : str
            Directory to save the dynamically generated Spatial Dimension GeoParquet.
        unaggregated_path : str, optional
            Filepath to the unaggregated relational fractions/classifications Parquet.
        aggregated_path : str, optional
            Filepath to the final aggregated vector cube Parquet.
        template_zarr_path : str, optional
            Filepath to the Zarr template, if generated.
        item_id : str, optional
            A unique identifier for the STAC Item. Defaults to dataset_name_timestamp.
        logger : logging.Logger, optional
            Pipeline logger.

        Returns
        -------
        pystac.Item
            The assembled STAC Item ready to be added to the bmd_cube Collection.
        """
        if not HAS_PYSTAC:
            raise ImportError("The 'pystac' library is required to generate STAC items. Run: pip install pystac")

        if logger: logger.info(f"=== Assembling STAC Item for {dataset_name.upper()} ===")

        # =====================================================================
        # 1. EXTRACT SPATIAL SPECS & TEMPORAL BOUNDS
        # =====================================================================
        spatial_cfg = recipe.get("spatial", {})
        target_grid = spatial_cfg.get("target_grid", "EEA")
        target_res = spatial_cfg.get("target_resolution", "1km")
        grid_name = f"{target_grid}_{target_res}"
        
        spec = self.GRID_REGISTRY[grid_name]
        res = spec["resolution"]
        master_minx, _, master_maxx, master_maxy = spec["bounds"]

        # Derive accurate temporal bounds from the actual dataset contents
        if 'datetime' in df.columns and not df['datetime'].isna().all():
            start_dt = df['datetime'].min().to_pydatetime().replace(tzinfo=timezone.utc)
            # Push end_dt to the end of the final month/year if applicable
            end_dt = df['datetime'].max().to_pydatetime().replace(tzinfo=timezone.utc)
        elif 'year' in df.columns:
            start_year = int(df['year'].min())
            end_year = int(df['year'].max())
            start_dt = datetime(start_year, 1, 1, tzinfo=timezone.utc)
            # If month is present, set to end of that month, else end of year
            end_month = int(df['month'].max()) if 'month' in df.columns else 12
            if end_month == 12:
                end_dt = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            else:
                # Handle end of specific month safely
                import calendar
                last_day = calendar.monthrange(end_year, end_month)[1]
                end_dt = datetime(end_year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        else:
            start_dt = end_dt = datetime.now(timezone.utc)

        # =====================================================================
        # 2. EXTRACT PROCESSING PROVENANCE METADATA
        # =====================================================================
        source_cfg = recipe.get("sources", {}).get(dataset_name, {})
        vector_cfg = source_cfg.get("vector_processing", {})
        aggregate_cfg = source_cfg.get("aggregate", {})
        
        topology = vector_cfg.get("topology", "point")
        mapping_mode = vector_cfg.get("mapping_mode", "classification")
        topology_cfg = vector_cfg.get("topology_config", {}).get(topology, {})

        stac_properties = {
            "cube:dataset": dataset_name,
            "cube:grid_registry_key": grid_name,
            "processing:topology": topology,
            "processing:mapping_mode": mapping_mode,
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
        }

        # Encode topology-specific parameters
        if topology == "point_cloud":
            stac_properties["processing:distribution"] = topology_cfg.get("distribution", "gaussian")
            stac_properties["processing:n_passes"] = topology_cfg.get("n_passes", 30)
            stac_properties["processing:random_seed"] = topology_cfg.get("random_seed", None)
        elif topology == "polygon":
            stac_properties["processing:quad_segs"] = topology_cfg.get("quad_segs", 8)

        # Encode aggregation parameters if an aggregation run was performed
        if aggregate_cfg:
            stac_properties["processing:aggregation_groups"] = aggregate_cfg.get("group_by_columns", [])
            stac_properties["processing:aggregation_metrics"] = aggregate_cfg.get("metrics", [])

        # =====================================================================
        # 3. GENERATE THE SPATIAL DIMENSION TABLE (CELL GEOMETRIES)
        # =====================================================================
        if logger: logger.info("Generating discrete Spatial Dimension Table from unique grid IDs...")
        
        # Accommodate 'grid_idx' or 'grid_id' naming conventions
        spatial_col = "grid_idx" if "grid_idx" in df.columns else "grid_id"
        unique_grid_ids = df[spatial_col].dropna().unique()
        
        total_global_cols = int(round((master_maxx - master_minx) / res))
        global_rows = unique_grid_ids // total_global_cols
        global_cols = unique_grid_ids % total_global_cols

        x_centers = master_minx + (global_cols * res) + (res / 2.0)
        y_centers = master_maxy - (global_rows * res) - (res / 2.0)

        half_res = res / 2.0
        polygons = shapely.box(
            x_centers - half_res, 
            y_centers - half_res, 
            x_centers + half_res, 
            y_centers + half_res
        )

        grid_gdf = gpd.GeoDataFrame(
            {spatial_col: unique_grid_ids}, 
            geometry=polygons, 
            crs=spec["crs"]
        )

        grid_mapping_path = os.path.join(output_dir, f"{dataset_name}_{grid_name}_dimension_table.parquet")
        grid_gdf.to_parquet(grid_mapping_path)
        if logger: logger.info(f"Spatial Dimension Table saved to: {grid_mapping_path}")

        # =====================================================================
        # 4. CALCULATE STAC-COMPLIANT BBOX (EPSG:4326)
        # =====================================================================
        wgs84_bounds = grid_gdf.to_crs("EPSG:4326").total_bounds
        bbox = [float(wgs84_bounds[0]), float(wgs84_bounds[1]), float(wgs84_bounds[2]), float(wgs84_bounds[3])]
        footprint = shapely.geometry.mapping(shapely.box(*bbox))

        # =====================================================================
        # 5. CONSTRUCT THE PYSTAC ITEM
        # =====================================================================
        stac_id = item_id or f"{dataset_name}_vector_cube_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        item = pystac.Item(
            id=stac_id,
            geometry=footprint,
            bbox=bbox,
            datetime=start_dt,
            properties=stac_properties
        )

        # =====================================================================
        # 6. ATTACH THE ASSETS (THE STAR SCHEMA LINKS)
        # =====================================================================
        
        # Asset 1: Original Source Geometries (Always Present)
        if source_geom_path and os.path.exists(source_geom_path):
            item.add_asset(
                "source_geometries",
                pystac.Asset(
                    href=source_geom_path,
                    media_type="application/x-parquet",
                    roles=["metadata", "spatial-index"],
                    title=f"Source Geometries ({topology.replace('_', ' ').title()})",
                    description="Original vector records containing src_uid and generated spatial topologies."
                )
            )

        # Asset 2: Unaggregated Relational Data (If Exported)
        if unaggregated_path and os.path.exists(unaggregated_path):
            item.add_asset(
                "unaggregated_data",
                pystac.Asset(
                    href=unaggregated_path,
                    media_type="application/x-parquet",
                    roles=["data"],
                    title="Unaggregated Relational Data",
                    description=f"Raw {mapping_mode} mapping linking source records to grid cell identifiers."
                )
            )

        # Asset 3: Final Aggregated Vector Cube (If Generated)
        if aggregated_path and os.path.exists(aggregated_path):
            item.add_asset(
                "aggregated_cube",
                pystac.Asset(
                    href=aggregated_path,
                    media_type="application/x-parquet",
                    roles=["data"],
                    title="Aggregated Vector Cube",
                    description="Final multidimensional grouped cube summarizing observations and spatial weights."
                )
            )

        # Asset 4: The Spatial Dimension Table (Grid Cells)
        item.add_asset(
            "grid_geometries",
            pystac.Asset(
                href=grid_mapping_path,
                media_type="application/x-parquet",
                roles=["metadata", "spatial-index"],
                title=f"{grid_name} Spatial Dimension Table",
                description="Explicit GeoParquet linking discrete grid identifiers to physical grid cell polygons."
            )
        )

        # Asset 5: (Optional) The Zarr Template
        if template_zarr_path and os.path.exists(template_zarr_path):
            item.add_asset(
                "zarr_template",
                pystac.Asset(
                    href=template_zarr_path,
                    media_type="application/x-zarr",
                    roles=["metadata"],
                    title=f"{grid_name} Zarr Blueprint",
                    description="Empty N-dimensional Zarr array defining the master grid topology."
                )
            )

        if logger: logger.info(f"=== STAC Item Assembly Complete: {stac_id} ===")
        return item