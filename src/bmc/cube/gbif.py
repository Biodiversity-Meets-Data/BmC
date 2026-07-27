from bmc.cube.spatiotemporal import *
import yaml
import os
import pandas as pd
from bmc.datasource.gbif import sql
import re
from shapely.geometry import box

class gbif_cube(spatiotemporal_vector_cube):

    def generate_gbif_query_from_recipe(self, recipe: dict, logger: Optional[logging.Logger] = None) -> str:
        """
        Parses a configuration recipe dictionary and generates a validated GBIF SQL query.
        Dynamically translates YAML metrics into GBIF Hive SQL statements.
        """
        from rasterio.warp import transform_bounds

        # 1. Base Configuration Extraction
        spatial_cfg = recipe.get("spatial", {})
        gbif_cfg = recipe.get("sources", {}).get("gbif", {})
        query_filters = gbif_cfg.get("query_filters", {})
        taxonomy_cfg = gbif_cfg.get("taxonomy", {})
        target_grid_name = self.resolve_target_grid(spatial_cfg, logger)

        # 2. Spatial extraction, densification, and WKT conversion
        bbox_cfg = spatial_cfg.get("bbox", {})
        raw_wgs84_bbox = (
            bbox_cfg.get("long_min"), bbox_cfg.get("lat_min"),
            bbox_cfg.get("long_max"), bbox_cfg.get("lat_max")
        )
        
        # Project the raw WGS84 user bounds into the target grid's native CRS and buffer[cite: 1]
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        target_bounds = transform_bounds("EPSG:4326", target_crs, *raw_wgs84_bbox)
        safe_wgs84_bbox = self.build_safe_fetch_envelope(
            target_grid_name=target_grid_name,
            target_bounds=target_bounds,
            source_crs_or_grid="EPSG:4326",
            pixel_buffer=5, 
            logger=logger
        )
        wkt_polygon = sql.bbox2polygon_wkt(safe_wgs84_bbox)

        # 3. Taxonomic & Standard Column Resolution
        target_level = taxonomy_cfg.get("lowest_level")
        columns = sql.resolve_taxonomic_columns(target_level) if target_level else ["speciesKey"]
        
        for col in gbif_cfg.get("columns", ["year", "month"]):
            if col not in columns:
                columns.append(col)

        col_backbone = taxonomy_cfg.get("col_backbone", False)
        col_uuid = taxonomy_cfg.get("col_uuid", "7ddf754f-d193-4cc9-b351-99906754a03b")

        temporal_cfg = recipe.get("temporal", {})
        year_range = [temporal_cfg.get("start_year"), temporal_cfg.get("end_year")]
        month_range = [temporal_cfg.get("start_month"), temporal_cfg.get("end_month")]

        yaml_record_type = query_filters.get("record_type", "presence")
        record_type = "occurrence" if yaml_record_type == "presence" else yaml_record_type

        default_uncert = query_filters.get("default_Uncertainty", 1000)
        max_uncertainty = query_filters.get("max_uncertainty", "auto")

        # 4. Query Mode Routing & Dynamic SQL Metrics
        processing_mode = gbif_cfg.get("processing_mode", "vector").lower()
        aggregate_cfg = gbif_cfg.get("aggregate", {})
        
        sql_group_cols = []
        sql_metric_selects = []
        
        if processing_mode == "api_cube":
            aggregate_flag = True
            # Fetch the raw base string directly from the recipe (e.g., "EEA")
            query_grid = spatial_cfg.get("target_grid")

            # Extract Grouping Columns
            sql_group_cols = aggregate_cfg.get("group_by_columns", [])
            for col in sql_group_cols:
                if col not in columns:
                    columns.append(col)

            # Translate Metrics to SQL
            method_to_sql = {
                "mean": "AVG({col})",
                "max": "MAX({col})",
                "min": "MIN({col})",
                "nunique": "COUNT(DISTINCT {col})",
                "count": "COUNT({col})"
            }
            
            for metric in aggregate_cfg.get("metrics", []):
                col = metric.get("column")
                method = metric.get("method", "count").lower()
                rename = metric.get("rename", f"{col}_{method}")
                
                # Skip local pipeline artifacts
                if col in ["areal_fraction", "fraction"]:
                    continue 
                
                # Catch empty DarwinCore counts and assume presence = 1
                if method == "sum":
                    sql_expr = f"SUM(COALESCE({col}, 1))"
                else:
                    sql_pattern = method_to_sql.get(method)
                    sql_expr = sql_pattern.format(col=col) if sql_pattern else None
                
                if sql_expr:
                    sql_metric_selects.append(f"{sql_expr} AS {rename}")

            # Master Grid Resolution Parsing
            res_str = str(spatial_cfg.get("target_resolution", "")).lower().strip()
            if "km" in res_str: query_res = int(float(res_str.replace("km", "")) * 1000)
            elif "m" in res_str: query_res = int(float(res_str.replace("m", "")))
            elif "sec" in res_str or "min" in res_str:
                query_res = {"0_3sec": 10, "3sec": 100, "7_5sec": 250, "15sec": 500, "30sec": 1000, "5min": 10000}.get(res_str, 1000)
            else:
                try: query_res = int(res_str)
                except ValueError: query_res = 1000
                    
        elif processing_mode in ["raw", "vector"]:
            aggregate_flag = False
            query_grid = False
            query_res = None
            
        else:
            raise ValueError(f"Invalid processing_mode '{processing_mode}'.")

        issue_flags = ["hasCoordinate = TRUE"]
        for issue in query_filters.get("exclude_issues", []):
            issue_flags.append(f"NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, '{issue}', TRUE)")

        raw_taxon_keys = query_filters.get("taxon_keys", [])
        mapped_taxon_keys = sql.map_taxonkeys_to_columns(raw_taxon_keys, col_backbone, col_uuid) if not query_filters.get("fetch_all_taxa", False) and raw_taxon_keys else []

        # 5. Dispatch to SQL generator (Ensure your sql.generate_query accepts the new dynamic SQL lists)
        gbif_sql_query = sql.generate_query(
            taxonKeys=mapped_taxon_keys,
            columns=columns,               
            record_type=record_type,
            wkt_polygon=wkt_polygon,
            year_range=year_range,
            month_range=month_range,
            aggregate=aggregate_flag,
            sql_group_cols=sql_group_cols,       # <-- NEW
            sql_metric_selects=sql_metric_selects, # <-- NEW
            grid=query_grid,                          
            grid_resolution=query_res,     
            coordinateUncertainty=default_uncert, 
            max_uncertainty=max_uncertainty,      
            issue_flags=issue_flags,
            col_backbone=col_backbone,
            col_uuid=col_uuid
        )

        return gbif_sql_query

    def _mine_crs_from_grid(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> str:
        """
        Inspects the DataFrame for a GBIF grid cell code column and mines the native CRS.
        Maps explicit strings (CRS3035, CRS4326) and implicit grid patterns (EEA, EQDG).
        """
        # Find the column dynamically generated by the SQL formatter (e.g., 'eeacellcode', 'dmsgcellcode')
        cell_col = next((col for col in df.columns if 'cellcode' in col.lower()), None)
        
        if not cell_col or df[cell_col].dropna().empty:
            log_execution(logger, "No grid cell codes found. Defaulting to EPSG:4326.", logging.DEBUG)
            return "EPSG:4326"
            
        # Sample the first valid cell code
        sample_code = str(df[cell_col].dropna().iloc[0]).upper()
        log_execution(logger, f"Mining native CRS from grid cell code: {sample_code}", logging.INFO)
        
        # 1. Explicit CRS strings (Eurostat & DMSG Grids)
        if sample_code.startswith("CRS3035"):
            return "EPSG:3035"
        if sample_code.startswith("CRS4326"):
            return "EPSG:4326"
            
        # 2. Implicit EEA Grid (EPSG:3035)
        # Matches formats like '100KME51N29' or '250ME510500N293350'
        if "KME" in sample_code or "ME" in sample_code:
            return "EPSG:3035"
            
        # 3. Implicit Extended Quarter-Degree Grid / EQDG (EPSG:4326)
        # Matches formats like 'W175S21' or 'E010N52BDB'
        if (sample_code.startswith('W') or sample_code.startswith('E')) and ('S' in sample_code or 'N' in sample_code):
            return "EPSG:4326"
            
        # Fallback if ISEA3H or MGRS is used
        log_execution(logger, f"Unrecognized CRS pattern in code {sample_code}. Defaulting to EPSG:4326.", logging.WARNING)
        return "EPSG:4326"

    def _parse_cellcode_to_polygon(self, code: str):
        """
        Dynamically parses GBIF grid cell codes into Shapely Polygons.
        Supports implicit EEA formats and explicit Eurostat/DMSG metric formats.
        Returns None if the code is invalid or unsupported.
        """
        if pd.isna(code):
            return None
            
        code = str(code).upper().strip()
        
        try:
            # 1. EEA Grid (e.g., '100KME51N29' or '250ME510500N293350')
            eea_match = re.match(r'^(\d+)(KM|M)E(\d+)N(\d+)$', code)
            if eea_match:
                res_val, res_unit, easting_code, northing_code = eea_match.groups()
                
                # Convert resolution to meters
                cell_size_m = int(res_val) * 1000 if res_unit == 'KM' else int(res_val)
                
                # Calculate the multiplier based on trailing zeros
                str_size = str(cell_size_m)
                trailing_zeros = len(str_size) - len(str_size.rstrip('0'))
                multiplier = 10 ** trailing_zeros
                
                # Calculate coordinates
                ll_easting = int(easting_code) * multiplier
                ll_northing = int(northing_code) * multiplier
                ur_easting = ll_easting + cell_size_m
                ur_northing = ll_northing + cell_size_m
                
                return box(ll_easting, ll_northing, ur_easting, ur_northing)
                
            # 2. Eurostat / DMSG Metric Grid (e.g., 'CRS3035RES10000MN2480000E4850000')
            euro_match = re.match(r'^CRS(\d+)RES(\d+)MN(\d+)E(\d+)$', code)
            if euro_match:
                epsg, res_m, northing, easting = euro_match.groups()
                
                cell_size_m = int(res_m)
                ll_easting = int(easting)
                ll_northing = int(northing)
                
                return box(ll_easting, ll_northing, ll_easting + cell_size_m, ll_northing + cell_size_m)
                
            # Note: Complex angular grids (like EQDG 'W175S21') are skipped 
            # and will drop out gracefully, as they cannot be perfectly boxed in planar meters.
            return None
            
        except Exception:
            return None

    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key
        that explicitly matches the parent class's GRID_REGISTRY format.
        """
        grid_base = spatial_cfg.get("target_grid")
        resolution = spatial_cfg.get("target_resolution")
        
        if not grid_base:
            log_execution(logger, "No 'target_grid' specified in recipe. Defaulting to Global_WGS84_30sec.", logging.WARNING)
            return "Global_WGS84_30sec"
            
        # 1. Check if the user already provided the full exact key (e.g., "EEA_1km")
        if grid_base in self.GRID_REGISTRY:
            log_execution(logger, f"GBIF strictly adhering to recipe master grid: {grid_base}", logging.INFO)
            return grid_base
            
        # 2. If they provided base and resolution separately, construct the key
        if resolution:
            constructed_key = f"{grid_base}_{resolution}"
            if constructed_key in self.GRID_REGISTRY:
                log_execution(logger, f"Constructed and resolved master grid key: {constructed_key}", logging.INFO)
                return constructed_key
                
        # 3. Fail gracefully if the grid doesn't exist in the registry
        valid_keys = list(self.GRID_REGISTRY.keys())
        raise KeyError(
            f"Could not resolve grid '{grid_base}' with resolution '{resolution}'. "
            f"Please ensure your recipe perfectly matches a key in the GRID_REGISTRY. Valid keys include: {valid_keys[:5]}..."
        )

    def fetch_data(self, recipe: dict, logger: Optional[logging.Logger] = None, downloaded_filepath: str = None, **kwargs) -> gpd.GeoDataFrame:
        """
        Loads raw GBIF data, parses the YAML processing mode, and generates the 
        appropriate base geometric topology (Points, Polygons, or Point Clouds).
        """
        if not downloaded_filepath or not os.path.exists(downloaded_filepath):
            log_execution(logger, "No pre-downloaded file provided. Initiating synchronous download...", logging.WARNING)
            query = self.generate_gbif_query_from_recipe(recipe, logger=logger)
            download_key = sql.submit_gbif_query(query)
            downloaded_filepath = sql.fetch_gbif_download(download_key, target_dir=recipe.get('base_dir', './downloads'))

        log_execution(logger, f"Loading GBIF data from {downloaded_filepath}...", logging.INFO)
        
        # Extract and load the CSV payload
        with zipfile.ZipFile(downloaded_filepath) as z:
            csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_filename) as f:
                # quoting=3 tells pandas to completely ignore quotation marks (csv.QUOTE_NONE)
                # low_memory=False prevents mixed-datatype warnings on massive GBIF files
                df = pd.read_csv(f, sep='\t', quoting=3, low_memory=False)

        # Extract Recipe Configurations
        gbif_cfg = recipe.get("sources", {}).get("gbif", {})
        processing_mode = gbif_cfg.get("processing_mode", "vector").lower()

        # =====================================================================
        # ROUTE A: GBIF API CUBE (Parse pre-aggregated SQL cell strings)
        # =====================================================================
        if processing_mode == "api_cube":
            native_crs = self._mine_crs_from_grid(df, logger)
            log_execution(logger, f"API Cube detected. Mapped to native CRS: {native_crs}", logging.INFO)
            
            cell_col = next((col for col in df.columns if 'cellcode' in col.lower()), None)
            if cell_col and not df[cell_col].dropna().empty:
                log_execution(logger, f"Parsing polygon geometries directly from '{cell_col}'...", logging.INFO)
                polygons = df[cell_col].apply(self._parse_cellcode_to_polygon)
                gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=native_crs)
                
                failed_parse_count = gdf.geometry.isna().sum()
                if failed_parse_count > 0:
                    log_execution(logger, f"Dropped {failed_parse_count} rows with malformed cell codes.", logging.WARNING)
                    gdf = gdf.dropna(subset=['geometry']).copy()
                return gdf
            else:
                raise ValueError("API Cube mode requested, but no 'cellcode' column found in downloaded data.")

        # =====================================================================
        # ROUTE B: RAW / VECTOR PROCESSING (Coordinate to Geometry)
        # =====================================================================
        lat_col = 'latitude' if 'latitude' in df.columns else 'decimallatitude'
        lon_col = 'longitude' if 'longitude' in df.columns else 'decimallongitude'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Required coordinate columns ({lat_col}, {lon_col}) missing from downloaded data.")

        # Identify uncertainty column for buffering/jittering
        uncert_col = 'coordinateuncertaintyinmeters'
        if uncert_col not in df.columns:
            uncert_col = next((c for c in df.columns if 'uncertainty' in c.lower()), None)

        if processing_mode == "raw":
            log_execution(logger, "Raw mode requested. Generating basic Point geometries with no vector transformations.", logging.INFO)
            return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

        # processing_mode == "vector"
        vector_cfg = gbif_cfg.get("vector_processing", {})
        topology = vector_cfg.get("topology", "point")
        topology_config = vector_cfg.get("topology_config", {})
        
        # Build geometries based on recipe topology
        gdf = self.coordinate_to_geometry(
            df=df,
            x_col=lon_col,
            y_col=lat_col,
            uncert_col=uncert_col,
            output_type=topology,
            input_crs="EPSG:4326",
            on_missing_uncertainty="fallback",
            quad_segs=topology_config.get("polygon", {}).get("quad_segs", 8),
            point_cloud_config=topology_config.get("point_cloud", {}),
            logger=logger
        )
        return gdf