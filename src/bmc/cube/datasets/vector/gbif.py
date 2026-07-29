from bmc.cube.spatiotemporal.vector import *
import yaml
import os
import pandas as pd
from bmc.datasource.gbif import sql
import re
from shapely.geometry import box
import zipfile

class gbif_cube(vector_cube):
    """
    Child class dedicated to generating and processing GBIF (Global Biodiversity 
    Information Facility) spatiotemporal vector data cubes.

    This orchestrator inherits from `vector_cube` and specializes in translating 
    abstract YAML recipes into GBIF Hive SQL queries. It handles the ingestion 
    of raw occurrence data, parses implicit and explicit GBIF grid cell codes 
    (EEA, EQDG), normalizes taxonomic representations, and dynamically generates 
    spatial topologies (Points, Polygons, or Point Clouds) based on coordinate 
    uncertainty.

    Methods
    -------
    generate_gbif_query_from_recipe(recipe, logger=None)
        Parses a configuration recipe dictionary and generates a validated GBIF SQL query.
    _mine_crs_from_grid(df, logger=None)
        Inspects the DataFrame for a GBIF grid cell code column and mines the native CRS.
    _parse_cellcode_to_polygon(code)
        Dynamically parses GBIF grid cell codes into Shapely Polygons.
    resolve_target_grid(spatial_cfg, logger=None)
        Translates user-defined spatial configurations into a validated master grid key.
    _validate_local_data(df, gbif_cfg, logger=None)
        Validates that a user-provided local file contains mandatory routing/metric columns.
    fetch_data(recipe, logger=None, downloaded_filepath=None, **kwargs)
        Loads data from file/download, validates inputs, and dispatches topology generation.
    """

    def generate_gbif_query_from_recipe(self, recipe: dict, logger: Optional[logging.Logger] = None) -> str:
        """
        Parses a configuration recipe dictionary and generates a validated GBIF SQL query.
        
        Dynamically translates YAML spatial bounds, taxonomic filters, temporal ranges, 
        and aggregation metrics into executable GBIF Hive SQL statements.

        Parameters
        ----------
        recipe : dict
            The master pipeline configuration dictionary containing spatial, temporal, 
            and data source constraints.
        logger : logging.Logger, optional
            Logger instance for recording execution telemetry.

        Returns
        -------
        str
            A completely formatted and valid GBIF Hive SQL query string ready for submission.
        """
        from rasterio.warp import transform_bounds

        # =====================================================================
        # 1. Base Configuration Extraction
        # =====================================================================
        spatial_cfg = recipe.get("spatial", {})
        gbif_cfg = recipe.get("sources", {}).get("gbif", {})
        query_filters = gbif_cfg.get("query_filters", {})
        taxonomy_cfg = gbif_cfg.get("taxonomy", {})
        target_grid_name = self.resolve_target_grid(spatial_cfg, logger)

        # =====================================================================
        # 2. Spatial Extraction, Densification, and WKT Conversion
        # =====================================================================
        bbox_cfg = spatial_cfg.get("bbox", {})
        raw_wgs84_bbox = (
            bbox_cfg.get("long_min"), bbox_cfg.get("lat_min"),
            bbox_cfg.get("long_max"), bbox_cfg.get("lat_max")
        )
        
        # Project the raw WGS84 user bounds into the target grid's native CRS.
        # A buffer is applied to capture geometries whose centroids fall just outside 
        # the border, preventing edge starvation in the final spatial cube.
        target_crs = self.GRID_REGISTRY[target_grid_name]["crs"]
        target_bounds = transform_bounds("EPSG:4326", target_crs, *raw_wgs84_bbox)
        safe_wgs84_bbox = self.build_safe_fetch_envelope(
            target_grid_name=target_grid_name,
            target_bounds=target_bounds,
            source_crs_or_grid="EPSG:4326",
            pixel_buffer=5, 
            logger=logger
        )
        # Convert the densified mathematical envelope into a WKT Polygon for the SQL query
        wkt_polygon = sql.bbox2polygon_wkt(safe_wgs84_bbox)

        # =====================================================================
        # 3. Taxonomic & Standard Column Resolution
        # =====================================================================
        target_level = taxonomy_cfg.get("lowest_level")
        columns = sql.resolve_taxonomic_columns(target_level) if target_level else ["speciesKey"]
        
        # Ensure temporal grouping columns are requested from the SQL database
        for col in gbif_cfg.get("columns", ["year", "month"]):
            if col not in columns:
                columns.append(col)

        col_backbone = taxonomy_cfg.get("col_backbone", False)
        col_uuid = taxonomy_cfg.get("col_uuid", "7ddf754f-d193-4cc9-b351-99906754a03b")

        temporal_cfg = recipe.get("temporal", {})
        year_range = [temporal_cfg.get("start_year"), temporal_cfg.get("end_year")]
        month_range = [temporal_cfg.get("start_month"), temporal_cfg.get("end_month")]

        # Determine the physical record type mapping (e.g. occurrences vs. events)
        yaml_record_type = query_filters.get("record_type", "presence")
        record_type = "occurrence" if yaml_record_type == "presence" else yaml_record_type

        default_uncert = query_filters.get("default_Uncertainty", 1000)
        max_uncertainty = query_filters.get("max_uncertainty", "auto")

        # =====================================================================
        # 4. Query Mode Routing & Dynamic SQL Metrics
        # =====================================================================
        processing_mode = gbif_cfg.get("processing_mode", "vector").lower()
        aggregate_cfg = gbif_cfg.get("aggregate", {})
        
        sql_group_cols = []
        sql_metric_selects = []
        
        # 'api_cube' offloads spatial aggregation directly to the GBIF SQL servers
        if processing_mode == "api_cube":
            aggregate_flag = True
            query_grid = spatial_cfg.get("target_grid")

            # Extract specific attributes requested for grouping (e.g., year, speciesKey)
            sql_group_cols = aggregate_cfg.get("group_by_columns", [])
            for col in sql_group_cols:
                if col not in columns:
                    columns.append(col)

            # Map YAML metric strings directly to Hive SQL aggregate functions
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
                
                # Ignore local spatial fraction weights as they don't exist in the remote DB
                if col in ["areal_fraction", "fraction"]:
                    continue 
                
                # Safely handle occurrences lacking explicit individualCount fields 
                # by coalescing empty values to 1
                if method == "sum":
                    sql_expr = f"SUM(COALESCE({col}, 1))"
                else:
                    sql_pattern = method_to_sql.get(method)
                    sql_expr = sql_pattern.format(col=col) if sql_pattern else None
                
                if sql_expr:
                    sql_metric_selects.append(f"{sql_expr} AS {rename}")

            # Translate resolution string into numerical units required by GBIF SQL API
            res_str = str(spatial_cfg.get("target_resolution", "")).lower().strip()
            if "km" in res_str: 
                query_res = int(float(res_str.replace("km", "")) * 1000)
            elif "m" in res_str: 
                query_res = int(float(res_str.replace("m", "")))
            elif "sec" in res_str or "min" in res_str:
                query_res = {"0_3sec": 10, "3sec": 100, "7_5sec": 250, "15sec": 500, "30sec": 1000, "5min": 10000}.get(res_str, 1000)
            else:
                try: 
                    query_res = int(res_str)
                except ValueError: 
                    query_res = 1000
                    
        # 'raw' or 'vector' mode downloads row-level data for local memory-intensive processing
        elif processing_mode in ["raw", "vector"]:
            aggregate_flag = False
            query_grid = False
            query_res = None
            
        else:
            raise ValueError(f"Invalid processing_mode '{processing_mode}'.")

        # Exclude flagged biodiversity data issues (e.g., zero coordinates, country coordinate mismatches)
        issue_flags = ["hasCoordinate = TRUE"]
        for issue in query_filters.get("exclude_issues", []):
            issue_flags.append(f"NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, '{issue}', TRUE)")

        # Link requested taxa to the GBIF Backbone if necessary
        raw_taxon_keys = query_filters.get("taxon_keys", [])
        mapped_taxon_keys = sql.map_taxonkeys_to_columns(raw_taxon_keys, col_backbone, col_uuid) if not query_filters.get("fetch_all_taxa", False) and raw_taxon_keys else []

        # =====================================================================
        # 5. Dispatch to SQL generator
        # =====================================================================
        gbif_sql_query = sql.generate_query(
            taxonKeys=mapped_taxon_keys,
            columns=columns,               
            record_type=record_type,
            wkt_polygon=wkt_polygon,
            year_range=year_range,
            month_range=month_range,
            aggregate=aggregate_flag,
            sql_group_cols=sql_group_cols,       
            sql_metric_selects=sql_metric_selects, 
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
        
        Maps explicit strings (e.g., 'CRS3035') and implicit visual patterns 
        (EEA '100KME', EQDG 'W175S') to their standard EPSG identifiers.

        Parameters
        ----------
        df : pandas.DataFrame
            The downloaded tabular data containing aggregated grid codes.
        logger : logging.Logger, optional
            Logger instance for recording execution telemetry.

        Returns
        -------
        str
            The resolved EPSG coordinate reference system string.
        """
        # Scan columns dynamically to locate the SQL formatter output (e.g., 'eeacellcode')
        cell_col = next((col for col in df.columns if 'cellcode' in col.lower()), None)
        
        if not cell_col or df[cell_col].dropna().empty:
            log_execution(logger, "No grid cell codes found. Defaulting to EPSG:4326.", logging.DEBUG)
            return "EPSG:4326"
            
        sample_code = str(df[cell_col].dropna().iloc[0]).upper()
        log_execution(logger, f"Mining native CRS from grid cell code: {sample_code}", logging.INFO)
        
        # 1. Explicit CRS strings (Eurostat & DMSG Grids)
        if sample_code.startswith("CRS3035"):
            return "EPSG:3035"
        if sample_code.startswith("CRS4326"):
            return "EPSG:4326"
            
        # 2. Implicit EEA Grid (EPSG:3035)
        # Identifies format signatures like '100KME51N29' or '250ME510500N293350'
        if "KME" in sample_code or "ME" in sample_code:
            return "EPSG:3035"
            
        # 3. Implicit Extended Quarter-Degree Grid / EQDG (EPSG:4326)
        # Identifies format signatures like 'W175S21' or 'E010N52BDB'
        if (sample_code.startswith('W') or sample_code.startswith('E')) and ('S' in sample_code or 'N' in sample_code):
            return "EPSG:4326"
            
        # Safe fallback for obscure angular grids (ISEA3H, MGRS)
        log_execution(logger, f"Unrecognized CRS pattern in code {sample_code}. Defaulting to EPSG:4326.", logging.WARNING)
        return "EPSG:4326"

    def _parse_cellcode_to_polygon(self, code: str):
        """
        Dynamically parses GBIF grid cell codes into Shapely Polygons.
        
        Supports extracting exact geometrical boundaries from implicit EEA formats 
        and explicit Eurostat/DMSG metric formats. Returns None if the code is 
        invalid or represents a complex angular grid unsuited for planar conversion.

        Parameters
        ----------
        code : str
            The specific grid cell string (e.g., '100KME51N29').

        Returns
        -------
        shapely.geometry.Polygon or None
            A strict bounding polygon constructed from the cell code geometry, 
            or None if the code cannot be geometrically parsed.
        """
        if pd.isna(code):
            return None
            
        code = str(code).upper().strip()
        
        try:
            # 1. Parse European Environment Agency (EEA) Standard Grids
            # Matches formats like '100KME51N29' indicating a 100km resolution cell
            eea_match = re.match(r'^(\d+)(KM|M)E(\d+)N(\d+)$', code)
            if eea_match:
                res_val, res_unit, easting_code, northing_code = eea_match.groups()
                
                # Convert resolution to base metric units (Meters)
                cell_size_m = int(res_val) * 1000 if res_unit == 'KM' else int(res_val)
                
                # Derive multiplier logic to determine physical coordinates:
                # E.g., '100KM' resolution implies coordinates need trailing zeros appended
                str_size = str(cell_size_m)
                trailing_zeros = len(str_size) - len(str_size.rstrip('0'))
                multiplier = 10 ** trailing_zeros
                
                # Project the mathematical boundary corners
                ll_easting = int(easting_code) * multiplier
                ll_northing = int(northing_code) * multiplier
                ur_easting = ll_easting + cell_size_m
                ur_northing = ll_northing + cell_size_m
                
                return box(ll_easting, ll_northing, ur_easting, ur_northing)
                
            # 2. Parse Universal Explicit Eurostat / DMSG Metric Grids
            # Matches formats like 'CRS3035RES10000MN2480000E4850000'
            euro_match = re.match(r'^CRS(\d+)RES(\d+)MN(\d+)E(\d+)$', code)
            if euro_match:
                epsg, res_m, northing, easting = euro_match.groups()
                
                cell_size_m = int(res_m)
                ll_easting = int(easting)
                ll_northing = int(northing)
                
                return box(ll_easting, ll_northing, ll_easting + cell_size_m, ll_northing + cell_size_m)
                
            # Complex angular grids (e.g. EQDG 'W175S21') are inherently non-square 
            # in planar projection. They are skipped and handled cleanly upstream.
            return None
            
        except Exception:
            return None

    def resolve_target_grid(self, spatial_cfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> str:
        """
        Translates user-defined spatial configurations into a validated master grid key.
        
        Ensures the requested grid matches a hardcoded configuration in the engine's 
        parent `GRID_REGISTRY` to guarantee perfect pixel alignment later in the pipeline.

        Parameters
        ----------
        spatial_cfg : dict
            The 'spatial' block dictionary loaded from the YAML recipe.
        logger : logging.Logger, optional
            Execution logger tracing validation outcomes.

        Returns
        -------
        str
            The precise dictionary key (e.g., 'EEA_10km') mapped to physical parameters.
        
        Raises
        ------
        KeyError
            If the requested grid configuration is unknown to the parent grid registry.
        """
        grid_base = spatial_cfg.get("target_grid")
        resolution = spatial_cfg.get("target_resolution")
        
        if not grid_base:
            log_execution(logger, "No 'target_grid' specified in recipe. Defaulting to Global_WGS84_30sec.", logging.WARNING)
            return "Global_WGS84_30sec"
            
        # 1. Exact Match Check (User provided "EEA_1km" directly)
        if grid_base in self.GRID_REGISTRY:
            log_execution(logger, f"GBIF strictly adhering to recipe master grid: {grid_base}", logging.INFO)
            return grid_base
            
        # 2. Assembly Check (User provided base and resolution separately)
        if resolution:
            constructed_key = f"{grid_base}_{resolution}"
            if constructed_key in self.GRID_REGISTRY:
                log_execution(logger, f"Constructed and resolved master grid key: {constructed_key}", logging.INFO)
                return constructed_key
                
        # 3. Validation failure
        valid_keys = list(self.GRID_REGISTRY.keys())
        raise KeyError(
            f"Could not resolve grid '{grid_base}' with resolution '{resolution}'. "
            f"Please ensure your recipe perfectly matches a key in the GRID_REGISTRY. Valid keys include: {valid_keys[:5]}..."
        )

    def _validate_local_data(self, df: pd.DataFrame, gbif_cfg: dict, logger: Optional[logging.Logger] = None) -> None:
        """
        Validates that a user-provided local file contains mandatory routing/metric columns.
        
        Uses case-insensitive matching to verify required attributes (like coordinates 
        and metric groupings) are present. Standardizes column names in-place to perfectly 
        mirror the YAML recipe to prevent downstream KeyErrors.

        Parameters
        ----------
        df : pandas.DataFrame
            The tabular data provided locally by the user.
        gbif_cfg : dict
            The GBIF-specific source configuration dictionary block.
        logger : logging.Logger, optional
            Pipeline telemetry tracker.

        Raises
        ------
        ValueError
            If critical geometric or grouping columns are completely missing.
        """
        processing_mode = gbif_cfg.get("processing_mode", "vector").lower()
        
        # Create a lowercase mapping to evaluate existence case-insensitively
        col_map = {str(c).lower(): c for c in df.columns}
        rename_dict = {}

        # =====================================================================
        # 1. Spatial/Geometry Pre-condition Checks
        # =====================================================================
        if processing_mode == "api_cube":
            # API processing strictly requires pre-aggregated grid codes
            cell_col = next((col for col in col_map.keys() if 'cellcode' in col), None)
            if not cell_col:
                raise ValueError("API Cube mode requested, but no 'cellcode' column found in the local data.")
        elif processing_mode in ["vector", "raw"]:
            # Vector mode requires precise latitude/longitude to generate planar topologies
            has_lat = 'latitude' in col_map or 'decimallatitude' in col_map
            has_lon = 'longitude' in col_map or 'decimallongitude' in col_map
            
            if not has_lat or not has_lon:
                raise ValueError("Required coordinate columns (latitude/decimallatitude, longitude/decimallongitude) missing from local data.")
            
            # Warn the user if they requested complex Monte Carlo/Buffer geometry 
            # without supplying an uncertainty radius column
            if processing_mode == "vector":
                topology = gbif_cfg.get("vector_processing", {}).get("topology", "point")
                if topology in ["polygon", "point_cloud"]:
                    if 'coordinateuncertaintyinmeters' not in col_map and not any('uncertainty' in c for c in col_map):
                        log_execution(logger, f"Topology '{topology}' usually requires an uncertainty column. Engine will fall back to default recipe values.", logging.WARNING)

        # =====================================================================
        # 2. Aggregation Schema Checks & In-Place Schema Normalization
        # =====================================================================
        aggregate_cfg = gbif_cfg.get("aggregate", {})
        if aggregate_cfg:
            required_cols_recipe = aggregate_cfg.get("group_by_columns", [])[:]
            for metric in aggregate_cfg.get("metrics", []):
                metric_col = metric.get("column")
                # Do not check for internal mathematical artifacts generated during runtime
                if metric_col not in ["fraction", "areal_fraction"]:
                    required_cols_recipe.append(metric_col)
            
            missing_cols = []
            for req_col in required_cols_recipe:
                req_lower = str(req_col).lower()
                if req_lower not in col_map:
                    missing_cols.append(req_col)
                else:
                    # If column exists but case mismatches, queue it for renaming 
                    # so the Pandas groupby engine doesn't crash
                    actual_df_col = col_map[req_lower]
                    if actual_df_col != req_col:
                        rename_dict[actual_df_col] = req_col

            if missing_cols:
                raise ValueError(f"The following required aggregation columns are missing from the local data (checked case-insensitively): {missing_cols}")

            if rename_dict:
                log_execution(logger, f"Aligning local column casing to match recipe: {rename_dict}", logging.INFO)
                df.rename(columns=rename_dict, inplace=True)

    def fetch_data(self, recipe: dict, logger: Optional[logging.Logger] = None, downloaded_filepath: str = None, **kwargs) -> gpd.GeoDataFrame:
        """
        Loads data from file/download, validates inputs, and dispatches topology generation.
        
        Acts as the primary I/O gateway for vector ingestion. It handles synchronous 
        downloads if local files are missing, parses various file types (Zip, CSV, Parquet), 
        validates schemas, and then converts the tabular coordinates into explicit 
        geometric representations (GeoDataFrames).

        Parameters
        ----------
        recipe : dict
            The full pipeline YAML execution recipe.
        logger : logging.Logger, optional
            System output streaming component tracking execution.
        downloaded_filepath : str, optional
            An explicit path to a previously downloaded or local dataset.
        **kwargs : dict
            Additional arguments absorbed safely from the orchestrator.

        Returns
        -------
        geopandas.GeoDataFrame
            The spatialized tabular dataframe containing mapped geometry objects.
            
        Raises
        ------
        ValueError
            If an unsupported file type is provided or critical coordinates are absent.
        """
        gbif_cfg = recipe.get("sources", {}).get("gbif", {})
        processing_mode = gbif_cfg.get("processing_mode", "vector").lower()
        is_local_user_file = "local_file_path" in gbif_cfg
        
        # =====================================================================
        # 1. HANDLE MISSING DOWNLOADS & SYNCHRONOUS FALLBACK
        # =====================================================================
        # If the asynchronous orchestrator failed to pre-download the data, execute a 
        # synchronous fallback query to ensure the pipeline proceeds.
        if not downloaded_filepath or not os.path.exists(downloaded_filepath):
            log_execution(logger, "No pre-downloaded file provided. Initiating synchronous download...", logging.WARNING)
            query = self.generate_gbif_query_from_recipe(recipe, logger=logger)
            
            raw_key_response = sql.submit_gbif_query(query)
            download_key = raw_key_response.get("key") if isinstance(raw_key_response, dict) else raw_key_response
            
            download_info = sql.fetch_gbif_download(download_key, target_dir=recipe.get('base_dir', './downloads'))
            downloaded_filepath = download_info.get("path") if isinstance(download_info, dict) else download_info

        log_execution(logger, f"Loading GBIF data from {downloaded_filepath}...", logging.INFO)
        
        # =====================================================================
        # 2. EXTRACT AND LOAD PAYLOAD I/O
        # =====================================================================
        # Parse Zipped GBIF archives dynamically to extract the underlying CSV
        if downloaded_filepath.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(downloaded_filepath) as z:
                csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f, sep='\t', quoting=3, low_memory=False)
                    
        elif downloaded_filepath.endswith('.csv'):
            # Detect separator (tab vs comma) dynamically based on GBIF header defaults
            with open(downloaded_filepath, 'r') as f:
                first_line = f.readline()
                sep = '\t' if '\t' in first_line else ','
            df = pd.read_csv(downloaded_filepath, sep=sep, quoting=3, low_memory=False)
            
        elif downloaded_filepath.endswith('.parquet'):
            df = pd.read_parquet(downloaded_filepath)
            
        else:
            raise ValueError(f"Unsupported file format provided: {downloaded_filepath}. Expected .zip, .csv, or .parquet.")

        # =====================================================================
        # 3. VALIDATE USER-PROVIDED LOCAL DATASETS
        # =====================================================================
        if is_local_user_file:
            self._validate_local_data(df, gbif_cfg, logger)

        # =====================================================================
        # ROUTE A: GBIF API CUBE (Parse pre-aggregated SQL cell strings)
        # =====================================================================
        if processing_mode == "api_cube":
            native_crs = self._mine_crs_from_grid(df, logger)
            log_execution(logger, f"API Cube detected. Mapped to native CRS: {native_crs}", logging.INFO)
            
            # Locate cell codes and generate geometric polygon footprints dynamically
            cell_col = next((col for col in df.columns if 'cellcode' in col.lower()), None)
            if cell_col and not df[cell_col].dropna().empty:
                log_execution(logger, f"Parsing polygon geometries directly from '{cell_col}'...", logging.INFO)
                polygons = df[cell_col].apply(self._parse_cellcode_to_polygon)
                gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=native_crs)
                
                # Prune rows where string cell codes failed geometric casting
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
        # Normalize coordinate column names defensively via lowercasing
        df_cols_lower = {str(c).lower(): c for c in df.columns}
        
        lat_col = df_cols_lower.get('latitude') or df_cols_lower.get('decimallatitude')
        lon_col = df_cols_lower.get('longitude') or df_cols_lower.get('decimallongitude')
        
        if not lat_col or not lon_col:
            raise ValueError("Required coordinate columns missing from downloaded data.")

        # Retrieve coordinate uncertainty radius for advanced metric buffering
        uncert_col = df_cols_lower.get('coordinateuncertaintyinmeters')
        if not uncert_col:
            uncert_col = next((df_cols_lower[c] for c in df_cols_lower if 'uncertainty' in c), None)

        if processing_mode == "raw":
            log_execution(logger, "Raw mode requested. Generating basic Point geometries with no vector transformations.", logging.INFO)
            return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

        # Fallthrough to processing_mode == "vector"
        vector_cfg = gbif_cfg.get("vector_processing", {})
        topology = vector_cfg.get("topology", "point")
        topology_config = vector_cfg.get("topology_config", {})
        
        # Dispatch geometric processing rules based on the user's topology request 
        # (e.g., executing point cloud arrays or single metric polygons)
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