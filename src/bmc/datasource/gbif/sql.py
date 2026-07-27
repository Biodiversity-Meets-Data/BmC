import shapely.geometry
import pygbif
from concurrent.futures import ThreadPoolExecutor
from bmc.datasource.gbif import interface
from bmc.utils import credentials
from functools import partial
import time
import os

def resolve_taxonomic_columns(target_level: str) -> list[str]:
    """Map a starting taxonomic rank level down to the terminal species rank.

    This function identifies the index of a specified target taxonomic rank within
    the standard Linnaean hierarchy and generates a downstream slice of corresponding
    GBIF SQL schema identifier key columns.

    Parameters
    ----------
    target_level : str
        The biological rank level to start extracting keys from. Must be one of:
        'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'.

    Returns
    -------
    list of str
        A list of GBIF SQL integer identifier column names spanning from the target
        level down to 'speciesKey'.

    Raises
    ------
    ValueError
        If the `target_level` string does not match any recognized rank in the
        canonical Linnaean hierarchy.

    Examples
    --------
    >>> resolve_taxonomic_columns("phylum")
    ['phylumKey', 'classKey', 'orderKey', 'familyKey', 'genusKey', 'speciesKey']

    >>> resolve_taxonomic_columns("GENUS  ")
    ['genusKey', 'speciesKey']

    >>> resolve_taxonomic_columns("")
    ['speciesKey']
    """
    rank_hierarchy = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    
    if not target_level:
        return ["speciesKey"]
        
    target_level = target_level.lower().strip()
    if target_level not in rank_hierarchy:
        raise ValueError(f"Unknown taxonomic level target: '{target_level}'")
        
    start_idx = rank_hierarchy.index(target_level)
    selected_ranks = rank_hierarchy[start_idx:]
    selected_keys = [f"{rank}Key" for rank in selected_ranks]
            
    return selected_keys

def map_taxonkeys_to_columns(
    taxon_keys: list[int | str], 
    max_workers: int = 8,
    col_backbone: bool = False,
    col_uuid: str = '7ddf754f-d193-4cc9-b351-99906754a03b'
) -> dict[str, list[int | str]]:
    """Group raw taxon keys into their optimal indexed GBIF SQL column positions.

    Takes an arbitrary array of identifier keys (GBIF integers or CoL strings), 
    fetches their structural ranks via parallel API requests, and buckets them into 
    the exact indexed column mappings required for high-speed SQL filtering. Non-major 
    intermediate ranks automatically route to the generic 'taxonKey'.

    Parameters
    ----------
    taxon_keys : list of int or str
        A collection of raw, unverified taxonomic sequence keys (integers for GBIF, 
        alphanumeric strings for CoL).
    max_workers : int, default 8
        The total allocation of worker threads reserved for execution inside the
        ThreadPoolExecutor.
    col_backbone : bool, default False
        If True, treats the keys as Catalogue of Life identifiers and passes the 
        toggle down to the underlying API lookup helper.
    col_uuid : str, default '7ddf754f-d193-4cc9-b351-99906754a03b'
        The target checklist dataset UUID parameter passed to the lookup helper.

    Returns
    -------
    dict of (str, list of int or str)
        A mapping lookup where keys are valid GBIF SQL column names (e.g., 'classKey')
        and values are lists of identifiers belonging to that specific rank.

    Examples
    --------
    >>> keys = [2, 216, 1043084]
    >>> map_taxonkeys_to_columns(keys, max_workers=4, col_backbone=False)
    {'kingdomKey': [2], 'classKey': [216], 'speciesKey': [1043084]}
    """
    if not taxon_keys:
        return {}
        
    # Safely handle both ints and strings, dropping None or empty values.
    # Crucial: The aggressive int(k) cast is removed to preserve CoL strings!
    unique_keys = list(set(k for k in taxon_keys if k is not None and str(k).strip() != ""))
    
    # Freeze our backbone configuration arguments into the single-argument callable 
    # expected by executor.map
    lookup_func = partial(interface.lookup_backbone_single, col_backbone=col_backbone, col_uuid=col_uuid)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Note: calling our optimized, flat-payload lookup_backbone_single function
        records = list(executor.map(lookup_func, unique_keys))
        
    # Exact indexed target column mappings for GBIF SQL downloads
    rank_to_column = {
        "KINGDOM": "kingdomKey",
        "PHYLUM": "phylumKey",
        "CLASS": "classKey",
        "ORDER": "orderKey",
        "FAMILY": "familyKey",
        "GENUS": "genusKey",
        "SUBGENUS": "subgenusKey",
        "SPECIES": "speciesKey"
    }
    
    column_mapping = {}
    
    # Zip up the unique keys with their flattened lookup metadata dicts
    for key, record in zip(unique_keys, records):
        if record and "rank" in record and record.get("matchType") != "NONE":
            rank = str(record["rank"]).upper()
            column_name = rank_to_column.get(rank, "taxonKey")
        else:
            # Route unmatchable keys or intermediate ranks cleanly to the generic catchment
            column_name = "taxonKey"
            
        if column_name not in column_mapping:
            column_mapping[column_name] = []
        column_mapping[column_name].append(key)
        
    return column_mapping

def bbox2polygon_wkt(bbox: tuple[float, float, float, float]) -> str:
    """Convert a bounding box coordinate tuple into an OGC standard WKT Polygon string.

    Parameters
    ----------
    bbox : tuple of float
        A 4-element numeric tuple indicating geographic bounds specified in the strict order:
        (min_x, min_y, max_x, max_y) or equivalently (min_lon, min_lat, max_lon, max_lat).

    Returns
    -------
    str
        The string representation conforming to Well-Known Text (WKT) geometry
        standards (e.g., 'POLYGON ((...))').

    Examples
    --------
    Convert a bounding box over Aarschot, Belgium into WKT format:
    
    >>> bounding_box = (4.8, 50.9, 4.9, 51.0)
    >>> bbox2polygon_wkt(bounding_box)
    'POLYGON ((4.9 50.9, 4.9 51, 4.8 51, 4.8 50.9, 4.9 50.9))'
    """
    polygon = shapely.geometry.box(*bbox)
    return polygon.wkt

GBIF_GRIDS = ["EEA", "EUROSTAT", "EQDG", "DMSG", "ISEA3H", "MGRS"]

GBIF_GRID_FUNCTIONS = [
    "GBIF_EEARGCode", 
    "GBIF_EuroStatCode", 
    "GBIF_EQDGCode", 
    "GBIF_DMSGCode",     
    "GBIF_ISEA3HCode",   
    "GBIF_MGRSCode"
]

# Updated based on GBIF documentation specifications
GBIF_GRID_RESOLUTIONS = [
    [25, 100, 250, 1000, 2000, 5000, 10000, 50000, 100000],  # EEA
    [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],    # EUROSTAT
    list(range(0, 8)),                                       # EQDG (Levels 0-7)
    [i for i in range(1, 3601) if 3600 % i == 0],            # DMSG (All divisors of 3600)
    list(range(1, 23)),                                      # ISEA3H
    [0, 1, 10, 100, 1000, 10000, 100000]                     # MGRS
]


GBIF_COLUMNS = [
    "gbifid", "accessrights", "bibliographiccitation", "language", "license",
    "modified", "publisher", "references", "rightsholder", "type",
    "institutionid", "collectionid", "datasetid", "institutioncode", "collectioncode",
    "datasetname", "ownerinstitutioncode", "basisofrecord", "informationwithheld",
    "datageneralizations", "dynamicproperties", "occurrenceid", "catalognumber",
    "recordnumber", "recordedby", "recordedbyid", "individualcount",
    "organismquantity", "organismquantitytype", "sex", "lifestage",
    "reproductivecondition", "caste", "behavior", "vitality",
    "establishmentmeans", "degreeofestablishment", "pathway",
    "georeferenceverificationstatus", "occurrencestatus", "preparations",
    "disposition", "associatedoccurrences", "associatedreferences",
    "associatedsequences", "associatedtaxa", "othercatalognumbers",
    "occurrenceremarks", "organismid", "organismname", "organismscope",
    "associatedorganisms", "previousidentifications", "organismremarks",
    "materialentityid", "materialentityremarks", "verbatimlabel",
    "materialsampleid", "eventid", "parenteventid", "eventtype",
    "fieldnumber", "eventdate", "eventtime", "startdayofyear", "enddayofyear",
    "year", "month", "day", "verbatimeventdate", "habitat", "samplingprotocol",
    "samplesizevalue", "samplesizeunit", "samplingeffort", "fieldnotes",
    "eventremarks", "locationid", "highergeographyid", "highergeography",
    "continent", "waterbody", "islandgroup", "island", "countrycode",
    "stateprovince", "county", "municipality", "locality", "verbatimlocality",
    "verbatimelevation", "verticaldatum", "verbatimdepth",
    "minimumdistanceabovesurfaceinmeters", "maximumdistanceabovesurfaceinmeters",
    "locationaccordingto", "locationremarks", "decimallatitude",
    "decimallongitude", "coordinateuncertaintyinmeters", "coordinateprecision",
    "pointradiusspatialfit", "verbatimcoordinatesystem", "verbatimsrs",
    "footprintwkt", "footprintsrs", "footprintspatialfit", "georeferencedby",
    "georeferenceddate", "georeferenceprotocol", "georeferencesources",
    "georeferenceremarks", "geologicalcontextid", "earliesteonorlowesteonothem",
    "latesteonorhighesteonothem", "earliesteraorlowesterathem",
    "latesteraorhighesterathem", "earliestperiodorlowestsystem",
    "latestperiodorhighestsystem", "earliestepochorlowestseries",
    "latestepochorhighestseries", "earliestageorloweststage",
    "latestageorhigheststage", "lowestbiostratigraphiczone",
    "highestbiostratigraphiczone", "lithostratigraphicterms", "group",
    "formation", "member", "bed", "identificationid", "verbatimidentification",
    "identificationqualifier", "typestatus", "identifiedby", "identifiedbyid",
    "dateidentified", "identificationreferences", "identificationverificationstatus",
    "identificationremarks", "taxonid", "scientificnameid", "acceptednameusageid",
    "parentnameusageid", "originalnameusageid", "nameaccordingtoid",
    "namepublishedinid", "taxonconceptid", "scientificname",
    "acceptednameusage", "parentnameusage", "originalnameusage",
    "nameaccordingto", "namepublishedin", "namepublishedinyear",
    "higherclassification", "kingdom", "phylum", "class", "order", "superfamily",
    "family", "subfamily", "tribe", "subtribe", "genus", "genericname",
    "subgenus", "infragenericepithet", "specificepithet",
    "infraspecificepithet", "cultivarepithet", "taxonrank",
    "verbatimtaxonrank", "vernacularname", "nomenclaturalcode",
    "taxonomicstatus", "nomenclaturalstatus", "taxonremarks", "datasetKey",
    "publishingcountry", "lastinterpreted", "elevation", "elevationaccuracy",
    "depth", "depthaccuracy", "distancefromcentroidinmeters", "issue",
    "taxonomicissue", "nontaxonomicissue", "mediatype", "hascoordinate",
    "hasgeospatialissues", "taxonKey", "acceptedtaxonKey", "kingdomKey",
    "phylumKey", "classKey", "orderKey", "familyKey", "genusKey",
    "subgenusKey", "speciesKey", "species", "acceptedscientificname",
    "typifiedname", "protocol", "lastparsed", "lastcrawled", "isinvasive",
    "repatriated", "relativeorganismquantity", "projectid", "issequenced",
    "gbifregion", "publishedbygbifregion", "level0gid", "level0name",
    "level1gid", "level1name", "level2gid", "level2name", "level3gid",
    "level3name", "iucnredlistcategory", "publishingorgKey", "installationKey",
    "institutionKey", "collectionKey", "programmeacronym",
    "hostingorganizationKey", "isincluster", "dwcaextension", "eventdategte",
    "eventdatelte"
]

def estimate_grid_size_meters(grid: str, resolution: int) -> float:
    """
    Approximates the physical size of a GBIF grid cell in meters.
    
    Args:
        grid (str): The GBIF grid system (e.g., 'EEA', 'DMSG', 'EQDG')
        resolution (int): The resolution parameter passed to the grid
        
    Returns:
        float: The approximate physical height of the cell in meters.
    """
    grid = grid.upper()
    
    # 1. Native Metric Grids (1:1 mapping)
    if grid in ["EEA", "EUROSTAT"]:
        return float(resolution)
        
    # 2. Degree-Minute-Second Grid (Resolution is in arc-seconds)
    elif grid == "DMSG":
        # 1 degree = 111,320 meters. 1 degree = 3600 arc-seconds.
        meters_per_second = 111320.0 / 3600.0
        return float(resolution * meters_per_second)
        
    # 3. Extended Quarter-Degree Grid (Resolution is exponential levels)
    elif grid == "EQDG":
        # Level 0 = 1 degree. Level 1 = 0.5 deg. Level 2 = 0.25 deg...
        degree_span = 1.0 / (2 ** resolution)
        return float(degree_span * 111320.0)
        
    else:
        raise NotImplementedError(f"Automated metric mapping for {grid} is not supported.")
    
def format_column_selects(columns: list[str], 
                          col_backbone: bool = False, 
                          col_uuid: str = '7ddf754f-d193-4cc9-b351-99906754a03b') -> list[str]:
    """
    Format column names for GBIF queries, mapping backbone JSON to standard column names.
    """
    reserved_columns = {"group", "order", "type", "references", "class", "language", "year", "month", "day"}
    
    taxonomic_terms = {
        "kingdom", "kingdomkey", "phylum", "phylumkey", "class", "classkey",
        "order", "orderkey", "family", "familykey", "genus", "genuskey",
        "species", "specieskey", "scientificname", "taxonkey", "acceptedtaxonkey"
    }

    formatted_columns = []
    
    for col in columns:
        col_lower = col.lower()
        
        # Determine the SQL-safe name up front (e.g., "order" vs genus)
        safe_name = f'"{col_lower}"' if col_lower in reserved_columns else col_lower
        
        if col_backbone and col_lower in taxonomic_terms:
            # Output: classificationdetails['...']['genus'] AS genus
            # Or:     classificationdetails['...']['order'] AS "order"
            formatted_columns.append(f"classificationdetails['{col_uuid}']['{col_lower}'] AS {safe_name}")
        else:
            # Output standard column with quotes if necessary
            formatted_columns.append(safe_name)
                
    return formatted_columns

def format_taxonomic_filters(taxon_keys: list[int | str] | dict[str, list[int | str]], 
                             col_backbone: bool = False,
                             col_uuid: str = '7ddf754f-d193-4cc9-b351-99906754a03b') -> str:
    """
    Format taxonomic WHERE clauses, supporting standard integer keys or CoL backbone string keys.
    Accepts a flat list (defaults to 'taxonkey') or a dictionary mapping specific ranks to keys.
    """
    if not taxon_keys:
        return ""
    
    # Normalize input: if it's a flat list, assume it belongs to the base 'taxonkey'
    if isinstance(taxon_keys, list):
        filters = {'taxonkey': taxon_keys}
    elif isinstance(taxon_keys, dict):
        filters = taxon_keys
    else:
        raise ValueError("taxon_keys must be a list or a dictionary.")

    sub_clauses = []
    
    for col_name, keys in filters.items():
        if not keys:
            continue
            
        col_lower = col_name.lower()
        
        if col_backbone:
            # CoL backbone: requires JSON map syntax and single-quoted string keys
            formatted_keys = [f"'{str(k)}'" for k in keys]
            col_expr = f"classificationdetails['{col_uuid}']['{col_lower}']"
        else:
            # Standard GBIF syntax
            # Safely quote strings if alphanumeric CoL keys are passed without the backbone flag,
            # otherwise just stringify the integers.
            formatted_keys = [f"'{k}'" if isinstance(k, str) and not str(k).isdigit() else str(k) for k in keys]
            col_expr = col_name

        sub_clauses.append(f"{col_expr} IN ({', '.join(formatted_keys)})")

    # Guard against empty dictionaries or empty lists inside dictionaries
    if not sub_clauses:
        return ""

    # Return single clause without wrapping parentheses, or join multiples with OR
    if len(sub_clauses) == 1:
        return sub_clauses[0]
        
    return f"({' OR '.join(sub_clauses)})"

def generate_query(taxonKeys: list[int | str] | dict[str, list[int | str]], 
                   columns: list[str], 
                   record_type: str, 
                   wkt_polygon: str,
                   year_range: int | list[int] | None = None,
                   month_range: int | list[int] | None = None,
                   aggregate: bool = False, 
                   sql_group_cols: list[str] | None = None,       
                   sql_metric_selects: list[str] | None = None, 
                   grid: str | bool = False, 
                   grid_resolution: int | None = None, 
                   coordinateUncertainty: int = 1000,
                   max_uncertainty: int | float | str | None = None,
                   includeUnknownStatus: bool = True,
                   include_uncertainty: bool = True,
                   issue_flags: list[str] | None = None,
                   col_backbone: bool = False,
                   col_uuid: str = '7ddf754f-d193-4cc9-b351-99906754a03b') -> str:
    """Generate a syntactically validated SQL string optimized for the GBIF query engine.

    Assembles SELECT, WHERE, and spatial or grid-based GROUP BY clauses into a 
    pretty-printed, readable SQL string. Includes dynamic bounding box optimization, 
    fallback coordinate uncertainty handling, and spatial precision filtering.
    """
    
    # ---- INJECT GBIF ID ----
    if not grid and not aggregate:
        if "gbifid" not in [col.lower() for col in columns]:
            columns = ["gbifid"] + columns

    if issue_flags is None:
        issue_flags = [
            "hasCoordinate = TRUE", 
            "NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, 'ZERO_COORDINATE', TRUE)",
            "NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, 'COORDINATE_OUT_OF_RANGE', TRUE)",
            "NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, 'COORDINATE_INVALID', TRUE)",
            "NOT GBIF_STRINGARRAYCONTAINS(occurrence.issue, 'COUNTRY_COORDINATE_MISMATCH', TRUE)"
        ]

    # ---- VALIDITY CHECK ----
    if not set(columns).issubset(GBIF_COLUMNS):
        invalid_cols = set(columns) - set(GBIF_COLUMNS)
        raise ValueError(f"The following column(s) ({invalid_cols}) are not present in the GBIF data table") 
        
    if record_type.lower() not in ["occurrence", "absence", "mixed"]:
        raise ValueError(f"Chosen record type '{record_type}' is invalid.")

    # ---- APPLY COLUMN FORMATTER ----
    quoted_columns = format_column_selects(columns, col_backbone=col_backbone, col_uuid=col_uuid)

    # ---- WHERE CONDITIONS ARRAY INITIALIZATION ----
    where_conditions = []

    # 1. Bounding Box Pre-Filter
    try:
        geom = shapely.wkt.loads(wkt_polygon)
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        where_conditions.append(f"decimalLatitude >= {min_lat} AND decimalLatitude <= {max_lat}")
        where_conditions.append(f"decimalLongitude >= {min_lon} AND decimalLongitude <= {max_lon}")
    except Exception as e:
        raise ValueError(f"Failed to parse bounding box from wkt_polygon: {str(e)}")

    # 2. Complex Polygon Filter
    where_conditions.append(f"GBIF_Within('{wkt_polygon}', decimalLatitude, decimalLongitude) = TRUE")

    # 3. Maximum Uncertainty Filter
    if max_uncertainty is not None:
        if str(max_uncertainty).lower() == "auto":
            if not grid or grid_resolution is None:
                raise ValueError("Cannot set max_uncertainty='auto' without specifying a grid and grid_resolution.")
            calculated_threshold = int(estimate_grid_size_meters(grid, grid_resolution))
            where_conditions.append(f"COALESCE(coordinateUncertaintyInMeters, {coordinateUncertainty}) <= {calculated_threshold}")
        else:
            where_conditions.append(f"COALESCE(coordinateUncertaintyInMeters, {coordinateUncertainty}) <= {max_uncertainty}")

    # 4. Temporal Filters
    time_columns_filter = " AND ".join([f'"{col}" IS NOT NULL' for col in columns if col in ["year", "month", "day"]])
    if time_columns_filter:
        where_conditions.append(time_columns_filter)
    
    if year_range is not None:
        if isinstance(year_range, (list, tuple)):
            if len(year_range) != 2: raise ValueError("Year range list must contain exactly two integers: [start_year, end_year].")
            if year_range[0] > year_range[1]: raise ValueError("Year range interval is invalid (start_year > end_year).")
            where_conditions.append(f'"year" >= {year_range[0]} AND "year" <= {year_range[1]}')
        else:
            where_conditions.append(f'"year" = {year_range}')

    if month_range is not None:
        if isinstance(month_range, (list, tuple)):
            if len(month_range) != 2: raise ValueError("Month range list must contain exactly two integers: [start_month, end_month].")
            if month_range[0] > month_range[1]: raise ValueError("Month range interval is invalid (start_month > end_month).")
            if not (1 <= month_range[0] <= 12) or not (1 <= month_range[1] <= 12): raise ValueError("Months must be integers between 1 and 12.")
            where_conditions.append(f'"month" >= {month_range[0]} AND "month" <= {month_range[1]}')
        else:
            if not (1 <= month_range <= 12): raise ValueError("Month must be an integer between 1 and 12.")
            where_conditions.append(f'"month" = {month_range}')
        
    # 5. Issue Flags
    if issue_flags:
        where_conditions.extend(issue_flags)

    # ---- OCCURRENCE STATUS ----
    status_map = {"occurrence": ["'PRESENT'"], "absence": ["'ABSENT'"], "mixed": ["'PRESENT'", "'ABSENT'"]}
    status_values = status_map[record_type.lower()].copy()
    if includeUnknownStatus:
        status_values.append("'UNKNOWN'")
    where_conditions.append(f"occurrenceStatus IN ({', '.join(status_values)})")
    status_str_map = {"occurrence": "occurrences", "absence": "absences", "mixed": "frequency"}

    # ---- OPTIMIZED TAXONOMIC FILTER GENERATION ----
    if taxonKeys:
        taxa_filter = format_taxonomic_filters(taxonKeys, col_backbone=col_backbone, col_uuid=col_uuid)
        if taxa_filter:
            where_conditions.append(taxa_filter)

    # ---- GRID SECTION ----
    gridding_str = ""
    if grid:
        if grid.upper() not in GBIF_GRIDS:
            raise ValueError(f"The specified grid '{grid}' is not a supported grid. Options: {GBIF_GRIDS}")
            
        grid_idx = GBIF_GRIDS.index(grid.upper())
        if grid_resolution not in GBIF_GRID_RESOLUTIONS[grid_idx]:
            raise ValueError(f"The specified resolution '{grid_resolution}' is not valid for {grid}.")
            
        selected_function = GBIF_GRID_FUNCTIONS[grid_idx]
        gridding_str = f"{selected_function}({grid_resolution}, decimalLatitude, decimalLongitude, COALESCE(coordinateUncertaintyInMeters, {coordinateUncertainty})) AS {grid.lower()}cellcode"

    # ---- SQL QUERY BUILD (NESTED FORMATTING) ----
    
    # Base columns (only those required for grouping if aggregate is True)
    cols_to_select = sql_group_cols if (aggregate and sql_group_cols) else columns
    quoted_selects = format_column_selects(cols_to_select, col_backbone=col_backbone, col_uuid=col_uuid)
    select_lines = [f"    {col}" for col in quoted_selects]
    
    if not aggregate:
        select_lines.extend(["    decimalLatitude", "    decimalLongitude"])
        
    if include_uncertainty and not aggregate:
        select_lines.append(f"    COALESCE(coordinateUncertaintyInMeters, {coordinateUncertainty}) AS coordinateUncertaintyInMeters")
        
    if aggregate:
        # Dynamically inject the SQL metrics derived from the YAML recipe
        if sql_metric_selects:
            select_lines.extend([f"    {metric}" for metric in sql_metric_selects])
        else:
            # Fallback if no custom metrics were provided
            select_lines.append(f"    COUNT(*) AS {status_str_map[record_type]}")
            
    if grid:
        select_lines.append(f"    {gridding_str}")
        
    select_block = "SELECT\n" + ",\n".join(select_lines)
    filter_block = "FROM occurrence\nWHERE\n    " + "\n    AND ".join(where_conditions)
    
    if aggregate:
        reserved_columns = {"group", "order", "type", "references", "class", "language", "year", "month", "day"}
        
        # Group by the dynamically provided columns, or default to all extracted base columns
        cols_to_group = sql_group_cols if sql_group_cols else columns
        group_cols = [f'"{col.lower()}"' if col.lower() in reserved_columns else col.lower() for col in cols_to_group]
        
        if grid:
            group_cols.append(f"{grid.lower()}cellcode")
            
        group_block = "\nGROUP BY\n    " + ",\n    ".join(group_cols)
    else:
        group_block = ""
        
    return f"{select_block}\n{filter_block}{group_block}"

def submit_gbif_query(gbif_query: str, creds: dict = None) -> str:
    """
    Submit a GBIF SQL download request.
    
    Args:
        gbif_query (str): The SQL query string to submit.
        creds (dict, optional): Dictionary containing 'GBIF_USER', 'GBIF_MAIL', 'GBIF_PWD'.
                                If None, credentials will be verified/prompted.
                                
    Returns:
        str: The GBIF download key used to track and fetch the request.
    """
    if not creds:
        creds = credentials.verify_gbif_credentials()

    try:
        download_key = pygbif.occurrences.download_sql(
            gbif_query,
            user=creds.get("GBIF_USER"),
            pwd=creds.get("GBIF_PWD")
        )
    except Exception as e:
        raise RuntimeError(f"Failed to submit GBIF download: {e}")

    print(f"GBIF download submitted successfully. Key: {download_key}")
    return download_key


def fetch_gbif_download(download_key: str,
                        target_dir: str = "",
                        max_time: int = 3600,
                        sleep_time: int = 30,
                        creds: dict = None) -> str:
    """
    Poll for completion of a GBIF download and save the ZIP file.
    Automatically creates the target directory if it does not exist.
    
    Args:
        download_key (str): The key returned by submit_gbif_query.
        target_dir (str): Directory where the ZIP file should be saved.
        max_time (int): Maximum polling time in seconds before throwing a TimeoutError.
        sleep_time (int): Seconds to wait between status checks.
        creds (dict, optional): Dictionary containing credentials. If None, verifies/prompts.
        
    Returns:
        str: The absolute or relative filepath to the downloaded ZIP.
    """
    if not creds:
        creds = credentials.verify_gbif_credentials()

    # ---- Directory existence check and auto-create --------------------------
    if target_dir:
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Could not create target directory '{target_dir}': {e}")
    else:
        target_dir = "."

    # ---- Poll download status ----------------------------------------------
    print(f"Polling status for download key: {download_key}...")
    start = time.time()
    
    while True:
        try:
            metadata = pygbif.occurrences.download_meta(download_key)
        except Exception as e:
            print(f"Warning: metadata fetch failed temporarily: {e}")
            metadata = {"status": "UNKNOWN"}

        status = metadata.get("status", "UNKNOWN")

        if status == "SUCCEEDED":
            print("Download succeeded!")
            break
        elif status in ("FAILED", "KILLED"):
            raise RuntimeError(f"GBIF download failed with status: {status}")

        if time.time() - start >= max_time:
            raise TimeoutError(
                f"GBIF download did not finish within {max_time} seconds. "
                f"Last status: {status}"
            )

        time.sleep(sleep_time)

    # ---- Download file to target directory ---------------------------------
    try:
        # download_get utilizes environment variables set by verify_gbif_credentials if needed
        filepath = pygbif.occurrences.download_get(download_key, path=target_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download GBIF file: {e}")
        
    return filepath

